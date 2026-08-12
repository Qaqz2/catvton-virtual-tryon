import os

import math
import PIL
import numpy as np
import torch
from PIL import Image
from accelerate.state import AcceleratorState
from packaging import version
import accelerate
from typing import List, Optional, Tuple, Set
from diffusers import UNet2DConditionModel, SchedulerMixin
from tqdm import tqdm


def compute_dream_and_update_latents_for_inpaint(
    unet, noise_scheduler, timesteps, noise, noisy_latents, target,
    encoder_hidden_states, dream_detail_preservation=1.0,
):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation
    pred = None
    with torch.no_grad():
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    noisy_latents_no_condition = noisy_latents[:, :4]
    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        _noisy_latents = noisy_latents_no_condition.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    _noisy_latents = torch.cat([_noisy_latents, noisy_latents[:, 4:]], dim=1)
    return _noisy_latents, _target


def prepare_inpainting_input(
    noisy_latents, mask_latents, condition_latents,
    enable_condition_noise=True, condition_concat_dim=-1,
):
    if not enable_condition_noise:
        condition_latents_ = condition_latents.chunk(2, dim=condition_concat_dim)[-1]
        noisy_latents = torch.cat([noisy_latents, condition_latents_], dim=condition_concat_dim)
    noisy_latents = torch.cat([noisy_latents, mask_latents, condition_latents], dim=1)
    return noisy_latents


def compute_vae_encodings(image, vae):
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input


from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

def init_accelerator(config):
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.project_name,
        logging_dir=os.path.join(config.project_name, "logs"),
    )
    accelerator_ddp_config = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_ddp_config],
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name,
            config={
                "learning_rate": config.learning_rate,
                "train_batch_size": config.train_batch_size,
                "image_size": f"{config.width}x{config.height}",
            },
        )
    return accelerator


def init_weight_dtype(wight_dtype):
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[wight_dtype]


def init_add_item_id(config):
    return torch.tensor([config.height, config.width * 2, 0, 0, config.height, config.width * 2]).repeat(config.train_batch_size, 1)


def resize_and_crop(image, size):
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    image = image.crop(((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2))
    image = image.resize(size, Image.LANCZOS)
    return image


def resize_and_padding(image, size):
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


def prepare_mask_image(mask_image):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]
        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
            mask_image = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0)
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)
    return mask_image
