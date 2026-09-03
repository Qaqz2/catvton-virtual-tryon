import os
import sys
from PIL import Image

# CatVTON 的模型代码在这个目录里，加进搜索路径才能 import
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CATVTON_DIR = os.path.join(_PROJECT_ROOT, "CatVTON")
if _CATVTON_DIR not in sys.path:
    sys.path.insert(0, _CATVTON_DIR)

# 让 huggingface 下载走 modelscope 镜像（国内网络）
try:
    from modelscope.utils.hf_util import patch_hub
    patch_hub()
except Exception:
    pass

_pipeline = None       # 推理管道
_automasker = None     # 蒙版生成
_mask_processor = None # 蒙版预处理


def load_model(
    base_model="runwayml/stable-diffusion-inpainting",
    resume_path="zhengchong/CatVTON",
    mixed_precision="fp16",
    device="cuda",
):
    global _pipeline, _automasker, _mask_processor
    import torch

    # 没有 CUDA 就退回 CPU（很慢，只是保证能跑）
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        mixed_precision = "no"

    # 只加载一次，之后直接复用
    if _pipeline is not None:
        return _pipeline, _automasker, _mask_processor

    from utils import init_weight_dtype
    from model.cloth_masker import AutoMasker
    from model.pipeline import CatVTONPipeline
    from huggingface_hub import snapshot_download
    from diffusers.image_processor import VaeImageProcessor

    local_models = os.path.join(_PROJECT_ROOT, "models")
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models")

    # CatVTON 权重：优先用本地 models/CatVTON，没有就自动下载
    repo_path = os.path.join(local_models, "CatVTON")
    if not os.path.exists(os.path.join(repo_path, "config.json")):
        print("models/CatVTON 不存在，尝试自动下载…")
        try:
            repo_path = snapshot_download(repo_id=resume_path)
        except Exception as e:
            raise RuntimeError(
                "CatVTON 权重下载失败，请手动下载后放到 models/CatVTON/（见 README）。"
            ) from e

    # SD 基础模型：本地 models/ → 之前下载过的缓存 → 都没有就按名字在线加载
    base_ckpt = None
    for p in (
        os.path.join(local_models, "stable-diffusion-inpainting"),
        os.path.join(hf_cache, "AI-ModelScope--stable-diffusion-inpainting", "snapshots", "master"),
    ):
        if os.path.exists(os.path.join(p, "model_index.json")):
            base_ckpt = p
            break
    if base_ckpt is None:
        base_ckpt = base_model

    # VAE 优先用独立的 sd-vae-ft-mse，没有就复用基础模型里的 vae
    vae_ckpt = os.path.join(local_models, "sd-vae-ft-mse")
    if not os.path.exists(os.path.join(vae_ckpt, "config.json")):
        vae_ckpt = os.path.join(base_ckpt, "vae") if base_ckpt != base_model else None
        if vae_ckpt and not os.path.exists(os.path.join(vae_ckpt, "config.json")):
            vae_ckpt = None

    _pipeline = CatVTONPipeline(
        base_ckpt=base_ckpt,
        vae_ckpt=vae_ckpt,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(mixed_precision),
        use_tf32=True,
        device=device,
        skip_safety_check=True,
    )
    _mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )
    _automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device=device,
    )
    print(f"模型加载完成，显存占用 {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")
    return _pipeline, _automasker, _mask_processor


def try_on(
    person_image,
    cloth_image,
    cloth_type="overall",
    num_inference_steps=30,
    guidance_scale=3.0,
    seed=42,
    width=512,
    height=768,
    progress_callback=None,
):
    if _pipeline is None:
        raise RuntimeError("模型未加载，请先调用 load_model()")

    from utils import resize_and_crop, resize_and_padding
    import torch

    # 人物图统一裁到 512x768
    person_image = resize_and_crop(person_image, (width, height))

    # 衣服图按类型只留上半/下半，再等比缩放到 512x768
    w, h = cloth_image.size
    if cloth_type == "upper":
        cloth_image = cloth_image.crop((0, 0, w, int(h * 0.70)))
    elif cloth_type == "lower":
        cloth_image = cloth_image.crop((0, int(h * 0.30), w, h))
    cloth_image = resize_and_padding(cloth_image, (width, height))

    # 自动生成换衣蒙版：蒙版内是要重绘新衣服的区域
    mask = _automasker(person_image, mask_type=cloth_type)["mask"]
    mask = resize_and_crop(mask, (width, height))

    with torch.inference_mode():
        result = _pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            callback=progress_callback,
        )

    result = result[0] if isinstance(result, list) else result
    return result.resize(person_image.size, Image.LANCZOS)
