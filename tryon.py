"""
虚拟试衣模型封装 (tryon.py)
基于 CatVTON (ICLR 2025) — 针对 6GB 显存优化

CatVTON 做了什么？
  1. 用 Stable Diffusion v1.5 Inpainting 作为"画布"基础模型
  2. 把人物图和衣物图在空间维度拼接，一起送入 UNet 去噪网络
  3. 通过 AutoMasker（DensePose + SCHP）自动识别人体区域，决定哪里换衣服
  4. 不需要额外编码器、不需要文字描述——结构极简

说明：
  这个文件封装了所有模型交互逻辑。app.py 只需要 import tryon
  然后调用 load_model() + try_on()，不碰模型细节。
  这是工程中"关注点分离"的体现——界面和模型各自独立。
"""

import os
import sys
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════
# 第一步：建立 CatVTON 的导入路径
# ═══════════════════════════════════════════════════════════════════════
# 说明：sys.path 是 Python 搜索模块的路径列表。
# 把 CatVTON 目录加进去，才能 import 它内部的模块。

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CATVTON_DIR = os.path.join(_PROJECT_ROOT, "CatVTON")

if not os.path.isdir(_CATVTON_DIR):
    raise FileNotFoundError(
        "未找到 CatVTON 目录。请确认克隆的是完整仓库，"
        "根目录下应包含 CatVTON/ 文件夹（模型代码已随仓库提供）。"
    )

if _CATVTON_DIR not in sys.path:
    sys.path.insert(0, _CATVTON_DIR)

# ═══════════════════════════════════════════════════════════════════════
# 网络重定向：让 HuggingFace 下载自动走 ModelScope（国内通道）
# ═══════════════════════════════════════════════════════════════════════
# 说明：CatVTON 代码内部调用 huggingface_hub.snapshot_download 和
# diffusers 的 from_pretrained，默认访问 huggingface.co。国内网络下
# 这些调用会超时失败。modelscope 的 patch_hub() 会把它们重定向到
# 魔搭社区（阿里云 OSS），下载速度稳定。
try:
    from modelscope.utils.hf_util import patch_hub
    patch_hub()
    print("[INFO] 已启用 ModelScope 网络重定向，HuggingFace 下载将走国内通道。")
except Exception as e:
    print(f"[WARN] ModelScope patch_hub 启用失败: {e}")

# ═══════════════════════════════════════════════════════════════════════
# 第二步：全局变量——模型只加载一次
# ═══════════════════════════════════════════════════════════════════════
# 说明：模型加载很慢（下载权重 + 初始化 GPU），所以用全局变量缓存。
# 首次调用 load_model() 时创建，之后的调用直接返回缓存，避免重复加载。

_pipeline = None       # CatVTONPipeline：核心推理管道
_automasker = None     # AutoMasker：自动生成人体蒙版
_mask_processor = None # VaeImageProcessor：蒙版预处理


def load_model(
    base_model: str = "runwayml/stable-diffusion-inpainting",
    resume_path: str = "zhengchong/CatVTON",
    mixed_precision: str = "fp16",
    device: str = "cuda",
):
    # 如果 CUDA 不可用，自动降级到 CPU（会慢很多，但能跑起来）
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("  [WARN] CUDA 不可用，自动切换到 CPU（非常慢，建议安装 CUDA 版 PyTorch）")
        device = "cpu"
    if device == "cpu":
        mixed_precision = "no"

    """
    加载 CatVTON 模型。
    首次运行会自动从 HuggingFace 下载权重（约 2-5 GB），请耐心等待。

    说明：
      fp16 是"半精度浮点数"——用 16 位而不是 32 位来存数据。
      好处：显存占用减半，对 6GB 显卡至关重要。
      为什么不用 bf16？GTX 3060 移动版是 Ampere 架构，bf16 支持不完整，
      fp16 在这个硬件上更稳定。

    参数:
        base_model:      基础模型 ID（HuggingFace）
        resume_path:     CatVTON 微调权重 ID
        mixed_precision: "fp16"（推荐）| "bf16" | "no"
        device:          "cuda"（GPU）| "cpu"（不推荐，会极慢）
    """
    global _pipeline, _automasker, _mask_processor

    if _pipeline is not None:
        return _pipeline, _automasker, _mask_processor

    # 延迟导入 CatVTON 的模块——确保路径已设置
    from utils import init_weight_dtype
    from model.cloth_masker import AutoMasker
    from model.pipeline import CatVTONPipeline

    import torch
    from huggingface_hub import snapshot_download
    from diffusers.image_processor import VaeImageProcessor

    print("=" * 60)
    print("  正在加载 CatVTON 虚拟试衣模型...")
    print(f"  基础模型: {base_model}")
    print(f"  精度模式: {mixed_precision}")
    print(f"  推理设备: {device}")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════════
    # 智能模型路径查找：本地目录 → ModelScope 缓存 → HuggingFace 网络
    # ═══════════════════════════════════════════════════════════════════
    local_models_dir = os.path.join(_PROJECT_ROOT, "models")
    local_catvton = os.path.join(local_models_dir, "CatVTON")
    local_base = os.path.join(local_models_dir, "stable-diffusion-inpainting")
    local_vae = os.path.join(local_models_dir, "sd-vae-ft-mse")

    # ModelScope 缓存路径
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models")
    ms_sd_cache = os.path.join(hf_cache, "AI-ModelScope--stable-diffusion-inpainting", "snapshots", "master")

    # ── CatVTON 适配器权重 ──
    catvton_found = False
    for candidate in [local_catvton]:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            repo_path = candidate
            catvton_found = True
            print(f"  [OK] CatVTON 权重: {candidate}")
            break

    if not catvton_found:
        print("  [INFO] CatVTON 权重未找到，尝试自动下载...")
        try:
            repo_path = snapshot_download(repo_id=resume_path)
            print(f"  [OK] CatVTON 下载到: {repo_path}")
        except Exception as e:
            raise RuntimeError(
                "CatVTON 权重自动下载失败。请手动下载后放到 models/CatVTON/ 目录，"
                "再重新运行。下载地址见 README「下载模型权重」一节。"
            ) from e

    # ── SD Inpainting 基础模型 ──
    base_found = False
    for candidate in [local_base, ms_sd_cache]:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "model_index.json")):
            base_ckpt = candidate
            base_found = True
            print(f"  [OK] SD 基础模型: {candidate}")
            break

    if not base_found:
        print(f"  [INFO] SD 模型未找到，将从 {base_model} 加载（可能触发下载）...")
        base_ckpt = base_model

    # ── VAE 模型 ──
    # 优先用独立 VAE；没有就用 SD Inpainting 自带的 vae 子目录
    vae_ckpt = None
    for candidate in [local_vae]:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            vae_ckpt = candidate
            print(f"  [OK] VAE 模型: {candidate}")
            break
    if vae_ckpt is None and base_found:
        sd_vae = os.path.join(base_ckpt, "vae")
        if os.path.isdir(sd_vae) and os.path.exists(os.path.join(sd_vae, "config.json")):
            vae_ckpt = sd_vae
            print(f"  [OK] VAE 模型: {sd_vae} (复用 SD 基础模型)")

    # 创建推理管道
    _pipeline = CatVTONPipeline(
        base_ckpt=base_ckpt,
        vae_ckpt=vae_ckpt,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(mixed_precision),
        use_tf32=True,          # RTX 30 系列支持 TF32 加速
        device=device,
        skip_safety_check=True, # 跳过 NSFW 检测，减少依赖
    )

    # 创建蒙版工具
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

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"  当前显存占用: {allocated:.1f} GB")
    print("  模型加载完成！\n")
    return _pipeline, _automasker, _mask_processor


def try_on(
    person_image,
    cloth_image,
    cloth_type: str = "overall",
    num_inference_steps: int = 30,
    guidance_scale: float = 3.0,
    seed: int = 42,
    width: int = 512,
    height: int = 768,
    progress_callback=None,
):
    """
    执行虚拟试衣推理——这是用户点击"生成"时实际调用的函数。

    参数:
        person_image:        人物自拍（PIL Image）
                            建议：全身照、正面站姿、简单背景
        cloth_image:         衣物商品图（PIL Image）
                            建议：正面平铺拍摄、轮廓清晰
        cloth_type:          衣物类别
                             "upper"   → 上衣（T恤、衬衫等）
                             "lower"   → 下装（裤子、裙子）
                             "overall" → 连衣裙 / 连体套装
        num_inference_steps: 去噪步数（默认 30）
                             越多越精细，但越慢（6GB 推荐 20-30）
        guidance_scale:      引导强度（默认 3.0）
                             越高→越贴合原衣物细节
                             越低→生成更自由，可能偏离原图
        seed:                随机种子（默认 42）
                             固定种子 + 相同输入 = 相同输出（可复现）
        width, height:       输出分辨率（默认 512×768）
                             6GB 显存不要超过这个值

    返回:
        试穿效果图（PIL Image）

    说明：
      torch.inference_mode() 是关键——它禁用梯度计算。
      梯度用于训练阶段的反向传播，推理不需要。
      关闭它能省约 30% 显存。
    """
    global _pipeline, _automasker, _mask_processor

    if _pipeline is None:
        raise RuntimeError("模型未加载！请先调用 load_model()")

    from utils import resize_and_crop, resize_and_padding
    import torch

    # ── 第 1 步：图像预处理 ──
    # resize_and_crop:   裁剪到目标尺寸（裁掉多余部分）
    # resize_and_padding: 填充到目标尺寸（加黑边，保留完整内容）
    person_image = resize_and_crop(person_image, (width, height))

    # 根据衣物类型裁剪参考图：让模型只关注对应区域，
    # 避免"上衣"类型却看到整件连衣裙导致生成混乱。
    w, h = cloth_image.size
    if cloth_type == "upper":
        cloth_image = cloth_image.crop((0, 0, w, int(h * 0.70)))
    elif cloth_type == "lower":
        cloth_image = cloth_image.crop((0, int(h * 0.30), w, h))
    cloth_image = resize_and_padding(cloth_image, (width, height))

    # ── 第 2 步：自动生成蒙版 ──
    # AutoMasker 内部做了两件事：
    #   DensePose → 检测人体关键区域（躯干、手臂、腿部等）
    #   SCHP      → 语义分割（区分衣服、皮肤、背景等）
    # 然后交叉分析：哪些像素应该被替换成新衣服，哪些必须保护
    mask_result = _automasker(person_image, mask_type=cloth_type)
    mask = mask_result["mask"]
    # 蒙版统一裁剪到与人物图相同的尺寸，交给 pipeline 内部处理（保持 PIL 格式）
    mask = resize_and_crop(mask, (width, height))

    # ── 第 3 步：模型推理 ──
    with torch.inference_mode():
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = _pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            callback=progress_callback,
        )

    # pipeline 返回图片列表，取第一张
    # pipeline 内部已做 inpainting：蒙版外=原图，蒙版内=生成新衣服
    result = result[0] if isinstance(result, list) else result
    return result.resize(person_image.size, Image.LANCZOS)
