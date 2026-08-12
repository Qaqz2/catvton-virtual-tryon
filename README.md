# AI 虚拟试衣项目

上传自拍 + 衣物图片，AI 自动生成试穿效果。

基于 **CatVTON**（ICLR 2025 论文），针对 6GB 显存的 GPU 做了轻量化适配。

---

## 目录

- [项目结构](#项目结构)
- [每个文件干什么](#每个文件干什么)
- [部署步骤（从零开始）](#部署步骤从零开始)
- [踩坑记录与解决方案](#踩坑记录与解决方案)
- [日常使用](#日常使用)
- [技术原理简述](#技术原理简述)

---

## 项目结构

```
ootd/
├── app.py                  # Web 界面入口（Gradio）
├── tryon.py                # 模型封装（加载模型 + 执行推理）
├── requirements.txt        # Python 依赖清单（版本已锁定）
├── .gitignore
├── README.md               # 你正在看的这个文件
│
├── CatVTON/                # 模型源码（从论文仓库改造，不是原始代码）
│   ├── __init__.py
│   ├── utils.py            # 图像预处理工具（裁剪、缩放、VAE 编码）
│   └── model/
│       ├── pipeline.py     # 核心推理管道（UNet 去噪 + VAE 解码）
│       ├── cloth_masker.py # 蒙版生成器（SCHP 人体解析 → 决定换哪块）
│       ├── attn_processor.py   # 注意力处理器（CatVTON 微调的交叉注意力）
│       ├── DensePose/
│       │   └── __init__.py     # DensePose 桩实现（原版需要 detectron2，太重）
│       └── SCHP/                # 人体语义分割模型
│           ├── __init__.py     # SCHP 加载器
│           ├── modules/bn.py   # 批归一化层（从 InPlaceABNSync 简化）
│           ├── networks/       # ResNet101 网络定义
│           └── utils/          # 数据变换工具
│
├── models/                 # 模型权重（不入 Git，单独下载）
│   └── CatVTON/
│       ├── config.json
│       ├── mix-48k-1024/      # CatVTON 微调权重（~500MB）
│       │   └── attention/model.safetensors
│       └── SCHP/              # 人体解析权重
│           ├── exp-schp-201908301523-atr.pth   # ATR 数据集
│           └── exp-schp-201908261155-lip.pth   # LIP 数据集
│
├── wheels/                 # 预下载的安装包（不入 Git，单独下载）
│   ├── torch-2.4.1+cu118-cp310-cp310-win_amd64.whl
│   ├── torchvision-0.19.1+cu118-cp310-cp310-win_amd64.whl
│   ├── gradio-4.44.1-py3-none-any.whl
│   └── gradio_client-1.3.0-py3-none-any.whl
│
└── env/                    # Python 3.10 虚拟环境（不入 Git）
```

**不入 Git 的目录**：`models/`、`wheels/`、`env/` — 这些文件太大（合计 10GB+），需要按部署步骤单独下载。

---

## 每个文件干什么

### app.py — Web 界面

用 **Gradio** 搭建一个网页界面。用户在浏览器里上传图片、点按钮，背后调用 `tryon.py` 的模型推理。

核心流程：
1. 启动时加载模型（`load_model()`）
2. 用 `gr.Blocks` 搭建左右分栏界面：左边上传图片+参数，右边显示结果
3. 用户点"生成"→ `generate_tryon()` 函数被调用
4. 这个函数开一个后台线程跑模型推理，主线程每 0.3 秒检查进度并更新状态文字
5. 模型跑完，结果图显示在右边

关键设计：
- `sources=["upload"]`：只允许上传图片，不显示相机拍照按钮
- `gr.Radio` 而不是 `gr.Dropdown`：衣物类别用单选按钮，点一下就切换
- `demo.queue()`：必须开队列，`yield` 的进度信息才能实时推送到页面
- `NO_PROXY` 环境变量：开着 VPN 时防止代理拦截 localhost 请求

### tryon.py — 模型封装

封装了所有与 AI 模型交互的逻辑。`app.py` 只需要调用两个函数：`load_model()` 和 `try_on()`。

**`load_model()`** — 加载三个组件：
- **CatVTONPipeline**：核心推理管道（UNet 去噪网络 + VAE 编解码器）
- **AutoMasker**：自动生成人体蒙版（决定图片里哪些区域要换成新衣服）
- **VaeImageProcessor**：蒙版预处理工具

模型加载顺序：先找本地 `models/` 目录 → 再找 ModelScope 缓存 → 最后从网络下载。

**`try_on()`** — 执行试穿推理，分 4 步：
1. **图像预处理**：人物图裁剪到 512×768，衣物图按类型裁剪后填充到同尺寸
2. **生成蒙版**：AutoMasker 用 SCHP 人体解析模型识别出"要换衣服的区域"
3. **模型推理**：把人物图、衣物图、蒙版一起送入 pipeline，UNet 逐步去噪生成试穿效果
4. **肤色保护**：模型可能在蒙版内重新生成偏暗的皮肤，用 YCrCb 肤色检测把原图肤色替换回去

### CatVTON/model/pipeline.py — 核心推理管道

CatVTON 的核心。继承自 `DiffusionPipeline`，工作原理：
1. 把人物图和衣物图分别通过 VAE 编码成"潜在表示"（latent）
2. 在空间维度（高度）上拼接两个 latent，一起送入 UNet
3. UNet 是 Stable Diffusion v1.5 Inpainting 的 UNet，经过 CatVTON 微调
4. 逐步去噪（默认 30 步），每步用衣物图作为条件引导生成
5. 去噪完成后，拆分 latent 取人物部分，通过 VAE 解码回图片

**本项目修改**：UNet 加载时指定 `variant="fp16"`，匹配本地权重文件名 `diffusion_pytorch_model.fp16.safetensors`。

### CatVTON/model/cloth_masker.py — 蒙版生成器

决定"图片里哪些像素要被替换成新衣服"。工作流程：
1. **SCHP 人体解析**：用 ResNet101 对人物图做语义分割，区分出头发、脸、上衣、裤子、皮肤、背景等 18 个类别
2. **DensePose**（本项目用桩实现）：原版会检测人体关键点和区域映射，但需要安装 detectron2（C++ 编译，太重），本项目简化为返回空白掩码
3. **蒙版合成**：根据衣物类型（上衣/下装/连衣裙），用 SCHP 的解析结果生成对应区域的蒙版，同时保护不该改的区域（如换上衣时保护腿部）

关键数据结构 `PROTECT_BODY_PARTS`：定义每种衣物类型不该被覆盖的身体部位。

### CatVTON/model/SCHP/ — 人体语义分割

SCHP（Self-Correction Human Parsing）模型，基于 ResNet101。

**本项目关键修改**（`modules/bn.py`）：
- 原版用 `InPlaceABNSync`（来自 C++ 扩展 InPlaceABN），需要编译
- 本项目让 `ABN` 直接继承 `nn.BatchNorm2d`，这样模型参数 key（如 `bn1.weight`）和权重文件里的 key 完全对齐
- **这个修改至关重要**：之前用 `self.bn = nn.BatchNorm2d(...)` 包了一层，导致 key 变成 `bn1.bn.weight`，`strict=False` 加载时所有 BN 参数被跳过，SCHP 输出全背景 → 蒙版全黑 → 衣服根本换不上去

### CatVTON/utils.py — 图像预处理工具

提供 4 个关键函数：
- `resize_and_crop(image, size)`：缩放并裁剪到目标尺寸（裁掉多余部分）
- `resize_and_padding(image, size)`：缩放并填充到目标尺寸（加黑边，保留完整内容）
- `prepare_image(image)`：PIL 图片 → PyTorch 张量（归一化到 [-1, 1]）
- `compute_vae_encodings(images, vae)`：图片张量 → VAE 编码 → latent 表示

---

## 部署步骤（从零开始）

### 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 |
| GPU | NVIDIA GPU，显存 ≥ 6GB（本项目用 RTX 3060 Laptop 6GB） |
| CUDA | 11.8（用 nvidia-smi 查看） |
| Python | 3.10（**不要用 3.12+**，CUDA 版 PyTorch 没有 3.12 的 wheel） |
| 磁盘空间 | ≥ 15GB（环境 5GB + 模型权重 8GB + wheel 缓存 2GB） |

### 第 1 步：创建 Python 3.10 虚拟环境

```bash
# 用 conda 创建（推荐）
conda create -n ootd python=3.10 -y
conda activate ootd

# 或者用 venv（需要系统已装 Python 3.10）
python3.10 -m venv env
env\Scripts\activate
```

### 第 2 步：安装 PyTorch（GPU 版）

**不要用 `pip install torch`** — 那会装 CPU 版本。必须指定 CUDA 版本：

```bash
# 方法 A：从 PyTorch 官方源安装（需要外网）
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 方法 B：从上海交大镜像下载 wheel（国内推荐，40MB/s）
# 下载地址：https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/
# 下载这两个文件到 wheels/ 目录：
#   torch-2.4.1+cu118-cp310-cp310-win_amd64.whl
#   torchvision-0.19.1+cu118-cp310-cp310-win_amd64.whl
# 然后本地安装：
pip install wheels/torch-2.4.1+cu118-cp310-cp310-win_amd64.whl
pip install wheels/torchvision-0.19.1+cu118-cp310-cp310-win_amd64.whl
```

验证 CUDA 可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

### 第 3 步：安装 Gradio

```bash
# 方法 A：从 PyPI 安装（可能装到太新的版本）
pip install gradio==4.44.1 gradio_client==1.3.0

# 方法 B：用预下载的 wheel（推荐，避免版本问题）
pip install wheels/gradio-4.44.1-py3-none-any.whl wheels/gradio_client-1.3.0-py3-none-any.whl
```

### 第 4 步：安装其他依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**如果遇到版本冲突**（某个包自动升级到太新的版本导致报错），手动降级：
```bash
pip install pydantic==2.9.2 pydantic_core==2.23.4 fastapi==0.115.6 starlette==0.41.3 --force-reinstall --no-deps
```

### 第 5 步：下载模型权重

需要两部分权重：

**A. CatVTON 微调权重（~500MB）**

从 HuggingFace 下载：`https://huggingface.co/zhengchong/CatVTON`

下载后放到 `models/CatVTON/` 目录，结构如下：
```
models/CatVTON/
├── config.json
├── mix-48k-1024/
│   └── attention/
│       └── model.safetensors      # ~500MB
└── SCHP/
    ├── exp-schp-201908301523-atr.pth   # ATR 人体解析权重
    └── exp-schp-201908261155-lip.pth   # LIP 人体解析权重
```

如果 HuggingFace 连不上，用 ModelScope：`https://modelscope.cn/models/Zheng-Chong/CatVTON`

**B. Stable Diffusion v1.5 Inpainting 基础模型（~5GB）**

从 ModelScope 下载（国内速度快）：`https://modelscope.cn/models/AI-ModelScope/stable-diffusion-inpainting`

下载后会缓存到 `C:\Users\<用户名>\.cache\huggingface\hub\models\AI-ModelScope--stable-diffusion-inpainting\`

代码会自动在以下位置查找，无需手动配置：
1. `models/stable-diffusion-inpainting/`（本地目录）
2. ModelScope 缓存路径
3. HuggingFace 在线下载（最后手段）

### 第 6 步：启动应用

```bash
python app.py
```

等待 1-2 分钟（模型加载到 GPU），看到以下输出说明成功：
```
Running on local URL:  http://127.0.0.1:7860
```

浏览器打开 `http://127.0.0.1:7860` 即可使用。

---

## 踩坑记录与解决方案

这个项目从零搭建过程中遇到了大量问题。以下是所有踩过的坑和最终解决方案，按类别整理。

### 一、网络下载问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| GitHub clone 超时 | 国内网络 | 用镜像站 gitclone.com 或手动下载源文件 |
| HuggingFace 大文件下载失败 | huggingface.co 被墙 | 用 ModelScope（modelscope.cn）替代 |
| pip install 很慢 | PyPI 默认源在国外 | 用清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| torch wheel 下载慢 | PyTorch 官方源在国外 | 用上海交大镜像：`https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/` |
| 阿里云镜像 pip 索引失败 | 阿里云镜像不支持 pip 索引 | 换清华镜像 |
| SD Inpainting 模型下载不完整 | 网络中断导致半完成文件 | 发现 ModelScope 缓存已有 5GB 完整下载，直接复用 |

### 二、Python 环境问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `AssertionError: Torch not compiled with CUDA` | 系统默认 Python 3.14 没有 CUDA wheel | 创建 Python 3.10 虚拟环境 |
| C 盘空间不足 | conda/pip 缓存堆积 | 把虚拟环境建在 E 盘：`E:\ai\ootd\env` |
| wheel 文件名不标准 | SJTU 镜像的文件名格式不同 | 重命名为标准格式：`torch-2.4.1+cu118-cp310-cp310-win_amd64.whl` |

### 三、版本兼容问题（最耗时）

CatVTON 是 2024 年初的代码，但 2026 年的默认包版本已经不兼容。**每个版本冲突都是独立的，必须逐个解决。**

| 包 | 冲突版本 | 正确版本 | 症状 |
|----|----------|----------|------|
| transformers | 5.x | 4.36.2 | API 变更导致 import 失败 |
| diffusers | 0.39+ | 0.24.0 | Pipeline 参数签名变更 |
| huggingface_hub | 0.25+ | 0.20.3 | snapshot_download 接口变更 |
| tokenizers | 0.20+ | 0.15.2 | transformers 4.36.2 要求 |
| gradio | 5.x/6.x | 4.44.1 | 界面 API 完全不兼容 |
| gradio_client | 2.x | 1.3.0 | 与 gradio 4.44.1 配对 |
| pydantic | 2.13+ | 2.9.2 | Gradio 4.44.1 的 schema 验证失败 |
| pydantic_core | 2.30+ | 2.23.4 | 与 pydantic 2.9.2 配对 |
| fastapi | 0.115+ | 0.115.6 | starlette 模板加载失败 |
| starlette | 1.6+ | 0.41.3 | Jinja2 模板路径变更 |

**解决方法**：手动删除冲突包目录，用 `--no-deps` 安装指定版本，防止 pip 自动升级其他包。

```bash
# 示例：修复 gradio 版本
# 1. 找到包位置
python -c "import gradio; print(gradio.__file__)"
# 2. 删除整个目录
rm -rf <包目录>
# 3. 用 wheel 安装指定版本
pip install wheels/gradio-4.44.1-py3-none-any.whl --no-deps
```

### 四、代码层面的 Bug

| Bug | 文件 | 原因 | 修复 |
|-----|------|------|------|
| 蒙版位运算崩溃 | `cloth_masker.py` | 高斯模糊后 mask 是 float32，和布尔数组做 `\|` 运算报错 | 加 `.astype(bool)` |
| 蒙版尺寸不匹配 | `tryon.py` | `VaeImageProcessor.preprocess` 返回 Tensor，和 PIL 图片比尺寸失败 | 直接传 PIL 蒙版给 pipeline |
| SCHP 权重加载失败 | `SCHP/modules/bn.py` | `ABN` 类内部包了 `self.bn = nn.BatchNorm2d()`，导致参数 key 多了一层 `.bn`，`strict=False` 跳过所有 BN 参数 | 让 `ABN` 直接继承 `nn.BatchNorm2d` |
| UNet 权重找不到 | `pipeline.py` | 代码找 `diffusion_pytorch_model.safetensors`，但本地文件名是 `diffusion_pytorch_model.fp16.safetensors` | 加载时指定 `variant="fp16"` |
| 输出图是上下拼接 | `tryon.py` | 后处理用 25px 羽化蒙版把原图混回结果，旧衣服在边缘漏出 | 删除羽化混合，pipeline 已做 inpainting |
| 页面 500 错误 | pydantic 版本 | pydantic 2.13 的 schema 验证和 gradio 4.44.1 不兼容 | 降级到 pydantic 2.9.2 |
| Jinja2 模板加载失败 | starlette 版本 | starlette 1.6 的模板路径变更 | 降级到 starlette 0.41.3 |

### 五、VPN/代理干扰

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Gradio 启动后页面打不开 | VPN 代理拦截 localhost 请求 | `app.py` 设置 `NO_PROXY=127.0.0.1,localhost` |
| Gradio 自检误判服务不可用 | 代理拦截 httpx 探测请求 | `_net.url_ok = lambda url: True` 跳过自检 |

---

## 日常使用

### 启动命令

```bash
cd /d E:\ai\ootd
E:\ai\ootd\env\python.exe app.py
```

### 使用建议

- **人物照片**：全身正面照、手臂自然下垂、简单背景，效果最好
- **衣物图片**：正面平铺拍摄、轮廓清晰、背景干净
- **贴合度滑块**：默认 3.0，觉得生硬就调低到 2.0，觉得衣服不够清晰就调高到 4.0
- **推理步数**：默认 30 步，约 15-20 秒出图。降到 20 步可加速但不精细

### 已知限制

1. **DensePose 是桩实现**：原版需要 detectron2（C++ 编译），本项目简化为返回空白掩码。蒙版主要靠 SCHP 人体解析生成，对大部分照片够用
2. **衣服版型贴合身体**：CatVTON 基于 inpainting（局部重绘），衣服会贴合人物原图的身体轮廓，不会产生独立的垂坠感或 oversize 版型。这是模型架构限制
3. **显存 6GB 上限**：分辨率固定 512×768，不能更大

---

## 技术原理简述

### 什么是 CatVTON？

CatVTON（Concatenation-based Virtual Try-On）是 2025 年 ICLR 论文提出的虚拟试衣模型。核心思路极简：

1. **不需要文字描述**：不写"红色碎花吊带裙"，直接给图片
2. **不需要额外编码器**：不像 OOTDiffusion 需要 CLIP/image encoder，CatVTON 直接把衣物图编码后拼到人物图旁边
3. **用现成的 SD Inpainting**：基础模型是 Stable Diffusion v1.5 Inpainting，CatVTON 只微调了 UNet 的注意力层

### 推理流程

```
用户上传：人物图 + 衣物图
        ↓
   图像预处理（裁剪到 512×768）
        ↓
   SCHP 人体解析 → 生成蒙版（标记"要换衣服的区域"）
        ↓
   VAE 编码：人物图 → latent_person，衣物图 → latent_cloth
        ↓
   空间拼接：latent = [latent_person; latent_cloth]（上下拼）
        ↓
   UNet 去噪（30步）：
     每步输入 = [拼接latent, 蒙版, 被蒙版遮住的人物latent]
     每步输出 = 预测的噪声
     用衣物图作为条件引导生成方向
        ↓
   拆分 latent → 取人物部分 → VAE 解码 → 输出图片
        ↓
   肤色保护后处理 → 最终结果
```

### 关键简化（相对于原版 CatVTON）

1. **InPlaceABNSync → nn.BatchNorm2d**：原版用 C++ 扩展 InPlaceABN 做同步批归一化，需要编译。本项目让 ABN 继承 BatchNorm2d，功能等价但纯 Python 实现
2. **DensePose → 桩实现**：原版用 detectron2 做人体姿态估计，安装极复杂。本项目简化为返回空白掩码，蒙版完全靠 SCHP 生成
3. **fp16 推理**：所有模型用半精度浮点数，显存占用从 4.5GB 降到 2.3GB

---

## 版本矩阵

以下是通过验证的完整版本组合。**不要随意升级任何包**，版本之间有严格的兼容关系。

```
Python:        3.10.20
torch:         2.4.1+cu118
torchvision:   0.19.1+cu118
gradio:        4.44.1
gradio_client: 1.3.0
diffusers:     0.24.0
transformers:  4.36.2
huggingface_hub: 0.20.3
tokenizers:    0.15.2
accelerate:    1.14.0
pydantic:      2.9.2
pydantic_core: 2.23.4
fastapi:       0.115.6
starlette:     0.41.3
modelscope:    1.39.1
pillow:        10.4.0
numpy:         2.2.6
opencv-python: 5.0.0.93
```

GPU: NVIDIA RTX 3060 Laptop, 6GB VRAM, CUDA 11.8
