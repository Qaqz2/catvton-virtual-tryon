# AI 虚拟试衣

上传一张人物照片和一张衣服图片，自动生成试穿效果。

基于 [CatVTON](https://github.com/Zheng-Chong/CatVTON)（ICLR 2025），界面用 Gradio 搭建。
在自己的笔记本（RTX 3060 6G 显存）上跑通的，主要是拿来练手 + 了解扩散模型的应用。

## 运行环境

- Windows 10/11
- Python 3.10（3.12 及以上装不了 CUDA 版的 torch）
- NVIDIA 显卡，显存 ≥ 6GB
- CUDA 11.8

## 快速开始

**1. 创建环境并安装依赖**

```bash
conda create -n ootd python=3.10 -y
conda activate ootd

# 必须装 CUDA 版 torch，直接 pip install torch 装的是 CPU 版
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

torch 的 wheel 比较大，国内下载慢的话可以从上海交大镜像下载后本地安装：
`https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/`

**2. 下载模型权重**

需要两个模型，建议手动下载放好（自动下载在国内网络下不稳定）：

- CatVTON 权重（约 500M），下载地址：
  - 国内：https://hf-mirror.com/zhengchong/CatVTON
  - 海外：https://huggingface.co/zhengchong/CatVTON
  只需要里面的 3 样东西，放到 `models/CatVTON/`：

```
models/CatVTON/
├── config.json
├── mix-48k-1024/
│   └── attention/model.safetensors
└── SCHP/
    ├── exp-schp-201908301523-atr.pth
    └── exp-schp-201908261155-lip.pth
```

- Stable Diffusion v1.5 inpainting 基础模型（约 5G）：https://modelscope.cn/models/AI-ModelScope/stable-diffusion-inpainting
  运行时会自动下载；如果下载太慢或失败，就把整个模型放到 `models/stable-diffusion-inpainting/`（目录里要有 model_index.json）。

**3. 启动**

```bash
python app.py
```

等模型加载完，浏览器打开 `http://127.0.0.1:7860`。

## 项目结构

```
├── app.py          # Gradio 界面，运行入口
├── tryon.py        # 模型加载 + 试穿推理
├── requirements.txt
├── CatVTON/        # 上游模型代码（做了少量改动）
└── models/         # 模型权重，太大不入库，需自行下载
```

- 推理统一用 fp16 半精度，6G 显存可以正常跑。
- 用 Gradio 写了个网页界面，上传图片 + 选衣物类别即可，不需要写文字描述。

## 已知限制

- 衣服会贴着原图的身体轮廓生成，oversize 之类的版型表现不出来，这是 inpainting 方法的通病。
- 只在 Windows + RTX 3060 6G 上验证过，其他环境没测。

## 参考

- CatVTON 论文代码：https://github.com/Zheng-Chong/CatVTON
- Stable Diffusion Inpainting：https://huggingface.co/runwayml/stable-diffusion-inpainting
