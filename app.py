"""
AI 虚拟试衣 — 用户界面 (app.py)

运行方法:
    python app.py

然后浏览器打开 http://127.0.0.1:7860 即可使用。

说明：
  Gradio 是 HuggingFace 推出的 AI 应用框架，专门用于快速搭建
  机器学习模型的 Web 演示界面。"gr.Interface"是最简单的用法，
  这里用 "gr.Blocks" 是因为需要更灵活的布局（左右分栏）。
"""

import os
import time
import threading
import gradio as gr

# ── 导入我们的模型封装模块 ──
from tryon import load_model, try_on

# ═══════════════════════════════════════════════════════════════════════
# 应用启动时加载模型（只加载一次）
# ═══════════════════════════════════════════════════════════════════════
# 说明：
#   把模型加载放在 if __name__ == "__main__" 之前、Gradio 界面启动之前，
#   这样 Gradio 启动时模型已经就绪。如果放得太晚，用户点"生成"时
#   会先等待模型下载（可能几分钟），体验不好。

print("正在初始化模型，请稍候...")
pipeline, automasker, mask_processor = load_model(mixed_precision="fp16")
print("模型就绪，启动 Web 界面...\n")


# ═══════════════════════════════════════════════════════════════════════
# 核心回调函数：用户点击"生成"时调用
# ═══════════════════════════════════════════════════════════════════════
def generate_tryon(person_img, cloth_img, cloth_type, steps, guidance):
    """
    Gradio 按钮的回调函数。
    接收界面输入 → 调用 try_on() → 返回结果图。

    说明：
      这个函数是"胶水层"——连接 Gradio 界面和 tryon 模型。
      它不做任何 AI 逻辑，只做参数传递和错误处理。
      这就是软件工程中"单一职责原则"：一个函数只做一件事。

      用 yield 返回（而不是 return）：Gradio 会把每次 yield 的内容
      实时显示在状态框里，所以用户能看到"第几步/共几步"的进度，
      而不是干等。
    """
    if person_img is None:
        yield None, "请先上传人物自拍。"
        return
    if cloth_img is None:
        yield None, "请先上传衣物图片。"
        return

    # 后台线程跑模型（很慢，不能阻塞界面），
    # 主线程循环检查进度并实时显示。
    state = {"stage": "准备中…", "step": 0, "error": None, "done": False, "result": None}

    def _run():
        try:
            state["stage"] = "正在解析人体轮廓…"

            def on_step(i, t, latents):
                if i == 0:
                    state["stage"] = "正在生成试穿效果…"
                state["step"] = i + 1

            state["result"] = try_on(
                person_image=person_img,
                cloth_image=cloth_img,
                cloth_type=cloth_type,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=42,  # 固定种子：相同输入 = 相同结果，方便复现
                progress_callback=on_step,
            )
            state["done"] = True
        except Exception as e:
            state["error"] = f"生成失败：{e}"
            state["done"] = True

    threading.Thread(target=_run, daemon=True).start()

    while not state["done"]:
        time.sleep(0.3)
        if state["step"] > 0:
            yield None, f"{state['stage']}（第 {state['step']}/{steps} 步）"
        else:
            yield None, state["stage"]

    if state["error"]:
        yield None, state["error"]
    else:
        yield state["result"], "生成完成！"


# ═══════════════════════════════════════════════════════════════════════
# 构建 Gradio 界面
# ═══════════════════════════════════════════════════════════════════════
# 说明：
#   gr.Blocks 是 Gradio 的"自由布局"模式。
#   每个 gr.Row / gr.Column 创建一行/一列。
#   组件放在容器里，从左到右、从上到下排列。

with gr.Blocks(
    title="AI 虚拟试衣",
    theme=gr.themes.Soft(),   # 简洁的预设主题
) as demo:

    gr.Markdown(
        """
        # 👔 AI 虚拟试衣
        **上传你的自拍，选择衣物，AI 自动生成试穿效果。**
        """
    )

    with gr.Row():
        # ── 左侧：输入区 ──
        with gr.Column(scale=1):
            person_input = gr.Image(
                label="📷 上传自拍照片",
                type="pil",
                height=300,
                sources=["upload"],  # 只允许从本地上传，不显示相机拍照按钮
            )
            cloth_input = gr.Image(
                label="👕 上传衣物图片",
                type="pil",
                height=300,
                sources=["upload"],
            )

            cloth_type = gr.Radio(
                label="衣物类别",
                choices=[
                    ("连衣裙 / 套装", "overall"),
                    ("上衣（T恤、衬衫等）", "upper"),
                    ("下装（裤子、裙子）", "lower"),
                ],
                value="overall",
            )

            with gr.Row():
                steps = gr.Slider(
                    label="推理步数（越多越精细、越慢）",
                    minimum=15,
                    maximum=50,
                    value=30,
                    step=5,
                )
                guidance = gr.Slider(
                    label="贴合度（低=更自然，高=更还原衣物）",
                    minimum=1.0,
                    maximum=4.0,
                    value=3.0,
                    step=0.5,
                )

            submit_btn = gr.Button(
                "✨ 生成试穿效果",
                variant="primary",
                size="lg",
            )

        # ── 右侧：输出区 ──
        with gr.Column(scale=1):
            result_output = gr.Image(
                label="试穿效果",
                type="pil",
                height=500,
            )
            status = gr.Textbox(
                label="状态",
                interactive=False,
            )

    # ── 使用说明 ──
    gr.Markdown(
        """
        ---
        ### 使用技巧
        - **人物照片**：全身照、正面站姿、简单背景效果最好
        - **衣物图片**：正面平铺拍摄、轮廓清晰
        - **显存不足？** 尝试降低推理步数到 20，或联系我调整分辨率
        """
    )

    # ── 绑定按钮事件 ──
    # 说明：
    #   .click() 是事件绑定——"当按钮被点击时，执行函数 X，
    #   用组件 A、B、C 作为输入，结果输出到组件 D、E"。
    #   这是 Gradio 的核心概念：组件 → 事件 → 函数 → 组件。

    submit_btn.click(
        fn=generate_tryon,
        inputs=[person_input, cloth_input, cloth_type, steps, guidance],
        outputs=[result_output, status],
    )


# ═══════════════════════════════════════════════════════════════════════
# 程序入口
# ═══════════════════════════════════════════════════════════════════════
# 说明：
#   if __name__ == "__main__" 是 Python 的标准写法。
#   当直接运行 python app.py 时，__name__ 的值是 "__main__"，条件为真。
#   当被其他文件 import 时，__name__ 是 "app"，条件为假，不会启动服务。
#   这防止了"import 别人的代码时意外启动了 Web 服务器"的问题。

if __name__ == "__main__":
    # 开着 VPN/代理时，gradio 会用 httpx 探测 localhost 是否可访问，
    # 探测请求会被代理拦截导致误判"localhost 不可访问"。
    # 服务器本身是正常的（URL 已打印），这里直接跳过探测。
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    import gradio.networking as _net
    _net.url_ok = lambda url: True  # 本地应用，跳过 localhost 自检

    demo.queue()  # 开启队列：进度信息才能实时推送到页面

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,    # 仅本地访问（需要外网链接时改为 True）
    )
