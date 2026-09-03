import os
import time
import threading

import gradio as gr
from tryon import load_model, try_on

# 启动时加载一次模型，点生成就不用等
load_model(mixed_precision="fp16")


def generate_tryon(person_img, cloth_img, cloth_type):
    if person_img is None:
        yield None, "请先上传人物照片"
        return
    if cloth_img is None:
        yield None, "请先上传衣物图片"
        return

    state = {"step": 0, "done": False, "error": None, "result": None}

    def run():
        try:
            state["result"] = try_on(
                person_image=person_img,
                cloth_image=cloth_img,
                cloth_type=cloth_type,
                progress_callback=lambda i, t, l: state.update(step=i + 1),
            )
        except Exception as e:
            state["error"] = str(e)
        finally:
            state["done"] = True

    # 推理慢，放后台线程跑，不然界面会卡住
    threading.Thread(target=run, daemon=True).start()

    while not state["done"]:
        time.sleep(0.3)
        if state["step"] > 0:
            yield None, f"正在生成试穿效果（第 {state['step']} 步）"
        else:
            yield None, "正在处理…"

    if state["error"]:
        yield None, "生成失败：" + state["error"]
    else:
        yield state["result"], "生成完成"


with gr.Blocks(title="AI 虚拟试衣") as demo:
    gr.Markdown("# AI 虚拟试衣\n上传一张人物照片和一张衣服图片，自动生成试穿效果。")

    with gr.Row():
        with gr.Column(scale=1):
            person_input = gr.Image(label="人物照片", type="pil", sources=["upload"])
            cloth_input = gr.Image(label="衣服图片", type="pil", sources=["upload"])
            cloth_type = gr.Radio(
                label="衣物类型",
                choices=[
                    ("上衣", "upper"),
                    ("下装", "lower"),
                    ("连衣裙/套装", "overall"),
                ],
                value="overall",
            )
            submit_btn = gr.Button("生成试穿效果", variant="primary")

        with gr.Column(scale=1):
            result_output = gr.Image(label="试穿效果", type="pil", height=500)
            status = gr.Textbox(label="状态", interactive=False)

    submit_btn.click(
        generate_tryon,
        inputs=[person_input, cloth_input, cloth_type],
        outputs=[result_output, status],
        api_name=False,  # 只做本地网页，不开放 API
    )

if __name__ == "__main__":
    # 开着代理时会误判 localhost 不可访问，直接跳过自检
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    import gradio.networking as net
    net.url_ok = lambda url: True

    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)
