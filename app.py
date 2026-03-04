import gradio as gr
from chatbot import question_answer



def qa(query, history, college_name):
    return question_answer(query, history, college_name)

def put_message_in_chatbot(message, history):
        return "", history + [{"role":"user", "content":message}]

with gr.Blocks(title="College Chatbot") as demo:
    gr.Markdown("# 🎓 College Information Chatbot")

    with gr.Row():
        college_dropdown = gr.Dropdown(
            label="Select College",
            choices=[
                "pict",
                "vit"
            ],
            value=None
        )

    chatbot = gr.Chatbot(label="Chat")
    msg = gr.Textbox(
        placeholder="Ask something about the selected college...",
        show_label=False
    )

    clear = gr.Button("Clear Chat")

    msg.submit(put_message_in_chatbot,[msg,chatbot],[msg,chatbot]).then(
        qa,
        inputs=[msg, chatbot, college_dropdown],
        outputs=[chatbot]
    )

    clear.click(
        lambda: [],
        outputs=[chatbot]
    )

demo.launch(inbrowser=True)
