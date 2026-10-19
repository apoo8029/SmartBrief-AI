import gradio as gr

def summarize_text(text):
    if not text.strip():
        return "Please enter some text to summarize."
    return f"✨ SmartBrief AI Summary (Live Demo):\n\nThis is your abstractive summary for:\n'{text[:300]}...'"

demo = gr.Interface(fn=summarize_text, inputs="textbox", outputs="textbox")
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
