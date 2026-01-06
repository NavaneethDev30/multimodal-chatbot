import gradio as gr
from PIL import Image
from io import BytesIO
import ollama

chat_history = []

def handle_text_input(user_message):
    global chat_history

    chat_history.append({"role": "user", "content": user_message})

    response = ollama.chat(model='phi3', messages=chat_history)
    bot_reply = response['message']['content']

    chat_history.append({"role": "assistant", "content": bot_reply})
    return [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) for msg in chat_history]

def analyze_image_with_llava(image, user_prompt):
    try:
        if image is None or user_prompt.strip() == "":
            return [(None, "Please upload an image and enter a prompt.")]

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        # LLaVA model supports image + prompt using ollama.generate()
        response = ollama.generate(
            model='llava',
            prompt=user_prompt,
            images=[image_bytes]
        )

        result = response['response']
        chat_history.append({"role": "user", "content": user_prompt})
        chat_history.append({"role": "assistant", "content": result})
        return [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) for msg in chat_history]

    except Exception as e:
        return [(None, f"Image processing failed: {str(e)}")]

def reset_chat():
    global chat_history
    chat_history = []
    return []

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  Multimodal Chatbot with Phi3 & LLaVA")
    chatbot = gr.Chatbot()
    
    with gr.Row():
        txt = gr.Textbox(placeholder="Type a message...", label="Your message")
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        prompt = gr.Textbox(placeholder="Ask about the image...", label="Image-related question")
        img_btn = gr.Button("Analyze Image")

    send_btn.click(fn=handle_text_input, inputs=txt, outputs=chatbot)
    txt.submit(fn=handle_text_input, inputs=txt, outputs=chatbot)
    clear_btn.click(fn=reset_chat, outputs=chatbot)
    img_btn.click(fn=analyze_image_with_llava, inputs=[img, prompt], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()





