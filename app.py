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
    gr.Markdown("# 🧠 Multimodal Chatbot with Phi3 & LLaVA")
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





# import os
# import json
# import gradio as gr
# from datetime import datetime
# import uuid
# from PIL import Image
# import ollama
# from image_handler import analyze_image

# # Configuration
# CHAT_HISTORY_DIR = "chat_history"
# os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# class ChatManager:
#     def __init__(self):
#         self.sessions = {}
#         self.load_sessions()
#         if not self.sessions:
#             self.create_session("New Chat")

#     def create_session(self, title="New Chat"):
#         session_id = str(uuid.uuid4())
#         self.sessions[session_id] = {
#             "id": session_id,
#             "title": title,
#             "created": datetime.now().isoformat(),
#             "updated": datetime.now().isoformat(),
#             "messages": []
#         }
#         self.save_session(session_id)
#         return session_id

#     def update_session_title(self, session_id, prompt):
#         if session_id in self.sessions and len(self.sessions[session_id]["messages"]) <= 1:
#             title = prompt[:30] + ("..." if len(prompt) > 30 else "")
#             self.sessions[session_id]["title"] = title
#             self.save_session(session_id)

#     def add_message(self, session_id, role, content):
#         if session_id in self.sessions:
#             self.sessions[session_id]["messages"].append({
#                 "role": role,
#                 "content": content,
#                 "timestamp": datetime.now().isoformat()
#             })
#             self.sessions[session_id]["updated"] = datetime.now().isoformat()
#             self.save_session(session_id)

#     def get_session(self, session_id):
#         return self.sessions.get(session_id)

#     def get_all_sessions(self):
#         return sorted(
#             self.sessions.values(),
#             key=lambda x: x["updated"],
#             reverse=True
#         )

#     def save_session(self, session_id):
#         with open(f"{CHAT_HISTORY_DIR}/{session_id}.json", "w") as f:
#             json.dump(self.sessions[session_id], f)

#     def load_sessions(self):
#         for filename in os.listdir(CHAT_HISTORY_DIR):
#             if filename.endswith(".json"):
#                 with open(f"{CHAT_HISTORY_DIR}/{filename}", "r") as f:
#                     session = json.load(f)
#                     self.sessions[session["id"]] = session

# chat_manager = ChatManager()

# def generate_response(prompt, session_id):
#     chat_manager.add_message(session_id, "user", prompt)
#     chat_manager.update_session_title(session_id, prompt)
#     try:
#         messages = [
#             {"role": msg["role"], "content": msg["content"]}
#             for msg in chat_manager.get_session(session_id)["messages"]
#         ]
#         response = ollama.chat(
#             model='phi3',
#             messages=messages,
#             options={'num_ctx': 1024, 'temperature': 0.3}
#         )
#         ai_response = response['message']['content']
#         chat_manager.add_message(session_id, "assistant", ai_response)
#         return ai_response
#     except Exception as e:
#         return f"Error: {str(e)}"

# def format_chat_display(messages):
#     display = []
#     for msg in messages:
#         if msg["role"] == "user":
#             display.append((msg["content"], None))
#         elif msg["role"] == "assistant" and display:
#             display[-1] = (display[-1][0], msg["content"])
#     return display

# def send_message(session_id, text, image):
#     if not text and not image:
#         return session_id, [], None, ""

#     response_parts = []

#     if image:
#         img_result = analyze_image(image)
#         response_parts.append(f"📸 Image Analysis:\n{img_result}")
#         chat_manager.add_message(session_id, "user", "[Image Attached]")
#         chat_manager.add_message(session_id, "system", f"Image analysis: {img_result}")

#     if text:
#         text_response = generate_response(text, session_id)
#         response_parts.append(text_response)

#     full_response = "\n\n".join(response_parts)
#     if text:
#         chat_manager.add_message(session_id, "assistant", full_response)

#     messages = chat_manager.get_session(session_id)["messages"]
#     return session_id, format_chat_display(messages), None, ""

# def new_chat():
#     session_id = chat_manager.create_session()
#     return session_id, [], None, ""

# def load_chat(evt: gr.SelectData):
#     session_id = evt.value["id"]
#     messages = chat_manager.get_session(session_id)["messages"]
#     return session_id, format_chat_display(messages), None, ""

# def get_session_list():
#     return [{"title": s["title"], "id": s["id"]} for s in chat_manager.get_all_sessions()]

# with gr.Blocks(theme=gr.themes.Soft(), title="AI Chat Assistant") as app:
#     session_id = gr.State()

#     with gr.Row():
#         with gr.Column(scale=1, min_width=250):
#             gr.Markdown("### Chat Sessions")
#             new_chat_btn = gr.Button("+ New Chat", size="sm")
#             sessions_list = gr.Dataset(
#                 components=[gr.Textbox(visible=False)],
#                 label="Your Chats",
#                 samples=get_session_list(),
#                 elem_id="sessions-list"
#             )

#         with gr.Column(scale=4):
#             chatbot = gr.Chatbot(
#                 elem_id="chatbot",
#                 bubble_full_width=False,
#                 height="70vh"
#             )

#             with gr.Row():
#                 image_input = gr.Image(
#                     type="pil",
#                     label="Upload Image"
#                 )
#                 upload_btn = gr.UploadButton(
#                     "📷 Upload Image",
#                     file_types=["image"],
#                     variant="secondary"
#                 )

#             with gr.Row():
#                 text_input = gr.Textbox(
#                     placeholder="Type your message...",
#                     container=False,
#                     scale=7
#                 )
#                 submit_btn = gr.Button("Send", variant="primary", scale=1)

#     if chat_manager.get_all_sessions():
#         app.load(
#             lambda: (
#                 chat_manager.get_all_sessions()[0]["id"],
#                 format_chat_display(chat_manager.get_all_sessions()[0]["messages"]),
#                 None,
#                 ""
#             ),
#             outputs=[session_id, chatbot, image_input, text_input]
#         )
#     else:
#         app.load(
#             new_chat,
#             outputs=[session_id, chatbot, image_input, text_input]
#         )

#     upload_btn.upload(
#         lambda f: f,
#         inputs=[upload_btn],
#         outputs=[image_input]
#     )

#     submit_btn.click(
#         send_message,
#         inputs=[session_id, text_input, image_input],
#         outputs=[session_id, chatbot, image_input, text_input]
#     ).then(
#         lambda: gr.Dataset.update(samples=get_session_list()),
#         outputs=[sessions_list]
#     )

#     text_input.submit(
#         send_message,
#         inputs=[session_id, text_input, image_input],
#         outputs=[session_id, chatbot, image_input, text_input]
#     ).then(
#         lambda: gr.Dataset.update(samples=get_session_list()),
#         outputs=[sessions_list]
#     )

#     new_chat_btn.click(
#         new_chat,
#         outputs=[session_id, chatbot, image_input, text_input]
#     ).then(
#         lambda: gr.Dataset.update(samples=get_session_list()),
#         outputs=[sessions_list]
#     )

#     sessions_list.select(
#         load_chat,
#         outputs=[session_id, chatbot, image_input, text_input]
#     )

# if __name__ == "__main__":
#     print("Please ensure Ollama is running with: ollama serve")
#     app.launch(server_port=7860, share=False)














# import os
# import json
# import gradio as gr
# from datetime import datetime
# import uuid
# from transformers import CLIPProcessor, CLIPModel
# import ollama
# import torch
# from PIL import Image

# # Configuration
# CHAT_HISTORY_DIR = "chat_history"
# os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# # Initialize AI Models
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
# torch.set_num_threads(4)

# # Warm-up models
# with torch.no_grad():
#     dummy_input = clip_processor(text=["test"], images=torch.rand(1,3,224,224).to(device), return_tensors="pt").to(device)
#     clip_model(**dummy_input)

# class ChatManager:
#     def __init__(self):
#         self.sessions = {}
#         self.load_sessions()
#         if not self.sessions:
#             self.create_session("New Chat")
    
#     def create_session(self, title="New Chat"):
#         session_id = str(uuid.uuid4())
#         self.sessions[session_id] = {
#             "id": session_id,
#             "title": title,
#             "created": datetime.now().isoformat(),
#             "updated": datetime.now().isoformat(),
#             "messages": []
#         }
#         self.save_session(session_id)
#         return session_id
    
#     def update_session_title(self, session_id, prompt):
#         if session_id in self.sessions and len(self.sessions[session_id]["messages"]) <= 1:
#             title = prompt[:30] + ("..." if len(prompt) > 30 else "")
#             self.sessions[session_id]["title"] = title
#             self.save_session(session_id)
    
#     def add_message(self, session_id, role, content):
#         if session_id in self.sessions:
#             self.sessions[session_id]["messages"].append({
#                 "role": role,
#                 "content": content,
#                 "timestamp": datetime.now().isoformat()
#             })
#             self.sessions[session_id]["updated"] = datetime.now().isoformat()
#             self.save_session(session_id)
    
#     def get_session(self, session_id):
#         return self.sessions.get(session_id)
    
#     def get_all_sessions(self):
#         return sorted(
#             self.sessions.values(),
#             key=lambda x: x["updated"],
#             reverse=True
#         )
    
#     def save_session(self, session_id):
#         with open(f"{CHAT_HISTORY_DIR}/{session_id}.json", "w") as f:
#             json.dump(self.sessions[session_id], f)
    
#     def load_sessions(self):
#         for filename in os.listdir(CHAT_HISTORY_DIR):
#             if filename.endswith(".json"):
#                 with open(f"{CHAT_HISTORY_DIR}/{filename}", "r") as f:
#                     session = json.load(f)
#                     self.sessions[session["id"]] = session

# chat_manager = ChatManager()

# def generate_response(prompt, session_id):
#     chat_manager.add_message(session_id, "user", prompt)
#     chat_manager.update_session_title(session_id, prompt)
    
#     try:
#         messages = [{"role": msg["role"], "content": msg["content"]} 
#                    for msg in chat_manager.get_session(session_id)["messages"]]
        
#         response = ollama.chat(
#             model='phi3',
#             messages=messages,
#             options={'num_ctx': 1024, 'temperature': 0.3}
#         )
#         ai_response = response['message']['content']
#         chat_manager.add_message(session_id, "assistant", ai_response)
#         return ai_response
#     except Exception as e:
#         return f"Error: {str(e)}"

# def analyze_image(image, session_id):
#     try:
#         if not isinstance(image, Image.Image):
#             image = Image.fromarray(image)
#         image = image.resize((224, 224))
        
#         inputs = clip_processor(
#             text=["photo", "drawing", "text", "person", "object"],
#             images=image, 
#             return_tensors="pt"
#         ).to(device)
        
#         with torch.no_grad():
#             outputs = clip_model(**inputs)
#             probs = outputs.logits_per_image.softmax(dim=1).tolist()[0]
        
#         labels = ["photo", "drawing", "text", "person", "object"]
#         result = {k: f"{v:.2%}" for k, v in zip(labels, probs)}
#         chat_manager.add_message(session_id, "system", f"Image analysis: {result}")
#         return result
#     except Exception as e:
#         return f"Image analysis error: {str(e)}"

# def format_chat_display(messages):
#     display = []
#     for msg in messages:
#         if msg["role"] == "user":
#             display.append((msg["content"], None))
#         elif msg["role"] == "assistant" and display:
#             display[-1] = (display[-1][0], msg["content"])
#     return display

# def send_message(session_id, text, image):
#     if not text and not image:
#         return session_id, [], None, ""
    
#     response_parts = []
    
#     if image:
#         img_result = analyze_image(image, session_id)
#         response_parts.append(f"📸 Image Analysis:\n{img_result}")
#         chat_manager.add_message(session_id, "user", "[Image Attached]")
    
#     if text:
#         text_response = generate_response(text, session_id)
#         response_parts.append(text_response)
    
#     full_response = "\n\n".join(response_parts)
#     if text:
#         chat_manager.add_message(session_id, "assistant", full_response)
    
#     messages = chat_manager.get_session(session_id)["messages"]
#     return session_id, format_chat_display(messages), None, ""

# def new_chat():
#     session_id = chat_manager.create_session()
#     return session_id, [], None, ""

# def load_chat(evt: gr.SelectData):
#     session_id = evt.value["id"]
#     messages = chat_manager.get_session(session_id)["messages"]
#     return session_id, format_chat_display(messages), None, ""

# def get_session_list():
#     return [{"title": s["title"], "id": s["id"]} 
#             for s in chat_manager.get_all_sessions()]

# with gr.Blocks(theme=gr.themes.Soft(), title="AI Chat Assistant") as app:
#     session_id = gr.State()
    
#     with gr.Row():
#         with gr.Column(scale=1, min_width=250):
#             gr.Markdown("### Chat Sessions")
#             new_chat_btn = gr.Button("+ New Chat", size="sm")
#             sessions_list = gr.Dataset(
#                 components=[gr.Textbox(visible=False)],
#                 label="Your Chats",
#                 samples=get_session_list(),
#                 elem_id="sessions-list"
#             )
        
#         with gr.Column(scale=4):
#             chatbot = gr.Chatbot(
#                 elem_id="chatbot",
#                 bubble_full_width=False,
#                 height="70vh"
#             )
            
#             with gr.Row():
#                 image_input = gr.Image(
#                     type="pil",
#                     label="Upload Image",
#                     visible=False
#                 )
#                 upload_btn = gr.UploadButton(
#                     "📷 Upload Image",
#                     file_types=["image"],
#                     variant="secondary"
#                 )
            
#             with gr.Row():
#                 text_input = gr.Textbox(
#                     placeholder="Type your message...",
#                     container=False,
#                     scale=7
#                 )
#                 submit_btn = gr.Button("Send", variant="primary", scale=1)
    
#     # Initialize with first session if exists
#     if chat_manager.get_all_sessions():
#         app.load(
#             lambda: (
#                 chat_manager.get_all_sessions()[0]["id"],
#                 format_chat_display(chat_manager.get_all_sessions()[0]["messages"]),
#                 None,
#                 ""
#             ),
#             outputs=[session_id, chatbot, image_input, text_input]
#         )
#     else:
#         app.load(
#             new_chat,
#             outputs=[session_id, chatbot, image_input, text_input]
#         )
    
#     upload_btn.upload(
#         lambda f: f,
#         inputs=[upload_btn],
#         outputs=[image_input]
#     )
    
#     submit_btn.click(
#         send_message,
#         inputs=[session_id, text_input, image_input],
#         outputs=[session_id, chatbot, image_input, text_input]
#     ).then(
#         lambda: gr.Dataset.update(samples=get_session_list()),
#         outputs=[sessions_list]
#     )
    
#     text_input.submit(
#         send_message,
#         inputs=[session_id, text_input, image_input],
#         outputs=[session_id, chatbot, image_input, text_input]
#     ).then(
#         lambda: gr.Dataset.update(samples=get_session_list()),
#         outputs=[sessions_list]
#     )
    
#     new_chat_btn.click(
#         new_chat,
#         outputs=[session_id, chatbot, image_input, text_input]
#     ).then(
#         lambda: gr.Dataset.update(samples=get_session_list()),
#         outputs=[sessions_list]
#     )
    
#     sessions_list.select(
#         load_chat,
#         outputs=[session_id, chatbot, image_input, text_input]
#     )

# if __name__ == "__main__":
#     # First ensure Ollama is running
#     print("Please ensure Ollama is running in another terminal with:")
#     print("ollama serve")
#     app.launch(server_port=7860, share=False)