import os
import json
import gradio as gr
from deepface import DeepFace
from PIL import Image

# Path to known faces
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load face info from JSON
INFO_FILE = "face_info.json"
if os.path.exists(INFO_FILE):
    with open(INFO_FILE, "r") as f:
        face_info = json.load(f)
else:
    face_info = {}

# Normalize names for consistent key usage
def normalize_name(name):
    return name.lower().replace(" ", "_")

# Save uploaded image to file
def save_uploaded_image(image):
    if image is None:
        raise ValueError("⚠️ No image was uploaded.")
    path = "uploaded.jpg"
    image.save(path)
    return path

# Main recognition logic
def recognize_and_describe(image):
    if image is None:
        return "⚠️ Please upload an image."

    try:
        img_path = save_uploaded_image(image)

        results = DeepFace.find(
            img_path=img_path,
            db_path=KNOWN_FACES_DIR,
            enforce_detection=False
        )

        if len(results) > 0 and len(results[0]) > 0:
            best_match_path = results[0].iloc[0]['identity']
            matched_filename = os.path.basename(best_match_path)
            matched_name = os.path.splitext(matched_filename)[0]
            key = normalize_name(matched_name)
            info = face_info.get(key, "📝 Info not available.")
            return f"✅ Match found: {matched_name.replace('_', ' ').title()}\n\n🧠 Info: {info}"
        else:
            return "❌ No matching face found."

    except Exception as e:
        return f"❌ Error: {str(e)}"

    finally:
        # Clean up the uploaded image
        if os.path.exists("uploaded.jpg"):
            os.remove("uploaded.jpg")

# Gradio UI setup
demo = gr.Interface(
    fn=recognize_and_describe,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Recognition Result"),
    title="🧠 Face Recognition App",
    description="Upload a photo to identify the person and get related information."
)

if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)
