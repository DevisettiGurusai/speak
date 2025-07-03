import os
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from gtts import gTTS
import torch

# Setup upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set Streamlit page config
st.set_page_config(page_title="SpeakSnap 🎤", layout="centered")
st.title("📸 SpeakSnap - Multilingual Image Captioning")

# Load models once with caching
@st.cache_resource
def load_captioning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

@st.cache_resource
def load_translation_model():
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return model, tokenizer

caption_model, caption_processor, caption_tokenizer = load_captioning_model()
translation_model, translation_tokenizer = load_translation_model()

# Supported languages
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
    "Arabic": "ar"
}

# Captioning
def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=16)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Translation
def translate_caption(caption, target_lang_code):
    translation_tokenizer.src_lang = "en"
    encoded = translation_tokenizer(caption, return_tensors="pt")
    generated_tokens = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code))
    translated = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

# Text-to-Speech
def speak_caption(text, lang_code, path="uploads/speech.mp3"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(path)
        return path
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# UI
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
language = st.selectbox("Choose Output Language", list(LANGUAGES.keys()))

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    if st.button("Generate Caption 🎤"):
        with st.spinner("Processing..."):
            caption = generate_caption(image)
            lang_code = LANGUAGES[language]

            if lang_code != "en":
                caption = translate_caption(caption, lang_code)

            st.success(f"Caption in {language}: {caption}")

            # Generate speech
            audio_path = speak_caption(caption, lang_code)
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
