
"""
OCR Engine for extracting text from PDF page images.
- Default: Uses Tesseract (offline, free)
- Optional: Gemini 1.5 Flash (for advanced OCR or complex layouts)
"""
import pytesseract
from PIL import Image
import streamlit as st
import hashlib
import google.generativeai as genai
from src.config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)


def _hash_image(image: Image.Image) -> str:
    """Return a unique hash for an image to cache results."""
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


@st.cache_data(show_spinner=False)
def ocr_image(image: Image.Image, use_gemini: bool = False) -> str:
    """
    Extract text from an image page using Tesseract or Gemini OCR.
    Cached for faster reprocessing.
    """
    img_hash = _hash_image(image)
    print(f"[INFO] OCR Cache key: {img_hash}")

    if use_gemini:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                [image, "Extract all readable text clearly and accurately."]
            )
            return response.text.strip()
        except Exception as e:
            print(f"[WARN] Gemini OCR failed: {e}, falling back to Tesseract.")

    # fallback to pytesseract
    return pytesseract.image_to_string(image)


def batch_ocr(images, use_gemini=False):
    """Apply OCR to a list of images."""
    texts = []
    for img in images:
        text = ocr_image(img, use_gemini=use_gemini)
        texts.append(text)
    return texts
