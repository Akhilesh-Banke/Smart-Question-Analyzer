import pytesseract
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extracts text from a single image using Tesseract OCR.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""


def extract_text_from_images(images: list[bytes]) -> str:
    """
    Extracts and concatenates text from multiple images.
    """
    all_text = []
    for idx, img_bytes in enumerate(images):
        text = extract_text_from_image(img_bytes)
        if text:
            logger.debug(f"OCR success on page {idx + 1}")
            all_text.append(text)
        else:
            logger.warning(f"OCR returned empty text on page {idx + 1}")
    return "\n\n".join(all_text)
