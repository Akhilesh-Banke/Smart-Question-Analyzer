import io
import logging
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from src.ocr_engine import extract_text_from_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text directly from a text-based PDF.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                logger.debug(f"Text extracted from page {page_num}")
                text_pages.append(text)
        return "\n".join(text_pages)
    except Exception as e:
        logger.error(f"Error extracting text-based PDF: {e}")
        return ""


def process_pdf(uploaded_file) -> str:
    """
    Main PDF processing function.
    Attempts text extraction first; if little or no text, uses OCR.
    """
    try:
        file_bytes = uploaded_file.read()
        logger.info("Starting PDF text extraction...")

        # Attempt direct extraction first
        text = extract_text_from_pdf(file_bytes)

        # If text is too short, switch to OCR
        if len(text.strip()) < 100:
            logger.info("Text-based extraction failed; switching to OCR.")
            images = convert_from_bytes(file_bytes)
            image_bytes_list = []
            for i, img in enumerate(images):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes_list.append(buf.getvalue())
                logger.debug(f"Converted page {i + 1} to image for OCR.")

            text = extract_text_from_images(image_bytes_list)

        logger.info("PDF processing complete.")
        return text.strip()

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return ""
