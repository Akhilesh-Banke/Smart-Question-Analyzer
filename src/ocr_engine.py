import numpy as np
from PIL import Image
import pytesseract
import easyocr
from src.config import CONFIG


# initialize easyocr reader lazily
_reader = None


def _get_easyocr_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'])
    return _reader


def ocr_image(pil_img):
    engine = CONFIG.get('ocr_engine', 'tesseract')
    if engine == 'easyocr':
        reader = _get_easyocr_reader()
        res = reader.readtext(np.array(pil_img))
        return "\n".join([r[1] for r in res])
    else:
        # tesseract default
        return pytesseract.image_to_string(pil_img)