import fitz # PyMuPDF
from PIL import Image
import io


def pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes()))
        images.append(img)
    return images