# utils/typhoon_utils.py
# โค้ดส่วนใหญ่นำมาจาก https://github.com/allenai/olmocr ภายใต้ Apache 2.0 license
# ปรับปรุงเพื่อใช้ในโปรเจกต์นี้

from dataclasses import dataclass
import re
import tempfile
from PIL import Image
import subprocess
import base64
from typing import List, Literal, Tuple
import random
import ftfy
from pypdf.generic import RectangleObject
from pypdf import PdfReader
from loguru import logger

# --- Data Structures ---
@dataclass(frozen=True)
class Element:
    pass

@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @staticmethod
    def from_rectangle(rect: RectangleObject) -> "BoundingBox":
        return BoundingBox(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))

@dataclass(frozen=True)
class TextElement(Element):
    text: str
    x: float
    y: float

@dataclass(frozen=True)
class ImageElement(Element):
    name: str
    bbox: BoundingBox

@dataclass(frozen=True)
class PageReport:
    mediabox: BoundingBox
    text_elements: List[TextElement]
    image_elements: List[ImageElement]

# --- PDF Report Generation ---

def _transform_point(x, y, m):
    x_new = m[0] * x + m[2] * y + m[4]
    y_new = m[1] * x + m[3] * y + m[5]
    return x_new, y_new

def _mult(m: List[float], n: List[float]) -> List[float]:
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]

def _pdf_report(local_pdf_path: str, page_num: int) -> PageReport:
    reader = PdfReader(local_pdf_path)
    page = reader.pages[page_num - 1]
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    text_elements, image_elements = [], []

    def visitor_body(text, cm, tm, font_dict, font_size):
        if text.strip(): # เก็บเฉพาะ text ที่มีเนื้อหา
            txt2user = _mult(tm, cm)
            text_elements.append(TextElement(text, txt2user[4], txt2user[5]))

    def visitor_op(op, args, cm, tm):
        if op == b"Do":
            try:
                xobject_name = args[0]
                xobject = xobjects.get(xobject_name)
                if xobject and xobject.get("/Subtype") == "/Image":
                    x0, y0 = _transform_point(0, 0, cm)
                    x1, y1 = _transform_point(1, 1, cm)
                    image_elements.append(ImageElement(xobject_name, BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))
            except Exception as e:
                logger.warning(f"Could not process image in PDF: {e}")


    page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    return PageReport(
        mediabox=BoundingBox.from_rectangle(page.mediabox),
        text_elements=text_elements,
        image_elements=image_elements,
    )

def _cap_split_string(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    head_length = max_length // 2 - 3
    tail_length = head_length
    head = text[:head_length].rsplit(" ", 1)[0] or text[:head_length]
    tail = text[-tail_length:].split(" ", 1)[-1] or text[-tail_length:]
    return f"{head} ... {tail}"

def _cleanup_element_text(element_text: str) -> str:
    MAX_TEXT_ELEMENT_LENGTH = 250
    TEXT_REPLACEMENTS = {"[": "\\[", "]": "\\]", "\n": " ", "\r": " ", "\t": " "}
    text_replacement_pattern = re.compile("|".join(re.escape(key) for key in TEXT_REPLACEMENTS.keys()))
    element_text = ftfy.fix_text(element_text).strip()
    element_text = text_replacement_pattern.sub(lambda match: TEXT_REPLACEMENTS[match.group(0)], element_text)
    return _cap_split_string(element_text, MAX_TEXT_ELEMENT_LENGTH)

def _linearize_pdf_report(report: PageReport, max_length: int = 8000) -> str:
    result = ""
    result += f"Page dimensions: {report.mediabox.x1:.1f}x{report.mediabox.y1:.1f}\n"
    
    all_elements = []
    for element in report.image_elements:
        image_str = f"[Image {element.bbox.x0:.0f}x{element.bbox.y0:.0f} to {element.bbox.x1:.0f}x{element.bbox.y1:.0f}]\n"
        all_elements.append(((element.bbox.y0, element.bbox.x0), image_str))

    for element in report.text_elements:
        if len(element.text.strip()) == 0:
            continue
        element_text = _cleanup_element_text(element.text)
        text_str = f"[{element.x:.0f}x{element.y:.0f}]{element_text}\n"
        all_elements.append(((element.y, element.x), text_str))

    # Sort elements by y then x coordinates (top-to-bottom, left-to-right)
    all_elements.sort(key=lambda x: x[0])
    
    for _, s in all_elements:
        if len(result) + len(s) > max_length:
            break
        result += s
        
    return result

# --- Main public function ---
def extract_and_linearize_pdf_report(local_pdf_path: str, page_num: int = 1, max_length: int = 8000) -> str:
    """
    Main function to extract a structured text report from a PDF page.
    """
    logger.info(f"Generating PDF report for {local_pdf_path} page {page_num}...")
    report = _pdf_report(local_pdf_path, page_num)
    linearized_text = _linearize_pdf_report(report, max_length)
    logger.info(f"Generated linearized report, length: {len(linearized_text)} chars.")
    return linearized_text