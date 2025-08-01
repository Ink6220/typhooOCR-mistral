# typhoon_tranformers_app_API.py
"""
This script processes a PDF or image file by extracting its layout and content,
then sends it to the OpenTyphoon API for OCR and document understanding.
The final result is saved as a text file.

The core PDF processing logic is adapted from https://github.com/allenai/olmocr
Under the Apache 2.0 license. All credit goes to the original authors.
"""

# --- 1. Imports ---
import base64
import json
import os
import re
import random
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import List, Literal, Callable

# ไลบรารีสำหรับโหลดไฟล์ .env
from dotenv import load_dotenv

import ftfy
from openai import OpenAI
from PIL import Image
from pypdf import PdfReader
from pypdf.generic import RectangleObject

# --- 2. Load Environment Variables & API Configuration ---
# โหลดตัวแปรจากไฟล์ .env เข้าสู่ environment ของระบบ
load_dotenv()

# ดึงค่า API Key จาก Environment Variable ที่โหลดมา
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"


# --- 3. Data Classes for PDF Structure ---
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
        return BoundingBox(rect[0], rect[1], rect[2], rect[3])

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


# --- 4. PDF and Image Processing Utilities (from olmocr) ---
def image_to_pdf(image_path):
    try:
        img = Image.open(image_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            filename = tmp.name
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(filename, "PDF")
            return filename
    except Exception as conv_err:
        print(f"Error converting image to PDF: {conv_err}")
        return None

def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    command = ["pdfinfo", "-f", str(page_num), "-l", str(page_num), "-box", "-enc", "UTF-8", local_pdf_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")
    output = result.stdout
    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str: List[str] = line.split(":")[1].strip().split()
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])
    raise ValueError("MediaBox not found in the PDF info.")

def render_pdf_to_base64png(local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048) -> str:
    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm", "-png", "-f", str(page_num), "-l", str(page_num),
            "-r", str(target_longest_image_dim * 72 / longest_dim),
            local_pdf_path,
        ],
        timeout=120, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")

def _cleanup_element_text(element_text: str) -> str:
    MAX_TEXT_ELEMENT_LENGTH = 250
    TEXT_REPLACEMENTS = {"[": "\\[", "]": "\\]", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    text_replacement_pattern = re.compile("|".join(re.escape(key) for key in TEXT_REPLACEMENTS.keys()))
    element_text = ftfy.fix_text(element_text).strip()
    element_text = text_replacement_pattern.sub(lambda match: TEXT_REPLACEMENTS[match.group(0)], element_text)
    if len(element_text) > MAX_TEXT_ELEMENT_LENGTH:
        head_length = MAX_TEXT_ELEMENT_LENGTH // 2 - 3
        tail_length = head_length
        head = element_text[:head_length].rsplit(" ", 1)[0] or element_text[:head_length]
        tail = element_text[-tail_length:].split(" ", 1)[-1] or element_text[-tail_length:]
        return f"{head} ... {tail}"
    return element_text

def _merge_image_elements(images: List[ImageElement], tolerance: float = 0.5) -> List[ImageElement]:
    if not images: return []
    n = len(images)
    parent = list(range(n))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_j] = root_i
    def bboxes_overlap(b1: BoundingBox, b2: BoundingBox) -> bool:
        return (max(b1.x0, b2.x0) - min(b1.x1, b2.x1) <= tolerance and
                max(b1.y0, b2.y0) - min(b1.y1, b2.y1) <= tolerance)
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(images[i].bbox, images[j].bbox):
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    merged_images = []
    for indices in groups.values():
        merged_bbox = images[indices[0]].bbox
        for idx in indices[1:]:
            bbox = images[idx].bbox
            merged_bbox = BoundingBox(
                x0=min(merged_bbox.x0, bbox.x0), y0=min(merged_bbox.y0, bbox.y0),
                x1=max(merged_bbox.x1, bbox.x1), y1=max(merged_bbox.y1, bbox.y1),
            )
        merged_images.append(ImageElement(name="+".join(images[i].name for i in indices), bbox=merged_bbox))
    return merged_images

def _linearize_pdf_report(report: PageReport, max_length: int = 4000) -> str:
    result = f"Page dimensions: {report.mediabox.x1:.1f}x{report.mediabox.y1:.1f}\n"
    if max_length < 20: return result
    images = _merge_image_elements(report.image_elements)
    all_elements = []
    for element in images:
        s = f"[Image {element.bbox.x0:.0f}x{element.bbox.y0:.0f} to {element.bbox.x1:.0f}x{element.bbox.y1:.0f}]\n"
        all_elements.append(((element.bbox.y0, element.bbox.x0), s))
    for element in report.text_elements:
        if len(element.text.strip()) == 0: continue
        element_text = _cleanup_element_text(element.text)
        s = f"[{element.x:.0f}x{element.y:.0f}]{element_text}\n"
        all_elements.append(((element.y, element.x), s))
    
    all_elements.sort(key=lambda x: x[0])
    
    for _, s in all_elements:
        if len(result) + len(s) > max_length:
            break
        result += s
    return result

def _pdf_report(local_pdf_path: str, page_num: int) -> PageReport:
    reader = PdfReader(local_pdf_path)
    page = reader.pages[page_num - 1]
    text_elements, image_elements = [], []
    def visitor_body(text, cm, tm, font_dict, font_size):
        if text.strip():
            text_elements.append(TextElement(text, tm[4], tm[5]))
    page.extract_text(visitor_text=visitor_body)
    return PageReport(
        mediabox=BoundingBox.from_rectangle(page.mediabox),
        text_elements=text_elements,
        image_elements=[],
    )

def get_anchor_text(local_pdf_path: str, page: int, pdf_engine: str, target_length: int = 4000) -> str:
    assert page > 0, "Pages are 1-indexed"
    if pdf_engine == "pdfreport":
        return _linearize_pdf_report(_pdf_report(local_pdf_path, page), max_length=target_length)
    else:
        raise NotImplementedError(f"Unknown engine: {pdf_engine}")


# --- 5. Prompt Generation Logic ---
PROMPTS_SYS = {
    "default": lambda base_text: (
        f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
    "structure": lambda base_text: (
        f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
}

def get_prompt(prompt_name: str) -> Callable[[str], str]:
    return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")


# --- 6. Main Processing Function (API Call) ---
def process_pdf_via_api(pdf_or_image_file: str, page_num: int, task_type: str):
    if pdf_or_image_file is None:
        return None, "No file provided"
    
    if not os.path.exists(pdf_or_image_file):
         return None, f"File not found: {pdf_or_image_file}"

    print(f"Processing file: {pdf_or_image_file}...")
    temp_pdf_created = False
    filename = pdf_or_image_file

    if not filename.lower().endswith(".pdf"):
        print("Input is an image, converting to temporary PDF...")
        filename = image_to_pdf(pdf_or_image_file)
        if filename is None:
            return None, "Error converting image to PDF"
        temp_pdf_created = True

    try:
        print(f"Rendering page {page_num} to image...")
        image_base64 = render_pdf_to_base64png(filename, page_num, target_longest_image_dim=1800)
        image_pil = Image.open(BytesIO(base64.b64decode(image_base64)))

        print("Extracting anchor text from PDF structure...")
        anchor_text = get_anchor_text(filename, page_num, pdf_engine="pdfreport", target_length=8000)

        prompt_template_fn = get_prompt(task_type)
        prompt = prompt_template_fn(anchor_text)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }]
        
        print("Sending request to Typhoon API...")
        openai = OpenAI(base_url=TYPHOON_BASE_URL, api_key=TYPHOON_API_KEY)
        response = openai.chat.completions.create(
            model="typhoon-ocr-preview",
            messages=messages,
            max_tokens=4096,
            extra_body={
                "repetition_penalty": 1.2,
                "temperature": 0.1,
                "top_p": 0.6,
            },
        )
        text_output = response.choices[0].message.content
        print("API response received.")

        try:
            json_data = json.loads(text_output)
            markdown_out = json_data.get('natural_text', "").replace("<figure>", "").replace("</figure>", "")
        except (json.JSONDecodeError, AttributeError):
            print("⚠️ Could not parse JSON from output, returning raw text.")
            markdown_out = f"⚠️ Could not extract `natural_text` from output.\nRaw output:\n{text_output}"

        return image_pil, markdown_out

    finally:
        if temp_pdf_created and os.path.exists(filename):
            print(f"Cleaning up temporary file: {filename}")
            os.remove(filename)


# --- 7. Execution Block ---
if __name__ == "__main__":
    # ตรวจสอบว่ามี API Key ใน environment หรือไม่
    if not TYPHOON_API_KEY:
        print("❌ Error: ไม่พบ TYPHOON_API_KEY")
        print("กรุณาสร้างไฟล์ .env และเพิ่ม 'TYPHOON_API_KEY=YOUR_KEY' ลงในไฟล์")
    else:
        # --- กำหนดค่า ที่นี่ ---
        # ระบุชื่อไฟล์รูปภาพหรือ PDF ของคุณ
        input_file = "/home/sayakaray/ByteDance_Dolphin/Dolphin_copy/examples/00002089.pdf" 
        page_to_process = 1
        task = "structure"
        # ----------------------

        image_result, markdown_result = process_pdf_via_api(
            pdf_or_image_file=input_file,
            page_num=page_to_process,
            task_type=task
        )

        if markdown_result:
            print("\n--- OCR Result ---")
            print(markdown_result)
            
            output_folder = 'ocr_results'
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_filename = f'{base_name}_page_{page_to_process}.txt'
            full_path = os.path.join(output_folder, output_filename)
            
            os.makedirs(output_folder, exist_ok=True)
            
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_result)
                print(f"\n✅ บันทึกไฟล์เรียบร้อยที่: {full_path}")
            except Exception as e:
                print(f"\n❌ เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")
        else:
            print(f"Could not process the file '{input_file}'.")