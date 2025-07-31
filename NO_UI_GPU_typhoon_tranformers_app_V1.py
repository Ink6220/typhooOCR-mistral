# NO_UI_GPU_typhoon_tranformers_app_V1.py
import io
import os
import tempfile
import time
import json
import base64
import re
import torch

import cv2
import pymupdf
from loguru import logger
from PIL import Image

from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel,
    Qwen2_5_VLForConditionalGeneration
)

# สมมติว่าไฟล์เหล่านี้อยู่ในโฟลเดอร์ utils/
from utils.utils import prepare_image, parse_layout_string, process_coordinates
from utils.markdown_utils import MarkdownConverter

# --- การตั้งค่าทั่วไป ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

OCR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_results")
os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)


# --- ส่วนสำหรับโมเดล ---
model, processor, tokenizer = None, None, None
thai_model, thai_processor = None, None

def initialize_model():
    """โหลดโมเดล Dolphin"""
    global model, processor, tokenizer
    if model is None:
        logger.info("Loading DOLPHIN model...")
        model_id = "ByteDance/Dolphin"
        processor = AutoProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        model.eval()
        model.to(device)
        tokenizer = processor.tokenizer
        logger.info(f"Dolphin model loaded successfully on {device}")


def initialize_thai_model():
    """โหลดโมเดลภาษาไทยผ่าน Transformers"""
    global thai_model, thai_processor
    if thai_model is None:
        logger.info(f"Loading Thai Typhoon-OCR model via Transformers to device: {device}")
        try:
            model_name = "scb10x/typhoon-ocr-7b"
            thai_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
            )
            thai_model.to(device)
            thai_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            logger.info(f"Thai model '{model_name}' (Transformers) loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Thai Typhoon-OCR (Transformers) model: {e}", exc_info=True)


# --- ฟังก์ชันประมวลผล ---
def model_chat(prompt, image):
    """เรียกใช้งานโมเดล Dolphin"""
    device_dolphin = model.device
    batch_pixel_values = processor(images=[image], return_tensors="pt").pixel_values.to(device_dolphin)
    prompt_inputs = tokenizer([f"<s>{prompt} <Answer/>"], add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(
        pixel_values=batch_pixel_values,
        decoder_input_ids=prompt_inputs.input_ids.to(device_dolphin),
        decoder_attention_mask=prompt_inputs.attention_mask.to(device_dolphin),
        max_length=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        repetition_penalty=1.1
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("<Answer/>")[-1].strip()

def process_with_thai_transformer_model(prompt: str, image: Image.Image) -> str:
    """เรียกใช้งานโมเดลภาษาไทยผ่าน Transformers"""
    if not thai_model or not thai_processor: return "[Error: Thai model not loaded.]"
    model_device = thai_model.device
    messages = [
        {"role": "system", "content": "You are an expert assistant that can analyze documents and images. Provide the answer in Thai."},
        {"role": "user", "content": [{"type": "image_url", "image_url": "data:image/jpeg;base64,dummy"}, {"type": "text", "text": prompt}]}
    ]
    text_template = thai_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = thai_processor(text=[text_template], images=[image], padding=True, return_tensors="pt").to(model_device)
    output = thai_model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    return thai_processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def extract_text_from_model_response(response_str: str) -> str:
    response_str = response_str.strip()
    match = re.search(r'\{.*\}', response_str, re.DOTALL)
    if match:
        try:
            json_data = json.loads(match.group(0))
            return json_data.get('natural_text', response_str).strip().strip('"')
        except json.JSONDecodeError:
            pass
    return response_str

# --- ฟังก์ชันเสริม ---
def cleanup_temp_file(file_path):
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except OSError as e:
            logger.warning(f"Could not clean up temp file: {e}")

def convert_to_image(file_path, page_num=0, dpi=200):
    temp_file = None
    try:
        if file_path.lower().endswith('.pdf'):
            with pymupdf.open(file_path) as doc:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi)
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_file = tmp.name
                    pil_image.save(temp_file, "PNG")
                return temp_file, pil_image
        else:
            pil_image = Image.open(file_path).convert("RGB")
            return file_path, pil_image
    except Exception as e:
        logger.error(f"Error converting file to image: {e}")
        if temp_file: cleanup_temp_file(temp_file)
        return None, None

def convert_results_to_html(results: list) -> str:
    html_parts = []
    for elem in results:
        text = elem.get('text', '')
        if not text: continue
        label = elem.get('label', 'para')
        if label == 'tab':
            html_parts.append(f'<pre><code>{text}</code></pre>')
        elif label in ['title', 'sec', 'sub_sec']:
            level = {'title': 2, 'sec': 3, 'sub_sec': 4}.get(label, 2)
            html_parts.append(f'<h{level}>{text}</h{level}>')
        elif label == 'fig':
            html_parts.append(f'<img src="data:image/png;base64,{text}" alt="Figure" style="max-width:100%;">')
        else:
            html_parts.append(f'<p>{text.replace(chr(10), "<br>")}</p>')
    return "\n".join(html_parts)

def html_table_to_markdown(html_content: str) -> str:
    if "<table>" not in html_content: return html_content
    try:
        content = html_content.replace('\n', '').replace('  ', ' ')
        rows = re.findall(r'<tr.*?>(.*?)</tr>', content)
        md_lines = []
        header_done = False
        for row_html in rows:
            cells = re.findall(r'<(?:th|td).*?>(.*?)</(?:th|td)>', row_html)
            if not cells: continue
            cleaned = [re.sub(r'<.*?>', '', c).strip() for c in cells]
            md_lines.append("| " + " | ".join(cleaned) + " |")
            if re.search(r'<th', row_html) and not header_done:
                md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                header_done = True
        return "\n".join(md_lines)
    except Exception as e:
        logger.error(f"Failed to convert HTML table to Markdown: {e}")
        return html_content

# --- Pipeline การทำงานหลัก ---
def run_ocr_pipeline(file_path: str, page_num: int):
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    page_idx = page_num - 1
    
    # === ขั้นตอนที่ 1: วิเคราะห์ Layout ===
    logger.info(f"[Step 1] Extracting layout for page {page_num}...")
    layout_temp_path, layout_image = convert_to_image(file_path, page_num=page_idx, dpi=150)
    if not layout_image: return "# Error: Could not create image for layout analysis."

    try:
        layout_json = model_chat("Parse the reading order of this document.", layout_image)
        layout_data = parse_layout_string(layout_json)
        if not layout_data: return f"# Error: Failed to parse layout. Response: {layout_json}"
        logger.info(f"Layout extracted for {len(layout_data)} elements.")
    finally:
        if layout_temp_path and layout_temp_path != file_path:
            cleanup_temp_file(layout_temp_path)

    # === ขั้นตอนที่ 2: ทำ OCR ตาม Layout ===
    logger.info(f"[Step 2] Processing OCR for page {page_num}...")
    ocr_temp_path, ocr_image = convert_to_image(file_path, page_num=page_idx, dpi=300)
    if not ocr_image: return "# Error: Could not create high-res image for OCR."

    try:
        padded_image, dims = prepare_image(ocr_image)
        recognition_results = []
        for i, (bbox, initial_label) in enumerate(layout_data):
            logger.info(f"  -> Processing element {i+1}/{len(layout_data)} ('{initial_label}')...")
            
            bbox_coords = json.loads(bbox.replace("'", "\"")) if isinstance(bbox, str) else bbox
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, _ = process_coordinates(bbox_coords, padded_image, dims)
            pil_crop = Image.fromarray(cv2.cvtColor(padded_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

            final_text, final_label = "", initial_label
            if initial_label == 'fig':
                buffered = io.BytesIO()
                pil_crop.save(buffered, format="PNG")
                final_text = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                type_response = process_with_thai_transformer_model("วิเคราะห์ภาพนี้ และตอบกลับมาเพียงคำเดียว: ถ้าเป็น 'ตาราง' ให้ตอบว่า TABLE, ถ้าเป็น 'ข้อความ' ให้ตอบว่า TEXT", pil_crop)
                
                if "TABLE" in type_response.upper():
                    final_label = 'tab'
                    prompt = "แปลงตารางในภาพนี้ให้เป็นตารางข้อความธรรมดา (Plain Text Table) ที่จัดรูปแบบให้สวยงาม โดยใช้สัญลักษณ์ไปป์ (|) คั่นระหว่างคอลัมน์ รักษาโครงสร้างและข้อมูลให้ครบถ้วน ห้ามมีคำอธิบายอื่นใดๆ ทั้งสิ้น"
                    raw_text = process_with_thai_transformer_model(prompt, pil_crop)
                    cleaned_text = extract_text_from_model_response(raw_text)
                    final_text = html_table_to_markdown(cleaned_text) if "<table>" in cleaned_text else cleaned_text
                else:
                    final_label = 'para'
                    prompt = "จงอ่านข้อความทั้งหมดในภาพนี้ และตอบกลับมาเป็นข้อความธรรมดา (Plain Text) เท่านั้น **ห้าม** สรุป, เปลี่ยนแปลง, หรือเพิ่มข้อมูลใดๆ ที่ไม่มีอยู่ในภาพโดยเด็ดขาด"
                    raw_text = process_with_thai_transformer_model(prompt, pil_crop)
                    final_text = extract_text_from_model_response(raw_text)

            recognition_results.append({"bbox": [orig_x1, orig_y1, orig_x2, orig_y2], "label": final_label, "reading_order": i, "text": final_text})

        # --- สร้างผลลัพธ์สุดท้ายและบันทึกไฟล์ ---
        final_html_output = convert_results_to_html(recognition_results)
        
        page_md = MarkdownConverter().convert(recognition_results)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_filename}_page_{page_num}_{int(time.time())}.txt"
        output_filepath = os.path.join(OCR_OUTPUT_DIR, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(page_md)
        logger.info(f"OCR result saved to: {output_filepath}")
        return final_html_output
    finally:
        if ocr_temp_path and ocr_temp_path != file_path:
            cleanup_temp_file(ocr_temp_path)

# --- จุดเริ่มต้นการทำงาน ---
if __name__ == '__main__':
    # --- ตัวอย่างการเรียกใช้งาน ---
    # 1. กำหนดค่าพาธของไฟล์และหน้าที่ต้องการ
    # TEST_FILE_PATH = "path/to/your/document.pdf"  # <-- แก้ไขตรงนี้
    # TEST_PAGE_NUMBER = 1                             # <-- แก้ไขตรงนี้

    # print("กรุณาแก้ไขตัวแปร TEST_FILE_PATH และ TEST_PAGE_NUMBER ในสคริปต์ก่อนรัน")

    # if 'TEST_FILE_PATH' in locals() and os.path.exists(TEST_FILE_PATH):
    #     print("Initializing models... (อาจใช้เวลาสักครู่ในการโหลดครั้งแรก)")
    #     initialize_model()
    #     initialize_thai_model()
    #     print("Models initialized successfully.")
        
    #     print(f"\nStarting OCR process for '{TEST_FILE_PATH}', page {TEST_PAGE_NUMBER}...")
    #     start_time = time.time()
    #     final_result = run_ocr_pipeline(file_path=TEST_FILE_PATH, page_num=TEST_PAGE_NUMBER)
        
    #     print("\n" + "="*25 + " OCR Result (HTML) " + "="*25)
    #     print(final_result)
    #     print("="*68)
    #     print(f"Processing completed in {time.time() - start_time:.2f} seconds.")
    # else:
    #     if 'TEST_FILE_PATH' in locals():
    #         print(f"File not found: {TEST_FILE_PATH}")
    
    print("Script is ready. To run, uncomment the code block in `if __name__ == '__main__':` and set your file path.")