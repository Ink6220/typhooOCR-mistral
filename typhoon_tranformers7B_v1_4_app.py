# typhoon_tranformers7B_v1_4_app.py
import io
import os
import tempfile
import time
import json
import base64
import re

import cv2
import gradio as gr
import pymupdf
import spaces
import torch
from gradio_pdf import PDF
from loguru import logger
from PIL import Image

from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel,
    Qwen2_5_VLForConditionalGeneration
)

from utils.utils import prepare_image, parse_layout_string, process_coordinates, ImageDimensions
from utils.markdown_utils import MarkdownConverter

# --- การตั้งค่าทั่วไป ---
device = "cpu"
logger.info(f"Using device: {device}")

OCR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_results")
os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)


# --- ส่วนสำหรับโมเดล ---
model, processor, tokenizer = None, None, None
thai_model, thai_processor = None, None

def load_css(): return ""

def initialize_model():
    """โหลดโมเดล Dolphin"""
    global model, processor, tokenizer
    if model is None:
        logger.info("Loading DOLPHIN model...")
        model_id = "ByteDance/Dolphin"
        processor = AutoProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        model.eval()
        model.to(device)
        tokenizer = processor.tokenizer
        logger.info(f"Dolphin model loaded successfully on {device}")


def initialize_thai_model():
    """โหลดโมเดลภาษาไทยผ่าน Transformers บน CPU"""
    global thai_model, thai_processor
    if thai_model is None:
        logger.info(f"Loading Thai Typhoon-OCR model via Transformers to device: {device}")
        try:
            model_name = "Adun/typhoon_ocr-7B-v1.4"
            thai_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
            thai_model.to(device)
            thai_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            logger.info(f"Thai model '{model_name}' (Transformers) loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Thai Typhoon-OCR (Transformers) model: {e}", exc_info=True)
            thai_model = None
            thai_processor = None

logger.info("Initializing models at startup...")
initialize_model()
initialize_thai_model()

# --- ฟังก์ชันประมวลผล ---
def model_chat(prompt, image):
    """เรียกใช้งานโมเดล Dolphin"""
    is_batch = isinstance(image, list); images = [image] if not is_batch else image
    prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
    device_dolphin = model.device
    batch_pixel_values = processor(images, return_tensors="pt").pixel_values.to(device_dolphin)
    prompts = [f"<s>{p} <Answer/>" for p in prompts]; batch_prompt_inputs = tokenizer(prompts, add_special_tokens=False, return_tensors="pt")
    batch_prompt_ids = batch_prompt_inputs.input_ids.to(device_dolphin); batch_attention_mask = batch_prompt_inputs.attention_mask.to(device_dolphin)
    outputs = model.generate(pixel_values=batch_pixel_values, decoder_input_ids=batch_prompt_ids, decoder_attention_mask=batch_attention_mask, max_length=4096, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True, num_beams=1, repetition_penalty=1.1)
    sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = [seq.split("<Answer/>")[-1].strip() for seq in sequences]
    return results[0] if not is_batch else results

def process_with_thai_transformer_model(prompt: str, image: Image.Image) -> str:
    """เรียกใช้งานโมเดลภาษาไทยผ่าน Transformers"""
    if not thai_model or not thai_processor: return "[Error: Thai model not loaded.]"
    try:
        model_device = thai_model.device
        messages = [
            {"role": "system", "content": "You are an expert assistant that can analyze documents and images. Provide the answer in Thai."},
            {"role": "user", "content": [{"type": "image_url", "image_url": "data:image/jpeg;base64,dummy"}, {"type": "text", "text": prompt}]}
        ]
        text_template = thai_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = thai_processor(text=[text_template], images=[image], padding=True, return_tensors="pt").to(model_device)
        input_length = inputs['input_ids'].shape[1]
        output = thai_model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_tokens = output[0, input_length:]
        assistant_response = thai_processor.decode(generated_tokens, skip_special_tokens=True).strip()
        return assistant_response
    except Exception as e:
        logger.error(f"Error during Thai Transformers model processing: {e}", exc_info=True)
        return f"[Error: {e}]"

def extract_text_from_model_response(response_str: str) -> str:
    try:
        response_str = response_str.strip()

        # กรณีเจอแพทเทิร์น ${...}$
        special_pattern_match = re.search(r'^\$\{(.*)\}\$$', response_str, re.DOTALL)
        if special_pattern_match:
            inner_json = "{" + special_pattern_match.group(1) + "}"
            try:
                json_data = json.loads(inner_json)
                if 'natural_text' in json_data:
                    return json_data['natural_text'].strip().strip('"')
            except json.JSONDecodeError:
                pass

        # กรณีเจอ JSON ธรรมดา
        generic_json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if generic_json_match:
            try:
                json_data = json.loads(generic_json_match.group(0))
                if 'natural_text' in json_data:
                    return json_data['natural_text'].strip().strip('"')
                if 'html' in json_data:
                    return json_data['html']
                if 'ภาษา' in json_data:
                    return json_data['ภาษา']
            except json.JSONDecodeError:
                pass

        # ไม่เจออะไร ให้คืนค่าต้นฉบับ
        return response_str

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return response_str


# --- ฟังก์ชันเสริม ---
def cleanup_temp_file(file_path):
    if file_path and os.path.exists(file_path) and "gradio" in file_path:
        try: os.unlink(file_path)
        except Exception as e: logger.warning(f"Could not clean up temp file: {e}")

def get_page_count(file_path):
    if not file_path: return 1
    try:
        if file_path.lower().endswith('.pdf'):
            with pymupdf.open(file_path) as doc: return len(doc)
    except Exception: pass
    return 1

def convert_to_image(file_path, page_num=0, dpi=200):
    if not file_path: return None
    temp_file = None
    try:
        if file_path.lower().endswith('.pdf'):
            with pymupdf.open(file_path) as doc:
                if page_num >= len(doc): page_num = 0
                page = doc[page_num]; pix = page.get_pixmap(dpi=dpi)
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_file = tmp.name; pil_image.save(temp_file, "PNG")
                return temp_file
        else: return file_path
    except Exception as e:
        logger.error(f"Error converting PDF to image: {e}");
        if temp_file: cleanup_temp_file(temp_file);
        return None

def convert_results_to_html(results: list) -> str:
    """
    แปลง list ของผลลัพธ์ OCR ให้เป็นสตริง HTML เดียวที่สมบูรณ์
    """
    html_parts = []
    for elem in results:
        text = elem.get('text', '')
        label = elem.get('label', 'para')
        if not text: continue
        if label == 'tab':
            html_parts.append(text)
        elif label in ['title', 'sec', 'sub_sec']:
            level = {'title': 2, 'sec': 3, 'sub_sec': 4}.get(label, 2)
            html_parts.append(f'<h{level}>{text}</h{level}>')
        elif label == 'fig':
            html_parts.append(f'<img src="data:image/png;base64,{text}" alt="Figure" style="max-width:100%;">')
        else:
            paragraph_text = text.replace('\n', '<br>')
            html_parts.append(f'<p>{paragraph_text}</p>')
    return "\n".join(html_parts)

def clean_and_extract_text(raw_html: str) -> str:
    """
    ฟังก์ชันสำหรับกรองและสกัดข้อความจากผลลัพธ์ OCR ที่อาจมี JSON ปนอยู่

    Args:
        raw_html: สตริงดิบที่ได้จาก OCR ซึ่งอาจมีลักษณะเป็น <p>{...}</p>

    Returns:
        สตริงข้อความที่ถูกทำความสะอาดแล้ว หรือสตริงเดิมหากไม่เข้าเงื่อนไข
    """
    try:
        # 1. ค้นหาเนื้อหาที่เป็น JSON ที่อาจซ่อนอยู่ในสตริง
        # ใช้ Regular Expression เพื่อหาทุกอย่างที่อยู่ระหว่าง { และ }
        match = re.search(r'\{.*\}', raw_html)

        if match:
            json_string = match.group(0)
            
            # 2. แปลงสตริง JSON เป็น Python Dictionary
            data = json.loads(json_string)
            
            # 3. ตรวจสอบและดึงค่าจากคีย์ 'natural_text'
            if 'natural_text' in data and data['natural_text']:
                # 4. ทำความสะอาดข้อความ: แทนที่การขึ้นบรรทัดใหม่ด้วยการเว้นวรรค
                # และลบช่องว่างที่ไม่จำเป็นที่หัวและท้ายของประโยค
                cleaned_text = data['natural_text'].replace('\n', ' ').strip()
                return cleaned_text

    except (json.JSONDecodeError, TypeError):
        # หากเกิดข้อผิดพลาดในการแปลง JSON หรือไม่พบข้อมูลตามรูปแบบ
        # ให้คืนค่าสตริงเดิมโดยลบแท็ก HTML ที่อาจมีออกไป
        return re.sub(r'<[^>]+>', '', raw_html).strip()

    # หากไม่พบ JSON ในสตริง ให้คืนค่าเดิมโดยลบแท็ก HTML ออก
    return re.sub(r'<[^>]+>', '', raw_html).strip()

# --- ฟังก์ชันสำหรับ UI ---

def extract_coordinates_step1(file_path, force_thai, page_num):
    if not file_path: return "กรุณาอัปโหลดไฟล์", "en", "Language: N/A"
    page_idx = int(page_num) - 1; logger.info(f"[Step 1] Extracting coordinates for page {page_idx + 1}...")
    temp_image_path = None
    try:
        temp_image_path = convert_to_image(file_path, page_num=page_idx, dpi=150)
        if not temp_image_path: raise Exception("Could not convert file to image.")
        pil_image = Image.open(temp_image_path).convert("RGB")
        layout_output = model_chat("Parse the reading order of this document.", pil_image)
        lang = 'th' if force_thai else 'en'
        lang_info = f"Language for Step 2: {lang.upper()}"
        return layout_output, lang, lang_info
    except Exception as e:
        logger.error(f"Error in Step 1 for page {page_idx + 1}: {e}", exc_info=True)
        return f"# Error\n\n{e}", "en", "Error"
    finally:
        if temp_image_path and temp_image_path != file_path: cleanup_temp_file(temp_image_path)


def process_data_step2(file_path, coordinates_json, language, page_num):
    """
    ประมวลผล OCR โดยใช้หลักการ "ตรวจสอบก่อนแปลงข้อมูล" เพื่อความแม่นยำสูงสุด
    """
    if not file_path or not coordinates_json:
        return "กรุณาอัปโหลดไฟล์ และสกัดพิกัดในขั้นตอนที่ 1 ก่อน", "", {}

    page_idx = int(page_num) - 1
    logger.info(f"[Step 2] Processing page {page_idx + 1} with 2-step verification OCR...")

    temp_image_path = None
    try:
        temp_image_path = convert_to_image(file_path, page_num=page_idx, dpi=300)
        if not temp_image_path: raise Exception("Could not convert file to image for processing.")

        pil_image = Image.open(temp_image_path).convert("RGB")
        padded_image, dims = prepare_image(pil_image)
        layout_data = parse_layout_string(coordinates_json)
        
        recognition_results = []

        for i, (bbox, initial_label) in enumerate(layout_data):
            logger.info(f"Processing element {i} (Dolphin suggested: '{initial_label}')...")

            bbox_coords = json.loads(bbox.replace("'", "\"")) if isinstance(bbox, str) else bbox
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, _ = process_coordinates(bbox_coords, padded_image, dims, None)
            pil_crop = Image.fromarray(cv2.cvtColor(padded_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

            final_label = initial_label
            final_text = ""

            if initial_label == 'fig':
                buffered = io.BytesIO()
                pil_crop.save(buffered, format="PNG")
                final_text = base64.b64encode(buffered.getvalue()).decode('utf-8')
                final_label = 'fig'
            else:
                # [ปรับปรุง] ใช้กระบวนการ 2 ขั้นตอน: 1. ตรวจสอบประเภท, 2. แปลงข้อมูล
                
                # --- ขั้นตอนที่ 1: ตรวจสอบประเภท ---
                classification_prompt = (
                    "วิเคราะห์ภาพนี้ และตอบกลับมาเพียงคำเดียว: ถ้าเป็น 'ตาราง' ให้ตอบว่า TABLE, "
                    "ถ้าเป็น 'ข้อความ' ให้ตอบว่า TEXT"
                )
                type_response = process_with_thai_transformer_model(classification_prompt, pil_crop)
                logger.info(f"Element {i} classification response: '{type_response}'")

                # --- ขั้นตอนที่ 2: แปลงข้อมูลตามประเภทที่ตรวจสอบได้ ---
                if "TABLE" in type_response.upper():
                    final_label = 'tab'
                    logger.info(f"Element {i} classified as TABLE. Performing HTML conversion.")
                    # ใช้ Prompt สำหรับตารางโดยเฉพาะ
                    html_prompt = (
                        "แปลงตารางในภาพนี้เป็นโค้ด HTML ที่สมบูรณ์และถูกต้องเท่านั้น โดยคำตอบต้องขึ้นต้นด้วย `<table>` "
                        "และจบด้วย `</table>` ห้ามมีคำอธิบายอื่น"
                    )
                    final_text = process_with_thai_transformer_model(html_prompt, pil_crop)
                else: # หากไม่มั่นใจหรือตอบเป็นอย่างอื่น ให้ถือว่าเป็นข้อความ
                    final_label = 'para'
                    logger.info(f"Element {i} classified as TEXT. Performing plain text OCR.")
                    # ใช้ Prompt สำหรับข้อความโดยเฉพาะ
                    text_prompt = (
                        "จงอ่านข้อความทั้งหมดในภาพนี้ และตอบกลับมาเป็นข้อความธรรมดา (Plain Text) เท่านั้น "
                        "**ห้าม** สรุป, เปลี่ยนแปลง, หรือเพิ่มข้อมูลใดๆ ที่ไม่มีอยู่ในภาพโดยเด็ดขาด"
                    )
                    raw_ocr_text = process_with_thai_transformer_model(text_prompt, pil_crop)

                    final_text = clean_and_extract_text(raw_ocr_text)

            recognition_results.append({
                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2], 
                "label": final_label, 
                "reading_order": i, 
                "text": final_text
            })

        final_output = convert_results_to_html(recognition_results)
        
        converter = MarkdownConverter()
        page_md = converter.convert(recognition_results)
        timestamp = int(time.time())
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_filename}_page_{page_num}_{timestamp}.txt"
        output_filepath = os.path.join(OCR_OUTPUT_DIR, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(page_md)
        
        logger.info(f"OCR result saved to: {output_filepath}")
        
        return final_output, final_output, {"results": recognition_results, "page": page_idx + 1}
        
    except Exception as e:
        logger.error(f"Error in Step 2 for page {page_idx + 1}: {e}", exc_info=True)
        return f"# Error\n\n{str(e)}", "", {"error": True, "message": str(e)}
    finally:
        if temp_image_path and temp_image_path != file_path:
            cleanup_temp_file(temp_image_path)