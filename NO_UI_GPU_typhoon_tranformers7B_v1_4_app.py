# NO_UI_GPU_typhoon_tranformers7B_v1_4.py
import io
import os
import tempfile
import time
import json
import base64
import re

import cv2
import pymupdf
import torch
from loguru import logger
from PIL import Image

from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel,
    Qwen2_5_VLForConditionalGeneration
)

# หมายเหตุ: ตรวจสอบให้แน่ใจว่าคุณมีไฟล์ utils.py และ markdown_utils.py อยู่ในโฟลเดอร์ utils/
from utils.utils import prepare_image, parse_layout_string, process_coordinates
from utils.markdown_utils import MarkdownConverter

# --- การตั้งค่าทั่วไป (GPU Version) ---
# ตรวจสอบว่ามี GPU (CUDA) ให้ใช้หรือไม่ ถ้าไม่มีให้ใช้ CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

OCR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_results")
os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)


# --- ส่วนสำหรับโมเดล ---
model, processor, tokenizer = None, None, None
thai_model, thai_processor = None, None

def initialize_model():
    """โหลดโมเดล Dolphin สำหรับวิเคราะห์ Layout"""
    global model, processor, tokenizer
    if model is None:
        logger.info("Loading DOLPHIN model...")
        model_id = "ByteDance/Dolphin"
        processor = AutoProcessor.from_pretrained(model_id)
        # โหลดโมเดลสำหรับ GPU (ใช้ half-precision)
        model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=torch.float16)
        model.eval()
        model.to(device)
        tokenizer = processor.tokenizer
        logger.info(f"Dolphin model loaded successfully on {device}")


def initialize_thai_model():
    """โหลดโมเดล Typhoon-OCR สำหรับอ่านภาษาไทย"""
    global thai_model, thai_processor
    if thai_model is None:
        logger.info(f"Loading Thai Typhoon-OCR model via Transformers to device: {device}")
        try:
            model_name = "Adun/typhoon_ocr-7B-v1.4"
            # โหลดโมเดลสำหรับ GPU (ใช้ half-precision)
            thai_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
            thai_model.to(device)
            thai_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            logger.info(f"Thai model '{model_name}' (Transformers) loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Thai Typhoon-OCR (Transformers) model: {e}", exc_info=True)

# (ส่วนที่เหลือของโค้ดเหมือนเดิมทั้งหมด)
# --- ฟังก์ชันประมวลผลหลักของโมเดล ---
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

# --- ฟังก์ชันเสริม ---
def cleanup_temp_file(file_path):
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file: {e}")

def convert_to_image(file_path, page_num=0, dpi=200):
    if not file_path: return None, None
    temp_image_path = None
    try:
        if file_path.lower().endswith('.pdf'):
            with pymupdf.open(file_path) as doc:
                if page_num >= len(doc): page_num = 0
                page = doc[page_num]; pix = page.get_pixmap(dpi=dpi)
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_image_path = tmp.name
                    pil_image.save(temp_image_path, "PNG")
                return temp_image_path, pil_image
        else: # หากเป็นไฟล์รูปภาพอยู่แล้ว
            pil_image = Image.open(file_path).convert("RGB")
            return file_path, pil_image
    except Exception as e:
        logger.error(f"Error converting file to image: {e}");
        if temp_image_path: cleanup_temp_file(temp_image_path);
        return None, None

def convert_results_to_html(results: list) -> str:
    """แปลง list ของผลลัพธ์ OCR ให้เป็นสตริง HTML เดียวที่สมบูรณ์"""
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
    """ฟังก์ชันสำหรับกรองและสกัดข้อความจากผลลัพธ์ OCR ที่อาจมี JSON ปนอยู่"""
    try:
        match = re.search(r'\{.*\}', raw_html, re.DOTALL)
        if match:
            json_string = match.group(0)
            data = json.loads(json_string)
            if 'natural_text' in data and data['natural_text']:
                cleaned_text = data['natural_text'].replace('\n', ' ').strip()
                return cleaned_text
    except (json.JSONDecodeError, TypeError):
        return re.sub(r'<[^>]+>', '', raw_html).strip()
    return re.sub(r'<[^>]+>', '', raw_html).strip()


# --- Pipeline การทำงานหลัก ---
def run_ocr_pipeline(file_path: str, page_num: int):
    """
    ฟังก์ชันหลักสำหรับรันกระบวนการ OCR ทั้งหมด ตั้งแต่การหา Layout จนถึงการสกัดข้อความ
    
    Args:
        file_path (str): พาธของไฟล์ PDF หรือรูปภาพ
        page_num (int): หมายเลขหน้า (เริ่มต้นที่ 1)
        
    Returns:
        str: ผลลัพธ์ในรูปแบบ HTML หรือข้อความแสดงข้อผิดพลาด
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    page_idx = page_num - 1
    
    # === ขั้นตอนที่ 1: วิเคราะห์ Layout ของเอกสาร ===
    logger.info(f"[Step 1] Extracting layout for page {page_num} from '{os.path.basename(file_path)}'...")
    layout_temp_path, layout_image = convert_to_image(file_path, page_num=page_idx, dpi=150)
    if not layout_image:
        return "# Error: Could not convert file to image for layout extraction."

    try:
        layout_json_str = model_chat("Parse the reading order of this document.", layout_image)
        layout_data = parse_layout_string(layout_json_str) # มาจาก utils.py
        if not layout_data:
            return f"# Error: Failed to parse layout from model. Response was: {layout_json_str}"
        logger.info(f"Successfully extracted layout for {len(layout_data)} elements.")
    except Exception as e:
        logger.error(f"Error in Step 1 (Layout Extraction): {e}", exc_info=True)
        return f"# Error: {e}"
    finally:
        if layout_temp_path and layout_temp_path != file_path:
            cleanup_temp_file(layout_temp_path)

    # === ขั้นตอนที่ 2: ทำ OCR ตาม Layout ที่ได้ ===
    logger.info(f"[Step 2] Processing OCR for page {page_num} with 2-step verification...")
    ocr_temp_path, ocr_image = convert_to_image(file_path, page_num=page_idx, dpi=300)
    if not ocr_image:
        return "# Error: Could not convert file to high-resolution image for OCR."
        
    try:
        padded_image, dims = prepare_image(ocr_image) # มาจาก utils.py
        recognition_results = []

        for i, (bbox, initial_label) in enumerate(layout_data):
            logger.info(f"  -> Processing element {i+1}/{len(layout_data)} (Suggested: '{initial_label}')")

            # ใช้ process_coordinates จาก utils.py
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, _ = process_coordinates(bbox, padded_image, dims)
            pil_crop = Image.fromarray(cv2.cvtColor(padded_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

            final_label = initial_label
            final_text = ""

            if initial_label == 'fig':
                buffered = io.BytesIO()
                pil_crop.save(buffered, format="PNG")
                final_text = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                # 2-Step Verification: 1. ตรวจสอบประเภท -> 2. แปลงข้อมูล
                classification_prompt = "วิเคราะห์ภาพนี้ และตอบกลับมาเพียงคำเดียว: ถ้าเป็น 'ตาราง' ให้ตอบว่า TABLE, ถ้าเป็น 'ข้อความ' ให้ตอบว่า TEXT"
                type_response = process_with_thai_transformer_model(classification_prompt, pil_crop)
                
                if "TABLE" in type_response.upper():
                    final_label = 'tab'
                    html_prompt = "แปลงตารางในภาพนี้เป็นโค้ด HTML ที่สมบูรณ์และถูกต้องเท่านั้น โดยคำตอบต้องขึ้นต้นด้วย `<table>` และจบด้วย `</table>` ห้ามมีคำอธิบายอื่น"
                    final_text = process_with_thai_transformer_model(html_prompt, pil_crop)
                else:
                    final_label = 'para'
                    text_prompt = "จงอ่านข้อความทั้งหมดในภาพนี้ และตอบกลับมาเป็นข้อความธรรมดา (Plain Text) เท่านั้น **ห้าม** สรุป, เปลี่ยนแปลง, หรือเพิ่มข้อมูลใดๆ ที่ไม่มีอยู่ในภาพโดยเด็ดขาด"
                    raw_ocr_text = process_with_thai_transformer_model(text_prompt, pil_crop)
                    final_text = clean_and_extract_text(raw_ocr_text)

            recognition_results.append({
                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2], 
                "label": final_label, 
                "reading_order": i, 
                "text": final_text
            })

        # --- สร้างผลลัพธ์สุดท้ายและบันทึกไฟล์ ---
        final_html_output = convert_results_to_html(recognition_results)
        
        converter = MarkdownConverter() # มาจาก markdown_utils.py
        page_md = converter.convert(recognition_results)
        timestamp = int(time.time())
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_filename}_page_{page_num}_{timestamp}.txt"
        output_filepath = os.path.join(OCR_OUTPUT_DIR, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(page_md)
        
        logger.info(f"OCR result saved to: {output_filepath}")
        return final_html_output

    except Exception as e:
        logger.error(f"Error in Step 2 (OCR Processing): {e}", exc_info=True)
        return f"# Error: {str(e)}"
    finally:
        if ocr_temp_path and ocr_temp_path != file_path:
            cleanup_temp_file(ocr_temp_path)


# --- จุดเริ่มต้นการทำงานของสคริปต์ ---
if __name__ == '__main__':
    # --- ตัวอย่างการเรียกใช้งาน ---
    # 1. กำหนดค่าพาธของไฟล์และหน้า
    # TEST_FILE_PATH = "path/to/your/document.pdf"  # <-- แก้ไขตรงนี้
    # TEST_PAGE_NUMBER = 1                             # <-- แก้ไขตรงนี้

    # print("กรุณาแก้ไขตัวแปร TEST_FILE_PATH และ TEST_PAGE_NUMBER ในสคริปต์ก่อนรัน")
    # print("โปรแกรมจะยังไม่ทำงานจนกว่าจะเอาเครื่องหมาย comment (#) หน้าโค้ดตัวอย่างออก")

    # if 'TEST_FILE_PATH' in locals() and os.path.exists(TEST_FILE_PATH):
    #     # 2. โหลดโมเดล (ทำครั้งเดียวตอนเริ่ม)
    #     print("Initializing models... (อาจใช้เวลาสักครู่ในการโหลดครั้งแรก)")
    #     initialize_model()
    #     initialize_thai_model()
    #     print("Models initialized successfully.")
        
    #     # 3. รันกระบวนการ OCR
    #     print(f"\nStarting OCR process for '{TEST_FILE_PATH}', page {TEST_PAGE_NUMBER}...")
    #     start_time = time.time()
    #     final_result = run_ocr_pipeline(file_path=TEST_FILE_PATH, page_num=TEST_PAGE_NUMBER)
    #     end_time = time.time()
        
    #     # 4. พิมพ์ผลลัพธ์
    #     print("\n" + "="*25 + " OCR Result (HTML) " + "="*25)
    #     print(final_result)
    #     print("="*68)
    #     print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    # else:
    #     if 'TEST_FILE_PATH' in locals():
    #         print(f"File not found: {TEST_FILE_PATH}")
    
    print("Script is ready. To run, uncomment the code block in `if __name__ == '__main__':` and set your file path.")