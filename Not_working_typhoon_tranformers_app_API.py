# typhoon_tranformers_app_API.py
import io
import os
import re
import json
import base64
from loguru import logger
import cv2
import pymupdf
from PIL import Image

# imports สำหรับ local model
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ตัวแปร global สำหรับเก็บโมเดลและ processor
model = None
processor = None

def initialize_local_typhoon_model():
    """
    โหลดโมเดล Typhoon OCR และ processor สำหรับการประมวลผลบน CPU
    """
    global model, processor
    if model is not None:
        logger.info("Local model is already initialized.")
        return

    logger.info("Initializing local Typhoon OCR model for CPU processing...")
    logger.warning("Loading a 7B model on CPU will consume significant RAM and be very slow.")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "scb10x/typhoon-ocr-7b",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        processor = AutoProcessor.from_pretrained(
            "scb10x/typhoon-ocr-7b",
            use_fast=True
        )
        logger.info("Local Typhoon OCR model loaded successfully on CPU.")
    except Exception as e:
        logger.error(f"Failed to load local model: {e}", exc_info=True)
        model = None
        processor = None

def run_local_typhoon_inference(prompt: str, image: Image.Image) -> str:
    """
    ประมวลผลภาพด้วยโมเดル Typhoon ที่โหลดไว้ในเครื่อง (Local Inference)
    """
    if not model or not processor:
        logger.error("Local model is not initialized.")
        return "[Error: Local model not available.]"

    try:
        logger.info(f"Running local inference...")
        
        # ======================= [การแก้ไข] =======================
        # เพิ่ม placeholder ของรูปภาพเข้าไปใน list ของ messages
        # เพื่อให้ apply_chat_template รู้ว่ามีรูปภาพอยู่ในการสนทนานี้ด้วย
        # แม้ว่าเราจะส่งอ็อบเจกต์รูปภาพจริงๆ ในขั้นตอนถัดไปก็ตาม
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # Placeholder นี้สำคัญมากในการแก้ปัญหา token mismatch
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ],
        }]
        
        text_template = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # ==========================================================

        inputs = processor(
            text=[text_template],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to("cpu")

        output = model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=4096,
            repetition_penalty=1.2,
            do_sample=True,
        )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        logger.info("Local inference completed.")
        return text_output

    except Exception as e:
        logger.error(f"An error occurred during local inference: {e}", exc_info=True)
        return f"[Error: An unexpected error occurred during local inference. {e}]"


def convert_to_image(file_path: str, page_num: int = 0, dpi: int = 300) -> Image.Image | None:
    if not file_path or not os.path.exists(file_path):
        logger.error(f"File not found at path: {file_path}")
        return None
    try:
        if file_path.lower().endswith('.pdf'):
            with pymupdf.open(file_path) as doc:
                if page_num >= len(doc): page_num = 0
                page = doc[page_num]
                pix = page.get_pixmap(dpi=dpi)
                return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            return Image.open(file_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error converting file to image for path '{file_path}': {e}", exc_info=True)
        return None

# --- Dummy Utility Functions ---
def parse_layout_string(layout_str: str) -> list:
    return [("full_page", "para")]

def convert_results_to_html(results: list) -> str:
    html_parts = [res.get('text', '').replace('\n', '<br>') for res in results]
    return "\n".join(html_parts)

class MarkdownConverter:
    def convert(self, results: list) -> str:
        return "\n\n".join([res.get('text', '') for res in results])
# --- End of Dummy Utility Functions ---

def process_data_step2(file_path: str, coordinates_json: str, page_num: int):
    if not file_path or not coordinates_json:
        return "กรุณาอัปโหลดไฟล์ และสกัดพิกัดในขั้นตอนที่ 1 ก่อน", "", {}
    if not model or not processor:
        error_msg = "# Error\n\nLocal model is not initialized. Please check the console log for errors."
        return error_msg, "Model not ready", {"error": True, "message": "Model not ready"}

    page_idx = int(page_num) - 1
    logger.info(f"[Local Typhoon] Processing page {page_idx + 1}...")

    try:
        pil_image = convert_to_image(file_path, page_num=page_idx, dpi=300)
        if not pil_image:
            raise Exception("Could not convert file to image for processing.")

        layout_data = parse_layout_string(coordinates_json)
        recognition_results = []

        for i, (bbox, initial_label) in enumerate(layout_data):
            logger.info(f"Processing full page based on '{initial_label}' label...")
            text_prompt = "จงอ่านข้อความทั้งหมดในภาพนี้ และตอบกลับมาเป็นข้อความธรรมดา (Plain Text) เท่านั้น"
            final_text = run_local_typhoon_inference(text_prompt, pil_image)

            recognition_results.append({
                "bbox": "full_page",
                "label": "para",
                "reading_order": i,
                "text": final_text
            })
            break

        final_html_render = convert_results_to_html(recognition_results)
        converter = MarkdownConverter()
        final_md_content = converter.convert(recognition_results)
        final_json_output = {"results": recognition_results, "page": page_idx + 1}

        return final_html_render, final_md_content, final_json_output

    except Exception as e:
        logger.error(f"Error in Typhoon Step 2 for page {page_idx + 1}: {e}", exc_info=True)
        error_msg = f"# Error\n\nAn unexpected error occurred: {str(e)}"
        return error_msg, str(e), {"error": True, "message": str(e)}