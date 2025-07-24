import io
import os
import tempfile
import time
import uuid
import json

import cv2
import gradio as gr
import pymupdf
import spaces
import torch
from gradio_pdf import PDF
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

from utils.utils import prepare_image, parse_layout_string, process_coordinates, ImageDimensions
from utils.markdown_utils import MarkdownConverter

# อ่าน外部CSS文件
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# 全局变量存储模型
model = None
processor = None
tokenizer = None

# 自动初始化模型
@spaces.GPU
def initialize_model():
    """初始化 Hugging Face 模型"""
    global model, processor, tokenizer
    
    if model is None:
        logger.info("Loading DOLPHIN model...")
        model_id = "ByteDance/Dolphin"
        
        # 加载处理器和模型
        processor = AutoProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        model.eval()
        
        # 强制使用 CPU เพื่อหลีกเลี่ยงปัญหา CUDA
        device = torch.device("cpu")
        model.to(device)
        # model = model.half()  # การใช้ half() บน CPU อาจไม่ได้รับการรองรับดีนัก จึงคอมเมนต์ไว้ก่อน
        
        # 设置tokenizer
        tokenizer = processor.tokenizer
        
        logger.info(f"Model loaded successfully on {device}")
    
    return "Model ready"

# 启动时自动初始化模型
logger.info("Initializing model at startup...")
try:
    initialize_model()
    logger.info("Model initialization completed")
except Exception as e:
    logger.error(f"Model initialization failed: {e}")

# 模型推理函数
@spaces.GPU
def model_chat(prompt, image):
    """使用模型进行推理"""
    global model, processor, tokenizer
    
    if model is None:
        initialize_model()
    
    is_batch = isinstance(image, list)
    
    if not is_batch:
        images = [image]
        prompts = [prompt]
    else:
        images = image
        prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
    
    device = torch.device("cpu")
    # batch_inputs = processor(images, return_tensors="pt", padding=True)
    # batch_pixel_values = batch_inputs.pixel_values.half().to(device)
    batch_pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

    prompts = [f"<s>{p} <Answer/>" for p in prompts]
    batch_prompt_inputs = tokenizer(
        prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True
    )

    batch_prompt_ids = batch_prompt_inputs.input_ids.to(device)
    batch_attention_mask = batch_prompt_inputs.attention_mask.to(device)
    
    outputs = model.generate(
        pixel_values=batch_pixel_values,
        decoder_input_ids=batch_prompt_ids,
        decoder_attention_mask=batch_attention_mask,
        min_length=1,
        max_length=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.1
    )
    
    sequences = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
    
    results = []
    for i, sequence in enumerate(sequences):
        cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
        results.append(cleaned)
        
    return results[0] if not is_batch else results

# --- ฟังก์ชันที่ปรับปรุงใหม่ตามโจทย์ ---

@spaces.GPU
def extract_bounding_boxes(file_path):
    """
    ขั้นตอนที่ 1: รับไฟล์เข้ามาแล้วใช้โมเดลสกัดพิกัด (bounding boxes)
    ขององค์ประกอบต่างๆ ในเอกสารหน้าแรก
    """
    if file_path is None:
        return "กรุณาอัปโหลดไฟล์ก่อน"
        
    logger.info(f"[Step 1] Extracting coordinates from: {file_path}")
    
    try:
        # แปลงไฟล์เป็นรูปภาพ (ใช้หน้าแรกเสมอสำหรับขั้นตอนนี้)
        image_path = convert_to_image(file_path, page_num=0)
        if not image_path:
            raise Exception("ไม่สามารถแปลงไฟล์เป็นรูปภาพได้")
            
        pil_image = Image.open(image_path).convert("RGB")
        
        # ใช้ prompt พิเศษเพื่อให้โมเดลคืนค่าเป็น layout/reading order
        layout_output = model_chat("Parse the reading order of this document.", pil_image)
        
        # จัดรูปแบบผลลัพธ์ให้เป็น JSON ที่สวยงาม
        try:
            parsed_json = json.loads(layout_output)
            pretty_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            logger.info("[Step 1] Successfully extracted and formatted coordinates.")
            return pretty_json
        except json.JSONDecodeError:
            logger.warning("[Step 1] Output is not a valid JSON, returning raw string.")
            return layout_output

    except Exception as e:
        logger.error(f"[Step 1] Error extracting coordinates: {str(e)}")
        return f"เกิดข้อผิดพลาดระหว่างการสกัดพิกัด: {str(e)}"
    finally:
        # Cleanup temp file if created
        if 'image_path' in locals() and image_path != file_path:
            cleanup_temp_file(image_path)


@spaces.GPU(duration=120)
def process_from_coordinates(file_path, coordinates_json):
    """
    ขั้นตอนที่ 2: รับไฟล์และข้อมูลพิกัดมาประมวลผลเพื่อสร้างผลลัพธ์สุดท้าย
    """
    if file_path is None or not coordinates_json:
        return "กรุณาอัปโหลดไฟล์และสกัดพิกัดก่อน", "", []

    logger.info(f"[Step 2] Processing from coordinates for: {file_path}")

    try:
        # แปลงไฟล์เป็นรูปภาพ (ใช้หน้าแรกเสมอสำหรับขั้นตอนนี้)
        image_path = convert_to_image(file_path, page_num=0)
        if not image_path:
            raise Exception("ไม่สามารถแปลงไฟล์เป็นรูปภาพได้")

        # โค้ดส่วนนี้ยกมาจาก process_page และ process_elements เดิม
        pil_image = Image.open(image_path).convert("RGB")
        padded_image, dims = prepare_image(pil_image)
        
        # เรียกใช้ฟังก์ชัน process_elements เดิมซึ่งเป็นหัวใจของการประมวลผล
        recognition_results = process_elements(coordinates_json, padded_image, dims)
        
        # สร้าง Markdown จากผลลัพธ์
        final_markdown = generate_markdown(recognition_results)
        
        logger.info("[Step 2] Successfully processed document from coordinates.")
        return final_markdown, final_markdown, recognition_results

    except Exception as e:
        logger.error(f"[Step 2] Error processing from coordinates: {str(e)}")
        error_data = [{"error": True, "message": str(e)}]
        return f"# เกิดข้อผิดพลาด\n\n{str(e)}", "", error_data
    finally:
        if 'image_path' in locals() and image_path != file_path:
            cleanup_temp_file(image_path)

# --- ฟังก์ชัน Helper เดิม (ไม่มีการเปลี่ยนแปลง) ---

def process_element_batch(elements, prompt, max_batch_size=16):
    results = []
    batch_size = min(len(elements), max_batch_size)
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        prompts_list = [prompt] * len(crops_list)
        batch_results = model_chat(prompts_list, crops_list)
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"], "bbox": elem["bbox"],
                "text": result.strip(), "reading_order": elem["reading_order"],
            })
    return results

def cleanup_temp_file(file_path):
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

def convert_to_image(file_path, target_size=896, page_num=0):
    if file_path is None: return None
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            doc = pymupdf.open(file_path)
            if page_num >= len(doc): page_num = 0
            page = doc[page_num]
            rect = page.rect
            scale = target_size / max(rect.width, rect.height)
            mat = pymupdf.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                pil_image.save(tmp_file.name, "PNG")
                doc.close()
                return tmp_file.name
        else:
            pil_image = Image.open(file_path).convert("RGB")
            w, h = pil_image.size
            if max(w, h) > target_size:
                if w > h: new_w, new_h = target_size, int(h * target_size / w)
                else: new_w, new_h = int(w * target_size / h), target_size
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if max(w, h) <= target_size: return file_path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                pil_image.save(tmp_file.name, "PNG")
                return tmp_file.name
    except Exception as e:
        logger.error(f"Error converting file to image: {e}")
        return file_path

def process_elements(layout_results, padded_image, dims, max_batch_size=16):
    layout_results = parse_layout_string(layout_results)
    text_elements, table_elements, figure_results = [], [], []
    previous_box = None
    reading_order = 0
    for bbox, label in layout_results:
        try:
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and (cropped.shape[0] > 3 and cropped.shape[1] > 3):
                if label == "fig":
                    try:
                        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        import base64
                        buffered = io.BytesIO()
                        pil_crop.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        figure_results.append({
                            "label": label, "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "text": img_base64, "reading_order": reading_order,
                        })
                    except Exception as e:
                        logger.error(f"Error encoding figure to base64: {e}")
                else:
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop, "label": label, "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    if label == "tab": table_elements.append(element_info)
                    else: text_elements.append(element_info)
            reading_order += 1
        except Exception as e:
            logger.error(f"Error processing bbox with label {label}: {str(e)}")
            continue
    recognition_results = figure_results.copy()
    if text_elements:
        recognition_results.extend(process_element_batch(text_elements, "Read text in the image.", max_batch_size))
    if table_elements:
        recognition_results.extend(process_element_batch(table_elements, "Parse the table in the image.", max_batch_size))
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))
    return recognition_results

def generate_markdown(recognition_results):
    converter = MarkdownConverter()
    return converter.convert(recognition_results)

def preview_file(file_path):
    """
    ฟังก์ชันสำหรับแสดงไฟล์ตัวอย่าง โดยจะแปลงไฟล์รูปภาพเป็น PDF ชั่วคราว
    เพื่อให้คอมโพเนนต์ PDF สามารถแสดงผลได้
    """
    if file_path is None:
        return None
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # ถ้าเป็นไฟล์ PDF อยู่แล้ว ให้ส่งคืน path ไปตรงๆ
            return file_path
        else:
            # ถ้าเป็นไฟล์รูปภาพ ให้แปลงเป็น PDF ก่อน
            logger.info(f"Converting image to PDF for preview: {file_path}")
            
            pil_image = Image.open(file_path).convert("RGB")
            
            # สร้างไฟล์ PDF ชั่วคราว
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                pil_image.save(tmp_file.name, "PDF", resolution=100.0)
                return tmp_file.name
                
    except Exception as e:
        logger.error(f"Error creating preview file: {e}")
        return None

# --- ส่วนของ Gradio Interface ที่ปรับปรุงใหม่ ---

latex_delimiters = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]
custom_css = load_css()
with open(os.path.join(os.path.dirname(__file__), "header.html"), "r", encoding="utf-8") as file:
    header = file.read()

with gr.Blocks(css=custom_css, title="Dolphin Document Parser (2-Step)") as demo:
    gr.HTML(header)
    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            file = gr.File(label="Choose PDF or image file", file_types=[".pdf", ".png", ".jpeg", ".jpg"], elem_id="file-upload")
            
            # --- ส่วนควบคุมที่ปรับปรุงใหม่ ---
            gr.Markdown("### 🕵️ ขั้นตอนที่ 1: สกัดพิกัด")
            extract_coords_button = gr.Button("1. Extract Coordinates", variant="secondary")
            
            gr.Markdown("### ⚙️ ขั้นตอนที่ 2: แปลงข้อมูล")
            process_final_button = gr.Button("2. Process from Coordinates", variant="primary")
            
            clear_btn = gr.ClearButton(value="清空/Clear")
            
            coordinates_output = gr.Textbox(
                label="Coordinates JSON", lines=15, interactive=True,
                info="ผลลัพธ์พิกัดจากขั้นตอนที่ 1 จะแสดงที่นี่"
            )
            
            example_root = os.path.join(os.path.dirname(__file__), "examples")
            if os.path.exists(example_root):
                gr.Examples(
                    examples=[os.path.join(example_root, f) for f in os.listdir(example_root) if not f.endswith(".py")],
                    inputs=file, examples_per_page=10, elem_id="example-files"
                )

        with gr.Column(scale=7):
            with gr.Row(elem_classes="main-content"):
                with gr.Column(scale=1, elem_classes="preview-panel"):
                    gr.HTML("文件预览/Preview")
                    pdf_show = PDF(label="", interactive=False, visible=True, height=600)
                with gr.Column(scale=1, elem_classes="output-panel"):
                    with gr.Tabs():
                        with gr.Tab("Markdown [Render]"):
                            md_render = gr.Markdown(label="", height=700, show_copy_button=True, latex_delimiters=latex_delimiters, line_breaks=True)
                        with gr.Tab("Markdown [Content]"):
                            md_content = gr.TextArea(lines=30, show_copy_button=True)
                        with gr.Tab("Json [Content]"):
                            json_output = gr.JSON(label="", height=700)

    # --- การเชื่อมโยง Event ของปุ่มต่างๆ ---
    
    # เมื่อไฟล์เปลี่ยน ให้เรียกใช้ฟังก์ชัน preview_file เพื่อสร้างไฟล์ตัวอย่าง
    file.change(fn=preview_file, inputs=file, outputs=pdf_show, queue=True)
    
    # ปุ่มขั้นตอนที่ 1
    extract_coords_button.click(
        fn=extract_bounding_boxes,
        inputs=[file],
        outputs=[coordinates_output]
    )

    # ปุ่มขั้นตอนที่ 2
    process_final_button.click(
        fn=process_from_coordinates,
        inputs=[file, coordinates_output],
        outputs=[md_render, md_content, json_output]
    )

    # ปุ่ม Clear
    def reset_all():
        return None, None, "", "", "", []
    clear_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[file, pdf_show, coordinates_output, md_render, md_content, json_output]
    )

if __name__ == "__main__":
    demo.launch()