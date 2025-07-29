import io
import os
import tempfile
import time
import uuid
import json 
import re 

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

# 读取外部CSS文件
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
        
        # 设置设备和精度
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device("cpu")
        model.to(device)
        model = model.half()  # 使用半精度
        
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
    # 模型将在首次使用时重新尝试初始化

# 模型推理函数
@spaces.GPU
def model_chat(prompt, image):
    """使用模型进行推理"""
    global model, processor, tokenizer
    
    # 确保模型已初始化
    if model is None:
        initialize_model()
    
    # 检查是否为批处理
    is_batch = isinstance(image, list)
    
    if not is_batch:
        images = [image]
        prompts = [prompt]
    else:
        images = image
        prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
    
    # 准备图像
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cpu")
    batch_inputs = processor(images, return_tensors="pt", padding=True)
    batch_pixel_values = batch_inputs.pixel_values.half().to(device)
    
    # 准备提示
    prompts = [f"<s>{p} <Answer/>" for p in prompts]
    batch_prompt_inputs = tokenizer(
        prompts,
        add_special_tokens=False,
        return_tensors="pt"
    )

    batch_prompt_ids = batch_prompt_inputs.input_ids.to(device)
    batch_attention_mask = batch_prompt_inputs.attention_mask.to(device)
    
    # 生成文本
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
    
    # 处理输出
    sequences = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
    
    # 清理提示文本
    results = []
    for i, sequence in enumerate(sequences):
        cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
        results.append(cleaned)
        
    # 返回单个结果或批处理结果
    if not is_batch:
        return results[0]
    return results

# 处理元素批次
@spaces.GPU
def process_element_batch(elements, prompt, max_batch_size=16):
    """处理同类型元素的批次"""
    results = []
    
    # 确定批次大小
    batch_size = min(len(elements), max_batch_size)
    
    # 分批处理
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        
        # 使用相同的提示
        prompts_list = [prompt] * len(crops_list)
        
        # 批量推理
        batch_results = model_chat(prompts_list, crops_list)
        
        # 添加结果
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })
    
    return results

# 清理临时文件
def cleanup_temp_file(file_path):
    """安全地删除临时文件"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

def convert_to_image(file_path, target_size=896, page_num=0):
    """将输入文件转换为图像格式，长边调整到指定尺寸"""
    if file_path is None:
        return None
    
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # PDF文件：转换指定页面为图像
            logger.info(f"Converting PDF page {page_num} to image: {file_path}")
            doc = pymupdf.open(file_path)
            
            # 检查页面数量
            if page_num >= len(doc):
                page_num = 0  # 如果页面超出范围，使用第一页
            
            page = doc[page_num]
            
            # 计算缩放比例，使长边为target_size
            rect = page.rect
            scale = target_size / max(rect.width, rect.height)
            
            # 渲染页面为图像
            mat = pymupdf.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            
            # 转换为PIL图像
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                pil_image.save(tmp_file.name, "PNG")
                doc.close()
                return tmp_file.name
                
        else:
            # 图像文件：调整尺寸（忽略page_num参数）
            logger.info(f"Resizing image: {file_path}")
            pil_image = Image.open(file_path).convert("RGB")
            
            # 计算新尺寸，保持长宽比
            w, h = pil_image.size
            if max(w, h) > target_size:
                if w > h:
                    new_w, new_h = target_size, int(h * target_size / w)
                else:
                    new_w, new_h = int(w * target_size / h), target_size
                
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 如果已是图像且尺寸合适，直接返回原文件
            if max(w, h) <= target_size:
                return file_path
            
            # 保存调整后的图像
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                pil_image.save(tmp_file.name, "PNG")
                return tmp_file.name
                
    except Exception as e:
        logger.error(f"Error converting file to image: {e}")
        return file_path  # 如果转换失败，返回原文件

def get_pdf_page_count(file_path):
    """获取PDF文件的页数"""
    try:
        if file_path and file_path.lower().endswith('.pdf'):
            doc = pymupdf.open(file_path)
            page_count = len(doc)
            doc.close()
            return page_count
        else:
            return 1  # 非PDF文件视为单页
    except Exception as e:
        logger.error(f"Error getting PDF page count: {e}")
        return 1

def convert_all_pdf_pages_to_images(file_path, target_size=896):
    """将PDF的所有页面转换为图像列表"""
    if file_path is None:
        return []
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            doc = pymupdf.open(file_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 计算缩放比例
                rect = page.rect
                scale = target_size / max(rect.width, rect.height)
                
                # 渲染页面为图像
                mat = pymupdf.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)
                
                # 转换为PIL图像
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # 保存为临时文件
                with tempfile.NamedTemporaryFile(suffix=f"_page_{page_num}.png", delete=False) as tmp_file:
                    pil_image.save(tmp_file.name, "PNG")
                    image_paths.append(tmp_file.name)
            
            doc.close()
            return image_paths
        else:
            # 非PDF文件，返回调整后的单个图像
            converted_path = convert_to_image(file_path, target_size)
            return [converted_path] if converted_path else []
            
    except Exception as e:
        logger.error(f"Error converting PDF pages to images: {e}")
        return []

def to_pdf(file_path):
    """为了兼容性保留的函数，现在调用convert_to_image"""
    return convert_to_image(file_path)

@spaces.GPU(duration=120)
def process_document(file_path):
    """处理文档的主要函数 - 支持多页PDF处理"""
    if file_path is None:
        return "", "", []
    
    start_time = time.time()
    original_file_path = file_path
    
    # 确保模型已初始化
    if model is None:
        initialize_model()
    
    try:
        # 获取页数
        page_count = get_pdf_page_count(file_path)
        logger.info(f"Document has {page_count} page(s)")
        
        # 将所有页面转换为图像
        image_paths = convert_all_pdf_pages_to_images(file_path)
        if not image_paths:
            raise Exception("Failed to convert document to images")
        
        # 记录需要清理的临时文件
        temp_files_created = []
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            temp_files_created.extend(image_paths)
        elif len(image_paths) == 1 and image_paths[0] != original_file_path:
            temp_files_created.append(image_paths[0])
        
        all_results = []
        md_contents = []
        
        # 逐页处理
        for page_idx, image_path in enumerate(image_paths):
            logger.info(f"Processing page {page_idx + 1}/{len(image_paths)}")
            
            # 处理当前页面
            recognition_results = process_page(image_path)
            
            # 生成当前页的markdown内容
            page_md_content = generate_markdown(recognition_results)
            
            md_contents.append(page_md_content)
            
            # 保存当前页的处理数据
            page_data = {
                "page": page_idx + 1,
                "elements": recognition_results,
                "total_elements": len(recognition_results)
            }
            all_results.append(page_data)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 合并所有页面的markdown内容
        if len(md_contents) > 1:
            final_md_content = "\n\n---\n\n".join(md_contents)
        else:
            final_md_content = md_contents[0] if md_contents else ""
        
        # 在结果数组最后添加总体信息
        summary_data = {
            "summary": True,
            "total_pages": len(image_paths),
            "total_elements": sum(len(page["elements"]) for page in all_results),
            "processing_time": f"{processing_time:.2f}s",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        all_results.append(summary_data)
        
        logger.info(f"Document processed successfully in {processing_time:.2f}s - {len(image_paths)} page(s)")
        return final_md_content, final_md_content, all_results
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        error_data = [{
            "error": True,
            "message": str(e),
            "original_file": original_file_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }]
        return f"# 处理错误\n\n处理文档时发生错误: {str(e)}", "", error_data
    
    finally:
        # 清理临时文件
        if 'temp_files_created' in locals():
            for temp_file in temp_files_created:
                if temp_file and os.path.exists(temp_file):
                    cleanup_temp_file(temp_file)

def process_page(image_path):
    """处理单页文档"""
    # 阶段1: 页面级布局解析
    pil_image = Image.open(image_path).convert("RGB")
    layout_output = model_chat("Parse the reading order of this document.", pil_image)

    # 阶段2: 元素级内容解析
    padded_image, dims = prepare_image(pil_image)
    recognition_results = process_elements(layout_output, padded_image, dims)

    return recognition_results

def process_elements(layout_results, padded_image, dims, max_batch_size=16):
    """解析所有文档元素"""
    layout_results = parse_layout_string(layout_results)

    # 分别存储不同类型的元素
    text_elements = []  # 文本元素
    table_elements = []  # 表格元素
    figure_results = []  # 图像元素（无需处理）
    previous_box = None
    reading_order = 0

    # 收集要处理的元素并按类型分组
    for bbox, label in layout_results:
        try:
            # 调整坐标
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # 裁剪并解析元素
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and (cropped.shape[0] > 3 and cropped.shape[1] > 3):
                if label == "fig":
                    # 对于图像区域，提取图像的base64编码
                    try:
                        # 将裁剪的图像转换为PIL图像
                        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        
                        # 转换为base64
                        import io
                        import base64
                        buffered = io.BytesIO()
                        pil_crop.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        figure_results.append(
                            {
                                "label": label,
                                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                                "text": img_base64,  # 存储base64编码而不是空字符串
                                "reading_order": reading_order,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error encoding figure to base64: {e}")
                        figure_results.append(
                            {
                                "label": label,
                                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                                "text": "",  # 如果编码失败，使用空字符串
                                "reading_order": reading_order,
                            }
                        )
                else:
                    # 准备元素进行解析
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    
                    # 按类型分组
                    if label == "tab":
                        table_elements.append(element_info)
                    else:  # 文本元素
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            logger.error(f"Error processing bbox with label {label}: {str(e)}")
            continue

    # 初始化结果列表
    recognition_results = figure_results.copy()
    
    # 处理文本元素（批量）
    if text_elements:
        text_results = process_element_batch(text_elements, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_results)
    
    # 处理表格元素（批量）
    if table_elements:
        table_results = process_element_batch(table_elements, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_results)

    # 按阅读顺序排序
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results

def generate_markdown(recognition_results):
    """从识别结果生成Markdown内容"""
    # 使用MarkdownConverter来处理所有类型的内容，包括图片
    converter = MarkdownConverter()
    return converter.convert(recognition_results)


######################################################################################
### START: โค้ดส่วนที่เพิ่มเข้ามาเพื่อรองรับ new_web.py ###
######################################################################################

def generate_enhanced_markdown(recognition_results):
    """
    สร้าง Markdown ที่มีทั้งข้อความ, ตาราง (HTML), และรูปภาพ (Base64)
    เพื่อให้ gr.Markdown สามารถแสดงผลได้ทั้งหมด
    """
    md_parts = []
    for index, result in enumerate(recognition_results):
        label = result.get('label', '')
        text = result.get('text', '').strip()

        if not text and label != 'fig':
            continue

        if label == 'fig':
            # สร้างแท็กรูปภาพใน Markdown โดยใช้ข้อมูล base64 โดยตรง
            figure_name = f"Figure {index + 1}"
            if len(text) > 100:
                md_parts.append(f"**{figure_name}**\n\n![{figure_name}](data:image/png;base64,{text})\n\n")
        
        elif label == 'tab':
            # นำโค้ด HTML ของตารางมาใช้โดยตรง
            md_parts.append(f"{text}\n\n")
            
        else:
            # จัดการกับ LaTeX ที่อาจปนมากับข้อความธรรมดา
            if ("_{" in text or "^{" in text or "\\" in text) and ("$" not in text):
                text = f"${text}$"
            md_parts.append(text + "\n\n")

    return "".join(md_parts)

@spaces.GPU
def extract_coordinates_step1(file_path, force_thai_checkbox, page_num):
    """
    ขั้นตอนที่ 1: สกัดเค้าโครงและพิกัดขององค์ประกอบในหน้าเอกสารที่เลือก
    """
    if not file_path:
        return "กรุณาอัปโหลดไฟล์ก่อน", "en", "Language: N/A"

    page_index = int(page_num) - 1
    if page_index < 0: page_index = 0

    logger.info(f"Step 1: Extracting coordinates for page {page_index + 1}...")
    
    if model is None: initialize_model()

    temp_image_path = None
    try:
        temp_image_path = convert_to_image(file_path, page_num=page_index)
        if not temp_image_path: raise Exception("ไม่สามารถแปลงไฟล์เป็นรูปภาพได้")

        pil_image = Image.open(temp_image_path).convert("RGB")
        layout_output = model_chat("Parse the reading order of this document.", pil_image)

        if force_thai_checkbox:
            lang_state = 'th'
            lang_info = "Language: Thai (Typhoon-OCR for Step 2)"
        else:
            lang_state = 'en'
            lang_info = "Language: English (Dolphin for Step 2)"
            
        logger.info(f"Step 1 successful. Detected language state: {lang_state}")
        return layout_output, lang_state, lang_info

    except Exception as e:
        logger.error(f"Error in Step 1 (extract_coordinates): {e}")
        return f"เกิดข้อผิดพลาดในขั้นตอนที่ 1: {e}", "en", "Language: Error"
    finally:
        if temp_image_path and temp_image_path != file_path:
             cleanup_temp_file(temp_image_path)

@spaces.GPU
def process_data_step2(file_path, coordinates_json, lang_state_val, page_num_val):
    """
    ขั้นตอนที่ 2: ทำ OCR และสร้าง Enhanced Markdown สำหรับแสดงผลใน UI และบันทึกเป็น .txt
    """
    if not file_path or not coordinates_json:
        return "กรุณาทำขั้นตอนที่ 1 ให้เสร็จก่อน", "ข้อมูลไม่ครบถ้วน", {}

    page_index = int(page_num_val) - 1
    if page_index < 0: page_index = 0

    logger.info(f"Step 2 (Dolphin): Processing page {page_index + 1}...")

    if model is None: initialize_model()

    temp_image_path = None
    try:
        temp_image_path = convert_to_image(file_path, page_num=page_index)
        if not temp_image_path: raise Exception("ไม่สามารถแปลงไฟล์เป็นรูปภาพได้")

        pil_image = Image.open(temp_image_path).convert("RGB")
        padded_image, dims = prepare_image(pil_image)
        recognition_results = process_elements(coordinates_json, padded_image, dims)
        
        markdown_for_render = generate_enhanced_markdown(recognition_results)
        
        json_output = {"page": page_index + 1, "processor": "Dolphin", "elements": recognition_results}

        # --- ส่วนสำหรับบันทึกไฟล์ ---
        try:
            save_dir = "/home/sayakaray/ByteDance_Dolphin/Dolphin_copy/ocr_results"
            os.makedirs(save_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            
            output_filename = f"{base_filename}_page_{page_index + 1}.txt"
            
            output_filepath = os.path.join(save_dir, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_for_render)
            
            logger.info(f"Successfully saved content to: {output_filepath}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
        
        logger.info(f"Step 2 (Dolphin) successful for page {page_index + 1}.")
        
        return markdown_for_render, markdown_for_render, json_output

    except Exception as e:
        logger.error(f"Error in Step 2 (Dolphin - process_data): {e}")
        error_md = f"## เกิดข้อผิดพลาดในขั้นตอนที่ 2 (Dolphin)\n\n**รายละเอียด:** {e}"
        return error_md, str(e), {"error": str(e)}
    finally:
        if temp_image_path and temp_image_path != file_path:
             cleanup_temp_file(temp_image_path)

######################################################################################
### END: โค้ดส่วนที่เพิ่มเข้ามา ###
######################################################################################


# ส่วน Gradio UI เดิมของไฟล์นี้จะยังคงอยู่ แต่ new_web.py จะไม่เรียกใช้มัน
if __name__ == "__main__":
    print("Dolphin processor module loaded.")