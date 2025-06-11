import os
import tempfile
import time
import uuid

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

def to_pdf(file_path):
    """将输入文件转换为PDF格式"""
    if file_path is None:
        return None
        
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            # 使用临时文件而不是保存到磁盘
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(pdf_bytes)
                return tmp_file.name

@spaces.GPU(duration=120)
def process_document(file_path):
    """处理文档的主要函数 - 集成完整的推理逻辑"""
    if file_path is None:
        return "", "", {}, {}
    
    start_time = time.time()
    original_file_path = file_path
    
    # 确保模型已初始化
    if model is None:
        initialize_model()
    
    # 转换为PDF（如果需要）
    converted_file_path = to_pdf(file_path)
    temp_file_created = converted_file_path != original_file_path
    
    try:
        logger.info(f"Processing document: {file_path}")
        
        # 处理页面
        recognition_results = process_page(converted_file_path)
        
        # 生成Markdown内容
        md_content = generate_markdown(recognition_results)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        debug_info = {
            "original_file": original_file_path,
            "converted_file": converted_file_path,
            "temp_file_created": temp_file_created,
            "status": "success",
            "processing_time": f"{processing_time:.2f}s",
            "total_elements": len(recognition_results)
        }
        
        processing_data = {
            "pages": [{"elements": recognition_results}],
            "total_elements": len(recognition_results),
            "processing_time": f"{processing_time:.2f}s"
        }
        
        logger.info(f"Document processed successfully in {processing_time:.2f}s")
        return md_content, md_content, processing_data, debug_info
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        error_info = {
            "original_file": original_file_path,
            "converted_file": converted_file_path,
            "temp_file_created": temp_file_created,
            "status": "error",
            "error": str(e)
        }
        return f"# 处理错误\n\n处理文档时发生错误: {str(e)}", "", {}, error_info
    
    finally:
        # 清理临时文件
        if temp_file_created:
            cleanup_temp_file(converted_file_path)

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
            if cropped.size > 0:
                if label == "fig":
                    # 对于图像区域，直接添加空文本结果
                    figure_results.append(
                        {
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "text": "",
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
    markdown_parts = []
    
    for result in recognition_results:
        text = result.get("text", "").strip()
        label = result.get("label", "")
        
        if text:
            if label == "tab":
                # 表格内容
                markdown_parts.append(f"\n{text}\n")
            else:
                # 普通文本内容
                markdown_parts.append(text)
    
    return "\n\n".join(markdown_parts)

# LaTeX 渲染配置
latex_delimiters = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# 加载自定义CSS
custom_css = load_css()

# 读取页面头部
with open("header.html", "r", encoding="utf-8") as file:
    header = file.read()

# 创建 Gradio 界面
with gr.Blocks(css=custom_css, title="Dolphin Document Parser") as demo:
    gr.HTML(header)

    with gr.Row():
        # 侧边栏 - 文件上传和控制
        with gr.Column(scale=1, elem_classes="sidebar"):
            # 文件上传组件
            file = gr.File(
                label="Choose PDF or image file", 
                file_types=[".pdf", ".png", ".jpeg", ".jpg"], 
                elem_id="file-upload"
            )

            gr.HTML("选择文件后，点击处理按钮开始解析<br>After selecting the file, click the Process button to start parsing")

            with gr.Row(elem_classes="action-buttons"):
                submit_btn = gr.Button("处理文档/Process Document", variant="primary")
                clear_btn = gr.ClearButton(value="清空/Clear")
            
            # 处理状态显示
            status_display = gr.Textbox(
                label="Processing Status", 
                value="Ready to process documents", 
                interactive=False,
                max_lines=2
            )

            # 示例文件
            example_root = os.path.join(os.path.dirname(__file__), "examples")
            if os.path.exists(example_root):
                gr.HTML("示例文件/Example Files")
                example_files = [
                    os.path.join(example_root, f) 
                    for f in os.listdir(example_root) 
                    if not f.endswith(".py")
                ]

                examples = gr.Examples(
                    examples=example_files, 
                    inputs=file, 
                    examples_per_page=10, 
                    elem_id="example-files"
                )

        # 主体内容区域
        with gr.Column(scale=7):
            with gr.Row(elem_classes="main-content"):
                # 预览面板
                with gr.Column(scale=1, elem_classes="preview-panel"):
                    gr.HTML("文件预览/Preview")
                    pdf_show = PDF(label="", interactive=False, visible=True, height=600)
                    debug_output = gr.JSON(label="Debug Info", height=100)

                # 输出面板
                with gr.Column(scale=1, elem_classes="output-panel"):
                    with gr.Tabs():
                        with gr.Tab("Markdown [Render]"):
                            md_render = gr.Markdown(
                                label="",
                                height=700,
                                show_copy_button=True,
                                latex_delimiters=latex_delimiters,
                                line_breaks=True,
                            )
                        with gr.Tab("Markdown [Content]"):
                            md_content = gr.TextArea(lines=30, show_copy_button=True)
                        with gr.Tab("Processing Data"):
                            json_output = gr.JSON(label="", height=700)

    # 事件处理
    file.change(fn=to_pdf, inputs=file, outputs=pdf_show)
    
    # 文档处理
    def process_with_status(file_path):
        """处理文档并更新状态"""
        if file_path is None:
            return "", "", {}, {}, "Please select a file first"
        
        # 更新状态为处理中
        status = "Processing document..."
        
        # 执行文档处理
        md_render_result, md_content_result, json_result, debug_result = process_document(file_path)
        
        # 更新完成状态
        if "错误" in md_render_result:
            status = "Processing failed - see debug info"
        else:
            status = "Processing completed successfully"
        
        return md_render_result, md_content_result, json_result, debug_result, status
    
    submit_btn.click(
        fn=process_with_status,
        inputs=[file],
        outputs=[md_render, md_content, json_output, debug_output, status_display],
    )
    
    # 清空所有内容
    def reset_all():
        return None, None, "", "", {}, {}, "Ready to process documents"
    
    clear_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[file, pdf_show, md_render, md_content, json_output, debug_output, status_display]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch() 