# new_web.py
import os
import tempfile
import gradio as gr
from gradio_pdf import PDF
from loguru import logger
from PIL import Image
import pymupdf

# Import โมดูลประมวลผลของแต่ละภาษา
import ByteDance_Dolphin_app as dolphin_processor
try:
    import typhoon_tranformers7B_v1_4_app_V2 as typhoon_processor
except ImportError:
    logger.warning("typhoon_tranformers7B_v1_4_app_V2.py not found.")
    typhoon_processor = None

def load_css(): return ""

# --- ฟังก์ชันสำหรับ UI (ไม่มีการเปลี่ยนแปลง) ---
def get_page_count(file_path):
    if not file_path: return 1
    try:
        if file_path.lower().endswith('.pdf'):
            with pymupdf.open(file_path) as doc: return len(doc)
    except Exception: pass
    return 1

def preview_file(file_path):
    count = get_page_count(file_path)
    md_text = f"Total Pages: {count}"
    num_update = gr.update(value=1, maximum=count if count > 0 else 1)
    if not file_path: return None, num_update, md_text
    try:
        if file_path.lower().endswith(('.png', '.jpeg', '.jpg')):
             with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                Image.open(file_path).convert("RGB").save(tmp.name, "PDF", resolution=100.0)
                return tmp.name, num_update, md_text
        return file_path, num_update, md_text
    except Exception as e:
        logger.error(f"Error creating preview: {e}")
        return None, gr.update(value=1, maximum=1), "Total Pages: 1"

def reset_all():
    return None, None, "", "", {}, "en", "Language: N/A", False, 1, "Total Pages: 1"

# --- UI Layout (ปรับปรุง) ---
latex_delimiters = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]
with gr.Blocks(css=load_css(), title="Hybrid Document Parser") as demo:
    gr.HTML("<h1>Hybrid Document Parser (Page-by-Page)</h1>")
    
    lang_state = gr.State(value="en")
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            file = gr.File(label="Choose PDF or image file", file_types=[".pdf", ".png", ".jpeg", ".jpg"])
            with gr.Row():
                page_info_output = gr.Markdown("Total Pages: 1")
                page_num_input = gr.Number(label="Page to Process", value=1, interactive=True, minimum=1, step=1)
            force_thai_checkbox = gr.Checkbox(label="🇹🇭 เอกสารเป็นภาษาไทย", value=False, info="ใช้ Typhoon-OCR สำหรับการประมวลผลในขั้นตอนที่ 2")
            gr.Markdown("### ➡️ ขั้นตอนที่ 1: สกัดพิกัด")
            lang_info_output = gr.Markdown("Language: N/A")
            extract_coords_button = gr.Button("1. Extract Coordinates for Selected Page", variant="secondary")
            gr.Markdown("### ➡️ ขั้นตอนที่ 2: แปลงข้อมูล (OCR)")
            process_final_button = gr.Button("2. Process Selected Page", variant="primary")
            clear_btn = gr.ClearButton(value="Clear")
            coordinates_output = gr.Textbox(label="Coordinates / Layout Text (Single Page)", lines=10, interactive=True)
            
        with gr.Column(scale=2):
            with gr.Row(elem_classes="main-content"):
                with gr.Column(scale=1, elem_classes="preview-panel"):
                    pdf_show = PDF(label="Preview", interactive=False, height=600)
                with gr.Column(scale=1, elem_classes="output-panel"):
                    with gr.Tabs():
                        with gr.Tab("Markdown [Render]"):
                            md_render = gr.Markdown(show_copy_button=True, line_breaks=True, latex_delimiters=latex_delimiters)
                        with gr.Tab("Markdown [Content]"):
                            md_content = gr.TextArea(lines=20, show_copy_button=True)
                        with gr.Tab("Json [Content]"):
                            json_output = gr.JSON(label="")
                        # --- เอา Tab แสดงรูปภาพออกไป ---

    # --- ฟังก์ชัน Wrapper และ Events (ปรับปรุง) ---
    def process_by_language(file_path, coordinates_json, lang_state_val, page_num_val):
        if lang_state_val == 'th' and typhoon_processor:
            logger.info("Redirecting to Typhoon processor for OCR...")
            # สมมติว่า typhoon_processor คืนค่า 3 ตัวแปรเหมือนกัน
            return typhoon_processor.process_data_step2(file_path, coordinates_json, lang_state_val, page_num_val)
        else:
            logger.info("Redirecting to Dolphin processor for OCR...")
            return dolphin_processor.process_data_step2(file_path, coordinates_json, lang_state_val, page_num_val)

    file.upload(fn=preview_file, inputs=file, outputs=[pdf_show, page_num_input, page_info_output], queue=True)
    
    extract_coords_button.click(
        fn=dolphin_processor.extract_coordinates_step1,
        inputs=[file, force_thai_checkbox, page_num_input],
        outputs=[coordinates_output, lang_state, lang_info_output]
    )

    process_final_button.click(
        fn=process_by_language,
        inputs=[file, coordinates_output, lang_state, page_num_input],
        outputs=[md_render, md_content, json_output] # Outputs เหลือ 3 ตัว
    )
    
    clear_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[file, pdf_show, md_render, md_content, json_output, lang_state, lang_info_output, force_thai_checkbox, page_num_input, page_info_output]
    )

if __name__ == "__main__":
    logger.info("Initializing models...")
    dolphin_processor.initialize_model()
    if typhoon_processor:
        typhoon_processor.initialize_thai_model()
    logger.info("Models initialized. Launching demo...")
    demo.launch()