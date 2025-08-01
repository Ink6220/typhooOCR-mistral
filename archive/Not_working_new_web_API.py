# new_web_API.py
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
    import Not_working_typhoon_tranformers_app_API as typhoon_processor
except ImportError:
    logger.warning("typhoon_tranformers_app_API.py not found. Thai processing will be unavailable.")
    typhoon_processor = None

def load_css(): return ""

# --- ฟังก์ชันสำหรับ UI ---
def get_page_count(file_path):
    if not file_path: return 1
    try:
        with pymupdf.open(stream=file_path.read(), filetype="pdf") as doc: return len(doc)
    except Exception:
        try:
            Image.open(file_path)
            return 1
        except Exception: pass
    return 1

def preview_file(file_obj):
    if not file_obj: return None, gr.update(value=1, maximum=1), "Total Pages: 1"
    count = get_page_count(file_obj)
    md_text = f"Total Pages: {count}"
    num_update = gr.update(value=1, maximum=count if count > 0 else 1)

    try:
        if file_obj.name.lower().endswith(('.png', '.jpeg', '.jpg')):
              with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                  Image.open(file_obj.name).convert("RGB").save(tmp.name, "PDF", resolution=100.0, save_all=True)
                  return tmp.name, num_update, md_text
        return file_obj.name, num_update, md_text
    except Exception as e:
        logger.error(f"Error creating preview: {e}")
        return None, gr.update(value=1, maximum=1), "Total Pages: 1"

def reset_all():
    return None, None, "", "", {}, "en", "Language: N/A", False, 1, "Total Pages: 1"

# --- UI Layout ---
latex_delimiters = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]
with gr.Blocks(css=load_css(), title="Hybrid Document Parser (Local CPU)") as demo:
    gr.HTML("<h1>Hybrid Document Parser (Local CPU Processing)</h1>")
    gr.Markdown("ประมวลผลเอกสารภาษาไทยด้วยโมเดล Typhoon ที่รันบนเครื่องของคุณ (ใช้ CPU)")

    lang_state = gr.State(value="en")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            file = gr.File(label="Choose PDF or image file", file_types=[".pdf", ".png", ".jpeg", ".jpg"])

            with gr.Row():
                page_info_output = gr.Markdown("Total Pages: 1")
                page_num_input = gr.Number(label="Page to Process", value=1, interactive=True, minimum=1, step=1)

            force_thai_checkbox = gr.Checkbox(label="🇹🇭 เอกสารเป็นภาษาไทย (ใช้ Local Typhoon)", value=False, info="ใช้โมเดลที่รันบนเครื่องนี้ในขั้นตอนที่ 2")
            gr.Markdown("### ➡️ ขั้นตอนที่ 1: สกัดพิกัด")
            lang_info_output = gr.Markdown("Language: N/A")
            extract_coords_button = gr.Button("1. Extract Coordinates for Selected Page", variant="secondary")
            gr.Markdown("### ➡️ ขั้นตอนที่ 2: แปลงข้อมูล (OCR)")
            process_final_button = gr.Button("2. Process Selected Page (Local CPU)", variant="primary")
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

    # --- ฟังก์ชัน Wrapper และ Events ---
    def process_by_language(file_obj, coordinates_json, lang_state_val, page_num_val):
        if not file_obj:
            return "# Error\nPlease upload a file first.", "", {}

        file_path_for_processing = file_obj.name

        if lang_state_val == 'th':
            if typhoon_processor:
                logger.info("Redirecting to local Typhoon processor for OCR...")
                return typhoon_processor.process_data_step2(file_path_for_processing, coordinates_json, page_num_val)
            else:
                logger.warning("Typhoon processor not available. Cannot process Thai document.")
                return "# Error\nTyphoon processor not loaded. Cannot process Thai.", "", {}
        else:
            logger.info("Redirecting to Dolphin processor for OCR...")
            # สมมติว่า dolphin_processor มีฟังก์ชันชื่อเดียวกัน
            return dolphin_processor.process_data_step2(file_path_for_processing, coordinates_json, lang_state_val, page_num_val)

    file.upload(fn=preview_file, inputs=file, outputs=[pdf_show, page_num_input, page_info_output], queue=True)

    extract_coords_button.click(
        fn=dolphin_processor.extract_coordinates_step1,
        inputs=[file, force_thai_checkbox, page_num_input],
        outputs=[coordinates_output, lang_state, lang_info_output]
    )

    process_final_button.click(
        fn=process_by_language,
        inputs=[file, coordinates_output, lang_state, page_num_input],
        outputs=[md_render, md_content, json_output]
    )

    clear_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[file, pdf_show, md_render, md_content, json_output, lang_state, lang_info_output, force_thai_checkbox, page_num_input, page_info_output]
    )

if __name__ == "__main__":
    logger.info("Initializing models...")
    # หากมี initialize_model() สำหรับ Dolphin ก็สามารถเปิดใช้งานได้
    # dolphin_processor.initialize_model()

    if typhoon_processor:
        # เรียกฟังก์ชันสำหรับโหลดโมเดล local Typhoon ตอนเริ่มแอป
        typhoon_processor.initialize_local_typhoon_model()

    logger.info("Models initialized. Launching demo...")
    demo.launch()