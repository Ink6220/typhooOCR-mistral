# ocr_evaluation.py
import re
import os
import time
import json
from jiwer import wer, cer
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.font_manager import FontProperties

# --- กำหนดโฟลเดอร์ต่างๆ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_RESULTS_DIR = os.path.join(BASE_DIR, "ocr_results")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
EVAL_GRAPH_DIR = os.path.join(BASE_DIR, "result_evaluation")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth.json")


def load_ground_truth(filepath: str) -> dict:
    """
    โหลดข้อมูล Ground Truth จากไฟล์ JSON
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            logger.info(f"กำลังโหลดข้อมูล Ground Truth จาก {filepath}")
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"ไม่พบไฟล์ Ground Truth ที่ {filepath} จะไม่มีการประเมินผล")
        return {}
    except json.JSONDecodeError:
        logger.error(f"เกิดข้อผิดพลาดในการถอดรหัส JSON จาก {filepath} กรุณาตรวจสอบรูปแบบไฟล์")
        return {}

def preprocess_text(text: str) -> str:
    """
    เตรียมข้อความสำหรับการประเมินผล
    """
    # [ใหม่] ลบบรรทัดที่มีแพทเทิร์น Figure ออกไปทั้งหมด
    # re.MULTILINE ช่วยให้ ^ ตรงกับจุดเริ่มต้นของแต่ละบรรทัด
    text = re.sub(r'^.*!\[Figure\s+\d+\].*$', '', text, flags=re.MULTILINE)
    
    # การล้างข้อมูลอื่นๆ ยังคงเดิม
    text = re.sub(r'!\[.*?\]\(data:image.*?\)', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_precision_recall_f1(ground_truth: str, ocr_output: str):
    gt_words = set(preprocess_text(ground_truth).split())
    ocr_words = set(preprocess_text(ocr_output).split())
    
    tp = len(gt_words.intersection(ocr_words))
    fp = len(ocr_words.difference(gt_words))
    fn = len(gt_words.difference(ocr_words))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1_score": f1}

def calculate_cer_wer(ground_truth: str, ocr_output: str):
    gt_processed = preprocess_text(ground_truth)
    ocr_processed = preprocess_text(ocr_output)

    wer_value = wer(gt_processed, ocr_processed)
    cer_value = cer(gt_processed, ocr_processed)

    return {"wer": wer_value, "cer": cer_value}

font_path = '/home/sayakaray/ByteDance_Dolphin/Dolphin_copy/Fonts/THSarabunNew.ttf'
font_prop = FontProperties(fname=font_path)

def save_results_as_graph(metrics: dict, save_path: str, filename: str):
    # กำหนด path ของฟอนต์และสร้าง object FontProperties
    font_path = '/home/sayakaray/ByteDance_Dolphin/Dolphin_copy/Fonts/THSarabunNew.ttf'
    font_prop = FontProperties(fname=font_path, size=12)
    title_font_prop = FontProperties(fname=font_path, size=16)

    os.makedirs(save_path, exist_ok=True)
    performance_metrics = {'Precision': metrics['precision'], 'Recall': metrics['recall'], 'F1-Score': metrics['f1_score']}
    error_metrics = {'WER': metrics['wer'], 'CER': metrics['cer']}

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # ตั้งค่าฟอนต์ให้ Title หลัก
    fig.suptitle(f'OCR Evaluation Metrics for\n{filename}', fontproperties=title_font_prop)

    # กราฟแท่งที่ 1: Performance Metrics
    bars1 = axs[0].bar(performance_metrics.keys(), performance_metrics.values(), color=['#4CAF50', '#2196F3', '#FFC107'])
    axs[0].set_title('Performance Metrics (Higher is Better)', fontproperties=font_prop)
    axs[0].set_ylabel('Score', fontproperties=font_prop)
    axs[0].set_ylim(0, 1)

    # ตั้งค่าฟอนต์ให้ตัวเลขบนแท่งกราฟ และแกน X
    for bar in bars1:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2%}', va='bottom', ha='center', fontproperties=font_prop)
    for label in axs[0].get_xticklabels():
        label.set_fontproperties(font_prop)

    # กราฟแท่งที่ 2: Error Rates
    bars2 = axs[1].bar(error_metrics.keys(), error_metrics.values(), color=['#F44336', '#9C27B0'])
    axs[1].set_title('Error Rates (Lower is Better)', fontproperties=font_prop)
    axs[1].set_ylabel('Error Rate', fontproperties=font_prop)
    # [ใหม่] ตั้งค่าสเกลสูงสุดของแกน Y เป็น 1.0
    axs[1].set_ylim(0, 1)

    # ตั้งค่าฟอนต์ให้ตัวเลขบนแท่งกราฟ และแกน X
    for bar in bars2:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2%}', va='bottom', ha='center', fontproperties=font_prop)
    for label in axs[1].get_xticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join(save_path, f"eval_{os.path.splitext(filename)[0]}.png")
    plt.savefig(output_filename)
    plt.close()
    logger.info(f"บันทึกกราฟไปที่: {output_filename}")

def find_ground_truth_key(filename_base: str, page_num_str: str, ground_truth_keys: list) -> str | None:
    for key in ground_truth_keys:
        try:
            gt_full_filename, gt_page = key.split('::')
            gt_filename_base = os.path.splitext(gt_full_filename)[0]
            if filename_base == gt_filename_base and page_num_str == gt_page:
                return key
        except ValueError:
            continue
    return None

def contains_figure_pattern(text: str) -> bool:
    """
    ตรวจสอบว่าข้อความมีแพทเทิร์นของ Figure หรือไม่
    เช่น "![Figure 1](...)"
    """
    # ใช้ regular expression เพื่อค้นหาแพทเทิร์น: ![Figure ตามด้วยเว้นวรรค และตัวเลข 1 ตัวขึ้นไป]
    pattern = re.compile(r'!\[Figure\s+\d+\]')
    if pattern.search(text):
        return True
    return False

def main_loop():
    logger.info("กำลังเริ่มโปรแกรมตรวจสอบผล OCR...")
    logger.info(f"กำลังตรวจสอบไฟล์ใหม่ใน: {OCR_RESULTS_DIR}")

    os.makedirs(OCR_RESULTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ground_truth_data = load_ground_truth(GROUND_TRUTH_FILE)
    if not ground_truth_data:
        logger.error("ไม่สามารถโหลดข้อมูล Ground Truth ได้, โปรแกรมจะหยุดทำงาน")
        return

    ground_truth_keys = list(ground_truth_data.keys())

    while True:
        try:
            ocr_files = [f for f in os.listdir(OCR_RESULTS_DIR) if f.endswith('.txt')]

            if not ocr_files:
                time.sleep(5)
                continue

            for filename in ocr_files:
                try:
                    logger.info(f"พบไฟล์ใหม่: {filename}")
                    filepath = os.path.join(OCR_RESULTS_DIR, filename)

                    page_key = None
                    try:
                        temp_name = os.path.splitext(filename)[0]
                        if '_page_' in temp_name:
                            parts = temp_name.rsplit('_page_', 1)
                            base_filename = parts[0]
                            page_num_full = parts[1]
                            page_num = page_num_full.split('_')[0]
                            page_key = find_ground_truth_key(base_filename, page_num, ground_truth_keys)
                            
                    except Exception as e:
                        logger.warning(f"ไม่สามารถแยกวิเคราะห์ชื่อไฟล์ {filename} ได้: {e}")

                    if page_key and page_key in ground_truth_data:
                        logger.info(f"พบข้อมูล Ground Truth สำหรับ {page_key}. กำลังประมวลผล...")
                        
                        # อ่านเนื้อหาไฟล์ในขั้นตอนนี้
                        with open(filepath, 'r', encoding='utf-8') as f:
                            ocr_text = f.read()
                        
                        ground_truth_text = ground_truth_data[page_key]
                        
                        word_metrics = calculate_precision_recall_f1(ground_truth_text, ocr_text)
                        error_rates = calculate_cer_wer(ground_truth_text, ocr_text)
                        all_metrics = {**word_metrics, **error_rates}
                        
                        logger.info(f"--- ผลการประเมินสำหรับ {filename} ---")
                        logger.info(f"Precision: {all_metrics['precision']:.2%}, Recall: {all_metrics['recall']:.2%}, F1: {all_metrics['f1_score']:.2%}")
                        logger.info(f"WER: {all_metrics['wer']:.2%}, CER: {all_metrics['cer']:.2%}")
                        
                        save_results_as_graph(all_metrics, EVAL_GRAPH_DIR, filename)
                        
                        os.rename(filepath, os.path.join(PROCESSED_DIR, filename))
                        logger.info(f"ย้ายไฟล์ที่ประมวลผลแล้วไปที่: {PROCESSED_DIR}")
                        
                    else:
                        logger.warning(f"ไม่พบข้อมูล Ground Truth สำหรับไฟล์ {filename}. ย้ายไฟล์ไปที่ processed.")
                        os.rename(filepath, os.path.join(PROCESSED_DIR, filename))

                except FileNotFoundError:
                    logger.warning(f"ไม่พบไฟล์ '{filename}' (อาจถูกลบไปแล้ว) กำลังข้ามไปไฟล์ถัดไป...")
                    continue

        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในลูปหลัก: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()