-----

# \============================================================ **Readme: Hybrid Document Parser & OCR Evaluation Suite**

### **คำอธิบาย**

-----

โปรเจกต์นี้คือชุดเครื่องมือประมวลผลเอกสารอัจฉริยะแบบครบวงจร ประกอบด้วย 3 ส่วนหลัก:

1.  **Web Application (Web UI)**: ส่วนติดต่อผู้ใช้ผ่านเว็บเบราว์เซอร์ สำหรับการใช้งานทั่วไปที่ง่ายและสะดวก
2.  **Command-Line Scripts (No-UI)**: สคริปต์สำหรับรันผ่าน Terminal เพื่อการประมวลผลเอกสารทีละมากๆ (Batch Processing) หรือการทำงานอัตโนมัติบนเซิร์ฟเวอร์ (โดยเฉพาะบน GPU)
3.  **Evaluation Script (สคริปต์ประเมินผล)**: เครื่องมือสำหรับวัดประสิทธิภาพและความแม่นยำของผลลัพธ์ OCR โดยอัตโนมัติ

หัวใจของระบบคือสถาปัตยกรรมแบบ Hybrid ที่ใช้โมเดลปัญญาประดิษฐ์ 2 ตัวทำงานร่วมกัน:

  * **Dolphin Model**: สำหรับวิเคราะห์โครงสร้างและลำดับการอ่าน (Layout Analysis)
  * **Typhoon-OCR Model**: สำหรับอ่านและแปลงข้อมูลภาษาไทยโดยเฉพาะ

### **คุณสมบัติหลัก**

-----

  - **รองรับหลายอินพุต**: สามารถประมวลผลได้ทั้งไฟล์ PDF และไฟล์รูปภาพ (JPG, PNG)
  - **สถาปัตยกรรม 2 ขั้นตอน**: แยกการวิเคราะห์โครงสร้างออกจากกันอ่านข้อความ ทำให้ผลลัพธ์มีระเบียบและแม่นยำ
  - **ระบบประมวลผลคู่ (Dual Processor)**:
      - **`new_web_*.py`**: Web UI ที่ใช้งานง่ายสำหรับผู้ใช้ทั่วไป
      - **`NO_UI_GPU_*.py`**: สคริปต์สำหรับผู้ใช้ขั้นสูงที่ต้องการประสิทธิภาพสูงสุดบน GPU
  - **การประเมินผลอัตโนมัติ**:
      - สคริปต์ `ocr_evaluation.py` จะคอยตรวจสอบผลลัพธ์ใหม่ๆ และนำมาประเมินเทียบกับข้อมูลต้นฉบับ (Ground Truth)
      - คำนวณค่าชี้วัดมาตรฐาน: Precision, Recall, F1-Score, Word Error Rate (WER), และ Character Error Rate (CER)
  - **แสดงผลเป็นกราฟ**: สร้างและบันทึกกราฟเปรียบเทียบประสิทธิภาพของการ OCR แต่ละไฟล์โดยอัตโนมัติ

### **สิ่งที่ต้องมี (Requirements)**

-----

คุณจำเป็นต้องติดตั้งไลบรารี Python ต่อไปนี้:

  - `torch`
  - `transformers`
  - `gradio`
  - `Pillow` (PIL)
  - `PyMuPDF`
  - `loguru`
  - `opencv-python`
  - `numpy`
  - `jiwer` (สำหรับคำนวณ WER, CER)
  - `matplotlib` (สำหรับสร้างกราฟ)

### **โครงสร้างไฟล์ (File Structure)**

-----

เพื่อให้สคริปต์ทั้งหมดทำงานร่วมกันได้อย่างถูกต้อง กรุณาจัดเรียงไฟล์และโฟลเดอร์ตามโครงสร้างนี้:
```
/your-project-folder/
|
|-- utils/
|   |-- __init__.py
|   |-- utils.py          # (ไฟล์ฟังก์ชันเสริมที่ใช้ร่วมกัน)
|   |-- markdown_utils.py # (ไฟล์สำหรับแปลงผลลัพธ์เป็น Markdown)
|
|-- ocr_results/          # (โฟลเดอร์สำหรับเก็บผลลัพธ์ .txt จะถูกสร้างอัตโนมัติ)
|
|-- ByteDance_Dolphin_app.py
|-- inference_hugg.py
|-- ocr_evaluation.py
|
|-- new_web.py
|-- new_web_V1.py
|-- new_web_fine_tune.py
|-- new_web_fine_tune_V2.py
|
|-- typhoon_tranformers_app.py
|-- typhoon_tranformers_app_V1.py
|-- typhoon_tranformers7B_v1_4_app.py
|-- typhoon_tranformers7B_v1_4_app_V2.py
|
|-- NO_UI_GPU_typhoon_tranformers_app_V1.py
|-- NO_UI_GPU_typhoon_tranformers7B_v1_4_app.py
|-- NO_UI_typhoon_tranformers7B_v1_4_app.py
```
### **ขั้นตอนการติดตั้ง (Installation)**

-----

1.  **สร้างสภาพแวดล้อม (Optional but Recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # บน Linux/macOS
    .\venv\Scripts\activate  # บน Windows
    ```

2.  **ติดตั้งไลบรารีที่จำเป็น**:

    ```bash
    pip install torch transformers gradio Pillow PyMuPDF loguru opencv-python numpy jiwer matplotlib
    ```

3.  **เตรียมไฟล์ `utils` และ `Fonts`**:

      - สร้างโฟลเดอร์ `utils` และ `Fonts`
      - นำไฟล์ `utils.py`, `markdown_utils.py` และไฟล์ฟอนต์ `THSarabunNew.ttf` ใส่ในโฟลเดอร์ที่ถูกต้อง

4.  **เตรียมไฟล์ `ground_truth.json` (สำหรับสคริปต์ประเมินผล)**:

      - สร้างไฟล์ชื่อ `ground_truth.json`
      - เพิ่มข้อมูลข้อความที่ถูกต้องสำหรับแต่ละหน้าของเอกสารที่คุณต้องการประเมินในรูปแบบ JSON
      - **Key**: ต้องอยู่ในรูปแบบ `"ชื่อไฟล์เอกสาร::หมายเลขหน้า"`
      - **Value**: ข้อความที่ถูกต้องทั้งหมดของหน้านั้น
      - **ตัวอย่าง**:

    <!-- end list -->

    ```json
    {
      "report-2024.pdf::1": "นี่คือข้อความที่ถูกต้องของหน้า 1...",
      "datasheet.pdf::5": "ข้อมูลทางเทคนิคของหน้า 5..."
    }
    ```

### **วิธีการใช้งาน (How to Use)**

-----

คุณสามารถรันโปรแกรมได้ 3 รูปแบบหลัก:

**1. 🚀 การใช้งานผ่าน Web UI (แนะนำสำหรับทั่วไป)**

  - **รันสคริปต์**: เปิด Terminal และรันคำสั่ง (แนะนำให้ใช้เวอร์ชันล่าสุด):
    ```bash
    python new_web_fine_tune_V2.py
    ```
  - **ใช้งาน**:
    1.  เปิด Web Browser ไปที่ URL ที่แสดงใน Terminal (เช่น `http://127.0.0.1:7860`)
    2.  อัปโหลดไฟล์ PDF หรือรูปภาพที่ช่อง "Choose PDF or image file"
    3.  เลือกหมายเลขหน้าที่ต้องการประมวลผลที่ "Page to Process"
    4.  หากเอกสารเป็นภาษาไทย ให้ติ๊กที่ช่อง "🇹🇭 เอกสารเป็นภาษาไทย"
    5.  กดปุ่ม "1. Extract Coordinates for Selected Page" เพื่อให้โมเดลวิเคราะห์โครงสร้าง
    6.  กดปุ่ม "2. Process Selected Page" เพื่อทำการ OCR และดูผลลัพธ์ในแท็บต่างๆ

**2. 💻 การใช้งานผ่าน Command-Line (สำหรับ Batch Processing/GPU)**

  - **แก้ไขสคริปต์**:
    1.  เปิดไฟล์ `NO_UI_GPU_typhoon_tranformers7B_v1_4_app.py`
    2.  เลื่อนไปที่ส่วนล่างสุดของไฟล์ (บล็อก `if __name__ == '__main__':`)
    3.  **เอาเครื่องหมาย comment (`#`) ออก** และแก้ไขค่าตัวแปร `TEST_FILE_PATH` และ `TEST_PAGE_NUMBER`
    <!-- end list -->
      - **ตัวอย่าง**:
    <!-- end list -->
    ```python
    # --- ตัวอย่างการเรียกใช้งาน ---
    # 1. กำหนดค่าพาธของไฟล์และหน้า
    TEST_FILE_PATH = "path/to/your/document.pdf"  # <-- แก้ไขตรงนี้
    TEST_PAGE_NUMBER = 1                         # <-- แก้ไขตรงนี้

    # ... (เอา comment โค้ดส่วนที่เหลือออก)
    ```
  - **รันสคริปต์**: เปิด Terminal และรันคำสั่ง:
    ```bash
    python NO_UI_GPU_typhoon_tranformers7B_v1_4_app.py
    ```

**3. 📊 การรันสคริปต์ประเมินผล (สำหรับวัดประสิทธิภาพ)**

  - **เงื่อนไข**: ต้องมีไฟล์ `ground_truth.json` ที่มีข้อมูลตรงกับเอกสารที่จะประเมิน
  - **ขั้นตอน**:
    1.  **(Terminal ที่ 1)** รัน OCR ผ่าน Web UI หรือ Command-Line ก่อน เพื่อให้มีไฟล์ผลลัพธ์ `.txt` ถูกสร้างขึ้นในโฟลเดอร์ `ocr_results`
    2.  **(Terminal ที่ 2)** เปิด Terminal อีกหน้าต่างหนึ่ง
    3.  รันสคริปต์ประเมินผล:
    <!-- end list -->
    ```bash
    python ocr_evaluation.py
    ```
    4.  สคริปต์จะทำงานวนลูปไปเรื่อยๆ เพื่อคอยตรวจจับและประเมินผลไฟล์ใหม่ที่เข้ามาในโฟลเดอร์ `ocr_results` โดยอัตโนมัติ

### **ผลลัพธ์ที่ได้ (Output)**

-----

1.  **จาก Web UI / Command-Line**:
      - **ไฟล์ `.txt`**: ผลลัพธ์การ OCR ของแต่ละหน้าจะถูกบันทึกไว้ในโฟลเดอร์ `ocr_results` รอการประเมินผล
2.  **จากสคริปต์ประเมินผล**:
      - **บนหน้าจอ (Console)**: แสดงค่า Precision, Recall, F1, WER, CER ของไฟล์ที่กำลังประเมิน
      - **ไฟล์กราฟ `.png`**: กราฟสรุปผลการประเมินจะถูกบันทึกไว้ในโฟลเดอร์ `result_evaluation`
      - **ไฟล์ที่ประมวลผลแล้ว**: ไฟล์ `.txt` ที่ถูกประเมินแล้วจะถูกย้ายจาก `ocr_results` ไปยังโฟลเดอร์ `processed` เพื่อป้องกันการทำงานซ้ำ