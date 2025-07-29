---
title: Dolphin
emoji: 🦀
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.32.1
app_file: app.py
pinned: false
license: mit
short_description: Dolphin Demo
---

Hybrid Document Parser (ตัววิเคราะห์เอกสารแบบผสม)

โปรเจกต์นี้คือเว็บแอปพลิเคชันสำหรับการวิเคราะห์และสกัดข้อมูลจากเอกสาร (PDF หรือรูปภาพ) ทีละหน้า โดยใช้สถาปัตยกรรมแบบผสม (Hybrid) ที่รวมความสามารถของโมเดลปัญญาประดิษฐ์สองตัวเข้าด้วยกัน:

1.  ByteDance/Dolphin: ใช้สำหรับวิเคราะห์โครงสร้างและเลย์เอาต์ของเอกสารโดยทั่วไป และมีความสามารถในการอ่านและแปลงข้อมูล (OCR) สำหรับภาษาอังกฤษและตารางที่ไม่ซับซ้อน
2.  Adun/typhoon_ocr-7B-v1.4: เป็นโมเดลที่เชี่ยวชาญและมีความแม่นยำสูงอย่างยิ่งสำหรับการอ่านและแปลงข้อมูล (OCR) ในภาษาไทยโดยเฉพาะ ทั้งข้อความและตาราง

เว็บแอปพลิเคชันนี้สร้างขึ้นโดยใช้ `Gradio` เพื่อให้มี User Interface (UI) ที่ใช้งานง่าย

## คุณสมบัติหลัก (Features)

* UI ที่ใช้งานง่าย: ขับเคลื่อนด้วย Gradio ทำให้ง่ายต่อการอัปโหลดไฟล์และดูผลลัพธ์
* รองรับหลายไฟล์: สามารถประมวลผลไฟล์ .pdf, .png, .jpeg, และ .jpg
* การประมวลผลแบบทีละหน้า: ผู้ใช้สามารถเลือกหน้าที่ต้องการประมวลผลจากเอกสาร PDF ที่มีหลายหน้าได้
* สถาปัตยกรรมแบบ Hybrid:
    * ใช้ Dolphin ในการวิเคราะห์โครงสร้าง (Layout Analysis) ในขั้นตอนแรกเสมอ
    * ผู้ใช้สามารถเลือกใช้ Typhoon-OCR สำหรับการสกัดข้อมูลภาษาไทยที่มีความแม่นยำสูงในขั้นตอนที่สองได้
* กระบวนการ 2 ขั้นตอน:
    1.  สกัดพิกัด (Extract Coordinates): วิเคราะห์โครงสร้างหน้าเอกสารเพื่อหาตำแหน่งขององค์ประกอบต่างๆ (เช่น หัวข้อ, ย่อหน้า, ตาราง, รูปภาพ)
    2.  แปลงข้อมูล (Process Page/OCR): ทำ OCR บนแต่ละองค์ประกอบตามพิกัดที่ได้มา โดยเลือกระหว่างโมเดล Dolphin (สำหรับภาษาอังกฤษ) หรือ Typhoon (สำหรับภาษาไทย)
* แสดงผลหลายรูปแบบ: สามารถดูผลลัพธ์ได้ทั้งในรูปแบบ Markdown ที่แสดงผลสวยงาม, Markdown แบบข้อความดิบ และโครงสร้างข้อมูลแบบ JSON

## สถาปัตยกรรมและการทำงาน (Architecture & How It Works)

โปรเจกต์ทำงานตามลำดับขั้นตอนที่ชัดเจน ซึ่งควบคุมโดย UI:

1.  การอัปโหลดและเตรียมการ:
    * ผู้ใช้อัปโหลดไฟล์เอกสารและเลือกหน้าที่ต้องการ
    * หากเอกสารเป็นภาษาไทย ผู้ใช้ติ๊กช่อง "🇹🇭 เอกสารเป็นภาษาไทย"

2.  ขั้นตอนที่ 1: การวิเคราะห์เลย์เอาต์ (Layout Analysis)
    * เมื่อกดปุ่ม `1. Extract Coordinates...` ไฟล์ `new_web_fine_tune.py` จะเรียกใช้ฟังก์ชัน `extract_coordinates_step1` จาก `ByteDance_Dolphin_app.py`
    * โมเดล Dolphin จะวิเคราะห์รูปภาพของหน้าที่เลือก และคืนค่าเป็นข้อความที่อธิบายเลย์เอาต์และพิกัด (Bounding Box) ของแต่ละองค์ประกอบ
    * UI จะเก็บ "สถานะภาษา" (lang_state) ไว้ว่าเป็น `th` (ไทย) หรือ `en` (อังกฤษ) ตามที่ผู้ใช้เลือก

3.  ขั้นตอนที่ 2: การสกัดเนื้อหา (Content Recognition / OCR)
    * เมื่อกดปุ่ม `2. Process Selected Page` ไฟล์ `new_web_fine_tune.py` จะเรียกใช้ฟังก์ชัน `process_by_language`
    * ฟังก์ชันนี้จะตรวจสอบ `lang_state` เพื่อเลือกว่าจะเรียกใช้โมดูลประมวลผลใด:
        * ถ้าเป็นภาษาไทย (`th`): จะเรียก `process_data_step2` จาก `typhoon_tranformers7B_v1_4_app.py` ซึ่งจะวนลูปแต่ละองค์ประกอบ, ทำการจำแนกประเภท (ข้อความ/ตาราง) เพื่อความแม่นยำสูงสุด, แล้วจึงทำ OCR ด้วยโมเดล Typhoon
        * ถ้าเป็นภาษาอังกฤษ (`en`): จะเรียก `process_data_step2` จาก `ByteDance_Dolphin_app.py` ซึ่งจะทำ OCR ด้วยโมเดล Dolphin
    * ผลลัพธ์ที่ได้ (ข้อความ, ตารางในรูปแบบ HTML, รูปภาพในรูปแบบ Base64) จะถูกรวบรวมและแปลงเป็น Markdown/JSON เพื่อแสดงผล

4.  การแสดงผล:
    * ผลลัพธ์สุดท้ายจะแสดงในแท็บต่างๆ (Markdown Render, Markdown Content, JSON) ใน UI

## โครงสร้างไฟล์ (File Descriptions)

* new_web_fine_tune.py
    * หน้าที่: เป็นไฟล์หลัก (Main Entry Point) ของแอปพลิเคชัน
    * รายละเอียด: สร้างหน้าเว็บ UI ด้วย Gradio, จัดการ Event จากการกดปุ่ม และเป็นตัวกลาง (Controller) ในการเรียกใช้ฟังก์ชันจากโมดูลประมวลผล (`dolphin_processor` หรือ `typhoon_processor`) ตามภาษาที่ผู้ใช้เลือก

* ByteDance_Dolphin_app.py
    * หน้าที่: โมดูลประมวลผลสำหรับโมเดล Dolphin
    * รายละเอียด: มีฟังก์ชันสำหรับโหลดโมเดล, วิเคราะห์เลย์เอาต์ (`extract_coordinates_step1`), และทำ OCR สำหรับภาษาอังกฤษและองค์ประกอบทั่วไป (`process_data_step2`) นอกจากนี้ยังจัดการกับการแปลงไฟล์ PDF/รูปภาพ

* typhoon_tranformers7B_v1_4_app.py
    * หน้าที่: โมดูลประมวลผลสำหรับโมเดล Typhoon-OCR
    * รายละเอียด: มีฟังก์ชันสำหรับโหลดโมเดลภาษาไทย และทำ OCR เฉพาะทางสำหรับภาษาไทย (`process_data_step2`) ที่มีความซับซ้อนและแม่นยำสูง โดยมีการตรวจสอบประเภทของข้อมูลก่อนทำการแปลง

## การติดตั้งและใช้งาน (Setup & Usage)

### ข้อกำหนดเบื้องต้น (Prerequisites)

* Python 3.8+
* Git

### ขั้นตอนการติดตั้ง (Installation)

1.  Clone a repository:
    `git clone <your-repository-url>`
    `cd <your-repository-directory>`

2.  สร้างและเปิดใช้งาน Virtual Environment (แนะนำ):
    `# สำหรับ macOS/Linux`
    `python3 -m venv venv`
    `source venv/bin/activate`

    `# สำหรับ Windows`
    `python -m venv venv`
    `.\venv\Scripts\activate`

3.  ติดตั้ง Dependencies:
    สร้างไฟล์ `requirements.txt` ที่มีเนื้อหาดังนี้:
    `gradio`
    `gradio_pdf`
    `loguru`
    `torch`
    `transformers`
    `Pillow`
    `PyMuPDF`
    `opencv-python`
    `huggingface_hub`
    
    จากนั้นรันคำสั่ง:
    `pip install -r requirements.txt`

### การรันโปรแกรม (Running the Application)

1.  ตรวจสอบให้แน่ใจว่าคุณอยู่ใน Virtual Environment ที่เปิดใช้งานอยู่
2.  รันไฟล์ `new_web_fine_tune.py` ผ่าน command line:
    `python new_web_fine_tune.py`
3.  โปรแกรมจะทำการโหลดโมเดลต่างๆ (อาจใช้เวลาสักครู่ในครั้งแรก) และเมื่อเสร็จสิ้นจะแสดง URL ของ Gradio (เช่น `Running on local URL:  http://127.0.0.1:7860`)
4.  เปิด URL นั้นในเว็บเบราว์เซอร์ของคุณเพื่อเริ่มใช้งาน

### วิธีใช้งานผ่านหน้าเว็บ (UI Guide)

1.  อัปโหลดไฟล์: คลิกที่กล่อง "Choose PDF or image file" เพื่ออัปโหลดเอกสาร
2.  เลือกหน้า: หากเป็นไฟล์ PDF ให้ระบุหมายเลขหน้าที่ต้องการในช่อง "Page to Process"
3.  เลือกภาษา: หากเป็นเอกสารภาษาไทย ให้ติ๊กที่ช่อง "🇹🇭 เอกสารเป็นภาษาไทย"
4.  สกัดพิกัด: กดปุ่ม `1. Extract Coordinates for Selected Page` รอจนกว่าพิกัดจะปรากฏในกล่องข้อความด้านล่าง
5.  แปลงข้อมูล: กดปุ่ม `2. Process Selected Page` เพื่อเริ่มทำ OCR
6.  ดูผลลัพธ์: ผลลัพธ์จะปรากฏในแท็บต่างๆ ทางด้านขวา

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
