#!/usr/bin/env python3
"""
Full Benchmark Tool (OCR + LLM Mapping)
Đo tổng thời gian End-to-End từ File PDF -> JSON chuẩn Schema.
Hỗ trợ: Azure, LlamaParse + Groq (Llama 3).
"""

import os
import json
import time
import argparse
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA SCHEMA (JSON FORMAT)
# ==============================================================================
# Để code chạy nhanh và gọn, ta dùng dict để mô tả schema cho LLM hiểu
# thay vì khai báo class Pydantic rườm rà.

INVOICE_SCHEMA = {
    "seller": {
        "name": "string", "taxCode": "string", "address": "string",
        "phone": "string", "fax": "string", "bankAccount": "string", "bankName": "string"
    },
    "buyer": {
        "name": "string", "taxCode": "string", "address": "string",
        "customerName": "string", "bankAccount": "string", "bankName": "string"
    },
    "invoice": {
        "serial": "string", "number": "string", "date": "YYYY-MM-DD",
        "paymentMethod": "string", "note": "string",
        "totalAmountBeforeTax": "number", "vatRate": "string",
        "vatAmount": "number", "totalAmountAfterTax": "number", "amountInWords": "string",
        "items": [
            {
                "description": "string", "unit": "string", "quantity": "number",
                "unitPrice": "number", "amount": "number"
            }
        ]
    }
}

CLAIM_SCHEMA = {
    "isMedicalDocument": "boolean",
    "claimType": "enum[OP, DEN, HOSP, MAT]",
    "treatmentDate": "DDMMYYYY",
    "clinicName": "string",
    "clinicSpeciality": "string",
    "doctorDetails": {
        "doctorName": "string", "doctorSpeciality": "string", "doctorCode": "string"
    },
    "diagnosis": {
        "description": "string", "code": "string (ICD-10)"
    },
    "currency": "string",
    "totalAmount": "number"
}

def get_schema_prompt(filename):
    """Chọn schema dựa trên tên file"""
    if "claim" in filename.lower():
        return "Insurance Claim", json.dumps(CLAIM_SCHEMA, indent=2)
    return "VAT Invoice", json.dumps(INVOICE_SCHEMA, indent=2)

# ==============================================================================
# PHẦN 2: LLM MAPPING (GROQ)
# ==============================================================================

def map_with_groq(raw_text, filename):
    """Gọi Groq để chuyển Text thô -> JSON chuẩn"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, 0 # Không có key thì bỏ qua bước này

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        doc_type, schema_str = get_schema_prompt(filename)
        
        start_llm = time.time()
        
        prompt = f"""
        You are a data extraction AI. Extract data from the OCR text below into JSON.
        Document Type: {doc_type}
        Target Schema:
        {schema_str}

        Rules:
        - Return ONLY valid JSON.
        - Use null for missing fields.
        - No markdown formatting (```json).
        
        --- OCR TEXT ---
        {raw_text[:20000]}
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Model tốt nhất cho task này trên Groq
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        duration = time.time() - start_llm
        return json.loads(completion.choices[0].message.content), duration

    except Exception as e:
        print(f"  ❌ LLM Error ({filename}): {e}")
        return None, 0

# ==============================================================================
# PHẦN 3: OCR PROCESSORS (AZURE & LLAMA)
# ==============================================================================

def save_result(data, original_filename, output_dir, stats):
    """Lưu kết quả cuối cùng + file thống kê thời gian"""
    # 1. Lưu JSON data
    out_name = f"{Path(original_filename).stem}_page_{stats['page']}.json"
    with open(output_dir / out_name, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 2. Lưu thống kê thời gian (để bạn vẽ biểu đồ nếu cần)
    stats_name = f"{Path(original_filename).stem}_page_{stats['page']}_stats.json"
    with open(output_dir / stats_name, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

def process_file_pipeline(pdf_path, output_dir, engine="azure"):
    """
    Pipeline xử lý đầy đủ:
    Start -> OCR -> (Có Text) -> LLM Mapping -> (Có JSON) -> End
    """
    total_start = time.time()
    file_name = pdf_path.name
    results_log = []

    # --- BƯỚC 1: OCR ---
    ocr_start = time.time()
    raw_pages = []
    
    try:
        if engine == "azure":
            # Azure Code
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
            client = DocumentIntelligenceClient(
                endpoint=os.getenv('AZURE_ENDPOINT'), 
                credential=AzureKeyCredential(os.getenv('AZURE_KEY'))
            )
            with open(pdf_path, 'rb') as f:
                poller = client.begin_analyze_document("prebuilt-layout", body=f, content_type="application/pdf")
            result = poller.result()
            
            # Cắt text theo trang
            full_content = result.content
            for page in result.pages:
                parts = [full_content[span.offset : span.offset + span.length] for span in page.spans] if page.spans else []
                raw_pages.append("".join(parts))
                
        elif engine == "llama":
            # Llama Code
            from llama_parse import LlamaParse
            parser = LlamaParse(api_key=os.getenv('LLAMA_CLOUD_API_KEY'), result_type="markdown")
            docs = parser.load_data(str(pdf_path))
            raw_pages = [doc.text for doc in docs]
            
    except Exception as e:
        return f"❌ {engine.upper()} Error: {e}"

    ocr_duration = time.time() - ocr_start

    # --- BƯỚC 2: LLM MAPPING (Chạy song song từng trang) ---
    # Vì OCR trả về cả file, giờ ta map từng trang một
    
    for i, raw_text in enumerate(raw_pages):
        page_num = i + 1
        
        # Gọi LLM
        structured_data, llm_duration = map_with_groq(raw_text, file_name)
        
        if structured_data is None:
            # Fallback nếu không có key LLM: Lưu raw text
            structured_data = {"raw_content": raw_text, "note": "LLM mapping skipped or failed"}
        
        # Tính tổng thời gian cho trang này
        # (Lưu ý: Thời gian OCR là chung cho cả file, nên ta có thể chia trung bình hoặc giữ nguyên tùy cách tính)
        # Ở đây mình giữ nguyên OCR time của cả file để thấy độ trễ ban đầu.
        
        stats = {
            "file": file_name,
            "page": page_num,
            "engine": engine,
            "time_ocr": round(ocr_duration, 2),
            "time_llm": round(llm_duration, 2),
            "time_total": round(ocr_duration + llm_duration, 2) # Đây là con số bạn cần
        }
        
        save_result(structured_data, file_name, output_dir, stats)
        results_log.append(f"P{page_num}: {stats['time_total']}s (OCR: {stats['time_ocr']}s + LLM: {stats['time_llm']}s)")

    return f"✅ {engine.upper()}: Processed {len(raw_pages)} pages. Times: {', '.join(results_log)}"

# ==============================================================================
# PHẦN 4: MAIN EXECUTOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark Full Process')
    parser.add_argument('--engine', choices=['all', 'azure', 'llama'], default='all')
    args = parser.parse_args()

    input_dir = Path("data/inputs")
    output_base = Path("data/outputs_full")
    
    pdf_files = sorted(list(input_dir.glob("*.pdf")))
    print(f"🚀 Benchmarking {len(pdf_files)} files (End-to-End Latency)...")
    
    engines = ["azure", "llama"] if args.engine == 'all' else [args.engine]
    
    # Tạo folder output
    for eng in engines: (output_base / eng).mkdir(parents=True, exist_ok=True)

    # Chạy Parallel (Cấp File)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for pdf in pdf_files:
            for eng in engines:
                futures.append(executor.submit(process_file_pipeline, pdf, output_base / eng, eng))
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print(f"\n✅ DONE. Results saved in '{output_base}'")

if __name__ == "__main__":
    main()