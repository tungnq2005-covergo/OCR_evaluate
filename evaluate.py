import json
import os
from rapidfuzz import fuzz
from tabulate import tabulate

# 1. Hàm tính điểm tương đồng (0 - 100)
def calculate_similarity(truth_val, pred_val):
    if not truth_val and not pred_val: return 100.0 # Cả 2 đều rỗng -> Đúng
    if not truth_val or not pred_val: return 0.0    # 1 cái rỗng -> Sai
    
    # Chuyển về string và lowercase để so sánh công bằng
    str_truth = str(truth_val).lower().strip()
    str_pred = str(pred_val).lower().strip()
    
    # Dùng Token Sort Ratio để không quan tâm thứ tự từ (Vd: "ABC Co." vs "Co. ABC")
    return fuzz.token_sort_ratio(str_truth, str_pred)

# 2. Hàm đệ quy để so sánh JSON (Nested Object)
def compare_objects(truth_obj, pred_obj, prefix=""):
    report = []
    
    for key, val_truth in truth_obj.items():
        current_key = f"{prefix}.{key}" if prefix else key
        val_pred = pred_obj.get(key, "")
        
        # Nếu là object con (Vd: seller, buyer) -> Đệ quy
        if isinstance(val_truth, dict):
            # Nếu pred không có dict tương ứng thì tạo dict rỗng để so sánh tiếp (sẽ ra 0 điểm)
            pred_sub = val_pred if isinstance(val_pred, dict) else {}
            report.extend(compare_objects(val_truth, pred_sub, current_key))
            continue
            
        # Nếu là Array (Vd: items) -> Xử lý riêng (Tạm thời so sánh tổng số item hoặc skip)
        if isinstance(val_truth, list):
            # TODO: Xử lý so sánh từng dòng item (Phức tạp hơn)
            # Tạm thời so sánh số lượng item lấy được
            count_truth = len(val_truth)
            count_pred = len(val_pred) if isinstance(val_pred, list) else 0
            score = 100.0 if count_truth == count_pred else 0.0
            report.append([current_key + "(count)", count_truth, count_pred, score])
            continue

        # So sánh giá trị đơn (String/Number)
        score = calculate_similarity(val_truth, val_pred)
        report.append([current_key, val_truth, val_pred, score])
        
    return report

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # GIẢ LẬP: Load file Ground Truth và File Kết quả từ Engine (sau khi đã normalize)
    # Thực tế bạn sẽ dùng: json.load(open('path/to/file.json'))
    
    # 1. Load Ground Truth (Chuẩn)
    try:
        with open('data/ground_truth/ground_truth_page_1.json', 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("⚠️ Chưa có file Ground Truth mẫu. Hãy tạo file json trong data/ground_truth/")
        exit()

    # 2. Load Prediction (Giả sử đây là kết quả từ LandingAI sau khi bạn đã parse)
    # Bạn hãy thay file này bằng file kết quả thực tế của bạn
    prediction_sample = {
        "seller": {
            "name": "CONG TY TNHH NHUA APCO", # Sai chính tả nhẹ
            "taxCode": "0107453450",
            "address": "BT65 Lam Vien..."
        },
        "invoice": {
            "number": "50",
            "totalAmountAfterTax": 839808000
        }
    }

    print(f"--- ĐANG SO SÁNH: Ground Truth vs Prediction ---")
    
    # 3. Chạy so sánh
    results = compare_objects(ground_truth, prediction_sample)
    
    # 4. Tính điểm trung bình
    total_score = sum([row[3] for row in results])
    avg_score = total_score / len(results) if results else 0
    
    # 5. Xuất báo cáo đẹp
    headers = ["Field", "Ground Truth", "Extracted Value", "Score (%)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print(f"\n🚀 ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY): {avg_score:.2f}%")