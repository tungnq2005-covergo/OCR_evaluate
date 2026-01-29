import json
import os
from rapidfuzz import fuzz
from tabulate import tabulate


def calculate_similarity(truth_val, pred_val):
    if not truth_val and not pred_val: return 100.0 
    if not truth_val or not pred_val: return 0.0 

    try:
        num_t = float(str(truth_val).replace(',', '').replace(' ', ''))
        num_p = float(str(pred_val).replace(',', '').replace(' ', ''))
        
        if abs(num_t - num_p) < 1.0:
            return 100.0
    except ValueError:
        pass
    
    str_truth = str(truth_val).lower().strip()
    str_pred = str(pred_val).lower().strip()
    
    return fuzz.token_sort_ratio(str_truth, str_pred)

# 2. Hàm đệ quy để so sánh JSON (Nested Object)
def compare_objects(truth_obj, pred_obj, prefix=""):
    report = []
    
    for key, val_truth in truth_obj.items():
        current_key = f"{prefix}.{key}" if prefix else key
        val_pred = pred_obj.get(key, "")
        
        if isinstance(val_truth, dict):
            pred_sub = val_pred if isinstance(val_pred, dict) else {}
            report.extend(compare_objects(val_truth, pred_sub, current_key))
            continue
            
        if isinstance(val_truth, list):
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
    try:
        with open('data/ground_truth/ground_truth_page_1.json', 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("Create json in data/ground_truth/")
        exit()

    prediction_sample = {
        "seller": {
            "name": "CONG TY TNHH NHUA APCO", 
            "taxCode": "0107453450",
            "address": "BT65 Lam Vien..."
        },
        "invoice": {
            "number": "50",
            "totalAmountAfterTax": 839808000
        }
    }

    print(f"--- Comparing: Ground Truth vs Prediction ---")
    
    # 3. Compare
    results = compare_objects(ground_truth, prediction_sample)
    
    # 4. Calculate Score
    total_score = sum([row[3] for row in results])
    avg_score = total_score / len(results) if results else 0
    
    # Report
    headers = ["Field", "Ground Truth", "Extracted Value", "Score (%)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print(f"\nACCURACY: {avg_score:.2f}%")