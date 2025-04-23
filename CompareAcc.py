import pandas as pd
import numpy as np

# ========================
# 配置路径
# ========================
PREDICT_CSV = "submission.csv"                 # 你的模型预测结果
ANSWER_CSV = "labels.csv"   # 正确答案文件

# ========================
# 分析函数
# ========================
def analyze_prediction(pred_path, answer_path):
    pred_df = pd.read_csv(pred_path)
    ans_df = pd.read_csv(answer_path)

    # 确保 id 对齐（防止顺序错乱）
    pred_df = pred_df.sort_values("id").reset_index(drop=True)
    ans_df = ans_df.sort_values("id").reset_index(drop=True)

    true_labels = ans_df["label"].to_numpy()
    pred_labels = pred_df["label"].to_numpy()

    # 总体准确率
    total = len(true_labels)
    correct = (true_labels == pred_labels).sum()
    accuracy = 100. * correct / total
    print(f"\n✅ 总体准确率: {accuracy:.2f}% ({correct}/{total})")

    # 初始化每个数字的统计
    stats = {i: {"total": 0, "wrong": 0} for i in range(10)}

    for true, pred in zip(true_labels, pred_labels):
        stats[true]["total"] += 1
        if true != pred:
            stats[true]["wrong"] += 1

    # 输出每个类别的错误率
    print("\n📊 每个数字的错误率分析：")
    print(f"{'数字':^6} {'总数':^6} {'错误':^6} {'错误率':^8}")
    print("-" * 30)
    for digit in range(10):
        total = stats[digit]["total"]
        wrong = stats[digit]["wrong"]
        error_rate = 100. * wrong / total if total > 0 else 0
        print(f"{digit:^6} {total:^6} {wrong:^6} {error_rate:>7.2f}%")

# ========================
# 主程序
# ========================
if __name__ == "__main__":
    analyze_prediction(PREDICT_CSV, ANSWER_CSV)