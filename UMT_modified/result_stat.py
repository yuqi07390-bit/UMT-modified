import json


def analyze_jsonl(path):
    metrics = ["Rank1@0.5", "Rank1@0.7", "Rank5@0.5", "Rank5@0.7"]

    # 用于保存每个指标的最佳结果
    best = {
        m: {"value": float("-inf"), "epoch": None, "full_result": None}
        for m in metrics
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            # 只处理 mode 为 val 的行
            if data.get("mode") != "val":
                continue

            for m in metrics:
                if m in data and data[m] > best[m]["value"]:
                    best[m]["value"] = data[m]
                    best[m]["epoch"] = data.get("epoch")
                    best[m]["full_result"] = data

    return best


# 示例调用
if __name__ == "__main__":
    path = "work_dirs/umt_base_vo_100e_charades_clipencoder_train_nofreeze_0/metrics.json"  # 修改为你的 jsonl 文件路径
    result = analyze_jsonl(path)

    for metric, info in result.items():
        print(f"\n=== {metric} ===")
        print(f"最大值: {info['value']}")
        print(f"对应 epoch: {info['epoch']}")
        print("完整结果:")
        print(info["full_result"])