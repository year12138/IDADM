import os
import json
from openai import OpenAI

# ========== 路径配置 ==========
DEV_JSON   = r"E:\IDADM\dataset\nq320k\dev.json"
ANS_JSONL  = r"E:\IDADM\out\direct_deepseek_answers.jsonl"

# ========== DeepSeek 配置 ==========
API_KEY = "sk-cc68687bb5eb462fa7930b60fa315ef2"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

# ========== 主流程 ==========
def main():
    # 1. 加载 dev.json 中的 query（每项是 [query_str, label]）
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data:
        if not isinstance(item, list) or len(item) < 1:
            continue  # 非法跳过

        query = item[0]
        prompt = f"[User Query]\n{query}\n\nPlease answer the user's query directly."

        # 2. 调用 DeepSeek API
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"[错误] 处理 query 时失败: {query}\n{e}")
            answer = "[ERROR] Failed to generate answer."

        # 3. 构建输出项
        out_obj = {
            "query": query,
            "answer": answer
        }
        results.append(out_obj)

        print(f"\n[Query] {query}")
        print(f"[Answer] {answer}")
        print("-" * 50)

    # 4. 保存为 .jsonl 文件
    os.makedirs(os.path.dirname(ANS_JSONL), exist_ok=True)
    with open(ANS_JSONL, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 全部完成，保存至：{ANS_JSONL}")

if __name__ == '__main__':
    main()
