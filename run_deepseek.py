import os
import json
import torch
from openai import OpenAI
from test_visual import save_single_query, test, parse_args  # 复用已有函数

# 1. 全局路径
DEV_JSON   = r"E:\GenRet\dataset\nq320k\dev.json"
PRED_JSON  = r"E:\GenRet\out\model-1-pre\top3_pred.json"
ANS_JSONL  = r"E:\GenRet\out\deepseek_answer.jsonl"

# 2. DeepSeek 配置
API_KEY   = "sk-cc68687bb5eb462fa7930b60fa315ef2"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

# 3. 主流程
def main():
    # 3-1 键盘输入问题并保存
    save_single_query()

    # 3-2 运行检索，生成 top3_pred.json
    print("正在检索 …")
    test({
        'model_name': 't5-base',
        'code_num': 512,
        'code_length': 6,
        'prev_model': './out/model-1-pre/100.pt',
        'prev_id': './out/model-1-pre/100.pt.all.code',
        'save_path': './out/model-1-pre',
        'epochs': 1,
        'dev_data': DEV_JSON,
        'corpus_data': './dataset/nq320k/corpus_lite.json'
    })

    # 3-3 读取检索结果
    with open(PRED_JSON, "r", encoding="utf-8") as f:
        sample = json.load(f)[0]

    query = sample["query"]
    docs  = sample["predicted_docs"]

    # 3-4 构造 prompt
    prompt = f"[User Query]\n{query}\n\nRetrieve Results:\n"
    for idx, doc in enumerate(docs, 1):
        prompt += f"{idx}. {doc}\n"
    prompt += "\nPlease answer the user's query based on the retrieved documents."

    # 3-5 调用 DeepSeek
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    answer = resp.choices[0].message.content

    # 3-6 保存 & 打印
    out_obj = {"query": query, "prompt": prompt, "answer": answer}
    os.makedirs(os.path.dirname(ANS_JSONL), exist_ok=True)
    with open(ANS_JSONL, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("\n=== 最终答案 ===")
    print(answer)
    print(f"\n已保存 → {ANS_JSONL}")

if __name__ == '__main__':
    main()