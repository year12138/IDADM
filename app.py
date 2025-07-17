from flask import Flask, render_template, request, jsonify
import json
import subprocess
import os

app = Flask(__name__)

# 配置路径
DEV_JSON = r"E:\IDADM\dataset\nq320k\dev.json"
RETRIEVE_SCRIPT = r"E:\IDADM\test_visual.py"
RETRIEVE_RESULT = r"E:\IDADM\out\model-1-pre\top3_pred.json"
ANSWER_SCRIPT = r"E:\IDADM\generate_model.py"
ANSWER_RESULT = r"E:\IDADM\out\deepseek_answers.jsonl"
DIRECT_ANSWER_SCRIPT = r"E:\IDADM\direct_generate_model.py"
DIRECT_ANSWER_RESULT = r"E:\IDADM\out\direct_deepseek_answers.jsonl"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_query():
    query = request.form['query']
    data = [[query, 0]]
    with open(DEV_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return jsonify({"status": "Query saved."})


@app.route('/retrieve', methods=['POST'])
def retrieve():
    subprocess.run(["python", RETRIEVE_SCRIPT], check=True)
    with open(RETRIEVE_RESULT, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    pred_docs = predictions[0]["predicted_docs"]
    output = [f"Document {i + 1}: {doc}" for i, doc in enumerate(pred_docs)]
    return jsonify(output)


@app.route('/answer', methods=['POST'])
def answer():
    # 运行脚本生成答案
    subprocess.run(["python", ANSWER_SCRIPT], check=True)

    # 读取整个文件内容（注意不是逐行）
    with open(ANSWER_RESULT, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        data = json.loads(content)
        answer_text = data.get("answer", "")
        return jsonify({"answer": answer_text})
    except json.JSONDecodeError as e:
        return jsonify({"answer": f"⚠ JSON 解析失败：{str(e)}"})


@app.route('/direct_answer', methods=['POST'])
def direct_answer():
    # 运行直接生成脚本
    subprocess.run(["python", DIRECT_ANSWER_SCRIPT], check=True)

    # 读取多行 JSON（假设是标准的 JSON 块）
    with open(DIRECT_ANSWER_RESULT, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        data = json.loads(content)
        answer_text = data.get("answer", "")
        return jsonify({"direct_answer": answer_text})
    except json.JSONDecodeError as e:
        return jsonify({"direct_answer": f"⚠ JSON 解析失败：{str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)
