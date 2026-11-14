import json
import re
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# ============ 配置区 ============
API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""
DB_PATH = "wrong_questions.db"

st.set_page_config(page_title="AI课程笔记自测", page_icon="📝", layout="centered")


def get_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============ 数据库相关函数 ============
def init_db():
    """初始化数据库，创建错题表（如果不存在）。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wrong_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL UNIQUE,
            options TEXT NOT NULL,
            answer_index INTEGER NOT NULL,
            explanation TEXT,
            knowledge_point TEXT DEFAULT '未分类',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def is_question_exists(question_text: str) -> bool:
    """检查某道题是否已经存在于错题本中（按题干去重）。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM wrong_questions WHERE question = ?", (question_text,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_wrong_question(question: str, options: list, answer_index: int, 
                       explanation: str, knowledge_point: str = "未分类"):
    """将错题保存到数据库（自动去重）。"""
    if is_question_exists(question):
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO wrong_questions (question, options, answer_index, explanation, knowledge_point)
            VALUES (?, ?, ?, ?, ?)
        """, (question, json.dumps(options, ensure_ascii=False), answer_index, explanation, knowledge_point))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


def get_wrong_questions_by_knowledge_point() -> dict:
    """按知识点分组获取所有错题。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT knowledge_point, question, options, answer_index, explanation, created_at
        FROM wrong_questions
        ORDER BY knowledge_point, created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    grouped = {}
    for row in rows:
        kp, question, options_json, answer_index, explanation, created_at = row
        if kp not in grouped:
            grouped[kp] = []
        grouped[kp].append({
            "question": question,
            "options": json.loads(options_json),
            "answer_index": answer_index,
            "explanation": explanation,
            "created_at": created_at
        })
    return grouped


def get_wrong_question_stats() -> dict:
    """获取每个知识点的错题数量统计。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT knowledge_point, COUNT(*) as count
        FROM wrong_questions
        GROUP BY knowledge_point
        ORDER BY count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


SYSTEM_PROMPT = """你是一位严谨的课程助教。请根据用户提供的课程笔记，出 3 道单项选择题。

【核心要求】
1. 严禁出“送分题”或“废话题”。不要问“本文讲了什么”、“标题是什么”这种看一眼标题就能猜到的问题。
2. 题目必须深入考察笔记中的具体概念、原理、步骤、对比或细节。
3. 干扰项（错误选项）必须具有迷惑性，不能一眼排除。
4. 必须严格只输出一个 JSON 对象，不要输出任何解释性文字、不要使用 Markdown 代码块包裹。

JSON 结构如下：
{
  "questions": [
    {
      "question": "题干文本",
      "options": ["选项A文本", "选项B文本", "选项C文本", "选项D文本"],
      "answer_index": 0,
      "explanation": "解析文本",
      "knowledge_point": "知识点标签"
    }
  ]
}

其他要求：
- questions 数组必须恰好包含 3 个元素。
- 每道题的 options 必须恰好包含 4 个选项。
- answer_index 是 0 到 3 之间的整数。
- 题目必须基于笔记内容，不要编造笔记中不存在的信息。
- 使用中文出题。
- knowledge_point 字段请从笔记中提取核心知识点，用简短的中文标签表示。
"""


def build_user_prompt(note: str) -> str:
    return f"以下是课程笔记内容：\n\n---\n{note}\n---\n\n请据此生成 3 道单项选择题，并以规定的 JSON 格式返回。"


def extract_json(text: str) -> dict:
    """从模型返回文本中稳健地提取 JSON 对象。"""
    text = text.strip()
    fence_match = re.search(r"`(?:json)?\s*(.*?)`", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("模型返回内容中未找到合法 JSON。")


def validate_questions(data: dict) -> list:
    """校验并规范化题目数据结构，强制添加 ABCD 前缀。"""
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("返回的题目数量不是 3 道。")
    cleaned = []
    labels = ['A', 'B', 'C', 'D']
    
    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            raise ValueError(f"第 {i} 题格式错误。")
        question = str(q.get("question", "")).strip()
        options = q.get("options")
        answer_index = q.get("answer_index")
        explanation = str(q.get("explanation", "")).strip()
        knowledge_point = str(q.get("knowledge_point", "未分类")).strip()
        
        if not question:
            raise ValueError(f"第 {i} 题缺少题干。")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"第 {i} 题的选项数量不是 4 个。")
            
        # 强制清洗并添加 ABCD 前缀
        normalized_options = []
        for idx, opt in enumerate(options):
            opt_str = str(opt).strip()
            opt_str = re.sub(r'^[A-D][\.、\s]+', '', opt_str).strip()
            normalized_options.append(f"{labels[idx]}. {opt_str}")
            
        if not isinstance(answer_index, int) or not (0 <= answer_index <= 3):
            raise ValueError(f"第 {i} 题的正确答案索引无效。")
            
        cleaned.append({
            "question": question,
            "options": normalized_options,
            "answer_index": answer_index,
            "explanation": explanation,
            "knowledge_point": knowledge_point if knowledge_point else "未分类"
        })
    return cleaned


def generate_questions(note: str) -> list:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(note)},
        ],
        temperature=0.3,
    )
    content = response.choices[0].message.content or ""
    data = extract_json(content)
    return validate_questions(data)


# ============ 页面 ============
st.title("AI课程笔记自测")
st.caption("粘贴课程笔记，自动生成 3 道单项选择题，边学边测。")

init_db()

note = st.text_area(
    "课程笔记",
    height=260,
    placeholder="在这里粘贴你的课程笔记内容……",
)

if st.button("生成题目", type="primary", use_container_width=True):
    note_clean = note.strip()
    # 【已取消字数限制】无论输入多少字都直接交给大模型处理
    if not note_clean:
        st.warning("请先粘贴课程笔记内容。")
    elif API_KEY == "sk-你的真实key":
        st.error("请先在代码中把 API_KEY 替换为你自己的真实密钥。")
    else:
        with st.spinner("正在根据笔记生成题目……"):
            try:
                questions = generate_questions(note_clean)
                st.session_state["questions"] = questions
                st.session_state["submitted"] = False
                st.success("题目生成成功，请作答。")
            except Exception as exc:
                st.session_state.pop("questions", None)
                st.error(f"生成题目失败：{exc}")

# ============ 答题区 ============
questions = st.session_state.get("questions")
if questions:
    st.divider()
    st.subheader("自测题目")
    user_answers = []
    for idx, q in enumerate(questions, start=1):
        st.markdown(f"**第 {idx} 题：{q['question']}**")
        choice = st.radio(
            label=f"第 {idx} 题选项",
            options=list(range(len(q["options"]))),
            format_func=lambda i, opts=q["options"]: opts[i],
            key=f"q_{idx}",
            label_visibility="collapsed",
        )
        user_answers.append(choice)

    if st.button("提交答案", type="secondary", use_container_width=True):
        st.session_state["submitted"] = True
        
    if st.session_state.get("submitted"):
        st.divider()
        score = 0
        for idx, (q, ans) in enumerate(zip(questions, user_answers), start=1):
            correct = q["answer_index"]
            is_right = ans == correct
            if is_right:
                score += 1
            else:
                save_wrong_question(
                    question=q["question"],
                    options=q["options"],
                    answer_index=q["answer_index"],
                    explanation=q["explanation"],
                    knowledge_point=q.get("knowledge_point", "未分类")
                )
            
            st.markdown(f"### 第 {idx} 题 {'✅ 正确' if is_right else '❌ 错误'}")
            st.markdown(f"- 你的答案：{q['options'][ans]}")
            st.markdown(f"- 正确答案：{q['options'][correct]}")
            if q["explanation"]:
                st.markdown(f"- 解析：{q['explanation']}")
        
        st.info(f"得分：{score} / {len(questions)}")

# ============ 侧边栏：独立折叠面板 ============
with st.sidebar:
    # 面板1：学习分析图表
    with st.expander("📊 学习分析", expanded=False):
        stats = get_wrong_question_stats()
        if stats:
            df_stats = pd.DataFrame(list(stats.items()), columns=["知识点", "错题数"])
            fig = px.bar(df_stats, x="知识点", y="错题数", color="错题数", 
                         color_continuous_scale="Reds", title="各知识点错题分布")
            fig.update_layout(xaxis_tickangle=-45, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("当前没有错题，继续保持！💪")

    # 面板2：错题本列表
    with st.expander("📕 错题本", expanded=False):
        grouped_questions = get_wrong_questions_by_knowledge_point()
        if not grouped_questions:
            st.caption("暂无错题记录。")
        else:
            st.markdown(f"**共 {sum(len(v) for v in grouped_questions.values())} 道错题**")
            st.divider()
            
            for kp, qs in grouped_questions.items():
                with st.expander(f"📌 {kp} ({len(qs)}题)", expanded=False):
                    for q_idx, q in enumerate(qs, 1):
                        st.markdown(f"**{q_idx}. {q['question']}**")
                        for opt_idx, opt in enumerate(q["options"]):
                            prefix = "✅" if opt_idx == q["answer_index"] else "○"
                            st.markdown(f"{prefix} {opt}")
                        if q["explanation"]:
                            st.markdown(f"*解析：{q['explanation']}*")
                        st.markdown(f"*做错时间：{q['created_at']}*")
                        st.divider()