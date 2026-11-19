import json
import re
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# ============ 配置区 ============
API_KEY = "ollama"                             # 本地模型固定写这个，不需要改
BASE_URL = "http://localhost:11434/v1"         # 本地 Ollama 的默认地址
MODEL_NAME = "qwen2.5:7b"                      # 你电脑上跑的模型名字
DB_PATH = "wrong_questions.db"

st.set_page_config(page_title="AI课程笔记自测", page_icon="📝", layout="centered")


def get_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============ 数据库相关函数 ============
def init_db():
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
            subject TEXT DEFAULT '未分类',
            grade TEXT DEFAULT '未分类',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("PRAGMA table_info(wrong_questions)")
    columns = {row[1] for row in cursor.fetchall()}
    if "subject" not in columns:
        cursor.execute("ALTER TABLE wrong_questions ADD COLUMN subject TEXT DEFAULT '未分类'")
    if "grade" not in columns:
        cursor.execute("ALTER TABLE wrong_questions ADD COLUMN grade TEXT DEFAULT '未分类'")
    conn.commit()
    conn.close()


def is_question_exists(question_text: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM wrong_questions WHERE question = ?", (question_text,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_wrong_question(question: str, options: list, answer_index: int,
                        explanation: str, knowledge_point: str = "未分类",
                        subject: str = "未分类", grade: str = "未分类"):
    if is_question_exists(question):
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO wrong_questions
                (question, options, answer_index, explanation, knowledge_point, subject, grade)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (question, json.dumps(options, ensure_ascii=False), answer_index,
              explanation, knowledge_point, subject, grade))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


def get_wrong_questions_grouped_by_subject_and_kp() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subject, knowledge_point, question, options, answer_index,
               explanation, grade, created_at
        FROM wrong_questions
        ORDER BY subject, knowledge_point, created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    grouped = {}
    for row in rows:
        subject, kp, question, options_json, answer_index, explanation, grade, created_at = row
        subject = subject or "未分类"
        kp = kp or "未分类"
        grouped.setdefault(subject, {}).setdefault(kp, []).append({
            "question": question,
            "options": json.loads(options_json) if options_json else [],
            "answer_index": answer_index,
            "explanation": explanation,
            "grade": grade or "未分类",
            "created_at": created_at,
        })
    return grouped


def get_filter_options() -> tuple:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT subject FROM wrong_questions
        WHERE subject IS NOT NULL AND subject != ''
        ORDER BY subject
    """)
    subjects = [r[0] for r in cursor.fetchall()]
    cursor.execute("""
        SELECT DISTINCT grade FROM wrong_questions
        WHERE grade IS NOT NULL AND grade != ''
        ORDER BY grade
    """)
    grades = [r[0] for r in cursor.fetchall()]
    conn.close()
    return subjects, grades


def get_filtered_stats(subject: str = "全部", grade: str = "全部") -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = """
        SELECT knowledge_point, COUNT(*) as count
        FROM wrong_questions
        WHERE 1=1
    """
    params = []
    if subject and subject != "全部":
        sql += " AND subject = ?"
        params.append(subject)
    if grade and grade != "全部":
        sql += " AND grade = ?"
        params.append(grade)
    sql += " GROUP BY knowledge_point ORDER BY count DESC"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def get_filtered_total(subject: str = "全部", grade: str = "全部") -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = "SELECT COUNT(*) FROM wrong_questions WHERE 1=1"
    params = []
    if subject and subject != "全部":
        sql += " AND subject = ?"
        params.append(subject)
    if grade and grade != "全部":
        sql += " AND grade = ?"
        params.append(grade)
    cursor.execute(sql, params)
    total = cursor.fetchone()[0]
    conn.close()
    return total



def get_subject_error_counts() -> dict:
    """按学科统计错题数量，返回 {学科: 错题数} 字典"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subject, COUNT(*) as count
        FROM wrong_questions
        WHERE subject IS NOT NULL AND subject != ""
        GROUP BY subject
        ORDER BY count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


# ============ JSON 解析与清洗 ============
def _sanitize_json_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # 去除 Markdown 代码块包裹（支持 ```json ... ``` 和 `...`）
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        fence_match = re.search(r"`(?:json)?\s*(.*?)`", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    def _fix_escapes(match):
        ch = match.group(1)
        if ch in '"\\/bfnrtu':
            return match.group(0)
        return "\\\\" + ch

    text = re.sub(r'\\(.)', _fix_escapes, text, flags=re.DOTALL)
    return text


def _fix_single_quotes(text: str) -> str:
    """将 JSON 中的单引号替换为双引号（简单场景）"""
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r"(?<!\w)'([^'\\]*(?:\\.[^'\\]*)*)'(?!\w)", r'"\1"', text)
    return text


def _ensure_json_safe(obj):
    """递归确保对象中不包含不可哈希类型（如 set），将 set 转为 list"""
    if isinstance(obj, dict):
        return {k: _ensure_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_ensure_json_safe(item) for item in obj]
    elif isinstance(obj, set):
        return [_ensure_json_safe(item) for item in obj]
    return obj


def _repair_truncated_json(text: str) -> str:
    """尝试修复被截断的 JSON：补全缺失的括号和引号"""
    if not text:
        return text
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    unescaped = re.sub(r'\\"', '', text)
    open_quotes = unescaped.count('"') % 2
    repaired = text.rstrip()
    if open_quotes:
        repaired += '"'
    repaired = re.sub(r',\s*$', '', repaired)
    for _ in range(max(open_brackets, 0)):
        repaired = re.sub(r',\s*$', '', repaired)
        repaired += ']'
    for _ in range(max(open_braces, 0)):
        repaired = re.sub(r',\s*$', '', repaired)
        repaired += '}'
    return repaired


def extract_json(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型返回内容为空。")

    sanitized = _sanitize_json_text(text)

    # 尝试 1：直接解析
    try:
        return json.loads(sanitized, strict=False)
    except json.JSONDecodeError:
        pass

    # 尝试 2：去除尾部逗号后解析
    cleaned = re.sub(r',\s*([}\]])', r'\1', sanitized)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # 尝试 3：单引号转双引号后解析
    fixed = _fix_single_quotes(cleaned)
    try:
        return json.loads(fixed, strict=False)
    except json.JSONDecodeError:
        pass

    # 尝试 4：使用 ast.literal_eval 兜底
    try:
        import ast
        result = ast.literal_eval(cleaned)
        if isinstance(result, dict):
            return _ensure_json_safe(result)
    except (ValueError, SyntaxError, TypeError):
        pass

    # 尝试 5：对单引号修复后的文本再用 ast.literal_eval
    try:
        import ast
        result = ast.literal_eval(fixed)
        if isinstance(result, dict):
            return _ensure_json_safe(result)
    except (ValueError, SyntaxError, TypeError):
        pass

    # 尝试 6：修复截断的 JSON
    repaired = _repair_truncated_json(cleaned)
    if repaired != cleaned:
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            pass
        repaired_fixed = _fix_single_quotes(repaired)
        try:
            return json.loads(repaired_fixed, strict=False)
        except json.JSONDecodeError:
            pass
        try:
            import ast
            result = ast.literal_eval(repaired)
            if isinstance(result, dict):
                return _ensure_json_safe(result)
        except (ValueError, SyntaxError, TypeError):
            pass

    raise ValueError(f"模型返回内容不是合法 JSON，无法解析。原始内容前200字符：{sanitized[:200]}")


# ============ 题目生成 ============
SYSTEM_PROMPT_TEMPLATE = """你是一位严谨的课程助教。请根据用户提供的课程笔记，出 3 道单项选择题。

【本次出题上下文】
- 学科：{subject}
- 年级/难度：{grade}
请严格按照上述学科和年级对应的知识深度与难度出题。

【语言格式约束 —— 必须遵守】
- 如果学科是\u201c英语\u201d：题干和选项以英文为主，但可以在题干末尾或选项中用括号附加中文释义来辅助理解（例如："What does the word \u2018ubiquitous\u2019 mean? (\u2018ubiquitous\u2019 \u7684\u610f\u601d\u662f\u4ec0\u4e48\uff1f)"）。解析（explanation）部分必须使用纯英文撰写，不要混入中文。
- 如果学科是其他（如数学、生物、计算机等），请使用中文生成题干、四个选项以及解析。

【考点覆盖与题目差异化约束 —— 最高优先级，违反将导致题目作废】
1. **禁止重复考点**：3 道题目必须分别考察笔记中 3 个不同的核心知识点，每道题的 knowledge_point 标签必须互不相同。
2. **禁止雷同设问**：3 道题的设问角度和题型必须有明显差异。例如：
   - 第 1 题可以是概念辨析题（\u201c以下哪个说法正确？\u201d）
   - 第 2 题可以是计算/推导题（\u201c求解...\u201d 或 \u201c...的值是多少？\u201d）
   - 第 3 题可以是条件判断题（\u201c以下哪个条件不是...的必要条件？\u201d）或应用题
   绝不能 3 道题都是\u201c以下哪个是正确的\u201d这种单一模式。
3. **禁止换皮题**：不能把同一道题换个数字或换个变量名就当作新题。每道题必须考察本质上不同的知识维度。

【选项质量约束 —— 必须遵守】
1. **选项要有区分度**：四个选项虽然属于同一题型维度（如同为公式、同为名词），但内容必须有明确差异，不能让考生觉得四个选项\u201c看起来都差不多\u201d。
   - \u2705 好的例子（数学）：["x = 2", "x = -2", "x = 0", "x = 1"] \u2014\u2014 同为方程解，但数值各不相同
   - \u274c 差的例子（数学）：["f\u2032(c) = 0", "f\u2032(c) = 1", "f\u2032(c) = -1", "f\u2032(c) = 2"] \u2014\u2014 只是改了个数字，毫无区分度
   - \u2705 好的例子（英语）：["ubiquitous (\u666e\u904d\u5b58\u5728\u7684)", "obsolete (\u8fc7\u65f6\u7684)", "ambiguous (\u6a21\u68f1\u4e24\u53ef\u7684)", "benevolent (\u6148\u5584\u7684)"] \u2014\u2014 同为形容词，但含义完全不同
   - \u274c 差的例子（英语）：["important", "very important", "extremely important", "most important"] \u2014\u2014 只是加了程度副词
2. **干扰项要有迷惑性**：错误选项应该是学生容易混淆的概念，而不是随意编造的无关内容。
3. **禁止无关项**：绝对禁止出现与题干语境完全无关的词汇（例如：数学题中出现\u201c公元前\u201d、生物题中出现\u201c操作系统\u201d）。

【题干硬性约束 —— 必须遵守，违反将导致题目作废】
1. 题干必须是一句简练的问句，以问号结尾（中文用？英文用?）。
2. **严禁泄露答案**：题干中绝对不能出现正确答案的内容或等价表述。题干应该只给出条件，让考生自己推导结论。
3. 题干中严禁包含任何选项内容，包括但不限于：A/B/C/D 标号、选项文本、\u201c以下哪项\u201d、\u201c下列选项\u201d等暗示选项的表述。
4. 题干只描述问题的条件和设问，不得把任何选项原文或答案结论写进题干。
5. 选项必须全部放在 options 数组中，不要把选项混入 question 字段。
6. 自检方法：写完题干后，检查题干是否已经包含了某个选项的关键内容\u2014\u2014如果是，说明你在泄露答案，必须重写。

【数学公式与特殊符号规范 —— 严格执行】
1. **纯文本优先**：所有数学公式、化学式、编程代码必须使用纯文本线性表达。
   - \u2705 正确：lim(x->0), sin^2(x), H2O, def func():
   - \u274c 禁止：\\lim_{{x\\to 0}}, \\sin^2 x, $H_2O$, ```python```
2. 严禁输出任何 LaTeX 控制序列（以反斜杠 \\ 开头的命令），这会导致系统解析崩溃。
3. 上下标请使用 ^ 和 _ 或直接线性书写，不要使用 LaTeX 语法。

【输出完整性自检】
在输出 JSON 前，请务必自检：
- options 数组是否严格包含 4 个非空字符串？
- 四个选项之间是否有足够的区分度（不能只是改个数字或加个副词）？
- 3 道题的知识点和设问角度是否互不重复？
- 题干是否泄露了答案？
- **answer_index 是否与题干的设问方向一致？** 如果题目问"哪个不正确"，answer_index 必须指向那个不正确的选项，而不是指向那个正确的描述。

【核心要求】
1. 严禁出\u201c送分题\u201d或\u201c废话题\u201d。不要问\u201c本文讲了什么\u201d、\u201c标题是什么\u201d这种看一眼标题就能猜到的问题。
2. 题目必须深入考察笔记中的具体概念、原理、步骤、对比或细节。
3. 必须严格只输出一个 JSON 对象，不要输出任何解释性文字、不要使用 Markdown 代码块包裹。

JSON 结构如下：
{{
  "questions": [
    {{
      "question": "一句简练的问句，不含任何选项内容",
      "options": ["选项A文本", "选项B文本", "选项C文本", "选项D文本"],
      "answer_index": 0,
      "explanation": "解析文本",
      "knowledge_point": "知识点标签"
    }}
  ]
}}

其他要求：
- questions 数组必须恰好包含 3 个元素。
- **options 必须是一个包含 4 个字符串的 JSON 数组**，不要返回一整个长字符串！
- answer_index 是 0 到 3 之间的整数，指向的是**符合题目要求的正确选项**。
  - 如果题目问"以下哪个是正确的？"，answer_index 指向那个正确的选项。
  - 如果题目问"以下哪个是不正确的？"或"哪一项是错误的？"，answer_index 必须指向那个**不正确/错误**的选项（因为那才是本题的正确答案）。
  - 如果题目问"以下哪个不是...的条件？"，answer_index 指向那个**不是**该条件的选项。
  - 总之：answer_index 永远指向"考生应该选的那个选项"，即符合题干设问方向的选项。
- 题目必须基于笔记内容，不要编造笔记中不存在的信息。
- knowledge_point 字段请从笔记中提取核心知识点，用简短的中文标签表示。
"""


def build_system_prompt(subject: str, grade: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(subject=subject or "未指定", grade=grade or "未指定")


def build_user_prompt(note: str, subject: str, grade: str) -> str:
    if subject == "英语":
        lang_req = "题干和选项以英文为主，可用括号附加中文释义辅助理解。解析（explanation）必须使用纯英文，不要混入中文。"
    else:
        lang_req = "请使用中文生成题干、选项和解析。"
    return (
        f"学科：{subject}\n年级：{grade}\n\n"
        f"以下是课程笔记内容：\n\n---\n{note}\n---\n\n"
        f"请据此生成 3 道单项选择题，并以规定的 JSON 格式返回。"
        f"注意：{lang_req}"
        f"3 道题必须考察不同知识点，设问角度要有差异（概念题/计算题/判断题各一道为佳）。"
        f"选项之间要有足够区分度，不能只是改个数字或加个副词。"
        f"题干严禁包含答案内容。"
        f"数学公式请用标准文本表达（如 lim(x->0)），不要使用带反斜杠的 LaTeX。"
    )


def _clean_option_text(opt) -> str:
    opt_str = str(opt).strip() if opt is not None else ""
    opt_str = re.sub(r'^\s*[A-Da-d]\s*[\.、:：\)\]\-]\s*', '', opt_str)
    return opt_str.strip()


def _strip_options_from_question(question: str) -> str:
    if not question:
        return question
    q = question.strip()
    m = re.search(r'(?:(?<=\s)|(?<=[。？?！!；;：:])|^)\s*[AＡ]\s*[\.、:：\)\]]\s*\S', q)
    if m:
        q = q[:m.start()].strip()
    q = re.sub(r'[（(]\s*(?:以下|下列)[^）)]*[）)]\s*$', '', q).strip()
    return q


def _normalize_options(options, labels=("A", "B", "C", "D")):
    normalized = []
    
    if isinstance(options, str):
        parts = re.split(r'\n|(?=\b[A-Da-d][\.、:：\)\]])', options)
        options = [p.strip() for p in parts if p.strip()]

    if isinstance(options, (list, tuple)):
        for opt in options:
            if isinstance(opt, dict):
                opt = opt.get("text") or opt.get("value") or opt.get("label") or str(opt)
            elif isinstance(opt, (list, tuple)):
                opt = str(opt[0]) if opt else ""
            elif not isinstance(opt, str):
                opt = str(opt)
            normalized.append(_clean_option_text(opt))
    elif isinstance(options, dict):
        for key in sorted(options.keys()):
            val = options[key]
            normalized.append(_clean_option_text(str(val) if not isinstance(val, str) else val))
            
    normalized = normalized[:4]
    while len(normalized) < 4:
        normalized.append("（选项缺失，请重新生成）" if not isinstance(options, str) else "(Option missing, please regenerate)")
    for i, opt in enumerate(normalized):
        if not opt:
            normalized[i] = "（选项缺失，请重新生成）" if not isinstance(options, str) else "(Option missing, please regenerate)"
            
    return [f"{labels[i]}. {normalized[i]}" for i in range(4)]


def _coerce_answer_index(answer_index, options_count=4) -> int:
    try:
        idx = int(answer_index)
    except (TypeError, ValueError):
        return 0
    if idx < 0 or idx >= options_count:
        return 0
    return idx


def _normalize_question_keys(data: dict) -> dict:
    """将中文键名统一映射为英文键名"""
    key_map = {
        "问题": "questions", "题目": "questions", "试题": "questions",
        "题干": "question", "选项": "options",
        "答案": "answer_index", "答案索引": "answer_index", "正确答案": "answer_index",
        "解析": "explanation", "解释": "explanation", "说明": "explanation",
        "知识点": "knowledge_point", "考点": "knowledge_point", "标签": "knowledge_point",
    }
    normalized = {}
    for k, v in data.items():
        mapped_key = key_map.get(k, k)
        if isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    new_item = {}
                    for ik, iv in item.items():
                        new_item[key_map.get(ik, ik)] = iv
                    new_list.append(new_item)
                else:
                    new_list.append(item)
            normalized[mapped_key] = new_list
        else:
            normalized[mapped_key] = v
    return normalized


def validate_questions(data: dict) -> list:
    if not isinstance(data, dict):
        raise ValueError("模型返回的 JSON 顶层不是对象。")
    data = _normalize_question_keys(data)
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("返回的题目列表为空或格式错误。")
    if len(questions) != 3:
        if len(questions) > 3:
            questions = questions[:3]
        else:
            raise ValueError(f"返回的题目数量为 {len(questions)}，不是 3 道。")

    cleaned = []
    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            raise ValueError(f"第 {i} 题格式错误（不是对象）。")
        question = _strip_options_from_question(str(q.get("question", "")).strip())
        if not question:
            raise ValueError(f"第 {i} 题缺少题干。")

        options = _normalize_options(q.get("options"))
        answer_index = _coerce_answer_index(q.get("answer_index"), len(options))
        explanation = str(q.get("explanation", "") or "").strip()
        knowledge_point = str(q.get("knowledge_point", "") or "").strip() or "未分类"

        cleaned.append({
            "question": question,
            "options": options,
            "answer_index": answer_index,
            "explanation": explanation,
            "knowledge_point": knowledge_point,
        })
    return cleaned


def _check_answer_leaked(question_text: str, options: list, answer_index: int) -> bool:
    """检查题干是否泄露了正确答案的关键内容"""
    if not question_text or not options or answer_index < 0 or answer_index >= len(options):
        return False
    correct_opt = options[answer_index]
    # 去掉选项前缀（如 "A. ", "B. "）
    clean_correct = re.sub(r'^[A-D]\.\s*', '', correct_opt).strip()
    if not clean_correct or len(clean_correct) < 2:
        return False
    # 如果正确答案的核心内容（>=4字符）完整出现在题干中，视为泄露
    if len(clean_correct) >= 4 and clean_correct in question_text:
        return True
    # 对于较短的答案，检查去除空格后是否匹配
    q_nospace = re.sub(r'\s+', '', question_text)
    c_nospace = re.sub(r'\s+', '', clean_correct)
    if len(c_nospace) >= 4 and c_nospace in q_nospace:
        return True
    return False


def _has_critical_defect(questions: list) -> bool:
    """检查题目是否存在严重缺陷（选项缺失过多或题干泄露答案），需要触发重试"""
    for q in questions:
        opts = q.get("options", [])
        # 如果超过 2 个选项是占位符或为空，视为严重缺陷
        placeholder_count = sum(
            1 for o in opts
            if not o or "选项缺失" in str(o) or "Option missing" in str(o)
        )
        if placeholder_count >= 2:
            return True
        # 检查题干是否泄露了正确答案
        answer_idx = q.get("answer_index", 0)
        question_text = q.get("question", "")
        if _check_answer_leaked(question_text, opts, answer_idx):
            return True
    return False


def generate_questions(note: str, subject: str, grade: str, max_retries: int = 2) -> list:
    client = get_client()
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": build_system_prompt(subject, grade)},
                    {"role": "user", "content": build_user_prompt(note, subject, grade)},
                ],
                temperature=0.3 if attempt == 0 else 0.5,
            )
            resp_content = response.choices[0].message.content or ""
            data = extract_json(resp_content)
            questions = validate_questions(data)
            
            if _has_critical_defect(questions) and attempt < max_retries:
                for q in questions:
                    opts = q.get("options", [])
                    placeholder_count = sum(
                        1 for o in opts
                        if not o or "选项缺失" in str(o) or "Option missing" in str(o)
                    )
                    if placeholder_count >= 2:
                        last_error = "生成的题目选项残缺过多，正在尝试重新生成..."
                        break
                    if _check_answer_leaked(q.get("question", ""), opts, q.get("answer_index", 0)):
                        last_error = "检测到题干泄露了答案内容，正在尝试重新生成..."
                        break
                else:
                    last_error = "生成的题目存在缺陷，正在尝试重新生成..."
                continue
                
            return questions
            
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                continue
            raise
    
    if last_error:
        raise last_error if isinstance(last_error, Exception) else ValueError(str(last_error))
    raise ValueError("生成题目失败：已达到最大重试次数。")


# ============ 页面 ============
st.title("AI课程笔记自测")
st.caption("粘贴课程笔记，自动生成 3 道单项选择题，边学边测。")

init_db()

SUBJECTS = ["数学", "生物", "计算机", "物理", "化学", "英语", "历史", "其他"]
GRADES = ["大一", "大二", "大三", "大四", "研一", "研二", "高中", "初中", "其他"]

col1, col2 = st.columns(2)
with col1:
    subject = st.selectbox("学科", SUBJECTS, index=0)
with col2:
    grade = st.selectbox("年级", GRADES, index=0)

note = st.text_area(
    "课程笔记",
    height=260,
    placeholder="在这里粘贴你的课程笔记内容……",
)

if st.button("生成题目", type="primary", use_container_width=True):
    note_clean = note.strip()
    if not note_clean:
        st.warning("请先粘贴课程笔记内容。")
    elif API_KEY == "sk-你的真实key":
        st.error("请先在代码中把 API_KEY 替换为你自己的真实密钥。")
    else:
        with st.spinner("正在根据笔记生成题目……"):
            try:
                questions = generate_questions(note_clean, subject, grade)
                st.session_state["questions"] = questions
                st.session_state["submitted"] = False
                st.session_state["subject"] = subject
                st.session_state["grade"] = grade
                st.session_state["note"] = note_clean
                st.success("题目生成成功，请作答。")
            except Exception as exc:
                st.session_state.pop("questions", None)
                st.error(f"生成题目失败：{exc}")

# ============ 答题区 ============
questions = st.session_state.get("questions")
if questions:
    st.divider()
    st.subheader("自测题目")
    cur_subject = st.session_state.get("subject", subject)
    cur_grade = st.session_state.get("grade", grade)
    st.caption(f"学科：{cur_subject} ｜ 年级：{cur_grade}")

    user_answers = []
    for idx, q in enumerate(questions, start=1):
        st.markdown(f"**第 {idx} 题：{q['question']}**")
        
        opts = q.get("options") or []
        if not isinstance(opts, list) or len(opts) < 4:
            opts = _normalize_options(opts)
            
        q["answer_index"] = _coerce_answer_index(q.get("answer_index"), len(opts))
        q["options"] = opts

        choice = st.radio(
            label=f"第 {idx} 题选项",
            options=list(range(len(opts))),
            index=0,
            format_func=lambda i, o=opts: o[i],
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
            correct = _coerce_answer_index(q.get("answer_index"), len(q["options"]))
            ans = ans if isinstance(ans, int) and 0 <= ans < len(q["options"]) else 0
            is_right = ans == correct
            if is_right:
                score += 1
            else:
                save_wrong_question(
                    question=q["question"],
                    options=q["options"],
                    answer_index=correct,
                    explanation=q.get("explanation", ""),
                    knowledge_point=q.get("knowledge_point", "未分类"),
                    subject=cur_subject,
                    grade=cur_grade,
                )

            st.markdown(f"### 第 {idx} 题 {'✅ 正确' if is_right else '❌ 错误'}")
            st.markdown(f"- 你的答案：{q['options'][ans]}")
            st.markdown(f"- 正确答案：{q['options'][correct]}")
            if q.get("explanation"):
                st.markdown(f"- 解析：{q['explanation']}")

        st.info(f"得分：{score} / {len(questions)}")

        # ---- 操作按钮 ----
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 再来一组", type="primary", use_container_width=True, key="regen_btn"):
                saved_note = st.session_state.get("note", "")
                saved_subject = st.session_state.get("subject", subject)
                saved_grade = st.session_state.get("grade", grade)
                if saved_note:
                    with st.spinner("正在重新生成题目……"):
                        try:
                            new_questions = generate_questions(saved_note, saved_subject, saved_grade)
                            st.session_state["questions"] = new_questions
                            st.session_state["submitted"] = False
                            # 清除旧的 radio 选择状态
                            for k in list(st.session_state.keys()):
                                if k.startswith("q_"):
                                    del st.session_state[k]
                            st.rerun()
                        except Exception as exc:
                            st.error(f"重新生成失败：{exc}")
                else:
                    st.warning("未找到之前的笔记内容，请返回上方重新粘贴笔记。")
        with btn_col2:
            if st.button("📝 换知识点", type="secondary", use_container_width=True, key="switch_kp_btn"):
                # 清除所有答题相关状态，回到输入区
                st.session_state.pop("questions", None)
                st.session_state.pop("submitted", None)
                st.session_state.pop("note", None)
                for k in list(st.session_state.keys()):
                    if k.startswith("q_"):
                        del st.session_state[k]
                st.rerun()


# ============ 侧边栏：独立折叠面板 ============
with st.sidebar:
    with st.expander("📊 薄弱知识点分析", expanded=True):
        subject_counts = get_subject_error_counts()
        grand_total = sum(subject_counts.values()) if subject_counts else 0

        if grand_total == 0:
            st.info("暂无错题数据，继续保持！💪")
        else:
            # ---- 圆环图：各学科错题比例 ----
            subjects_list = list(subject_counts.keys())
            counts_list = list(subject_counts.values())

            # 配色方案
            color_palette = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
                "#BB8FCE", "#85C1E9"
            ]
            colors = [color_palette[i % len(color_palette)] for i in range(len(subjects_list))]

            fig_donut = go.Figure(data=[go.Pie(
                labels=subjects_list,
                values=counts_list,
                hole=0.45,
                marker=dict(colors=colors),
                textinfo="label+percent",
                textfont_size=13,
                hovertemplate="<b>%{label}</b><br>错题数：%{value}<br>占比：%{percent}<extra></extra>",
                pull=[0.03] * len(subjects_list),
            )])
            fig_donut.update_layout(
                title=dict(text=f"各学科错题分布（共 {grand_total} 题）", font=dict(size=15)),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            # 圆环中心注释
            fig_donut.add_annotation(
                text=f"<b>{grand_total}</b><br>总错题",
                x=0.5, y=0.5,
                font=dict(size=16, color="#333"),
                showarrow=False,
            )
            st.plotly_chart(fig_donut, width="stretch", key="donut_subject_chart")

            # ---- 学科知识点明细（可折叠） ----
            with st.expander("🔍 查看各学科薄弱知识点明细", expanded=False):
                selected_subject = st.selectbox(
                    "选择学科",
                    subjects_list,
                    key="analysis_subject_select",
                    label_visibility="collapsed",
                )

                if selected_subject:
                    kp_stats = get_filtered_stats(subject=selected_subject)
                    kp_total = get_filtered_total(subject=selected_subject)

                    if kp_stats:
                        kp_names = list(kp_stats.keys())
                        kp_values = list(kp_stats.values())

                        kp_colors = [color_palette[(i + 3) % len(color_palette)] for i in range(len(kp_names))]
                        fig_kp = go.Figure(data=[go.Pie(
                            labels=kp_names,
                            values=kp_values,
                            hole=0.4,
                            marker=dict(colors=kp_colors),
                            textinfo="label+value",
                            textfont_size=12,
                            hovertemplate="<b>%{label}</b><br>错题数：%{value}<br>占比：%{percent}<extra></extra>",
                        )])
                        fig_kp.update_layout(
                            title=dict(text=f"📚 {selected_subject} — 薄弱知识点分布（共 {kp_total} 题）", font=dict(size=14)),
                            showlegend=True,
                            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=300,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        fig_kp.add_annotation(
                            text=f"<b>{kp_total}</b><br>错题",
                            x=0.5, y=0.5,
                            font=dict(size=14, color="#333"),
                            showarrow=False,
                        )
                        st.plotly_chart(fig_kp, width="stretch", key=f"donut_kp_{selected_subject}")

                        st.markdown(f"**{selected_subject} 薄弱知识点排行：**")
                        for rank, (kp_name, kp_count) in enumerate(kp_stats.items(), 1):
                            pct = kp_count / kp_total * 100
                            bar_fill = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                            pct_str = f"{pct:.0f}%"
                            st.markdown(f"{rank}. **{kp_name}** — {kp_count} 题 ({pct_str})  `{bar_fill}`")
                    else:
                        st.info(f"该学科暂无错题数据。")

    with st.expander("📕 错题本", expanded=False):
        grouped = get_wrong_questions_grouped_by_subject_and_kp()
        if not grouped:
            st.caption("暂无错题记录。")
        else:
            total_wrong = sum(len(qs) for kps in grouped.values() for qs in kps.values())
            st.markdown(f"**共 {total_wrong} 道错题**")
            st.divider()

            for subj, kp_map in grouped.items():
                subj_count = sum(len(qs) for qs in kp_map.values())
                with st.expander(f"📚 {subj}（{subj_count}题）", expanded=False):
                    for kp, qs in kp_map.items():
                        with st.expander(f"📌 {kp}（{len(qs)}题）", expanded=False):
                            for q_idx, q in enumerate(qs, 1):
                                st.markdown(f"**{q_idx}. {q['question']}**")
                                for opt_idx, opt in enumerate(q.get("options", [])):
                                    prefix = "✅" if opt_idx == q.get("answer_index") else "○"
                                    st.markdown(f"{prefix} {opt}")
                                if q.get("explanation"):
                                    st.markdown(f"*解析：{q['explanation']}*")
                                st.markdown(f"*年级：{q.get('grade', '未分类')}*")
                                st.markdown(f"*做错时间：{q['created_at']}*")
                                st.divider()
