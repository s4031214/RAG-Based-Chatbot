import ollama
import regex as re

RUBRIC = """You are a strict evaluator. Given a QUESTION, CONTEXT (the only allowed source of truth), and an ANSWER:

Score two dimensions from 1 (poor) to 5 (excellent), integers only:
- FAITHFULNESS: Are the answer's claims supported by the CONTEXT? Penalize any unsupported claim, policy, date, number, or jurisdictional detail.
- COMPLETENESS: Does the answer cover the key points needed to address the QUESTION, given the CONTEXT? Penalize omissions.

Rules:
- Use ONLY the CONTEXT as ground truth. If the answer uses info not in CONTEXT, reduce FAITHFULNESS.
- If the answer honestly says it doesn’t know and points to relevant articles, FAITHFULNESS may be high (4–5), COMPLETENESS medium (2–3) depending on helpfulness.
- Return a single JSON object with keys: faithfulness, completeness, critique.
"""

JUDGE_PROMPT = """QUESTION:
{q}

CONTEXT:
{ctx}

ANSWER:
{ans}

Return JSON only, like:
{{"faithfulness": 4, "completeness": 3, "critique": "short reason"}}
"""

def judge(model: str, question: str, context: str, answer: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role":"system", "content": RUBRIC},
            {"role":"user", "content": JUDGE_PROMPT.format(q=question, ctx=context, ans=answer)}
        ],
        "options": {"temperature": 0.0, "num_ctx": 8192}
    }
    resp = ollama.chat(**payload)
    txt = resp["message"]["content"].strip()
    # naive JSON grab (model should comply)
    m = re.search(r'\{.*\}', txt, flags=re.S)
    if not m:
        return {"faithfulness": None, "completeness": None, "critique": "No JSON from judge."}
    try:
        import json
        data = json.loads(m.group(0))
        # sanitize types
        f = int(data.get("faithfulness")) if data.get("faithfulness") is not None else None
        c = int(data.get("completeness")) if data.get("completeness") is not None else None
        crit = str(data.get("critique") or "")
        return {"faithfulness": f, "completeness": c, "critique": crit[:500]}
    except Exception:
        return {"faithfulness": None, "completeness": None, "critique": "Bad JSON from judge."}
