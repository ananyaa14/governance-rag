import json
import random
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PAGES = json.loads(
    (ROOT / "data" / "processed" / "pages_clean.json").read_text(encoding="utf-8")
)
OUT = ROOT / "eval" / "questions_synth.json"

PER_SOURCE = {
    "eu_ai_act.pdf": 12,
    "nist_ai_rmf.pdf": 8,
    "nist_genai_profile.pdf": 8,
    "oecd_ai_principles.pdf": 4,
}
MIN_WORDS = 150
MODEL = "claude-sonnet-5-5"

PROMPT = """Below is one page from an AI governance document.

Write ONE question that a compliance officer or engineer might ask, which this page answers.
Rules:
- Paraphrase. Do NOT copy distinctive phrases or section numbers from the page into the question.
- The question must make sense on its own. Never write "this page", "the passage" or "the document above".
- The reference answer must be 1-3 sentences and use only facts stated on the page.
- If the page has no substantive content (contents list, references, blank layout), set "skip" to true.

Return ONLY JSON: {"skip": false, "question": "...", "reference_answer": "..."}

PAGE:
"""

client = anthropic.Anthropic()
random.seed(42)


def ask(page):
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": PROMPT + page["text"][:6000]}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {"skip": True}


questions = []
for source, quota in PER_SOURCE.items():
    pool = [p for p in PAGES if p["source"] == source and len(p["text"].split()) >= MIN_WORDS]
    random.shuffle(pool)
    made = 0
    for page in pool:
        if made >= quota:
            break
        try:
            item = ask(page)
        except Exception as e:
            print(f"  skipped p{page['page']} ({e})")
            continue
        if item.get("skip") or not item.get("question"):
            continue
        made += 1
        questions.append(
            {
                "id": f"s{len(questions) + 1:02d}",
                "question": item["question"],
                "type": "answerable",
                "source": source,
                "expected_pages": [page["page"]],
                "reference_answer": item["reference_answer"],
            }
        )
    print(f"{source}: {made} questions")

OUT.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nWrote {len(questions)} questions to {OUT}\n")
for q in questions:
    print(f"{q['id']} [{q['source']} p{q['expected_pages'][0]}] {q['question']}")