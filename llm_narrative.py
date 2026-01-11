import os
from typing import List
from openai import OpenAI
import os
print("OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an assistant generating explanatory text for an educational machine-learning demo.

Rules:
- Do NOT provide medical advice, diagnosis, or treatment.
- Do NOT suggest actions, interventions, or thresholds.
- Do NOT interpret results clinically.
- Only explain what the model used and its limitations.
- If unsure, say so explicitly.

If a response would violate these rules, respond with a neutral explanation of limitations and uncertainty instead.
"""

def build_prompt(
    risk: float,
    threshold: float,
    flag: str,
    top_up: List[str],
    top_down: List[str],
    imputation_notes: List[str],
) -> str:
    return f"""
Context:
This is an educational ML research prototype using a public diabetes dataset.

Model output:
- Predicted probability: {risk:.2f}
- Decision threshold (demo-only): {threshold:.2f}
- Flag status: {flag}

Top features increasing risk:
{', '.join(top_up) if top_up else 'None'}

Top features decreasing risk:
{', '.join(top_down) if top_down else 'None'}

Missing or imputed inputs:
{', '.join(imputation_notes) if imputation_notes else 'None'}

Task:
Write a short explanation with the following sections:

1. What the model used
2. Main drivers of the prediction
3. What the model cannot know
4. Questions this result raises (questions only, no advice)

Keep the tone neutral and non-clinical.
""".strip()

def generate_narrative(prompt: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
