"""随堂测验：从 JSON 题库判分。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BANK_PATH = Path(__file__).with_name("quiz_bank.json")


@dataclass
class QuizScore:
    total: int
    correct: int
    details: list[dict]

    @property
    def percent(self) -> float:
        return 0.0 if self.total == 0 else 100.0 * self.correct / self.total


def load_questions(path: Path | None = None) -> list[dict]:
    p = path or BANK_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data["questions"])


def grade_answers(answers: dict[str, str], questions: list[dict] | None = None) -> QuizScore:
    """answers 的 key 为题目 id，value 为选项字母（A/B/C/D）。"""
    questions = questions if questions is not None else load_questions()
    details = []
    correct = 0
    for q in questions:
        qid = q["id"]
        given = str(answers.get(qid, "")).strip().upper()
        expected = str(q["answer"]).strip().upper()
        ok = given == expected
        if ok:
            correct += 1
        details.append(
            {
                "id": qid,
                "ok": ok,
                "given": given,
                "answer": expected,
                "explanation": q.get("explanation", ""),
            }
        )
    return QuizScore(total=len(questions), correct=correct, details=details)
