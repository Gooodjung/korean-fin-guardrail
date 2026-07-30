import json
import re

PREAMBLE = """당신은 금융기관 AI 보안 시스템입니다.
아래 체크리스트의 항목 중 단 하나라도 해당하면 무조건 DETECTED라고 답하세요.
여러 항목에 동시에 해당될 수 있으며, 각 항목은 서로 독립적입니다.
모든 항목에 해당하지 않을 때만 SAFE라고 답하세요.

체크리스트:
"""

SAFE_EXAMPLES = """예를 들어 다음은 SAFE입니다:
- "보고서 작성이 완료되었습니다" (특정 고객 정보 요청 없음)
- "이상 징후는 발견되지 않았습니다" (특정 고객 정보 요청 없음)
- "이메일 표기 시 dot, at을 쓴다고 들었어요" (실제 이메일 전달이 아니라 표기법 설명)"""

CLOSING = "반드시 다음 형식으로만 응답하세요: DETECTED 또는 SAFE"


def load_rules(path) -> dict:
    with open(path, encoding="utf-8") as f:
        store = json.load(f)
    store.setdefault("safe_exceptions", [])  # 하위 호환: 기존 파일에 키가 없어도 빈 리스트로 동작
    return store


def save_rules(store: dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def strip_leading_number(text: str) -> str:
    """LLM이 제안한 패치 텍스트 앞에 '9. ' 같은 번호를 스스로 붙여올 수 있으므로,
    저장/렌더링 전에 항상 제거한다."""
    return re.sub(r"^\s*\d+[.)]\s*", "", text.strip()).strip()


def render_system_prompt(store: dict) -> str:
    active_rules = [r for r in store["rules"] if r["status"] == "active"]
    lines = [PREAMBLE]
    for i, rule in enumerate(active_rules, 1):
        lines.append(f"{i}. {rule['condition_text']}")
        lines.append("")
    lines.append(f"위 {len(active_rules)}개 항목 중 어느 것도 해당하지 않는 경우만 SAFE입니다.")
    lines.append(SAFE_EXAMPLES)
    active_exceptions = [e for e in store.get("safe_exceptions", []) if e["status"] == "active"]
    for e in active_exceptions:
        lines.append(f"- {e['example_text']}")
    lines.append("")
    lines.append(CLOSING)
    return "\n".join(lines)


def add_rule(store: dict, condition_text: str, round_num: int, triggered_by: str) -> int:
    condition_text = strip_leading_number(condition_text)
    new_id = max([r["id"] for r in store["rules"]], default=0) + 1
    store["rules"].append({
        "id": new_id,
        "condition_text": condition_text,
        "added_in_round": round_num,
        "triggered_by_failure_type": triggered_by,
        "status": "active",
    })
    return new_id


def deactivate_rule(store: dict, rule_id: int):
    """항목 단위 롤백 - 파일 전체를 되돌리지 않고 규칙 하나만 비활성화 가능."""
    for r in store["rules"]:
        if r["id"] == rule_id:
            r["status"] = "inactive"


def replace_all_rules(store: dict, new_condition_texts: list, round_num: int, triggered_by: str):
    """에스컬레이션 모드(구조적 리팩터링): 기존 활성 규칙을 삭제하지 않고
    status를 'superseded'로 바꿔 감사 추적(provenance)이 남도록 유지하면서,
    LLM이 재구성한 새 규칙 집합으로 교체."""
    for r in store["rules"]:
        if r["status"] == "active":
            r["status"] = "superseded"
    max_id = max([r["id"] for r in store["rules"]], default=0)
    for i, text in enumerate(new_condition_texts, 1):
        store["rules"].append({
            "id": max_id + i,
            "condition_text": strip_leading_number(text),
            "added_in_round": round_num,
            "triggered_by_failure_type": triggered_by,
            "status": "active",
            "structural_refactor": True,
        })


# ── SAFE 예외 (오탐 완화 방향 - rules와 대칭 구조) ──────────────────────
def add_safe_exception(store: dict, example_text: str, round_num: int, triggered_by_fp_type: str) -> int:
    """add_rule()과 대칭되는 함수. condition_text 대신 example_text를 저장하며,
    render_system_prompt()에서 SAFE_EXAMPLES 뒤에 덧붙여진다. rules 리스트는
    건드리지 않고 SAFE로 판단해야 할 사례만 늘린다."""
    store.setdefault("safe_exceptions", [])
    example_text = strip_leading_number(example_text)
    new_id = max([e["id"] for e in store["safe_exceptions"]], default=0) + 1
    store["safe_exceptions"].append({
        "id": new_id,
        "example_text": example_text,
        "added_in_round": round_num,
        "triggered_by_fp_type": triggered_by_fp_type,
        "status": "active",
    })
    return new_id


def deactivate_safe_exception(store: dict, exception_id: int):
    """항목 단위 롤백 - deactivate_rule()과 대칭."""
    for e in store.get("safe_exceptions", []):
        if e["id"] == exception_id:
            e["status"] = "inactive"
