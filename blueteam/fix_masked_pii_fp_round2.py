import os
import re
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys.path.append(str(ROOT / "blueteam"))
import rule_store  # noqa: E402
from fix_masked_pii_fp import sync_layer3_llm_source  # noqa: E402

MODEL = "gpt-4.1-mini"
RULES_PATH = ROOT / "guardrail" / "layer3_rules_v4.json"
LAYER3_LLM_PATH = ROOT / "guardrail" / "layer3_llm.py"
AFTERFIX_RESULT_PATH = ROOT / "experiments" / "redteam_hardneg_v3_afterfix.json"

# 남은 실패 샘플을 형식별로 대략 분류 (프롬프트 근거용 분류일 뿐, 판정 로직과 무관)
PATTERN_GROUPS = {
    "주민등록번호": re.compile(r"\d{6}-\d\*+"),
    "카드번호": re.compile(r"\d{4}-\*+-\*+-\d{4}"),
}


def extract_remaining_fp_samples(result_path: Path) -> dict:
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data["fpr_report"]["results"]
    fp_texts = [r["text"] for r in results
                if r.get("normal_type") == "masked_pii" and r.get("false_positive")]
    grouped = defaultdict(list)
    for t in fp_texts:
        matched = False
        for name, pattern in PATTERN_GROUPS.items():
            if pattern.search(t):
                grouped[name].append(t)
                matched = True
                break
        if not matched:
            grouped["기타"].append(t)
    return grouped


def propose_exception_for_group(group_name: str, samples: list) -> str:
    samples_text = "\n".join(f'- "{s}"' for s in samples[:5])
    meta_prompt = f"""당신은 금융권 AI 보안 시스템의 프롬프트 엔지니어입니다.
아래는 현재 가드레일이 "위험(DETECTED)"으로 잘못 차단한 정상 문장들입니다(오탐).
공통적으로 '{group_name}'의 일부가 이미 마스킹(*)되어 있는 값을 보고하는 문장입니다.

[오탐된 정상 문장들]
{samples_text}

이 유형(마스킹된 {group_name})을 대표하는 새 SAFE 예시 하나를 제안하세요.
1. 실제 값이 아닌 예시용 값을 사용하고, 어떤 자릿수가 마스킹된 형태인지 명확히 보이게 하세요.
2. 값을 다시 알려달라거나 마스킹을 풀어달라는 요청에는 해당하지 않는다는 것을 이유에 명시하세요.
3. 다음 형식으로 한 줄만 출력하세요, 다른 설명은 포함하지 마세요:
"<예시 문장>" (<안전한 이유, 20자 내외 - "{group_name} 마스킹"과 "재요청 아님" 포함>)"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": meta_prompt}],
                max_tokens=200, temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"제안 생성 오류 (시도 {attempt+1}/3): {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rules-path", type=str, default=str(RULES_PATH))
    parser.add_argument("--layer3-llm-path", type=str, default=str(LAYER3_LLM_PATH))
    parser.add_argument("--afterfix-result", type=str, default=str(AFTERFIX_RESULT_PATH))
    args = parser.parse_args()

    rules_path = Path(args.rules_path)
    layer3_llm_path = Path(args.layer3_llm_path)
    afterfix_result_path = Path(args.afterfix_result)

    print("=" * 60)
    print("masked_pii 오탐 완화 패치 2라운드 (주민번호/카드번호 타겟)")
    print("=" * 60)

    grouped = extract_remaining_fp_samples(afterfix_result_path)
    for name, samples in grouped.items():
        print(f"  {name}: {len(samples)}건")

    proposals = {}
    for name, samples in grouped.items():
        if not samples:
            continue
        print(f"\nGPT-4.1-mini에게 '{name}' 형식 SAFE 예시 요청 중...")
        proposal = propose_exception_for_group(name, samples)
        if proposal:
            proposals[name] = proposal
            print(f"  제안: {proposal}")
        else:
            print(f"'{name}'제안 실패")

    if not proposals:
        print("제안이 하나도 생성되지 않았습니다.")
        sys.exit(1)

    if args.dry_run:
        print("\n(--dry-run 모드: 저장하지 않았습니다)")
        return

    if not args.auto_approve:
        ans = input(f"\n위 {len(proposals)}개 예외를 저장소에 추가하시겠습니까? (y/n): ").strip().lower()
        if ans != "y":
            print("거부됨 - 종료")
            return

    store = rule_store.load_rules(rules_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_json = rules_path.with_suffix(f".backup_r2_{ts}.json")
    shutil.copy(rules_path, backup_json)
    print(f"백업: {backup_json}")
    backup_py = layer3_llm_path.with_suffix(f".backup_r2_{ts}.py")
    shutil.copy(layer3_llm_path, backup_py)
    print(f"백업: {backup_py}")

    for name, proposal in proposals.items():
        new_id = rule_store.add_safe_exception(store, proposal, round_num=2, triggered_by_fp_type=f"masked_pii:{name}")
        print(f"추가됨 (id={new_id}, {name})")

    rule_store.save_rules(store, rules_path)
    print(f"저장 완료: {rules_path}")

    rendered = rule_store.render_system_prompt(store)
    sync_layer3_llm_source(rendered, layer3_llm_path)
    print(f"layer3_llm.py의 SYSTEM_PROMPT 동기화 완료: {layer3_llm_path}")

    print("\n" + "=" * 60)
    print("다시 재검증하세요:")
    print("=" * 60)
    print("1. python redteam/redteam.py --layers 1 2 3 "
          "--dataset data/mixed_dataset_v3_hardneg.json "
          "--output experiments/redteam_hardneg_v3_afterfix2.json")
    print("2. python experiments/layer_funnel_analysis.py experiments/redteam_hardneg_v3_afterfix2.json")
    print("3. python redteam/redteam.py --layers 1 2 3 "
          "--dataset data/mixed_dataset_v3.json "
          "--output experiments/redteam_v3_afterfix2_asr_check.json")


if __name__ == "__main__":
    main()
