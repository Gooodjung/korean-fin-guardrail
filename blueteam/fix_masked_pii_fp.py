import os
import re
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys.path.append(str(ROOT / "blueteam"))
import rule_store  # noqa: E402

MODEL = "gpt-4.1-mini"
RULES_PATH = ROOT / "guardrail" / "layer3_rules_v4.json"
LAYER3_LLM_PATH = ROOT / "guardrail" / "layer3_llm.py"
HARDNEG_RESULT_PATH = ROOT / "experiments" / "redteam_hardneg_v3.json"
FP_TYPE = "masked_pii"
N_EXAMPLES = 5


def sync_layer3_llm_source(rendered_prompt: str, layer3_llm_path: Path):
    """[중요] guardrail/guardrail.py와 redteam/redteam.py는 rule_store.json을
    런타임에 렌더링해서 쓰지 않고, layer3_llm.py에 하드코딩된 SYSTEM_PROMPT
    상수를 그대로 import해서 쓴다(auto_tuning_v4.py의 apply_rules_to_layer3()만
    실행 중에 메모리상으로 갈아끼운다 - 파일 자체는 안 건드림). 그래서
    guardrail/layer3_rules_v4.json에 safe_exceptions를 추가하는 것만으로는
    실제 redteam.py 재검증에 반영되지 않는다 - 렌더링된 최종 프롬프트를
    layer3_llm.py의 SYSTEM_PROMPT 상수에도 그대로 동기화해야 한다(9항목
    체크리스트가 처음 이 파일에 반영됐을 때와 동일한 방식)."""
    src = layer3_llm_path.read_text(encoding="utf-8")
    pattern = re.compile(r'SYSTEM_PROMPT = """.*?"""', re.S)
    if not pattern.search(src):
        raise RuntimeError(f"{layer3_llm_path}에서 SYSTEM_PROMPT 블록을 찾지 못했습니다 - 수동으로 동기화하세요.")
    escaped = rendered_prompt.replace("\\", "\\\\")
    new_src = pattern.sub(lambda m: 'SYSTEM_PROMPT = """' + escaped + '"""', src, count=1)
    layer3_llm_path.write_text(new_src, encoding="utf-8")


def extract_fp_samples(result_path: Path, fp_type: str, n: int) -> list:
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data["fpr_report"]["results"]
    matched = [r["text"] for r in results
               if r.get("normal_type") == fp_type and r.get("false_positive")]
    if not matched:
        raise RuntimeError(f"{result_path}에서 normal_type={fp_type} 오탐 샘플을 찾지 못했습니다.")
    return matched[:n]


def propose_safe_exception(fp_type: str, samples: list) -> str:
    samples_text = "\n".join(f'- "{s}"' for s in samples)
    meta_prompt = f"""당신은 금융권 AI 보안 시스템의 프롬프트 엔지니어입니다.
아래는 현재 가드레일이 "위험(DETECTED)"으로 잘못 차단한 정상 문장들입니다(오탐).
이 문장들의 공통 유형은 '{fp_type}'(이미 마스킹·비식별 처리된 정보를 언급하는
완료형 보고 문장)입니다.

[오탐된 정상 문장들]
{samples_text}

이 문장들을 안전(SAFE)하다고 판단하도록 돕는, 일반화된 SAFE 예시 항목 하나를
제안하세요. 다음 원칙을 반드시 지키세요:
1. 개별 샘플을 그대로 복사하지 말고, "이미 마스킹된 값을 보고하는 문장은 안전하다"는
   일반 규칙을 대표하는 예시를 새로 하나 만드세요 (실제 값이 아닌 예시용 값 사용).
2. 값의 일부가 실제로 *, 마스킹 처리 등으로 가려져 있는 경우에만 해당하고, 값을
   다시 알려달라거나 마스킹을 풀어달라는 요청에는 해당하지 않는다는 것을 이유에
   명시하세요 (그런 요청은 여전히 DETECTED여야 합니다 - 이 예외가 그런 요청까지
   안전하다고 오판하게 만들면 안 됩니다).
3. 다음 형식으로 한 줄만 출력하세요, 다른 설명은 포함하지 마세요:
"<일반화된 예시 문장>" (<안전한 이유, 20자 내외 - 마스킹 언급과 "재요청 아님" 포함>)"""

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
    parser.add_argument("--hardneg-result", type=str, default=str(HARDNEG_RESULT_PATH))
    args = parser.parse_args()

    rules_path = Path(args.rules_path)
    layer3_llm_path = Path(args.layer3_llm_path)
    hardneg_result_path = Path(args.hardneg_result)

    print("=" * 60)
    print(f"{FP_TYPE} 오탐 완화 패치 제안")
    print("=" * 60)

    samples = extract_fp_samples(hardneg_result_path, FP_TYPE, N_EXAMPLES)
    print(f"근거로 사용할 실제 오탐 샘플 {len(samples)}개:")
    for s in samples:
        print(f"  - {s}")

    print("\nGPT-4.1-mini에게 일반화된 SAFE 예외 제안 요청 중...")
    proposal = propose_safe_exception(FP_TYPE, samples)
    if not proposal:
        print("제안 생성 실패")
        sys.exit(1)

    print("\n제안된 SAFE 예외:")
    print("-" * 60)
    print(proposal)
    print("-" * 60)

    if args.dry_run:
        print("\n(--dry-run 모드: 저장하지 않았습니다)")
        return

    if not args.auto_approve:
        ans = input("\n이 예외를 저장소에 추가하시겠습니까? (y/n): ").strip().lower()
        if ans != "y":
            print("거부됨 - 종료")
            return

    store = rule_store.load_rules(rules_path)

    # 저장 전 백업 (안전장치) - JSON과 layer3_llm.py 둘 다
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_json = rules_path.with_suffix(f".backup_{ts}.json")
    shutil.copy(rules_path, backup_json)
    print(f"백업: {backup_json}")
    backup_py = layer3_llm_path.with_suffix(f".backup_{ts}.py")
    shutil.copy(layer3_llm_path, backup_py)
    print(f"백업: {backup_py}")

    new_id = rule_store.add_safe_exception(store, proposal, round_num=1, triggered_by_fp_type=FP_TYPE)
    rule_store.save_rules(store, rules_path)
    print(f"저장 완료 (safe_exceptions id={new_id}): {rules_path}")

    # [중요] guardrail.py/redteam.py가 실제로 쓰는 layer3_llm.py의 SYSTEM_PROMPT
    # 상수도 함께 동기화해야 재검증에 반영된다 (위 sync_layer3_llm_source 참고).
    rendered = rule_store.render_system_prompt(store)
    sync_layer3_llm_source(rendered, layer3_llm_path)
    print(f"layer3_llm.py의 SYSTEM_PROMPT 동기화 완료: {layer3_llm_path}")

    print("\n" + "=" * 60)
    print("다음 단계 (반드시 실행) — 이 패치가 회귀를 유발하지 않았는지 확인:")
    print("=" * 60)
    print("1. python redteam/redteam.py --layers 1 2 3 "
          "--dataset data/mixed_dataset_v3_hardneg.json "
          "--output experiments/redteam_hardneg_v3_afterfix.json")
    print("   (masked_pii FPR이 낮아졌는지 확인)")
    print("2. python redteam/redteam.py --layers 1 2 3 "
          "--dataset data/mixed_dataset_v3.json "
          "--output experiments/redteam_v3_afterfix_asr_check.json")
    print("   (원본 공격 1,000개 ASR이 악화되지 않았는지 확인 - 가장 중요)")


if __name__ == "__main__":
    main()
