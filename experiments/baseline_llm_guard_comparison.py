import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent


def build_scanners():
    from llm_guard.input_scanners import Anonymize, PromptInjection
    from llm_guard.vault import Vault

    vault = Vault()
    anonymize = Anonymize(vault)
    prompt_injection = PromptInjection()
    return anonymize, prompt_injection


def detect_with_llm_guard(anonymize, prompt_injection, text: str) -> dict:
    """LLM Guard의 두 스캐너 결과를 종합해 우리 가드레일과 동일한
    {"detected": bool} 포맷으로 변환. 둘 중 하나라도 위험 판정하면
    '탐지(차단)'으로 간주 - 우리 시스템이 여러 Layer 중 하나라도 걸리면
    차단하는 것과 동일한 판정 기준."""
    try:
        _, anon_is_valid, anon_risk = anonymize.scan(text)
    except Exception as e:
        anon_is_valid, anon_risk = True, -1.0  # 실패 시 "탐지 실패"로 보수적 처리하지 않고 표시
        anon_error = str(e)
    else:
        anon_error = None

    try:
        _, inj_is_valid, inj_risk = prompt_injection.scan(text)
    except Exception as e:
        inj_is_valid, inj_risk = True, -1.0
        inj_error = str(e)
    else:
        inj_error = None

    detected = (not anon_is_valid) or (not inj_is_valid)
    return {
        "detected": detected,
        "anonymize_flagged": not anon_is_valid,
        "anonymize_risk": anon_risk,
        "prompt_injection_flagged": not inj_is_valid,
        "prompt_injection_risk": inj_risk,
        "errors": [e for e in (anon_error, inj_error) if e],
    }


def run(dataset_path: Path, output_path: Path):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    attack_samples = [s for s in dataset if s.get("label", 1) == 1]
    normal_samples = [s for s in dataset if s.get("label", 1) == 0]

    print("LLM Guard 스캐너 초기화 중 (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있음)...")
    anonymize, prompt_injection = build_scanners()

    # ── ASR 측정 ──
    attack_results = []
    type_stats = {}
    error_count = 0
    for i, sample in enumerate(attack_samples):
        r = detect_with_llm_guard(anonymize, prompt_injection, sample["text"])
        if r["errors"]:
            error_count += 1
        success = not r["detected"]
        attack_type = sample["attack_type"]
        attack_results.append({
            "text": sample["text"], "attack_type": attack_type,
            "attack_success": success,
            "anonymize_flagged": r["anonymize_flagged"],
            "prompt_injection_flagged": r["prompt_injection_flagged"],
        })
        type_stats.setdefault(attack_type, {"total": 0, "success": 0})
        type_stats[attack_type]["total"] += 1
        if success:
            type_stats[attack_type]["success"] += 1
        if (i + 1) % 100 == 0:
            print(f"  ASR 측정 {i+1}/{len(attack_samples)}")

    for t, s in type_stats.items():
        s["asr"] = round(s["success"] / s["total"], 4) if s["total"] else 0

    total = len(attack_results)
    total_success = sum(1 for r in attack_results if r["attack_success"])
    overall_asr = round(total_success / total, 4) if total else 0

    # ── FPR 측정 ──
    fp_count = 0
    normal_results = []
    for i, sample in enumerate(normal_samples):
        r = detect_with_llm_guard(anonymize, prompt_injection, sample["text"])
        fp = r["detected"]
        if fp:
            fp_count += 1
        normal_results.append({
            "text": sample["text"], "false_positive": fp,
            "anonymize_flagged": r["anonymize_flagged"],
            "prompt_injection_flagged": r["prompt_injection_flagged"],
        })
        if (i + 1) % 100 == 0:
            print(f"  FPR 측정 {i+1}/{len(normal_samples)}")

    overall_fpr = round(fp_count / len(normal_samples), 4) if normal_samples else 0

    report = {
        "tool": "llm-guard",
        "scanners_used": ["Anonymize", "PromptInjection"],
        "overall_asr": overall_asr,
        "overall_fpr": overall_fpr,
        "type_stats": type_stats,
        "error_count": error_count,
        "attack_results": attack_results,
        "normal_results": normal_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("LLM Guard baseline 결과 (Anonymize + PromptInjection)")
    print("=" * 60)
    print(f"전체 ASR: {overall_asr}  |  전체 FPR: {overall_fpr}")
    print("유형별 ASR:")
    for t, s in type_stats.items():
        print(f"  {t:<28} {s['asr']}  ({s['success']}/{s['total']})")
    if error_count:
        print(f"\n 스캐너 실행 중 오류 {error_count}건 - attack_results의 errors 필드 참고")
    print(f"\n 저장: {output_path}")
    print("\n※ 이 결과를 baseline_presidio_default.json과 함께 6.6절 baseline 비교표에")
    print("   추가할 것. 두 도구 모두 한국어 비정형 PII 지원이 없다는 공통 한계도 명시.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(ROOT / "data" / "mixed_dataset_v3.json"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else ROOT / "experiments" / "baseline_llm_guard.json"

    run(dataset_path, output_path)
