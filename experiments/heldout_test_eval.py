import sys
import json
import math
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "guardrail"))
sys.path.append(str(ROOT / "blueteam"))


def wilson(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    lo = (center - adj) / denom
    hi = (center + adj) / denom
    return p * 100, lo * 100, hi * 100


def evaluate(dataset_path, layer3_llm):
    from layer1_regex import detect_pii_regex
    from layer2_ner import detect_pii_ner

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    attacks = [s for s in dataset if s.get("label", 1) == 1]
    normals = [s for s in dataset if s.get("label", 1) == 0]

    attack_success = 0
    for s in attacks:
        text = s["text"]
        blocked = detect_pii_regex(text)["detected"]
        if not blocked:
            blocked = detect_pii_ner(text)["detected"]
        if not blocked:
            blocked = layer3_llm.detect_pii_llm(text)["detected"]
        if not blocked:
            attack_success += 1

    fp = 0
    for s in normals:
        text = s["text"]
        flagged = detect_pii_regex(text)["detected"]
        if not flagged:
            flagged = detect_pii_ner(text)["detected"]
        if not flagged:
            flagged = layer3_llm.detect_pii_llm(text)["detected"]
        if flagged:
            fp += 1

    asr_p, asr_lo, asr_hi = wilson(attack_success, len(attacks))
    fpr_p, fpr_lo, fpr_hi = wilson(fp, len(normals))
    return {
        "n_attacks": len(attacks),
        "n_normals": len(normals),
        "attack_success": attack_success,
        "fp": fp,
        "asr": {"point": round(asr_p, 2), "lo": round(asr_lo, 2), "hi": round(asr_hi, 2)},
        "fpr": {"point": round(fpr_p, 2), "lo": round(fpr_lo, 2), "hi": round(fpr_hi, 2)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-path", type=str,
                         default=str(ROOT / "guardrail" / "layer3_rules_v4_traintest.json"))
    parser.add_argument("--train-dataset", type=str, default=str(ROOT / "data" / "mixed_dataset_v3_train.json"))
    parser.add_argument("--test-dataset", type=str, default=str(ROOT / "data" / "mixed_dataset_v3_test.json"))
    parser.add_argument("--output", type=str,
                         default=str(ROOT / "experiments" / "heldout_test_eval_result.json"))
    args = parser.parse_args()

    import rule_store
    import layer3_llm
    store = rule_store.load_rules(args.rules_path)
    layer3_llm.SYSTEM_PROMPT = rule_store.render_system_prompt(store)
    n_active = len([r for r in store["rules"] if r["status"] == "active"])
    print(f"규칙 로드: {args.rules_path} (활성 규칙 {n_active}개)")

    print("\n[1/2] train 세트 평가 중 (튜닝에 실제로 쓰인 데이터)...")
    train_result = evaluate(args.train_dataset, layer3_llm)
    print(f"  ASR {train_result['asr']['point']}% [95% CI {train_result['asr']['lo']}-{train_result['asr']['hi']}%]"
          f"  FPR {train_result['fpr']['point']}% [95% CI {train_result['fpr']['lo']}-{train_result['fpr']['hi']}%]")

    print("\n[2/2] test 세트 평가 중 (튜닝 과정에서 한 번도 보지 못한 데이터)...")
    test_result = evaluate(args.test_dataset, layer3_llm)
    print(f"  ASR {test_result['asr']['point']}% [95% CI {test_result['asr']['lo']}-{test_result['asr']['hi']}%]"
          f"  FPR {test_result['fpr']['point']}% [95% CI {test_result['fpr']['lo']}-{test_result['fpr']['hi']}%]")

    asr_ci_overlap = not (train_result["asr"]["hi"] < test_result["asr"]["lo"] or
                           test_result["asr"]["hi"] < train_result["asr"]["lo"])
    fpr_ci_overlap = not (train_result["fpr"]["hi"] < test_result["fpr"]["lo"] or
                           test_result["fpr"]["hi"] < train_result["fpr"]["lo"])

    report = {
        "rules_path": args.rules_path,
        "n_active_rules": n_active,
        "train": train_result,
        "test": test_result,
        "asr_ci_overlap": asr_ci_overlap,
        "fpr_ci_overlap": fpr_ci_overlap,
        "interpretation": (
            "train/test 95% CI가 겹쳐 통계적으로 유의한 성능 차이를 확인할 수 없음 "
            "(과적합 근거 없음)" if (asr_ci_overlap and fpr_ci_overlap) else
            "train/test 95% CI가 겹치지 않는 지표가 있음 - 해당 지표에서 과적합 가능성을 배제할 수 없음"
        ),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("train vs test 비교")
    print("=" * 60)
    print(f"ASR  train {train_result['asr']['point']}%  vs  test {test_result['asr']['point']}%  "
          f"(CI 겹침: {asr_ci_overlap})")
    print(f"FPR  train {train_result['fpr']['point']}%  vs  test {test_result['fpr']['point']}%  "
          f"(CI 겹침: {fpr_ci_overlap})")
    print(f"\n해석: {report['interpretation']}")
    print(f"\n 저장: {args.output}")


if __name__ == "__main__":
    main()
