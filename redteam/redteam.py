import json
import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent / "guardrail"))
from layer1_regex import detect_pii_regex
from layer2_ner import detect_pii_ner
from layer3_llm import detect_pii_llm


# ── warmup ───────────────────────────────────────────────────────
def warmup_layers(layers: list):
    print("레이어 워밍업 중 (모델 가중치 로딩)...")
    dummy_text = "테스트 문장입니다."
    if 1 in layers:
        detect_pii_regex(dummy_text)
    if 2 in layers:
        detect_pii_ner(dummy_text)
    print("   완료\n")


# ── run the guardrail with latency measurement ──────────────────────────────────
def run_guardrail_with_latency(text: str, layers: list) -> dict:
    results = {}
    latency_ms = {}

    if 1 in layers:
        t0 = time.perf_counter()
        layer1 = detect_pii_regex(text)
        latency_ms["layer1"] = round((time.perf_counter() - t0) * 1000, 3)
        results["layer1"] = layer1
        if layer1["detected"]:
            return _result(True, "layer1", results, latency_ms)

    if 2 in layers:
        t0 = time.perf_counter()
        layer2 = detect_pii_ner(text)
        latency_ms["layer2"] = round((time.perf_counter() - t0) * 1000, 3)
        results["layer2"] = layer2
        if layer2["detected"]:
            return _result(True, "layer2", results, latency_ms)

    if 3 in layers:
        t0 = time.perf_counter()
        layer3 = detect_pii_llm(text)
        latency_ms["layer3"] = round((time.perf_counter() - t0) * 1000, 3)
        results["layer3"] = layer3
        if layer3["detected"]:
            return _result(True, "layer3", results, latency_ms)

    return _result(False, None, results, latency_ms)


def _result(blocked, blocked_by, results, latency_ms):
    return {
        "blocked": blocked,
        "blocked_by": blocked_by,
        "results": results,
        "latency_ms": latency_ms,
        "total_latency_ms": round(sum(latency_ms.values()), 3),
    }


def _avg(values):
    return round(sum(values) / len(values), 3) if values else 0.0


# ── measure ASR (attack samples) ─────────────────────────────────────────
def run_attack_evaluation(attack_samples: list, layers: list) -> dict:
    """공격 샘플을 가드레일에 투입해 ASR(공격 성공률) 측정"""
    results = []
    attack_type_stats = {}
    layer_latencies = {f"layer{i}": [] for i in layers}
    total_latencies = []

    for sample in tqdm(attack_samples, desc=f"ASR 측정 (layers={layers})"):
        text = sample["text"]
        attack_type = sample["attack_type"]

        guardrail_result = run_guardrail_with_latency(text, layers)
        attack_success = not guardrail_result["blocked"]
        lat = guardrail_result["latency_ms"]
        total_lat = guardrail_result["total_latency_ms"]

        for layer_key, ms in lat.items():
            if layer_key in layer_latencies:
                layer_latencies[layer_key].append(ms)
        total_latencies.append(total_lat)

        results.append({
            "text": text,
            "attack_type": attack_type,
            "pii_type": sample.get("pii_type", ""),
            "difficulty": sample.get("difficulty", ""),
            "subtype": sample.get("subtype", ""),
            "attack_success": attack_success,
            "blocked_by": guardrail_result.get("blocked_by"),
            "latency_ms": lat,
            "total_latency_ms": total_lat,
        })

        attack_type_stats.setdefault(attack_type, {"total": 0, "success": 0})
        attack_type_stats[attack_type]["total"] += 1
        if attack_success:
            attack_type_stats[attack_type]["success"] += 1

    total = len(results)
    total_success = sum(1 for r in results if r["attack_success"])
    overall_asr = round(total_success / total, 3) if total > 0 else 0

    for t, s in attack_type_stats.items():
        s["asr"] = round(s["success"] / s["total"], 3) if s["total"] > 0 else 0

    latency_summary = {
        layer_key: {
            "avg_ms": _avg(ms_list),
            "min_ms": round(min(ms_list), 3) if ms_list else 0,
            "max_ms": round(max(ms_list), 3) if ms_list else 0,
            "sample_count": len(ms_list),
        }
        for layer_key, ms_list in layer_latencies.items() if ms_list
    }
    latency_summary["total"] = {
        "avg_ms": _avg(total_latencies),
        "min_ms": round(min(total_latencies), 3) if total_latencies else 0,
        "max_ms": round(max(total_latencies), 3) if total_latencies else 0,
    }

    return {
        "total": total,
        "total_success": total_success,
        "overall_asr": overall_asr,
        "attack_type_stats": attack_type_stats,
        "latency_summary": latency_summary,
        "results": results,
    }


# ── measure FPR (benign samples) ─────────────────────────────────────────
def run_fpr_evaluation(normal_samples: list, layers: list) -> dict:
    """정상 샘플을 가드레일에 투입해 FPR(오탐지율) 측정"""
    results = []
    layer_fp_counts = {f"layer{i}": 0 for i in layers}

    for sample in tqdm(normal_samples, desc=f"FPR 측정 (layers={layers})"):
        text = sample["text"]
        guardrail_result = run_guardrail_with_latency(text, layers)
        false_positive = guardrail_result["blocked"]

        if false_positive:
            blocked_by = guardrail_result["blocked_by"]
            if blocked_by in layer_fp_counts:
                layer_fp_counts[blocked_by] += 1

        results.append({
            "text": text,
            "normal_type": sample.get("normal_type", ""),
            "false_positive": false_positive,
            "blocked_by": guardrail_result.get("blocked_by"),
        })

    total = len(results)
    total_fp = sum(1 for r in results if r["false_positive"])
    overall_fpr = round(total_fp / total, 3) if total > 0 else 0

    return {
        "total": total,
        "total_false_positive": total_fp,
        "overall_fpr": overall_fpr,
        "fp_by_layer": layer_fp_counts,
        "results": results,
    }


# ── combined run ────────────────────────────────────────────────────
def run_full_evaluation(dataset_path, output_path, layers=[1, 2, 3], round_num=1):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    attack_samples = [s for s in dataset if s.get("label", 1) == 1]
    normal_samples = [s for s in dataset if s.get("label", 1) == 0]

    print(f"공격 샘플 {len(attack_samples)}개 / 정상 샘플 {len(normal_samples)}개\n")

    warmup_layers(layers)

    print("\n[1/2] ASR 측정 (공격 샘플)")
    asr_report = run_attack_evaluation(attack_samples, layers)

    print("\n[2/2] FPR 측정 (정상 샘플)")
    fpr_report = run_fpr_evaluation(normal_samples, layers)

    report = {
        "round": round_num,
        "layers_used": layers,
        "asr_report": asr_report,
        "fpr_report": fpr_report,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def print_report(report):
    asr = report["asr_report"]
    fpr = report["fpr_report"]

    print("\n" + "=" * 65)
    print(f"레드팀 결과 Round {report['round']} (레이어: {report['layers_used']})")
    print("=" * 65)
    print(f"[ASR] 총 공격 수: {asr['total']} / 성공: {asr['total_success']} / 전체 ASR: {asr['overall_asr']}")
    print()
    print("유형별 ASR:")
    for attack_type, stats in asr["attack_type_stats"].items():
        print(f"  {attack_type:<30} ASR {stats['asr']}  ({stats['success']}/{stats['total']})")
    print()
    print(f"[FPR] 총 정상 샘플: {fpr['total']} / 오탐지: {fpr['total_false_positive']} / 전체 FPR: {fpr['overall_fpr']}")
    print("레이어별 False Positive:")
    for layer_key, count in fpr["fp_by_layer"].items():
        print(f"  {layer_key}: {count}건")
    print()
    print("레이어별 평균 Latency (공격 샘플 기준):")
    for layer_key, lat in asr["latency_summary"].items():
        if layer_key == "total":
            print(f"  {'합계(total)':<12} avg {lat['avg_ms']:>8.3f} ms")
        else:
            print(f"  {layer_key:<12} avg {lat['avg_ms']:>8.3f} ms  n={lat['sample_count']}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="레드팀 ASR + FPR + Latency 통합 측정")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    dataset_path = args.dataset or str(Path(__file__).parent.parent / "data" / "mixed_dataset.json")
    output_path = args.output or str(Path(__file__).parent.parent / "experiments" / f"redteam_round{args.round}.json")

    print(f"레드팀 통합 평가 시작 (Round {args.round}, 레이어: {args.layers})")
    print(f"   데이터셋: {dataset_path}\n")

    report = run_full_evaluation(dataset_path, output_path, args.layers, args.round)
    print_report(report)
    print(f"\n 저장: {output_path}")
