import json
import sys
from pathlib import Path
from collections import defaultdict


def analyze_attack_funnel(asr_report: dict) -> dict:
    """공격 유형별로 레이어 퍼널(도달/차단/통과) 계산.
    주의: blocked_by가 기록된 실행(--layers 1 2 3 전체)에서만 의미가 있다.
    Layer1만 돌린 리포트를 넣으면 layer2/3는 애초에 도달 자체가 없다."""
    layers_used = None
    by_type = defaultdict(lambda: {"total": 0, "reached": defaultdict(int), "caught": defaultdict(int), "escaped": 0})
    overall = {"total": 0, "reached": defaultdict(int), "caught": defaultdict(int), "escaped": 0}

    for r in asr_report["results"]:
        attack_type = r["attack_type"]
        blocked_by = r.get("blocked_by")
        success = r["attack_success"]

        stat = by_type[attack_type]
        stat["total"] += 1
        overall["total"] += 1

        # Cascade structure: reached in order layer1 -> layer2 -> layer3.
        # If blocked_by is layerN, layer1..layerN were all "reached" and layerN "blocked" it.
        # If blocked_by is None (attack succeeded), every existing layer was
        # reached but none of them blocked it.
        layer_order = ["layer1", "layer2", "layer3"]
        stopped_at = layer_order.index(blocked_by) if blocked_by in layer_order else None

        reach_upto = stopped_at if stopped_at is not None else len(layer_order) - 1
        for i, lname in enumerate(layer_order):
            # If latency_ms has a value, that layer was actually run (= reached)
            if lname in r.get("latency_ms", {}):
                stat["reached"][lname] += 1
                overall["reached"][lname] += 1

        if blocked_by in layer_order:
            stat["caught"][blocked_by] += 1
            overall["caught"][blocked_by] += 1
        if success:
            stat["escaped"] += 1
            overall["escaped"] += 1

    return {"by_type": by_type, "overall": overall}


def analyze_fp_distribution(fpr_report: dict) -> dict:
    by_type = defaultdict(lambda: {"total": 0, "fp_by_layer": defaultdict(int)})
    overall = {"total": fpr_report["total"], "fp_by_layer": defaultdict(int)}

    for r in fpr_report["results"]:
        ntype = r.get("normal_type", "unknown") or "unknown"
        stat = by_type[ntype]
        stat["total"] += 1
        if r["false_positive"]:
            blocked_by = r.get("blocked_by")
            stat["fp_by_layer"][blocked_by] += 1
            overall["fp_by_layer"][blocked_by] += 1

    return {"by_type": by_type, "overall": overall}


def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def print_attack_funnel(funnel: dict):
    print("=" * 78)
    print("공격 유형별 레이어 퍼널 (도달 수 대비 그 레이어의 차단 기여, 최종 통과율)")
    print("=" * 78)
    header = f"{'유형':<24}{'총':>6}{'L1도달':>8}{'L1차단':>8}{'L2도달':>8}{'L2차단':>8}{'L3도달':>8}{'L3차단':>8}{'최종통과(FN)':>12}"
    print(header)
    print("-" * 78)

    rows = list(funnel["by_type"].items()) + [("전체", funnel["overall"])]
    for name, stat in rows:
        total = stat["total"]
        r1, c1 = stat["reached"].get("layer1", 0), stat["caught"].get("layer1", 0)
        r2, c2 = stat["reached"].get("layer2", 0), stat["caught"].get("layer2", 0)
        r3, c3 = stat["reached"].get("layer3", 0), stat["caught"].get("layer3", 0)
        esc = stat["escaped"]
        print(f"{name:<24}{total:>6}{r1:>8}{c1:>8}{r2:>8}{c2:>8}{r3:>8}{c3:>8}"
              f"{esc:>7}({pct(esc,total):>5.1f}%)")
    print()
    print("해석: 'L2도달'은 layer1을 통과해 layer2까지 넘어온 공격 수(=layer1의 miss).")
    print("      'L2도달' 대비 'L2차단' 비율이 낮을수록 layer2가 그 유형에 취약하다는 뜻.")
    print("      '최종통과(FN)'은 3개 레이어를 전부 통과한, 파이프라인 전체의 실패 사례.")
    print("=" * 78)


def print_fp_distribution(fp_analysis: dict):
    print("\n" + "=" * 60)
    print("정상 샘플 레이어별 오탐지(FP) 분포")
    print("=" * 60)
    header = f"{'정상 유형':<24}{'총':>6}{'L1 FP':>8}{'L2 FP':>8}{'L3 FP':>8}{'FPR':>8}"
    print(header)
    print("-" * 60)

    rows = list(fp_analysis["by_type"].items()) + [("전체", fp_analysis["overall"])]
    for name, stat in rows:
        total = stat["total"]
        fpl = stat["fp_by_layer"]
        l1, l2, l3 = fpl.get("layer1", 0), fpl.get("layer2", 0), fpl.get("layer3", 0)
        total_fp = l1 + l2 + l3
        print(f"{name:<24}{total:>6}{l1:>8}{l2:>8}{l3:>8}{pct(total_fp,total):>7.1f}%")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("사용법: python layer_funnel_analysis.py <redteam_결과.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    layers_used = report.get("layers_used", [])
    if layers_used != [1, 2, 3]:
        print(f"이 리포트는 layers={layers_used}로 실행된 결과입니다.")
        print("   레이어별 병목 분석은 --layers 1 2 3 전체 실행 결과에서 가장 의미가 큽니다")
        print("   (일부 레이어만 실행하면 그 레이어의 '도달' 수만 존재).\n")

    attack_funnel = analyze_attack_funnel(report["asr_report"])
    fp_analysis = analyze_fp_distribution(report["fpr_report"])

    print_attack_funnel(attack_funnel)
    print_fp_distribution(fp_analysis)

    out_path = path.parent / f"{path.stem}_funnel_analysis.json"

    def _default(o):
        if isinstance(o, defaultdict):
            return dict(o)
        raise TypeError

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"attack_funnel": attack_funnel, "fp_distribution": fp_analysis}, f,
                   ensure_ascii=False, indent=2, default=_default)
    print(f"\n 저장: {out_path}")


if __name__ == "__main__":
    main()
