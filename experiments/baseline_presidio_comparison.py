import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent


def build_analyzer(korean_nlp: bool):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    if korean_nlp:
        # 다국어 모델로 교체 시도 (완전한 한국어 NER은 아님 - 한계로 명시할 것)
        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "ko", "model_name": "xx_ent_wiki_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=config)
        nlp_engine = provider.create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["ko"])
    else:
        return AnalyzerEngine()  # 기본 영어(en) 파이프라인


def detect_with_presidio(analyzer, text: str, language: str) -> dict:
    """Presidio 분석 결과를 우리 가드레일과 동일한 {"detected": bool} 포맷으로 변환.
    엔티티가 1개라도 잡히면 '탐지(차단)'으로 간주 - 우리 가드레일의
    detect_pii_* 함수들과 동일한 판정 기준."""
    try:
        results = analyzer.analyze(text=text, language=language)
    except Exception as e:
        # 언어 미지원 등으로 실패하면 "탐지 실패"로 처리 (보수적으로 미탐 처리하지 않고
        # 명시적으로 에러 표시 - 통계에서 구분 가능하도록)
        return {"detected": False, "entities": [], "error": str(e)}
    return {
        "detected": len(results) > 0,
        "entities": [{"type": r.entity_type, "score": round(r.score, 3)} for r in results],
    }


def run(dataset_path: Path, korean_nlp: bool, output_path: Path):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    attack_samples = [s for s in dataset if s.get("label", 1) == 1]
    normal_samples = [s for s in dataset if s.get("label", 1) == 0]
    language = "ko" if korean_nlp else "en"

    print(f"Presidio 분석 엔진 초기화 중 (korean_nlp={korean_nlp})...")
    analyzer = build_analyzer(korean_nlp)

    # ── ASR 측정 ──
    attack_results = []
    type_stats = {}
    error_count = 0
    for i, sample in enumerate(attack_samples):
        r = detect_with_presidio(analyzer, sample["text"], language)
        if "error" in r:
            error_count += 1
        success = not r["detected"]
        attack_type = sample["attack_type"]
        attack_results.append({
            "text": sample["text"], "attack_type": attack_type,
            "attack_success": success, "entities_found": r["entities"],
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
        r = detect_with_presidio(analyzer, sample["text"], language)
        fp = r["detected"]
        if fp:
            fp_count += 1
        normal_results.append({"text": sample["text"], "false_positive": fp, "entities_found": r["entities"]})
        if (i + 1) % 100 == 0:
            print(f"  FPR 측정 {i+1}/{len(normal_samples)}")

    overall_fpr = round(fp_count / len(normal_samples), 4) if normal_samples else 0

    report = {
        "tool": "presidio-analyzer",
        "korean_nlp_mode": korean_nlp,
        "language_used": language,
        "overall_asr": overall_asr,
        "overall_fpr": overall_fpr,
        "type_stats": type_stats,
        "nlp_error_count": error_count,
        "attack_results": attack_results,
        "normal_results": normal_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Presidio baseline 결과 (korean_nlp={korean_nlp})")
    print("=" * 60)
    print(f"전체 ASR: {overall_asr}  |  전체 FPR: {overall_fpr}")
    print("유형별 ASR:")
    for t, s in type_stats.items():
        print(f"  {t:<28} {s['asr']}  ({s['success']}/{s['total']})")
    if error_count:
        print(f"\n 분석 중 오류(언어 미지원 등) {error_count}건 - nlp_error_count 참고")
    print(f"\n 저장: {output_path}")
    print("\n※ 이 결과를 우리 가드레일의 redteam.py 결과(같은 mixed_dataset_v3.json 기준)와")
    print("   나란히 놓고 보고서 Related Work 절에 baseline 비교표로 반영할 것.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(ROOT / "data" / "mixed_dataset_v3.json"))
    parser.add_argument("--korean-nlp", action="store_true", help="다국어(xx_ent_wiki_sm) 모델로 재시도")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    suffix = "korean" if args.korean_nlp else "default"
    output_path = Path(args.output) if args.output else ROOT / "experiments" / f"baseline_presidio_{suffix}.json"

    run(dataset_path, args.korean_nlp, output_path)
