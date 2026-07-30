import json
import time
import hashlib
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "guardrail"))
from layer1_regex import detect_pii_regex  # noqa: E402
from layer2_ner import detect_pii_ner  # noqa: E402
from layer3_llm import detect_pii_llm  # noqa: E402


def normalize_key(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class CachedLayer3:
    """detect_pii_llm()을 감싸는 캐시 래퍼. 원본 함수는 전혀 수정하지 않는다."""

    def __init__(self):
        self.cache: dict = {}
        self.hits = 0
        self.misses = 0
        self.hit_latency_ms: list = []
        self.miss_latency_ms: list = []

    def detect(self, text: str) -> dict:
        key = normalize_key(text)
        t0 = time.perf_counter()
        if key in self.cache:
            result = self.cache[key]
            elapsed = (time.perf_counter() - t0) * 1000
            self.hits += 1
            self.hit_latency_ms.append(elapsed)
            return result
        result = detect_pii_llm(text)
        elapsed = (time.perf_counter() - t0) * 1000
        self.cache[key] = result
        self.misses += 1
        self.miss_latency_ms.append(elapsed)
        return result


def run_pass(dataset: list, cached_layer3: CachedLayer3, use_cache: bool, desc: str) -> dict:
    """dataset 전체를 Layer1->Layer2->Layer3 순으로 평가. use_cache=False면
    캐시를 아예 참조하지 않고 매번 새로 호출(순수 cold 상태 재현)."""
    t_start = time.perf_counter()
    per_sample_latency_ms = []
    l3_call_count = 0

    for i, sample in enumerate(dataset):
        text = sample["text"]
        t0 = time.perf_counter()

        l1 = detect_pii_regex(text)
        blocked = l1["detected"]
        if not blocked:
            l2 = detect_pii_ner(text)
            blocked = l2["detected"]
        if not blocked:
            if use_cache:
                r3 = cached_layer3.detect(text)
            else:
                r3 = detect_pii_llm(text)
            blocked = r3["detected"]
            l3_call_count += 1

        per_sample_latency_ms.append((time.perf_counter() - t0) * 1000)

        if (i + 1) % 100 == 0:
            print(f"  [{desc}] {i+1}/{len(dataset)}건 처리")

    total_elapsed_sec = time.perf_counter() - t_start
    return {
        "desc": desc,
        "n_samples": len(dataset),
        "l3_call_count": l3_call_count,
        "total_elapsed_sec": round(total_elapsed_sec, 2),
        "avg_latency_ms_per_sample": round(sum(per_sample_latency_ms) / len(per_sample_latency_ms), 2) if per_sample_latency_ms else 0,
    }


def run(dataset_path: Path, output_path: Path, sample_n: int = None, seed: int = 42):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if sample_n:
        import random
        random.seed(seed)
        dataset = random.sample(dataset, min(sample_n, len(dataset)))

    print(f"데이터셋 {len(dataset)}건으로 캐싱/레이턴시 실험 시작\n")

    # ── Pass 1: Cold (no cache) ──
    print("=" * 60)
    print("Pass 1/2: Cold (캐시 미사용, 매번 Layer3 실호출)")
    print("=" * 60)
    cold_result = run_pass(dataset, cached_layer3=None, use_cache=False, desc="cold")

    # ── Pass 2: Warm (re-evaluate the same dataset, using cache) ──
    print("\n" + "=" * 60)
    print("Pass 2/2: Warm (동일 입력 반복 시나리오, 캐시 사용)")
    print("=" * 60)
    cached_layer3 = CachedLayer3()
    # Rather than pre-filling the cache with the cold-pass results, let this pass
    # naturally fill itself: the first occurrence is a miss, and a later
    # reoccurrence of the same text (duplicates within the dataset, or this
    # rerun) becomes a hit — so to demonstrate the effect of "repeated input"
    # using just this one pass, the dataset is concatenated with itself twice.
    warm_dataset = dataset + dataset
    warm_result = run_pass(warm_dataset, cached_layer3=cached_layer3, use_cache=True, desc="warm(2x)")

    speedup = None
    if warm_result["total_elapsed_sec"] > 0:
        # For a fair comparison, scale warm(2x)'s time down to a single pass before comparing to cold
        warm_per_full_pass_sec = warm_result["total_elapsed_sec"] / 2
        speedup = round(cold_result["total_elapsed_sec"] / warm_per_full_pass_sec, 2) if warm_per_full_pass_sec else None

    report = {
        "n_samples": len(dataset),
        "cold_pass": cold_result,
        "warm_pass_2x": warm_result,
        "cache_hits": cached_layer3.hits,
        "cache_misses": cached_layer3.misses,
        "cache_hit_rate": round(cached_layer3.hits / (cached_layer3.hits + cached_layer3.misses), 4) if (cached_layer3.hits + cached_layer3.misses) else 0,
        "avg_hit_latency_ms": round(sum(cached_layer3.hit_latency_ms) / len(cached_layer3.hit_latency_ms), 4) if cached_layer3.hit_latency_ms else None,
        "avg_miss_latency_ms": round(sum(cached_layer3.miss_latency_ms) / len(cached_layer3.miss_latency_ms), 2) if cached_layer3.miss_latency_ms else None,
        "estimated_speedup_on_full_repeat_traffic": speedup,
        "caveat": (
            "이 실험은 '완전히 동일한 문장이 반복 입력되는' 상한선 시나리오만 측정한다. "
            "실제 운영 트래픽에서 문장이 얼마나 반복되는지는 실제 로그가 있어야 알 수 있으며, "
            "본 프로젝트는 이를 측정할 프로덕션 데이터를 보유하고 있지 않다. "
            "따라서 이 결과는 '캐싱이 이론적으로 얼마나 효과적일 수 있는가'에 대한 것이며, "
            "'실제로 얼마나 절감되는가'에 대한 답은 아니다."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("캐싱/레이턴시 실험 결과")
    print("=" * 60)
    print(f"Cold pass: {cold_result['total_elapsed_sec']}초 (Layer3 호출 {cold_result['l3_call_count']}건, "
          f"평균 {cold_result['avg_latency_ms_per_sample']}ms/건)")
    print(f"Warm pass(2x): {warm_result['total_elapsed_sec']}초 (Layer3 호출 {warm_result['l3_call_count']}건, "
          f"평균 {warm_result['avg_latency_ms_per_sample']}ms/건)")
    print(f"캐시 히트율: {report['cache_hit_rate']*100:.1f}% ({cached_layer3.hits}/{cached_layer3.hits+cached_layer3.misses})")
    print(f"히트 시 평균 지연: {report['avg_hit_latency_ms']}ms  |  미스 시 평균 지연: {report['avg_miss_latency_ms']}ms")
    if speedup:
        print(f"완전 반복 트래픽 가정 시 추정 속도 향상: {speedup}배")
    print(f"\n 저장: {output_path}")
    print("\n※ caveat 필드에 명시된 대로, 이 결과는 이론적 상한선이며 실제 운영 반복률은 별도 검증 필요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(ROOT / "data" / "mixed_dataset_v3.json"))
    parser.add_argument("--sample", type=int, default=None, help="빠른 시연을 위한 표본 수 제한")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else ROOT / "experiments" / "caching_latency_test.json"

    run(dataset_path, output_path, sample_n=args.sample, seed=args.seed)
