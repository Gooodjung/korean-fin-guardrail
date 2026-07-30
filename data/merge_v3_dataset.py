import json
from pathlib import Path

ROOT = Path(__file__).parent


def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    lib_part = load_json(ROOT / "attack_dataset_v3_lib_partial.json")
    llm_part = load_json(ROOT / "attack_dataset_v3_llm_partial.json")

    if not llm_part:
        print("attack_dataset_v3_llm_partial.json이 없습니다.")
        print("   먼저 'python data/generate_dataset_v3_llm.py'를 실행하세요")
        print("   (contextual_leakage/roleplay_jailbreak/multiturn_accumulation).")
        print(f"   지금은 lib 파트({len(lib_part)}개)만 있는 상태로 부분 병합만 진행합니다.\n")

    # Reuse the existing benign samples (unrelated to the PII generation issue, so not something that needs redoing)
    mixed_original = load_json(ROOT / "mixed_dataset.json")
    normal_samples = [s for s in mixed_original if s.get("label", 1) == 0]
    if not normal_samples:
        print("기존 mixed_dataset.json에서 정상 샘플을 찾지 못했습니다. label=0 확인 필요.")

    all_samples = lib_part + llm_part + normal_samples

    output_path = ROOT / "mixed_dataset_v3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    type_counts = {}
    for s in all_samples:
        t = s.get("attack_type", "normal") if s.get("label", 1) == 1 else "normal"
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"병합 완료: 총 {len(all_samples)}개")
    for t, c in type_counts.items():
        print(f"  {t:<25} {c}개")
    print(f"\n 저장: {output_path}")

    if llm_part and len(all_samples) < 1000:
        print("\n 참고: 전체가 목표(1,200개 안팎)보다 적습니다. 각 생성 스크립트의")
        print("   n 값을 조정했는지, 일부 배치가 실패하진 않았는지 확인하세요.")


if __name__ == "__main__":
    main()
