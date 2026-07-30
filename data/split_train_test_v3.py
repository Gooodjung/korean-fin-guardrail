import json
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "mixed_dataset_v3.json"
TRAIN_OUT = ROOT / "data" / "mixed_dataset_v3_train.json"
TEST_OUT = ROOT / "data" / "mixed_dataset_v3_test.json"

TEST_RATIO = 0.30
SEED = 42


def split(dataset: list, test_ratio: float, seed: int):
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for sample in dataset:
        by_type[sample["attack_type"]].append(sample)

    train, test = [], []
    split_report = {}
    dup_groups_moved = 0
    for attack_type, samples in sorted(by_type.items()):
        # 텍스트 단위로 그룹핑 (완전 동일 문장이 여러 레코드로 중복 존재하는
        # 경우, 그룹 전체가 항상 같은 쪽에만 속하도록 하기 위함)
        groups = defaultdict(list)
        for s in samples:
            groups[s["text"]].append(s)
        unique_texts = list(groups.keys())
        rng.shuffle(unique_texts)

        n_test_records = round(len(samples) * test_ratio)
        test_part, train_part = [], []
        for text in unique_texts:
            group = groups[text]
            if len(group) > 1:
                dup_groups_moved += 1
            target = test_part if len(test_part) < n_test_records else train_part
            target.extend(group)

        test.extend(test_part)
        train.extend(train_part)
        split_report[attack_type] = {"total": len(samples), "train": len(train_part), "test": len(test_part)}

    rng.shuffle(train)
    rng.shuffle(test)
    if dup_groups_moved:
        print(f"[안내] 중복 텍스트 그룹 {dup_groups_moved}개를 그룹 단위로 한쪽에만 배치함")
    return train, test, split_report


def main():
    with open(SRC, encoding="utf-8") as f:
        dataset = json.load(f)

    train, test, split_report = split(dataset, TEST_RATIO, SEED)

    # 무결성 체크: 겹치는 샘플이 없어야 하고, 합쳐서 원본과 개수가 같아야 함
    train_texts = {s["text"] for s in train}
    test_texts = {s["text"] for s in test}
    overlap = train_texts & test_texts
    assert len(overlap) == 0, f"train/test 겹침 발견: {len(overlap)}건"
    assert len(train) + len(test) == len(dataset), "분리 후 총 개수 불일치"

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"원본: {len(dataset)}개 -> train {len(train)}개 / test {len(test)}개")
    print("=" * 60)
    for attack_type, counts in split_report.items():
        print(f"  {attack_type:25s} total={counts['total']:4d}  train={counts['train']:4d}  test={counts['test']:4d}")
    print("=" * 60)
    print(f"train 저장: {TRAIN_OUT}")
    print(f"test  저장: {TEST_OUT}")
    print("무결성 체크 통과: train/test 텍스트 중복 없음, 총합 일치")


if __name__ == "__main__":
    main()
