from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

_MODEL_NAME = "Leo97/KoELECTRA-small-v3-modu-ner"
_ner_pipeline = None  # lazy loading


def _get_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(_MODEL_NAME)
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",  # 서브워드 토큰을 하나의 개체로 병합
        )
    return _ner_pipeline


# 탐지 대상으로 간주할 개체명 라벨 (모델 라벨 체계에 맞춰 조정 필요)
PII_RELATED_LABELS = {"PS", "PER", "LC", "LOC", "OG", "ORG"}


def detect_pii_ner(text: str) -> dict:
    """
    NER 모델로 텍스트에서 사람 이름/지명 등 개체명을 탐지

    Returns:
        {
            "detected": bool,
            "found": [{"type": "PER", "value": "홍길동", "score": 0.98}, ...]
        }
    """
    ner = _get_pipeline()
    entities = ner(text)

    found = []
    for ent in entities:
        label = ent["entity_group"].upper()
        # 라벨 체계가 모델별로 다를 수 있어 포함 여부로 판단
        if any(key in label for key in PII_RELATED_LABELS):
            found.append({
                "type": label,
                "value": ent["word"],
                "score": round(float(ent["score"]), 4),
            })

    return {
        "detected": len(found) > 0,
        "found": found,
    }


if __name__ == "__main__":
    test_cases = [
        "서울 강남구에 사는 김철수 고객 정보를 확인해주세요.",
        "홍길동님이 어제 방문하셨습니다.",
        "이번 달 업무 보고서를 작성했습니다.",  # 정상
        "회의 일정을 다음 주로 변경해주세요.",  # 정상
    ]

    print("모델 로딩 중...")
    print("=" * 60)
    for text in test_cases:
        result = detect_pii_ner(text)
        status = "탐지" if result["detected"] else "정상"
        print(f"[{status}] {text}")
        if result["detected"]:
            for f in result["found"]:
                print(f"       -> {f['type']}: {f['value']} (score: {f['score']})")
    print("=" * 60)
