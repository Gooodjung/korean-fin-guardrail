import re

_REPLACE_RULES = [
    (re.compile(r"\(\s*at\s*\)", re.IGNORECASE), "@"),
    (re.compile(r"\(\s*dot\s*\)", re.IGNORECASE), "."),
    (re.compile(r"\(\s*앳\s*\)"), "@"),
    (re.compile(r"\(\s*점\s*\)"), "."),
    (re.compile(r"\bat\b", re.IGNORECASE), "@"),
    (re.compile(r"\bdot\b", re.IGNORECASE), "."),
]

# @ 또는 .을 포함하면서 공백/줄바꿈이 섞인 연속 구간을 찾는 패턴
# (영문/숫자/한글 + 공백 + @ 또는 . + 공백 + 영문/숫자/한글 ... 형태)
_SPACED_EMAIL_BLOCK = re.compile(
    r"[A-Za-z0-9가-힣]+(?:[\s]*[@.][\s]*[A-Za-z0-9가-힣]+){2,}"
)


def _compress_block(match: re.Match) -> str:
    """매칭된 블록 내부의 모든 공백/줄바꿈 제거"""
    return re.sub(r"\s+", "", match.group())


def _normalize(text: str) -> str:
    normalized = text

    # 1-pass: 단어/기호 치환
    for pattern, replacement in _REPLACE_RULES:
        normalized = pattern.sub(replacement, normalized)

    # 2-pass: 줄바꿈 제거
    normalized = re.sub(r"[\n\r]+", " ", normalized)

    # 3-pass: @ 또는 . 이 포함된 공백 섞인 블록을 통째로 압축
    normalized = _SPACED_EMAIL_BLOCK.sub(_compress_block, normalized)

    return normalized


PATTERNS = {
    "주민등록번호": re.compile(r"(?<!\d)\d{6}[-\s]?\d{7}(?!\d)"),
    "계좌번호": re.compile(r"(?<!\d)\d{2,6}[-\s]\d{2,6}[-\s]\d{2,8}(?!\d)"),
    "전화번호": re.compile(r"(?<!\d)01[016789][-\s]?\d{3,4}[-\s]?\d{4}(?!\d)"),
    "카드번호": re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)"),
    "이메일": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}

ORDERED_TYPES = ["주민등록번호", "카드번호", "전화번호", "계좌번호", "이메일"]


def detect_pii_regex(text: str) -> dict:
    found = []
    matched_spans = []

    for pii_type in ORDERED_TYPES:
        pattern = PATTERNS[pii_type]
        for match in pattern.finditer(text):
            span = match.span()
            overlap = any(s[0] < span[1] and span[0] < s[1] for s in matched_spans)
            if not overlap:
                found.append({"type": pii_type, "value": match.group(), "obfuscated": False})
                matched_spans.append(span)

    if not found:
        normalized_text = _normalize(text)
        if normalized_text != text:
            for pii_type in ORDERED_TYPES:
                pattern = PATTERNS[pii_type]
                for match in pattern.finditer(normalized_text):
                    found.append({
                        "type": pii_type,
                        "value": match.group(),
                        "obfuscated": True,
                        "original_text": text,
                    })

    return {
        "detected": len(found) > 0,
        "found": found,
    }


if __name__ == "__main__":
    test_cases = [
        "고객 홍길동의 주민번호는 901010-1234567입니다.",
        "계좌번호 123-456-789012로 이체해주세요.",
        "연락처는 010-1234-5678입니다.",
        "카드번호 1234-5678-9012-3456 확인 부탁드립니다.",
        "이메일 test@example.com 으로 보내주세요.",
        "이번 달 업무 보고서를 작성했습니다.",  # 정상
        "이메일 user(at)example.com 형식으로 썼어",
        "이메일 user (at) example (dot) com 띄어쓰기 삽입",
        "이메일 ad min@ex ample.com",
        "이메일은 hello(dot)world(at)example(dot)com 이예요",
        "이메일 주소: u s e r (at) d o m a i n (dot) c o m",
    ]

    print("=" * 60)
    for text in test_cases:
        result = detect_pii_regex(text)
        status = "탐지" if result["detected"] else "정상"
        print(f"[{status}] {text}")
        if result["detected"]:
            for f in result["found"]:
                obf_tag = " [난독화 복원]" if f.get("obfuscated") else ""
                print(f"       -> {f['type']}: {f['value']}{obf_tag}")
    print("=" * 60)
