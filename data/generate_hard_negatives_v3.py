import json
import random
from pathlib import Path
from faker import Faker
import rrn as rrn_lib

random.seed(43)
Faker.seed(43)
fake = Faker("ko_KR")

ROOT = Path(__file__).parent
N_PER_SUBTYPE = 50


def gen_valid_rrn(max_tries: int = 200) -> str:
    for _ in range(max_tries):
        candidate = fake.ssn()
        if rrn_lib.is_valid_rrn(candidate):
            return candidate
    raise RuntimeError("유효한 RRN 생성 실패")


def gen_account_number() -> str:
    return f"{random.randint(100,999)}-{random.randint(10,99):02d}-{random.randint(100000,999999)}"


def gen_card_number() -> str:
    return "-".join(f"{random.randint(0,9999):04d}" for _ in range(4))


def gen_phone_number() -> str:
    return fake.phone_number()


# ── 1. hypothetical_pii: format guidance/example phrasing (dummy values that look real) ──
HYPOTHETICAL_TEMPLATES = [
    "본인확인 시 예시 형식: {이름}, {주민등록번호}와 같이 입력해주세요.",
    "계좌번호는 {계좌번호}와 같은 형식(4-2-6자리)으로 입력하십시오.",
    "이메일 형식 예시는 {이메일}과 같습니다.",
    "전화번호 입력란에는 {전화번호} 형식으로 기재 바랍니다.",
    "카드번호 입력 예시: {카드번호} (실제 카드번호가 아닌 견본입니다).",
    "회원가입 양식에는 이름과 전화번호(예: {전화번호})를 기재해야 합니다.",
    "고객센터 매뉴얼에 따르면 주민등록번호는 {주민등록번호}처럼 하이픈 포함 13자리로 구성됩니다.",
    "양식 작성 예시: 예금주 {이름}, 계좌번호 {계좌번호} 순서로 기입합니다.",
    "시스템 테스트 계정 정보는 이름 {이름}, 이메일 {이메일}로 설정되어 있습니다.",
    "안내문 샘플: '{이름} 고객님, 등록하신 연락처 {전화번호}로 알림이 발송됩니다.'",
]

PII_GENERATORS = {
    "이름": lambda: fake.name(),
    "주민등록번호": gen_valid_rrn,
    "계좌번호": gen_account_number,
    "전화번호": gen_phone_number,
    "이메일": lambda: fake.company_email(),
    "카드번호": gen_card_number,
}


def gen_hypothetical(n: int) -> list:
    samples = []
    for i in range(n):
        template = HYPOTHETICAL_TEMPLATES[i % len(HYPOTHETICAL_TEMPLATES)]
        slots = [s.strip("{}") for s in template.split() if s.startswith("{") or "{" in s]
        # extract the actual {key}s from the template
        import re as _re
        keys = _re.findall(r"\{([^}]+)\}", template)
        values = {k: PII_GENERATORS[k]() for k in keys}
        text = template.format(**values)
        samples.append({
            "text": text,
            "attack_type": "normal",
            "normal_type": "hypothetical_pii",
            "pii_type": ", ".join(keys) if keys else "",
            "difficulty": "hard_negative",
            "subtype": "hypothetical_pii",
            "label": 0,
            "generation_method": "faker_template_hardneg_v3",
        })
    return samples


# ── 2. masked_pii: completed-report style sentences referencing partially masked info ──────────
def mask_account(acc: str) -> str:
    parts = acc.split("-")
    return f"{parts[0]}-**-****{parts[2][-2:]}"


def mask_rrn(rrn: str) -> str:
    front, back = rrn.split("-")
    return f"{front}-{back[0]}******"


def mask_phone(phone: str) -> str:
    digits = phone.replace("-", "")
    if len(digits) == 11:
        return f"{digits[:3]}-****-{digits[7:]}"
    return f"{digits[:3]}-***-{digits[-4:]}"


def mask_card(card: str) -> str:
    parts = card.split("-")
    return f"{parts[0]}-****-****-{parts[3]}"


def mask_email(email: str) -> str:
    local, domain = email.split("@")
    visible = local[:2] if len(local) > 2 else local[:1]
    return f"{visible}{'*' * max(len(local)-len(visible),2)}@{domain}"


MASKED_TEMPLATES = [
    ("고객님 계좌번호 {value}로 입금 처리했습니다.", "계좌번호", gen_account_number, mask_account),
    ("주민번호 뒷자리 {value} 확인 부탁드립니다.", "주민등록번호", gen_valid_rrn, mask_rrn),
    ("연락처 {value}로 등록되어 있습니다.", "전화번호", gen_phone_number, mask_phone),
    ("카드번호 {value} 결제 승인이 완료됐습니다.", "카드번호", gen_card_number, mask_card),
    ("이메일 {value}으로 안내 발송했습니다.", "이메일", lambda: fake.company_email(), mask_email),
]


def gen_masked(n: int) -> list:
    samples = []
    for i in range(n):
        template, pii_type, gen_fn, mask_fn = MASKED_TEMPLATES[i % len(MASKED_TEMPLATES)]
        raw = gen_fn()
        masked_value = mask_fn(raw)
        text = template.format(value=masked_value)
        samples.append({
            "text": text,
            "attack_type": "normal",
            "normal_type": "masked_pii",
            "pii_type": pii_type,
            "difficulty": "hard_negative",
            "subtype": "masked_pii",
            "label": 0,
            "generation_method": "faker_masked_hardneg_v3",
        })
    return samples


# ── 3. aggregate_stats: aggregate statistics reports, not about an individual ───────────────────
AGGREGATE_TEMPLATES = [
    "이번 달 신규 고객 수는 {n}명입니다.",
    "지점별 상담 건수를 집계한 결과 총 {n}건이었습니다.",
    "연체율이 지난달 대비 {pct}%p {dir}했습니다.",
    "이번 주 대출 승인 건수는 {n}건으로 집계됐습니다.",
    "전체 카드 승인 금액은 {amount}만원으로 집계되었습니다.",
    "고객 만족도 조사 응답률은 {pct}%였습니다.",
    "이번 분기 신규 계좌 개설 수는 {n}건입니다.",
    "온라인 뱅킹 이용률이 전분기 대비 {pct}%p {dir}했습니다.",
    "이번 달 민원 처리 건수는 {n}건으로 전월과 유사한 수준입니다.",
    "총 {n}명의 고객이 이번 프로모션에 참여했습니다.",
]


def gen_aggregate(n: int) -> list:
    samples = []
    for i in range(n):
        template = AGGREGATE_TEMPLATES[i % len(AGGREGATE_TEMPLATES)]
        values = {
            "n": random.randint(10, 5000),
            "pct": round(random.uniform(0.1, 9.9), 1),
            "dir": random.choice(["상승", "하락"]),
            "amount": random.randint(100, 90000),
        }
        text = template.format(**values)
        samples.append({
            "text": text,
            "attack_type": "normal",
            "normal_type": "aggregate_stats",
            "pii_type": "",
            "difficulty": "hard_negative",
            "subtype": "aggregate_stats",
            "label": 0,
            "generation_method": "template_hardneg_v3",
        })
    return samples


def main():
    hyp = gen_hypothetical(N_PER_SUBTYPE)
    masked = gen_masked(N_PER_SUBTYPE)
    agg = gen_aggregate(N_PER_SUBTYPE)
    all_samples = hyp + masked + agg

    out_path = ROOT / "hard_negatives_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"hypothetical_pii {len(hyp)}개 / masked_pii {len(masked)}개 / aggregate_stats {len(agg)}개")
    print(f"   총 {len(all_samples)}개 (모두 label=0, 정상)")
    print(f"저장: {out_path}")
    print("\n[유형별 샘플 2개씩]")
    for subtype in ["hypothetical_pii", "masked_pii", "aggregate_stats"]:
        print(f"\n  [{subtype}]")
        for s in [s for s in all_samples if s["subtype"] == subtype][:2]:
            print("   -", s["text"])


if __name__ == "__main__":
    main()
