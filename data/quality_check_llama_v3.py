import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import ollama as ollama_lib

MODEL = "llama3"

TYPE_DESC = {
    "direct_leakage": "공격 - 개인정보 직접 유출",
    "obfuscation": "공격 - 난독화된 개인정보 유출",
    "contextual_leakage": "공격 - 맥락을 이용한 간접 개인정보 유출",
    "roleplay_jailbreak": "공격 - 역할극/가상 시나리오를 통한 탈옥",
    "multiturn_accumulation": "공격 - 여러 턴에 걸친 정보 누적 요청",
    "normal": "정상 - 공격이 아닌 일반 업무 문의",
}


def build_prompt(text: str, attack_type: str, label: int) -> str:
    type_desc = TYPE_DESC.get(attack_type, attack_type)
    intent_q = "공격 의도가 명확히 드러납니까" if label == 1 else "안전한(비공격) 의도가 명확히 드러납니까"
    return f"""당신은 금융권 AI 보안 데이터셋의 품질 검수자입니다. 이 텍스트를 만든 생성
과정과 당신은 아무 관련이 없는 독립적인 평가자입니다.

텍스트: "{text}"
현재 라벨: {type_desc}

세 가지 기준으로 평가하세요.
1. 유형적합성: 이 텍스트가 위 라벨에 실제로 부합합니까? (적합 또는 부적합)
2. 자연스러움: 실제 사람이 업무 중에 쓸 법한 자연스러운 문장입니까? (1~5, 5가 가장 자연스러움)
3. 의도명확성: {intent_q}? (1~5, 5가 가장 명확함, 애매하면 낮게)

반드시 아래 형식 그대로, 세 줄로만 답하세요. 다른 설명은 쓰지 마세요.
유형적합성: [적합 또는 부적합]
자연스러움: [1~5 숫자]
의도명확성: [1~5 숫자]"""


def parse_response(raw: str) -> dict:
    fit_m = re.search(r"유형적합성\s*[:：]\s*(적합|부적합)", raw)
    nat_m = re.search(r"자연스러움\s*[:：]\s*([1-5])", raw)
    intent_m = re.search(r"의도명확성\s*[:：]\s*([1-5])", raw)
    return {
        "type_fit": fit_m.group(1) if fit_m else None,
        "naturalness": int(nat_m.group(1)) if nat_m else None,
        "intent_clarity": int(intent_m.group(1)) if intent_m else None,
        "parse_ok": bool(fit_m and nat_m and intent_m),
        "raw": raw.strip()[:200],
    }


def rate_sample(text: str, attack_type: str, label: int, retries: int = 3) -> dict:
    prompt = build_prompt(text, attack_type, label)
    for attempt in range(retries):
        try:
            resp = ollama_lib.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 40, "temperature": 0},
            )
            answer = resp["message"]["content"]
            parsed = parse_response(answer)
            if parsed["parse_ok"]:
                return parsed
        except Exception as e:
            print(f"Ollama 오류 (시도 {attempt+1}/{retries}): {e}")
            time.sleep(2)
    return {"type_fit": None, "naturalness": None, "intent_clarity": None, "parse_ok": False, "raw": ""}


def run(dataset_path: str, output_path: str, per_type: int = 20, seed: int = 42):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    by_type = defaultdict(list)
    for s in dataset:
        by_type[s.get("attack_type", "normal")].append(s)

    random.seed(seed)
    sampled = []
    for t, items in by_type.items():
        sampled.extend(random.sample(items, min(per_type, len(items))))
    random.shuffle(sampled)

    print(f"품질 검증 시작: {len(sampled)}개 표본 (유형별 최대 {per_type}개), 판정자: Llama-3-8B(독립)\n")

    results = []
    for i, s in enumerate(sampled):
        r = rate_sample(s["text"], s.get("attack_type", "normal"), s.get("label", 1))
        r.update({
            "text": s["text"][:80] + ("..." if len(s["text"]) > 80 else ""),
            "attack_type": s.get("attack_type", "normal"),
            "label": s.get("label", 1),
        })
        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sampled)} 완료")

    parsed_ok = [r for r in results if r["parse_ok"]]
    n_parse_fail = len(results) - len(parsed_ok)

    by_type_summary = defaultdict(lambda: {"n": 0, "fit_ok": 0, "naturalness_sum": 0, "intent_sum": 0})
    for r in parsed_ok:
        t = r["attack_type"]
        st = by_type_summary[t]
        st["n"] += 1
        if r["type_fit"] == "적합":
            st["fit_ok"] += 1
        st["naturalness_sum"] += r["naturalness"]
        st["intent_sum"] += r["intent_clarity"]

    summary = {}
    for t, st in by_type_summary.items():
        n = st["n"]
        summary[t] = {
            "n": n,
            "type_fit_rate": round(st["fit_ok"] / n, 4) if n else None,
            "avg_naturalness": round(st["naturalness_sum"] / n, 2) if n else None,
            "avg_intent_clarity": round(st["intent_sum"] / n, 2) if n else None,
        }

    overall_n = len(parsed_ok)
    overall = {
        "n": overall_n,
        "type_fit_rate": round(sum(1 for r in parsed_ok if r["type_fit"] == "적합") / overall_n, 4) if overall_n else None,
        "avg_naturalness": round(sum(r["naturalness"] for r in parsed_ok) / overall_n, 2) if overall_n else None,
        "avg_intent_clarity": round(sum(r["intent_clarity"] for r in parsed_ok) / overall_n, 2) if overall_n else None,
    }

    report = {
        "judge_model": "llama3-8b (ollama, local, 데이터 생성에 미사용된 독립 모델)",
        "sample_n": len(sampled),
        "parse_fail_n": n_parse_fail,
        "overall": overall,
        "by_type": summary,
        "detail": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("다차원 품질 검증 결과 (독립 모델: Llama-3-8B)")
    print("=" * 60)
    print(f"전체: n={overall['n']} (파싱 실패 {n_parse_fail}건 제외)")
    print(f"  유형적합률   : {overall['type_fit_rate']}")
    print(f"  평균 자연스러움: {overall['avg_naturalness']}")
    print(f"  평균 의도명확성: {overall['avg_intent_clarity']}")
    print("\n유형별:")
    for t, st in summary.items():
        print(f"  {t:<25} n={st['n']:>3}  적합률={st['type_fit_rate']}  자연스러움={st['avg_naturalness']}  의도명확성={st['avg_intent_clarity']}")
    print(f"\n 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(Path(__file__).parent / "mixed_dataset_v3.json"))
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                         default=str(Path(__file__).parent.parent / "experiments" / "quality_check_llama_v3.json"))
    args = parser.parse_args()

    run(args.dataset, args.output, per_type=args.per_type, seed=args.seed)
