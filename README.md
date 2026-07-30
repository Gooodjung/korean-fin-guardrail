# 금융 도메인 Input Guardrail 자동 튜닝 프레임워크

레드팀(공격 생성) → 측정(ASR/FPR) → 블루팀(패치 제안) → 회귀 검증(McNemar) 
→채택/롤백/에스컬레이션이 자동으로 반복되는 가드레일 자동 튜닝 프레임워크.

## 아키텍처 요약

```
입력 텍스트
   │
   ▼
Layer 1 (정규식) ──detected──▶ 차단
   │ 통과
   ▼
Layer 2 (NER)  ──detected──▶ 차단
   │ 통과
   ▼
Layer 3 (LLM, 맥락 분석 + 인젝션 통합) ──detected──▶ 차단
   │ 통과
   ▼
정상 처리
```

자동 튜닝 루프(v4)는 Layer 3의 판단 기준(SYSTEM_PROMPT)을 구조화된 규칙
저장소(`guardrail/layer3_rules_v4.json`)로 관리하며, 레드팀 평가에서 취약
유형이 발견되면 (1) 국소 패치를 제안 → (2) 누적 회귀 코퍼스 전체에 대해
McNemar 통계 검정 → (3) 유의미하게 개선되고 회귀가 없을 때만 채택, 아니면
롤백 → (4) 같은 유형이 2회 연속 롤백되면 구조적 리팩터링 모드로 자동 전환
하는 사이클을 사람 개입 없이 반복한다.

## 폴더 구조

```
guardrail/      
3-Layer 가드레일 본체 (Layer1 정규식 / Layer2 NER / Layer3 LLM)
redteam/         
공격 생성·ASR/FPR/Latency 측정, 다중 모델 비교, ablation study
blueteam/        
자동 튜닝 루프(v4, 구조화 규칙 + 에스컬레이션)와 오탐 완화 패치
data/            
데이터셋 생성·검증 스크립트 (mixed_dataset_v3.json 기준)
experiments/     
baseline 비교, held-out 검증, 캐싱/레이턴시 등 실험 스크립트
cli.py           
위 스크립트들을 통일된 인터페이스로 실행하는 진입점
streamlit_demo.py   
실시간 검사·자동 튜닝·오탐 완화를 보여주는 라이브 데모
README_DEMO.md   
streamlit_demo.py 실행 가이드
```

## 사용법 (cli.py)

```bash
# 문장 하나를 3-Layer 파이프라인에 통과시켜보기
python cli.py check "홍길동 고객님 계좌번호 123-456-789012로 처리해주세요"

# 레드팀 평가 (ASR/FPR/Latency 측정)
python cli.py redteam --dataset data/mixed_dataset_v3.json --output experiments/redteam_new.json

# v4 규칙 저장소 최초 초기화 (한 번만)
python cli.py seed-rules

# v4 자동 튜닝 루프 실행
python cli.py tune --dataset data/mixed_dataset_v3.json --threshold 0.05 --max-rounds 3 --auto-approve

# 다중 모델(방어자) 비교
python cli.py multi-model --defender claude-sonnet --sample 50
python cli.py multi-model --defender llama-3 --sample 50

# baseline 도구 비교
python cli.py baseline-presidio --korean-nlp
python cli.py baseline-llm-guard

# Layer3 판단 기준 자체의 프롬프트 인젝션 견고성 테스트
python cli.py judge-injection-test --n 100
```

각 서브커맨드는 실제로는 `redteam/`, `blueteam/`, `experiments/` 아래의
독립 스크립트를 그대로 실행하는 얇은 래퍼다 — 내부 로직을 바꾸지 않고
사용법만 통일했다. `python cli.py <subcommand> -h`로 하위 스크립트의 실제
옵션을 확인할 수 있다.

## 자동 프레임워크 실행

```bash
pip install streamlit
streamlit run streamlit_demo.py
```

실시간 검사, 자동 튜닝(놓친 공격 → 패치 제안 → 재검증), 오탐 완화(잘못
차단된 정상 문장 → SAFE 예외 제안 → 재검증) 세 화면으로 구성되어 있다.
기존 가드레일 코드(`guardrail/*.py`)나 디스크에 저장된 규칙 파일은 건드리지
않고, 메모리 상의 규칙 사본만 가지고 동작한다. 
(자세한 실행 가이드는 `README_DEMO.md` 참고.)

## 재현 순서 (처음부터 다시 실행할 때)

1. `python cli.py seed-rules` — v4 규칙 저장소 초기화
2. `python cli.py redteam --dataset data/mixed_dataset_v3.json` — baseline 측정
3. `python cli.py tune --dataset data/mixed_dataset_v3.json --auto-approve` — 자동 튜닝
4. `python cli.py multi-model --defender claude-sonnet` / `--defender llama-3` — 모델 비교
5. `python cli.py baseline-presidio` / `baseline-llm-guard` — 기존 오픈소스 도구 비교
6. `python cli.py judge-injection-test` — 판단 기준 견고성 검증
7. `python experiments/caching_latency_test.py` — 캐싱 효과 측정
