#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def _run(cmd: list):
    """Run the underlying script as-is and propagate its exit code."""
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_check(args):
    sys.path.append(str(ROOT / "guardrail"))
    from guardrail import run_guardrail  # noqa: E402
    text = " ".join(args)
    if not text:
        print("사용법: python cli.py check \"<검사할 텍스트>\"")
        sys.exit(1)
    result = run_guardrail(text, verbose=True)
    print("\n" + "=" * 50)
    print("차단됨" if result["blocked"] else "통과", "→", result.get("blocked_by") or "-")


def cmd_redteam(args):
    _run([sys.executable, str(ROOT / "redteam" / "redteam.py")] + args)


def cmd_seed_rules(args):
    _run([sys.executable, str(ROOT / "blueteam" / "seed_rules_v4.py")] + args)


def cmd_tune(args):
    _run([sys.executable, str(ROOT / "blueteam" / "auto_tuning_v4.py")] + args)


def cmd_multi_model(args):
    _run([sys.executable, str(ROOT / "redteam" / "multi_model_redteam.py")] + args)


def cmd_baseline_presidio(args):
    _run([sys.executable, str(ROOT / "experiments" / "baseline_presidio_comparison.py")] + args)


def cmd_baseline_llm_guard(args):
    _run([sys.executable, str(ROOT / "experiments" / "baseline_llm_guard_comparison.py")] + args)


def cmd_judge_injection_test(args):
    _run([sys.executable, str(ROOT / "experiments" / "test_judge_prompt_injection.py")] + args)


COMMANDS = {
    "check": cmd_check,
    "redteam": cmd_redteam,
    "seed-rules": cmd_seed_rules,
    "tune": cmd_tune,
    "multi-model": cmd_multi_model,
    "baseline-presidio": cmd_baseline_presidio,
    "baseline-llm-guard": cmd_baseline_llm_guard,
    "judge-injection-test": cmd_judge_injection_test,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("사용 가능한 서브커맨드:")
        for name in COMMANDS:
            print(f"  - {name}")
        print("\n(라이브 시연용 대화형 데모는 streamlit_demo.py 참고: streamlit run streamlit_demo.py)")
        sys.exit(0)

    sub = sys.argv[1]
    rest = sys.argv[2:]
    if sub not in COMMANDS:
        print(f"알 수 없는 서브커맨드: {sub}")
        print("사용 가능:", ", ".join(COMMANDS))
        sys.exit(1)

    COMMANDS[sub](rest)


if __name__ == "__main__":
    main()
