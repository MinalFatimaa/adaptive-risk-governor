from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


FILES = [
    # Environment / world
    "src/environment/world.py",
    "src/environment/event_log.py",

    # Adaptive customer / agent simulation
    "src/environment/adaptive_customer_simulator.py",
    "src/environment/adaptive_agent_simulator.py",
    "src/environment/adaptive_customer.py",
    "src/environment/strategic_agent.py",
    "src/environment/strategic_state.py",

    # Governor
    "src/governor/governor.py",
    "src/governor/__init__.py",

    # Schemas used by the simulation
    "src/schemas/interaction.py",
    "src/schemas/customer.py",
    "src/schemas/order.py",
    "src/schemas/refund.py",
    "src/schemas/graph.py",

    # Intelligence / signals
    "src/intelligence/semantic_complaint.py",

    # Existing normal simulation — useful for matching conventions
    "src/environment/normal_customer_simulator.py",

    # Tests can reveal the intended contracts
    "tests/test_environment.py",
    "tests/test_governor.py",
    "tests/test_adaptive_customer.py",
    "tests/test_adaptive_agent.py",
    "tests/test_strategic.py",
]


def read_file(relative_path: str) -> None:
    path = ROOT / relative_path

    print()
    print("=" * 110)
    print(f"FILE: {relative_path}")
    print("=" * 110)

    if not path.exists():
        print("[FILE NOT FOUND]")
        return

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        print(f"[READ ERROR] {exc}")
        return

    print(text)


def main() -> None:
    print("=" * 110)
    print("ADAPTIVE RISK GOVERNOR — STEPS 89–95 PROJECT INSPECTION")
    print("=" * 110)

    print()
    print(f"PROJECT ROOT: {ROOT}")

    print()
    print("Checking required files...")

    for relative_path in FILES:
        path = ROOT / relative_path
        status = "FOUND" if path.exists() else "NOT FOUND"
        print(f"{status:<12} {relative_path}")

    print()
    print("=" * 110)
    print("BEGIN FILE CONTENT")
    print("=" * 110)

    for relative_path in FILES:
        read_file(relative_path)

    print()
    print("=" * 110)
    print("END OF STEPS 89–95 PROJECT INSPECTION")
    print("=" * 110)
    print()
    print("Paste this complete output back into ChatGPT.")
    print("Do not modify the generated output.")


if __name__ == "__main__":
    main()