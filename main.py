"""Run both projects from the command line (no GUI).
    python main.py            -> runs both, saves graphs + reports to ./outputs
    python main.py --student  -> only Project 1
    python main.py --house    -> only Project 2
"""
import argparse
import re
from pathlib import Path

from data_generator import ensure_datasets
from house_price import HousePriceSystem
from student_performance import StudentPerformanceSystem

OUT = Path(__file__).parent / "outputs"


def save(prefix, system, log_lines):
    OUT.mkdir(exist_ok=True)
    for name, fig in system.figures.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        fig.savefig(OUT / f"{prefix}_{slug}.png", dpi=130)
    (OUT / f"{prefix}_report.txt").write_text("\n".join(log_lines), encoding="utf-8")


def run(prefix, system_cls, csv):
    lines = []

    def log(msg):
        print(msg)
        lines.append(str(msg))

    system = system_cls(log=log)
    system.run_all(csv)
    save(prefix, system, lines)
    return system


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", action="store_true")
    ap.add_argument("--house", action="store_true")
    args = ap.parse_args()
    both = not (args.student or args.house)
    student_csv, house_csv = ensure_datasets()

    if args.student or both:
        print("=" * 70 + "\n PROJECT 1: STUDENT PERFORMANCE PREDICTION SYSTEM\n" + "=" * 70)
        s = run("student", StudentPerformanceSystem, student_csv)
        r = s.predict({"study_hours": 6, "attendance": 85, "previous_score": 70, "sleep_hours": 7,
                       "assignments_completed": 8, "parental_support": "Medium", "extra_classes": "Yes"})
        print(f"Sample prediction -> {r['result']} | marks ~ {r['marks']:.1f} | "
              f"pass probability {r['pass_probability'] * 100:.0f}%\n")

    if args.house or both:
        print("=" * 70 + "\n PROJECT 2: HOUSE PRICE PREDICTION SYSTEM\n" + "=" * 70)
        h = run("house", HousePriceSystem, house_csv)
        r = h.predict({"area_sqft": 1500, "bedrooms": 3, "bathrooms": 2, "age_years": 5, "parking": 1,
                       "distance_to_center_km": 8, "location": "Suburb", "furnishing": "Semi-Furnished"})
        print(f"Sample prediction -> {r['formatted']}\n")

    print(f"Graphs and reports saved in: {OUT}")
