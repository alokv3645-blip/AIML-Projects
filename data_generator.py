"""Creates realistic sample datasets (with deliberate missing values, duplicates
and outliers) so the preprocessing steps have real work to do.

Replace these CSVs with your own data any time - just keep the column names.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
STUDENT_CSV = DATA_DIR / "student_data.csv"
HOUSE_CSV = DATA_DIR / "house_data.csv"


def _messy(df, rng, cols, frac):
    """Blank out a fraction of values in the given columns."""
    df = df.copy()
    for c in cols:
        df.loc[rng.random(len(df)) < frac, c] = np.nan
    return df


def generate_student_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(4.5, 2.0, n), 0, 12).round(1)
    attendance = np.clip(rng.normal(78, 12, n), 35, 100).round(1)
    prev = np.clip(rng.normal(62, 15, n), 15, 100).round(1)
    sleep = np.clip(rng.normal(6.8, 1.1, n), 3.5, 10).round(1)
    assign = rng.integers(0, 11, n)
    support = rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.5, 0.25])
    extra = rng.choice(["No", "Yes"], n, p=[0.6, 0.4])

    bonus = pd.Series(support).map({"Low": -3, "Medium": 0, "High": 3}).to_numpy()
    marks = (-14 + 3.2 * study + 0.18 * attendance + 0.32 * prev + 0.9 * assign
             + 1.2 * np.minimum(sleep, 8) + bonus + 2 * (extra == "Yes")
             + rng.normal(0, 4.5, n))
    df = pd.DataFrame({
        "study_hours": study, "attendance": attendance, "previous_score": prev,
        "sleep_hours": sleep, "assignments_completed": assign,
        "parental_support": support, "extra_classes": extra,
        "final_marks": np.clip(marks, 0, 100).round(1),
    })
    df = _messy(df, rng, ["study_hours", "attendance", "sleep_hours", "parental_support"], 0.03)
    df.loc[[5, 77, 140], "attendance"] = [150, 240, 130]      # impossible values
    df.loc[[20, 300], "study_hours"] = [45, 60]               # impossible values
    return pd.concat([df, df.sample(6, random_state=seed)], ignore_index=True)  # duplicates


LOCATIONS = ["Rural", "Outskirts", "Suburb", "IT Corridor", "City Center", "Coastal"]
RATE = {"Rural": 2500, "Outskirts": 3800, "Suburb": 5000,
        "IT Corridor": 6200, "City Center": 8000, "Coastal": 7000}   # Rs per sq ft
DIST = {"Rural": 25, "Outskirts": 15, "Suburb": 9, "IT Corridor": 12, "City Center": 3, "Coastal": 8}


def generate_house_data(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    loc = rng.choice(LOCATIONS, n, p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1])
    area = np.clip(rng.normal(1400, 500, n), 450, 4000).round(0)
    beds = np.clip(np.round(area / 500 + rng.normal(0, 0.6, n)), 1, 6).astype(int)
    baths = np.clip(beds + rng.integers(-1, 2, n), 1, 6)
    age = rng.integers(0, 31, n)
    parking = rng.integers(0, 4, n)
    dist = np.clip(pd.Series(loc).map(DIST).to_numpy() + rng.normal(0, 3, n), 0.5, 40).round(1)
    furn = rng.choice(["Unfurnished", "Semi-Furnished", "Fully-Furnished"], n, p=[0.4, 0.4, 0.2])

    rate = pd.Series(loc).map(RATE).to_numpy()
    fbonus = pd.Series(furn).map({"Unfurnished": 0, "Semi-Furnished": 3, "Fully-Furnished": 6}).to_numpy()
    price = (area * rate / 1e5 * (1 - 0.012 * age) * (1 - 0.006 * dist)
             + 1.5 * beds + 1.2 * baths + 2.5 * parking + fbonus)
    price = price * rng.lognormal(0, 0.07, n)

    df = pd.DataFrame({
        "area_sqft": area, "bedrooms": beds, "bathrooms": baths, "age_years": age,
        "parking": parking, "distance_to_center_km": dist, "location": loc,
        "furnishing": furn, "price_lakhs": price.round(2),
    })
    df = _messy(df, rng, ["area_sqft", "bathrooms", "location", "furnishing"], 0.04)
    df.loc[[10, 500], "price_lakhs"] = df["price_lakhs"].max() * 8   # price outliers
    df.loc[[42], "area_sqft"] = -1                                   # invalid area
    return pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)


def ensure_datasets():
    DATA_DIR.mkdir(exist_ok=True)
    if not STUDENT_CSV.exists():
        generate_student_data().to_csv(STUDENT_CSV, index=False)
    if not HOUSE_CSV.exists():
        generate_house_data().to_csv(HOUSE_CSV, index=False)
    return STUDENT_CSV, HOUSE_CSV


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    generate_student_data().to_csv(STUDENT_CSV, index=False)
    generate_house_data().to_csv(HOUSE_CSV, index=False)
    print("Created:", STUDENT_CSV, HOUSE_CSV)
