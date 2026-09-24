"""
Project 2: House Price Prediction System
----------------------------------------
Objective : Predict house prices from features such as area and location.

Key concepts : Regression Models | Feature Engineering | Evaluation Metrics (R2, MAE)
Steps        : 1 Load (Pandas)  2 Handle missing values  3 Encode categoricals
               4 Train regression models  5 Evaluate and predict
Output       : Predicted house price and comparison graph
Prices are in Rs. lakhs (1 lakh = 100,000).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "price_lakhs"
LOCATIONS = ["Rural", "Outskirts", "Suburb", "IT Corridor", "City Center", "Coastal"]
FURNISHING = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully-Furnished": 2}
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years", "parking", "distance_to_center_km"]
CATEGORICAL = ["location", "furnishing"]
ENGINEERED = ["total_rooms", "area_per_bedroom", "is_new", "log_area"]
LOC_COLS = [f"loc_{loc}" for loc in LOCATIONS[1:]]          # 'Rural' = baseline (drop-first)
FEATURES = NUMERIC + ENGINEERED + ["furnishing_level"] + LOC_COLS
BLUE, GREEN, RED, ORANGE = "#4C78A8", "#2E9E5B", "#D9534F", "#F58518"


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def format_price(lakhs: float) -> str:
    return f"Rs. {lakhs / 100:.2f} Crore" if lakhs >= 100 else f"Rs. {lakhs:.2f} Lakh"


class HousePriceSystem:
    def __init__(self, log=print):
        self.log = log
        self.figures: "OrderedDict[str, Figure]" = OrderedDict()
        self.summary = ""

    # ------------------------------------------------------------ 1. LOAD
    def load_data(self, path):
        df = pd.read_csv(path)
        missing = [c for c in NUMERIC + CATEGORICAL + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")
        self.raw = df
        self.log(f"[1] LOAD DATA\n    {df.shape[0]} rows x {df.shape[1]} columns loaded")
        self.log(f"    Columns: {', '.join(df.columns)}\n")

    # ------------------------------------------------ 2. HANDLE MISSING VALUES
    def clean_data(self):
        df = self.raw.copy()
        self.log("[2] HANDLE MISSING VALUES & CLEAN")
        n = len(df)
        df = df.drop_duplicates()
        self.log(f"    - Removed {n - len(df)} duplicate rows")

        for c in NUMERIC + [TARGET]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in CATEGORICAL:
            df[c] = df[c].str.strip()
        na = df[NUMERIC + CATEGORICAL + [TARGET]].isna().sum()
        self.log("    - Missing values found: " +
                 (", ".join(f"{k}={v}" for k, v in na[na > 0].items()) or "none"))

        n = len(df)
        df = df.dropna(subset=[TARGET])
        df = df[(df[TARGET] > 0) & ~(df["area_sqft"] <= 0)]      # invalid rows (NaN area is kept, imputed)
        self.log(f"    - Dropped {n - len(df)} rows with missing/invalid price or area")
        for c in NUMERIC:
            df[c] = df[c].fillna(df[c].median())
        for c in CATEGORICAL:
            df[c] = df[c].fillna(df[c].mode()[0])
        df.loc[~df["location"].isin(LOCATIONS), "location"] = df["location"].mode()[0]
        df.loc[~df["furnishing"].isin(FURNISHING), "furnishing"] = df["furnishing"].mode()[0]
        self.log("    - Numeric gaps -> median | categorical gaps -> most frequent value")

        q1, q3 = df[TARGET].quantile([0.25, 0.75])
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
        n = len(df)
        df = df[(df[TARGET] >= lo) & (df[TARGET] <= hi)]
        self.log(f"    - Removed {n - len(df)} extreme price outliers (3 x IQR rule)")
        self.clean = df.reset_index(drop=True)
        self.log(f"    Clean dataset: {len(self.clean)} rows\n")

    # ---------------------------------- FEATURE ENGINEERING + 3. ENCODING
    @staticmethod
    def engineer_and_encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # feature engineering
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
        df["area_per_bedroom"] = df["area_sqft"] / df["bedrooms"].clip(lower=1)
        df["is_new"] = (df["age_years"] <= 5).astype(int)
        df["log_area"] = np.log1p(df["area_sqft"])
        # encoding categorical variables
        df["furnishing_level"] = df["furnishing"].map(FURNISHING)                    # ordinal
        for loc in LOCATIONS[1:]:
            df[f"loc_{loc}"] = (df["location"] == loc).astype(int)                    # one-hot
        return df

    def prepare(self, test_size=0.2, seed=42):
        self.data = self.engineer_and_encode(self.clean)
        self.log("[3] FEATURE ENGINEERING & ENCODING")
        self.log("    - New features: total_rooms, area_per_bedroom, is_new (<=5 yrs), log_area")
        self.log("    - furnishing -> ordinal (0/1/2) | location -> one-hot dummies "
                 "(Rural = baseline)")
        X, y = self.data[FEATURES], self.data[TARGET]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        self.scaler = StandardScaler().fit(self.X_train)
        self.Xs_train = self.scaler.transform(self.X_train)
        self.Xs_test = self.scaler.transform(self.X_test)
        self.log(f"    - {len(FEATURES)} features | Train {len(self.X_train)} / Test {len(self.X_test)} "
                 "(80/20), features standardised\n")

    # ------------------------------------------------------------------ EDA
    def eda(self):
        df = self.clean
        self.log("[EDA] Price summary (Rs. lakhs)")
        self.log("    " + df[TARGET].describe().round(2).to_string().replace("\n", "\n    "))
        self.log("\n    Average price by location:")
        avg = df.groupby("location")[TARGET].mean().reindex(LOCATIONS)
        self.log("\n".join(f"      {k:<14}{v:8.2f}" for k, v in avg.items()) + "\n")

        fig = Figure(figsize=(12, 8), layout="constrained")
        a1, a2, a3, a4 = fig.subplots(2, 2).ravel()
        a1.hist(df[TARGET], bins=30, color=BLUE, edgecolor="white")
        a1.set_title("Price Distribution"); a1.set_xlabel("Price (Rs. lakhs)")
        a2.boxplot([df.loc[df["location"] == l, TARGET] for l in LOCATIONS])
        a2.set_xticks(range(1, len(LOCATIONS) + 1), LOCATIONS, rotation=30)
        a2.set_title("Price by Location"); a2.set_ylabel("Price (Rs. lakhs)")
        sc = a3.scatter(df["area_sqft"], df[TARGET], c=df["age_years"], cmap="viridis", s=10, alpha=0.7)
        fig.colorbar(sc, ax=a3, label="Age (years)")
        a3.set_title("Area vs Price"); a3.set_xlabel("Area (sq ft)"); a3.set_ylabel("Price (Rs. lakhs)")
        cols = NUMERIC + [TARGET]
        cm = df[cols].corr()
        a4.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
        a4.set_xticks(range(len(cols)), [pretty(c) for c in cols], rotation=45, ha="right", fontsize=8)
        a4.set_yticks(range(len(cols)), [pretty(c) for c in cols], fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                a4.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        a4.set_title("Correlation Heatmap")
        fig.suptitle("Exploratory Data Analysis", fontweight="bold")
        self.figures["EDA"] = fig

    # --------------------------------------------------- 4. TRAIN REGRESSION
    def train(self):
        self.models = OrderedDict([
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge(alpha=1.0)),
            ("Random Forest", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        ])
        self.log("[4] TRAIN REGRESSION MODELS (5-fold cross-validation on training set)")
        self.cv = {}
        for name, model in self.models.items():
            self.cv[name] = cross_val_score(model, self.Xs_train, self.y_train, cv=5, scoring="r2").mean()
            model.fit(self.Xs_train, self.y_train)
            self.log(f"    - {name:<19} CV R2 = {self.cv[name]:.3f}")
        self.best_name = max(self.cv, key=self.cv.get)
        self.best = self.models[self.best_name]
        self.log(f"    Best model by CV: {self.best_name}\n")

    # ----------------------------------------------------- 5. EVALUATE
    def evaluate(self):
        self.results = {}
        for name, model in self.models.items():
            p = model.predict(self.Xs_test)
            self.results[name] = {
                "r2": r2_score(self.y_test, p), "mae": mean_absolute_error(self.y_test, p),
                "rmse": mean_squared_error(self.y_test, p) ** 0.5, "pred": p}
        self.log("[5] EVALUATION ON UNSEEN TEST DATA")
        self.log(f"    {'Model':<20}{'R2':>8}{'MAE (lakh)':>13}{'RMSE (lakh)':>14}")
        for name, r in self.results.items():
            star = "  <- best" if name == self.best_name else ""
            self.log(f"    {name:<20}{r['r2']:>8.3f}{r['mae']:>13.2f}{r['rmse']:>14.2f}{star}")
        b = self.results[self.best_name]
        self.log(f"\n    {self.best_name}: explains {b['r2'] * 100:.1f}% of price variation, "
                 f"typical error about {format_price(b['mae'])}\n")

        names = list(self.results)
        # model comparison
        fig = Figure(figsize=(11, 4.6), layout="constrained")
        a1, a2 = fig.subplots(1, 2)
        colors = [GREEN if n == self.best_name else BLUE for n in names]
        a1.bar([n.replace(" ", "\n") for n in names], [self.results[n]["r2"] for n in names], color=colors)
        a1.set_title("R2 Score (higher is better)"); a1.set_ylim(0, 1.05)
        a2.bar([n.replace(" ", "\n") for n in names], [self.results[n]["mae"] for n in names], color=colors)
        a2.set_title("MAE in Rs. lakhs (lower is better)")
        for ax in (a1, a2):
            for p in ax.patches:
                ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.2f}",
                        ha="center", va="bottom", fontsize=9)
        fig.suptitle("Model Comparison", fontweight="bold")
        self.figures["Model Comparison"] = fig

        # actual vs predicted (the required comparison graph)
        pred = b["pred"]
        y = self.y_test.to_numpy()
        fig = Figure(figsize=(13, 5), layout="constrained")
        a1, a2 = fig.subplots(1, 2)
        a1.scatter(y, pred, s=12, alpha=0.6, color=BLUE)
        lim = [0, max(y.max(), pred.max()) * 1.05]
        a1.plot(lim, lim, "r--", lw=1.2, label="Perfect prediction")
        a1.set_xlabel("Actual Price (lakhs)"); a1.set_ylabel("Predicted Price (lakhs)")
        a1.set_title(f"Actual vs Predicted (R2 = {b['r2']:.3f})"); a1.legend()
        k = 50
        a2.plot(range(k), y[:k], "o-", color=BLUE, label="Actual", ms=4)
        a2.plot(range(k), pred[:k], "s--", color=ORANGE, label="Predicted", ms=4)
        a2.set_xlabel("Test house #"); a2.set_ylabel("Price (lakhs)")
        a2.set_title("Comparison Graph: first 50 test houses"); a2.legend()
        fig.suptitle("Predicted vs Actual House Prices", fontweight="bold")
        self.figures["Actual vs Predicted"] = fig

        # feature importance + residuals
        if hasattr(self.best, "feature_importances_"):
            imp = pd.Series(self.best.feature_importances_, index=FEATURES)
        else:
            imp = pd.Series(np.abs(self.best.coef_), index=FEATURES)
            imp = imp / imp.sum()
        imp = imp.sort_values().tail(12)
        fig = Figure(figsize=(12, 4.8), layout="constrained")
        a1, a2 = fig.subplots(1, 2)
        a1.barh([pretty(i) for i in imp.index], imp.values, color=BLUE)
        a1.set_title("Top Features Driving Price")
        a2.hist(y - pred, bins=30, color=ORANGE, edgecolor="white")
        a2.axvline(0, color="k", lw=1)
        a2.set_title("Prediction Errors (Actual - Predicted)"); a2.set_xlabel("Error (lakhs)")
        self.figures["Feature Importance"] = fig

        self.summary = (f"Best: {self.best_name}\nR2: {b['r2']:.3f}   MAE: Rs. {b['mae']:.2f} lakh")
        return self.results

    # ------------------------------------------------------------ PREDICTION
    def predict(self, values: dict) -> dict:
        row = self.engineer_and_encode(pd.DataFrame([values]))[FEATURES]
        X = self.scaler.transform(row)
        by_model = {n: float(m.predict(X)[0]) for n, m in self.models.items()}
        price = by_model[self.best_name]
        return {"price_lakhs": price, "formatted": format_price(price), "by_model": by_model}

    # ---------------------------------------------------------------- PIPELINE
    def run_all(self, path):
        self.figures = OrderedDict()
        self.load_data(path)
        self.clean_data()
        self.prepare()
        self.eda()
        self.train()
        self.evaluate()
        return {"figures": self.figures, "summary": self.summary}
