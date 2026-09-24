"""
Project 1: Student Performance Prediction System
------------------------------------------------
Objective : Predict whether a student will PASS / FAIL (Logistic Regression) and
            estimate the final MARKS (Linear Regression) from study habits.

Key concepts : Data Preprocessing | EDA | Linear & Logistic Regression | Feature Scaling
Steps        : 1 Load (Pandas)  2 Clean/Preprocess  3 EDA (visualisation)
               4 Train/Test split (+ scaling)  5 Train & evaluate
Output       : Predicted result and model accuracy
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PASS_MARK = 40
TARGET = "final_marks"
NUMERIC = ["study_hours", "attendance", "previous_score", "sleep_hours", "assignments_completed"]
ENCODINGS = {"parental_support": {"Low": 0, "Medium": 1, "High": 2},
             "extra_classes": {"No": 0, "Yes": 1}}
FEATURES = NUMERIC + list(ENCODINGS)
VALID_RANGE = {"study_hours": (0, 16), "attendance": (0, 100), "previous_score": (0, 100),
               "sleep_hours": (2, 12), "assignments_completed": (0, 10), TARGET: (0, 100)}
PASS_C, FAIL_C, BLUE = "#2E9E5B", "#D9534F", "#4C78A8"


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


class StudentPerformanceSystem:
    def __init__(self, log=print):
        self.log = log
        self.figures: "OrderedDict[str, Figure]" = OrderedDict()
        self.summary = ""

    # ------------------------------------------------------------ 1. LOAD
    def load_data(self, path):
        df = pd.read_csv(path)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")
        self.raw = df
        self.log(f"[1] LOAD DATA\n    {df.shape[0]} rows x {df.shape[1]} columns loaded")
        self.log(f"    Columns: {', '.join(df.columns)}\n")
        return df

    # ---------------------------------------------------- 2. CLEAN / PREPROCESS
    def clean_data(self):
        df = self.raw.copy()
        self.log("[2] CLEAN & PREPROCESS")

        n = len(df)
        df = df.drop_duplicates()
        self.log(f"    - Removed {n - len(df)} duplicate rows")

        for c in NUMERIC + [TARGET]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ENCODINGS:
            df[c] = df[c].str.strip().str.title()

        fixed = 0
        for c, (lo, hi) in VALID_RANGE.items():                # impossible value -> treat as missing
            bad = (df[c] < lo) | (df[c] > hi)
            fixed += int(bad.sum())
            df.loc[bad, c] = np.nan
        self.log(f"    - Flagged {fixed} impossible values (e.g. attendance > 100%) as missing")

        na = df[FEATURES + [TARGET]].isna().sum()
        self.log("    - Missing values found: " +
                 (", ".join(f"{k}={v}" for k, v in na[na > 0].items()) or "none"))
        n = len(df)
        df = df.dropna(subset=[TARGET])
        self.log(f"    - Dropped {n - len(df)} rows with no target ({TARGET})")
        for c in NUMERIC:
            df[c] = df[c].fillna(df[c].median())          # median: robust to outliers
        for c in ENCODINGS:
            df[c] = df[c].fillna(df[c].mode()[0])          # mode for categories
        self.log("    - Filled numeric gaps with the median, categorical gaps with the mode")

        self.clean = df.reset_index(drop=True)
        enc = self.clean.copy()
        for c, mapping in ENCODINGS.items():
            enc[c] = enc[c].map(mapping)
            enc[c] = enc[c].fillna(enc[c].mode()[0])
        enc["passed"] = (enc[TARGET] >= PASS_MARK).astype(int)
        self.encoded = enc
        self.log("    - Encoded categories: parental_support (Low/Medium/High -> 0/1/2), "
                 "extra_classes (No/Yes -> 0/1)")
        self.log(f"    - Created label 'passed' (final_marks >= {PASS_MARK})")
        self.log(f"    Clean dataset: {len(enc)} rows\n")
        return enc

    # ------------------------------------------------------------------ 3. EDA
    def eda(self):
        df, enc = self.clean, self.encoded
        self.log("[3] EXPLORATORY DATA ANALYSIS")
        self.log(enc[NUMERIC + [TARGET]].describe().round(2).to_string())
        corr = enc[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
        self.log("\n    Correlation with final_marks:\n" +
                 "\n".join(f"      {pretty(k):<24}{v:+.2f}" for k, v in corr.items()))
        self.log(f"\n    Pass rate: {enc['passed'].mean() * 100:.1f}%\n")

        # (a) distributions
        fig = Figure(figsize=(11, 6.5), layout="constrained")
        for ax, c in zip(fig.subplots(2, 3).ravel(), NUMERIC + [TARGET]):
            ax.hist(enc[c], bins=20, color=BLUE, edgecolor="white")
            ax.set_title(pretty(c))
        fig.suptitle("EDA - Feature Distributions", fontweight="bold")
        self.figures["EDA: Distributions"] = fig

        # (b) correlation heatmap
        cm = enc[FEATURES + [TARGET]].corr()
        fig = Figure(figsize=(8, 6.5), layout="constrained")
        ax = fig.subplots()
        im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cm)), [pretty(c) for c in cm.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(cm)), [pretty(c) for c in cm.columns])
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("EDA - Correlation Heatmap", fontweight="bold")
        self.figures["EDA: Correlation"] = fig

        # (c) relationships
        fig = Figure(figsize=(11, 7), layout="constrained")
        a1, a2, a3, a4 = fig.subplots(2, 2).ravel()
        colors = np.where(enc["passed"] == 1, PASS_C, FAIL_C)
        for ax, col in ((a1, "study_hours"), (a2, "attendance")):
            ax.scatter(enc[col], enc[TARGET], c=colors, s=14, alpha=0.6)
            m, b = np.polyfit(enc[col], enc[TARGET], 1)
            xs = np.linspace(enc[col].min(), enc[col].max(), 50)
            ax.plot(xs, m * xs + b, color="black", lw=1.5)
            ax.axhline(PASS_MARK, ls="--", color="gray", lw=1)
            ax.set_xlabel(pretty(col)); ax.set_ylabel("Final Marks")
            ax.set_title(f"{pretty(col)} vs Marks (green = pass, red = fail)")
        order = list(ENCODINGS["parental_support"])
        a3.boxplot([df.loc[df["parental_support"] == k, TARGET] for k in order])
        a3.set_xticks([1, 2, 3], order)
        a3.set_title("Marks by Parental Support")
        counts = enc["passed"].value_counts().reindex([1, 0]).fillna(0)
        a4.bar(["Pass", "Fail"], counts.values, color=[PASS_C, FAIL_C])
        for i, v in enumerate(counts.values):
            a4.text(i, v, int(v), ha="center", va="bottom")
        a4.set_title("Pass / Fail Count")
        fig.suptitle("EDA - Study Habits vs Performance", fontweight="bold")
        self.figures["EDA: Relationships"] = fig

    # --------------------------------------------- 4. SPLIT + FEATURE SCALING
    def split_and_scale(self, test_size=0.2, seed=42):
        enc = self.encoded
        X, y_reg, y_clf = enc[FEATURES], enc[TARGET], enc["passed"]
        (self.X_train, self.X_test, self.yr_train, self.yr_test,
         self.yc_train, self.yc_test) = train_test_split(
            X, y_reg, y_clf, test_size=test_size, random_state=seed, stratify=y_clf)
        self.scaler = StandardScaler().fit(self.X_train)   # fit on TRAIN only -> no leakage
        self.Xs_train = self.scaler.transform(self.X_train)
        self.Xs_test = self.scaler.transform(self.X_test)
        self.log("[4] TRAIN / TEST SPLIT + FEATURE SCALING")
        self.log(f"    Train: {len(self.X_train)} rows | Test: {len(self.X_test)} rows (stratified 80/20)")
        self.log("    StandardScaler fitted on the training set only (prevents data leakage)\n")

    # -------------------------------------------------- 5. TRAIN AND EVALUATE
    def train(self):
        self.lin = LinearRegression().fit(self.Xs_train, self.yr_train)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            self.Xs_train, self.yc_train)

    def evaluate(self):
        yr_pred = np.clip(self.lin.predict(self.Xs_test), 0, 100)
        yc_pred = self.clf.predict(self.Xs_test)
        m = {
            "accuracy": accuracy_score(self.yc_test, yc_pred),
            "train_accuracy": accuracy_score(self.yc_train, self.clf.predict(self.Xs_train)),
            "precision": precision_score(self.yc_test, yc_pred),
            "recall": recall_score(self.yc_test, yc_pred),
            "f1": f1_score(self.yc_test, yc_pred),
            "r2": r2_score(self.yr_test, yr_pred),
            "mae": mean_absolute_error(self.yr_test, yr_pred),
            "rmse": mean_squared_error(self.yr_test, yr_pred) ** 0.5,
        }
        self.metrics = m
        cm = confusion_matrix(self.yc_test, yc_pred)
        self.log("[5] MODEL EVALUATION")
        self.log("    Logistic Regression (Pass/Fail)")
        self.log(f"      Accuracy  : {m['accuracy'] * 100:.2f}%   (train: {m['train_accuracy'] * 100:.2f}%)")
        self.log(f"      Precision : {m['precision']:.3f}   Recall: {m['recall']:.3f}   F1: {m['f1']:.3f}")
        self.log("    Linear Regression (Final Marks)")
        self.log(f"      R2 score  : {m['r2']:.3f}   MAE: {m['mae']:.2f} marks   RMSE: {m['rmse']:.2f}\n")

        fig = Figure(figsize=(13, 4.6), layout="constrained")
        a1, a2, a3 = fig.subplots(1, 3)
        a1.scatter(self.yr_test, yr_pred, s=16, alpha=0.6, color=BLUE)
        a1.plot([0, 100], [0, 100], "k--", lw=1)
        a1.set_xlabel("Actual Marks"); a1.set_ylabel("Predicted Marks")
        a1.set_title(f"Linear Regression (R2 = {m['r2']:.2f})")
        a2.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a2.text(j, i, v, ha="center", va="center", fontsize=16,
                    color="white" if v > cm.max() / 2 else "black")
        a2.set_xticks([0, 1], ["Fail", "Pass"]); a2.set_yticks([0, 1], ["Fail", "Pass"])
        a2.set_xlabel("Predicted"); a2.set_ylabel("Actual")
        a2.set_title(f"Confusion Matrix (Accuracy = {m['accuracy'] * 100:.1f}%)")
        coef = pd.Series(self.clf.coef_[0], index=[pretty(f) for f in FEATURES]).sort_values()
        a3.barh(coef.index, coef.values, color=[PASS_C if v > 0 else FAIL_C for v in coef])
        a3.set_title("What drives passing?")
        fig.suptitle("Model Results", fontweight="bold")
        self.figures["Results"] = fig

        self.summary = (f"Accuracy: {m['accuracy'] * 100:.1f}%   F1: {m['f1']:.2f}\n"
                        f"R2: {m['r2']:.2f}   MAE: {m['mae']:.2f} marks")
        return m

    # ------------------------------------------------------------ PREDICTION
    def predict(self, values: dict) -> dict:
        row = pd.DataFrame([values])[FEATURES]
        for c, mapping in ENCODINGS.items():
            row[c] = row[c].map(mapping)
        X = self.scaler.transform(row)
        marks = float(np.clip(self.lin.predict(X)[0], 0, 100))
        prob = float(self.clf.predict_proba(X)[0, 1])
        return {"result": "PASS" if prob >= 0.5 else "FAIL", "pass_probability": prob, "marks": marks}

    # ---------------------------------------------------------------- PIPELINE
    def run_all(self, path):
        self.figures = OrderedDict()
        self.load_data(path)
        self.clean_data()
        self.eda()
        self.split_and_scale()
        self.train()
        self.evaluate()
        return {"figures": self.figures, "summary": self.summary}
