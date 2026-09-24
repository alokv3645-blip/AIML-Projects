# AI / ML Projects Studio

Two complete machine-learning projects with a desktop GUI.

| Project | Objective | Models | Output |
|---|---|---|---|
| 1. Student Performance Prediction | Predict pass/fail and estimate marks from study habits | Logistic + Linear Regression | Predicted result + model accuracy |
| 2. House Price Prediction | Predict price from area, location, etc. | Linear, Ridge, Random Forest, Gradient Boosting | Predicted price + comparison graph |

## Run
```bash
pip install -r requirements.txt
python app.py      # GUI (tkinter ships with Python; on Linux: sudo apt install python3-tk)
python main.py     # command line, saves graphs + reports to ./outputs
```

## How each requirement is covered

### Project 1 - `student_performance.py`
| Requirement | Where |
|---|---|
| Load dataset using Pandas | `load_data()` |
| Clean and preprocess | `clean_data()` - duplicates, impossible values, missing values (median/mode), categorical encoding |
| EDA using visualisation | `eda()` - distributions, correlation heatmap, scatter + trend lines, box plot, pass/fail counts |
| Split into train/test | `split_and_scale()` - stratified 80/20 |
| Feature scaling | `StandardScaler`, fitted on the training set only (no data leakage) |
| Linear & Logistic Regression | `train()` - marks (linear), pass/fail (logistic) |
| Evaluate accuracy | `evaluate()` - accuracy, precision, recall, F1, confusion matrix, R2, MAE, RMSE |
| Output: predicted result + accuracy | `predict()` + GUI result panel |

### Project 2 - `house_price.py`
| Requirement | Where |
|---|---|
| Load dataset using Pandas | `load_data()` |
| Handle missing values | `clean_data()` - median/mode imputation, invalid rows, outliers (3xIQR) |
| Feature engineering | `engineer_and_encode()` - total_rooms, area_per_bedroom, is_new, log_area |
| Encode categorical variables | furnishing -> ordinal, location -> one-hot |
| Regression models | Linear, Ridge, Random Forest, Gradient Boosting (5-fold CV picks the best) |
| Metrics (R2, MAE) | `evaluate()` - R2, MAE, RMSE for every model |
| Output: price + comparison graph | `predict()`; "Actual vs Predicted" scatter + line graph |

## Use your own data
Datasets in `data/` are generated samples (with deliberate missing values, duplicates
and outliers). Use **Browse CSV** in the GUI to load your own file with the same columns:

* Student: `study_hours, attendance, previous_score, sleep_hours, assignments_completed, parental_support (Low/Medium/High), extra_classes (No/Yes), final_marks`
* House: `area_sqft, bedrooms, bathrooms, age_years, parking, distance_to_center_km, location, furnishing, price_lakhs`

Pass mark is `PASS_MARK = 40` in `student_performance.py`.

## Files
```
app.py                  GUI (Tkinter + embedded matplotlib)
main.py                 CLI runner + exports
student_performance.py  Project 1
house_price.py          Project 2
data_generator.py       sample data
data/  outputs/         datasets, saved graphs and reports
```
