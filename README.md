# Mindshift_challange

This README file provides a comprehensive overview of the **MindShift Analytics Haul-Mark Challenge** solution. The project utilizes an ensemble of gradient-boosted decision trees to predict fuel consumption ($acons$) based on vehicle telemetry and operational data.

---

## 📌 Project Overview
This repository contains a high-performance machine learning pipeline designed to predict fuel consumption for heavy-duty dumpers. The solution addresses the challenge by combining environmental factors (altitude, lift) with operational metrics (speed, distance) using a weighted ensemble of **CatBoost**, **LightGBM**, and **XGBoost**.

### Key Features
* **Ensemble Modeling**: A weighted combination of three state-of-the-art GBMs.
* **Automated Hyperparameter Tuning**: Integrated **Optuna** optimization specifically for CatBoost parameters.
* **Secondary Analytics**: Methodology for calculating a "Route-Level Fuel Benchmark" independent of dumper efficiency.
* **Operational Insights**: Generation of efficiency reports for both vehicles and operators.

---

## 🛠 Model Architecture
The pipeline utilizes a 5-fold cross-validation strategy. The final prediction is a weighted ensemble calculated as follows:
$$Final\ Prediction = 0.45(CatBoost) + 0.40(LightGBM) + 0.15(XGBoost)$$

### 1. Data Pipeline
* **Fetch**: Aggregates training summaries and merges them with cached telemetry features.
* **Feature Prep**: Handles categorical missing values as "UNKNOWN" and converts them to string format.
* **Encoding**: Implements custom transformation logic for each model, including native categorical support for CatBoost and LightGBM, and label encoding for XGBoost.

### 2. Secondary Outputs & Methodology
As per the project requirements, the system generates several analytical outputs:
* **Route-Level Fuel Benchmark**: An LGBM-based estimate of expected fuel consumption based on topography (distance, lift, altitude).
* **Efficiency Component**: Captures dumper and operator variation by analyzing the "fuel wasted" (the residual between actual and benchmark fuel).

---

## 📂 File Structure
* `Main_output.py`: The core production script focused on generating the ensemble submission.
* `Secondary_output.py`: An extended version including the `ReportService` for efficiency analytics and benchmark reports.

---

## 🚀 Usage

### Execution
To generate the competition submission and all operational reports, run:
```bash
python Secondary_output.py
```

### Outputs
The following files will be generated in the working directory:
* `submission_ensemble.csv`: Final competition predictions.
* `benchmark_full.csv`: Full dataset with route benchmarks and "fuel wasted" calculations.
* `dumper_eff.csv`: Average efficiency rating per vehicle.
* `operator_eff.csv`: Average efficiency rating per operator.

---

## 📊 Feature Set
The models utilize the following key features:
* **Categorical**: `vehicle`, `shift`, `operator_id`.
* **Numerical**: `mean_speed`, `max_speed`, `total_pings`, `distance_travelled`, `net_lift`, and various altitude metrics.
