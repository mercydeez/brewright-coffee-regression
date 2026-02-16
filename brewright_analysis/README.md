# BrewRight Coffee — Store Revenue Regression Analysis

A complete machine learning regression analysis investigating the factors that drive monthly revenue for **BrewRight Coffee** store locations. This project covers simple and multiple linear regression, Ridge and Lasso regularization, feature engineering, and data-driven business recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Assignment Breakdown](#assignment-breakdown)
- [Key Findings](#key-findings)
- [Models & Techniques](#models--techniques)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)

---

## Overview

BrewRight Coffee's marketing team hypothesizes that **local marketing spend is the primary driver of monthly revenue**. This analysis tests that claim using regression modeling on data from 150 store locations, progressing from simple linear regression through regularized models, and ultimately delivering actionable business recommendations.

## Dataset

| Property | Value |
|----------|-------|
| **File** | `data/brewright_stores.csv` |
| **Observations** | 150 stores |
| **Features** | 16 predictors + 1 target |
| **Target Variable** | `monthly_revenue_K` (monthly revenue in $K) |

### Predictor Variables

| Feature | Description |
|---------|-------------|
| `marketing_spend_K` | Local marketing spend ($K) |
| `store_sqft` | Store size (sq ft) |
| `avg_daily_foot_traffic` | Average daily visitors |
| `num_employees` | Number of employees |
| `neighborhood_median_income_K` | Neighborhood median income ($K) |
| `drive_through` | Has drive-through (0/1) |
| `competitor_count` | Number of nearby competitors |
| `yelp_rating` | Yelp rating (1–5) |
| `avg_latte_price` | Average latte price ($) |
| `parking_spots` | Number of parking spots |
| `num_menu_items` | Menu item count |
| `seating_capacity` | Indoor seating capacity |
| `wifi_speed_mbps` | WiFi speed (Mbps) |
| `distance_to_nearest_atm_miles` | Distance to nearest ATM (miles) |
| `avg_barista_experience_months` | Average barista experience (months) |
| `loyalty_program` | Has loyalty program (0/1) |

## Project Structure

```
brewright_analysis/
├── README.md                    # This file
├── assignment_workbook.ipynb    # Complete analysis notebook (Q1–Q21)
├── assignment.pdf               # Original assignment worksheet
├── data/
│   └── brewright_stores.csv     # Dataset (150 stores, 17 columns)
└── outputs/                     # Saved visualizations
    ├── q1_scatter_plot.png
    ├── q2_slr_fit.png
    ├── q6_ols_coefficients.png
    ├── q10_coefficient_comparison.png
    ├── q11_summary_table.png
    └── q12_lasso_importance.png
```

## Assignment Breakdown

### Part A — Simple Linear Regression (Q1–Q3)
- Scatter plot of marketing spend vs. revenue (r = 0.67)
- SLR model: `revenue = 107.90 + 2.78 × marketing_spend`
- Prediction for $15K marketing spend → **$149.55K** revenue

### Part B — Multiple Linear Regression (Q4–Q7)
- 5-feature MLR (R² = 0.80) vs. 16-feature MLR (R² = 0.93)
- 80/20 train-test split with `random_state=42`
- OLS coefficient analysis and VIF-based multicollinearity diagnostics
- High VIF detected: `store_sqft` (91.1), `seating_capacity` (78.4)

### Part C — Regularization (Q8–Q11)
- **Ridge** (RidgeCV, α = 1.0): Test R² = 0.946 — shrinks but keeps all 16 features
- **Lasso** (LassoCV, α = 0.35): Test R² = 0.948 — eliminates 5 features
- Coefficient comparison (OLS vs. Ridge vs. Lasso) with visualizations
- Model comparison summary table

### Part D — Business Recommendations (Q12–Q14)
- Top 3 actionable recommendations from Lasso feature importance
- Analysis of eliminated features (WiFi speed, loyalty program)
- Model selection recommendation for production deployment

### Part E — Critical Thinking (Q15–Q21)
- Extreme regularization (α = 0.00001 vs. α = 1000) — bias-variance tradeoff
- College-town extrapolation risk assessment
- Confidence vs. prediction intervals
- Interaction effect testing (marketing × drive-through) — **not supported**
- Feature selection stability across random seeds
- Budget-constrained 5-feature model (Test R² = 0.79)
- Impact of tripling dataset size on model performance

## Key Findings

1. **Marketing spend is the strongest predictor** (standardized Lasso coef: +18.1), confirming the marketing team's hypothesis
2. **Drive-through stores earn ~$11.2K more** per month (OLS coefficient), the second-largest effect
3. **Lasso eliminates 5 features** as noise: `wifi_speed_mbps`, `avg_barista_experience_months`, `loyalty_program`, `seating_capacity`, `avg_latte_price`
4. **All three models perform similarly** (Test R² ≈ 0.94–0.95), indicating a well-behaved dataset
5. **Marketing does NOT work differently** at drive-through stores (interaction term eliminated by Lasso)

## Models & Techniques

| Technique | Implementation |
|-----------|---------------|
| Simple Linear Regression | `sklearn.linear_model.LinearRegression` |
| Multiple Linear Regression | `sklearn.linear_model.LinearRegression` |
| Ridge Regression | `sklearn.linear_model.RidgeCV` (5-fold CV) |
| Lasso Regression | `sklearn.linear_model.LassoCV` (5-fold CV) |
| Feature Scaling | `sklearn.preprocessing.StandardScaler` |
| VIF Analysis | `statsmodels.stats.outliers_influence.variance_inflation_factor` |
| Train/Test Split | 80/20, `random_state=42` |

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- `pandas` — data manipulation
- `numpy` — numerical computing
- `matplotlib` — plotting
- `seaborn` — statistical visualization
- `scikit-learn` — ML models and preprocessing
- `statsmodels` — VIF calculation
- `jupyter` — notebook environment

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/brewright-coffee-regression.git
   cd brewright-coffee-regression/brewright_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook assignment_workbook.ipynb
   ```

4. Run all cells sequentially (Cell → Run All) to reproduce the full analysis.

---

*Built with Python, scikit-learn, and statsmodels.*
