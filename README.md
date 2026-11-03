# Marketing Mix Modeling & Budget Optimization for a Pharma Brand

> Bayesian marketing mix model (adstock + Hill saturation) on 5.7M rows of multi-channel data, followed by a constrained budget optimizer and an Excel scenario tool.
> Built with **Databricks (PySpark + Delta Lake)**, **SQL**, **Python (PyMC, scipy)** and **Excel**.

---

## 1. Business Problem

A pharma brand (**Brand X**) spends about **$160M per year** across 7 channels: rep calls, email, display, paid search, DTC TV, speaker programs and samples. Leadership cannot answer:

- Which channels actually drive prescriptions (TRx)?
- Where is spend already saturated, and where is there room to grow?
- How should the same budget be reallocated for the highest return?

**Question:** What is the return on each channel, and what budget mix maximizes incremental TRx at constant total spend?

## 2. Solution Summary

| Step | What was done | Tool |
|---|---|---|
| 1 | Reconciled call-level and daily digital logs to the weekly DMA panel | Databricks, PySpark, SQL |
| 2 | Cleaned the panel: duplicates, missing spend, outliers, negative adjustments | PySpark, SQL |
| 3 | Fit a Bayesian MMM with adstock and Hill saturation per channel | Python (PyMC) |
| 4 | Validated on a holdout period (train 2020-2023, test 2024) | Python |
| 5 | Compared recovered parameters to the known ground truth | Python |
| 6 | Optimized the budget under channel min/max constraints | Python (scipy) |
| 7 | Built an Excel scenario tool and a 1-page summary | Excel, PDF |

## 3. Key Results

> Replace the placeholders with your actual numbers after running the notebooks.

- Holdout (2024) accuracy: **MAPE = [x]%**, **R² = [0.xx]**.
- Channel ROI ranking (incremental TRx per $1M): **[channel 1] > [channel 2] > ... > [channel 7]**.
- Saturated channels: **[e.g., email]**. Underinvested channels: **[e.g., search, speaker programs]**.
- Reallocating the same **$[160]M** budget projects **+[X]% incremental TRx** (90% credible interval **[x–x]%**).
- Parameter recovery: **[x] of 7** channel decay rates and **[x] of 7** channel response curves recovered within **[x]%** of the ground truth.

## 4. Data

The data is **fully synthetic**, built to mimic pharma commercial data. No real company or patient information is used.

| File | Rows | Description |
|---|---|---|
| `dma_week_panel.csv.gz` | 164,923 | 3 brands x 210 DMAs x 261 weeks (2020-2024): TRx, NRx, net sales, volume and spend for 7 channels, covariates |
| `rep_call_log.csv.gz` | 2,128,530 | One row per rep call: date, HCP, rep, call type (in-person, virtual, phone), duration, samples left |
| `daily_digital_log.csv.gz` | 3,453,030 | Daily email, display and search volume, spend, delivered/viewable and engagements by DMA |
| `dma_master.csv` | 210 | Population, 65+ share, diabetes prevalence, urbanicity, target HCP count |
| `budget_constraints_brand_x.csv` | 7 | 2024 spend per channel with 0.5x to 2x limits |
| `ground_truth_parameters.csv` | 21 | True model parameters (used only for validation in Step 5) |
| `generate_mmm_dataset.py` | n/a | Regenerates all files (seed 11) |

**Covariates in the panel:** average copay, formulary coverage, WAC price, competitor pressure index, competitor launch flag (Sept 2022), holiday weeks and COVID period.

## 5. Reconciliation & Cleaning Log

### Step 1: Reconciliation

| Check | Method | Result |
|---|---|---|
| Rep calls: call log vs. panel | Count call rows by brand, DMA, week; compare to `rep_calls` | **[match / differences]** |
| Digital: daily log vs. panel | Sum daily volume and spend to week; compare to panel | **[within x%]** |

### Step 2: Cleaning

| # | Issue | Approximate rows affected | Fix |
|---|---|---|---|
| 1 | Duplicate DMA-week rows | ~490 | Dropped on brand, DMA, week |
| 2 | Missing display spend and impressions (vendor outage) | ~1,950 | Rebuilt from the daily digital log where available; otherwise interpolated within DMA |
| 3 | Missing email spend | ~640 | Rebuilt from the daily digital log |
| 4 | Extreme TRx values (about 10x neighbors) | 15 | Detected with rolling median and z-score; replaced with the DMA's rolling median |
| 5 | Negative TRx (adjustments/returns) | 20 | Set to 0 and flagged |

## 6. Methodology

### Model
For each DMA-week, TRx is modeled as a baseline multiplied by marketing lift:

```
TRx = Baseline x (1 + sum over channels of  beta_c x Hill( Adstock_c ))
Adstock_c(t) = x_c(t) + decay_c x Adstock_c(t-1)
Hill(a)      = a^s / (a^s + k^s)
```

- **Baseline:** DMA-level scale, trend, Fourier seasonality, holiday, COVID, competitor pressure, competitor launch, formulary coverage and price.
- **Per channel:** `beta` (max lift), `decay` (carryover), `s` (shape) and `k` (half-saturation point).
- **Bayesian estimation in PyMC** with weakly informative priors (decay in 0 to 1, beta positive).
- **Partial pooling** across regions (hierarchical) as an extension to the pooled model.

### Validation
- Train on 2020-2023, test on the 2024 holdout (52 weeks).
- Metrics: MAPE, R², and residual diagnostics (autocorrelation, heteroscedasticity).
- Convergence checks: R-hat, effective sample size, posterior predictive checks.

### Ground-truth comparison (Step 5)
- Open `ground_truth_parameters.csv` only after the model is final.
- Compare decay, shape and the implied **response curves and ROI** per channel. Raw `beta` and `k` are partly interchangeable, so curve-level comparison is the fair test.

### Budget optimization
- **Objective:** maximize total incremental TRx for Brand X.
- **Constraints:** total 2024 budget fixed; each channel between 0.5x and 2.0x of its 2024 spend (`budget_constraints_brand_x.csv`).
- **Solver:** `scipy.optimize.minimize` (SLSQP) on posterior mean response curves; uncertainty from re-running across posterior draws.

## 7. Deliverables

| Deliverable | Location |
|---|---|
| Databricks notebooks | `notebooks/` |
| SQL scripts | `sql/` |
| Excel scenario tool | `excel/mmm_scenario_tool.xlsx` |
| Charts (response curves, contribution decomposition, holdout fit) | `images/` |
| 1-page executive summary | `docs/executive_summary.pdf` |

**Excel scenario tool:** channel spend inputs, response curves driven by the fitted parameters, current vs. optimized budget, and incremental TRx and revenue under each scenario. Formulas are live so the user can test their own allocations.

## 8. Repository Structure

```
pharma-mmm-budget-optimization/
├── data/
│   ├── dma_week_panel.csv.gz
│   ├── rep_call_log.csv.gz
│   ├── daily_digital_log.csv.gz
│   ├── dma_master.csv
│   ├── budget_constraints_brand_x.csv
│   ├── ground_truth_parameters.csv
│   └── generate_mmm_dataset.py
├── notebooks/
│   ├── 01_reconciliation.py
│   ├── 02_data_cleaning.py
│   ├── 03_mmm_pymc.py
│   ├── 04_holdout_validation.py
│   ├── 05_ground_truth_comparison.py
│   └── 06_budget_optimizer.py
├── sql/
│   ├── 01_reconcile_calls.sql
│   ├── 02_reconcile_digital.sql
│   └── 03_clean_panel.sql
├── excel/
│   └── mmm_scenario_tool.xlsx
├── images/
├── docs/
│   └── executive_summary.pdf
└── README.md
```

## 9. Assumptions & Limitations

- Data is synthetic. The patterns demonstrate the method, not real-world results.
- MMM is **observational**. Correlated channel spend (multicollinearity) and spend that responds to sales (endogeneity) can bias estimates. In practice, validate with geo-experiments or holdout tests.
- The optimizer assumes response curves hold outside the observed spend range. The 0.5x to 2x limits keep recommendations near the data.
- The ground-truth comparison is possible only because the data is simulated; real projects do not have this.
- Results apply to Brand X only; cross-brand effects are not modeled.

## 10. How to Run

1. Upload the data files to a Databricks Volume. Spark reads gzipped CSV directly:
   ```python
   panel = spark.read.csv("/Volumes/main/default/raw/dma_week_panel.csv.gz", header=True, inferSchema=True)
   ```
2. Save each table as a Delta table.
3. Run notebooks 01 to 06 in order. Install packages in the first cell:
   ```python
   %pip install pymc arviz
   ```
4. Load the fitted parameters from notebook 05 into `excel/mmm_scenario_tool.xlsx`.

To regenerate the data:

```bash
python data/generate_mmm_dataset.py
```

## 11. Tech Stack

- **Databricks:** PySpark, Delta Lake, notebooks, Volumes
- **SQL:** Spark SQL for log reconciliation and panel cleaning
- **Python:** pandas, NumPy, PyMC, ArviZ, scipy.optimize, matplotlib
- **Excel:** scenario tool with live formulas

## 12. Author

**Rahul** | [GitHub](https://github.com/rahulpatil0001) | [LinkedIn](https://linkedin.com/in/rahulpatil0001)
