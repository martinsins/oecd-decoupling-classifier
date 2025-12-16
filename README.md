# OECD Decoupling Classification

This project implements a fully reproducible machine-learning pipeline to classify whether OECD country–year observations exhibit **economic–environmental decoupling**, defined as positive GDP growth combined with declining CO₂ emissions.

The analysis is designed as an applied data-science exercise with an emphasis on **modular code structure, chronological validation, baseline comparisons, and reproducibility**. All core results reported in the accompanying paper are generated programmatically from a single entry point.

---

## Research Question

**Can contemporaneously observable macroeconomic and environmental indicators predict economic–environmental decoupling at the country–year level?**

---

## Data

- **Unit of observation:** OECD country–year  
- **Time span:** 2010–2023  
- **Sources:**  
  - World Bank, *World Development Indicators* (GDP growth)  
  - Our World in Data, *CO₂ and Greenhouse Gas Emissions*  
  
Details on raw data acquisition and source URLs are provided in `notebooks/01_build_panel.ipynb`.


### Outcome Variable
A binary indicator `decoupled` is constructed such that:
- `decoupled = 1` if GDP growth is strictly positive **and** total CO₂ emissions decline relative to the previous year  
- `decoupled = 0` otherwise  

The year-over-year change in emissions (`delta_co2`) is used **exclusively** to define the outcome and is excluded from the feature set to avoid target leakage.

---

## Features

The final feature set includes:
- GDP growth  
- CO₂ emissions per capita  
- Total CO₂ emissions  

These variables are consistently available across countries and years and jointly capture economic performance, emissions intensity, and emissions scale.

---

## Models and Baselines

The pipeline estimates and compares:

### Supervised Models
- **Logistic regression**
  - Standardized inputs
  - Classification threshold tuned on the 2022 validation set to maximize the F1-score
- **Shallow decision tree**
  - Captures non-linear and threshold-based patterns
  - Constrained depth for interpretability

### Baselines
- **Majority-class baseline**
- **Carry-forward baseline** (predicts the previous year’s outcome)

Model performance is evaluated on a **held-out 2023 test set** using accuracy and the F1-score.

---

## Project Structure

```text
oecd-decoupling-classifier/
├── src/
│   ├── data_loader.py        # Data loading and chronological splitting
│   ├── models.py             # Model definitions
│   ├── evaluation.py         # Metrics, threshold selection, diagnostics
│   └── interpretation.py     # Feature importance and interpretation tools
├── notebooks/
│   ├── 01_build_panel.ipynb      # Data construction and validation
│   └── 02_appendix_figures.ipynb # Supplementary figures for the appendix
├── data/
│   ├── raw/                   # Raw source files (ignored by Git; not all used)
│   └── processed/             # Generated panel data (ignored by Git)
├── results/
│   ├── metrics/               # Generated evaluation outputs (ignored by Git)
│   └── figures/               # Generated figures (ignored by Git)
├── main.py                    # End-to-end training and evaluation entry point
├── environment.yml            # Conda environment specification
├── .gitignore
└── README.md
```

--- 

## How to Run the Pipeline

From the repository root:
`python main.py`  

Running this command:
- trains all models on 2010–2021 data,
- tunes the logistic classification threshold on 2022,
- evaluates performance on the 2023 test set,
- saves metrics to `results/metrics/`,
- saves figures (confusion matrices and feature importances) to `results/figures/`.

Generated outputs are intentionally not tracked by Git and can be reproduced at any time.

--- 

## Notebooks

Notebooks are included for transparency and documentation but are not required to reproduce the main results:
- `01_build_panel.ipynb` documents data construction and cleaning steps.
- `02_appendix_figures.ipynb` generates supplementary figures reported in Appendix A of the paper.

All results in the main text are produced via `main.py`.

---

## Data Dictionary
| Variable       | Description                                   | Unit / Type   |
| -------------- | --------------------------------------------- | ------------- |
| country        | Country name                                  | string        |
| iso3           | ISO-3 country code                            | string        |
| year           | Calendar year                                 | integer       |
| gdp_growth     | Annual GDP growth rate                        | percent       |
| co2            | Total CO₂ emissions                           | MtCO₂         |
| co2_per_capita | CO₂ emissions per capita                      | tCO₂ / person |
| delta_co2      | Year-over-year change in total CO₂ emissions  | MtCO₂         |
| decoupled      | 1 if GDP growth > 0 and delta_co2 < 0, else 0 | binary        |

---

## Reproducibility

All results reported in the accompanying report are generated programmatically via `main.py`.

Raw data files are intentionally excluded from the repository via `.gitignore`.  
The data sources and download logic are documented explicitly in the notebook `notebooks/01_build_panel.ipynb`, which retrieves the original public datasets (OECD, OWID, World Bank) and constructs the final processed panel.
