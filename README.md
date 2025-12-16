#OECD Decoupling Classification
This project implements a reproducible machine-learning pipeline to predict whether OECD country–year observations exhibit economic–environmental decoupling, defined as positive GDP growth combined with declining CO₂ emissions.

The full pipeline runs from data loading to evaluation via a single entry point (python main.py) and saves all outputs to disk.

#Research Question

Can macroeconomic and environmental indicators predict decoupling outcomes at the country–year level?

#Data
- Source: OECD country–year panel
- Time span: 2010–2023
- Unit of observation: country–year
- Target variable: binary indicator of decoupling

The target equals 1 if GDP growth is positive and total CO₂ emissions decline relative to the previous year.

#Features
The final feature set includes:
- GDP growth
- CO₂ emissions per capita
- Total CO₂ emissions

These variables are consistently available across countries and years and capture both economic performance and environmental pressure.

#Models and Baselines
The pipeline estimates:
- Logistic regression (with standardized features and validation-based threshold selection)
- Shallow decision tree (for non-linear patterns and interpretability)
- Performance is benchmarked against:
- Majority-class baseline
- Carry-forward baseline (previous year’s outcome)

#Project structure
├── data/
│   └── processed/
│       └── oecd_panel_2010_2023.csv
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluation.py
│   └── interpretation.py
├── results/
│   ├── metrics/
│   └── figures/
├── main.py
└── README.md

#How to run
From the repository root:  python main.py

Running this command:
- trains all models,
- evaluates performance on the 2023 test set,
- saves metrics to results/metrics/,
- saves figures (confusion matrices and feature importance) to results/figures/.

#Data Dictionary
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

Note: The variable delta_co2 is used exclusively to define the decoupling outcome and is deliberately excluded from the feature set to avoid target leakage.

#Reproducibility
All results reported in the accompanying report are generated programmatically via main.py. No manual steps or notebooks are required.
