# ML_For_Finance_Project-LiessGröli-327521-AdrienAïtLalim-326588

ML-For-Finance-Project/
│
├── data/
│   ├── raw/
│   │   ├── crsp_monthly.csv
│   │   ├── compustat.csv
│   │   ├── jkp_factors.csv
│   │   └── chen_zimmerman.csv
│   │
│   ├── processed/
│   │   ├── crsp_clean.parquet
│   │   ├── features_panel.parquet
│   │   ├── model_dataset.parquet
│   │   └── predictions.parquet
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── portfolios/
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess_crsp.py
│   ├── preprocess_compustat.py
│   ├── preprocess_factors.py
│   ├── build_features.py
│   ├── build_targets.py
│   ├── models.py
│   ├── rolling_train.py
│   ├── backtest.py
│   ├── metrics.py
│   └── plots.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_checks.ipynb
│   └── 03_results_analysis.ipynb
│
├── main.py
├── requirements.txt
└── README.md
