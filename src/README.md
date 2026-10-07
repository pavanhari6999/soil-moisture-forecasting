# Analysis scripts

The src directory contains the reproducibility implementation for the paper.

Recommended execution order:
1. 01_quality_control.py
2. 02_feature_engineering.py
3. 03_primary_forecasting.py
4. 04_bootstrap_uncertainty.py
5. 05_robustness_analysis.py
6. 06_generate_figures.py

The scripts expect the original AgriDataValue source files to be downloaded locally according to data/README.md.
