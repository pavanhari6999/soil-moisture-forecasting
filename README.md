# Beyond Persistence: Soil Moisture Forecasting

Reproducibility repository for **Beyond Persistence: A Chronological Benchmark of Machine Learning for Short-Term Soil Moisture Forecasting Across Station-Year Datasets**.

**Author:** Daggumati Pavan Hari Krishna  
**Affiliation:** Department of Computer Science and Engineering, K L Deemed to be University, Vaddeswaram, Andhra Pradesh, India.

## Research objective

This study evaluates whether machine learning provides incremental predictive value beyond a persistence baseline for short-term soil-moisture forecasting.

### Models
- Persistence baseline
- Ridge regression
- Random Forest
- Histogram-based Gradient Boosting
- Controlled LSTM robustness baseline

### Forecast horizons
1 h, 6 h, 12 h, and 24 h.

### Validation and uncertainty
Chronological expanding-window validation is used to avoid random temporal leakage. Temporal block bootstrap resampling with 2,000 replicates is used to quantify uncertainty in ML-minus-persistence absolute-error differences.

## Dataset

Source: **AgriDataValue – Environmental Data Useful for Smart Irrigation**  
Zenodo DOI: https://doi.org/10.5281/zenodo.18959383

The source contains two monitoring stations across four station-year datasets. The raw dataset contained 22,111 observations; 22,050 were retained after quality control.

The original raw dataset is **not redistributed in this repository** because the current Zenodo record does not state an explicit redistribution license. See data/README.md.

## Repository structure

- src/ — reproducibility and analysis scripts
- data/README.md — dataset provenance and acquisition instructions
- results/ — final numerical result tables
- figures/ — manuscript figures
- paper/ — manuscript source and bibliography
- requirements.txt — software environment
- CITATION.cff — citation metadata

## Reproduction

1. Download the original dataset from Zenodo.
2. Place the required source files under data/raw/.
3. Install dependencies with: pip install -r requirements.txt
4. Run the scripts in src/ in numerical order.
5. Compare generated results with the committed result tables.

## Research integrity

This repository preserves the reported negative result. Results must not be selectively changed to make machine learning outperform persistence.

## Citation

Please cite both the research paper and the original AgriDataValue dataset when using these materials.
