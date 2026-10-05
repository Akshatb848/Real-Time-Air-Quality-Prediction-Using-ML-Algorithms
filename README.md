# Air Quality Prediction with Machine Learning

A regression study that estimates hourly carbon monoxide concentration, `CO(GT)`, from the other readings of a multisensor air quality device (metal-oxide sensor responses, reference-analyzer pollutant readings, temperature and humidity). The notebook trains a Random Forest regressor on the UCI/Kaggle Air Quality dataset, tunes it with `RandomizedSearchCV`, and reports MSE and R² on a 20% random hold-out split. It is an offline, single-notebook experiment developed in Google Colab; despite the repository name, it does not include a real-time data pipeline or a deployed model.

## Problem

Reference-grade gas analyzers are expensive, while low-cost metal-oxide sensors are cheap but noisy. The question explored here is how well the true CO concentration measured by a reference analyzer can be estimated from the other columns recorded by the device in the same hour.

- Target: `CO(GT)`, true hourly averaged CO concentration (mg/m³)
- Features: the remaining 12 numeric columns (`PT08.S1(CO)`, `NMHC(GT)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH`)

## Dataset

- Name: Air Quality Data Set (originally the UCI Machine Learning Repository "Air Quality" dataset, De Vito et al.)
- Source used by the notebook: [Kaggle, fedesoriano/air-quality-data-set](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set), file `AirQuality.csv`
- Content: hourly averaged readings from a device deployed in an Italian city, March 2004 to April 2005; `Date`, `Time` and 13 numeric measurement columns
- Format: semicolon-separated with comma decimals (read with `sep=';', decimal=','`)
- Not included in this repository. The first notebook cell downloads it with `kagglehub`, which requires Kaggle API credentials (`~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

The notebook does not print the dataset shape. Its missing-value report shows 114 rows that are empty in every column, including `Date` and `Time`, which come from trailing blank lines in the CSV. Separately, the original dataset encodes missing sensor values as `-200`; the notebook does not treat these as missing (see Limitations).

## Approach

All steps are in `AQI_ML.ipynb`:

1. Download the dataset with `kagglehub` and load `AirQuality.csv` with pandas, dropping the empty `Unnamed` columns.
2. Fill `NaN` values in numeric columns with the column mean (computed on the full dataset), then drop `Date` and `Time`.
3. Split into train and test sets with `train_test_split(test_size=0.2, random_state=42)` (random, not time-ordered).
4. Standardize features with `StandardScaler` fitted on the training set.
5. Baseline: `RandomForestRegressor(n_estimators=100, random_state=42)`.
6. Tuning: `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` and `bootstrap`, with 10 candidates, 5-fold CV on the training set, scored by negative MSE.
7. Evaluate both models on the test split and plot the tuned model's feature importances.

`XGBRegressor` is imported in cell 2 but never trained.

## Results

Numbers below are copied from the executed outputs saved in the notebook. Both are measured on the same 20% random test split (`random_state=42`).

| Model | Notebook cell | Test MSE | Test R² |
|---|---|---|---|
| Random Forest, `n_estimators=100` (baseline) | 5 (exec `[1]`) | 2134.2465 | 0.6165 |
| Random Forest, tuned with RandomizedSearchCV | 6 (exec `[4]`) | 2066.2100 | 0.6287 |

Best hyperparameters printed by the search (cell 6):

```python
{'n_estimators': 250, 'min_samples_split': 5, 'min_samples_leaf': 1,
 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}
```

Cell numbers are 1-based positions in the notebook; the bracketed values are the saved execution counts. The notebook also saves a true-vs-predicted line plot (cell 5) and a feature-importance bar chart (cell 7) as output images. Feature importances are only plotted; they are not used to select features.

## Limitations

- **`-200` missing-value sentinel is not handled.** In this dataset, missing measurements are recorded as `-200` rather than left blank. The notebook only imputes `NaN`, so `-200` values remain in both features and target. This is the most likely reason the MSE is in the thousands while typical CO concentrations are single-digit mg/m³: the error is dominated by sentinel rows, so the reported MSE and R² should not be read as the model's accuracy on real CO values.
- **Random split on time-series data.** Hourly readings are split randomly, so neighbouring hours appear in both train and test. This tends to overstate performance compared with a chronological split.
- **Imputation before the split.** Column means are computed on the full dataset before splitting, a minor form of test-set leakage. The 114 imputed rows are entirely empty and would be better dropped.
- **Co-located reference readings used as features.** Several features (`NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`) come from reference analyzers like the target and often share its missing (`-200`) hours, so the model can partly learn the missingness pattern rather than the sensor-to-pollutant relationship.
- **Single split, no simple baseline.** Each model is scored on one test split, with no repeated evaluation and no comparison against a linear model or mean predictor.
- **Notebook state.** Execution counts are out of order (cells were run non-linearly in Colab), the last cell is empty, and the data path is hard-coded to `/kaggle/input/air-quality-data-set/AirQuality.csv`. Results may differ slightly when re-run top to bottom or with different library versions.

## How to run

Dependencies were verified to install and import on Python 3.11.

```bash
git clone https://github.com/Akshatb848/real-time-air-quality-prediction-using-ml-algorithms.git
cd real-time-air-quality-prediction-using-ml-algorithms

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook AQI_ML.ipynb
```

Before running, configure Kaggle credentials so `kagglehub` can download the dataset. Outside Kaggle/Colab, replace the hard-coded path in cells 4 and 5 (`/kaggle/input/air-quality-data-set/AirQuality.csv`) with the path returned by `kagglehub.dataset_download(...)`, for example `os.path.join(dataset_path, 'AirQuality.csv')`.

## Project structure

```
.
├── AQI_ML.ipynb       # Data download, preprocessing, Random Forest training, tuning and evaluation
├── requirements.txt   # Python dependencies
└── README.md
```
