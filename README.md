# SDSS Classification System — Stars, Galaxies & Quasars

A machine learning project that classifies astronomical objects observed by the **Sloan Digital Sky Survey (SDSS), Data Release 14**, into one of three classes: **Star**, **Galaxy**, or **Quasar (QSO)**.

The full pipeline (data exploration → feature engineering → model comparison → hyperparameter tuning → evaluation) is implemented in `SDSS_Full_System_XGBoost.ipynb`.

---

## 📂 Repository Contents

| File | Description |
|---|---|
| `SDSS_Full_System_XGBoost.ipynb` | Main notebook — the entire data science workflow |
| `Skyserver_SQL2_27_2018 6_51_39 PM.csv` | Raw dataset pulled from SDSS CasJobs (10,000 rows, 18 columns) |
| `sdss_data.csv` | Working dataset — read in at the start of the notebook, and **overwritten at the end** with the final feature-engineered, scaled dataset (used for the hyperparameter-tuning step) |

> ⚠️ **Note:** Both the raw input and the processed output currently share the filename `sdss_data.csv` inside the notebook (cell that reads it in vs. cell that does `sdss.to_csv('sdss_data.csv')`). If you re-run the notebook from scratch, keep a backup of the original raw CSV, otherwise it gets overwritten by the processed version on the final run.

---

## 🛰️ About the Data

Data comes from the SDSS public archive, queried via **CasJobs** (SQL interface) joining the `PhotoObj` (imaging) and `SpecObj` (spectroscopy) tables.

**Original features (18):**
`objid, ra, dec, u, g, r, i, z, run, rerun, camcol, field, specobjid, class, redshift, plate, mjd, fiberid`

- `u, g, r, i, z` — brightness magnitudes captured at 5 different wavelength filters
- `ra, dec` — celestial coordinates (right ascension / declination)
- `redshift` — proxy for an object's distance from Earth
- `class` — the target label: `STAR`, `GALAXY`, or `QSO`
- `objid, specobjid, run, rerun, camcol, field` — identifiers/instrument metadata, dropped before modeling since they carry no real predictive signal

Class distribution: ~50% Galaxy, ~40% Star, ~10% QSO.

---

## 🧰 Libraries Used & Why

| Library | Used For | Purpose in This Project |
|---|---|---|
| **pandas** | Data loading & manipulation | Reading the CSV, dropping irrelevant columns, building summary tables, merging PCA outputs back into the dataframe |
| **numpy** | Numerical operations | Underlying array math, computing accuracy (`(preds == y_test).sum()`), unique class counts |
| **matplotlib** | Plotting | Base plotting engine for all charts (histograms, heatmaps, bar charts) |
| **seaborn** | Statistical visualization | `distplot` (redshift distribution per class), `boxenplot`/letter-value plot (dec by class), `heatmap` (correlation between u/g/r/i/z), `lmplot` (ra vs dec scatter by class) |
| **scikit-learn** (`sklearn`) | Core ML toolkit | Multiple sub-modules used: |
| &nbsp;&nbsp;↳ `train_test_split`, `cross_val_score`, `cross_val_predict` | Model validation | Splitting data into train/test, and doing 10-fold cross-validation for a more reliable accuracy estimate |
| &nbsp;&nbsp;↳ `KNeighborsClassifier` | Baseline model | K-Nearest Neighbors classifier — one of 5 algorithms compared |
| &nbsp;&nbsp;↳ `GaussianNB` | Baseline model | Naive Bayes classifier (paired with `MaxAbsScaler` since it assumes normally-distributed features) |
| &nbsp;&nbsp;↳ `RandomForestClassifier` | Baseline/competitor model | Ensemble tree-based classifier compared against XGBoost |
| &nbsp;&nbsp;↳ `SVC` | Baseline model | Support Vector Machine classifier |
| &nbsp;&nbsp;↳ `SGDClassifier` | Imported (linear model) | Available for use as an additional linear baseline |
| &nbsp;&nbsp;↳ `PCA` | Dimensionality reduction | Compresses the 5 correlated `u,g,r,i,z` magnitude columns into 3 principal components (`PCA_1, PCA_2, PCA_3`), reducing redundancy/training time |
| &nbsp;&nbsp;↳ `LabelEncoder` | Preprocessing | Converts the text target column (`STAR`/`GALAXY`/`QSO`) into numeric labels for the models |
| &nbsp;&nbsp;↳ `MinMaxScaler` / `MaxAbsScaler` | Preprocessing | Scales numeric features into a fixed range so distance-based and gradient-based algorithms train properly/converge faster |
| &nbsp;&nbsp;↳ `confusion_matrix`, `precision_score`, `recall_score`, `f1_score` | Evaluation metrics | Measures how well the final model distinguishes the 3 classes |
| **xgboost** (`XGBClassifier`) | Main/final model | Gradient-boosted decision trees — the best-performing classifier in this project, later hyperparameter-tuned for the final system |
| **time** | Benchmarking | Measures training time and prediction time for each model, used to compare efficiency alongside accuracy |
| **warnings** | Cleanup | Suppresses noisy deprecation warnings in notebook output |

---

## 🔬 Project Workflow

1. **Data Acquisition** — Query SDSS CasJobs SQL database (PhotoObj + SpecObj join) for 10,000 labeled observations.
2. **Data Exploration** — Check structure, missing values (none found), class balance, and basic stats.
3. **Feature Filtering** — Drop identifier/instrument columns (`objid`, `specobjid`, `run`, `rerun`, `camcol`, `field`) that carry no physical relationship to the class label.
4. **Univariate & Multivariate Analysis** — Visualize `redshift` and `dec` distributions per class, and correlation between the 5 photometric bands.
5. **Feature Engineering** — Apply PCA on `u, g, r, i, z` → 3 components (`PCA_1`, `PCA_2`, `PCA_3`), reducing dimensionality while preserving most of the signal.
6. **Model Comparison** — Train & benchmark 5 algorithms (KNN, Naive Bayes, XGBoost, Random Forest, SVM) on accuracy + training/prediction time.
7. **Cross-Validation** — 10-fold CV on the top 2 models (XGBoost, Random Forest) to confirm results aren't due to a lucky train/test split.
8. **Feature Importance** — Use XGBoost's feature importances to identify `redshift` as the strongest predictor; drop the least useful feature (`mjd`).
9. **Hyperparameter Tuning** — Tune XGBoost (`max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha`) for the final model.
10. **Final Evaluation** — Confusion matrix, precision, recall, and F1-score on the tuned XGBoost model.

---

## 📊 Results

XGBoost and Random Forest were the top performers among the 5 models tested. The final tuned **XGBoost** model achieved:

- **Precision / Recall / F1-score:** ~99.36% (micro-averaged)
- Of ~10,000 objects, only **64 were misclassified** in cross-validated predictions — most confusion occurred between stars and galaxies/quasars at the edges of their `redshift` distributions.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook SDSS_Full_System_XGBoost.ipynb
```

Run all cells top to bottom. The notebook reads `sdss_data.csv` initially — make sure that file is the **raw** SDSS export before running the full pipeline (see the note in the Repository Contents section above).

---

## 🙏 Acknowledgements

- Data: [Sloan Digital Sky Survey, DR14](http://www.sdss.org/dr14/)
- Workflow approach inspired by Niklas Donges' Titanic classification notebook structure
- Naive Bayes scaling tip from Adithya Raman (Kaggle)
