# Movie Revenue Prediction Before Production

Final machine learning project by Yohanan Ben-Gad and Kabhilesh Giri.

This project predicts movie box office revenue during the development stage, before post-release signals such as ratings, reviews, or audience popularity are available. The work combines multiple Kaggle movie datasets, creates a custom modeling dataset, enriches it with actor popularity and MPAA rating features, engineers numeric/categorical/text features, and compares tuned machine learning models.

The final writeup and presentation are included:

- `FinalProjectReport.pdf`
- `Presentation.pdf`

## Project Goal

Predict expected movie revenue from pre-production or early development information such as budget, runtime, release date, genre, production companies, production countries, language, MPAA rating, cast popularity, keywords, tagline, overview, and title.

The project deliberately excludes variables such as movie popularity, ratings, votes, and reviews because those are not realistically known before release and would make the forecasting task less useful for greenlighting decisions.

## Data Pipeline

The project starts with two Kaggle datasets, merges overlapping movies, removes rows without usable revenue and budget, filters older/outlier records, and creates the ML-ready dataset used for modeling.

Key dataset stages:

| Stage | Rows | Columns | File |
| --- | ---: | ---: | --- |
| Combined merged dataset | 16,441 | 41 | `Dataset/comparison_outputs/Movies-Dataset.csv` |
| Cleaned ML-ready dataset | 3,856 | 22 | `Dataset/comparison_outputs/Movies-Dataset-no-missing-2000-2026-ML-Ready.csv` |
| Encoded modeling dataset | 3,856 | 135 | `Dataset/comparison_outputs/Movies-Dataset-no-missing-2000-2026-ML-Ready-encoded.csv` |

The two raw Kaggle CSVs are not tracked because they exceed GitHub's 100 MB file limit. See `DATA_SOURCES.md` for expected paths and setup notes.

## Feature Engineering

Final feature processing included:

- Numeric scaling for budget, runtime, release year, and actor popularity.
- Multi-label binarization for genres, production companies, production countries, and spoken languages.
- One-hot encoding for original language and MPAA rating.
- CountVectorizer features for keywords.
- TF-IDF features for title, tagline, and overview.
- TMDB enrichment for actor popularity and MPAA rating.
- Experiments with Lasso feature selection and text clustering before settling on TF-IDF and CountVectorizer.

## Modeling

The final tuning stage used 5-fold cross validation across 11 model configurations. The report notes that the larger grid search ran on Northeastern's CPU cluster with 26 cores and 42 GB RAM and took about 14 to 16 hours.

Top report results:

| Model | Candidates | CV Fits | Train R2 | Test R2 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost, no Lasso | 96 | 480 | 0.985 | 0.595 |
| KNN, no Lasso | 24 | 120 | 1.000 | 0.594 |
| Random Forest, no Lasso | 54 | 270 | 0.925 | 0.585 |
| Linear Regression | 720 | 3,600 | 0.917 | 0.583 |
| Elastic Net, no Lasso | 18 | 90 | 0.616 | 0.574 |
| Ridge, no Lasso | 6 | 30 | 0.616 | 0.574 |
| CatBoost | 20 | 100 | 0.618 | 0.563 |

The strongest model in the report was XGBoost without Lasso, with a test R2 of 0.595. The final discussion also identifies budget, runtime, actor popularity, and franchise/studio/title signals as important predictors.

## Repository Layout

```text
.
├── data_integration_and_filtering.ipynb
├── 01_movies_eda_cleaning_feature_prep.ipynb
├── 02_movies_feature_processing_modeling1.ipynb
├── 02_movies_feature_processing_modeling2.ipynb
├── HyperParameterTuning_ModelFinalization.ipynb
├── FullNotebook.ipynb
├── tmdb_enrichment.py
├── Dataset/
│   └── comparison_outputs/
├── best_model_checkpoints/
├── FinalProjectReport.pdf
└── Presentation.pdf
```

## Reproducing

Install dependencies:

```powershell
pip install -r requirements.txt
```

For TMDB enrichment, set:

```powershell
$env:TMDB_API_KEY="your_tmdb_api_key"
```

Then run the notebooks in this order:

1. `data_integration_and_filtering.ipynb`
2. `01_movies_eda_cleaning_feature_prep.ipynb`
3. `FullNotebook.ipynb` or the `02_*` modeling notebooks
4. `HyperParameterTuning_ModelFinalization.ipynb`

The tracked derived datasets allow the modeling workflow to be inspected without re-downloading the two oversized raw CSV files.
