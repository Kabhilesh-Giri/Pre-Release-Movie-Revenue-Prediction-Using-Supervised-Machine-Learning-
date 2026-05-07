# Pre-Release Movie Revenue Prediction Using Supervised Machine Learning 

This repository documents a supervised machine learning project for estimating a movie's box-office revenue before a project is greenlit. The workflow combines two large movie datasets, filters them into a clean supervised regression table, engineers director-facing movie features, tunes multiple model families on a 24-core CPU environment, and finishes with a deployment-style prediction cell for entering a future movie concept and receiving an estimated revenue.

The project is intentionally focused on the business question: given information a director, producer, or studio could know before release, what revenue range should they expect?

## Final Artifacts

| Artifact | Purpose |
| --- | --- |
| `Dataset/comparison_outputs/Movies-Dataset.csv` | Combined and cleaned supervised dataset used as the project base table. |
| `data_integration_and_filtering.ipynb` | Combines the two source datasets, resolves overlapping movies, filters invalid revenue/budget rows, and creates `Movies-Dataset.csv`. |
| `01_movies_eda_cleaning_feature_prep.ipynb` | Explores the cleaned movie data, missing values, feature distributions, correlations, PCA, and early feature preparation choices. |
| `FullNotebook.ipynb` | End-to-end workflow: ML-ready transformation, feature encoding, final model fit, diagnostics, and the user-input revenue prediction cell. |
| `HyperParameterTuning_ModelFinalization.ipynb` | HPC/24-core hyperparameter search notebook used to compare model families and lock the final modeling direction. |
| `FinalProjectReport.pdf` | Full written report. |
| `Presentation.pdf` | Results presentation. |

The raw source files `Dataset/movies.csv` and `Dataset/TMDB_movie_dataset_v11.csv` were used locally, but they are not committed because they are approximately 334 MB and 574 MB, which is beyond GitHub's normal per-file limit. The committed dataset is the smaller supervised project dataset created from them.

## Problem Formulation

The target variable is movie revenue. This is a supervised regression problem because every training row contains known movie features and a known historical revenue value.

The project avoids relying on post-release information for the final modeling workflow. Popularity, vote counts, ratings, reviews, and other audience-response variables are useful for analysis, but they are not dependable before release. The model is therefore framed around information that can exist before or during greenlight evaluation:

- budget and runtime
- production country, studio, genre, and spoken language
- original language and MPAA rating
- actor popularity features from TMDB enrichment
- keywords, tagline, and overview text

## Data Story

Two movie datasets were merged using normalized title and release-date keys. For movies appearing in both sources, the notebooks compare overlapping columns, preserve useful fields, and report mismatches before creating the final combined table.

The integration notebook starts from 1,349,515 merged records. Most raw rows do not contain usable supervised targets: 1,326,310 rows have revenue equal to zero and 1,275,849 rows have budget equal to zero. After filtering for usable positive revenue and budget, the project keeps 16,441 movies in `Movies-Dataset.csv`.

The full modeling workflow then removes one invalid negative-revenue row, clips extreme budget and revenue values at the 99.5th percentile, applies a log transform, restricts the modeling period to 2000-2026, and keeps complete cases. This produces 3,856 final modeling rows and 135 encoded columns, with 132 columns used as model features after dropping title, release date, and target revenue.

## Feature Engineering

The final feature table contains 35 continuous features and 97 binary encoded features.

Important transformations include:

- budget and revenue capped at the 99.5th percentile before log transformation
- budget cap: $205,000,000
- revenue cap: $1,000,000,000
- multi-label encoding for production countries, studios, genres, and spoken languages
- one-hot encoding for original language and MPAA rating
- keyword, tagline, and overview text signals converted into model features
- missing actor popularity filled conservatively during preprocessing

The final Elastic Net diagnostic table identifies budget as the strongest positive coefficient, followed by number of keywords, sequel-related keywords, runtime, and actor popularity signals. R-rated movies and drama genre indicators had negative coefficients in that fitted linear model.

## Hyperparameter Tuning

The exhaustive tuning notebook was run on a 24-core CPU environment. The run used:

- 5-fold cross-validation
- 80/20 train-test split
- 5 parallel CV jobs
- 5 estimator threads per model
- 8 model variants across Ridge Regression, Elastic Net, KNN, and Random Forest
- with-Lasso and without-Lasso feature-selection modes
- 1,224 searched candidates and 6,120 planned CV fits

Best tuning result from `HyperParameterTuning_ModelFinalization.ipynb`:

| Model | Test RMSE | Test MAE | Test R2 | Notes |
| --- | ---: | ---: | ---: | --- |
| KNN without Lasso | $116,845,939 | $63,968,001 | 0.595 | Best pure test-score result in the HPC search. |
| Random Forest without Lasso | $118,192,075 | $64,206,987 | 0.585 | Strong tree-based benchmark. |
| Elastic Net without Lasso | $119,669,261 | $74,229,794 | 0.575 | Interpretable linear benchmark. |
| Ridge without Lasso | $119,697,648 | $74,262,349 | 0.574 | Similar to Elastic Net. |
| Elastic Net with Lasso Sweep | $121,214,082 | $73,399,814 | 0.564 | Lightweight model family used for the deployment-style notebook flow. |

The tuning result shows that non-linear neighborhood/tree methods scored slightly better, while linear regularized models remained competitive and easier to explain.

## Final Workflow Result

`FullNotebook.ipynb` locks an `Elastic Net | With Lasso Sweep` pipeline for the final explainable workflow and demonstrates how a future movie can be entered as a dictionary of planned production values.

Final notebook metrics after inverse-transforming predictions back to capped revenue scale:

| Split | RMSE | MAE | MedianAE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Train | $140,430,021 | $68,843,800 | $22,921,733 | 0.456 |
| Test | $131,992,591 | $67,575,349 | $23,725,675 | 0.421 |

The example deployment cell uses a planned action/adventure/science-fiction movie with a $120M budget, PG-13 rating, major studio signals, cast popularity values, keywords, tagline, and overview. The trained pipeline returns an estimated revenue of **$153,521,637** for that sample input.

## Project Narrative

1. Dataset combining and cleaning: the project merges two large movie sources, keeps only movies with usable budget and revenue, enriches selected people/rating fields, and creates the supervised project dataset.
2. Feature formulation and HPC tuning: the cleaned data is transformed into numeric, binary, categorical, and text-derived model features, then tuned across model families on a 24-core CPU setup.
3. Deployment model: the final workflow notebook provides a ready prediction pattern where a future movie concept can be entered and passed through the trained pipeline to estimate revenue before a greenlight decision.
