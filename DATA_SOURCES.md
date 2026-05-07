# Data Sources

This project combines two Kaggle movie datasets and enriches the merged result with TMDB API lookups.

## Raw Source Files

The original raw CSV files are intentionally not tracked because they exceed GitHub's 100 MB per-file limit:

| Expected path | Approx. local size | Purpose |
| --- | ---: | --- |
| `Dataset/movies.csv` | 334 MB | Raw Kaggle movie metadata source |
| `Dataset/TMDB_movie_dataset_v11.csv` | 574 MB | Raw TMDB daily update movie source |

To rerun the full integration notebook, place the downloaded raw datasets at those paths.

## Tracked Derived Outputs

The repository keeps the smaller derived datasets used by the modeling notebooks:

| File | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `Dataset/comparison_outputs/Movies-Dataset.csv` | 16,441 | 41 | Combined post-merge dataset |
| `Dataset/comparison_outputs/Movies-Dataset-no-missing-2000-2026-ML-Ready.csv` | 3,856 | 22 | Cleaned modeling table |
| `Dataset/comparison_outputs/Movies-Dataset-no-missing-2000-2026-ML-Ready-encoded.csv` | 3,856 | 135 | Encoded feature table |

## API Setup

`tmdb_enrichment.py` expects a TMDB key in the environment:

```powershell
$env:TMDB_API_KEY="your_tmdb_api_key"
```

The API key is intentionally not stored in the repository.
