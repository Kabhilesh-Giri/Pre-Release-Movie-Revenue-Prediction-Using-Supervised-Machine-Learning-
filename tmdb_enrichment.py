from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import local
import os
import time
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests


EMPTY_ACTOR_LOOKUP = (pd.NA, pd.NA, None)
EMPTY_ACTOR_TUPLE = (pd.NA, pd.NA, pd.NA)


def enrich_cleaned_movies(
    cleaned_movies: pd.DataFrame,
    *,
    final_merged_movies: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_unique: pd.DataFrame,
    df2_unique: pd.DataFrame,
    common_combined: pd.DataFrame,
    unique_src1_export: pd.DataFrame,
    unique_src2_export: pd.DataFrame,
    mismatch_summary: pd.DataFrame,
    mismatch_details_path: Union[Path, str],
    cleaned_output_path: Union[Path, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    print("Inside enrich_cleaned_movies function")
    if "credits" not in cleaned_movies.columns:
        raise KeyError(
            "The merged dataset does not contain a 'credits' column required for "
            "actor popularity enrichment."
        )

    if "id" not in cleaned_movies.columns:
        raise KeyError(
            "The merged dataset does not contain an 'id' column required for MPAA "
            "rating enrichment."
        )

    cleaned_movies = cleaned_movies.copy()
    mismatch_details_path = Path(mismatch_details_path)
    cleaned_output_path = Path(cleaned_output_path)
    mismatch_details_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)

    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if not tmdb_api_key:
        raise ValueError(
            "A TMDB API key is required to populate actor popularity features and "
            "MPAA ratings."
        )

    tmdb_request_max_attempts = max(1, int(os.getenv("TMDB_REQUEST_MAX_ATTEMPTS", "4")))
    tmdb_request_timeout_seconds = float(
        os.getenv("TMDB_REQUEST_TIMEOUT_SECONDS", "20")
    )
    tmdb_parse_workers = max(1, int(os.getenv("TMDB_PARSE_WORKERS", "8")))
    tmdb_lookup_workers = max(
        1, int(os.getenv("TMDB_LOOKUP_WORKERS", str(tmdb_parse_workers)))
    )
    tmdb_movie_workers = max(1, int(os.getenv("TMDB_MOVIE_WORKERS", "8")))
    tmdb_movie_delay_seconds = max(
        0.0, float(os.getenv("TMDB_MOVIE_DELAY_SECONDS", "0"))
    )
    actor_parse_max_hyphen_parts = max(
        1, int(os.getenv("TMDB_ACTOR_PARSE_MAX_HYPHEN_PARTS", "2"))
    )

    thread_local = local()

    def get_thread_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    def normalize_name_key(value: object) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace(chr(8217), "'")
        return " ".join(normalized.lower().split())

    def get_naive_actor_name(parts: object, index: int) -> object:
        if not isinstance(parts, list) or len(parts) <= index:
            return pd.NA

        name = str(parts[index]).strip()
        return name if name else pd.NA

    def get_json_with_retries(
        url: str,
        params: Dict[str, object],
        *,
        timeout: float = tmdb_request_timeout_seconds,
        max_attempts: int = tmdb_request_max_attempts,
    ) -> Optional[Dict[str, object]]:
        requester = get_thread_session()
        for attempt in range(max_attempts):
            try:
                response = requester.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt + 1 == max_attempts:
                    return None

                time.sleep(min(2**attempt, 8))

        return None

    @lru_cache(maxsize=None)
    def search_person(name: str) -> Optional[Dict[str, object]]:
        clean_name = str(name).strip()
        if not clean_name:
            return None

        data = get_json_with_retries(
            "https://api.themoviedb.org/3/search/person",
            {
                "api_key": tmdb_api_key,
                "query": clean_name,
            },
        )

        results = data.get("results", []) if data else []
        return results[0] if results else None

    def is_exact_person_match(name: str) -> bool:
        result = search_person(name)
        matched_name = result.get("name") if result else ""
        return bool(matched_name) and (
            normalize_name_key(matched_name) == normalize_name_key(name)
        )

    def should_try_hyphen_merge(raw_parts: List[str], cursor: int) -> bool:
        if cursor + 1 >= len(raw_parts):
            return False

        current_tokens = raw_parts[cursor].split()
        next_tokens = raw_parts[cursor + 1].split()
        return len(current_tokens) == 1 or len(next_tokens) == 1

    @lru_cache(maxsize=None)
    def extract_actor_names(credits_value: str) -> Tuple[object, object, object]:
        clean_credits = str(credits_value).strip()
        if not clean_credits:
            return EMPTY_ACTOR_TUPLE

        raw_parts = [
            part.strip() for part in clean_credits.split("-") if str(part).strip()
        ]
        parsed_names: List[object] = []
        cursor = 0

        while cursor < len(raw_parts) and len(parsed_names) < 3:
            chosen_name = raw_parts[cursor]
            chosen_width = 1
            max_span = min(actor_parse_max_hyphen_parts, len(raw_parts) - cursor)

            # Only test merged names when the split looks like a hyphenated name.
            if max_span > 1 and should_try_hyphen_merge(raw_parts, cursor):
                for span in range(max_span, 1, -1):
                    candidate = "-".join(raw_parts[cursor : cursor + span]).strip()
                    if is_exact_person_match(candidate):
                        chosen_name = candidate
                        chosen_width = span
                        break

            parsed_names.append(chosen_name if chosen_name else pd.NA)
            cursor += chosen_width

        while len(parsed_names) < 3:
            parsed_names.append(pd.NA)

        return tuple(parsed_names[:3])  # type: ignore[return-value]

    @lru_cache(maxsize=None)
    def lookup_actor(name: str) -> Tuple[object, object, object]:
        clean_name = str(name).strip()
        if not clean_name:
            return EMPTY_ACTOR_LOOKUP

        result = search_person(clean_name)
        if not result:
            return EMPTY_ACTOR_LOOKUP

        person_id = result.get("id")
        matched_name = result.get("name") or pd.NA
        popularity = result.get("popularity")
        return (
            matched_name,
            person_id if person_id is not None else pd.NA,
            popularity,
        )

    @lru_cache(maxsize=None)
    def get_mpaa_rating(movie_id: int) -> Optional[str]:
        data = get_json_with_retries(
            f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates",
            {
                "api_key": tmdb_api_key,
            },
        )
        if data is None:
            return None

        if tmdb_movie_delay_seconds > 0:
            time.sleep(tmdb_movie_delay_seconds)

        for country in data.get("results", []):
            if country.get("iso_3166_1") != "US":
                continue

            for release in country.get("release_dates", []):
                certification = release.get("certification")
                if certification:
                    return certification

        return None

    print("Start")
    credits_series = cleaned_movies["credits"].fillna("").astype(str).str.strip()
    naive_credits_split = credits_series.str.split("-")
    naive_actor_names = {
        f"actor_{idx + 1}": naive_credits_split.apply(
            lambda parts, i=idx: get_naive_actor_name(parts, i)
        )
        for idx in range(3)
    }

    print("Done")

    unique_credits = credits_series[credits_series.ne("")].drop_duplicates().tolist()
    parsed_credit_map = {"": EMPTY_ACTOR_TUPLE}

    print("Done1")
    if unique_credits:
        parse_workers = min(tmdb_parse_workers, len(unique_credits))
        with ThreadPoolExecutor(max_workers=parse_workers) as executor:
            parsed_unique_actor_lists = list(
                executor.map(extract_actor_names, unique_credits)
            )

        parsed_credit_map.update(dict(zip(unique_credits, parsed_unique_actor_lists)))

    parsed_actor_lists = credits_series.apply(
        lambda value: parsed_credit_map.get(value, EMPTY_ACTOR_TUPLE)
    )
    parsed_actor_df = pd.DataFrame(
        parsed_actor_lists.tolist(),
        index=cleaned_movies.index,
        columns=["actor_1", "actor_2", "actor_3"],
    )
    cleaned_movies[["actor_1", "actor_2", "actor_3"]] = parsed_actor_df

    unique_actor_names = (
        pd.concat(
            [
                cleaned_movies["actor_1"],
                cleaned_movies["actor_2"],
                cleaned_movies["actor_3"],
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
        .str.strip()
    )
    unique_actor_names = (
        unique_actor_names[unique_actor_names.ne("")].drop_duplicates().tolist()
    )

    actor_lookup_by_name: Dict[str, Tuple[object, object, object]] = {}
    if unique_actor_names:
        lookup_workers = min(tmdb_lookup_workers, len(unique_actor_names))
        with ThreadPoolExecutor(max_workers=lookup_workers) as executor:
            actor_lookup_results = list(executor.map(lookup_actor, unique_actor_names))

        actor_lookup_by_name = dict(zip(unique_actor_names, actor_lookup_results))

    def get_actor_lookup_tuple(name: object) -> Tuple[object, object, object]:
        if pd.isna(name):
            return EMPTY_ACTOR_LOOKUP

        clean_name = str(name).strip()
        if not clean_name:
            return EMPTY_ACTOR_LOOKUP

        return actor_lookup_by_name.get(clean_name, EMPTY_ACTOR_LOOKUP)

    for idx in range(3):
        actor_col = f"actor_{idx + 1}"
        actor_tmdb_name_col = f"{actor_col}_tmdb_name"
        actor_tmdb_id_col = f"{actor_col}_tmdb_id"
        actor_popularity_col = f"{actor_col}_popularity"

        actor_lookup_df = pd.DataFrame(
            cleaned_movies[actor_col].apply(get_actor_lookup_tuple).tolist(),
            index=cleaned_movies.index,
            columns=[
                actor_tmdb_name_col,
                actor_tmdb_id_col,
                actor_popularity_col,
            ],
        )
        cleaned_movies[
            [actor_tmdb_name_col, actor_tmdb_id_col, actor_popularity_col]
        ] = actor_lookup_df

    movie_id_series = pd.to_numeric(cleaned_movies["id"], errors="coerce").astype(
        "Int64"
    )
    unique_movie_ids = (
        movie_id_series.dropna().astype("int64").drop_duplicates().tolist()
    )

    mpaa_rating_by_movie_id: Dict[int, Optional[str]] = {}
    if unique_movie_ids:
        movie_workers = min(tmdb_movie_workers, len(unique_movie_ids))
        with ThreadPoolExecutor(max_workers=movie_workers) as executor:
            mpaa_values = list(executor.map(get_mpaa_rating, unique_movie_ids))

        mpaa_rating_by_movie_id = dict(zip(unique_movie_ids, mpaa_values))

    cleaned_movies["mpaa_rating"] = movie_id_series.map(mpaa_rating_by_movie_id)
    cleaned_movies.to_csv(cleaned_output_path, index=False, encoding="utf-8-sig")

    credits_available_rows = int(credits_series.ne("").sum())
    reconstructed_actor_rows = int(
        pd.DataFrame(
            {
                actor_col: cleaned_movies[actor_col].fillna("")
                != naive_actor_names[actor_col].fillna("")
                for actor_col in ["actor_1", "actor_2", "actor_3"]
            }
        )
        .any(axis=1)
        .sum()
    )

    revenue_numeric = pd.to_numeric(final_merged_movies["revenue"], errors="coerce")
    budget_numeric = pd.to_numeric(final_merged_movies["budget"], errors="coerce")

    summary_lines = [
        "Data integration and filtering summary",
        "=" * 60,
        f"Dataset 1 raw rows: {len(df1)}",
        f"Dataset 2 raw rows: {len(df2)}",
        f"Dataset 1 duplicate merge keys removed: {len(df1) - len(df1_unique)}",
        f"Dataset 2 duplicate merge keys removed: {len(df2) - len(df2_unique)}",
        f"Common movies overlapped into one row: {len(common_combined)}",
        f"Unique movies appended from dataset 1: {len(unique_src1_export)}",
        f"Unique movies appended from dataset 2: {len(unique_src2_export)}",
        f"Rows in raw merged dataset: {len(final_merged_movies)}",
        "",
        "Revenue and budget counts before filtering:",
        (
            f"- revenue: <0={(revenue_numeric < 0).sum()}, =0={(revenue_numeric == 0).sum()}, "
            f">0={(revenue_numeric > 0).sum()}, missing={revenue_numeric.isna().sum()}"
        ),
        (
            f"- budget: <0={(budget_numeric < 0).sum()}, =0={(budget_numeric == 0).sum()}, "
            f">0={(budget_numeric > 0).sum()}, missing={budget_numeric.isna().sum()}"
        ),
        "",
        f"Rows after removing missing or zero revenue and budget: {len(cleaned_movies)}",
        f"Rows removed by final filter: {len(final_merged_movies) - len(cleaned_movies)}",
        f"Rows with non-empty credits available for actor enrichment: {credits_available_rows}",
        f"Rows without credits after the merge: {len(cleaned_movies) - credits_available_rows}",
        f"Rows with hyphen-aware actor reconstruction changes: {reconstructed_actor_rows}",
        f"Unique actor names queried from TMDB: {len(unique_actor_names)}",
        f"Unique movie IDs queried for MPAA ratings: {len(unique_movie_ids)}",
        "",
        f"Cleaned dataset: {cleaned_output_path}",
        f"Mismatch details text report: {mismatch_details_path}",
        "",
    ]

    if mismatch_summary.empty:
        summary_lines.append(
            "No comparable mismatches were found across shared columns."
        )
    else:
        summary_lines.append("Mismatch counts by column:")
        for row in mismatch_summary.itertuples(index=False):
            summary_lines.append(f"- {row.column}: {row.mismatch_rows}")

    summary_text = chr(10).join(summary_lines)

    output_summary = pd.DataFrame(
        {
            "artifact": [
                "mismatch_details_text_report",
                "cleaned_dataset",
            ],
            "path": [
                str(mismatch_details_path),
                str(cleaned_output_path),
            ],
        }
    )

    actor_enrichment_summary = pd.DataFrame(
        {
            "actor_column": ["actor_1", "actor_2", "actor_3"],
            "parsed_rows": [
                int(cleaned_movies["actor_1"].notna().sum()),
                int(cleaned_movies["actor_2"].notna().sum()),
                int(cleaned_movies["actor_3"].notna().sum()),
            ],
            "popularity_rows": [
                int(cleaned_movies["actor_1_popularity"].notna().sum()),
                int(cleaned_movies["actor_2_popularity"].notna().sum()),
                int(cleaned_movies["actor_3_popularity"].notna().sum()),
            ],
        }
    )

    return cleaned_movies, actor_enrichment_summary, output_summary, summary_text
