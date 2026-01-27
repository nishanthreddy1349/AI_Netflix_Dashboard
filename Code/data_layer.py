from pathlib import Path
import pandas as pd

# Get project root (Netflix_Dashboard/)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets"

def load_data():
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    watch_df = pd.read_csv(DATA_DIR / "watch_history.csv")

    movies_df.columns = movies_df.columns.str.lower().str.strip()
    watch_df.columns = watch_df.columns.str.lower().str.strip()

    watch_df["watch_date"] = pd.to_datetime(watch_df["watch_date"])
    watch_df["watch_duration_minutes"] = watch_df["watch_duration_minutes"].fillna(0)

    final_df = watch_df.merge(
        movies_df[
            [
                "movie_id",
                "title",
                "genre_primary",
                "content_type",
                "duration_minutes",
                "release_year",
                "rating",
            ]
        ],
        on="movie_id",
        how="left",
    )

    return final_df