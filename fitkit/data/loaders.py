"""Data loading adapters for Wikipedia edit analysis.

This module provides data-loading implementations that conform to the
DataLoader protocol defined in fitkit.types. Data loaders handle I/O,
authentication, caching, and return standardized in-memory DataBundle objects.

Authentication is environment-aware:
- In Google Colab: uses google.colab.auth.authenticate_user()
- In local environments: uses Application Default Credentials (ADC)
- In tests: not needed (use SyntheticLoader from fixtures module)
"""

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from fitkit.types import DataBundle


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for Wikipedia data query.

    Attributes:
        max_authors: Maximum number of authors/users to sample.
        min_comments_per_author: Minimum edits required per author.
        max_docs_per_author: Maximum edits to aggregate per author.
        specific_users: Optional list of specific users to prioritize.
        min_df: Minimum document frequency for vocabulary (passed to CountVectorizer).
        max_features: Maximum vocabulary size (passed to CountVectorizer).
        binary: If True, use binary incidence; if False, use word counts.
        min_user_mass: Minimum total word count to keep a user.
        min_word_mass: Minimum total occurrence to keep a word.
    """
    max_authors: int = 1000
    min_comments_per_author: int = 20
    max_docs_per_author: int = 2000
    specific_users: tuple[str, ...] = field(default_factory=tuple)

    # Vocabulary filtering
    min_df: int = 3
    max_features: int = 5000
    binary: bool = False

    # Post-filtering
    min_user_mass: int = 5
    min_word_mass: int = 5


class WikipediaLoader:
    """Data loader for Wikipedia edit data via BigQuery with caching.

    This loader:
    - Queries BigQuery for Wikipedia edit comments (or loads from cache)
    - Builds a user × word matrix using CountVectorizer
    - Filters low-mass users/words
    - Returns a DataBundle with matrix, labels, and provenance metadata

    Authentication is environment-aware:
    - In Colab: uses google.colab.auth.authenticate_user()
    - Locally: uses google.auth.default() (ADC)
    """

    def __init__(self, config: QueryConfig, cache_path: str):
        """Initialize WikipediaLoader.

        Args:
            config: QueryConfig with query and filtering parameters.
            cache_path: Path to parquet cache file. If exists, load from cache;
                       otherwise query BigQuery and save to cache.
        """
        self.config = config
        self.cache_path = cache_path

    def load(self) -> DataBundle:
        """Load Wikipedia edit data and return a DataBundle.

        Returns:
            DataBundle with:
                - matrix: sparse user × word matrix (CSR format)
                - row_labels: user IDs (strings)
                - col_labels: word tokens (strings)
                - metadata: source, cache_path, config, created_at

        Raises:
            IOError: If BigQuery query fails and no cache exists.
            AuthenticationError: If credentials are missing/invalid.
        """
        # Load raw data (from cache or BigQuery)
        if os.path.exists(self.cache_path):
            print(f"Loading cached Wikipedia data from {self.cache_path}...")
            df = pd.read_parquet(self.cache_path)
        else:
            print("No cache found. Querying BigQuery...")
            df = self._query_bigquery()
            # Save to cache
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            df.to_parquet(self.cache_path, index=False)
            print(f"Saved cache to {self.cache_path}")

        # Build user × word matrix
        print("Building user × word matrix...")
        X, user_ids, vocab = self._build_matrix(df)

        # Return DataBundle
        metadata = {
            "source": "bigquery:wikipedia",
            "cache_path": self.cache_path,
            "config": self.config,
            "created_at": datetime.utcnow().isoformat(),
            "n_users_raw": len(df),
            "n_users_filtered": len(user_ids),
            "n_words": len(vocab),
        }

        return DataBundle(
            matrix=X,
            row_labels=np.array(user_ids),
            col_labels=np.array(vocab),
            metadata=metadata,
        )

    def _query_bigquery(self) -> pd.DataFrame:
        """Query BigQuery for Wikipedia edit data.

        Returns:
            DataFrame with columns: author, user_text, n_comments

        Raises:
            IOError: If query fails.
            AuthenticationError: If credentials are missing/invalid.
        """
        # Import BigQuery dependencies here (only needed if actually querying)
        try:
            import google.auth
            from google.cloud import bigquery
        except ImportError as e:
            raise ImportError(
                "BigQuery dependencies not installed. "
                "Install with: pip install google-cloud-bigquery google-auth"
            ) from e

        # Environment-aware authentication
        try:
            # Check if running in Colab
            import google.colab  # noqa: F401
            in_colab = True
        except ImportError:
            in_colab = False

        if in_colab:
            print("Detected Colab environment, using colab.auth...")
            from google.colab import auth
            auth.authenticate_user()
            credentials, project = google.auth.default()
        else:
            print("Using Application Default Credentials (ADC)...")
            print("Ensure you've run: gcloud auth application-default login")
            credentials, project = google.auth.default()

        # Use project from config if provided, otherwise use detected project
        if self.config.project_id:
            project = self.config.project_id
        
        if not project:
            raise RuntimeError(
                "No GCP project ID found. Please specify in QueryConfig:\n"
                "  cfg = QueryConfig(project_id='your-project-id', ...)\n"
                "Or set via: gcloud config set project YOUR_PROJECT_ID"
            )

        print(f"Using GCP project: {project}")
        client = bigquery.Client(project=project, credentials=credentials)

        # Wikipedia query
        query = """
        WITH edits AS (
          SELECT
            contributor_username AS author,
            REGEXP_REPLACE(comment, r'/\\*.*?\\*/|<[^>]+>', '') AS body
          FROM `bigquery-public-data.samples.wikipedia`
          WHERE
            contributor_username IS NOT NULL
            AND comment IS NOT NULL
            AND NOT REGEXP_CONTAINS(LOWER(contributor_username), r'bot$')
        ),
        valid_edits AS (
          SELECT author, body
          FROM edits
          WHERE LENGTH(TRIM(body)) > 10
        ),
        sampled_authors AS (
          SELECT author
          FROM valid_edits
          GROUP BY author
          HAVING COUNT(*) >= @min_comments_per_author
          ORDER BY RAND()
          LIMIT @max_authors
        )
        SELECT
          author,
          ARRAY_TO_STRING(
              ARRAY_AGG(body ORDER BY LENGTH(body) DESC LIMIT @max_docs_per_author),
              '\\n'
          ) AS user_text,
          COUNT(*) AS n_comments
        FROM valid_edits
        JOIN sampled_authors
        USING (author)
        GROUP BY author
        ORDER BY n_comments DESC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("max_authors", "INT64", self.config.max_authors),
                bigquery.ScalarQueryParameter(
                    "min_comments_per_author", "INT64",
                    self.config.min_comments_per_author
                ),
                bigquery.ScalarQueryParameter(
                    "max_docs_per_author", "INT64",
                    self.config.max_docs_per_author
                ),
            ]
        )

        print("Running BigQuery...")
        df = client.query(query, job_config=job_config).to_dataframe()
        print(f"Retrieved {len(df)} users")

        return df

    def _build_matrix(self, df: pd.DataFrame) -> tuple[sp.spmatrix, list[str], np.ndarray]:
        """Build user × word matrix from raw text.

        Args:
            df: DataFrame with columns: author, user_text, n_comments

        Returns:
            X: sparse user × word matrix (CSR format)
            user_ids: list of user IDs (strings)
            vocab: array of word tokens (strings)
        """
        # Vectorize text into user × word matrix
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            binary=self.config.binary,
        )

        X = vectorizer.fit_transform(df["user_text"].fillna(""))
        X = X.astype(np.float64)

        vocab = vectorizer.get_feature_names_out()
        user_ids = df["author"].astype(str).tolist()

        print(f"Raw matrix: {X.shape[0]} users × {X.shape[1]} words")

        # Filter low-mass users/words
        user_strength = np.asarray(X.sum(axis=1)).ravel()
        word_strength = np.asarray(X.sum(axis=0)).ravel()

        keep_users = user_strength >= self.config.min_user_mass
        keep_words = word_strength >= self.config.min_word_mass

        X = X[keep_users][:, keep_words]
        user_ids = [u for u, ok in zip(user_ids, keep_users) if ok]
        vocab = vocab[keep_words]

        print(f"Filtered matrix: {X.shape[0]} users × {X.shape[1]} words")

        return X.tocsr(), user_ids, vocab
