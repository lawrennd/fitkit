"""Synthetic data fixtures for testing and offline development.

This module provides synthetic data generators that conform to the DataLoader
protocol, enabling offline testing without BigQuery access or cloud credentials.
"""

from datetime import datetime

import numpy as np
import scipy.sparse as sp

from fitkit.data.loaders import QueryConfig
from fitkit.types import DataBundle

class SyntheticLoader:
    """Generate synthetic Wikipedia-like data for testing.

    This loader creates random user × word matrices with configurable
    size and sparsity. Useful for:
    - Offline testing (no BigQuery required)
    - CI/CD pipelines (no credentials needed)
    - Quick experimentation
    """

    def __init__(self, config: QueryConfig, seed: int = 42):
        """Initialize SyntheticLoader.

        Args:
            config: QueryConfig with size parameters (max_authors, max_features).
            seed: Random seed for reproducibility (default: 42).
        """
        self.config = config
        self.seed = seed

    def load(self) -> DataBundle:
        """Generate synthetic Wikipedia-like data.

        Returns:
            DataBundle with:
                - matrix: sparse user × word matrix (CSR format)
                - row_labels: synthetic user IDs
                - col_labels: synthetic word tokens
                - metadata: source, config, created_at
        """
        rng = np.random.default_rng(self.seed)

        n_users = self.config.max_authors
        n_words = min(self.config.max_features, 100)  # Keep vocab small for tests

        # Generate user IDs and word tokens
        user_ids = np.array([f"user_{i}" for i in range(n_users)])
        vocab = np.array([
            "data", "science", "python", "learning", "machine", "big", "query",
            "analysis", "matrix", "complexity", "fitness", "network", "plot",
            "code", "algorithm", "statistics", "neural", "deep", "model",
            "optimization", "linear", "algebra", "visualization", "mining",
            "edit", "wiki", "page", "revision", "history", "link",
            "article", "content", "contributor", "user", "community", "project",
            "source", "reference", "citation", "category", "template", "policy",
            "discussion", "consensus", "vote", "admin", "moderator", "spam",
            "vandalism", "revert", "undo", "merge", "split", "redirect",
            "stub", "featured", "quality", "assessment", "review", "feedback",
            "suggestion", "improvement", "expansion", "cleanup", "formatting",
            "grammar", "spelling", "punctuation", "syntax", "style", "tone",
            "neutrality", "bias", "copyright", "license", "permission", "attribution",
            "image", "media", "video", "audio", "file", "upload", "caption",
            "thumbnail", "gallery", "infobox", "list", "table", "chart", "graph",
            "map", "timeline", "diagram", "illustration", "photo", "screenshot",
        ][:n_words])

        # Generate sparse random matrix
        # Each user has words with varying frequencies (some heavy, some light)
        data = []
        row = []
        col = []

        for i in range(n_users):
            # Each user has a random number of distinct words
            n_user_words = rng.integers(5, min(n_words, 30))
            word_ids = rng.choice(n_words, size=n_user_words, replace=False)

            for word_id in word_ids:
                # Word frequency (geometric distribution for realistic sparsity)
                if self.config.binary:
                    count = 1.0
                else:
                    count = float(rng.geometric(p=0.3))  # Most words appear 1-3 times

                row.append(i)
                col.append(word_id)
                data.append(count)

        X = sp.csr_matrix((data, (row, col)), shape=(n_users, n_words), dtype=np.float64)

        print(f"Generated synthetic matrix: {n_users} users × {n_words} words")
        print(f"Sparsity: {X.nnz / (n_users * n_words):.1%}")

        metadata = {
            "source": "synthetic",
            "config": self.config,
            "seed": self.seed,
            "created_at": datetime.utcnow().isoformat(),
            "n_users": n_users,
            "n_words": n_words,
            "nnz": X.nnz,
        }

        return DataBundle(
            matrix=X,
            row_labels=user_ids,
            col_labels=vocab,
            metadata=metadata,
        )


def create_small_fixture() -> DataBundle:
    """Create a small deterministic fixture for unit tests.

    Returns a 5-user × 8-word matrix with known structure for testing
    algorithm correctness and edge cases.

    Returns:
        DataBundle with small, fully-specified matrix.
    """
    # Small test matrix with interesting structure:
    # - Some isolated users/words (to test edge case handling)
    # - Some heavily connected users/words
    # - Deterministic for reproducibility

    user_ids = np.array(["alice", "bob", "charlie", "dave", "eve"])
    vocab = np.array(["python", "data", "science", "learning", "code", "test", "example", "demo"])

    # Define explicit connections (user, word, count)
    connections = [
        # alice: broad interests (python, data, science)
        (0, 0, 5.0), (0, 1, 3.0), (0, 2, 2.0),
        # bob: focused on coding (python, code)
        (1, 0, 10.0), (1, 4, 7.0),
        # charlie: data science (data, science, learning)
        (2, 1, 4.0), (2, 2, 6.0), (2, 3, 3.0),
        # dave: testing (test, example, code)
        (3, 4, 2.0), (3, 5, 8.0), (3, 6, 5.0),
        # eve: isolated, only uses "demo"
        (4, 7, 1.0),
    ]

    row, col, data = zip(*connections)
    X = sp.csr_matrix((data, (row, col)), shape=(5, 8), dtype=np.float64)

    metadata = {
        "source": "fixture:small",
        "created_at": datetime.utcnow().isoformat(),
        "description": "Small deterministic fixture for unit tests",
        "n_users": 5,
        "n_words": 8,
        "nnz": X.nnz,
    }

    return DataBundle(
        matrix=X,
        row_labels=user_ids,
        col_labels=vocab,
        metadata=metadata,
    )
