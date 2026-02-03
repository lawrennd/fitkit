"""Tests for data loaders (offline, no BigQuery required)."""

import numpy as np
import pytest

from fitkit.data.loaders import QueryConfig
from fitkit.data.fixtures import SyntheticLoader, create_small_fixture
from fitkit.types import DataBundle


def test_create_small_fixture():
    """Test that create_small_fixture returns a valid DataBundle."""
    bundle = create_small_fixture()
    
    # Check type
    assert isinstance(bundle, DataBundle)
    
    # Check shapes
    assert bundle.matrix.shape == (5, 8)
    assert len(bundle.row_labels) == 5
    assert len(bundle.col_labels) == 8
    
    # Check metadata
    assert bundle.metadata["source"] == "fixture:small"
    assert "created_at" in bundle.metadata


def test_synthetic_loader_basic():
    """Test that SyntheticLoader generates valid data."""
    config = QueryConfig(max_authors=20, max_features=30)
    loader = SyntheticLoader(config, seed=42)
    bundle = loader.load()
    
    # Check type
    assert isinstance(bundle, DataBundle)
    
    # Check shapes match config
    assert bundle.matrix.shape[0] == 20  # max_authors
    assert bundle.matrix.shape[1] <= 30  # max_features (may be less due to vocab size)
    assert len(bundle.row_labels) == 20
    assert len(bundle.col_labels) == bundle.matrix.shape[1]
    
    # Check metadata
    assert bundle.metadata["source"] == "synthetic"
    assert bundle.metadata["seed"] == 42
    assert "created_at" in bundle.metadata


def test_synthetic_loader_deterministic():
    """Test that SyntheticLoader is deterministic with fixed seed."""
    config = QueryConfig(max_authors=10, max_features=20)
    
    loader1 = SyntheticLoader(config, seed=42)
    bundle1 = loader1.load()
    
    loader2 = SyntheticLoader(config, seed=42)
    bundle2 = loader2.load()
    
    # Matrices should be identical
    np.testing.assert_array_equal(bundle1.matrix.toarray(), bundle2.matrix.toarray())
    np.testing.assert_array_equal(bundle1.row_labels, bundle2.row_labels)
    np.testing.assert_array_equal(bundle1.col_labels, bundle2.col_labels)


def test_synthetic_loader_different_seeds():
    """Test that different seeds produce different data."""
    config = QueryConfig(max_authors=10, max_features=20)
    
    loader1 = SyntheticLoader(config, seed=42)
    bundle1 = loader1.load()
    
    loader2 = SyntheticLoader(config, seed=99)
    bundle2 = loader2.load()
    
    # Matrices should be different (with high probability)
    assert not np.array_equal(bundle1.matrix.toarray(), bundle2.matrix.toarray())


def test_synthetic_loader_sparsity():
    """Test that SyntheticLoader produces sparse matrices."""
    config = QueryConfig(max_authors=50, max_features=100)
    loader = SyntheticLoader(config, seed=42)
    bundle = loader.load()
    
    # Check that matrix is sparse
    sparsity = bundle.matrix.nnz / (bundle.matrix.shape[0] * bundle.matrix.shape[1])
    
    # Sparsity should be reasonable (not fully dense)
    assert 0 < sparsity < 0.5
    
    # Check that nnz is reported in metadata
    assert "nnz" in bundle.metadata
    assert bundle.metadata["nnz"] == bundle.matrix.nnz


def test_synthetic_loader_binary_mode():
    """Test that binary mode produces binary matrices."""
    config = QueryConfig(max_authors=10, max_features=20, binary=True)
    loader = SyntheticLoader(config, seed=42)
    bundle = loader.load()
    
    # All nonzero entries should be 1.0
    data = bundle.matrix.data
    assert np.all(data == 1.0)


def test_synthetic_loader_count_mode():
    """Test that count mode produces count matrices."""
    config = QueryConfig(max_authors=10, max_features=20, binary=False)
    loader = SyntheticLoader(config, seed=42)
    bundle = loader.load()
    
    # Entries should be positive counts (not all 1.0)
    data = bundle.matrix.data
    assert np.all(data > 0)
    # At least some entries should be > 1.0 (geometric distribution)
    assert np.any(data > 1.0)


def test_query_config_defaults():
    """Test that QueryConfig has sensible defaults."""
    config = QueryConfig()
    
    assert config.max_authors == 1000
    assert config.min_comments_per_author == 20
    assert config.max_docs_per_author == 2000
    assert config.min_df == 3
    assert config.max_features == 5000
    assert config.binary == False
    assert config.min_user_mass == 5
    assert config.min_word_mass == 5


def test_data_bundle_structure():
    """Test that DataBundle has the expected structure."""
    bundle = create_small_fixture()
    
    # Check required attributes
    assert hasattr(bundle, "matrix")
    assert hasattr(bundle, "row_labels")
    assert hasattr(bundle, "col_labels")
    assert hasattr(bundle, "metadata")
    
    # Check types
    assert isinstance(bundle.metadata, dict)
    assert isinstance(bundle.row_labels, np.ndarray)
    assert isinstance(bundle.col_labels, np.ndarray)
