# Fitkit Examples

This directory contains example notebooks and scripts demonstrating the fitkit package.

## Atlas of Economic Complexity Examples

### `atlas_fitness_comparison.ipynb` - Comprehensive Fitness Analysis

A complete demonstration of the Fitness-Complexity algorithm using real-world trade data from the Harvard Atlas of Economic Complexity. This notebook:

**Compares economic fitness between 2000 and 2020:**
- Loads and processes trade data for both years
- Computes country fitness and product complexity
- Analyzes convergence behavior
- Identifies top-performing and declining countries

**Key visualizations:**
- Fitness scatter plots showing country evolution
- Convergence plots for the iterative algorithm
- Distribution comparisons across time
- Diversification vs. fitness relationships
- Product complexity rankings and changes

**Analysis includes:**
- Top 20 countries by economic fitness in each year
- Countries with largest fitness gains/losses
- Product complexity evolution
- Diversification patterns
- Statistical correlations

**Perfect for:**
- Understanding how the Fitness-Complexity algorithm works
- Demonstrating economic evolution over time
- Learning to work with Atlas trade data
- Creating publication-quality visualizations

### `atlas_data_usage.md` - Quick Start Guide

Concise examples showing how to:
- Load Atlas data for different years and classifications
- List available years
- Compare time periods
- Create basic visualizations

## Other Examples

### `bipartite_community_detection.ipynb`

Community detection in bipartite networks using spectral methods (from the spectral package).

### `spectral_entropic_scenarios.ipynb`

Advanced spectral clustering demonstrations with various scenarios.

## Running the Examples

All notebooks require the fitkit package and its dependencies:

```bash
# Install fitkit
pip install -e .

# Launch Jupyter
jupyter notebook examples/
```

**Note**: The first time you run `atlas_fitness_comparison.ipynb`, it will download ~475 MB of data from Harvard Dataverse. This takes a few minutes but the data is cached locally.

## Data Sources

- **Atlas of Economic Complexity**: Harvard Growth Lab (https://atlas.hks.harvard.edu/)
- **R economiccomplexity package**: World trade data 1998-2000 (included in `fitkit/data/`)
