#!/usr/bin/env Rscript
# Extract Demo Data from R economiccomplexity Package
#
# This script extracts the world trade dataset from the R economiccomplexity
# package and saves it in a format suitable for use with Python fitkit.
#
# USAGE:
#   Rscript scripts/extract_r_demo_data.R
#
# OUTPUT:
#   Creates fitkit/data/ directory with:
#   - world_trade_1998_2000.csv: Long-format trade data (country, product, value)
#   - world_trade_1998_2000_balassa.csv: Binary Balassa index matrix
#   - world_trade_1998_2000_metadata.txt: Dataset description
#
# REFERENCE:
#   R Package: https://cran.r-project.org/package=economiccomplexity
#   Data Source: World Bank / UN Comtrade

library(economiccomplexity)

# Create output directory
output_dir <- "fitkit/data"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("======================================================================\n")
cat("EXTRACTING R ECONOMICCOMPLEXITY DEMO DATA\n")
cat("======================================================================\n\n")

# Load world trade data
data(world_trade_avg_1998_to_2000)
cat("Loaded world_trade_avg_1998_to_2000\n")
cat(sprintf("  Unique countries: %d\n", length(unique(world_trade_avg_1998_to_2000$country))))
cat(sprintf("  Unique products:  %d\n", length(unique(world_trade_avg_1998_to_2000$product))))
cat(sprintf("  Total observations: %d\n\n", nrow(world_trade_avg_1998_to_2000)))

# Save raw trade data
output_file <- file.path(output_dir, "world_trade_1998_2000.csv")
write.csv(world_trade_avg_1998_to_2000, output_file, row.names = FALSE)
cat(sprintf("Saved: %s\n", output_file))

# Compute Balassa index (RCA) and save binary version
cat("\nComputing Balassa index...\n")
balassa <- balassa_index(world_trade_avg_1998_to_2000, discrete = TRUE, cutoff = 1)
cat(sprintf("  Matrix dimensions: %d countries × %d products\n", nrow(balassa), ncol(balassa)))
cat(sprintf("  Density: %.4f\n", sum(balassa) / (nrow(balassa) * ncol(balassa))))

# Save as sparse format (row, col, value) for efficient loading in Python
library(Matrix)
M_sparse <- as(Matrix(balassa, sparse = TRUE), "TsparseMatrix")
M_triplet <- summary(M_sparse)

# Create mapping of country/product indices to names
countries <- rownames(balassa)
products <- colnames(balassa)

# Save Balassa matrix in sparse format
balassa_file <- file.path(output_dir, "world_trade_1998_2000_balassa.csv")
write.csv(data.frame(
  country_idx = M_triplet$i - 1,  # Convert to 0-indexed
  product_idx = M_triplet$j - 1,
  value = M_triplet$x
), balassa_file, row.names = FALSE)
cat(sprintf("Saved: %s\n", balassa_file))

# Save country mapping
country_file <- file.path(output_dir, "world_trade_1998_2000_countries.csv")
write.csv(data.frame(
  idx = 0:(length(countries) - 1),
  country = countries
), country_file, row.names = FALSE)
cat(sprintf("Saved: %s\n", country_file))

# Save product mapping
product_file <- file.path(output_dir, "world_trade_1998_2000_products.csv")
write.csv(data.frame(
  idx = 0:(length(products) - 1),
  product = products
), product_file, row.names = FALSE)
cat(sprintf("Saved: %s\n", product_file))

# Save metadata
metadata_file <- file.path(output_dir, "world_trade_1998_2000_metadata.txt")
cat(sprintf("\nSaving metadata: %s\n", metadata_file))

writeLines(c(
  "World Trade Data (1998-2000 Average)",
  "=====================================",
  "",
  "Source: R economiccomplexity package v2.0.0",
  "Data: World Bank / UN Comtrade",
  "Period: Average of years 1998, 1999, 2000",
  "",
  sprintf("Countries: %d", length(countries)),
  sprintf("Products:  %d (SITC Rev. 2 classification)", length(products)),
  sprintf("Matrix density: %.4f", sum(balassa) / (nrow(balassa) * ncol(balassa))),
  "",
  "Files:",
  "  - world_trade_1998_2000.csv: Raw trade data (country, product, value)",
  "  - world_trade_1998_2000_balassa.csv: Binary Balassa index (sparse format)",
  "  - world_trade_1998_2000_countries.csv: Country index to name mapping",
  "  - world_trade_1998_2000_products.csv: Product index to name mapping",
  "",
  "Citation:",
  "  Hausmann, R., Hidalgo, C. A., Bustos, S., Coscia, M., Simoes, A., & Yildirim, M. A. (2014).",
  "  The Atlas of Economic Complexity: Mapping Paths to Prosperity.",
  "  MIT Press.",
  "",
  "R Package:",
  "  https://cran.r-project.org/package=economiccomplexity"
), metadata_file)

cat("\n======================================================================\n")
cat("COMPLETED: Demo data extracted successfully!\n")
cat("======================================================================\n\n")
cat("Python usage example:\n\n")
cat("  import pandas as pd\n")
cat("  import scipy.sparse as sp\n")
cat("  from fitkit.algorithms import fitness_complexity\n\n")
cat("  # Load Balassa index\n")
cat("  data = pd.read_csv('fitkit/data/world_trade_1998_2000_balassa.csv')\n")
cat("  countries = pd.read_csv('fitkit/data/world_trade_1998_2000_countries.csv')\n")
cat("  products = pd.read_csv('fitkit/data/world_trade_1998_2000_products.csv')\n\n")
cat("  # Create sparse matrix\n")
cat("  M = sp.csr_matrix(\n")
cat("      (data['value'], (data['country_idx'], data['product_idx'])),\n")
cat("      shape=(len(countries), len(products))\n")
cat("  )\n\n")
cat("  # Compute fitness-complexity\n")
cat("  F, Q, history = fitness_complexity(M)\n\n")
