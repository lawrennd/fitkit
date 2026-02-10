#!/usr/bin/env Rscript
# ECI/PCI Validation: Generate R Reference Data
#
# This script generates reference data from the R economiccomplexity package
# for validating the Python fitkit ECI/PCI implementation. See CIP-0008 for design.
#
# INSTALLATION (one-time):
#   # macOS
#   brew install --cask r
#
#   # Install R package
#   R -e 'install.packages("economiccomplexity")'
#
# USAGE:
#   Rscript tests/test_r_eci_comparison.R
#
# OUTPUT:
#   Creates tests/r_comparison_data/ with CSV files containing:
#   - Test matrices (nested, random, modular)
#   - R ECI and PCI results using both "eigenvalues" and "reflections" methods
#
# NEXT STEP:
#   Run Python comparison: pytest tests/test_r_eci_comparison.py -v
#
# REFERENCE:
#   R Package: https://cran.r-project.org/package=economiccomplexity
#   Method: Hidalgo & Hausmann (2009) PNAS 106(26):10570-10575

library(economiccomplexity)
library(Matrix)

#' Save matrix and ECI/PCI results to files for Python comparison
#' 
#' @param M Binary matrix (countries × products)
#' @param output_prefix Prefix for output files
#' @param method Method to use: "eigenvalues" (default) or "reflections"
save_comparison_data <- function(M, output_prefix, method = "eigenvalues") {
  cat(sprintf("\n=== Testing with method: %s ===\n", method))
  
  # Convert sparse matrix to base R matrix for economiccomplexity package
  M_matrix <- as.matrix(M)
  
  # Filter out rows/columns with all zeros (required for complexity measures)
  row_sums <- rowSums(M_matrix)
  col_sums <- colSums(M_matrix)
  valid_rows <- which(row_sums > 0)
  valid_cols <- which(col_sums > 0)
  
  cat(sprintf("Original matrix dimensions: %d x %d\n", nrow(M_matrix), ncol(M_matrix)))
  cat(sprintf("Filtered matrix dimensions: %d x %d\n", length(valid_rows), length(valid_cols)))
  
  M_filtered <- M_matrix[valid_rows, valid_cols, drop = FALSE]
  
  # Add row and column names (required by economiccomplexity package)
  rownames(M_filtered) <- paste0("c", 1:nrow(M_filtered))
  colnames(M_filtered) <- paste0("p", 1:ncol(M_filtered))
  
  # Compute complexity measures using specified method
  # Use 200 iterations to match Python (R default is only 20)
  results <- complexity_measures(balassa_index = M_filtered, method = method, iterations = 200)
  
  # Extract ECI and PCI
  eci <- results$complexity_index_country  # Economic Complexity Index
  pci <- results$complexity_index_product  # Product Complexity Index
  
  # Print summary statistics
  cat(sprintf("R ECI (country):   mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
              mean(eci), sd(eci), min(eci), max(eci)))
  cat(sprintf("R PCI (product):   mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
              mean(pci), sd(pci), min(pci), max(pci)))
  
  # Save filtered matrix as CSV (sparse format: row, col, value)
  # Note: indices here refer to the filtered matrix
  M_sparse_filtered <- as(Matrix(M_filtered, sparse = TRUE), "TsparseMatrix")
  M_triplet <- summary(M_sparse_filtered)
  write.csv(data.frame(
    row = M_triplet$i - 1,  # Convert to 0-indexed
    col = M_triplet$j - 1,
    value = M_triplet$x
  ), file = paste0(output_prefix, "_matrix.csv"), row.names = FALSE)
  
  # Save row/column mapping (which rows/cols from original were kept)
  write.csv(data.frame(
    original_row = valid_rows - 1,  # Convert to 0-indexed
    filtered_row = 0:(length(valid_rows)-1)
  ), file = paste0(output_prefix, "_row_mapping.csv"), row.names = FALSE)
  
  write.csv(data.frame(
    original_col = valid_cols - 1,  # Convert to 0-indexed
    filtered_col = 0:(length(valid_cols)-1)
  ), file = paste0(output_prefix, "_col_mapping.csv"), row.names = FALSE)
  
  # Save filtered matrix dimensions (not original M)
  write.csv(data.frame(
    n_countries = nrow(M_filtered),
    n_products = ncol(M_filtered)
  ), file = paste0(output_prefix, "_dims.csv"), row.names = FALSE)
  
  # Save R results
  write.csv(data.frame(
    index = 0:(length(eci)-1),  # 0-indexed for Python
    eci = eci,
    complexity_country = eci  # Same as ECI for countries
  ), file = paste0(output_prefix, "_r_eci.csv"), row.names = FALSE)
  
  write.csv(data.frame(
    index = 0:(length(pci)-1),
    pci = pci,
    complexity_product = pci
  ), file = paste0(output_prefix, "_r_pci.csv"), row.names = FALSE)
  
  cat(sprintf("Saved results to: %s_*.csv\n", output_prefix))
  
  invisible(list(eci = eci, pci = pci))
}

#' Create perfectly nested matrix (classic ECI test case)
create_nested_matrix <- function(n_countries = 20, n_products = 30) {
  M <- Matrix(0, nrow = n_countries, ncol = n_products, sparse = TRUE)
  for (i in 1:n_countries) {
    # Each country includes all products up to its index
    n_prods <- min(i + 4, n_products)
    M[i, 1:n_prods] <- 1
  }
  M
}

#' Create random sparse matrix
create_random_matrix <- function(n_countries = 50, n_products = 75, density = 0.15, seed = 42) {
  set.seed(seed)
  M <- Matrix(0, nrow = n_countries, ncol = n_products, sparse = TRUE)
  n_entries <- round(n_countries * n_products * density)
  rows <- sample(1:n_countries, n_entries, replace = TRUE)
  cols <- sample(1:n_products, n_entries, replace = TRUE)
  for (k in 1:n_entries) {
    M[rows[k], cols[k]] <- 1
  }
  M
}

#' Create modular block-diagonal matrix
create_modular_matrix <- function(n_blocks = 2, countries_per_block = 5, products_per_block = 10) {
  n_countries <- n_blocks * countries_per_block
  n_products <- n_blocks * products_per_block
  M <- Matrix(0, nrow = n_countries, ncol = n_products, sparse = TRUE)
  
  for (b in 1:n_blocks) {
    row_start <- (b - 1) * countries_per_block + 1
    row_end <- b * countries_per_block
    col_start <- (b - 1) * products_per_block + 1
    col_end <- b * products_per_block
    M[row_start:row_end, col_start:col_end] <- 1
  }
  M
}

#' Main execution
run_comparison_tests <- function() {
  # Create output directory
  output_dir <- "tests/r_comparison_data"
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("\n")
  cat("======================================================================\n")
  cat("R ECONOMICCOMPLEXITY PACKAGE - ECI/PCI VALIDATION DATA GENERATION\n")
  cat("======================================================================\n")
  cat(sprintf("Package version: %s \n", packageVersion("economiccomplexity")))
  cat(sprintf("Output directory: %s \n", output_dir))
  
  # Test 1: Nested matrix
  cat("\n--- Test 1: Nested Matrix (Eigenvalues Method) ---\n")
  M_nested <- create_nested_matrix()
  save_comparison_data(M_nested, file.path(output_dir, "eci_nested"), method = "eigenvalues")
  
  # Test 2: Random matrix
  cat("\n--- Test 2: Random Matrix (Eigenvalues Method) ---\n")
  M_random <- create_random_matrix()
  save_comparison_data(M_random, file.path(output_dir, "eci_random"), method = "eigenvalues")
  
  # Test 3: Modular matrix
  cat("\n--- Test 3: Modular Matrix (Eigenvalues Method) ---\n")
  M_modular <- create_modular_matrix()
  save_comparison_data(M_modular, file.path(output_dir, "eci_modular"), method = "eigenvalues")
  
  # Test 4: Nested with reflections method (iterative alternative)
  cat("\n--- Test 4: Nested Matrix (Reflections Method) ---\n")
  save_comparison_data(M_nested, file.path(output_dir, "eci_nested_reflections"), method = "reflections")
  
  cat("\n")
  cat("======================================================================\n")
  cat("COMPLETED: All ECI/PCI test data generated successfully!\n")
  cat("======================================================================\n")
  cat("\nNext step: Run Python comparison script:\n")
  cat("  python tests/test_r_eci_comparison.py\n\n")
}

# Run the tests
run_comparison_tests()
