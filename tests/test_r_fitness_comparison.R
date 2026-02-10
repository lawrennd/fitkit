#!/usr/bin/env Rscript
# Fitness-Complexity Validation: Generate R Reference Data
#
# This script generates reference data from the R economiccomplexity package
# for validating the Python fitkit implementation. See CIP-0008 for design.
#
# INSTALLATION (one-time):
#   # macOS
#   brew install --cask r
#
#   # Install R package
#   R -e 'install.packages("economiccomplexity")'
#
# USAGE:
#   Rscript tests/test_r_fitness_comparison.R
#
# OUTPUT:
#   Creates tests/r_comparison_data/ with CSV files containing:
#   - Test matrices (nested, random, modular)
#   - R fitness and complexity results
#
# NEXT STEP:
#   Run Python comparison: pytest tests/test_r_fitness_comparison.py -v
#
# REFERENCE:
#   R Package: https://cran.r-project.org/package=economiccomplexity
#   Method: Tacchella et al. (2012) Scientific Reports 2:723

library(economiccomplexity)
library(Matrix)

#' Save matrix and results to files for Python comparison
#' 
#' @param M Binary matrix (countries × products)
#' @param output_prefix Prefix for output files
#' @param method Method to use: "fitness" (default), "reflections", or "eigenvalues"
save_comparison_data <- function(M, output_prefix, method = "fitness") {
  cat(sprintf("\n=== Testing with method: %s ===\n", method))
  
  # Convert sparse matrix to base R matrix for economiccomplexity package
  M_matrix <- as.matrix(M)
  
  # Filter out rows/columns with all zeros (required for fitness-complexity)
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
  
  # Extract Fitness and Complexity
  fitness <- results$complexity_index_country  # Country fitness (ECI in their naming)
  complexity <- results$complexity_index_product  # Product complexity (PCI in their naming)
  
  # Print summary statistics
  cat(sprintf("R Fitness (country):   mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
              mean(fitness), sd(fitness), min(fitness), max(fitness)))
  cat(sprintf("R Complexity (product): mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
              mean(complexity), sd(complexity), min(complexity), max(complexity)))
  
  # Save filtered matrix as CSV (sparse format: row, col, value)
  # Note: indices here refer to the filtered matrix
  M_sparse_filtered <- as(Matrix(M_filtered, sparse = TRUE), "dgTMatrix")
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
  
  # Save dimensions
  # Save filtered matrix dimensions (not original M)
  write.csv(data.frame(
    n_countries = nrow(M_filtered),
    n_products = ncol(M_filtered)
  ), file = paste0(output_prefix, "_dims.csv"), row.names = FALSE)
  
  # Save R results
  write.csv(data.frame(
    index = 0:(length(fitness)-1),  # 0-indexed for Python
    fitness = fitness,
    complexity_country = fitness  # Same as fitness for countries
  ), file = paste0(output_prefix, "_r_fitness.csv"), row.names = FALSE)
  
  write.csv(data.frame(
    index = 0:(length(complexity)-1),
    complexity = complexity,
    complexity_product = complexity
  ), file = paste0(output_prefix, "_r_complexity.csv"), row.names = FALSE)
  
  cat(sprintf("Saved results to: %s_*.csv\n", output_prefix))
  
  invisible(list(fitness = fitness, complexity = complexity))
}

#' Create perfectly nested matrix (classic ECI/FC test case)
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

#' Create modular (block-diagonal) matrix
create_modular_matrix <- function(n_countries = 10, n_products = 20, seed = 42) {
  set.seed(seed)
  M <- Matrix(0, nrow = n_countries, ncol = n_products, sparse = TRUE)
  
  # Module 1: countries 1-5, products 1-10
  for (i in 1:5) {
    prods <- sample(1:10, size = 7, replace = FALSE)
    M[i, prods] <- 1
  }
  
  # Module 2: countries 6-10, products 11-20
  for (i in 6:10) {
    prods <- sample(11:20, size = 7, replace = FALSE)
    M[i, prods] <- 1
  }
  
  M
}

#' Main comparison test
run_comparison_tests <- function(output_dir = "tests/r_comparison_data") {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  cat("\n")
  cat("=" %R% 70)
  cat("\nR ECONOMICCOMPLEXITY PACKAGE - VALIDATION DATA GENERATION\n")
  cat("=" %R% 70)
  cat("\n")
  cat("Package version:", as.character(packageVersion("economiccomplexity")), "\n")
  cat("Output directory:", output_dir, "\n")
  
  # Test 1: Nested matrix
  cat("\n--- Test 1: Nested Matrix ---\n")
  M_nested <- create_nested_matrix()
  save_comparison_data(M_nested, file.path(output_dir, "nested"), method = "fitness")
  
  # Test 2: Random matrix
  cat("\n--- Test 2: Random Matrix ---\n")
  M_random <- create_random_matrix()
  save_comparison_data(M_random, file.path(output_dir, "random"), method = "fitness")
  
  # Test 3: Modular matrix
  cat("\n--- Test 3: Modular Matrix ---\n")
  M_modular <- create_modular_matrix()
  save_comparison_data(M_modular, file.path(output_dir, "modular"), method = "fitness")
  
  # Also test with different methods for nested case (to document behavior)
  cat("\n--- Nested Matrix with 'reflections' method ---\n")
  save_comparison_data(M_nested, file.path(output_dir, "nested_reflections"), method = "reflections")
  
  cat("\n--- Nested Matrix with 'eigenvalues' method ---\n")
  save_comparison_data(M_nested, file.path(output_dir, "nested_eigenvalues"), method = "eigenvalues")
  
  cat("\n")
  cat("=" %R% 70)
  cat("\nCOMPLETED: All test data generated successfully!\n")
  cat("=" %R% 70)
  cat("\n")
  cat("\nNext step: Run Python comparison script:\n")
  cat("  python tests/test_r_fitness_comparison.py\n\n")
}

# Allow R string repetition (for separator lines)
`%R%` <- function(str, n) paste(rep(str, n), collapse = "")

# Run if called as script
if (!interactive()) {
  run_comparison_tests()
}
