# Backward kNN Imputation for Time Series Data
# =========================================================

# Load required libraries
library(dplyr)
library(data.table)

# Main imputation function
impute_missing_values <- function(data, variables_to_impute, k = 5, max_days_lookback = 30) {
  # Start timing
  start_time <- Sys.time()
  
  # Ensure time is in POSIXct format
  if (!inherits(data$time, "POSIXct")) {
    data$time <- as.POSIXct(data$time)
  }
  
  # Create summary of missing values before imputation
  missing_before <- sapply(variables_to_impute, function(var) {
    var_data <- data[data$variable == var, ]
    return(sum(is.na(var_data$value)))
  })
  
  cat("Missing values before imputation:\n")
  print(data.frame(variable = variables_to_impute, 
                   missing_count = missing_before))
  
  # Convert to data.table for faster processing
  dt <- as.data.table(data)
  setkey(dt, id, variable, time)  # Set keys for faster lookups
  
  # Calculate global means for each variable
  variable_means <- sapply(variables_to_impute, function(var) {
    mean_val <- mean(dt[variable == var, value], na.rm = TRUE)
    # Round and constrain to -2 to 2 range
    mean_val <- round(mean_val)
    mean_val <- max(-2, min(2, mean_val))
    return(mean_val)
  })
  
  # Initialize counters
  total_nas <- sum(missing_before)
  processed <- 0
  knn_count <- 0
  mean_count <- 0
  
  cat("Total NAs to impute:", total_nas, "\n")
  
  # Process each variable
  for (var in variables_to_impute) {
    # Skip if no NAs for this variable
    if (missing_before[var] == 0) {
      cat("Skipping", var, "- no missing values\n")
      next
    }
    
    cat("Processing variable:", var, "\n")
    
    # Get global mean for this variable
    global_mean <- variable_means[var]
    
    # Extract data for this variable
    var_dt <- dt[variable == var]
    
    # Find rows with NA values
    na_rows <- which(is.na(var_dt$value))
    
    if (length(na_rows) == 0) {
      next
    }
    
    # Get unique users with missing values
    users_with_na <- unique(var_dt[na_rows]$id)
    
    # Process each user
    for (user_id in users_with_na) {
      # Get data for this user and variable
      user_var_dt <- var_dt[id == user_id]
      setkey(user_var_dt, time)  # Sort by time
      
      # Find NA rows for this user
      user_na_rows <- which(is.na(user_var_dt$value))
      
      if (length(user_na_rows) == 0) next
      
      # Process each NA value
      for (i in seq_along(user_na_rows)) {
        row_idx <- user_na_rows[i]
        current_time <- user_var_dt$time[row_idx]
        
        # Get previous observations
        prev_data <- user_var_dt[time < current_time]
        
        # Check if this is the first record (no previous data)
        if (nrow(prev_data) == 0) {
          # First record - use global mean
          var_dt[id == user_id & time == current_time, value := global_mean]
          
          # Update counters
          processed <- processed + 1
          mean_count <- mean_count + 1
          
          # Display progress
          if (processed %% 10 == 0 || processed == total_nas) {
            cat("Progress:", processed, "/", total_nas, 
                sprintf("(%.1f%%)", 100 * processed / total_nas), "\n")
          }
          
          next
        }
        
        # Calculate time differences
        time_diffs <- as.numeric(difftime(current_time, prev_data$time, units = "days"))
        
        # Find valid indices within lookback period
        valid_indices <- which(time_diffs >= 0 & time_diffs <= max_days_lookback)
        
        # If no valid previous values or all are NA, use global mean
        if (length(valid_indices) == 0 || all(is.na(prev_data$value[valid_indices]))) {
          var_dt[id == user_id & time == current_time, value := global_mean]
          
          # Update counters
          processed <- processed + 1
          mean_count <- mean_count + 1
          
          # Display progress
          if (processed %% 10 == 0 || processed == total_nas) {
            cat("Progress:", processed, "/", total_nas, 
                sprintf("(%.1f%%)", 100 * processed / total_nas), "\n")
          }
          
          next
        }
        
        # There are valid previous values, use kNN imputation
        # Take k nearest or all available if fewer
        k_actual <- min(k, length(valid_indices))
        
        # Sort by time difference and select k nearest
        nearest_indices <- valid_indices[order(time_diffs[valid_indices])][1:k_actual]
        
        # Get weights based on time difference
        nearest_diffs <- time_diffs[nearest_indices]
        weights <- 1 / (nearest_diffs + 0.01)
        
        # Normalize weights
        weights <- weights / sum(weights)
        
        # Get values for imputation
        prev_values <- prev_data$value[nearest_indices]
        
        # Handle NA values in neighbors
        na_neighbors <- is.na(prev_values)
        if (any(na_neighbors)) {
          weights[na_neighbors] <- 0
          if (sum(weights) > 0) {
            weights <- weights / sum(weights)
          } else {
            # Use global mean if all weights become zero
            var_dt[id == user_id & time == current_time, value := global_mean]
            
            # Update counters
            processed <- processed + 1
            mean_count <- mean_count + 1
            
            # Display progress
            if (processed %% 10 == 0 || processed == total_nas) {
              cat("Progress:", processed, "/", total_nas, 
                  sprintf("(%.1f%%)", 100 * processed / total_nas), "\n")
            }
            
            next
          }
        }
        
        # Calculate weighted average
        imputed_value <- sum(prev_values * weights, na.rm = TRUE)
        
        # Round to nearest integer between -2 and 2
        imputed_value <- round(imputed_value)
        imputed_value <- max(-2, min(2, imputed_value))
        
        # Update the value
        var_dt[id == user_id & time == current_time, value := imputed_value]
        
        # Update counters
        processed <- processed + 1
        knn_count <- knn_count + 1
        
        # Display progress
        if (processed %% 10 == 0 || processed == total_nas) {
          cat("Progress:", processed, "/", total_nas, 
              sprintf("(%.1f%%)", 100 * processed / total_nas), "\n")
        }
      }
    }
    
    # Update main data.table with imputed values
    dt[variable == var, value := var_dt$value]
  }
  
  # Convert back to data.frame
  imputed_data <- as.data.frame(dt)
  
  # Create summary of missing values after imputation
  missing_after <- sapply(variables_to_impute, function(var) {
    var_data <- imputed_data[imputed_data$variable == var, ]
    return(sum(is.na(var_data$value)))
  })
  
  # End timing
  end_time <- Sys.time()
  time_taken <- end_time - start_time
  
  cat("\nImputation completed in:", format(time_taken), "\n")
  cat("Values imputed using kNN:", knn_count, "\n")
  cat("Values imputed using global mean:", mean_count, "\n")
  
  # Print imputation results
  cat("\nMissing values after imputation:\n")
  
  imputation_summary <- data.frame(
    variable = variables_to_impute, 
    missing_before = missing_before,
    missing_after = missing_after,
    imputed_count = missing_before - missing_after,
    imputed_percent = round(100 * (missing_before - missing_after) / 
                              pmax(missing_before, 1), 2)
  )
  
  print(imputation_summary)
  
  return(imputed_data)
}

# Main execution
# ===============================================

cat("Reading dataset...\n")
data_path <- "data/cleaned_data.csv"

# Check if the cleaned data exists, otherwise use the original dataset
if (file.exists(data_path)) {
  data <- read.csv(data_path, stringsAsFactors = FALSE)
  cat("Loaded cleaned dataset after outlier removal\n")
} else {
  data <- read.csv("dataset_mood_smartphone.csv", stringsAsFactors = FALSE)
  cat("Warning: Cleaned dataset not found. Using original dataset instead\n")
}

# Convert time column to POSIXct for processing
data$time <- as.POSIXct(data$time)

# Identify variables with missing values
variable_na_counts <- aggregate(is.na(value) ~ variable, data = data, FUN = sum)
variables_with_na <- variable_na_counts$variable[variable_na_counts$`is.na(value)` > 0]

cat("Variables with missing values:\n")
print(variable_na_counts[variable_na_counts$`is.na(value)` > 0, ])

# If no variables have NAs, stop
if (length(variables_with_na) == 0) {
  cat("No variables have missing values. Imputation not needed.\n")
} else {
  # Set parameters for imputation
  k_neighbors <- 5  # Number of neighbors to use
  max_lookback_days <- 30  # Maximum days to look back
  
  # Create a copy of the original data for comparison
  original_data <- data
  
  # Apply imputation
  cat("\nApplying backward kNN imputation with k =", k_neighbors, 
      "and max lookback of", max_lookback_days, "days...\n")
  
  imputed_data <- impute_missing_values(data, 
                                        variables_with_na, 
                                        k = k_neighbors, 
                                        max_days_lookback = max_lookback_days)
  
  # Verify all NAs are gone
  remaining_nas <- FALSE
  for (var in variables_with_na) {
    na_count <- sum(is.na(imputed_data$value[imputed_data$variable == var]))
    if (na_count > 0) {
      cat("WARNING: Variable", var, "still has", na_count, "missing values!\n")
      remaining_nas <- TRUE
    }
  }
  
  if (!remaining_nas) {
    cat("SUCCESS: All missing values were successfully imputed.\n")
  }
  
  # Most reliable approach - format all times consistently with milliseconds
  imputed_data$time <- format(imputed_data$time, "%Y-%m-%d %H:%M:%OS3")
  
  # Write the imputed data to file
  write.csv(imputed_data, file.path("data", "imputed_data.csv"), row.names = FALSE)
  
  cat("\nImputation complete. Imputed data saved to 'data/imputed_data.csv'\n")
  
  # Generate basic imputation summary
  cat("\nFinal Imputation Summary:\n")
  cat("------------------------\n")
  for (var in variables_with_na) {
    original_na <- sum(is.na(original_data$value[original_data$variable == var]))
    imputed_na <- sum(is.na(imputed_data$value[imputed_data$variable == var]))
    
    cat(sprintf("Variable: %s\n", var))
    cat(sprintf("  Values imputed: %d/%d (%.1f%%)\n", 
                original_na - imputed_na, original_na, 
                100 * (original_na - imputed_na) / original_na))
    
    # Verify no NAs remain
    if (imputed_na > 0) {
      cat("  WARNING: There are still", imputed_na, "missing values after imputation.\n")
    } else {
      cat("  SUCCESS: All missing values were successfully imputed.\n")
    }
  }
}

