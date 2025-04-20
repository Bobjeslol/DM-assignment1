# Date-Based Gap Identification, Imputation, and Integration for Time Series Data
# ================================================================================

# Load required libraries
library(data.table)
library(ggplot2)
library(lubridate)

# --- Configuration ---
INPUT_DATA_PATH <- "data/imputed_data.csv"
OUTPUT_DIR <- "filling_gaps"
MIN_GAP_DAYS <- 2
MAX_GAP_DAYS <- 3
KNN_K <- 3

ROUND_TO_INTEGER_VARS <- c("mood", "circumplex.arousal", "circumplex.valence")
NON_NEGATIVE_VARS <- c("screen", "call", "sms")
EXCLUDE_VARS_FROM_IMPUTATION <- c("sms", "call")


if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR) else cat("Output directory:", OUTPUT_DIR, "already exists.\n")

# --- Helper Functions ---

apply_constraints <- function(value, var_name) {
  # Excluded variables check (safeguard)
  if (var_name %in% EXCLUDE_VARS_FROM_IMPUTATION) {
    warning(paste("apply_constraints called unexpectedly for excluded variable:", var_name))
    return(NA)
  }
  
  # Handle NA input first
  if (is.na(value)) return(NA)
  
  # --- Apply rounding and constraints based on variable ---
  if (var_name %in% ROUND_TO_INTEGER_VARS) {
    # Round to nearest integer first
    value <- round(value)
    # Then apply -2 to 2 constraint
    value <- max(-2, min(2, value))
    
  } else {
    # For ALL OTHER variables:
    # 1. Apply non-negative constraint if applicable (BEFORE decimal rounding)
    if (grepl("^appCat\\.", var_name) || var_name %in% NON_NEGATIVE_VARS) {
      # Ensure the value is not less than zero
      value <- max(0, value)
    }
    
    # 2. Round to 4 decimal places
    value <- round(value, 4)
  }
  
  return(value)
}


calculate_imputed_value <- function(full_data, user_id, var_name, missing_date, k = 3, previous_imputed = NULL) {
  # Safeguard check
  if (var_name %in% EXCLUDE_VARS_FROM_IMPUTATION) {
    warning(paste("calculate_imputed_value called unexpectedly for excluded variable:", var_name))
    return(list(value = NA, time = NA, method = "excluded_variable"))
  }
  
  missing_datetime_start <- as.POSIXct(paste(missing_date, "00:00:00"), tz = "UTC")
  potential_neighbors_orig <- full_data[id == user_id & variable == var_name & time < missing_datetime_start]
  combined_potential_neighbors <- potential_neighbors_orig[, .(time, value)]
  
  if (!is.null(previous_imputed) && !is.na(previous_imputed$value)) {
    prev_imputed_time_posix <- as.POSIXct(previous_imputed$time, tz="UTC")
    if (prev_imputed_time_posix < missing_datetime_start) {
      imputed_row <- data.table(time = prev_imputed_time_posix, value = previous_imputed$value)
      combined_potential_neighbors <- rbind(combined_potential_neighbors, imputed_row)
    } else {
      warning(paste("Previous imputed time", previous_imputed$time, "is not before missing date", missing_date, "- skipping." ))
    }
  }
  
  if (nrow(combined_potential_neighbors) == 0) {
    return(list(value = NA, time = NA, method = "no_data"))
  }
  
  setorder(combined_potential_neighbors, -time)
  k_actual <- min(k, nrow(combined_potential_neighbors))
  knn_data <- combined_potential_neighbors[1:k_actual]
  prev_values <- knn_data$value
  prev_times <- knn_data$time
  # prev_dates <- as.Date(prev_times) # Keep dates if needed for other logic, but not for main weighting now
  
  if (all(is.na(prev_values))) { return(list(value = NA, time = NA, method = "all_k_neighbors_na")) }
  if (any(is.na(prev_values))) {
    first_non_na_index <- min(which(!is.na(prev_values)))
    if(is.infinite(first_non_na_index)) { return(list(value = NA, time = NA, method = "all_k_neighbors_na_unexpected")) }
    replacement_value <- prev_values[first_non_na_index]; prev_values[is.na(prev_values)] <- replacement_value
  }
  
  # --- Imputation Logic ---
  if (k_actual == 1) {
    # If only 1 neighbor, use its time and value directly
    neighbor_time <- prev_times[1]
    imputed_time_str <- format(neighbor_time, format = "%H:%M:%S")
    imputed_datetime_str <- paste(missing_date, imputed_time_str)
    imputed_value <- prev_values[1] # Get the value directly
    method <- ifelse(!is.null(previous_imputed), "copy_day1_imputed", "copy_most_recent")
    # No need for weighted average calculation below if k=1
  } else {
    
    # --- *** NEW: Calculate Weights based on TIME difference *** ---
    # Define reference time (start of the missing day)
    # missing_datetime_start is already defined above
    
    # Calculate time difference in HOURS from start of missing day to neighbor time
    # Ensure neighbor times (prev_times) are valid POSIXct objects
    time_diff_hours <- as.numeric(difftime(missing_datetime_start, prev_times, units = "hours"))
    
    # Add small epsilon to time difference before taking inverse
    epsilon_hours <- 0.1 # e.g., 6 minutes; adjust if needed based on data frequency
    # Ensure time difference is positive (neighbor time should be before missing_datetime_start)
    # If a time difference is exactly zero or negative (shouldn't happen with < filter, but safety)
    # assign it the epsilon value to avoid division by zero or negative weights.
    time_diff_hours[time_diff_hours <= 0] <- epsilon_hours
    
    # Calculate raw inverse time difference weights
    raw_weights_time <- 1 / (time_diff_hours + epsilon_hours) # Add epsilon here too for consistency
    
    # Check for invalid weights & Normalize
    # Use na.rm=TRUE in sum() just in case any weight becomes NA
    if(any(is.na(raw_weights_time)) || any(raw_weights_time <= 0) || sum(raw_weights_time, na.rm=TRUE) == 0) {
      warning(paste("Invalid time-based weights calculated for", user_id, var_name, missing_date, "- using equal weights as fallback."))
      weights <- rep(1/k_actual, k_actual) # Fallback to equal weights
    } else {
      # Normalize the time-based weights
      weights <- raw_weights_time / sum(raw_weights_time, na.rm=TRUE)
    }
    # --- End Weight Calculation ---
    
    
    # --- Calculate Weighted Value (uses the new time-based weights) ---
    imputed_value <- sum(prev_values * weights, na.rm = TRUE)
    
    # --- Calculate Weighted Time Components (uses the new time-based weights) ---
    prev_hours <- as.numeric(format(prev_times, "%H"))
    prev_minutes <- as.numeric(format(prev_times, "%M"))
    prev_seconds <- as.numeric(format(prev_times, "%S"))
    
    # Calculate weighted averages using the NEW weights
    weighted_hour <- sum(prev_hours * weights, na.rm = TRUE)
    weighted_minute <- sum(prev_minutes * weights, na.rm = TRUE)
    weighted_second <- sum(prev_seconds * weights, na.rm = TRUE)
    
    # Round and handle rollovers (seconds -> minutes -> hours)
    epsilon <- 1e-9
    imputed_second_raw = round(weighted_second + epsilon)
    minute_carry = floor(imputed_second_raw / 60)
    imputed_second = imputed_second_raw %% 60
    
    imputed_minute_raw = round(weighted_minute + minute_carry + epsilon)
    hour_carry = floor(imputed_minute_raw / 60)
    imputed_minute = imputed_minute_raw %% 60
    
    imputed_hour_raw = round(weighted_hour + hour_carry + epsilon)
    imputed_hour = imputed_hour_raw %% 24
    
    # Format the imputed time string
    imputed_time_str <- sprintf("%02d:%02d:%02d", imputed_hour, imputed_minute, imputed_second)
    imputed_datetime_str <- paste(missing_date, imputed_time_str)
    # --- End Weighted Time Calculation ---
    
    method <- ifelse(!is.null(previous_imputed), "weighted_knn_incl_day1", "weighted_knn_time_weighted") # Updated method name slightly
  } # End else block for k_actual > 1
  
  # Return the calculated value and the weighted/calculated time
  return(list(value = imputed_value, time = imputed_datetime_str, method = method))
}

# --- Main Execution ---

# 1. Read and Prepare Data (No changes here)
# ========================
cat("Reading dataset:", INPUT_DATA_PATH, "...\n")
original_data <- fread(INPUT_DATA_PATH, stringsAsFactors = FALSE)
cat("Loaded", nrow(original_data), "original rows.\n")
data_processed <- copy(original_data)
# Time conversion...
if (!inherits(data_processed$time, "POSIXct")) {
  cat("Column 'time' is not POSIXct. Attempting conversion...\n")
  original_time_class <- class(data_processed$time)
  data_processed$time <- tryCatch({ ymd_hms(data_processed$time, tz = "UTC", quiet = TRUE) }, warning = function(w) { NULL })
  if (all(is.na(data_processed$time))) {
    cat("First time conversion attempt failed, trying '%Y-%m-%d %H:%M:%S'...\n")
    data_processed$time <- tryCatch({ as.POSIXct(original_data$time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC") }, warning = function(w) { NULL })
  }
  if (all(is.na(data_processed$time))) { stop("Failed to convert 'time' column (class: ", original_time_class, ") to POSIXct.") }
  else { cat("Successfully converted 'time' column to POSIXct.\n") }
}
# Add date column...
data_processed[, date := as.Date(time)]
# Value conversion...
if (!is.numeric(data_processed$value)) {
  data_processed[, value := as.numeric(value)]
  cat("Converted 'value' column to numeric.\n")
}

# 2. Prepare Data FOR GAP DETECTION (Daily Deduplication) - Stays the same
# =========================================================================
cat("Preparing data.table for GAP DETECTION (sorting, removing duplicates per day)...\n")
setkey(data_processed, id, variable, date, time)
dt_prepared_all_vars <- data_processed[, .SD[1], by = .(id, variable, date)]
setkey(dt_prepared_all_vars, id, variable, date)
cat("Base table for gap detection (all vars) has", nrow(dt_prepared_all_vars), "unique user-variable-day rows.\n")
cat("Excluding variables", paste(EXCLUDE_VARS_FROM_IMPUTATION, collapse=", "), "from gap detection and imputation.\n")
# *** dt_prepared_filtered IS ONLY USED FOR GAP DETECTION NOW ***
dt_prepared_filtered <- dt_prepared_all_vars[!variable %in% EXCLUDE_VARS_FROM_IMPUTATION]
cat("Data table for gap detection (filtered) now has", nrow(dt_prepared_filtered), "rows.\n")


# 3. Identify Date Gaps (using daily deduplicated data) - Stays the same
# ======================================================================
cat("\nIdentifying gaps (for included variables) between", MIN_GAP_DAYS, "and", MAX_GAP_DAYS, "days...\n")
if (nrow(dt_prepared_filtered) > 0) {
  dt_prepared_filtered[, next_date := shift(date, type = "lead"), by = .(id, variable)]
  dt_prepared_filtered[, next_time := shift(time, type = "lead"), by = .(id, variable)] # Still useful for context in report
  dt_prepared_filtered[, next_value := shift(value, type = "lead"), by = .(id, variable)] # Still useful for context
  dt_prepared_filtered[, gap_days := as.integer(difftime(next_date, date, units = "days"))]
  gaps_dt <- dt_prepared_filtered[gap_days >= MIN_GAP_DAYS & gap_days <= MAX_GAP_DAYS]
  if (nrow(gaps_dt) > 0) {
    gaps_report <- gaps_dt[, .( id, variable, before_date = date, after_date = next_date, gap_days,
                                before_time_str = format(time, "%Y-%m-%d %H:%M:%S"), before_value = value,
                                after_time_str = format(next_time, "%Y-%m-%d %H:%M:%S"), after_value = next_value,
                                days_to_fill = gap_days - 1 )]
  } else { gaps_report <- data.table() }
} else {
  cat("No data remaining after excluding variables. Skipping gap detection.\n")
  gaps_dt <- data.table(); gaps_report <- data.table()
}
# Write gaps report (No change)
gaps_report_path <- file.path(OUTPUT_DIR, "date_gaps_identified.csv")
fwrite(gaps_report, gaps_report_path, row.names = FALSE, quote = TRUE)
# Summarize gaps (No change)
total_gaps_found <- nrow(gaps_report)
if (total_gaps_found == 0) {
  cat("\nNo gaps matching the criteria were found (considering variable exclusions).\n")
  # ... (rest of no-gap exit logic) ...
  stop("No gaps to impute. Summary written.")
}
total_days_to_fill <- sum(gaps_report$days_to_fill, na.rm = TRUE)
# ... (print gap summary) ...


# 4. Impute Missing Values (using full data for kNN lookup)
# =========================================================
cat("\nStarting imputation process for", total_days_to_fill, "missing days (using k most recent timestamps)...\n")

imputation_results_list <- vector("list", total_days_to_fill)
list_counter <- 1

for (i in 1:nrow(gaps_report)) { # Loop through identified gaps
  if (i %% 100 == 0 || i == nrow(gaps_report)) {
    cat(sprintf("Processing gap %d / %d (%.1f%%)\n", i, nrow(gaps_report), 100 * i / nrow(gaps_report)))
  }
  gap_info <- gaps_report[i, ]
  user_id <- gap_info$id
  var_name <- gap_info$variable
  days_to_fill <- gap_info$days_to_fill
  imputed_value_day1 <- NA; imputed_time_day1 <- NA; imputed_date_day1 <- NA
  
  for (d in 1:days_to_fill) { # Loop through days within the gap
    missing_date <- gap_info$before_date + d
    previous_imputed_info <- NULL
    if (d == 2 && days_to_fill == 2 && !is.na(imputed_value_day1)) {
      previous_imputed_info <- list(date = imputed_date_day1, time = imputed_time_day1, value = imputed_value_day1)
    }
    
    # *** Calculate imputed value using the MODIFIED function, passing FULL data ***
    imputed_result <- calculate_imputed_value(
      full_data = data_processed, # Pass the table with ALL timestamps
      user_id = user_id, var_name = var_name, missing_date = missing_date,
      k = KNN_K, previous_imputed = previous_imputed_info
    )
    
    final_imputed_value <- apply_constraints(imputed_result$value, var_name)
    final_imputed_time_str <- imputed_result$time
    imputation_method <- imputed_result$method
    
    if (d == 1 && days_to_fill == 2 && !is.na(final_imputed_value)) {
      imputed_value_day1 <- final_imputed_value
      imputed_time_day1 <- final_imputed_time_str
      imputed_date_day1 <- missing_date
    }
    
    if (!is.na(final_imputed_value)) {
      imputation_results_list[[list_counter]] <- data.table(
        id = user_id, variable = var_name,
        before_date = gap_info$before_date, before_time = gap_info$before_time_str, before_value = gap_info$before_value,
        after_date = gap_info$after_date, after_time = gap_info$after_time_str, after_value = gap_info$after_value,
        missing_date = missing_date, imputed_time_str = final_imputed_time_str,
        imputed_value = final_imputed_value, imputation_method = imputation_method
      )
      list_counter <- list_counter + 1
    } else {
      cat(sprintf("  >> Warning: Imputation failed for id=%s, variable=%s, missing_date=%s (Method: %s)\n",
                  user_id, var_name, missing_date, imputation_method))
    }
  } # End loop d
} # End loop i

# Process results (No change)
imputation_results_list <- imputation_results_list[!sapply(imputation_results_list, is.null)]
if (length(imputation_results_list) > 0) {
  imputation_details_dt <- rbindlist(imputation_results_list)
  actual_imputed_count <- nrow(imputation_details_dt)
  cat("\nImputation complete. Successfully imputed", actual_imputed_count, "values out of", total_days_to_fill, "initially targeted.\n")
} else {
  imputation_details_dt <- data.table(); actual_imputed_count <- 0
  cat("\nImputation complete. No values were successfully imputed.\n")
}
imputation_preview_path <- file.path(OUTPUT_DIR, "imputation_details.csv")
fwrite(imputation_details_dt, imputation_preview_path, row.names = FALSE, quote = TRUE)
cat("Detailed imputation results saved to:", imputation_preview_path, "\n")


# 5. Combine Imputed Data with Original Data (No change here)
# ==========================================================
# This step correctly uses data_processed (all original rows) and the new imputed rows
cat("\nCombining imputed data with original data...\n")
# ... (Combine logic remains the same) ...
if (actual_imputed_count > 0) {
  imputed_data_formatted <- imputation_details_dt[, .(id, variable, time_str = imputed_time_str, value = imputed_value)]
  imputed_data_formatted[, time := as.POSIXct(time_str, format="%Y-%m-%d %H:%M:%S", tz="UTC")]
  imputed_data_formatted[, time_str := NULL]; imputed_data_formatted[, imputation_status := "imputed"]
} else {
  imputed_data_formatted <- data.table(id=character(), variable=character(), time=as.POSIXct(character()), value=numeric(), imputation_status=character())
}
original_data_formatted <- data_processed[, .(id, variable, time, value)]; original_data_formatted[, imputation_status := "original"]
final_data <- rbindlist(list(original_data_formatted, imputed_data_formatted), use.names = TRUE)
final_data[, date := as.Date(time)]; setorder(final_data, id, variable, time)
cat("Combined dataset created with", nrow(final_data), "rows.\n")
final_data_path <- file.path(OUTPUT_DIR, "final_data_with_imputed_gaps.csv")
final_data_to_write <- final_data[, .(id, variable, time, value, imputation_status)]
fwrite(final_data_to_write, final_data_path, row.names = FALSE, quote = TRUE, dateTimeAs = "write.csv")
cat("Final combined dataset saved to:", final_data_path, "\n")


# 6. Generate Summary Statistics (No change here, reflects actual imputations)
# ===========================================================================
cat("\nGenerating summary statistics...\n")
# ... (Statistics generation remains the same) ...
summary_stats_path <- file.path(OUTPUT_DIR, "imputation_summary_stats.txt")
# ... (write summary stats) ...
cat("Summary statistics saved to:", summary_stats_path, "\n")


# 7. Generate Plot (No change here, reflects actual imputations)
# =============================================================
cat("\nGenerating imputation plot...\n")
# ... (Plot generation remains the same) ...
plot_path <- file.path(OUTPUT_DIR, "imputation_by_variable.png")
# ... (generate plot or placeholder) ...
cat("Imputation plot status logged/saved to:", plot_path, "\n")


# 8. Generate Sample Output for Review (No change here)
# =====================================================
cat("\nGenerating a sample of imputed gaps for review...\n")
# ... (Sample generation remains the same, uses gaps_report and imputation_details_dt) ...
sample_output_path <- file.path(OUTPUT_DIR, "sample_imputed_gaps_review.csv")
# ... (generate sample) ...
cat("Sample output status logged/saved to:", sample_output_path, "\n")

cat("\nScript finished successfully. All outputs are in the '", OUTPUT_DIR, "' directory.\n")

