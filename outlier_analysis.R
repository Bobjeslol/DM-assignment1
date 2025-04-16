# Comprehensive Distribution-Based LOF Outlier Analysis
# =====================================================================

# Load required libraries
library(dplyr)
library(dbscan)
library(ggplot2)
library(fitdistrplus)  # For distribution fitting
library(MASS)          # For additional distribution functions
library(gridExtra)     # For arranging multiple plots
library(EnvStats)      # For additional goodness of fit tests
library(actuar)        # For additional distributions

# Create directory for saving images
dir.create("outlier_analysis", showWarnings = FALSE)

# Read the dataset
data <- read.csv("data/dataset_mood_smartphone.csv")

# Define the usage variables (screen and app categories)
app_cat_vars <- unique(data$variable)[grep("^appCat\\.", unique(data$variable))]
usage_vars <- c("screen", app_cat_vars)

# Create empty dataframes to store results
lof_outlier_summary <- data.frame(
  variable = character(),
  total_points = integer(),
  outlier_count = integer(),
  outlier_percent = numeric(),
  mean_value = numeric(),
  max_value = numeric(),
  outlier_mean = numeric()
)

distribution_fit_summary <- data.frame(
  variable = character(),
  stage = character(),
  best_distribution = character(),
  aic = numeric(),
  bic = numeric(),
  ks_stat = numeric(),
  ks_pvalue = numeric(),
  distribution_params = character()
)

# Define the distributions to test
distributions_to_test <- c("norm", "lnorm", "gamma", "exp")

# Function to fit distributions and identify the best one
fit_best_distribution <- function(data_vector, variable_name, stage = "before") {
  # Handle edge cases
  if (length(data_vector) <= 3 || length(unique(data_vector)) <= 1) {
    return(data.frame(
      variable = variable_name,
      stage = stage,
      best_distribution = "insufficient_data",
      aic = NA,
      bic = NA,
      ks_stat = NA,
      ks_pvalue = NA,
      distribution_params = "NA"
    ))
  }
  
  # Remove any zero or negative values for certain distributions
  data_vector_pos <- data_vector[data_vector > 0]
  
  # If too many values were removed, log this and continue with what we have
  if (length(data_vector_pos) < length(data_vector) * 0.9) {
    cat("Warning: More than 10% of values were zero or negative for", variable_name, "\n")
  }
  
  # Initialize results storage
  fit_results <- list()
  fit_errors <- character()
  
  # Try fitting each distribution
  for (dist in distributions_to_test) {
    tryCatch({
      # Different fitting approaches based on distribution
      if (dist == "norm") {
        fit <- fitdist(data_vector, dist)
      } else if (dist %in% c("lnorm", "gamma", "weibull", "exp")) {
        # These require positive values
        fit <- fitdist(data_vector_pos, dist)
      } else {
        next
      }
      
      # Store the fit result
      fit_results[[dist]] <- fit
    }, error = function(e) {
      fit_errors[dist] <- paste("Error fitting", dist, ":", e$message)
    })
  }
  
  # If no distributions could be fit, return an error result
  if (length(fit_results) == 0) {
    return(data.frame(
      variable = variable_name,
      stage = stage,
      best_distribution = "fitting_failed",
      aic = NA,
      bic = NA,
      ks_stat = NA,
      ks_pvalue = NA,
      distribution_params = paste(fit_errors, collapse = "; ")
    ))
  }
  
  # Compare fits based on AIC
  fit_aic <- sapply(fit_results, function(x) x$aic)
  best_dist_aic <- names(fit_results)[which.min(fit_aic)]
  
  # Get the best fit
  best_fit <- fit_results[[best_dist_aic]]
  
  # Get fit metrics
  # Using tryCatch here to handle potential issues with gofstat
  tryCatch({
    gof <- gofstat(best_fit)
    ks_stat <- gof$ks
    if (is.list(ks_stat)) {
      ks_statistic <- ks_stat$statistic
      ks_pvalue <- ks_stat$p.value
    } else {
      # If ks is not a list (atomic vector), handle differently
      ks_statistic <- NA
      ks_pvalue <- NA
    }
    bic_value <- gof$bic
  }, error = function(e) {
    # Set default values if gofstat fails
    ks_statistic <<- NA
    ks_pvalue <<- NA
    bic_value <<- NA
  })
  
  # Format the distribution parameters as a string
  params_string <- paste(names(best_fit$estimate), round(best_fit$estimate, 4), 
                         sep = "=", collapse = ", ")
  
  # Return summary of best fit
  return(data.frame(
    variable = variable_name,
    stage = stage,
    best_distribution = best_dist_aic,
    aic = best_fit$aic,
    bic = ifelse(is.null(bic_value), NA, bic_value),
    ks_stat = ks_statistic,
    ks_pvalue = ks_pvalue,
    distribution_params = params_string
  ))
}

# Function to create distribution plots comparing empirical data with fitted distributions
create_distribution_plots <- function(data_vector, variable_name, fit_result, stage = "before") {
  # If fitting failed, create just a histogram
  if (fit_result$best_distribution %in% c("insufficient_data", "fitting_failed")) {
    png(file.path("outlier_analysis", 
                  paste0("dist_", gsub("\\.", "_", variable_name), "_", stage, ".png")),
        width = 800, height = 600)
    
    hist(data_vector, 
         main = paste("Histogram of", variable_name, "(", stage, "outlier removal)"),
         xlab = "Value",
         col = "lightblue")
    
    dev.off()
    return()
  }
  
  # Create comparison plots
  png(file.path("outlier_analysis", 
                paste0("dist_", gsub("\\.", "_", variable_name), "_", stage, ".png")),
      width = 1200, height = 900)
  
  # Set up a 2x2 plot layout
  par(mfrow = c(2, 2))
  
  # Extract distribution name and parameters from the summary
  dist_name <- fit_result$best_distribution
  
  # Parse parameters from the string
  param_list <- strsplit(fit_result$distribution_params, ", ")[[1]]
  params <- list()
  for (param in param_list) {
    parts <- strsplit(param, "=")[[1]]
    params[[parts[1]]] <- as.numeric(parts[2])
  }
  
  # Deal with distributions that require positive values
  if (dist_name %in% c("lnorm", "gamma", "weibull", "exp")) {
    data_vector_pos <- data_vector[data_vector > 0]
  } else {
    data_vector_pos <- data_vector
  }
  
  # Histogram with density overlay
  hist(data_vector_pos, 
       freq = FALSE, 
       main = paste(variable_name, "with fitted", dist_name, "density (", stage, ")"),
       xlab = "Value",
       col = "lightblue")
  
  # Add the theoretical density line
  x_range <- seq(min(data_vector_pos), max(data_vector_pos), length.out = 1000)
  
  # Different density functions based on distribution
  if (dist_name == "norm") {
    y_density <- dnorm(x_range, mean = params$mean, sd = params$sd)
  } else if (dist_name == "lnorm") {
    y_density <- dlnorm(x_range, meanlog = params$meanlog, sdlog = params$sdlog)
  } else if (dist_name == "gamma") {
    y_density <- dgamma(x_range, shape = params$shape, rate = params$rate)
  } else if (dist_name == "weibull") {
    y_density <- dweibull(x_range, shape = params$shape, scale = params$scale)
  } else if (dist_name == "exp") {
    y_density <- dexp(x_range, rate = params$rate)
  }
  
  lines(x_range, y_density, col = "red", lwd = 2)
  
  # Q-Q plot
  if (dist_name == "norm") {
    qqnorm(data_vector_pos, main = paste("Q-Q Plot vs", dist_name, "(", stage, ")"))
    qqline(data_vector_pos, col = "red", lwd = 2)
  } else {
    # Create a custom QQ plot for non-normal distributions
    theoretical_quantiles <- switch(dist_name,
                                    lnorm = qlnorm(ppoints(length(data_vector_pos)), 
                                                   meanlog = params$meanlog, sdlog = params$sdlog),
                                    gamma = qgamma(ppoints(length(data_vector_pos)), 
                                                   shape = params$shape, rate = params$rate),
                                    weibull = qweibull(ppoints(length(data_vector_pos)), 
                                                       shape = params$shape, scale = params$scale),
                                    exp = qexp(ppoints(length(data_vector_pos)), rate = params$rate)
    )
    
    plot(sort(theoretical_quantiles), sort(data_vector_pos),
         main = paste("Q-Q Plot vs", dist_name, "(", stage, ")"),
         xlab = paste("Theoretical", dist_name, "Quantiles"),
         ylab = "Sample Quantiles")
    abline(0, 1, col = "red", lwd = 2)
  }
  
  # P-P plot
  if (dist_name == "norm") {
    theoretical_probs <- pnorm(sort(data_vector_pos), 
                               mean = params$mean, sd = params$sd)
  } else if (dist_name == "lnorm") {
    theoretical_probs <- plnorm(sort(data_vector_pos), 
                                meanlog = params$meanlog, sdlog = params$sdlog)
  } else if (dist_name == "gamma") {
    theoretical_probs <- pgamma(sort(data_vector_pos), 
                                shape = params$shape, rate = params$rate)
  } else if (dist_name == "weibull") {
    theoretical_probs <- pweibull(sort(data_vector_pos), 
                                  shape = params$shape, scale = params$scale)
  } else if (dist_name == "exp") {
    theoretical_probs <- pexp(sort(data_vector_pos), rate = params$rate)
  }
  
  plot(ppoints(length(data_vector_pos)), theoretical_probs,
       main = paste("P-P Plot for", dist_name, "(", stage, ")"),
       xlab = "Empirical Probabilities",
       ylab = paste("Theoretical", dist_name, "Probabilities"))
  abline(0, 1, col = "red", lwd = 2)
  
  # CDF comparison
  plot(ecdf(data_vector_pos), 
       main = paste("Empirical vs Theoretical CDF (", stage, ")"),
       xlab = "Value",
       ylab = "Cumulative Probability")
  
  # Add the theoretical CDF
  lines(x_range, switch(dist_name,
                        norm = pnorm(x_range, mean = params$mean, sd = params$sd),
                        lnorm = plnorm(x_range, meanlog = params$meanlog, sdlog = params$sdlog),
                        gamma = pgamma(x_range, shape = params$shape, rate = params$rate),
                        weibull = pweibull(x_range, shape = params$shape, scale = params$scale),
                        exp = pexp(x_range, rate = params$rate)),
        col = "red", lwd = 2)
  
  legend("bottomright", 
         legend = c("Empirical", paste("Fitted", dist_name)),
         col = c("black", "red"),
         lwd = c(1, 2))
  
  dev.off()
}

# Function to compare distributions before and after outlier removal
create_comparison_plot <- function(before_data, after_data, variable_name, before_fit, after_fit) {
  # Skip if either fit failed
  if (before_fit$best_distribution %in% c("insufficient_data", "fitting_failed") ||
      after_fit$best_distribution %in% c("insufficient_data", "fitting_failed")) {
    return()
  }
  
  # Create plot
  png(file.path("outlier_analysis", 
                paste0("dist_compare_", gsub("\\.", "_", variable_name), ".png")),
      width = 1200, height = 600)
  
  # Set up a 1x2 plot layout
  par(mfrow = c(1, 2))
  
  # Ensure positive data for certain distributions
  if (before_fit$best_distribution %in% c("lnorm", "gamma", "weibull", "exp")) {
    before_data_plot <- before_data[before_data > 0]
  } else {
    before_data_plot <- before_data
  }
  
  if (after_fit$best_distribution %in% c("lnorm", "gamma", "weibull", "exp")) {
    after_data_plot <- after_data[after_data > 0]
  } else {
    after_data_plot <- after_data
  }
  
  # Histogram before with density overlay
  hist(before_data_plot, 
       freq = FALSE, 
       main = paste(variable_name, "\nBefore: ", before_fit$best_distribution),
       xlab = "Value",
       col = "lightblue")
  
  # Parse parameters for before
  before_params <- list()
  before_param_list <- strsplit(before_fit$distribution_params, ", ")[[1]]
  for (param in before_param_list) {
    parts <- strsplit(param, "=")[[1]]
    before_params[[parts[1]]] <- as.numeric(parts[2])
  }
  
  # Add theoretical density for before
  x_range_before <- seq(min(before_data_plot), max(before_data_plot), length.out = 1000)
  before_density <- switch(before_fit$best_distribution,
                           norm = dnorm(x_range_before, mean = before_params$mean, sd = before_params$sd),
                           lnorm = dlnorm(x_range_before, meanlog = before_params$meanlog, sdlog = before_params$sdlog),
                           gamma = dgamma(x_range_before, shape = before_params$shape, rate = before_params$rate),
                           weibull = dweibull(x_range_before, shape = before_params$shape, scale = before_params$scale),
                           exp = dexp(x_range_before, rate = before_params$rate)
  )
  lines(x_range_before, before_density, col = "red", lwd = 2)
  
  # Histogram after with density overlay
  hist(after_data_plot, 
       freq = FALSE, 
       main = paste(variable_name, "\nAfter: ", after_fit$best_distribution),
       xlab = "Value",
       col = "lightgreen")
  
  # Parse parameters for after
  after_params <- list()
  after_param_list <- strsplit(after_fit$distribution_params, ", ")[[1]]
  for (param in after_param_list) {
    parts <- strsplit(param, "=")[[1]]
    after_params[[parts[1]]] <- as.numeric(parts[2])
  }
  
  # Add theoretical density for after
  x_range_after <- seq(min(after_data_plot), max(after_data_plot), length.out = 1000)
  after_density <- switch(after_fit$best_distribution,
                          norm = dnorm(x_range_after, mean = after_params$mean, sd = after_params$sd),
                          lnorm = dlnorm(x_range_after, meanlog = after_params$meanlog, sdlog = after_params$sdlog),
                          gamma = dgamma(x_range_after, shape = after_params$shape, rate = after_params$rate),
                          weibull = dweibull(x_range_after, shape = after_params$shape, scale = after_params$scale),
                          exp = dexp(x_range_after, rate = after_params$rate)
  )
  lines(x_range_after, after_density, col = "red", lwd = 2)
  
  dev.off()
}

# Create a list to store the outlier data for each variable
outlier_data_list <- list()

# Loop through each usage variable
for (var in usage_vars) {
  cat("\n=== Processing variable:", var, "===\n")
  
  # Filter data for current variable, remove NAs
  var_data <- data %>%
    filter(variable == var, !is.na(value))
  
  # Skip if insufficient data points
  if (nrow(var_data) <= 10) {
    cat("Skipping", var, "- insufficient data points\n")
    next
  }
  
  # Step 1: Initial distribution fitting (before outlier removal)
  cat("Fitting distributions to raw data...\n")
  before_fit <- fit_best_distribution(var_data$value, var, "before")
  distribution_fit_summary <- rbind(distribution_fit_summary, before_fit)
  
  # Create distribution plots for the raw data
  create_distribution_plots(var_data$value, var, before_fit, "before")
  
  # Step 2: Perform LOF outlier detection
  cat("Performing LOF outlier detection...\n")
  k_value <- min(20, max(10, floor(nrow(var_data) * 0.005)))
  value_matrix <- matrix(var_data$value, ncol = 1)
  lof_scores <- lof(value_matrix, minPts = k_value + 1)
  
  # Calculate statistics for outlier detection
  lof_mean <- mean(lof_scores)
  lof_sd <- sd(lof_scores)
  threshold <- lof_mean + 2 * lof_sd
  
  # Identify outliers
  outliers <- which(lof_scores > threshold)
  
  # Log outlier detection results
  cat("Detected", length(outliers), "outliers out of", nrow(var_data), "data points\n")
  
  # Save LOF score distribution plot
  png(file.path("outlier_analysis", paste0("lof_scores_", gsub("\\.", "_", var), ".png")),
      width = 800, height = 600)
  
  # Plot LOF scores with threshold
  hist(lof_scores, 
       main = paste("LOF Score Distribution for", var),
       xlab = "LOF Score",
       col = "lightblue")
  abline(v = threshold, col = "red", lwd = 2, lty = 2)
  text(x = threshold * 1.05, y = par("usr")[4] * 0.9, 
       labels = paste("Threshold:", round(threshold, 2)),
       pos = 4, col = "red", cex = 0.9)
  
  dev.off()
  
  # If no outliers were detected, record and skip to next variable
  if (length(outliers) == 0) {
    cat("No outliers detected for", var, "\n")
    
    # Record in summary
    lof_outlier_summary <- rbind(
      lof_outlier_summary,
      data.frame(
        variable = var,
        total_points = nrow(var_data),
        outlier_count = 0,
        outlier_percent = 0,
        mean_value = mean(var_data$value),
        max_value = max(var_data$value),
        outlier_mean = NA
      )
    )
    next
  }
  
  # Store outlier information
  outlier_data_list[[var]] <- data.frame(
    id = var_data$id[outliers],
    time = var_data$time[outliers],
    value = var_data$value[outliers],
    lof_score = lof_scores[outliers],
    stringsAsFactors = FALSE
  )
  
  # Create data without outliers
  var_data_clean <- var_data[-outliers, ]
  
  # Step 3: Re-fit distributions after outlier removal
  cat("Fitting distributions to cleaned data...\n")
  after_fit <- fit_best_distribution(var_data_clean$value, var, "after")
  distribution_fit_summary <- rbind(distribution_fit_summary, after_fit)
  
  # Create distribution plots for the cleaned data
  create_distribution_plots(var_data_clean$value, var, after_fit, "after")
  
  # Step 4: Compare distributions before and after
  create_comparison_plot(var_data$value, var_data_clean$value, var, before_fit, after_fit)
  
  # Calculate statistical summaries before and after
  mean_before <- mean(var_data$value)
  mean_after <- mean(var_data_clean$value)
  mean_outliers <- mean(var_data$value[outliers])
  
  # Record in summary
  lof_outlier_summary <- rbind(
    lof_outlier_summary,
    data.frame(
      variable = var,
      total_points = nrow(var_data),
      outlier_count = length(outliers),
      outlier_percent = round(100 * length(outliers) / nrow(var_data), 2),
      mean_value = mean_before,
      max_value = max(var_data$value),
      outlier_mean = mean_outliers
    )
  )
  
  # Step 5: Create a detailed before/after comparison plot of all distributions
  png(file.path("outlier_analysis", paste0("all_dist_compare_", gsub("\\.", "_", var), ".png")),
      width = 1200, height = 900)
  
  par(mfrow = c(2, 3))
  
  # Raw histograms
  hist(var_data$value,
       main = paste("Histogram Before (", var, ")"),
       xlab = "Value", 
       col = "lightblue")
  
  hist(var_data_clean$value,
       main = paste("Histogram After (", var, ")"),
       xlab = "Value", 
       col = "lightgreen")
  
  # Boxplots
  boxplot(var_data$value,
          main = paste("Boxplot Before (", var, ")"),
          ylab = "Value",
          col = "lightblue")
  
  boxplot(var_data_clean$value,
          main = paste("Boxplot After (", var, ")"),
          ylab = "Value",
          col = "lightgreen")
  
  # ECDF plots
  plot(ecdf(var_data$value),
       main = paste("ECDF Before (", var, ")"),
       xlab = "Value",
       ylab = "Cumulative Probability",
       col = "blue",
       lwd = 2)
  
  plot(ecdf(var_data_clean$value),
       main = paste("ECDF After (", var, ")"),
       xlab = "Value",
       ylab = "Cumulative Probability",
       col = "green",
       lwd = 2)
  
  dev.off()
}

# Save the distribution fit summary
write.csv(distribution_fit_summary, file.path("outlier_analysis", "distribution_fit_summary.csv"), 
          row.names = FALSE)

# Save the outlier summary
write.csv(lof_outlier_summary, file.path("outlier_analysis", "lof_outlier_summary.csv"), 
          row.names = FALSE)

# Create a summary of distribution changes using base R instead of dplyr
before_dist <- distribution_fit_summary[distribution_fit_summary$stage == "before", 
                                        c("variable", "best_distribution", "aic", "distribution_params")]
after_dist <- distribution_fit_summary[distribution_fit_summary$stage == "after", 
                                       c("variable", "best_distribution", "aic", "distribution_params")]

# Only proceed if we have data for both stages
if (nrow(before_dist) > 0 && nrow(after_dist) > 0) {
  # Merge the data
  distribution_changes <- merge(before_dist, after_dist, by = "variable")
  
  # Rename columns for clarity
  names(distribution_changes) <- c("variable", "before_distribution", "before_aic", "before_params",
                                   "after_distribution", "after_aic", "after_params")
  
  # Add calculated columns
  distribution_changes$distribution_changed <- distribution_changes$before_distribution != distribution_changes$after_distribution
  distribution_changes$aic_improvement <- distribution_changes$before_aic - distribution_changes$after_aic
} else {
  # Create empty dataframe with right structure if no data
  distribution_changes <- data.frame(
    variable = character(),
    before_distribution = character(),
    before_aic = numeric(),
    before_params = character(),
    after_distribution = character(),
    after_aic = numeric(),
    after_params = character(),
    distribution_changed = logical(),
    aic_improvement = numeric(),
    stringsAsFactors = FALSE
  )
}

# Save distribution changes summary
write.csv(distribution_changes, file.path("outlier_analysis", "distribution_changes.csv"), 
          row.names = FALSE)

# Create visualizations to summarize findings
# 1. Bar chart of best distributions before and after
distribution_counts_before <- table(distribution_fit_summary$best_distribution[
  distribution_fit_summary$stage == "before"])
distribution_counts_after <- table(distribution_fit_summary$best_distribution[
  distribution_fit_summary$stage == "after"])

png(file.path("outlier_analysis", "best_distribution_counts.png"), 
    width = 1000, height = 800)

par(mfrow = c(2, 1))

barplot(distribution_counts_before,
        main = "Best-Fitting Distributions Before Outlier Removal",
        col = "lightblue",
        xlab = "Distribution",
        ylab = "Number of Variables")

barplot(distribution_counts_after,
        main = "Best-Fitting Distributions After Outlier Removal",
        col = "lightgreen",
        xlab = "Distribution",
        ylab = "Number of Variables")

dev.off()

# 2. Plot showing AIC improvement after outlier removal
# Calculate AIC improvements directly
before_aic <- distribution_fit_summary[distribution_fit_summary$stage == "before", c("variable", "aic")]
after_aic <- distribution_fit_summary[distribution_fit_summary$stage == "after", c("variable", "aic")]

# Use merge instead of dplyr joins
aic_change_data <- merge(before_aic, after_aic, by = "variable")
names(aic_change_data) <- c("variable", "aic_before", "aic_after")

# Calculate improvement
aic_change_data$aic_improvement <- aic_change_data$aic_before - aic_change_data$aic_after
aic_change_data <- aic_change_data[order(-aic_change_data$aic_improvement), ]

# Identify significant improvements
significant_improvements <- aic_change_data[aic_change_data$aic_improvement > 0, ]

# Save to file if we have any
if (nrow(significant_improvements) > 0) {
  write.csv(significant_improvements, 
            file.path("outlier_analysis", "aic_improvements.csv"), row.names = FALSE)
}

# Create AIC improvement plot
if (nrow(aic_change_data) > 0) {
  png(file.path("outlier_analysis", "aic_improvement.png"), width = 1000, height = 800)
  
  par(mar = c(8, 4, 4, 2))  # Increase bottom margin for variable names
  
  barplot(aic_change_data$aic_improvement,
          names.arg = aic_change_data$variable,
          main = "AIC Improvement After Outlier Removal",
          xlab = "",
          ylab = "AIC Improvement",
          col = ifelse(aic_change_data$aic_improvement > 0, "green", "red"),
          las = 2)  # Rotate variable names
  
  abline(h = 0, lty = 2)
  
  dev.off()
}

# Print summary information
cat("\n==== Distribution-Based LOF Outlier Detection Summary ====\n")
cat("Total variables analyzed:", nrow(lof_outlier_summary), "\n")

cat("\nVariables with highest outlier percentages:\n")
temp <- lof_outlier_summary[order(-lof_outlier_summary$outlier_percent), ]
print(head(temp[, c("variable", "outlier_percent", "outlier_count", "total_points")], 5))

cat("\nDistribution changes after outlier removal:\n")
if (exists("distribution_changes")) {
  distribution_changes_filtered <- distribution_changes[distribution_changes$distribution_changed == TRUE, ]
  if (nrow(distribution_changes_filtered) > 0) {
    print(distribution_changes_filtered[, c("variable", "before_distribution", "after_distribution")])
  } else {
    cat("No distribution type changes detected after outlier removal\n")
  }
} else {
  cat("No distribution changes information available\n")
}

cat("\nVariables with most improved fit (by AIC) after outlier removal:\n")
# Calculate AIC improvements directly here
aic_changes <- merge(
  distribution_fit_summary[distribution_fit_summary$stage == "before", c("variable", "aic")],
  distribution_fit_summary[distribution_fit_summary$stage == "after", c("variable", "aic")],
  by = "variable",
  suffixes = c("_before", "_after")
)
if (nrow(aic_changes) > 0) {
  aic_changes$aic_improvement <- aic_changes$aic_before - aic_changes$aic_after
  aic_changes <- aic_changes[order(-aic_changes$aic_improvement), ]
  significant_improvements <- aic_changes[aic_changes$aic_improvement > 0, ]
  if (nrow(significant_improvements) > 0) {
    # Get corresponding distribution names
    significant_improvements <- merge(
      significant_improvements,
      distribution_fit_summary[distribution_fit_summary$stage == "before", c("variable", "best_distribution")],
      by = "variable"
    )
    names(significant_improvements)[ncol(significant_improvements)] <- "best_distribution_before"
    
    significant_improvements <- merge(
      significant_improvements,
      distribution_fit_summary[distribution_fit_summary$stage == "after", c("variable", "best_distribution")],
      by = "variable"
    )
    names(significant_improvements)[ncol(significant_improvements)] <- "best_distribution_after"
    
    print(head(significant_improvements[, c("variable", "aic_improvement", "best_distribution_before", "best_distribution_after")], 5))
    
    # Save to file for future reference
    write.csv(significant_improvements, 
              file.path("outlier_analysis", "aic_improvements.csv"), row.names = FALSE)
  } else {
    cat("No variables showed AIC improvement after outlier removal\n")
  }
} else {
  cat("No AIC comparison information available\n")
}

cat("\nDistribution fitting summary:\n")
dist_summary <- table(distribution_fit_summary$best_distribution, distribution_fit_summary$stage)
print(dist_summary)

# Collect and save the cleaned data
# =====================================

# Create a dataframe to store all cleaned data
cleaned_data <- data.frame()

# First, collect all variables that were subject to outlier detection
processed_vars <- unique(c(names(outlier_data_list), usage_vars))

# 1. Add data for variables that had outliers detected and removed
for (var in names(outlier_data_list)) {
  # Get all data for this variable
  var_data_all <- data %>% filter(variable == var)
  
  # Get outlier information
  outliers_info <- outlier_data_list[[var]]
  
  # Create a key for outliers to identify them for removal
  outlier_keys <- paste(outliers_info$id, outliers_info$time)
  
  # Create a key for all data points
  var_data_all$key <- paste(var_data_all$id, var_data_all$time)
  
  # Keep only non-outlier points
  var_data_clean <- var_data_all[!var_data_all$key %in% outlier_keys, ]
  var_data_clean$key <- NULL  # Remove the temporary key column
  
  # Add to the cleaned data dataframe
  cleaned_data <- rbind(cleaned_data, var_data_clean)
}

# 2. Add data for usage variables that were processed but had no outliers 
# or insufficient data points
for (var in usage_vars) {
  # Skip if already processed (outliers were detected and removed)
  if (var %in% names(outlier_data_list)) {
    next
  }
  
  # Add all data points for this variable
  var_data_all <- data %>% filter(variable == var)
  cleaned_data <- rbind(cleaned_data, var_data_all)
}

# 3. Add all other variables that weren't subject to outlier detection
# Get all variables in the dataset
all_vars <- unique(data$variable)

# Find variables that weren't processed for outliers
non_processed_vars <- setdiff(all_vars, processed_vars)

# Add these variables to the cleaned data
for (var in non_processed_vars) {
  var_data_all <- data %>% filter(variable == var)
  cleaned_data <- rbind(cleaned_data, var_data_all)
}

# Print summary of cleaned data
cat("\n=== Cleaned Data Summary ===\n")
cat("Total variables in original dataset:", length(all_vars), "\n")
cat("Variables processed for outliers:", length(processed_vars), "\n")
cat("Variables with outliers detected:", length(names(outlier_data_list)), "\n")
cat("Variables not processed for outliers:", length(non_processed_vars), "\n")
cat("Total variables in cleaned dataset:", length(unique(cleaned_data$variable)), "\n")

# Save the cleaned dataset
write.csv(cleaned_data, file.path("data", "cleaned_data.csv"), row.names = FALSE)
cat("\nCleaned dataset (with outliers removed) saved to 'data/cleaned_data.csv'\n")

