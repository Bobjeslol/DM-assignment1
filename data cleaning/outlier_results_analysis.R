# Generate a comprehensive distribution comparison table
# This adds to our existing LOF outlier analysis

# If running as a standalone script, load necessary libraries
# Otherwise these will already be loaded from the main analysis
library(dplyr)
library(ggplot2)
library(gridExtra)
library(knitr)
library(kableExtra)  # For better table formatting

# Ensure output directory exists
dir.create("outlier_analysis", showWarnings = FALSE)

# This assumes distribution_fit_summary has been created in the main analysis
# If not, you'll need to run the distribution fitting code first

# Create a comprehensive comparison table
comparison_table <- function(distribution_fit_summary) {
  # Extract before and after data
  before_data <- distribution_fit_summary[distribution_fit_summary$stage == "before", 
                                          c("variable", "best_distribution", "aic", "ks_stat", "distribution_params")]
  
  after_data <- distribution_fit_summary[distribution_fit_summary$stage == "after", 
                                         c("variable", "best_distribution", "aic", "ks_stat", "distribution_params")]
  
  # Merge the data
  comparison <- merge(before_data, after_data, by = "variable", all = TRUE)
  colnames(comparison) <- c("variable", "before_distribution", "before_aic", "before_ks", 
                            "before_params", "after_distribution", "after_aic", "after_ks", "after_params")
  
  # Calculate improvement metrics
  comparison$aic_change <- comparison$before_aic - comparison$after_aic
  comparison$distribution_changed <- comparison$before_distribution != comparison$after_distribution
  
  # Replace NA with "No outliers removed" where appropriate
  comparison$after_distribution[is.na(comparison$after_distribution)] <- "No outliers removed"
  comparison$after_aic[is.na(comparison$after_aic)] <- NA
  comparison$aic_change[is.na(comparison$aic_change)] <- NA
  
  # Fix NA values in distribution_changed column
  comparison$distribution_changed[is.na(comparison$distribution_changed)] <- FALSE
  
  # Sort by AIC improvement
  comparison <- comparison[order(-comparison$aic_change, comparison$variable), ]
  
  # Create a simplified version for display
  display_table <- comparison[, c("variable", "before_distribution", "before_aic", 
                                  "after_distribution", "after_aic", "aic_change", "distribution_changed")]
  
  # Round numeric columns
  display_table$before_aic <- round(display_table$before_aic, 2)
  display_table$after_aic <- round(display_table$after_aic, 2)
  display_table$aic_change <- round(display_table$aic_change, 2)
  
  # Save the full comparison table to CSV
  write.csv(comparison, file.path("outlier_analysis", "distribution_comparison_table.csv"), row.names = FALSE)
  
  # Also save the display version
  write.csv(display_table, file.path("outlier_analysis", "distribution_comparison_display.csv"), row.names = FALSE)
  
  # Return the display table for immediate viewing
  return(display_table)
}

# Generate an HTML table for better visualization
create_html_table <- function(display_table) {
  # Format the table with kableExtra
  html_table <- kable(display_table, format = "html", 
                      caption = "Distribution Comparison Before and After Outlier Removal") %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed"), 
                  full_width = FALSE) %>%
    column_spec(1, bold = TRUE) %>%
    column_spec(6, color = ifelse(display_table$aic_change > 0, "green", 
                                  ifelse(display_table$aic_change < 0, "red", "black"))) %>%
    column_spec(7, background = ifelse(display_table$distribution_changed == TRUE, "lightyellow", "white"))
  
  # Save the HTML table
  write(html_table, file = file.path("outlier_analysis", "distribution_comparison_table.html"))
}

# Create a visual representation of the distribution changes
create_distribution_change_plot <- function(display_table) {
  # Count distribution types before and after
  dist_before <- table(display_table$before_distribution)
  dist_after <- table(display_table$after_distribution[display_table$after_distribution != "No outliers removed"])
  
  # Combine into a data frame for plotting
  dist_counts <- data.frame(
    distribution = names(c(dist_before, dist_after)),
    count = c(dist_before, dist_after),
    stage = rep(c("Before", "After"), c(length(dist_before), length(dist_after)))
  )
  
  # Create bar plot
  png(file.path("outlier_analysis", "distribution_type_changes.png"), width = 1000, height = 600)
  
  barplot(table(display_table$before_distribution, display_table$after_distribution), 
          main = "Distribution Type Changes After Outlier Removal",
          xlab = "After Removal", 
          ylab = "Before Removal",
          col = rainbow(length(unique(display_table$before_distribution))))
  
  dev.off()
  
  # Create a second visualization showing distribution frequencies
  png(file.path("outlier_analysis", "distribution_frequencies.png"), width = 1000, height = 800)
  
  par(mfrow = c(2, 1))
  
  # Before removal
  barplot(dist_before, 
          main = "Best-Fitting Distributions Before Outlier Removal",
          xlab = "Distribution Type",
          ylab = "Number of Variables",
          col = "lightblue")
  
  # After removal
  barplot(dist_after, 
          main = "Best-Fitting Distributions After Outlier Removal",
          xlab = "Distribution Type",
          ylab = "Number of Variables",
          col = "lightgreen")
  
  dev.off()
}

# Example execution - run this part assuming distribution_fit_summary exists
if (exists("distribution_fit_summary")) {
  # Generate the comparison table
  display_table <- comparison_table(distribution_fit_summary)
  
  # Print the table to console
  print(display_table)
  
  # Try to create HTML version if kableExtra is available
  tryCatch({
    create_html_table(display_table)
    cat("HTML table created successfully.\n")
  }, error = function(e) {
    cat("HTML table generation skipped:", e$message, "\n")
  })
  
  # Create visualization of distribution changes
  tryCatch({
    create_distribution_change_plot(display_table)
    cat("Distribution change plots created successfully.\n")
  }, error = function(e) {
    cat("Distribution change plot generation skipped:", e$message, "\n")
  })
  
  cat("\nDistribution comparison table generated and saved to 'outlier_analysis' directory.\n")
  cat("Files created:\n")
  cat("- distribution_comparison_table.csv (full details)\n")
  cat("- distribution_comparison_display.csv (simplified version)\n")
  cat("- distribution_comparison_table.html (formatted HTML, if kableExtra available)\n")
  cat("- distribution_type_changes.png (visualization of distribution changes)\n")
  cat("- distribution_frequencies.png (distribution frequencies before and after)\n")
} else {
  cat("Error: distribution_fit_summary not found.\n")
  cat("Please run the distribution fitting analysis first.\n")
}