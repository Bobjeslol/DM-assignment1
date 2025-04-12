library(dplyr)

# Read the data
data <- read.csv("dataset_mood_smartphone.csv")
head(data)

# Get colnames and variable names to reference in analysis
colnames(data)
unique(data$variable)


# Pull all mood values, grouped
mood_values <- data %>%
  filter(variable == "mood") %>%
  pull(value)

# Plots all mood values combined
hist(mood_values,
     main = "Moods grouped values",    # plot title
     xlab = "Mood rating",          # x-axis label
     ylab = "Frequency",            # y-axis label
     cex.main = 1.7,                # title size
     cex.lab = 1.5,                  # axis label size
     col = "lightblue",
     xlim = c(1, 10),
     xaxt = "n"
)

axis(side = 1, at = 1:10, cex.axis = 1.2)


# Pull all mood values by user, then take the mean of those values
mood_summary <- data %>%
  filter(variable == "mood") %>%
  group_by(id) %>%
  summarise(
    mean_mood = mean(value, na.rm = TRUE),
    sd_mood = sd(value, na.rm = TRUE),
    count = n()
  )

summary(mood_summary)

# Plot all the means per user
hist(mood_summary$mean_mood,
     main = "Individual user mood means",    # plot title
     xlab = "Mean mood rating",          # x-axis label
     ylab = "Frequency",            # y-axis label
     cex.main = 1.7,                # title size
     cex.lab = 1.5,                  # axis label size
     col = "lightblue",
     xlim = c(5.5, 8)
)


# Count the amount of missing values per user
missing_values_per_user <- data %>%
  filter(is.na(value)) %>%
  group_by(id) %>%
  summarise(missing_count = n())


# Plot the amount of missing values per user
hist(missing_values_per_user$missing_count,
     main = "Distribution of Missing Values per User",
     xlab = "Number of Missing Values",
     ylab = "Number of Users",
     col = "lightblue",
     cex.main = 1.7,                   # title size
     cex.lab = 1.5,                    # axis label size
     cex.axis = 1.2                    # axis ticks size
)

# Get all relevant variables: appCat.* + screen
appcat_vars <- unique(data$variable)
usage_vars <- c("screen", appcat_vars[grepl("^appCat\\.", appcat_vars)])

# Loop over each variable and plot histograms
for (var in usage_vars) {
  cat("Plotting for:", var, "\n")
  
  # Filter data for current variable
  var_data <- data %>%
    filter(variable == var, !is.na(value))
  
  # Skip if no data
  if (nrow(var_data) == 0) next
  
  # Raw histogram
  hist(var_data$value,
       main = paste("Raw Distribution of", var, "(All Users)"),
       xlab = "Usage Time",
       ylab = "Frequency",
       col = "salmon",
       cex.main = 1.7,
       cex.lab = 1.5,
       cex.axis = 1.2
  )
  
  # Filter out negative values for log plot
  nonneg_values <- var_data$value[var_data$value >= 0]
  
  # Log-transformed histogram
  hist(log1p(nonneg_values),
       main = paste("Log-Transformed Distribution of", var, "(All Users)"),
       xlab = "log(Usage Time + 1)",
       ylab = "Frequency",
       col = "darkseagreen2",
       cex.main = 1.7,
       cex.lab = 1.5,
       cex.axis = 1.2
  )
  
