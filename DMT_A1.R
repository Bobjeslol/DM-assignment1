data=read.csv(file="dataset_mood_smartphone.csv",header=TRUE)

num_users <- length(unique(data$id))

print(num_users)
print(unique(data$id))

# Average number of missing values per user
missing_per_id <- tapply(is.na(data$value), data$id, sum)
average_missing <- mean(missing_per_id)
print(average_missing)

# Average number of entries per user
entries_per_user <- table(data$id)
average_entries <- mean(entries_per_user)
print(average_entries)

# Average number of entries per user per day
data$date <- as.Date(data$time)
entries_per_user_day <- table(data$id, data$date)
average_entries_per_user_day <- mean(entries_per_user_day)
print(average_entries_per_user_day)

# Variable boxplots
data$value <- as.numeric(data$value)
data_clean <- data[!is.na(data$value), ]
variables <- unique(data_clean$variable)
for (v in variables) {
  var_data <- subset(data_clean, variable == v)
  
  boxplot(var_data$value,
          main = paste("Boxplot of", v),
          ylab = "Value",
          col = "lightblue",
          border = "darkblue")
}


# Intervals for variables
data$value <- as.numeric(data$value)
data_clean <- data[!is.na(data$value), ]
coverage <- 0.99
lower_p <- (1 - coverage) / 2      
upper_p <- 1 - lower_p              
variables <- unique(data_clean$variable)
for (v in variables) {
  var_data <- subset(data_clean, variable == v)$value
  
  bounds <- quantile(var_data, probs = c(lower_p, upper_p), na.rm = TRUE)
  
  cat(sprintf("Variable: %s\n%.1f%% Interval: [%.2f, %.2f]\n\n", 
              v, coverage * 100, bounds[1], bounds[2]))
}