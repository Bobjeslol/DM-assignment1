data=read.csv(file="imputed_data.csv",header=TRUE)
new_mood_data <- data[data$variable == "mood", ]
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


# Ensure time is in proper format and hour is extracted
data$time <- as.POSIXct(data$time, format = "%Y-%m-%d %H:%M:%OS")
data$hour <- as.numeric(format(data$time, "%H"))

# Filter missing values
missing_data <- data[is.na(data$value), ]

# Reorder hours starting from 8
hour_levels <- c(8:23, 0)
missing_data$hour_factor <- factor(missing_data$hour, levels = hour_levels)

# Count missing values per hour
counts <- table(missing_data$hour_factor)

# Create barplot and store midpoints of bars
bar_midpoints <- barplot(counts,
                         main = "Missing Values by Hour (Starting at 8 AM)",
                         xlab = "Hour of Day",
                         ylab = "Number of Missing Values",
                         col = "tomato",
                         border = "black",
                         names.arg = FALSE, 
                         las = 1,
                         ylim = c(0, 35),
                         space = 0)   

# Add custom x-axis labels aligned with bars
axis(side = 1, at = bar_midpoints, labels = hour_levels)

# Add surrounding box
box()

######### Feature Engineering ########

library(dplyr)

# Convert time to POSIXct and extract the date
data$time <- as.POSIXct(data$time)
data$date <- as.Date(data$time)

# Define sum and mean variables
sum_vars <- c("call", "sms")
mean_vars <- setdiff(unique(data$variable), sum_vars)

# Create two separate datasets
sum_data <- data[data$variable %in% sum_vars, ]
mean_data <- data[data$variable %in% mean_vars, ]

# Aggregate sum variables
sum_agg <- aggregate(value ~ id + date + variable, data = sum_data, FUN = sum, na.rm = TRUE)

# Aggregate mean variables
mean_agg <- aggregate(value ~ id + date + variable, data = mean_data, FUN = mean, na.rm = TRUE)

# Combine them
aggregated <- rbind(sum_agg, mean_agg)

# Reshape to wide format
wide_data <- reshape(aggregated,
                     timevar = "variable",
                     idvar = c("id", "date"),
                     direction = "wide")

# Clean column names
names(wide_data) <- gsub("^value\\.", "", names(wide_data))

# Sort the data
wide_data <- wide_data[order(wide_data$id, wide_data$date), ]



instances <- list()
user_ids <- unique(wide_data$id)

# Helper to check if dates are consecutive
is_consecutive <- function(dates) {
  all(diff(as.integer(dates)) == 1)
}

for (uid in user_ids) {
  user_data <- wide_data[wide_data$id == uid, ]
  user_data <- user_data[order(user_data$date), ]
  
  if (nrow(user_data) < 6) next
  
  dates <- user_data$date
  period_index <- 1
  
  for (i in 1:(nrow(user_data) - 5)) {
    date_window <- dates[i:(i + 5)]
    
    if (!is_consecutive(date_window)) next
    
    window <- user_data[i:(i + 4), ]
    target_day <- user_data[i + 5, ]
    
    # Ensure all mood values are present in the 5-day window AND target day
    if (any(is.na(window$mood)) || is.na(target_day$mood)) next
    
    row <- data.frame(
      id = uid,
      period = paste0(period_index, "-", period_index + 5),
      date_period = paste0(as.character(date_window[1]), ", ", as.character(date_window[6])),
      circumplex.arousal = round(mean(window$circumplex.arousal, na.rm = TRUE)),
      circumplex.valence = round(mean(window$circumplex.valence, na.rm = TRUE)),
      activity = mean(window$activity, na.rm = TRUE),
      screen = mean(window$screen, na.rm = TRUE),
      call = sum(window$call, na.rm = TRUE),
      sms = sum(window$sms, na.rm = TRUE),
      appCat = mean(as.matrix(window[ , grep("^appCat\\.", names(window))]), na.rm = TRUE),
      period_mood = round(mean(window$mood, na.rm = TRUE)),
      mood = round(target_day$mood)
    )
    
    instances[[length(instances) + 1]] <- row
    period_index <- period_index + 1
  }
}

# Combine all rows
instance_dataset <- do.call(rbind, instances)
# Missing values replaced with 0
instance_dataset[is.na(instance_dataset)] <- 0
write.csv(instance_dataset, "instance_dataset.csv", row.names = FALSE)

##### Mood value Distribution #####

og_data=read.csv(file="dataset_mood_smartphone.csv", header=TRUE)
mood_data <- og_data[og_data$variable == "mood", ]
breaks <- seq(min(mood_data$value) - 0.5, max(mood_data$value) + 0.5, by = 1)
# Create histogram
hist(mood_data$value,
     breaks = breaks,
     col = "steelblue",
     border = "black",
     main = "Distribution of Mood Values",
     xlab = "Mood",
     ylab = "Count",
     xaxt = "n")

# Add custom x-axis with integer labels
axis(1, at = seq(floor(min(mood_data$value)), ceiling(max(mood_data$value)), by = 1))
box()
