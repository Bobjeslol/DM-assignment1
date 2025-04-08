library(dplyr)


data <- read.csv("dataset_mood_smartphone.csv")
head(data)

colnames(data)

variables <- unique(data$variable)

for (var in variables) {
  
  mood_values <- data %>%
    filter(variable == var) %>%
    pull(value)
  
  xlim_range <- range(mood_values, na.rm = TRUE)
  
  hist(mood_values,
       main = paste(var, "grouped values"),    # plot title
       xlab = paste(var, "rating"),          # x-axis label
       ylab = "Frequency",            # y-axis label
       cex.main = 1.7,                # title size
       cex.lab = 1.5,                  # axis label size
       col = "lightblue",
       xlim = xlim_range,  # dynamic limits
       xaxt = "n"
  )
  
  axis(side = 1, at = 1:10, cex.axis = 1.2)
  
  mood_summary <- data %>%
    filter(variable == var) %>%
    group_by(id) %>%
    summarise(
      mean_mood = mean(value, na.rm = TRUE),
      sd_mood = sd(value, na.rm = TRUE),
      count = n()
    )
  
  summary(mood_summary)
  
  hist(mood_summary$mean_mood,
       main = paste("Individual user", var, "means"),    # plot title
       xlab = paste("Mean", var, "rating"),          # x-axis label
       ylab = "Frequency",            # y-axis label
       cex.main = 1.7,                # title size
       cex.lab = 1.5,                  # axis label size
       col = "lightblue",
       xlim = xlim_range,  # dynamic limits
  )
}

