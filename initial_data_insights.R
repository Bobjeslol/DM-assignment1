library(dplyr)


data <- read.csv("dataset_mood_smartphone.csv")
head(data)

colnames(data)


mood_values <- data %>%
  filter(variable == "mood") %>%
  pull(value)

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

mood_summary <- data %>%
  filter(variable == "mood") %>%
  group_by(id) %>%
  summarise(
    mean_mood = mean(value, na.rm = TRUE),
    sd_mood = sd(value, na.rm = TRUE),
    count = n()
  )

summary(mood_summary)

hist(mood_summary$mean_mood,
     main = "Individual user mood means",    # plot title
     xlab = "Mean mood rating",          # x-axis label
     ylab = "Frequency",            # y-axis label
     cex.main = 1.7,                # title size
     cex.lab = 1.5,                  # axis label size
     col = "lightblue",
     xlim = c(5.5, 8)
)


missing_values_per_user <- data %>%
  filter(is.na(value)) %>%           # Filter rows with NA in 'value' column
  group_by(id) %>%                   # Group by user id
  summarise(missing_count = n())      # Count missing values per user


hist(missing_values_per_user$missing_count,
     main = "Distribution of Missing Values per User",
     xlab = "Number of Missing Values",
     ylab = "Number of Users",
     col = "lightblue",
     cex.main = 1.7,                   # title size
     cex.lab = 1.5,                    # axis label size
     cex.axis = 1.2                    # axis ticks size
)
