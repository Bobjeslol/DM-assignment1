library(dplyr)
library(dbscan)
library(ggplot2)

# Read the data
data <- read.csv("data/dataset_mood_smartphone.csv")
head(data)

# Get colnames and variable names to reference in analysis
colnames(data)
unique(data$variable)

# Create folder
if (!dir.exists("insights")) {
  dir.create("insights")
}

# 1. Plot all mood values combined
png("insights/plot_all_moods.png", width = 800, height = 600)
hist(mood_values,
     main = "Moods grouped values",
     xlab = "Mood rating",
     ylab = "Frequency",
     cex.main = 1.7,
     cex.lab = 1.5,
     col = "lightblue",
     xlim = c(1, 10),
     xaxt = "n")
axis(side = 1, at = 1:10, cex.axis = 1.2)
dev.off()

# 2. Mood summary per user
mood_summary <- data %>%
  filter(variable == "mood") %>%
  group_by(id) %>%
  summarise(
    mean_mood = mean(value, na.rm = TRUE),
    sd_mood = sd(value, na.rm = TRUE),
    count = n()
  )
summary(mood_summary)

# 3. Plot mean mood per user
png("insights/plot_mean_moods_per_user.png", width = 800, height = 600)
hist(mood_summary$mean_mood,
     main = "Individual user mood means",
     xlab = "Mean mood rating",
     ylab = "Frequency",
     cex.main = 1.7,
     cex.lab = 1.5,
     col = "lightblue",
     xlim = c(5.5, 8))
dev.off()

# 4. Count missing values per user
missing_values_per_user <- data %>%
  filter(is.na(value)) %>%
  group_by(id) %>%
  summarise(missing_count = n())

# 5. Plot missing values
png("insights/plot_missing_values_per_user.png", width = 800, height = 600)
hist(missing_values_per_user$missing_count,
     main = "Distribution of Missing Values per User",
     xlab = "Number of Missing Values",
     ylab = "Number of Users",
     col = "lightblue",
     cex.main = 1.7,
     cex.lab = 1.5,
     cex.axis = 1.2)
dev.off()
