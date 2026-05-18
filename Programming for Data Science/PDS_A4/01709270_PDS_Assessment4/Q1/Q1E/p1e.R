# q1e.R
# Demonstration: NA is a first-class missing value in R with consistent behavior.

cat("Examples 1: Propagation and comparisons")
cat("____________________________________\n")
x <- c(1, NA, 3)

cat("x:\n"); print(x)
cat("x + 1:\n"); print(x + 1)

cat("x > 2:\n"); print(x > 2)       # NA yields NA (unknown)
cat("is.na(x):\n"); print(is.na(x))

cat("Examples 2: Aggregation with/without na.rm")
cat("________________________________________\n")
cat("sum(x):\n"); print(sum(x))                 # NA
cat("sum(x, na.rm=TRUE):\n"); print(sum(x, na.rm = TRUE))

cat("mean(x):\n"); print(mean(x))               # NA
cat("mean(x, na.rm=TRUE):\n"); print(mean(x, na.rm = TRUE))

cat("Examples 3: Data frame filtering with NA")
cat("________________________________________\n")
df <- data.frame(
  id = 1:6,
  score = c(10, NA, 7, NA, 4, 9),
  group = c("A", "A", "B", "B", "B", "A")
)

cat("df:\n"); print(df)

cat("\nRows with missing score:\n")
print(df[is.na(df$score), ])

cat("\nRows without missing score:\n")
print(df[!is.na(df$score), ])

cat("\nGroup means (na.rm=TRUE):\n")
print(tapply(df$score, df$group, mean, na.rm = TRUE))

cat("\nGroup means (default, na.rm=FALSE):\n")
print(tapply(df$score, df$group, mean))  # groups containing NA -> NA