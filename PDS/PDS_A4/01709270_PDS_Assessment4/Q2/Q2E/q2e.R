# q2e.R
# Equivalent functionality in base R (no external packages)

files <- c(
  "report.csv",
  "data.csv",
  "analysis.txt",
  "notes.txt",
  "script.py",
  "archive.tar"
)

# 1) Extract (filename, extension) pairs
split_names <- lapply(files, function(x) strsplit(x, "\\.")[[1]])
extensions <- sapply(split_names, function(x) tail(x, 1))

file_ext_pairs <- cbind(files, extensions)

# 2) Count files by extension
ext_counts <- table(extensions)

print("File–extension pairs:")
print(file_ext_pairs)

print("Extension counts:")
print(ext_counts)