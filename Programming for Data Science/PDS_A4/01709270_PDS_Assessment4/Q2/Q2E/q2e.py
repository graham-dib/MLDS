# q2e.py
# Demonstration: list and dictionary comprehensions in Python

files = [
    "report.csv",
    "data.csv",
    "analysis.txt",
    "notes.txt",
    "script.py",
    "archive.tar"
]

# 1) List comprehension: extract (filename, extension) pairs
file_ext_pairs = [
    (name, name.split(".")[-1])
    for name in files
]

# 2) Dictionary comprehension: count files by extension
ext_counts = {
    ext: sum(1 for _, e in file_ext_pairs if e == ext)
    for ext in {e for _, e in file_ext_pairs}
}

print("File–extension pairs:")
print(file_ext_pairs)

print("\nExtension counts:")
print(ext_counts)