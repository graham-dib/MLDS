# q1e.py
# Demonstration: Python commonly uses NaN/None; behavior differs by container/library.

import math

import numpy as np
import pandas as pd

print("Examples 1: Propagation and comparisons (NumPy)")
print("__________________________________________________\n")
x = np.array([1.0, np.nan, 3.0])

print("x:")
print(x)

print("x + 1:")
print(x + 1)

print("x > 2:")
print(x > 2)  # np.nan comparisons yield False, not an explicit "unknown"

print("np.isnan(x):")
print(np.isnan(x))

print("Examples 2: Aggregation with/without skipping missing")
print("__________________________________________________\n")
print("np.sum(x):")
print(np.sum(x))  # nan
print("np.nansum(x):")
print(np.nansum(x))

print("np.mean(x):")
print(np.mean(x))  # nan
print("np.nanmean(x):")
print(np.nanmean(x))

print("Examples 3: DataFrame filtering with missing (pandas)")
print("__________________________________________________\n")
df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "score": [10.0, np.nan, 7.0, np.nan, 4.0, 9.0],
        "group": ["A", "A", "B", "B", "B", "A"],
    }
)

print("df:")
print(df)

print("\nRows with missing score:")
print(df[df["score"].isna()])

print("\nRows without missing score:")
print(df[~df["score"].isna()])

print("\nGroup means (default skipna=True in pandas mean):")
print(df.groupby("group")["score"].mean())

print("\nGroup means if you force missing to propagate (skipna=False):")
print(df.groupby("group")["score"].mean(skipna=False))

print("\n=== Part 4: None vs NaN (plain Python / math) ===")
y = [1.0, None, 3.0]
print("y:", y)
try:
    print("y + 1 (fails):", [v + 1 for v in y])
except TypeError as e:
    print("TypeError:", e)

z = [1.0, float("nan"), 3.0]
print("z:", z)
print("sum(z) (nan propagates):", sum(z))
print("math.isnan(z[1]):", math.isnan(z[1]))