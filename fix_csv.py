import pandas as pd

df = pd.read_csv("DatasetA.csv")

df.to_csv("dataset_A.csv", index=False, encoding="utf-8-sig")

print("Done!")