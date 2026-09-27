import pandas as pd

df = pd.read_csv("Dataset_B_clean.csv")

df.columns = df.columns.str.strip()

print(df["Authenticity"].value_counts())
print(df["Authenticity"].value_counts(normalize=True) * 100)