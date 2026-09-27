import pandas as pd

df = pd.read_csv("DL Project/Dataset_B_clean.csv")

print("Columns:")
print(df.columns)

print("\nAuthenticity counts:")
print(df["Authenticity"].value_counts())

print("\n==============================")
print("Examples where Authenticity = 0")
print("==============================")
print(
    df[df["Authenticity"] == 0][["News", "Authenticity"]].head(10)
)

print("\n==============================")
print("Examples where Authenticity = 1")
print("==============================")
print(
    df[df["Authenticity"] == 1][["News", "Authenticity"]].head(10)
)