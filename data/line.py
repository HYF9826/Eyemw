import pandas as pd

df = pd.read_excel("testdata_study3.xlsx")

df["UniqueSpeed1"] = df["UniqueSpeed1"].interpolate(method="linear")

df.to_excel("output.xlsx", index=False)