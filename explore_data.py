import pandas as pd

df1 = pd.read_csv('PHICAD-part1.csv')
df2 = pd.read_csv('PHICAD-part2.csv')

print(df1.head())
print(df1.columns)
print(len(df1))