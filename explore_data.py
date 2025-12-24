import pandas as pd

df1 = pd.read_csv('PHICAD-part1.csv', sep='\t', encoding='utf-8', skiprows=[1])
df2 = pd.read_csv('PHICAD-part2.csv', sep='\t', encoding='utf-8', skiprows=[1])

df= pd.concat([df1, df2], ignore_index=True)

print(df.head())
print(df.columns)
print(len(df))

print("\nتوزیع لیبل‌ها در دیتاست:")
print(df['class'].value_counts())