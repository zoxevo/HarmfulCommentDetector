import pandas as pd

print("در حال لود و تمیز کردن دیتاست PHICAD...")

df1 = pd.read_csv('PHICAD-part1.csv', sep='\t', encoding='utf-8', header=0, skiprows=[1])
df2 = pd.read_csv('PHICAD-part2.csv', sep='\t', encoding='utf-8', header=0, skiprows=[1])

df = pd.concat([df1, df2], ignore_index=True)

df = df[['comment_normalized', 'hate', 'spam', 'obscene', 'class']]

print(f"تعداد ردیف قبل: {len(df)}")
df = df.dropna(subset=['class'])
df = df[df['class'].str.strip() != '']
print(f"تعداد ردیف بعد: {len(df)}")

print("\nتوزیع کلاس‌ها:")
print(df['class'].value_counts())

print("\nچند نمونه از کامنت‌ها:")
for cls in df['class'].unique():
    print(f"\n--- {cls} ---")
    samples = df[df['class'] == cls]['comment_normalized'].sample(min(3, len(df[df['class'] == cls]))).values
    for s in samples:
        print(s)

df.to_csv('phicad_clean.csv', index=False, encoding='utf-8')
print("\nدیتاست تمیز ذخیره شد!")