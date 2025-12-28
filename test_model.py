import joblib

# لود مدل
pipeline = joblib.load('models/toxic_model.pkl')

# کامنت‌های تست (سالم و harmful)
tests = [
    "فیلم عالی بود دمت گرم",  # باید clean
    "عالی بود ممنون از زحماتتون",  # clean
    "کیر تو دهنت خفه شو",  # باید obscene یا hateobscene
    "کانال تلگرام پورن بیا خصوصی",  # spamobscene
    "لعنتی برو گم شو",  # hate
    "محصولات جنسی اصل 💦 دایرکت"  # spamobscene
]

print("تست پیش‌بینی مدل:\n")
for comment in tests:
    pred = pipeline.predict([comment])[0]
    print(f"کامنت: {comment}")
    print(f"تشخیص: {pred}\n")