import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("در حال لود دیتا تمیز...")


df = pd.read_csv('phicad_clean.csv', encoding='utf-8')

X = df['comment_normalized'].fillna('')
y = df['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"تعداد train: {len(X_train)}, test: {len(X_test)}")


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=50000, min_df=2, lowercase=False)),
    ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight='balanced', n_jobs=-1))
])

print("شروع train... (۵-۱۵ دقیقه طول می‌کشه)")
pipeline.fit(X_train, y_train)

# تست مدل
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقت مدل: {accuracy*100:.2f}%")
print("\nگزارش کامل:")
print(classification_report(y_test, y_pred))

# ذخیره مدل
joblib.dump(pipeline, 'models/toxic_model.pkl')
print("\nمدل ذخیره شد در models/toxic_model.pkl")