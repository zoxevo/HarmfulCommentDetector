from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__, template_folder='../views')

# لود مدل
model_path = '../models/toxic_model.pkl'
if not os.path.exists(model_path):
    print("مدل پیدا نشد! اول train.py رو اجرا کن")
pipeline = joblib.load(model_path)

# مپ برای نمایش فارسی و رنگ
persian_label = {
    'clean': 'سالم',
    'hate': 'سخنرانی نفرت‌انگیز',
    'spam': 'هرزنامه',
    'obscene': 'مبتذل',
    'hateobscene': 'نفرت‌انگیز + مبتذل',
    'spamobscene': 'هرزنامه + مبتذل'
}

color_map = {
    'clean': 'success',  # سبز
    'hate': 'danger',    # قرمز
    'spam': 'warning',
    'obscene': 'danger',
    'hateobscene': 'danger',
    'spamobscene': 'warning'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    comment = None
    if request.method == 'POST':
        comment = request.form['comment']
        if comment.strip():
            pred = pipeline.predict([comment])[0]
            prediction = {
                'label': persian_label.get(pred, pred),
                'color': color_map.get(pred, 'secondary')
            }
    return render_template('index.html', prediction=prediction, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)