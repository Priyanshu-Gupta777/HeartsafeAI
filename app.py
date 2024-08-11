from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('model1.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    cp = request.form.get('cp')
    trestbps= request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    result = model.predict(np.array([age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,13))

    if result[0] == 1:
        result = "Warning: Our analysis indicates a significant risk of heart disease. We strongly recommend consulting with a healthcare professional for a comprehensive evaluation and appropriate intervention."
    else:
        result = "Good news! Our analysis shows no significant risk of heart disease at this time. Continue maintaining a healthy lifestyle, and always consult with your healthcare provider for regular check-ups."
    return render_template("index.html", result = result)

if __name__ == '__main__':
    app.run(debug=True)