#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask, render_template,request
import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', "rb"))
print("after model2")

@app.route('/')
def hello():
    print("inside function home")
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    age = request.form["age"]
    sex = request.form["sex"]
    bp = request.form["bp"]
    ch = request.form["ch"]
    na = request.form["na"]

    sample_df=pd.DataFrame({'Age':age,'Sex':sex,'BP':bp,'Cholesterol':ch,'Na_to_K':na},index=[0])

    drug_pred=model.predict(sample_df)
    print("After Predict")
    print(drug_pred)
    if drug_pred[0]==0:
        drug_pred="Drug X"
    elif drug_pred[0]==1:
        drug_pred="Drug Y"
    elif drug_pred[0]==2:
        drug_pred="Drug C"
    elif drug_pred[0]==3:
        drug_pred="Drug A"
    elif drug_pred[0]==4:
        drug_pred="Drug B"
    return render_template("index.html", drug='Proposed drug is {}'.format(drug_pred))
if __name__ == '__main__':
    app.run(debug=True)