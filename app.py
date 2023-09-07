import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

scaler_model = pickle.load(open('models/scaler.pkl','rb'))
ridge_model  = pickle.load(open('models/ridge.pkl','rb'))


@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('temp'))
        RH = float(request.form.get('rh'))
        Ws = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('classes'))
        Region = float(request.form.get('reg'))

        new_data_scaled= scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('pred.html',result=result[0])

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")