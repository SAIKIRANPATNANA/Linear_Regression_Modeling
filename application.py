from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler 

application = Flask(__name__)
app = application
ridge_model = pickle.load(open('/config/workspace/models/ridge_regressor.pkl','rb'))
standard_scaler = pickle.load(open('/config/workspace/models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
       Temperature =float(request.form.get('Temperature'))
       RH = float(request.form.get('RH'))
       Ws = float(request.form.get('Ws'))
       Rain = float(request.form.get('Rain'))
       FFMC = float(request.form.get('FFMC'))
       DMC = float(request.form.get('DMC'))
       ISI = float(request.form.get('ISI'))
       Classes = float(request.form.get('Classes'))
       scaled_data = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes]])
       result = ridge_model.predict(scaled_data)
       return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")

