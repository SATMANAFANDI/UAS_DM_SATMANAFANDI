import numpy as np
from flask import Flask, request, render_template
import pickle
from model import PrepoceesingData

flask_app = Flask(__name__)


@flask_app.route("/",methods=["POST", "GET"],endpoint='')
def home():
    dataM = PrepoceesingData()
    dataM.proses("dataset/PENCEMARAN SUUUU.csv")
    dataM.DataSelection()
    result = ''
    dataInputan = []
    if request.method=='POST':
        pm10 = float(request.form["pm10"])
        pm25 = float(request.form["pm25"])
        so2 = float(request.form["so2"])
        co = float(request.form["co"]) 
        o3 = float(request.form["o3"])
        no2 = float(request.form["no2"])
        input_features = [[pm10, pm25, so2, co, o3, no2,]]


        dataM.MetodeKnn()
        model = pickle.load(open("modelKnnPencemaran.pkl", "rb"))
        resultKNN = model.predict(input_features)[0]
     

        dataM.MetodeTree()
        model = pickle.load(open("modelTreePencemaran.pkl", "rb"))
        resultTree = model.predict(input_features)[0]

        
        dataM.MetodeNaiveBayes()
        model = pickle.load(open("modelNBPencemaran.pkl", "rb"))
        resultNB = model.predict(input_features)[0]
        dataInputan = [pm10, pm25, so2, co, o3, no2]
        
        return render_template("index.html", result=result, dataInputan=dataInputan, resultKNN=resultKNN,resultNB=resultNB,resultTree=resultTree)        
           
    else: 
        return render_template("index.html", result=result,resultKNN='', dataInputan=dataInputan)
    
if __name__=="__main__":
    flask_app.run(debug=True)