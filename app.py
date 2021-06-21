import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
import pickle
app = Flask(__name__)
from flask import json
model = pickle.load(open('Occupancy.pkl', 'rb'))
modela = pickle.load(open('flightfare.pkl', 'rb'))

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predicts',methods=['POST'])
def predicts():
    data = request.get_json()
    data=data['dat']
    occ=[]
    for i in data:
        y=model.predict([[i[0],i[1],i[2],int(i[3]),i[4],i[5]]])
        occ.append(y[0])
    actp=[]
    j=0
    for i in data:
        ocx=occ[j]
        j=j+1
        y=modela.predict([[i[0],i[1],i[2],int(i[3]),i[5],ocx,i[6],i[7]]])
        actp.append(int(y[0]))
    return str(actp)

if __name__ == "__main__":
    app.run(debug = True)
    
