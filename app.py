# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:29:21 2021

@author: Nagesh
"""

# Reference: https://github.com/krishnaik06/Heroku-Demo/blob/master/app.py

#%%
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from werkzeug.utils import secure_filename


#%%
with open("./LR_L1_reg.pkl", "rb") as f: 
    model = pickle.load(f)

with open("./x_scaler.pkl", "rb") as f: 
    x_scaler = pickle.load(f)

with open("./y_scaler.pkl", "rb") as f: 
    y_scaler = pickle.load(f)


#%%

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Reference: https://stackoverflow.com/a/39808586/6645883
@app.route("/predict", methods = ["GET", "POST"])
def getFileAndPredict():
    if request.method == "POST":
        file = request.files["myfile"]
        filename = secure_filename(file.filename)
        file.save(filename)
        
        if not filename.endswith(".txt"): 
            delete_file(filename)
            return render_template("index.html", prediction_text1 = "Please give a text file as input", prediction_text2 = "")
        
        with open(filename) as f:
            file_content = f.read()
        
        y_pred, y_true = predict(file_content)
        delete_file(filename)
        return_string1 = "Actual closing index value for next working day is " + str(y_true) + "."
        return_string2 = "Predicted closing index value for next working day is " + str(np.round(y_pred[0,0], decimals = 3)) + "."
        return render_template("index.html", prediction_text1 = return_string1, prediction_text2 = return_string2)
    else:
        return render_template("index.html")

def delete_file(filename):
    try: 
        os.remove(filename)
    except Exception as e: 
        print("error: ", e)

def predict(file_content):
    xy = file_content.split(",")
    x = np.array(xy[1 : -1])
    x_scaled = x_scaler.transform(x.reshape(1, -1))
    y_pred_scaled = model.predict(x_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))
    return y_pred, xy[-1]

if __name__ == "__main__":
    app.run(debug = True)
        
      