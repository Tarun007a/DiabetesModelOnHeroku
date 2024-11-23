# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:25:17 2024

@author: tarun
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()
class model_input(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int
    
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameter : model_input):
    input_data_as_json = input_parameter.json()
    input_dictonary = json.loads(input_data_as_json)
    
    preg = input_dictonary['Pregnancies']
    glu = input_dictonary['Glucose']
    bp = input_dictonary['BloodPressure']
    skin = input_dictonary['SkinThickness']
    insulin = input_dictonary['Insulin']
    bmi = input_dictonary['BMI']
    dpf = input_dictonary['DiabetesPedigreeFunction']
    age = input_dictonary['Age']
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    predicion = loaded_model.predict([input_list])
    if predicion[0] == 0 :
        return "The person is not diabetic"
    else : 
        return "The person is diabetic"

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    