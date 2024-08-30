from flask import Flask, render_template, request
from utils import *
import pandas as pd
import sklearn
import joblib
# import xgboost as xgb


app = Flask(__name__, static_url_path="/static", static_folder="static")

# load saved model
loaded_pipeline = joblib.load("../model/housing_pipeline.pkl")


@app.route("/")
def hello_world():
    return render_template("index.html",locations=locations, types=types)

@app.route("/predict", methods=["POST"])
def predict():
    # fetch all user input from the form
    location = request.form["location"]
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    toilets = int(request.form["toilets"])
    parking_spaces = int(request.form["parking_space"])
    house_type = request.form["type"]
    
    # dictionary
    input_dict = {
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "parking_spaces": [parking_spaces],
    "location": [location],
    "toilets": [toilets],
    "type": [house_type]
    }
   
    # create a DataFrame
    input_df = pd.DataFrame.from_dict(input_dict)
    
    # make prediction
    prediction = loaded_pipeline.predict(input_df)
    
    return f"Predicted price {prediction:,.2f}"
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)