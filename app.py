import pickle
from flask import Flask,request,app,render_template,jsonify,url_for
import numpy as np
import pandas as pd


app = Flask(__name__)

## Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
## Load the standard scaler
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    new_data= scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(i) for i in request.form.values()]
    final_data = scaler.transform(np.array(data).reshape(1,-1))
    predicted_output = regmodel.predict(final_data)[0]
    return render_template('home.html',predicted_value="The predicted House price is {}".format(np.round(predicted_output),2))


if __name__ == '__main__':
    app.run()
