import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn import preprocessing
en = preprocessing.LabelEncoder()

from sklearn.preprocessing import StandardScaler
stdscale = StandardScaler()

app = Flask (__name__)    #Initialising the Flask application
model = pickle.load(open('model.pkl', 'rb'))    #Opening the pickled model file in read bytes mode

@app.route('/')
def man():
    return render_template('Home.html')    #WORKS TILL HERE

@app.route('/', methods=['POST'])
def home():
    d1 = float(request.form['fuel'])
    d2 = float(request.form['seller'])
    d3 = float(request.form['tran'])
    d4 = float(request.form['owner'])
    d5 = float(request.form['Age'])
    d6 = float(request.form['km'])
    d7 = float(request.form['cp'])
    arr = np.array([[d1, d2, d3, d4, d5, d6, d7]])
    j, k = arr.shape
    arr = pd.DataFrame(arr)
    for i in range(4, 7, 1):
        arr[[i]] = stdscale.fit_transform(arr[[i]])
    z = model.predict(arr.iloc[:, :].values)
    z = stdscale.inverse_transform(z)
    return render_template('Home.html', data=round(float(z), 2))

    #unproc_col = [int(x) for x in request.form.values()]
    #proc_col = [np.array(unproc_col)]
    #output = round(prediction[0], 2)    
    #prediction = model.predict(proc_col) 

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__=='__main__':
    app.run(debug=True)

