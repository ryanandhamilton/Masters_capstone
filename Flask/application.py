# save this as app.py
from flask import Flask, escape, request, render_template
import numpy as np
import pandas as pd
import dill
import sklearn
import joblib

application = Flask(__name__)

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ML Model Code --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/')
@application.route('/about')
def about():

    return render_template("about.html")
    
@application.route('/projects')
def projects():

    return render_template("projects.html")

@application.route('/Predictor')
def Predictor():

    return render_template("Predictor.html")
    
def preprocessDataAndPredict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10):
    # keep all inputs in array
    data = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])
    
    # open file
    file = open("finalModel.pkl", "rb")
    
    # load trained model
    trained_model = joblib.load(file)

    # predict
    prediction = trained_model.predict(data.reshape(1, -1))

    return round(prediction[0], 0)

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # get form data
        x1 = request.form.get('x1')
        x2 = request.form.get('x2')
        x3 = request.form.get('x3')
        x4 = request.form.get('x4')
        x5 = request.form.get('x5')
        x6 = request.form.get('x6')
        x7 = request.form.get('x7')
        x8 = request.form.get('x8')
        x9 = request.form.get('x9')
        x10 = request.form.get('x10')
        t1 = request.form.get('t1')
        t2 = request.form.get('t2')
        t3 = request.form.get('t3')
        t4 = request.form.get('t4')
        t5 = request.form.get('t5')
        t6 = request.form.get('t6')
        t7 = request.form.get('t7')
        t8 = request.form.get('t8')
        t9 = request.form.get('t9')
        t10 = request.form.get('t10')
        p1 = request.form.get('p1')
        p2 = request.form.get('p2')
        p3 = request.form.get('p3')
        p4 = request.form.get('p4')
        p5 = request.form.get('p5')
        p6 = request.form.get('p6')
        p7 = request.form.get('p7')
        p8 = request.form.get('p8')
        p9 = request.form.get('p9')
        p10 = request.form.get('p10')

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)
            # pass prediction to template
            if(prediction==0):
                return render_template('predict_0.html')
            if(prediction>0):
                return render_template('predict_1.html')

        except ValueError:
            return "Please Enter valid values"

        pass
    pass


# Run on Correct Port
if __name__ == '__main__':
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run(host="localhost", port=5000, debug=True)