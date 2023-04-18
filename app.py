from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
app = Flask(__name__)

modelLR=pickle.load(open('modelLR.pkl','rb'))
modelKNN=pickle.load(open('modelKNN.pkl','rb'))
modelDT=pickle.load(open('modelDT.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x  for x in request.form.values()]
    print(int_features)
    final=[np.array(int_features)]
    print(final)
    prediction1=modelLR.predict(final)
    prediction2=modelKNN.predict(final)
    prediction3=modelDT.predict(final)
    output = prediction1[i]*0.1 + prediction2[i]*0.3 + prediction3[i]*0.6
    return render_template('index.html',pred='{}'.format(output))
 
if __name__ == '__main__':
    app.run(debug=True)
  
