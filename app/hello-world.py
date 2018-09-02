"""Filename: hello-world.py
  """
import flask
from flask import Flask, render_template, request
import pickle
from sklearn.externals import joblib






app = Flask(__name__)

@app.route('/')

@app.route("/index")
def index():
   return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        #return render_template('index.html', label="3")
#       data=request.get_json()
#       years_of_experience = float(data["yearsOfExperience"])
#       regressor = joblib.load("./model.pkl")
#       return jsonify(lin_reg.predict(years_of_experience).tolist())
       lin_reg = joblib.load('../model.pkl')
       regressor = pickle.loads(lin_reg)
       y_pred = regressor.predict(['RSPM/PM10'])
       return y_pred,render_template('index.html', label="3")
       
    
if __name__ == '__main__':

	#modelfile = 'model.pkl'    

	#model = pickle.load(open(modelfile, 'rb'))

	#print("loaded OK")

	app.run(debug=True)

    