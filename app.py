# importing the necessary dependencies

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

# initializing a flask app
app = Flask(__name__)


@app.route("/", methods=['GET'])
@cross_origin()
def homepage():
    return render_template("index.html")


@app.route('/predict', methods=['POST','GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    """'Try to read the data inputs given by the user and predict the
        result by using the loaded model(pickle file) and showing the
        result through a web UI """
    if request.method == 'POST':
        try:
            # CRIM: Per capita crime rate by town
            crime_rate = float(request.form['CRIME_RATE'])

            # CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            chas = request.form['CHAS']

            if chas == "1":
                chas = 1
            else:
                chas = 0

            # RM: Average number of rooms per dwelling
            rm = float(request.form['RM'])

            # DIS: Weighted distances to five Boston employment centers
            dis = float(request.form['DIS'])

            # RAD: Index of accessibility to radial highways
            rad = float(request.form['RAD'])

            # TAX: Full-value property tax rate per 10,000 us Dollars
            tax = float(request.form['TAX'])

            # PTRATIO : Pupil-teacher ratio by town
            ptratio = float(request.form['PTRATIO'])

            # B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
            b = float(request.form['B'])

            # LSTAT: Percentage of lower status of the population
            lstat = float(request.form['LSTAT'])
            scalable_input = scalar.fit_transform([[crime_rate, chas, rm, dis, rad, tax, ptratio, b, lstat ]])

            physical_file = "elasticnet.pickle"

            loaded_model = pickle.load(open(physical_file, 'rb'))

            prediction = loaded_model.predict(scalable_input)

            return render_template('results.html', prediction=round(prediction[0],2))

        except Exception as err:
            print("The exception message is : ",  err)
            return "Something Wrong!!"

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
