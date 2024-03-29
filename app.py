from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    input_features = [float(x) for x in request.form.values()]
    final_features = [np.array(input_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Iris Type should be {}'.format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # For direct API calls trought request
    data = request.get_json(force=True)
    prediction = model.predict(([np.array(list(data.values()))]))
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
