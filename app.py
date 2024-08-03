from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Admission Chance: {output * 100}%')

if __name__ == "__main__":
    app.run(debug=True)
