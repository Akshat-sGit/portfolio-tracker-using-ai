from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    features = np.array(data['Predictions']).reshape(1, -1)  # Convert to array
    prediction = model.predict(features)  # Get prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
