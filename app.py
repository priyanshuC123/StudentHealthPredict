from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from model import predict_depression

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()
        risk_score = predict_depression(user_input)
        risk_category = categorize_score(risk_score)
        return jsonify({"score": risk_score, "category": risk_category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def categorize_score(score):
    if score <= 20:
        return 'Minimal Risk (0-20)'
    elif score <= 40:
        return 'Mild Risk (21-40)'
    elif score <= 60:
        return 'Moderate Risk (41-60)'
    elif score <= 80:
        return 'High Risk (61-80)'
    else:
        return 'Very High Risk (81-100)'


if __name__ == '__main__':
    app.run(debug=True)


