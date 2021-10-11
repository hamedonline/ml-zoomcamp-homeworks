import pickle
from flask import Flask, request, jsonify
from waitress import serve


def load_model(path_to_model, path_to_dv):
    with open(path_to_model, 'rb') as model_file, open(path_to_dv, 'rb') as dv_file:
        model = pickle.load(model_file)
        dv = pickle.load(dv_file)
    return model, dv

def predict_churn(model, dv, customer_data):
    X = dv.transform([customer_data])
    y_pred = model.predict_proba(X)[0, 1]
    is_churning = y_pred >= 0.5

    result = {
        'churn': bool(is_churning),
        'churn_probability': float(y_pred)
    }
    return jsonify(result)


app = Flask('churn')
@app.route('/churn/predict', methods=['POST'])
def predict():
    customer_data = request.get_json()

    model, dv = load_model(path_to_model='models/model1.bin',
                           path_to_dv='models/dv.bin')
    prediction_result = predict_churn(model=model, dv=dv, customer_data=customer_data)

    return prediction_result


if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=9696)
    serve(app, host='0.0.0.0', port=9696)
