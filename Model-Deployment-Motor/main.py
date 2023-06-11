import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify

class RSquared(tf.keras.metrics.Metric):
    def __init__(self, name='r_squared', **kwargs):
        super(RSquared, self).__init__(name=name, **kwargs)
        self.total_residual = self.add_weight(name='total_residual', initializer='zeros')
        self.total_total = self.add_weight(name='total_total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        self.total_residual.assign_add(residual)
        self.total_total.assign_add(total)

    def result(self):
        r2_score = 1 - tf.math.divide_no_nan(self.total_residual, self.total_total)
        return r2_score

    def reset_states(self):
        self.total_residual.assign(0.0)
        self.total_total.assign(0.0)

normalized_df = pandas.read_csv('data_motor.csv')

def prepare_data(mileage, brand, model, tahun, cc):
    # Input nilai untuk setiap fitur
    input_value1 = (mileage)
    input_value2 = (brand)
    input_value3 = (model)
    input_value4 = (tahun)
    input_value5 = (cc)

    # Nama-nama kolom
    column_names2 = normalized_df.columns[1:14]
    column_names3 = normalized_df.columns[14:137]
    column_names4 = normalized_df.columns[137:160]
    column_names5 = normalized_df.columns[160:207]

    features = [input_value1] + \
               [1 if col == input_value2 else 0 for col in column_names2] + \
               [1 if col == input_value3 else 0 for col in column_names3] + \
               [1 if col == input_value4 else 0 for col in column_names4] + \
               [1 if col == input_value5 else 0 for col in column_names5]

    return features

def predict(x):
    scaler = MinMaxScaler()
    scaler.fit(normalized_df.iloc[:,1:])
    data_predict = scaler.transform([x])
    predictions = model.predict(data_predict)
    return predictions.tolist()

app = Flask(__name__)

@app.route("/", methods=["POST"])
def index():
    if request.method == "POST":
        try:
            input_data = request.get_json()
            if not input_data:
                return jsonify({"error": "no data"})

            feature1 = input_data.get('feature1')
            feature2 = input_data.get('feature2')
            feature3 = input_data.get('feature3')
            feature4 = input_data.get('feature4')
            feature5 = input_data.get('feature5')
            features = prepare_data(feature1, feature2, feature3, feature4, feature5)
            print(len(features))
            prediction = predict(features)

            data = {"prediction": prediction}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

if __name__ == "__main__":
    # Register custom metric as a custom object
    keras.utils.get_custom_objects().update({'RSquared': RSquared})

    # Load the model
    model = keras.models.load_model("model-motor.h5")

    app.run(debug=True)
