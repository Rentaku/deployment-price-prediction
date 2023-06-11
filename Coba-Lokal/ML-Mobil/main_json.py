import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

normalized_df = pandas.read_csv('data_mobil.csv')

def prepare_data(mileage, manufacture, model, category, year, gear_box_type):
    # Input nilai untuk setiap fitur
    input_value0 = mileage
    input_value1 = str(manufacture)
    input_value2 = str(model)
    input_value3 = str(category)
    input_value4 = str(year)
    input_value5 = str(gear_box_type)

    column_names1 = normalized_df.columns[1:31]  # Nama-nama kolom
    column_names2 = normalized_df.columns[31:1597]
    column_names3 = normalized_df.columns[1597:1608]
    column_names4 = normalized_df.columns[1608:1629]
    column_names5 = normalized_df.columns[1629:1633]

    features = [input_value0] + \
               [1 if col == input_value1 else 0 for col in column_names1] + \
               [1 if col == input_value2 else 0 for col in column_names2] + \
               [1 if col == input_value3 else 0 for col in column_names3] + \
               [1 if col == input_value4 else 0 for col in column_names4] + \
               [1 if col == input_value5 else 0 for col in column_names5]

    return features

def predict(x):
    scaler = MinMaxScaler()
    data_predict = scaler.fit_transform([x])
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
            feature6 = input_data.get('feature6')
            features = prepare_data(feature1, feature2, feature3, feature4, feature5, feature6)
            prediction = predict(features)

            data = {"prediction": prediction}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"

if __name__ == "__main__":
    keras.utils.get_custom_objects().update({'RSquared': RSquared})

    with open('model.json','r') as f:
        model_json = f.read()

    model = keras.models.load_model(model_json)  # Load the model inside the custom object scope
    model.load_weights('group1-shard1of3.bin')
    model.load_weights('group1-shard2of3.bin')
    model.load_weights('group1-shard3of3.bin')

    app.run(debug=True)