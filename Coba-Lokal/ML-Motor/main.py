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

#list_fitur = ['mileage', 'aprilia', 'demak', 'ducati', 'hero', 'honda', 'ktm', 'kawasaki', 'kinetic', 'loncin', 'mahindra', 'suzuki', 'tvs', 'yamaha', '250', '90', 'ax1', 'achiever', 'apache', 'balius', 'bandit', 'benly', 'blaze', 'boxer', 'cb 125', 'cb hornet', 'cb shine', 'cb trigger', 'cb unicorn', 'cb4', 'cbr', 'cbz', 'cd', 'cd 110', 'cd 125', 'cd 70', 'cd 90', 'cd down', 'cm custom', 'centra', 'centuro', 'civic', 'd tracker', 'd7', 'dr', 'dt', 'dtm', 'dzm', 'dzr', 'dash', 'dawn', 'djebel', 'dream', 'duke', 'duke 200', 'estrella', 'ftr', 'fz', 'fz s', 'fz25', 'fazer', 'flame', 'gn 125', 'gn 250', 'gs 125', 'gixxer', 'gladiator', 'glamour', 'grass tracker', 'gusto', 'hf dawn', 'hf deluxe', 'hornet', 'hunk', 'ignitor', 'intruder', 'jade', 'karizma', 'ld', 'lx', 'libero', 'little cub', 'md', 'mt 15', 'magna', 'mate', 'metro', 'monster', 'nv400', 'navi', 'ninja', 'ntorq', 'pcx', 'passion', 'passion plus', 'passion pro', 'r15', 'rc', 'rebel', 'rio', 'sx', 'sz-rr', 'safari', 'saluto', 'savage supra', 'scooty zest', 'sky born', 'skyline', 'splender', 'splender plus', 'splender i smart', 'star city plus', 'star sport', 'streak', 'stryker', 'stunner', 'super club', 'super splender', 'ttr', 'tw', 'tzr', 'tropica', 'tuono', 'twister', 'uzo 125', 'vtr', 'victor', 'virago', 'warrior', 'wego', 'x-blade', 'xl 100', 'xl super', 'xlr', 'xr', 'xtream', 'zoomer', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '1,500 cc', '100 cc', '110 cc', '111 cc', '113 cc', '115 cc', '120 cc', '125 cc', '140 cc', '149 cc', '150 cc', '152 cc', '153 cc', '154 cc', '155 cc', '156 cc', '157 cc', '160 cc', '177 cc', '180 cc', '197 cc', '2,010 cc', '2,017 cc', '200 cc', '223 cc', '225 cc', '228 cc', '230 cc', '249 cc', '250 cc', '366 cc', '400 cc', '48 cc', '49 cc', '50 cc', '555 cc', '60 cc', '650 cc', '696 cc', '70 cc', '75 cc', '8,000 cc', '80 cc', '85 cc', '89 cc', '90 cc', '99 cc', 'price']

#normalized_df = pandas.DataFrame(columns=list_fitur)
normalized_df = pandas.read_csv('data_motor.csv')

def prepare_data(mileage, brand, model, tahun, cc):
    # Input nilai untuk setiap fitur
    input_value1 = mileage
    input_value2 = brand
    input_value3 = model
    input_value4 = tahun
    input_value5 = cc

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
    return predictions.to_list()

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
