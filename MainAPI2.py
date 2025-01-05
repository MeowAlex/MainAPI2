from flask import Flask, jsonify, Response, send_file
import requests
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import io
from datetime import timedelta, datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import openpyxl

app = Flask(__name__)

class GraphGenerator24:
    def __init__(self):
        # Initialize any necessary variables if needed
        pass

    def create_graph(self, Forecast_Dates, arima_FORECAST, Recent_VALUES, Observed_Dates):
        # Create the matplotlib figure
        fig, ax = plt.subplots()

        ax.plot(Observed_Dates, Recent_VALUES, label="Observed", marker='o')
        ax.plot(Forecast_Dates, arima_FORECAST, label="Forecast", marker='^')

        # Customize the plot
        ax.set_xlabel("MM-DD T")
        ax.set_ylabel("Ap Index")
        ax.legend()
        ax.grid()

        ax.tick_params(axis='x', labelrotation=20)
        ax.tick_params(axis='y', labelrotation=90)

        # Adjust layout to fix padding
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.35)  # Adjust margins as needed

        # Set figure background to transparent
        fig.patch.set_facecolor('none')  # Transparent figure background
        ax.set_facecolor('none')  # Transparent axes background

        return fig

class GeomagneticPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.xgb = XGBClassifier(
            n_estimators=180,
            max_depth=1000,
            learning_rate=10000,
            random_state=42,
            use_label_encoder=False
        )

        # Load datasets
        self.SC19_24 = pd.read_excel('Daily Solar Cycle 19-24 Classifier.xlsx')
        self.SC25 = pd.read_excel('Daily Solar Cycle 25 Classifier.xlsx')

        # Prepare training data
        self.X_train = self.SC19_24[['Daily Ap', 'SN']].dropna()
        self.y_train = self.SC19_24['Latitudes Day'].loc[self.X_train.index]

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Map labels to integers
        self.unique_classes = sorted(set(self.y_train))
        self.class_mapping = {label: idx for idx, label in enumerate(self.unique_classes)}
        self.inverse_mapping = {idx: label for label, idx in self.class_mapping.items()}
        self.y_train_mapped = self.y_train.map(self.class_mapping)

        # Train the model
        self.xgb.fit(self.X_train_scaled, self.y_train_mapped)

    def get_latest_ap_index(self):
        url = 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            last_entry = data[-1]
            if isinstance(last_entry, list):
                return last_entry[-2]
        print("Failed to retrieve Ap index or unexpected data structure.")
        return None

    def predict_latitude(self):
        latest_ap_index = self.get_latest_ap_index()
        if latest_ap_index is None:
            return {"latitude": "Unavailable", "accuracy": "Unavailable"}

        latest_feature = self.scaler.transform([[latest_ap_index, 0]])
        predicted_class = self.xgb.predict(latest_feature)
        predicted_latitude = self.inverse_mapping[predicted_class[0]]

        X_test = self.SC25[['Daily Ap', 'SN']].dropna()
        y_test = self.SC25['Latitudes Day'].loc[X_test.index]
        X_test_scaled = self.scaler.transform(X_test)
        y_test_mapped = y_test.map(self.class_mapping)

        y_pred_mapped = self.xgb.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_mapped, y_pred_mapped)

        return {
            "latitude": predicted_latitude,
            "accuracy": f"{int(accuracy * 100)}%"
        }

class AuroraFuture:
    @staticmethod
    def TwentyFourHour():
        try:
            # Step 1: Fetch the Ap Index data
            url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

            # Fetch data from NOAA JSON URL
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()

            # Convert JSON to DataFrame
            df = pd.DataFrame(json_data[1:], columns=json_data[0])
            df['time_tag'] = pd.to_datetime(df['time_tag'])
            df['a_running'] = pd.to_numeric(df['a_running'], errors='coerce')

            # Ensure data is sorted by time
            df.sort_values(by='time_tag', inplace=True)

            # Extract the last 3 days of data (24 values per day)
            Previous_Data = df.tail(1 * 8)
            Recent_VALUES = Previous_Data['a_running'].values

            # Step 2: Fit the ARIMA model
            arima_MODEL = ARIMA(Recent_VALUES, order=(4, 1, 2))  # Order can be adjusted
            arima_FITTED = arima_MODEL.fit()

            # Step 3: Forecast the next 3 days (24 values each)
            forecast_steps = 1 * 8
            arima_FORECASt = arima_FITTED.forecast(steps=forecast_steps)

            # Ensure no negative forecast values by clipping to a minimum of 0
            arima_FORECAST = np.clip(np.round(arima_FORECASt).astype(int), 0, None)

            # Generate forecast timestamps (next 3 days)
            Last_Dates = df['time_tag'].iloc[-1]
            Forecast_Dates = [Last_Dates + timedelta(hours=3 * i) for i in range(1, forecast_steps + 1)]

            # Step 4: Calculate accuracy metrics
            observed_LAST_day = df['a_running'].iloc[-8:].values
            predicted_LAST_day = arima_FORECAST[:8]

            arima_MAE = mean_absolute_error(observed_LAST_day, predicted_LAST_day)
            arima_RMSE = np.sqrt(mean_squared_error(observed_LAST_day, predicted_LAST_day))

            Observed_Dates = df['time_tag'].iloc[-1 * 8:]

            return Forecast_Dates, Last_Dates, observed_LAST_day, predicted_LAST_day, arima_MAE, arima_RMSE, arima_FORECAST, Recent_VALUES, Observed_Dates

        except Exception as e:
            print("Error in predicting latitudes for the next 24 hours:", e)
            return None


    def predict_latitudes_next_24h(self):
        """
        Predicts aurora latitudes for the next 24 hours using ARIMA-predicted Ap Index values.
        """
        try:
            # Step 1: Fetch predicted Ap Index values from TwentyFourHour
            result = AuroraFuture.TwentyFourHour()
            if result is None:
                return None

            Forecast_Dates, Last_Dates, observed_LAST_day, predicted_LAST_day, arima_MAE, arima_RMSE, arima_FORECAST, Recent_VALUES, Observed_Dates = result

            # Step 2: Predict latitudes for each forecasted Ap value
            geomagnetic_predictor = GeomagneticPredictor()
            predicted_latitudes = []

            for ap_value in arima_FORECAST:
                # Scale the input features (assuming SN = 0 for predictions)
                scaled_features = geomagnetic_predictor.scaler.transform([[ap_value, 0]])
                predicted_class = geomagnetic_predictor.xgb.predict(scaled_features)
                predicted_latitude = geomagnetic_predictor.inverse_mapping[predicted_class[0]]
                predicted_latitudes.append(predicted_latitude)

            # Combine predictions with forecast timestamps
            predictions_with_time = list(zip(Forecast_Dates, predicted_latitudes))

            # Convert results to JSON-serializable format
            serialized_forecast_dates = [str(date) for date in Forecast_Dates]
            return {
                "Forecast Dates": serialized_forecast_dates,
                "Predicted Latitudes": predicted_latitudes,
            }

        except Exception as e:
            print("Error in predicting latitudes for the next 24 hours:", e)
            return None

@app.route('/TwentyFourHourLats', methods=['GET'])
def TwentyFourHourLats():
    try:
        aurora_future = AuroraFuture()
        prediction_results = aurora_future.predict_latitudes_next_24h()

        if prediction_results is None:
            return jsonify({"error": "Failed to generate predictions."}), 500

        return jsonify(prediction_results), 200
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route('/TwentyFourHour', methods=['GET'])
def TwentyFourHour():
    try:
        # Step 1: Fetch the Ap Index data
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

        # Fetch data from NOAA JSON URL
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()

        # Convert JSON to DataFrame
        df = pd.DataFrame(json_data[1:], columns=json_data[0])
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        df['a_running'] = pd.to_numeric(df['a_running'], errors='coerce')

        # Ensure data is sorted by time
        df.sort_values(by='time_tag', inplace=True)

        # Extract the last 3 days of data (24 values per day)
        Previous_Data = df.tail(1 * 8)
        Recent_VALUES = Previous_Data['a_running'].values

        # Step 2: Fit the ARIMA model
        arima_MODEL = ARIMA(Recent_VALUES, order=(4, 1, 2))  # Order can be adjusted
        arima_FITTED = arima_MODEL.fit()

        # Step 3: Forecast the next 3 days (24 values each)
        forecast_steps = 1 * 8
        arima_FORECASt = arima_FITTED.forecast(steps=forecast_steps)

        # Ensure no negative forecast values by clipping to a minimum of 0
        arima_FORECAST = np.clip(np.round(arima_FORECASt).astype(int), 0, None)

        # Generate forecast timestamps (next 3 days)
        Last_Dates = df['time_tag'].iloc[-1]
        Forecast_Dates = [Last_Dates + timedelta(hours=3 * i) for i in range(1, forecast_steps + 1)]

        # Step 4: Calculate accuracy metrics
        observed_LAST_day = df['a_running'].iloc[-8:].values
        predicted_LAST_day = arima_FORECAST[:8]

        arima_MAE = mean_absolute_error(observed_LAST_day, predicted_LAST_day)
        arima_RMSE = np.sqrt(mean_squared_error(observed_LAST_day, predicted_LAST_day))

        Observed_Dates = df['time_tag'].iloc[-1 * 8:]

        # Prepare the data for the JSON response
        response_data = {
            "forecast_dates": [date.strftime('%Y-%m-%d %H:%M:%S') for date in Forecast_Dates],
            "predicted_values": predicted_LAST_day.tolist(),
            "MAE": arima_MAE,
            "RMSE": arima_RMSE,
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/TwentyFourHourGraph', methods=['GET'])
def TwentyFourHourGraph():
    try:
        # Fetch data
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

        # Fetch data from NOAA JSON URL
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()

        # Convert JSON to DataFrame
        df = pd.DataFrame(json_data[1:], columns=json_data[0])
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        df['a_running'] = pd.to_numeric(df['a_running'], errors='coerce')

        # Ensure data is sorted by time
        df.sort_values(by='time_tag', inplace=True)

        # Extract the last 3 days of data (24 values per day)
        Previous_Data = df.tail(1 * 8)
        Recent_VALUES = Previous_Data['a_running'].values

        # Step 2: Fit the ARIMA model
        arima_MODEL = ARIMA(Recent_VALUES, order=(4, 1, 2))  # Order can be adjusted
        arima_FITTED = arima_MODEL.fit()

        # Step 3: Forecast the next 3 days (24 values each)
        forecast_steps = 1 * 8
        arima_FORECASt = arima_FITTED.forecast(steps=forecast_steps)

        # Ensure no negative forecast values by clipping to a minimum of 0
        arima_FORECAST = np.clip(np.round(arima_FORECASt).astype(int), 0, None)

        # Generate forecast timestamps (next 3 days)
        Last_Dates = df['time_tag'].iloc[-1]
        Forecast_Dates = [Last_Dates + timedelta(hours=3 * i) for i in range(1, forecast_steps + 1)]

        # Step 4: Calculate accuracy metrics
        observed_LAST_day = df['a_running'].iloc[-8:].values
        predicted_LAST_day = arima_FORECAST[:8]

        arima_MAE = mean_absolute_error(observed_LAST_day, predicted_LAST_day)
        arima_RMSE = np.sqrt(mean_squared_error(observed_LAST_day, predicted_LAST_day))

        Observed_Dates = df['time_tag'].iloc[-1 * 8:]

        # Generate graph using GraphGenerator class
        graph_generator24 = GraphGenerator24()
        fig = graph_generator24.create_graph(Forecast_Dates, arima_FORECAST, Recent_VALUES, Observed_Dates)

        # Convert graph to PNG and return as a response
        output = io.BytesIO()
        fig.savefig(output, format='png')
        output.seek(0)
        plt.close(fig)

        return Response(output.getvalue(), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
