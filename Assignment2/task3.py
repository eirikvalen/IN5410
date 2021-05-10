import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from Assignment2.read_from_file import read_training_data, read_input_data, read_solution
from Assignment2.plot_methods import RMSE, plot_results, plot_results_3_inputs

training_data = read_training_data()
windspeed_training = training_data[0]
power_training = training_data[1]

input_data = read_input_data()
windspeed_input = input_data[0]
wind_direction = input_data[1]
windspeed_and_direction_input = input_data[2]

solution = read_solution()
actual_power = solution[0]
timestamp = solution[1]


def create_timeseries(powers, windowsize):
    rows = len(powers) - windowsize
    x = np.zeros((rows, windowsize))
    y = np.zeros(rows)

    for i in range(rows):
        for j in range(windowsize):
            x[i][j] = powers[i + j]

        y[i] = powers[i + windowsize]

    return x, y


def timeseries_linear_regression(x, y, forecast_input, actual_power, windowsize, show_plots):
    model = LinearRegression()
    model.fit(x, y)
    predicted_power = model.predict(forecast_input)

    actual_power_comparison = actual_power[windowsize:]

    rmse = RMSE(actual_power_comparison, predicted_power)

    if show_plots:
        plot_results(actual_power[windowsize:], predicted_power, "timeseries linear regression")

    return predicted_power, rmse


def timeseries_supported_vector_regression(x, y, forecast_input, actual_power, windowsize, show_plots):
    model = SVR()
    model.fit(x, y)

    predicted_power = model.predict(forecast_input)
    actual_power_comparison = actual_power[windowsize:]

    rmse = RMSE(actual_power_comparison, predicted_power)

    if show_plots:
        plot_results(actual_power[windowsize:], predicted_power, "timeseries SVR")

    return predicted_power, rmse


def timeseries_artificial_neural_networks(x, y, forecast_input, actual_power, windowsize, show_plots):
    # Layers and number of neurons in hidden layers
    inputs = Input(shape=(windowsize,))
    layer = Dense(32, activation='relu')(inputs)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(100, activation='relu')(layer)
    predictions = Dense(1)(layer)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mse'])

    model.fit(x, y, epochs=100, batch_size=100, verbose=1, validation_split=0.2)

    predicted_power = model.predict(forecast_input)
    actual_power_comparison = actual_power[windowsize:].reshape((-1, 1))

    rmse = RMSE(actual_power_comparison, predicted_power)

    if show_plots:
        plot_results(actual_power[windowsize:], predicted_power, "timeserie ANN")

    return predicted_power, rmse


def timeseries_recurrant_neural_network(x, y, forecast_input, actual_power, windowsize, show_plots):
    # reshaping input to (batch_size, timesteps, input_dim)
    x = x.reshape(-1, 1, windowsize)
    forecast_input3_d = np.reshape(forecast_input, (len(forecast_input), 1, windowsize))

    # using LSTM layer, used to accept a sequence of input data and make prediction
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, windowsize), activation='relu', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['mse'],
    )

    model.fit(x, y, epochs=100, batch_size=100, verbose=1)
    predicted_power = model.predict(forecast_input3_d)
    actual_power_comparison = actual_power[windowsize:].reshape((-1, 1))

    rmse = RMSE(actual_power_comparison, predicted_power)

    if show_plots:
        plot_results(actual_power[windowsize:], predicted_power, "timeseries RNN")

    return predicted_power, rmse


def run_all_models_task3(actual_power, windowsize, show_plots, write_to_csv):
    x, y = create_timeseries(power_training, windowsize)
    forecast_input = create_timeseries(actual_power, windowsize)[0]

    ts_lin_reg = timeseries_linear_regression(x, y, forecast_input, actual_power, windowsize, show_plots)
    ts_svr = timeseries_supported_vector_regression(x, y, forecast_input, actual_power, windowsize, show_plots)
    ts_ann = timeseries_artificial_neural_networks(x, y, forecast_input, actual_power, windowsize, show_plots)
    ts_rnn = timeseries_recurrant_neural_network(x, y, forecast_input, actual_power, windowsize, show_plots)

    if show_plots:
        plot_results_3_inputs(actual_power, ts_lin_reg[0], ts_svr[0], "actual power", "linear regression",
                              "supported vector regression",
                              "actual power, linear regression and SVR comparison for timeseries")

        plot_results_3_inputs(actual_power, ts_ann[0], ts_rnn[0], "actual power", "artificial neural network",
                              "recurrant neural network",
                              "actual power, ANN and RNN comparison for timeseries")

    if write_to_csv:
        header = "TIMESTAMP, POWER"

        pd.DataFrame({"TIMESTAMP": timestamp[windowsize:], "POWER": ts_lin_reg[0]}).to_csv(
            "../ForecastResults/ForecastTemplate3-LR.csv",
            index=None, header=header)
        pd.DataFrame({"TIMESTAMP": timestamp[windowsize:], "POWER": ts_svr[0]}).to_csv(
            "../ForecastResults/ForecastTemplate3-SVR.csv",
            index=None, header=header)
        pd.DataFrame({"TIMESTAMP": timestamp[windowsize:], "POWER": ts_ann[0].flat}).to_csv(
            "../ForecastResults/ForecastTemplate3-ANN.csv",
            index=None, header=header)
        pd.DataFrame({"TIMESTAMP": timestamp[windowsize:], "POWER": ts_rnn[0].flat}).to_csv(
            "../ForecastResults/ForecastTemplate3-RNN.csv",
            index=None, header=header)

    print("RMSE timeseries LinReg:" + str(ts_lin_reg[1]) + ", windowsize = " + str(windowsize))
    print("RMSE timeseries SVR:" + str(ts_svr[1]) + ", windowsize = " + str(windowsize))
    print("RMSE timeseries ANN:" + str(ts_ann[1]) + ", windowsize = " + str(windowsize))
    print("RMSE timeseries RNN:" + str(ts_rnn[1]) + ", windowsize = " + str(windowsize))


run_all_models_task3(actual_power, 1, False, True)
