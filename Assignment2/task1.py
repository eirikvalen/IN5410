import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.layers import Dense
from Assignment2.read_from_file import read_training_data, read_input_data, read_solution
from Assignment2.plot_methods import RMSE, plot_results, scatter_plot

trainingData = read_training_data()
windspeed_training = trainingData[0]
power_training = trainingData[1]

windspeed_input = read_input_data()[0]

solution = read_solution()
actual_power = solution[0]
timestamp = solution[1]


def linear_regression(windspeed, power, forecast_input, actual_power, show_plots):
    model = LinearRegression()
    model.fit(windspeed, power)

    predicted_power = model.predict(forecast_input)

    rmse = RMSE(actual_power, predicted_power)
    scatter_plot(forecast_input, actual_power, predicted_power, "linear regression")

    if show_plots:
        plot_results(actual_power, predicted_power, "linear regression")

    return predicted_power, rmse


def k_nearest_neighbor(windspeed, power, forecast_input, actual_power, show_plots):
    k = 20
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(windspeed, power)

    predicted_power = model.predict(forecast_input)

    rmse = RMSE(predicted_power, actual_power)
    scatter_plot(forecast_input, actual_power, predicted_power, "knn")


    if show_plots:
        plot_results(actual_power, predicted_power, "KNN")

    return predicted_power, rmse


def supported_vector_regression(windspeed, power, forecast_input, actual_power, show_plots):
    model = SVR()
    model.fit(windspeed, power)

    predicted_power = model.predict(forecast_input)

    rmse = RMSE(actual_power, predicted_power)
    scatter_plot(forecast_input, actual_power, predicted_power, "svr")


    if show_plots:
        plot_results(actual_power, predicted_power, "SVR")

    return predicted_power, rmse


def artificial_neural_network(windspeed, power, forecast_input, actual_power, show_plots):
    # This returns a tensor. Since the input only has one column
    inputs = Input(shape=(1,))

    # Layers and number of neurons in hidden layers
    x = Dense(32, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    predictions = Dense(1)(x)

    # The model that includes the input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mse'])

    model.fit(windspeed, power, epochs=100, batch_size=100, verbose=1, validation_split=0.2)

    # Predicting power_training with new data
    predicted_power = model.predict(forecast_input)

    actual_power = actual_power.reshape((-1, 1))

    rmse = RMSE(actual_power, predicted_power)

    scatter_plot(forecast_input, actual_power, predicted_power, "ann")


    if show_plots:
        plot_results(actual_power, predicted_power, "neural network")

    return predicted_power, rmse


def run_all_models_task1(show_plots, write_to_csv):
    lin_reg = linear_regression(windspeed_training, power_training, windspeed_input, actual_power, show_plots)
    knn = k_nearest_neighbor(windspeed_training, power_training, windspeed_input, actual_power, show_plots)
    svr = supported_vector_regression(windspeed_training, power_training, windspeed_input, actual_power, show_plots)
    ann = artificial_neural_network(windspeed_training, power_training, windspeed_input, actual_power, show_plots)

    if write_to_csv:
        header = "TIMESTAMP, POWER"

        pd.DataFrame({"TIMESTAMP": timestamp, "POWER": lin_reg[0]}).to_csv(
            "../ForecastResults/ForecastTemplate1-LR.csv",
            index=None, header=header)
        pd.DataFrame({"TIMESTAMP": timestamp, "POWER": knn[0]}).to_csv("../ForecastResults/ForecastTemplate1-kNN.csv",
                                                                       index=None, header=header)
        pd.DataFrame({"TIMESTAMP": timestamp, "POWER": svr[0]}).to_csv("../ForecastResults/ForecastTemplate1-SVR.csv",
                                                                       index=None, header=header)
        pd.DataFrame({"TIMESTAMP": timestamp, "POWER": ann[0].flat}).to_csv(
            "../ForecastResults/ForecastTemplate1-NN.csv",
            index=None, header=header)

    print("RMSE timeseries LinReg:" + str(lin_reg[1]))
    print("RMSE timeseries KNN:" + str(knn[1]))
    print("RMSE timeseries SVR:" + str(svr[1]))
    print("RMSE timeseries ANN:" + str(ann[1]))



run_all_models_task1(False, False)
