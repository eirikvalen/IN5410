import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# ---------------- Reading from files ----------------
# Training data
df = pd.read_csv('TrainData.csv', sep=',')
windspeed_training = df["WS10"].to_numpy().reshape((-1, 1))
power_training = df["POWER"].to_numpy()
u = df["U10"].to_numpy()
v = df["V10"].to_numpy()
wind_direction = np.mod(180 + np.rad2deg(np.arctan2(v, u)), 360)

# Input data
df = pd.read_csv("WeatherForecastInput.csv", sep=',')
windspeed_input = df["WS10"].to_numpy().reshape((-1, 1))
u_input = df["U10"].to_numpy()
v_input = df["V10"].to_numpy()
direction_input = np.mod(180 + np.rad2deg(np.arctan2(v_input, u_input)), 360).reshape((-1, 1))
windspeed_and_direction_input = np.concatenate((windspeed_input, direction_input), axis=1)

# Solution (actual power)
df = pd.read_csv("Solution.csv", sep=',')
time = df["TIMESTAMP"].to_numpy()
actualPower = df["POWER"].to_numpy()


def RMSE(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def plotResults(actualPower, predictedPower, model):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xticklabels([])
    plt.plot(actualPower, color='lightseagreen', label="actual power")
    plt.plot(predictedPower, color='darkorange', label="predicted power")
    plt.legend()
    # plt.legend( bbox_to_anchor=(1.05, 1), loc='upper center')

    plt.title("True vs. predicted wind power, " + model)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()


def plotResults3Inputs(actualPower, predictedpower1, predictedPower, label1, label2, label3, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xticklabels([])
    plt.plot(actualPower, color='gray', label=label1)
    plt.plot(predictedpower1, color='lightseagreen', label=label2)
    plt.plot(predictedPower, color='darkorange', label=label3)
    plt.legend()

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()


def scatterPlot(forecastInput, actualPower, predictedPower, model):
    plt.scatter(forecastInput, actualPower, c="r", s=1.5, label="Actual")
    plt.scatter(forecastInput, predictedPower, c="b", s=1.5, label="Predicted")
    plt.title("True vs. predicted wind power, " + model)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()


# ---------------- For task 1 ----------------
def linearRegression(windspeed, power, forecastInput, actualPower, showPlots):
    model = LinearRegression()
    model.fit(windspeed, power)

    predictedPower = model.predict(forecastInput)

    rmse = RMSE(actualPower, predictedPower)

    if showPlots:
        plotResults(actualPower, predictedPower, "linear regression")

    return predictedPower, rmse


def kNearestNeighbor(windspeed, power, forecastInput, actualPower, showPlots):
    for k in [20]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(windspeed, power)

        predictedPower = model.predict(forecastInput)

        rmse = RMSE(predictedPower, actualPower)

        if showPlots:
            plotResults(actualPower, predictedPower, "KNN")

        return predictedPower, rmse


def supportedVectorRegression(windspeed, power, forecastInput, actualPower, showPlots):
    model = SVR()
    model.fit(windspeed, power)

    predictedPower = model.predict(forecastInput)

    rmse = RMSE(actualPower, predictedPower)

    if showPlots:
        plotResults(actualPower, predictedPower, "SVR")

    return predictedPower, rmse

def artificialNeuralNetworks(windspeed, power, forecastInput, actualPower, showPlots):
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
    predictedPower = model.predict(forecastInput)

    actualPower = actualPower.reshape((-1, 1))

    rmse = RMSE(actualPower, predictedPower)

    if showPlots:
        plotResults(actualPower, predictedPower, "neural network")

    return predictedPower, rmse


def runAllModelsTask1(showPlots):
    linReg = linearRegression(windspeed_training, power_training, windspeed_input, actualPower, showPlots)
    KNN = kNearestNeighbor(windspeed_training, power_training, windspeed_input, actualPower, showPlots)
    SVR = supportedVectorRegression(windspeed_training, power_training, windspeed_input, actualPower, showPlots)
    ANN = artificialNeuralNetworks(windspeed_training, power_training, windspeed_input, actualPower, showPlots)

    print("RMSE timeseries LinReg:" + str(linReg[1]))
    print("RMSE timeseries KNN:" + str(KNN[1]))
    print("RMSE timeseries SVR:" + str(SVR[1]))
    print("RMSE timeseries ANN:" + str(ANN[1]))


# runAllModelsTask1(True)



# ---------------- For task 2 ----------------
def multipleLinReg(windspeed, direction, power, forecastInput, actualPower, showPlots):
    direction = direction.reshape((-1, 1))

    x = np.concatenate((windspeed, direction), axis=1)

    model = LinearRegression()
    model.fit(x, power)

    predictedPower = model.predict(forecastInput)

    rmse = RMSE(actualPower, predictedPower)

    if showPlots:
        plotResults(actualPower, predictedPower, "multiple linear regression")

    return predictedPower, rmse


def runAllModelsTask2(showPlots):
    simpleLinearRegression = linearRegression(windspeed_training, power_training, windspeed_input, actualPower, showPlots)
    multipleLinearRegression = multipleLinReg(windspeed_training, wind_direction, power_training,
                                              windspeed_and_direction_input,
                                              actualPower,
                                              showPlots)
    if showPlots:
        plotResults3Inputs(actualPower,
                           simpleLinearRegression[0],
                           multipleLinearRegression[0],
                           "actual power",
                           "linear regression",
                           "multiple regression",
                           "Simple, multiple and actual wind power")

    print("RMSE simple linear regression: " + str(simpleLinearRegression[1]))
    print("RMSE multiple linear regression: " + str(multipleLinearRegression[1]))


runAllModelsTask2(True)



# ---------------- For task 3 ----------------
def createTimeserie(powers, windowsize):
    rows = len(powers) - windowsize
    x = np.zeros((rows, windowsize))
    y = np.zeros(rows)

    for i in range(rows):
        for j in range(windowsize):
            x[i][j] = powers[i + j]

        y[i] = powers[i + windowsize]

    return x, y


def timeserieLinReg(x, y, forecastInput, actualPower, windowsize, showPlots):
    model = LinearRegression()
    model.fit(x, y)
    predictedPower = model.predict(forecastInput)

    actualPowerComparison = actualPower[windowsize:]

    rmse = RMSE(actualPowerComparison, predictedPower)

    if showPlots:
        plotResults(actualPower[windowsize:], predictedPower, "timeseries linear regression")

    return predictedPower, rmse


def timeserieSupportedVectorRegression(x, y, forecastInput, actualPower, windowsize, showPlots):
    model = SVR()
    model.fit(x, y)

    predictedPower = model.predict(forecastInput)
    actualPowerComparison = actualPower[windowsize:]

    rmse = RMSE(actualPowerComparison, predictedPower)

    if showPlots:
        plotResults(actualPower[windowsize:], predictedPower, "timeseries SVR")

    return predictedPower, rmse


def timeserieArtificialNeuralNetworks(x, y, forecastInput, actualPower, windowsize, showPlots):
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

    predictedPower = model.predict(forecastInput)
    actualPowerComparison = actualPower[windowsize:].reshape((-1, 1))

    rmse = RMSE(actualPowerComparison, predictedPower)

    if showPlots:
        plotResults(actualPower[windowsize:], predictedPower, "timeserie ANN")

    return predictedPower, rmse


def timeserieRecurrantNeuralNetwork(x, y, forecastInput, actualPower, windowsize, showPlots):
    # reshaping input to (batch_size, timesteps, input_dim)
    x = x.reshape(-1, 1, windowsize)
    forecastInput3D = np.reshape(forecastInput, (len(forecastInput), 1, windowsize))

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
    predictedPower = model.predict(forecastInput3D)
    actualPowerComparison = actualPower[windowsize:].reshape((-1, 1))

    rmse = RMSE(actualPowerComparison, predictedPower)

    if showPlots:
        plotResults(actualPower[windowsize:], predictedPower, "timeseries RNN")

    return predictedPower, rmse

def runAllTimeseriesModelsWindowsize(actualPower, windowsize, showPlots):
    timeseries = createTimeserie(power_training, windowsize)
    x = timeseries[0]
    y = timeseries[1]
    forecastInput = createTimeserie(actualPower, windowsize)

    tsLinReg = timeserieLinReg(x, y, forecastInput[0], actualPower, windowsize, plotResults)
    tsSVR = timeserieSupportedVectorRegression(x, y, forecastInput[0], actualPower, windowsize, plotResults)
    tsANN = timeserieArtificialNeuralNetworks(x, y, forecastInput[0], actualPower, windowsize, plotResults)
    tsRNN = timeserieRecurrantNeuralNetwork(x, y, forecastInput[0], actualPower, windowsize, plotResults)

    if showPlots:
        plotResults3Inputs(actualPower, tsLinReg[0], tsSVR[0], "actual power", "linear regression", "supported vector regression",
                            "actual power, linear regression and SVR comparison for timeseries")

        plotResults3Inputs(actualPower, tsANN[0], tsRNN[0], "actual power", "artificial neural network", "recurrant neural network",
                            "actual power, ANN and RNN comparison for timeseries")

    print("RMSE timeseries LinReg:" + str(tsLinReg[1]) + ", windowsize = " + str(windowsize))
    print("RMSE timeseries SVR:" + str(tsSVR[1]) + ", windowsize = " + str(windowsize))
    print("RMSE timeseries ANN:" + str(tsANN[1]) + ", windowsize = " + str(windowsize))
    print("RMSE timeseries RNN:" + str(tsRNN[1]) + ", windowsize = " + str(windowsize))


# runAllTimeseriesModelsWindowsize(actualPower, 1, True)
