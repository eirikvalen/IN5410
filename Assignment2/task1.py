import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dense
from keras.models import Model


# ---------------- Reading from file ----------------
# Training data
df = pd.read_csv('TrainData.csv', sep=',')
windspeed = df["WS10"].to_numpy().reshape((-1, 1))
power = df["POWER"].to_numpy()
u = df["U10"].to_numpy()
v = df["V10"].to_numpy()
direction = np.mod(180 + np.rad2deg(np.arctan2(v, u)), 360)

# Input data
df = pd.read_csv("WeatherForecastInput.csv", sep=',')
ws = df["WS10"].to_numpy().reshape((-1, 1))
u_input = df["U10"].to_numpy()
v_input = df["V10"].to_numpy()
direction_input = np.mod(180 + np.rad2deg(np.arctan2(v_input, u_input)), 360).reshape((-1, 1))
x_input = np.concatenate((ws, direction_input), axis=1)

# Solution - actual power
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

def plotResults3Inputs(actualPower, predictedpower1, predictedPower):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xticklabels([])
    plt.plot(actualPower, color='gray', label="actual power")
    plt.plot(predictedpower1, color='lightseagreen', label="linear regression")
    plt.plot(predictedPower, color='darkorange', label="multiple linear regression")
    plt.legend()
    # plt.legend( bbox_to_anchor=(1.05, 1), loc='upper center')

    plt.title("Simple, multiple and actual wind power")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()

def scatterPlot(forecastInput, actualPower, predictedPower, model):
    plt.scatter(forecastInput, actualPower, c="r", s=1.5)
    plt.scatter(forecastInput, predictedPower, c="b", s=1.5)
    plt.title("True vs. predicted wind power, " + model)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()



# ---------------- For task 1 ----------------
def linearRegression(windspeed, power, forecastInput, actualPower):
    model = LinearRegression()
    model.fit(windspeed, power)

    # Predicting power with new data
    predictedPower = model.predict(forecastInput)

    print("RMSE linear regression ", RMSE(actualPower, predictedPower))

    # plotResults(actualPower, predictedPower, "linear regression")

    return predictedPower


def knn(windspeed, power, forecastInput, actualPower):
    for k in [20]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(windspeed, power)

        predictedPower = model.predict(forecastInput)

        print("RMSE knn k = %d : %f" % (k, RMSE(predictedPower, actualPower)))

        plotResults(actualPower, predictedPower, "knn")


def svr(windspeed, power, forecastInput, actualPower):
    model = SVR()
    model.fit(windspeed, power)

    # Predicting power with new data
    predictedPower = model.predict(forecastInput)

    print("RMSE svr", RMSE(actualPower, predictedPower))

    plotResults(actualPower, predictedPower, "svr")


def neuralNetwork(windspeed, power, forecastInput, actualPower):
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

    model.fit(windspeed, power, epochs=500, batch_size=100, verbose=1, validation_split=0.2)  # starts training

    # Predicting power with new data
    predictedPower = model.predict(forecastInput)

    actualPower = actualPower.reshape((-1, 1))

    print("RMSE Neural networks", RMSE(actualPower, predictedPower))

    scatterPlot(forecastInput, actualPower, predictedPower, "Neural networks")
    plotResults(actualPower, predictedPower, "neural network")


# linearRegression(windspeed, power, ws, actualPower)
# knn(windspeed, power, ws, actualPower)
# svr(windspeed, power, ws, actualPower)
# neuralNetwork(windspeed, power, ws, actualPower)




# ---------------- For task 2 ----------------
def multipleLinReg(windspeed, direction, power, forecastInput, actualPower):
    direction = direction.reshape((-1, 1))

    x = np.concatenate((windspeed, direction), axis=1)

    model = LinearRegression()
    model.fit(x, power)

    predictedPower = model.predict(forecastInput)

    print("RMSE multiple linear regression", RMSE(actualPower, predictedPower))

    # plotResults(actualPower, predictedPower, "multiple linear regression")

    return predictedPower



# plotResults3Inputs(actualPower,
#              linearRegression(windspeed, power, ws, actualPower),
#              multipleLinReg(windspeed, direction, power, x_input, actualPower))




# ---------------- For task 3 ----------------
def timeseriesLinReg(powers, windowsize, time, actualPower):
    # powers = np.array(range(10))

    rows = len(powers) - windowsize
    x = np.zeros((rows, windowsize))
    y = np.zeros(rows)

    for i in range(rows):
        for j in range(windowsize):
            x[i][j] = powers[i + j]

        y[i] = powers[i + windowsize]

    model = LinearRegression()
    model.fit(x, y)

    predictedPower = model.predict(actualPower[:-1].reshape(-1, 1))

    powerSolution = actualPower[1:]

    print("RMSE time series linear regression ", RMSE(powerSolution, predictedPower))

    time = time[1:]
    plt.plot(time, powerSolution)
    plt.plot(time, predictedPower)
    plt.show()

    # predicted = model.predict(powers[-1].reshape(1,-1))
    # nextPredicted = model.predict(predicted.reshape(1,-1))

# time_series_lin_reg(power, 1, time, actualPower)