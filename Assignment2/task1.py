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


def kNearestNeighbor(windspeed, power, forecastInput, actualPower):
    for k in [20]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(windspeed, power)

        predictedPower = model.predict(forecastInput)

        print("RMSE KNN k = %d : %f" % (k, RMSE(predictedPower, actualPower)))

        plotResults(actualPower, predictedPower, "KNN")


def supportedVectorRegression(windspeed, power, forecastInput, actualPower):
    model = SVR()
    model.fit(windspeed, power)

    # Predicting power with new data
    predictedPower = model.predict(forecastInput)

    print("RMSE SVR", RMSE(actualPower, predictedPower))

    plotResults(actualPower, predictedPower, "SVR")


def artificialNeuralNetworks(windspeed, power, forecastInput, actualPower):
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

    # scatterPlot(forecastInput, actualPower, predictedPower, "Neural networks")
    # plotResults(actualPower, predictedPower, "neural network")


# linearRegression(windspeed, power, ws, actualPower)
# kNearestNeighbor(windspeed, power, ws, actualPower)
# supportedVectorRegression(windspeed, power, ws, actualPower)
# artificialNeuralNetworks(windspeed, power, ws, actualPower)




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

def createTimeserie(powers, windowsize):
    rows = len(powers) - windowsize
    x = np.zeros((rows, windowsize))
    y = np.zeros(rows)

    for i in range(rows):
        for j in range(windowsize):
            x[i][j] = powers[i + j]

        y[i] = powers[i + windowsize]

    return x, y


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return X, y


def timeserieLinReg(x, y, actualPower, forecastInput):
    model = LinearRegression()
    model.fit(x, y)

    predictedPower = model.predict(actualPower[:-1].reshape(-1, 1))
    powerSolution = actualPower[1:]

    print("RMSE timeseries linear regression ", RMSE(powerSolution, predictedPower))

    plotResults(actualPower, predictedPower, "timeseries linear regression")

    plt.scatter(actualPower[:-1], powerSolution)
    plt.plot(actualPower[:-1], predictedPower)
    plt.show()

    # predicted = model.predict(powers[-1].reshape(1,-1))
    # nextPredicted = model.predict(predicted.reshape(1,-1))


def timeserieSupportedVectorRegression(x, y, actualPower):
    model = SVR()
    model.fit(x, y)
    predictedPower = model.predict(actualPower[:-1].reshape(-1, 1))

    powerSolution = actualPower[1:]

    print("RMSE timeserie SVR", RMSE(powerSolution, predictedPower))

    plotResults(actualPower, predictedPower, "timeserie SVR")


def timeserieArtificialNeuralNetworks(x, y, actualPower):
    inputs = Input(shape=(1,))

    # Layers and number of neurons in hidden layers
    layer = Dense(32, activation='relu')(inputs)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(100, activation='relu')(layer)
    predictions = Dense(1)(layer)

    # The model that includes the input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mse'])

    model.fit(x, y, epochs=500, batch_size=100, verbose=1, validation_split=0.2)  # starts training

    # Predicting power with new data
    predictedPower = model.predict(actualPower[:-1].reshape(-1, 1))

    actualPower = actualPower.reshape((-1, 1))

    print("RMSE Neural networks", RMSE(actualPower[0:len(actualPower)-1], predictedPower))

    plotResults(actualPower[0:len(actualPower)-1], predictedPower, "timeserie ANN")

def timeserieRecurrantNeuralNetwork(x, y, actualPower):
    # reshaping x input
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    actualPower = np.reshape(actualPower, (len(actualPower), 1, 1))

    # using LSTM layer, used to accept a sequence of input data and make prediction
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 1), activation='relu', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # model.add(LSTM(128, input_shape=(1,1), activation='relu', return_sequences=True)) #input_shape=(n_steps,1)
    # model.add(Dropout(0.2))
    # model.add(LSTM(128, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='relu'))

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mse'],
    )
    model.summary()

    model.fit(x, y, epochs=50, batch_size=100, verbose=1)
    predictedPower = model.predict(actualPower)
    print("RMSE RNN", RMSE(actualPower[0:len(actualPower) - 1], predictedPower))




timeserie = createTimeserie(power, 1)
x = timeserie[0]
y = timeserie[1]




# print(x.shape, "x shape")
# timeserieLinReg(x, y, actualPower, ws)
# timeserieSupportedVectorRegression(x, y, actualPower)
# timeserieArtificialNeuralNetworks(x, y, actualPower)
timeserieRecurrantNeuralNetwork(x, y, actualPower)


# a = [1,2,3,4,5,6,7,8,9] [[1,2], [2,3], ]
# res = split_sequence(a, 3)
# print(res[0])
# print(res[1])