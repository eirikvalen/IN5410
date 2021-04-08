import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler



#TODO scale data?


def RMSE(x, y):
   return np.sqrt(np.mean((x-y)**2))

#Training data
df=pd.read_csv('TrainData.csv', sep=',')

windspeed = df["WS10"].to_numpy().reshape((-1,1))
power = df["POWER"].to_numpy()

u = df["U10"].to_numpy()
v = df["V10"].to_numpy()

direction = np.mod(180+np.rad2deg(np.arctan2(v,u)), 360)

#Input data
df = pd.read_csv("WeatherForecastInput.csv", sep=',')
ws = df["WS10"].to_numpy().reshape((-1,1))

u_input = df["U10"].to_numpy()
v_input = df["V10"].to_numpy()

direction_input = np.mod(180+np.rad2deg(np.arctan2(v_input,u_input)), 360).reshape((-1,1))

x_input = np.concatenate((ws,direction_input),axis=1)

#solution - actual power
df = pd.read_csv("Solution.csv", sep = ',')
time = df["TIMESTAMP"].to_numpy()
actualPower = df["POWER"].to_numpy()




def linearRegression(windspeed, power, forecastInput, actualPower):
    model = LinearRegression()
    model.fit(windspeed, power)
 

    #using the model to predict new data

    predictedPower = model.predict(forecastInput)
   
    print("RMSE linear regression ", RMSE(actualPower, predictedPower))
    #print("RMSE2  ", np.sqrt(metrics.mean_squared_error(actualPower, predictedPower)))

    plt.scatter(forecastInput, actualPower, s = 1.0)
    plt.plot(forecastInput, predictedPower, color = "r")
    plt.show()



def knn(windspeed, power, forecastInput, actualPower):
   
    for k in range(190,191):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(windspeed, power)

        predictedPower = model.predict(forecastInput)

        plt.scatter(forecastInput, actualPower, c = "r", s = 1.5)
        plt.scatter(forecastInput, predictedPower, c = "b", s = 1.5)
        plt.show()

        print("RMSE knn k = %d : %f" %  (k, RMSE(predictedPower, actualPower)))

def svr(windspeed, power, forecastInput, actualPower):
    model = SVR()
    model.fit(windspeed, power)

    predictedPower = model.predict(forecastInput)
    plt.scatter(forecastInput, actualPower, c = "r", s = 1.0)
    
    plt.scatter(forecastInput, predictedPower, c = "b", s = 1.0)
    plt.show()

    print("RMSE svr", RMSE(actualPower, predictedPower))



def neuralNetwork(windspeed, power, forecastInput, actualPower):
    scaler = MinMaxScaler()
    windspeed = scaler.fit_transform(windspeed) #scaling the data between 0 and 1

    nn = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000)

    nn.fit(windspeed, power)

    forecastInput = scaler.fit_transform(forecastInput)
    predictedPower = nn.predict(forecastInput)

    plt.scatter(forecastInput, actualPower, c = "r", s = 1.0)
    plt.plot(forecastInput, predictedPower, c = "b")
    plt.show()

    print("RMSE neural network", RMSE(actualPower, predictedPower))
    

#linearRegression(windspeed, power, ws, actualPower)
#knn(windspeed, power, ws, actualPower)
#svr(windspeed, power, ws, actualPower)
#neuralNetwork(windspeed, power, ws, actualPower)


def multiple_lin_reg(windspeed, direction, power, forecastInput, actualPower):
    direction = direction.reshape((-1,1))

    x = np.concatenate((windspeed, direction), axis=1)
    
    model = LinearRegression()
    model.fit(x,power)

    predictedPower = model.predict(forecastInput)

    print("RMSE multiple linear regression", RMSE(actualPower, predictedPower))
    

multiple_lin_reg(windspeed, direction, power, x_input, actualPower)


def time_series_lin_reg(powers, windowsize, time, actualPower):

    #powers = np.array(range(10))


    rows = len(powers) - windowsize
    x = np.zeros((rows, windowsize))
    y = np.zeros(rows)
    
    for i in range(rows):
        for j in range(windowsize):
            x[i][j] = powers[i+j] 

        y[i] = powers[i+windowsize]

    model = LinearRegression()
    model.fit(x,y)

 

    predictedPower = model.predict(actualPower[:-1].reshape(-1,1))
    
    powerSolution = actualPower[1:]

    print("RMSE time series linear regression ", RMSE(powerSolution, predictedPower))

    time = time[1:]
    plt.plot(time, powerSolution)
    plt.plot(time, predictedPower)
    plt.show()






    #predicted = model.predict(powers[-1].reshape(1,-1))
    #nextPredicted = model.predict(predicted.reshape(1,-1))







    

time_series_lin_reg(power, 1, time, actualPower)




