import pandas as pd
import numpy as np

def read_training_data():
    df = pd.read_csv('TrainData.csv', sep=',')
    windspeed_training = df["WS10"].to_numpy().reshape((-1, 1))
    power_training = df["POWER"].to_numpy()
    u = df["U10"].to_numpy()
    v = df["V10"].to_numpy()
    wind_direction = np.mod(180 + np.rad2deg(np.arctan2(v, u)), 360)

    return windspeed_training, power_training, wind_direction

def read_input_data():
    df = pd.read_csv("WeatherForecastInput.csv", sep=',')
    windspeed_input = df["WS10"].to_numpy().reshape((-1, 1))
    u_input = df["U10"].to_numpy()
    v_input = df["V10"].to_numpy()
    direction_input = np.mod(180 + np.rad2deg(np.arctan2(v_input, u_input)), 360).reshape((-1, 1))
    windspeed_and_direction_input = np.concatenate((windspeed_input, direction_input), axis=1)

    return windspeed_input, direction_input, windspeed_and_direction_input

def read_solution():
    df = pd.read_csv("Solution.csv", sep=',')
    actual_power = df["POWER"].to_numpy()

    return actual_power

