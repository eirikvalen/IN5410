import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Assignment2.read_from_file import read_training_data, read_input_data, read_solution
from Assignment2.plot_methods import RMSE, plot_results, plot_results_3_inputs
from Assignment2.task1 import linear_regression

training_data = read_training_data()
windspeed_training = training_data[0]
power_training = training_data[1]
wind_direction_training = training_data[2]

input_data = read_input_data()
windspeed_input = input_data[0]
windspeed_and_direction_input = input_data[2]

solution = read_solution()
actual_power = solution[0]
timestamp = solution[1]


def multiple_linear_regression(windspeed, direction, power, forecast_input, actual_power, show_plots):
    direction = direction.reshape((-1, 1))

    x = np.concatenate((windspeed, direction), axis=1)

    model = LinearRegression()
    model.fit(x, power)

    predictedPower = model.predict(forecast_input)

    rmse = RMSE(actual_power, predictedPower)

    if show_plots:
        plot_results(actual_power, predictedPower, "multiple linear regression")

    return predictedPower, rmse


def run_all_models_task2(show_plots, write_to_csv):
    simple_linear_regression = linear_regression(windspeed_training, power_training, windspeed_input,
                                                 actual_power,
                                                 show_plots)
    multiple_lin_reg = multiple_linear_regression(windspeed_training, wind_direction_training, power_training,
                                                  windspeed_and_direction_input,
                                                  actual_power,
                                                  show_plots)
    if show_plots:
        plot_results_3_inputs(actual_power,
                              simple_linear_regression[0],
                              multiple_lin_reg[0],
                              "actual power",
                              "linear regression",
                              "multiple regression",
                              "Simple, multiple and actual wind power")

    if write_to_csv:
        header = "TIMESTAMP, POWER"

        pd.DataFrame({"TIMESTAMP": timestamp, "POWER": multiple_lin_reg[0].flat}).to_csv(
            "../ForecastResults/ForecastTemplate2.csv",
            index=None, header=header)

    print("RMSE simple linear regression: " + str(simple_linear_regression[1]))
    print("RMSE multiple linear regression: " + str(multiple_lin_reg[1]))


run_all_models_task2(False, True)
