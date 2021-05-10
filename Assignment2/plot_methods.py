import matplotlib.pyplot as plt
import numpy as np


def RMSE(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def plot_results(actual_power, predicted_power, model):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xticklabels([])
    plt.plot(actual_power, color='lightseagreen', label="actual power")
    plt.plot(predicted_power, color='darkorange', label="predicted power")
    plt.legend()

    plt.title("True vs. predicted wind power, " + model)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()


def plot_results_3_inputs(actual_power, predicted_power1, predicted_power, label1, label2, label3, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xticklabels([])
    plt.plot(actual_power, color='gray', label=label1)
    plt.plot(predicted_power1, color='lightseagreen', label=label2)
    plt.plot(predicted_power, color='darkorange', label=label3)
    plt.legend()

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()


def scatter_plot(forecast_input, actual_power, predicted_power, model):
    plt.scatter(forecast_input, actual_power, c="r", s=1.5, label="Actual power")
    plt.scatter(forecast_input, predicted_power, c="b", s=1.5, label="Predicted power")
    plt.title("True vs. predicted wind power, " + model)
    plt.xlabel("Windspeed")
    plt.ylabel("Power")
    plt.legend()
    plt.show()