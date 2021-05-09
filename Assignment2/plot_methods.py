import matplotlib.pyplot as plt
import numpy as np


def RMSE(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def plot_results(actualPower, predictedPower, model):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xticklabels([])
    plt.plot(actualPower, color='lightseagreen', label="actual power")
    plt.plot(predictedPower, color='darkorange', label="predicted power")
    plt.legend()

    plt.title("True vs. predicted wind power, " + model)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.show()


def plot_results_3_inputs(actualPower, predictedpower1, predictedPower, label1, label2, label3, title):
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


def scatter_plot(forecastInput, actualPower, predictedPower, model):
    plt.scatter(forecastInput, actualPower, c="r", s=1.5, label="Actual power")
    plt.scatter(forecastInput, predictedPower, c="b", s=1.5, label="Predicted power")
    plt.title("True vs. predicted wind power, " + model)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()
    plt.show()