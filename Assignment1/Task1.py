from scipy.optimize import linprog
from matplotlib import pyplot as plt
import pandas as pd

# The appliances to be used in this task
appliances = ["Washing machine", "Electric vehicle", "Dishwasher"]
numAppliances = len(appliances)

# Right side values for optimization problem
rhs_eq = [1.94, 9.9, 1.44]
rhs_ineq_values = [1.5, 3.0, 1.8]
rhs_ineq = []

# Left side vectors for optimization problem
lhs_eq = []
lhs_ineq = []


def price(hour):
    """
    Function that creates the time-of-Use (ToU) pricing scheme defined in
    the assignment.
    :param hour:  list of the hours [0-23]
    :return:      list of prices in the different hours
    """
    return 1 if 17 <= hour <= 20 else 0.5


# Adding placeholders for vectors in equality constraints (total energy consumption)
for i in range(numAppliances):
    lhs_eq.append([0] * numAppliances * 24)


def add_operation_time_lhs(start, end, appliance):
    """
    Function adding vectors (1) to equality constraint. These vectors represent the
    operation time for the appliance (when the appliance can be used).
    :param start:       start hour for appliance
    :param end:         end hour for appliance
    :param appliance:   which appliance the operation time
    """
    appliance_index = appliances.index(appliance)

    interval = appliance_index * 24

    for i in range(start + interval, end + interval + 1):
        lhs_eq[appliance_index][i] = 1


# Defining the operation time for the appliances
add_operation_time_lhs(7, 23, "Washing machine")
add_operation_time_lhs(0, 7, "Electric vehicle")
add_operation_time_lhs(17, 23, "Electric vehicle")
add_operation_time_lhs(12, 18, "Dishwasher")


def add_max_power_use_hours_lhs():
    """
    Adding vectors for hourly max energy use for the appliances.
    """
    for i in range(numAppliances):
        for j in range(24):
            lst = [0] * 72
            lst[j + 24 * i] = 1
            lhs_ineq.append(lst)


add_max_power_use_hours_lhs()


def add_max_power_use_hours_rhs(rhs_ineq):
    """
    Adding the actual values for max hourly energy use for all the appliances.
    :param rhs_ineq:
    :return:
    """
    for value in rhs_ineq_values:
        rhs_ineq += [value] * 24


add_max_power_use_hours_rhs(rhs_ineq)

# Using linprog to solve optimization problem.
objective = [price(i) for i in range(24)] * numAppliances
bnd = [(0, float("inf")) for i in range(72)]
opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd)

# The new energy consumption for each appliance
y_values = [opt.x[i:i + 24] for i in range(0, len(opt.x), 24)]


def plot_bars(names, y_values):
    """
    Function to plot a graphical representation (bar graph) of the energy consumption
    for all the different appliances.
    :param names:      Name of the appliances included in the graph
    :param y_values:   energy consumption for each appliance
    """
    # combining energy consumptions and their appliance label
    consumptions = {name: y for (name, y) in zip(names, y_values)}

    # Details for plotting
    df = pd.DataFrame(consumptions)
    ax = df.plot(kind="bar", stacked=True, rot=0, figsize=(5, 4))
    ax.set_ylim(0, 1.2)
    plt.title("Energy consumption")
    plt.legend(loc="lower left", bbox_to_anchor=(0.8, 1.0))
    plt.show()


plot_bars(appliances, y_values)


def plot_bars_independent():
    """
    Function to plot a graphical representation (bar graph) of the energy consumption
    for all the different appliances, where each appliance has its own bar graph.
    """
    hours = list(range(24))
    fig, axs = plt.subplots(3)

    plt.setp(axs, xticks=hours)

    axs[0].bar(hours, y_values[0], color="blue")
    axs[0].set_title("Washing machine")

    axs[1].bar(hours, y_values[1], color="orange")
    axs[1].set_title("Electric vehicle")

    axs[2].bar(hours, y_values[2], color="green")
    axs[2].set_title("dishwasher")
    plt.show()
