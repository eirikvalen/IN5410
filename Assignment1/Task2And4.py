import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd


# Shiftable appliances that always are included in energy consumption
appliances_base = ["Dishwasher", "Laundry machine",
                   "Cloth dryer", "Electric vehicle"]

# Non-shiftable appliances that always are included in energy consumption.
# How to interpret the values:
#    name :[total consumption, hourly max consumption, start hour, end hour]
#    name :[ rhs_eq          , rhs_ineq,             ,start     , end      ]
non_shiftable = {"Lighting": [1.5, 0, 10, 20],
                 "Heating": [8.0, 0, 4, 8],
                 "Refrigerator": [1.32, 0, 0, 23],
                 "Electric stove": [3.9, 0, 16, 19],
                 "TV": [0.22, 0, 18, 21],
                 "Computer": [0.3, 0, 16, 21]}

# Random shiftable appliances where some will be included
appliances_random = {"Coffee maker": [0.76, 1.52, 8, 12],
                     "Ceiling fan": [0.97, 0.075, 7, 20],
                     "Hair dryer": [0.9, 1.5, 7, 8],
                     "Toaster": [0.6, 1.2, 7, 8],
                     "Microwave": [0.96, 1.2, 12, 18],
                     "Router": [0.14, 0.006, 0, 23],
                     "Cellphone charger": [0.03, 0.005, 18, 23],
                     "Cloth iron": [0.55, 1.1, 18, 21],
                     "Freezer": [0.84, 0.035, 0, 23]}

# Max peak load constraint
L = 5

numAppliances = 6

# Right side values for optimization problem
rhs_eq = [1.44, 1.94, 2.5, 9.9]
rhs_ineq_values = [1.8, 1.5, 2, 4]
rhs_ineq = []

# Left side vectors for optimization problem
lhs_ineq = []
lhs_eq = []

# Prices in nok/mWh retrieved from nordpool 14.03.21. Source:
# https://www.nordpoolgroup.com/Market-data1/Dayahead/Area-Prices/NO/Hourly/?view=table
prices = [389.86, 382.99,375.11, 371.48, 387.94, 417.42, 457.71, 530.92, 634.92, 504.76,
          479.32, 453.37, 447.61, 435.90, 411.67, 408.94, 430.25, 450.24, 453.87, 455.99,
          452.56, 440.24, 431.05, 412.98]

# Adding placeholders for vectors in equality constraints for base appliances
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
    appliance_index = appliances_base.index(appliance)

    interval = appliance_index * 24

    for i in range(start + interval, end + interval + 1):
        lhs_eq[appliance_index][i] = 1


def add_random_appliances(number):
    """
    Adding a number of random appliances. Adding the chosen appliances operation
    time and hourly max usage.
    :param number:  the number of random appliances to add
    """
    for i in range(number):
        random.seed(10 + i)
        item = random.choice(list(appliances_random.keys()))
        item_rhs = appliances_random[item][0]
        appliances_base.append(item)
        rhs_eq.append(item_rhs)

        # Adding operation times
        add_operation_time_lhs(
            appliances_random[item][2], appliances_random[item][3], item)

        # Adding hourly max
        rhs_ineq_values.append(appliances_random[item][1])


combined_consumptions_non_shiftable = [0] * 24

def add_times_non_shiftable():
    """
    Adding the energy consumption for each hour for all the non-shiftable applaiances.
    Since their operation time and therefore hourly consumption can't be changed, only
    one value will represent the total energy consumption.
    """
    for values in non_shiftable.values():
        start_hour = values[2]
        end_hour = values[3]
        hourly_usage = values[0] / (end_hour - start_hour)
        for i in range(start_hour, end_hour):
            combined_consumptions_non_shiftable[i] += hourly_usage


add_times_non_shiftable()


def add_max_power_use_hours_lhs():
    """
    Adding vectors for hourly max energy use for all the appliances to be used (base + random).
    """
    for i in range(numAppliances):
        for j in range(24):
            lst = [0] * 24 * numAppliances
            lst[j + 24 * i] = 1
            lhs_ineq.append(lst)

add_max_power_use_hours_lhs()



def add_max_power_use_hours_rhs(rhs_ineq):
    """
    Adding the actual values for max hourly energy use for all the appliances that are
    used (base + random).
    :param rhs_ineq:
    :return:
    """
    for value in rhs_ineq_values:
        rhs_ineq += [value] * 24


def add_max_peak_load():
    """
    Adding the constraint on max peak load for every hour. Both the vectors and the
    actual value L is added to the inequality left and right side. The energy consumption
    for the non-shiftable appliances are taken into considerations by subtracting their energy
    consumption from the value of L.
    :return:
    """
    for hour in range(24):  
        lst = [0]*24*numAppliances
        for j in range(numAppliances):
            lst[j*24 + hour] = 1

        lhs_ineq.append(lst)
        rhs_ineq.append(L - combined_consumptions_non_shiftable[hour])


# Adding random appliances, energy consumption for all appliances, peak load variable L and
# creating operation times for the base appliances.
add_random_appliances(2)
add_max_power_use_hours_rhs(rhs_ineq)
add_operation_time_lhs(7, 23, "Cloth dryer")
add_operation_time_lhs(7, 23, "Laundry machine")
add_operation_time_lhs(0, 7, "Electric vehicle")
add_operation_time_lhs(17, 23, "Electric vehicle")
add_operation_time_lhs(12, 18, "Dishwasher")

add_max_peak_load()

# Using linprog to calculate optimization problem
objective = prices * numAppliances
bnd = [(0, float("inf")) for i in range(24 * numAppliances)]
opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq,
              A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd)


def calculate_total_price(df):
    """
    Calculating the total energy consumption for the household and the price for this consumption.
    :param df:      dataframe of results from linprog
    """
    sums = df.sum(axis=1)
    energy_consumption = 0
    total_sum = 0
    for i in range(len(sums)):
        total_sum+= sums[i] * (prices[i] / 1000)  #price in kWH
        energy_consumption+=sums[i]

    print("L: " + str(L) + " | Energy consumption: " + str(energy_consumption) + " | Total price: " + str(total_sum))



def plot_bars(names, task4):
    """
    Function to plot a graphical representation (bar graph) of the energy consumption
    for the different appliances.
    :param names:      Name of the appliances included in the graph
    """
    # The new energy consumption for each appliance
    y_values = [opt.x[i:i + 24] for i in range(0, len(opt.x), 24)]
    y_values.append(combined_consumptions_non_shiftable)
    appliances_base.append("Non-shiftable appliances")

    # Combining energy consumptions and their appliance label
    consumptions = {name: y for (name, y) in zip(names, y_values)}

    df = pd.DataFrame(consumptions)
    calculate_total_price(df)

    if (task4):
        plt.axhline(y=L, color = 'r', linestyle="-", label="Max peak load (L)")

    # Plotting details
    ax = df.plot(kind="bar", stacked=True, rot=0, figsize=(10, 5))
    ax.set_ylim(0, 10)
    plt.yticks(range(0,10,1))
    plt.xlabel("Hour")
    plt.ylabel("Energy consumption (kWh)")
    plt.title("Energy consumption")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


plot_bars(appliances_base, True)

def plot_price_curve():
    """
    Function to genereate a graphical representation of the price curve used.
    """
    hours = list(range(24))

    plt.plot(hours, prices)
    plt.xticks(range(0,24,1))
    plt.xlabel("Hour")
    plt.ylabel("Price (NOK / MWh)")
    plt.show()

plot_price_curve()
