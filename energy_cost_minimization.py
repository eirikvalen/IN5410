from scipy.optimize import linprog
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#time-of-Use (ToU) pricing scheme
def price(hour):
    return 1 if 17 <= hour <= 20 else 0.5

appliances = ["Washing machine", "Electric vehicle", "Dishwasher"]
numAppliances = len(appliances)

rhs_ineq_values = [1.5, 3.0, 1.8]
rhs_ineq = []
lhs_ineq = []

objective = [price(i) for i in range(24)] * numAppliances

rhs_eq = [1.94, 9.9, 1.44]
lhs_eq = []
for i in range(numAppliances):
    lhs_eq.append([0] * numAppliances*24)


def add_operation_time_lhs(start, end, appliance):
    appliance_index = appliances.index(appliance)

    interval = appliance_index * 24

    for i in range(start+interval, end+interval+1):
        lhs_eq[appliance_index][i] = 1


add_operation_time_lhs(7,23,"Washing machine")
add_operation_time_lhs(0,7,"Electric vehicle")
add_operation_time_lhs(17,23,"Electric vehicle")
add_operation_time_lhs(12,18,"Dishwasher")

def add_max_power_use_hours_lhs():
    for i in range(numAppliances):
        for j in range(24):
            lst = [0] * 72
            lst[j+24*i] = 1
            lhs_ineq.append(lst)

add_max_power_use_hours_lhs()
    

def add_max_power_use_hours_rhs(rhs_ineq):
    for value in rhs_ineq_values:
        rhs_ineq += [value]*24
    

add_max_power_use_hours_rhs(rhs_ineq)


bnd = [(0, float("inf")) for i in range(72)]
opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd)
hours = list(range(24))

y_values = [opt.x[i:i + 24] for i in range(0, len(opt.x), 24)]


def plot_bars(names, y_values):
    consumptions = {name : y for (name,y) in zip(names, y_values)}

    df = pd.DataFrame(consumptions)

    ax = df.plot(kind="bar",stacked=True, rot=0,figsize=(5,4))
    ax.set_ylim(0,1.2)
    plt.title("Energy consumption")
    plt.legend(loc="lower left",bbox_to_anchor=(0.8,1.0))
    plt.show()
    
plot_bars(appliances, y_values)


def plot_bars_independent():

    fig, axs = plt.subplots(3)

    plt.setp(axs, xticks=hours)

    axs[0].bar(hours, y_values[0], color = "blue")
    axs[0].set_title("Washing machine")

    axs[1].bar(hours, y_values[1], color = "orange")
    axs[1].set_title("Electric vehicle")

    axs[2].bar(hours, y_values[2], color="green")
    axs[2].set_title("dishwasher")
    plt.show()
