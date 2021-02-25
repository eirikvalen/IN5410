import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd


# TODO non-SA restrictions: finnes noen med satt tid og noen med "5 timer"??
# TODO price is very random, should the function be more "realistic"??
# TODO How to find the daily usage for random applaiances??

appliances_base = ["Lighting", "Heating", "Refrigerator", "Electric stove", "TV", "Computer",
                   "Dishwasher", "Laundry machine", "Cloth dryer", "Electric vehicle"]
appliances_random = ["Coffee maker", "Ceiling fan", "Hair dryer", "Toaster", "Microwave",
                     "Router", "Cellphone charger", "Cloth iron", "Freezer"]
rhs_eq = [1.5, 8.0, 1.32, 3.9, 0.22, 0.3, 1.44, 1.94, 2.5, 9.9]
rhs_eq_random = [0.1, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.4, 0.2, 1.9]
numAppliances = len(appliances_base)
print(numAppliances)
rhs_ineq = []
lhs_ineq = []

def price(hour):
    random.seed(10+hour)
    return random.uniform(1, 2) if 17 <= hour <= 20 else random.uniform(0.1, 1)


def add_random_appliances(number):
    for i in range(number):
        random.seed(10 + i)
        item = random.choice(appliances_random)
        item_index = appliances_random.index(item)
        appliances_base.append(item)
        rhs_eq.append(rhs_eq_random[item_index])

add_random_appliances(2)
numAppliances = len(appliances_base)
objective = [price(i) for i in range(24)] * numAppliances

lhs_eq = []
for i in range(numAppliances):
    lhs_eq.append([0] * numAppliances*24)

def add_operation_time_lhs(start, end, appliance):
    appliance_index = appliances_base.index(appliance)

    interval = appliance_index * 24

    for i in range(start+interval, end+interval+1):
        lhs_eq[appliance_index][i] = 1


add_operation_time_lhs(10, 20, "Lighting") # TODO denne må være uten for minimiaztion greia
add_operation_time_lhs(0,23, "Heating")
add_operation_time_lhs(0,23, "Refrigerator")
add_operation_time_lhs(16,18, "Electric stove")
add_operation_time_lhs(18,20, "TV")
add_operation_time_lhs(0,23, "Computer")
add_operation_time_lhs(7,23, "Cloth dryer")
add_operation_time_lhs(7,23,"Laundry machine")
add_operation_time_lhs(0,7,"Electric vehicle")
add_operation_time_lhs(17,23,"Electric vehicle")
add_operation_time_lhs(12,18,"Dishwasher")

#print(lhs_eq)

#print(len(rhs_eq))
print(len(lhs_eq[0]))
print(len(objective))

bnd = [(0, float("inf")) for i in range(24*numAppliances)]
opt = linprog(c=objective, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd)
hours = list(range(24))

print(opt.x)

y_values = [opt.x[i:i + 24] for i in range(0, len(opt.x), 24)]



def plot_bars(names, y_values):
    consumptions = {name: y for (name, y) in zip(names, y_values)}

    df = pd.DataFrame(consumptions)

    ax = df.plot(kind="bar", stacked=True, rot=0, figsize=(5, 4))
    ax.set_ylim(0, 1.2)
    plt.title("Energy consumption")
    plt.legend(loc="lower left", bbox_to_anchor=(0.8, 1.0))
    plt.show()


plot_bars(appliances_base, y_values)