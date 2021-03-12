import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd

# TODO find values for: hourly max usage , total usage
# TODO doesnt work for all hours, need fix


appliances_base = ["Dishwasher", "Laundry machine",
                   "Cloth dryer", "Electric vehicle"]

non_shiftable = {"Lighting": [1.5, 0, 10, 20],
                 "Heating": [8.0, 0, 4, 8],
                 "Refrigerator": [1.32, 0, 0, 23],
                 "Electric stove": [3.9, 0, 16, 19],
                 "TV": [0.22, 0, 18, 21],
                 "Computer": [0.3, 0, 16, 21]}

appliances_random = {"Coffee maker": [0.1, 0.1, 8, 12],
                     "Ceiling fan": [0.3, 0.3, 7, 20],
                     "Hair dryer": [0.2, 0.2, 7, 8],
                     "Toaster": [0.2, 0.2, 7, 8],
                     "Microwave": [0.3, 0.3, 12, 18],
                     # TODO needs to be consistant with use of 24/23
                     "Router": [0.2, 0, 0, 24],
                     "Cellphone charger": [0.2, 0.2, 18, 23],
                     "Cloth iron": [0.4, 0.4, 18, 21],
                     "Freezer": [1.9, 1.0, 0, 23]}
# name : rhs_eq, rhs_ineq, start, end

L = 6

rhs_eq = [1.44, 1.94, 2.5, 9.9]
numAppliances = 6
rhs_ineq_values = [1.8, 1.5, 2, 4]  # , L]
rhs_ineq = []
lhs_ineq = []


prices = [38.30, 37.84, 37.73, 37.73, 37.84, 39.11, 42.08, 49.90, 64.68, 56.15, 47.80, 43.42, 42.87,
          41.52, 41.94, 42.28, 42.73, 43.95, 44.17, 41.92, 38.67, 37.61, 37.31, 36.82]  # RTP schme with numbers from Nordpool

lhs_eq = []
for i in range(numAppliances):
    lhs_eq.append([0] * numAppliances * 24)


def price(hour):
    random.seed(10 + hour)
    return random.uniform(1, 2) if 17 <= hour <= 20 else random.uniform(0.1, 1)


def add_operation_time_lhs(start, end, appliance):
    appliance_index = appliances_base.index(appliance)

    interval = appliance_index * 24

    for i in range(start + interval, end + interval + 1):
        lhs_eq[appliance_index][i] = 1


def add_random_appliances(number):
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
    for values in non_shiftable.values():
        start_hour = values[2]
        end_hour = values[3]
        hourly_usage = values[0] / (end_hour - start_hour)
        for i in range(start_hour, end_hour):
            combined_consumptions_non_shiftable[i] += hourly_usage


add_times_non_shiftable()


def add_max_power_use_hours_lhs():
    for i in range(numAppliances):
        for j in range(24):
            lst = [0] * 24 * numAppliances
            lst[j + 24 * i] = 1
            lhs_ineq.append(lst)


add_max_power_use_hours_lhs()


def add_max_power_use_hours_rhs(rhs_ineq):
    print(len(rhs_ineq_values), "lengde")
    for value in rhs_ineq_values:
        rhs_ineq += [value] * 24
        print(value)


def add_max_peak_load_per_hour():
    for hour in range(24):
        lst = [0]*24*numAppliances
        for j in range(numAppliances):
            lst[j*24 + hour] = 1

        lhs_ineq.append(lst)


def add_1hour_max_peak_load():  # for testing
    hour = 23
    lst = [0]*24*numAppliances
    for j in range(numAppliances):
        lst[j*24 + hour] = 1

    lhs_ineq.append(lst)
    rhs_ineq.append(L)


add_random_appliances(2)
numAppliances = len(appliances_base)

add_max_power_use_hours_rhs(rhs_ineq)

print(rhs_ineq_values)


objective = prices * numAppliances

# add_max_peak_load_per_hour()
add_1hour_max_peak_load()

add_operation_time_lhs(7, 23, "Cloth dryer")
add_operation_time_lhs(7, 23, "Laundry machine")
add_operation_time_lhs(0, 7, "Electric vehicle")
add_operation_time_lhs(17, 23, "Electric vehicle")
add_operation_time_lhs(12, 18, "Dishwasher")

print(len(rhs_ineq))

bnd = [(0, float("inf")) for i in range(24 * numAppliances)]
opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq,
              A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd)


hours = list(range(24))


y_values = [opt.x[i:i + 24] for i in range(0, len(opt.x), 24)]

y_values.append(combined_consumptions_non_shiftable)
appliances_base.append("Non-shiftable appliances")


def plot_bars(names, y_values):
    consumptions = {name: y for (name, y) in zip(names, y_values)}

    df = pd.DataFrame(consumptions)

    ax = df.plot(kind="bar", stacked=True, rot=0, figsize=(10, 5))
    ax.set_ylim(0, 15)
    plt.title("Energy consumption")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


plot_bars(appliances_base, y_values)
