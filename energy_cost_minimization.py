from scipy.optimize import linprog
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#time-of-Use (ToU) pricing scheme
def price(hour):
    return 1 if 17 <= hour <= 20 else 0.5

#washing machine: xi (i = 0,1..24)
#EV             : yi (i = 0,1..24)
#dishwasher     : zi (i = 0,1..24)

#cost: p(x0) + p(x1) + ... + p(x24) + p(y0) + p(y1) + ... + p(y24) + p(z0) + p(z1) + ... + p(z24)


#constraint1: x1 + x2 + .. + x24 = 1.94
#constraint2: y1 + y2 + .. + y24 = 9.9
#constraint3: z1 + z2 + .. + z24  = 1.44

#constraint4: y1 < 0.9   y2 < 0.9 ..... y24 < 0.9

numAppliances = 3

objective = [price(i) for i in range(24)] * numAppliances


constraint1 = [1 if  7 <= i < 24 else 0 for i in range(72)] # Washing machine operation time constraint 07-24
constraint2 = [1 if (24 <= i < 31 or 41 <= i < 48)  else 0 for i in range(72)] # Electric vehicle operation time constraint 17-07
constraint3 = [1 if 60 <= i <= 66 else 0 for i in range(72)] # dishwasher operation time constraint 12-18

lhs_ineq = []

#Electric vehicle inequality constraints
for i in range(24):
    lst = [0] * 72
    lst[i+24] = 1
    lhs_ineq.append(lst)

#dishwasher inequality constraints
for i in range(24):
    lst = [0] * 72
    lst[i+48] = 1
    lhs_ineq.append(lst)


rhs_ineq = [0.9] * 24 + [0.25] * 24

lhs_eq = [constraint1, constraint2, constraint3]
rhs_eq = [1.94, 9.9, 1.44]


bnd = [(0, float("inf")) for i in range(72)]

opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd)

hours = list(range(24))


y_values = [opt.x[i:i + 24] for i in range(0, len(opt.x), 24)] #

names = ["Washing machine", "Electric vehicles", "Dishwasher"]


def plot_bars(names, y_values):
    #consumptions = {}
    #for name,y in zip(names, y_values):
    #    consumptions[name] = y

    consumptions = {name : y for (name,y) in zip(names, y_values)}

    df = pd.DataFrame(consumptions)

    ax = df.plot(kind="bar",stacked=True, rot=0,figsize=(5,4))
    ax.set_ylim(0,1.2)
    plt.title("Energy consumption")
    plt.legend(loc="lower left",bbox_to_anchor=(0.8,1.0))
    plt.show()
    
plot_bars(names,y_values)




fig, axs = plt.subplots(3)

plt.setp(axs, xticks=hours)

axs[0].bar(hours, y_values[0], color = "blue")
axs[0].set_title("Washing machine")

axs[1].bar(hours, y_values[1], color = "orange")
axs[1].set_title("Electric vehicle")

axs[2].bar(hours, y_values[2], color="green")
axs[2].set_title("dishwasher")
plt.show()
