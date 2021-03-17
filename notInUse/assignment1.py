
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import numpy as np

def price(hour):
    if 17 <= hour <= 20:
        return 1
    else:
        return 0.5


#z = 0.5*x1 + + 0.5*x2 .. 

#washing machine: xi (i = 0,1..24)
#EV             : yi (i = 0,1..24)
#dishwasher     : zi (i = 0,1..24)


#cost p(x0) + p(x1) + ... + p(x24) + p(y0) + p(y1) + ... + p(y24) + p(z0) + p(z1) + ... + p(z24)


#constraint1: x1 + x2 + .. + x24 = 1.94
#constraint2: y1 + y2 + .. + y24 = 9.9
#consttraint3: z1 + z2 +.. + z24 = 1.44





objective =  [1.0 if 17 <= i <= 20 else 0.5 for i in range(24)] * 3


numAppliances = 3

objective2 = [price(i) for i in range(24)] * numAppliances

for i in range(len(objective)):
    if(objective[i] != objective2[i]):
        print("not equal")



constraint1 = [1.0 if 0 <= i < 24 else 0 for i in range(72)] # Washing machine operation time constraint 
constraint2 = [1 if 24 <= i < 48 else 0 for i in range(72)]
constraint3 = [1.0 if 60 <= i <= 66 else 0 for i in range(72)]


#for i in range(72):
#    print(constraint2a[i], constraint2[i])


lhs_eq = [constraint1, constraint2, constraint3]
rhs_eq = [1.94, 9.9, 1.44]


bnd = [(0, float("inf")) for i in range(72)]

opt = linprog(c=objective2, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd)

hours = [i for i in range(24)]

y1 = opt.x[0:24]
y2 = opt.x[24:48]
y3 = opt.x[48:72]

print(sum(y1))
print(sum(y2))
print(sum(y3))


fig, axs = plt.subplots(3)

plt.setp(axs, xticks=list(range(24)))

axs[0].bar(hours, y1, color = "blue")
axs[0].set_title("Washing machine")

axs[1].bar(hours, y2, color = "orange")
axs[1].set_title("Electric vehicle")

axs[2].bar(hours, y3, color="green")
axs[2].set_title("dishwasher")

plt.show()


""" plt.bar(hours, y3)
plt.bar(hours, y3)
plt.bar(hours, y3)

width = 0.35
fig, ax = plt.subplots()

x = np.arange(len(hours))

rects1 = ax.bar(x - width/2, y1, width, label='Washing machine')
rects2 = ax.bar(x + width/2, y2, width, label='Electric vehicle')

ax.set_xticks(x)
ax.set_xticklabels(hours)
ax.legend()

fig.tight_layout()


plt.show() """