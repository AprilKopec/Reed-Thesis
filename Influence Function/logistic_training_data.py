import random

xdata = []
ydata = []

for _ in range(10):
    y = random.randint(0,1)
    if y:
        p = 3/4
    else:
        p = 1/4

    x = sum([random.random() < p for _ in range(10)])
    xdata += [x]
    ydata += [y]

print(xdata)
print(ydata)