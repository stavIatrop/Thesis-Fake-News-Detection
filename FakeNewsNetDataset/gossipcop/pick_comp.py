import matplotlib.pyplot as plt

comps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
variance = [27, 40, 48, 55, 61, 65, 70, 73, 76]

plt.plot(comps, variance)
plt.grid(True)

plt.xlabel("Number of components")
plt.ylabel("Variance (%)")
plt.axis([min(comps) - 10, max(comps) + 10, min(variance) - 12, max(variance) + 6])
plt.title("Variance regarding number of components")



plt.show()
