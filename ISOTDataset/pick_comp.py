import matplotlib.pyplot as plt

comps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
variance = [28, 40, 48, 55, 60, 64, 68, 70, 73, 75]

plt.plot(comps, variance)
plt.grid(True)

plt.xlabel("Number of components")
plt.ylabel("Variance (%)")
plt.axis([min(comps) - 10, max(comps) + 10, min(variance) - 12, max(variance) + 6])
plt.title("Variance regarding number of components")



plt.show()
