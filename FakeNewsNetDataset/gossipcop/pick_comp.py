import matplotlib.pyplot as plt

comps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
variance = [34, 49, 60, 68, 75, 80, 85, 89, 92]

plt.plot(comps, variance)
plt.grid(True)

plt.xlabel("Number of components")
plt.ylabel("Variance (%)")
plt.axis([min(comps) - 10, max(comps) + 10, min(variance) - 12, max(variance) + 6])
plt.title("Variance regarding number of components")



plt.show()
