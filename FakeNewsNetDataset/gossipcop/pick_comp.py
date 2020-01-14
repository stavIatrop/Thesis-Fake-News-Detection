import matplotlib.pyplot as plt

comps = [100, 600, 1100, 1600, 2100, 2600, 3100, 3600, 4100, 4600, 5100, 5600, 6100, 6600, 7100, 7600]
variance = [12, 32, 44, 53, 60, 65, 70, 79, 78, 81, 84, 86, 89, 90, 93, 94]

plt.plot(comps, variance)
plt.grid(True)

plt.xlabel("Number of components")
plt.ylabel("Variance (%)")
plt.axis([min(comps) - 10, max(comps) + 10, min(variance) - 12, max(variance) + 6])
plt.title("Variance regarding number of components")



plt.show()
