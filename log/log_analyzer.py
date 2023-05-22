import matplotlib.pyplot as plt

log_file = "log"

lower_bounds = []
upper_bounds = []

with open(log_file, "r") as file:
    for line in file:
        if "len of fractional solutions" in line:
            lower_bound = float(line.split("len of fractional solutions")[0].split()[-1])
            lower_bounds.append(lower_bound)

        if line.startswith("obj"):
            if "inf" in line:
                upper_bound = float('inf')
            else:
                upper_bound = float(line.split()[1])
            upper_bounds.append(upper_bound)

# Plotting the lower and upper bounds
time_steps = range(len(lower_bounds))

plt.plot(time_steps, lower_bounds, label="Lower Bound")
plt.plot(time_steps, upper_bounds, label="Upper Bound")

plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Lower Bound and Upper Bound Convergence")

plt.legend()
plt.grid(True)
plt.show()