import numpy as np
import matplotlib.pyplot as plt
# Define the function x^a and its derivative
def power_function(x, a):
    return x ** a

def derivative_power_function(x, a):
    return a * x ** (a - 1)

# Choose a value for 'a'
a = 0.1

# Adjust the x values range to cover from -1 to 1 for the function x^a and its derivative
x_values_power_adjusted = np.linspace(0, 1, 400)

# Recalculate y values for the adjusted range
y_values_power_adjusted = power_function(x_values_power_adjusted, a)
y_values_derivative_power_adjusted = derivative_power_function(x_values_power_adjusted, a)

# Create the plot for the adjusted range
plt.figure(figsize=(12, 6))

# Plot x^a for adjusted range
plt.subplot(1, 2, 1)
plt.plot(x_values_power_adjusted, y_values_power_adjusted, label=f"f(x) = x^{a}")
plt.title(f"Plot of f(x) = x^{a} from -1 to 1")
plt.xlabel('x')
plt.ylabel(f"f(x)")
plt.grid(True)
plt.legend()

# Plot derivative of x^a for adjusted range
plt.subplot(1, 2, 2)
plt.plot(x_values_power_adjusted, y_values_derivative_power_adjusted, label=f"f'(x) = {a}x^{a-1}", color='red')
plt.title(f"Derivative of f(x) = x^{a} from -1 to 1")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
