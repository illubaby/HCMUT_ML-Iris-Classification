import matplotlib.pyplot as plt
import numpy as np

# Create an array of values to represent the X-axis
x_values = np.linspace(-10, 10, 200)
# Define the sigmoid function
sigmoid = lambda x: 1 / (1 + np.exp(-x))
# Apply the sigmoid function to each x value
y_values = sigmoid(x_values)

plt.figure(figsize=(8, 4))  # Set the figure size
# Plot the sigmoid function
plt.plot(x_values, y_values, label='Sigmoid Function')

# Customize the plot
plt.title('Logistic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.yticks([0, 0.5, 1], ['y=0', '0.5', 'y=1'])  # Set custom ticks for y-axis
plt.axhline(y=1, color='grey', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
plt.text(10, 0.8, 'Predicted Y lies between 0 and 1 range', fontsize=9, ha='right')
plt.text(0, 0.2, 'S-shaped curve', fontsize=9, ha='center')

# Add a grid for better readability
plt.grid(True)
# Show the legend
plt.legend()
# Show the plot
plt.show()
