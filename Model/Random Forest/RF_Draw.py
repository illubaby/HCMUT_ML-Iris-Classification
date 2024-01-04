import matplotlib.pyplot as plt
import numpy as np

# Assuming 'cm' is your confusion matrix as a numpy array
cm = np.array([[19, 0, 0],
               [0, 20, 1],
               [0, 1, 19]])

# Labels for the classes
class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Create a plot
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap='viridis')

# Add color bar for reference
plt.colorbar(cax)

# Set ticks and labels
ax.set_xticklabels([''] + class_labels)
ax.set_yticklabels([''] + class_labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# Rotate the tick labels for better readability
plt.xticks(rotation=45)

# Add the numerical values inside the squares
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center', color='black')

plt.show()
