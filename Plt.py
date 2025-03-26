import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(confusion_matrix, title, x_labels, y_labels):
    """
    Plots a confusion matrix with individual numbers and labels.

    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix to plot.
        title (str): The title of the plot.
        x_labels (list): The labels for the x-axis.
        y_labels (list): The labels for the y-axis.

    Returns:
        None
    """

    fig, ax = plt.subplots()
    ax.imshow(confusion_matrix, cmap='BuGn')

    # Add labels and title with increased fontsize
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_yticklabels(y_labels, fontsize=14)
    plt.title(title, fontsize=18)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)

    # Display individual numbers in the matrix with increased fontsize
    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, z, ha='center', va='center', fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Example usage
confusion_matrix = np.array([
    [117,  3],
    [ 6, 114]
    
])

x_labels = ['Healthy', 'PD']
y_labels = ['Healthy', 'PD']

title = "Dataset 2"

plot_confusion_matrix(confusion_matrix, title, x_labels, y_labels)
