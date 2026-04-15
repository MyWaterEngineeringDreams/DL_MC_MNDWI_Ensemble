#13VizConfusionMatrixFromBestThresholds_UResNetMNDWIandSen2.py

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Custom colormap and bounds
colors = ["#99ccff", "#ff9999", "#d73027", "#87ceeb"]
custom_cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 4]
norm = BoundaryNorm(bounds, custom_cmap.N)

# Function to plot the confusion matrix with normalization to percentages
def plot_confusion_matrix(ax, cm, title):
    cm = cm.astype(int)  # Convert to integer for count representation
    total = cm.sum()  # Total count for percentage calculation
    percentages = cm / total * 100  # Normalize to percentage

    # Only create labels with percentages
    labels = [
        f"{percentage:.1f}%"
        for percentage in percentages.flatten()
    ]
    labels = np.array(labels).reshape(2, 2)

    # Define the matrix category (for tick labels)
    category_matrix = np.array([[0, 1], [2, 3]])

    # Plot the heatmap
    sns.heatmap(
        category_matrix,
        annot=labels,  # Only annotate with percentage
        fmt="",
        cmap=custom_cmap,
        norm=norm,
        cbar=False,
        ax=ax,
        xticklabels=["Dry", "Wet"],
        yticklabels=["Dry", "Wet"],
        annot_kws={"fontsize": 18},
        alpha=0.8,
    )

    # Set titles and labels
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("True", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=17)

# Normalize and plot the confusion matrices for BF and WF
def normalize_and_plot_conf_matrices(wf_conf_matrices, bf_conf_matrices):
    # Prepare the figure with 2 rows and 6 columns (for BF and WF)
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    axes = axes.flatten()

    # Normalize and plot BF matrices (first row)
    for i, conf_matrix in enumerate(bf_conf_matrices):
        conf_matrix = np.array(conf_matrix)  # Convert to numpy array
        plot_confusion_matrix(axes[i], conf_matrix, f'BF {2019 + i}')

    # Normalize and plot WF matrices (second row)
    for i, conf_matrix in enumerate(wf_conf_matrices):
        conf_matrix = np.array(conf_matrix)  # Convert to numpy array
        plot_confusion_matrix(axes[i + 6], conf_matrix, f'WF {2019 + i}')

    # Add the legend for categories
    legend_elements = [
        Patch(facecolor="#99ccff", edgecolor="black", label="True Neg"),
        Patch(facecolor="#ff9999", edgecolor="black", label="False Pos"),
        Patch(facecolor="#d73027", edgecolor="black", label="False Neg"),
        Patch(facecolor="#87ceeb", edgecolor="black", label="True Pos"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=16,
        title="Pixel Category",
        title_fontsize=18,
        frameon=True,
        bbox_to_anchor=(1.001, 0.083), 
        borderaxespad=0.3, 
    )
    #plt.subplots_adjust(hspace=0.4, wspace = 0.2) 
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout
    #plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\DLSeg\Fig_DLSeg\Reservoir Performance against sentinel2 thresh 2.2.png", dpi =600)
    plt.show()

bf_conf_matrices = bf_data['Confusion Matrix']
wf_conf_matrices = wf_data['Confusion Matrix']

# Normalize and plot the confusion matrices
normalize_and_plot_conf_matrices(wf_conf_matrices, bf_conf_matrices)
