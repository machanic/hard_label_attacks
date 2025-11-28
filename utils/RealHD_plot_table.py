import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Add Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Data
methods = ['Resnet-50', 'Xception', 'Efficientformer', 'LNP']
datasets = ['GenImage', 'DiffusionForensics', 'DMimageDetection', 'RealHD(Ours)']
colors = ["#FFD65A", "#80B3FF", "#9AD0C2", "#F75A5A"]
markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond

# Data points
acc_data = {
    'Resnet-50': [59.40, 55.44, 56.65, 72.57],
    'Xception': [59.36, 60.87, 57.33, 66.25],
    'Efficientformer': [59.45, 57.61, 57.31, 71.47],
    'LNP': [68.18, 60.27, 61.21, 61.77]
}

roc_auc_data = {
    'Resnet-50': [0.5389, 0.5341, 0.4121, 0.7824],
    'Xception': [0.4915, 0.6549, 0.3291, 0.7202],
    'Efficientformer': [0.7692, 0.6090, 0.4796, 0.7811],
    'LNP': [0.4371, 0.3879, 0.2603, 0.4384]
}

# Create the plot
plt.figure(figsize=(8, 6))

# Plot each method
for i, method in enumerate(methods):
    for j, dataset in enumerate(datasets):
        plt.scatter(acc_data[method][j], roc_auc_data[method][j], 
                   color=colors[j], marker=markers[i], s=100,
                   label=f'{method} - {dataset}' if i == 0 else "")

# Customize the plot
plt.xlabel('Accuracy (%)', fontsize=16, fontfamily='Times New Roman')
plt.ylabel('ROC-AUC', fontsize=16, fontfamily='Times New Roman')
plt.title('Dataset Generalisability Evaluation', fontsize=16, fontfamily='Times New Roman', pad=20)
plt.grid(True, linestyle='--', alpha=0.5)

# Create legend elements in two groups
dataset_legends = []
method_legends = []

# Dataset legend (colors)
for i, dataset in enumerate(datasets):
    dataset_legends.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=colors[i], label=dataset, markersize=12))
# Method legend (markers)
for i, method in enumerate(methods):
    method_legends.append(plt.Line2D([0], [0], marker=markers[i], color='black', 
                                   label=method, markersize=12))

# Combine legends side by side
plt.legend(handles=dataset_legends + method_legends, 
          loc='lower right', 
          bbox_to_anchor=(0.98, 0.02),
          fontsize=14, 
          ncol=2,
          columnspacing=1.0,
          framealpha=0.9)

# Set axis limits to ensure data points are visible
plt.xlim(55, 75)  # Adjust based on your accuracy range
plt.ylim(0.2, 0.85)  # Adjust based on your ROC-AUC range

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('table8.svg', bbox_inches='tight', format='svg')
plt.savefig('table8.png', bbox_inches='tight', dpi=300)
plt.close() 