import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib import rcParams, rc
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rc('pdf', fonttype=42)


# Data
methods = ['ResNet-50', 'Xception', 'EfficientFormer', 'LNP']
datasets = ['GenImage', 'DiffusionForensics', 'DMimageDetection', 'RealHD (Ours)']
colors = ["#FFC107", "#80B3FF", "#9AD0C2", "#F75A5A"]
markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond

# Data points
acc_data = {
    'ResNet-50': [58.61, 55.44, 56.65, 60.45],
    'Xception': [59.36, 60.87, 57.33, 66.25],
    'EfficientFormer': [59.45, 57.61, 57.31, 71.47],
    'LNP': [55.44, 49.94, 57.09, 55.99]
}

roc_auc_data = {
    'ResNet-50': [0.5389, 0.5341, 0.4121, 0.6549],
    'Xception': [0.4915, 0.6549, 0.3291, 0.7202],
    'EfficientFormer': [0.7692, 0.6090, 0.4796, 0.7811],
    'LNP': [0.4815, 0.5375, 0.4722, 0.5793]
}

# Create the plot
plt.figure(figsize=(8, 7))

# Plot each method
for i, method in enumerate(methods):
    for j, dataset in enumerate(datasets):
        plt.scatter(acc_data[method][j], roc_auc_data[method][j], 
                   color=colors[j], marker=markers[i], s=100,
                   label=f'{method} - {dataset}' if i == 0 else "")

# Customize the plot
x_ticks = np.arange(0, 81, 5)
y_ticks = np.arange(0, 0.85, 0.1)
plt.xticks(ticks=x_ticks, fontsize=30, fontfamily='Times New Roman')
plt.yticks(ticks=y_ticks, fontsize=30, fontfamily='Times New Roman')
plt.xlabel('Accuracy (%)', fontsize=30, fontfamily='Times New Roman')
plt.ylabel('ROC-AUC', fontsize=30, fontfamily='Times New Roman')
plt.title('Dataset Generalizability Evaluation', fontsize=40, fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.5)

# Create legend elements in two groups
dataset_legends = []
method_legends = []

# Dataset legend (colors)
for i, dataset in enumerate(datasets):
    dataset_legends.append(plt.Line2D([0], [0],  # 单点，避免线段干扰
        color=colors[i],
        lw=0,       # 隐藏线段（仅显示 marker）
        marker='_',  # 水平线条
        markersize=30,  # 控制宽度
        markeredgewidth=10,  # 控制高度（粗细）
        markeredgecolor=colors[i],
        markerfacecolor=colors[i],
        label=dataset))


# Method legend (markers)
for i, method in enumerate(methods):
    method_legends.append(plt.Line2D([0], [0], marker=markers[i], color='black',
                                   label=method, markersize=10,linestyle='',lw=0))

# Combine legends side by side
plt.legend(handles=dataset_legends + method_legends,
          loc='lower right',
          bbox_to_anchor=(0.98, 0.02),
          fontsize=18,
          ncol=2,
          labelcolor='linecolor',
          columnspacing=0.8,
          framealpha=0.4,
        edgecolor=(0.4, 0.4, 0.4, 0.4))

# Set axis limits to ensure data points are visible
plt.xlim(55, 80)  # Adjust based on your accuracy range
plt.ylim(0, 0.85)  # Adjust based on your ROC-AUC range

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('E:/table8.svg', bbox_inches='tight', format='svg')
plt.savefig('E:/table8.png', bbox_inches='tight')
plt.close()