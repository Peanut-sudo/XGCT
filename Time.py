import matplotlib.pyplot as plt
import numpy as np

# 数据集训练时间
datasets = ['Dataset 1', 'Dataset 2']
times_with_fs = [91.336, 21.025]  # 有特征选择的时间（秒）
times_without_fs = [598.525, 26.906]  # 无特征选择的时间（秒）

# 设置柱状图的宽度和位置
bar_width = 0.35
index = np.arange(len(datasets))

# 选择SCI顶刊常用的颜色（柔和、清晰的对比色）
color_with_fs = '#4E79A7'  # 蓝色，通常用于干净专业的外观
color_without_fs = '#F28E2B'  # 橙色，对比鲜明但视觉平衡

# 绘制图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制有特征选择和无特征选择的柱状图
bars1 = ax.bar(index, times_with_fs, bar_width, color=color_with_fs, label='With Feature Selection')
bars2 = ax.bar(index + bar_width, times_without_fs, bar_width, color=color_without_fs, label='Without Feature Selection')

# 添加标签和标题
ax.set_xlabel('Datasets', fontsize=12)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
# ax.set_title('Training Time with and without Feature Selection for Different Datasets', fontsize=14)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend()

# 在柱状图上显示时间，保留三位小数
for bar in bars1 + bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{yval:.3f} s', ha='center', va='bottom', fontsize=10)

# 调整图表布局
plt.tight_layout()
plt.show()
