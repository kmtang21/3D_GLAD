import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据配置
excel_file = 'Rawdata.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)

# 明确指定 SARS2 的 4 个目标（按此顺序排列）
keep_targets = ['nsp5', 'nsp14_N7-MTase', 'nsp14_ExoN', 'spike_omicron']
rename_map = {'spike_omicron': 'Spike'}

group_data = {}
# 按照 keep_targets 的顺序提取数据，确保横坐标顺序可控
for target in keep_targets:
    if target in all_sheets:
        df = all_sheets[target]
        display_name = rename_map.get(target, target)
        
        factor = 100 if 'nsp14' in target else 1
        inhib_cols = df.columns[-2:]
        
        for col in inhib_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float) * factor
        
        df['Max_Inhib'] = df[inhib_cols].max(axis=1).clip(lower=0)
        group_data[display_name] = df

# 2. 准备数据与计算阳性率
combined_data_max = []
x_labels = []
last_x = 0
positive_rates = [] # 存储 (x_pos, rate_value) 用于标注

for target_name, df in group_data.items():
    n = len(df)
    x = np.arange(last_x, last_x + n)
    
    # 格式化标签
    label_name = target_name.replace('_', '\n(') + ')' if '_' in target_name else target_name
    center_x = x[n // 2]
    x_labels.append((center_x, label_name))

    # 计算阳性率
    pos_rate = (df['Max_Inhib'] > 30).sum() / n * 100
    positive_rates.append((center_x, pos_rate))

    combined_data_max.append(pd.DataFrame({
        'Position': x,
        'Inhibition': df['Max_Inhib'],
        'Group': target_name
    }))
    last_x += n

plot_data_max = pd.concat(combined_data_max)

# 3. 绘图函数
def create_final_plot(data, title, filename, labels, rates):
    plt.figure(figsize=(15, 8))
    plt.axhline(y=30, color='black', linestyle='dashdot', linewidth=1.5, alpha=0.7)

    # 4个靶点对应 4 种颜色
    hex_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"] 
    groups = list(data['Group'].unique())
    
    for i, group in enumerate(groups):
        group_df = data[data['Group'] == group]
        # 设置点的大小和透明度
        plt.scatter(
            group_df['Position'], group_df['Inhibition'],
            c=hex_colors[i], alpha=0.5, edgecolors='none', s=45
        )
    
    # 标注阳性率数值
    for x_pos, rate in rates:
        plt.text(x_pos, 105, f"{rate:.1f}%", 
                 ha='center', va='bottom', fontsize=18, 
                 fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # 坐标轴与字体设置
    positions, ticks = zip(*labels)
    plt.xticks(positions, ticks, fontsize=22)
    plt.yticks(np.arange(0, 121, 20), fontsize=20)
    plt.xlabel('Proteins', fontsize=24, fontweight='bold') 
    plt.ylabel('Inhibition Rate (%)', fontsize=24, fontweight='bold')
    plt.title(title, fontsize=28, pad=30, fontweight='bold')
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.ylim(-5, 125) # 稍微留高一点空间给百分比标注
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

# 执行
create_final_plot(plot_data_max, 'SARS-CoV-2', 'Panel_A_SARS2_Final.png', x_labels, positive_rates)