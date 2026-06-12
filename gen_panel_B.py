import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
excel_file = 'CHIKV_rawdata.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)

# 定义 CHIKV 的 4 个目标顺序
# 请确保这些名称与你的 Excel Sheet 名称完全匹配
keep_targets_B = ['nsP1', 'nsP2_helicase', 'nsP2_protease', 'nsP3_Mac']

group_data_B = {}
for target in keep_targets_B:
    if target in all_sheets:
        df = all_sheets[target]
        
        # CHIKV 数据处理逻辑：取最后三列的平均/最大值
        # 根据你之前的代码，CHIKV 有 3 个重复孔
        inhib_cols = df.columns[-3:] 
        
        for col in inhib_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
        # 计算最大抑制率并裁剪负值
        df['Max_Inhib'] = df[inhib_cols].max(axis=1).clip(lower=0)
        group_data_B[target] = df

# 2. 准备绘图数据并计算阳性率
combined_data_B = []
x_labels_B = []
positive_rates_B = []
last_x = 0

for target_name in keep_targets_B:
    if target_name not in group_data_B:
        continue
        
    df = group_data_B[target_name]
    n = len(df)
    x = np.arange(last_x, last_x + n)
    
    # 格式化标签：将 nsP2_helicase 变为 nsP2\n(helicase)
    display_label = target_name.replace('_', '\n(') + ')' if '_' in target_name else target_name
    center_x = x[n // 2]
    x_labels_B.append((center_x, display_label))
    
    # 计算阳性率 (>30%)
    rate = (df['Max_Inhib'] > 30).sum() / n * 100
    positive_rates_B.append((center_x, rate))
    
    combined_data_B.append(pd.DataFrame({
        'Position': x,
        'Inhibition': df['Max_Inhib'],
        'Group': target_name
    }))
    last_x += n

plot_data_B = pd.concat(combined_data_B)

# 3. 绘图函数 (与 Panel A 保持高度一致)
def create_panel_b_plot(data, title, filename, labels, rates):
    plt.figure(figsize=(15, 8))
    
    # 基准线
    plt.axhline(y=30, color='black', linestyle='dashdot', linewidth=1.5, alpha=0.7)

    # 颜色方案 (可根据喜好调整，这里选用了另一组明亮的颜色以区分病毒种类)
    hex_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"] 
    groups = list(data['Group'].unique())
    
    for i, group in enumerate(groups):
        group_df = data[data['Group'] == group]
        plt.scatter(
            group_df['Position'], group_df['Inhibition'],
            c=hex_colors[i], alpha=0.5, edgecolors='none', s=45
        )
    
    # 标注阳性率
    for x_pos, rate in rates:
        plt.text(x_pos, 105, f"{rate:.1f}%", 
                 ha='center', va='bottom', fontsize=18, 
                 fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # 坐标轴与字体设置 (与 A 图完全一致)
    positions, ticks = zip(*labels)
    plt.xticks(positions, ticks, fontsize=22)
    plt.yticks(np.arange(0, 121, 20), fontsize=20)
    plt.xlabel('Proteins', fontsize=24, fontweight='bold') # B图原代码有xlabel，A图也可以加上
    plt.ylabel('Inhibition Rate (%)', fontsize=24, fontweight='bold')
    plt.title(title, fontsize=28, pad=30, fontweight='bold')
    
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.ylim(-5, 125) # 统一高度
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Panel B saved as: {filename}")

# 执行绘图
create_panel_b_plot(plot_data_B, 'CHIKV', 'Panel_B_CHIKV_Final.png', x_labels_B, positive_rates_B)