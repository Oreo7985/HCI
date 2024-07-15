import pandas as pd
import matplotlib.pyplot as plt

# Data preparation
data_regular = {
    "Participant": ["Ann", "Chris", "Kai", "Oreo", "Andy", "Julian", "Nina", "Sebastian", "Max", "Caro"],
    "Start HR": [94, 104, 83, 95, 91, 87, 89, 92, 90, 93],
    "Max HR": [170, 169, 166, 150, 157, 162, 168, 165, 159, 160],
    "Average HR": [156.00, 149.00, 130.33, 117.67, 128.67, 129.67, 134.67, 133.33, 128.67, 130.67]
}

data_vr = {
    "Participant": ["Ann", "Chris", "Kai", "Oreo", "Andy", "Julian", "Nina", "Sebastian", "Max", "Caro"],
    "Start HR": [93, 101, 87, 100, 95, 88, 90, 98, 91, 94],
    "Max HR": [123, 131, 118, 120, 128, 121, 130, 127, 124, 132],
    "Average HR": [108.00, 114.67, 104.00, 109.33, 112.67, 104.33, 110.00, 113.00, 106.67, 112.33]
}

df_regular = pd.DataFrame(data_regular)
df_vr = pd.DataFrame(data_vr)

# Plotting function
def plot_combined_bars_with_labels_and_save(filepath):
    fig, ax = plt.subplots(figsize=(14, 8))
    participants = df_regular['Participant']
    x = range(len(participants))
    width = 0.35

    # Plot bars for regular activity
    bars_regular = ax.bar([xi - width/2 for xi in x], df_regular['Max HR'] - df_regular['Start HR'], width=width, bottom=df_regular['Start HR'], color='skyblue', label='Real HR Range')
    # Plot bars for VR activity
    bars_vr = ax.bar([xi + width/2 for xi in x], df_vr['Max HR'] - df_vr['Start HR'], width=width, bottom=df_vr['Start HR'], color='lightgreen', label='VR HR Range')

    # Add labels
    for bars, df, color, label_avg_hr in zip([bars_regular, bars_vr], [df_regular, df_vr], ['red', 'green'], ['Real Average HR', 'VR Average HR']):
        for bar, start, max_hr, avg_hr in zip(bars, df['Start HR'], df['Max HR'], df['Average HR']):
            ax.text(bar.get_x() + bar.get_width()/2, start, f'{start}', ha='center', va='bottom', fontsize=10)
            ax.text(bar.get_x() + bar.get_width()/2, max_hr, f'{max_hr}', ha='center', va='bottom', fontsize=10)
            ax.hlines(avg_hr, bar.get_x(), bar.get_x() + bar.get_width(), colors=color, label=label_avg_hr if bar is bars[0] else "")
            ax.text(bar.get_x() + bar.get_width()/2, avg_hr, f'{avg_hr}', ha='center', va='top', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(participants)
    ax.set_ylabel('Heart Rate')
    ax.set_title('Comparison of Heart Rate Data: Regular vs. VR Activity')
    ax.legend()
    plt.xlabel('Participant')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()

# Save path for the plot
save_path = 'Heart_Rate_Comparison.png'
plot_combined_bars_with_labels_and_save(save_path)