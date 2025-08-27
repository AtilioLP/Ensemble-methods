import pandas as pd
import numpy as np
from matplotlib import use, get_backend
use('TkAgg', force=True)
from matplotlib import pyplot as plt
print("Switched to:", get_backend())
import seaborn as sns
from matplotlib.patches import Patch


df_copy = pd.read_csv("data/merged_df_SP.csv")

df_long = pd.melt(
    df_copy,
    id_vars=[col for col in df_copy.columns if col not in ['met1_CRPS', 'met2_CRPS']],
    value_vars=['met1_CRPS', 'met2_CRPS'],
    var_name='method',
    value_name='CRPS'
)

df_long['method'] = df_long['method'].map({
    'met1_CRPS': 1,
    'met2_CRPS': 2
})

# Ploting boxplots
sns.set(style="whitegrid")
period_palette = {1: 'blue', 2: 'green', 3: 'red'}

fig, ax = plt.subplots(figsize=(14, 6))

model_ids = df_long['model_id'].unique()
valid_tests = sorted(df_long['valid_test'].unique())
methods = [1, 2]

box_width = 0.2
spacing = 0.35
group_spacing = 0.5

positions = []
labels = []

pos_counter = 0

for model in model_ids:
    for method in methods:
        for vt in valid_tests:
            subset = df_long[
                (df_long['model_id'] == model) &
                (df_long['valid_test'] == vt) &
                (df_long['method'] == method)
            ]
            if not subset.empty:
                box = ax.boxplot(
                    subset['CRPS'],
                    positions=[pos_counter],
                    widths=box_width,
                    patch_artist=True,
                    medianprops=dict(linewidth=2.5)
                )
                for patch in box['boxes']:
                    patch.set_facecolor(period_palette[int(vt)])
                    if method == 2:
                        patch.set_hatch('//')
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.2)
                labels.append(f'{model}\nP{vt}\nM{method}')
                positions.append(pos_counter)
                pos_counter += spacing
    pos_counter += group_spacing

# Legend
legend_elements = [
    Patch(facecolor='blue', label='Period 1'),
    Patch(facecolor='green', label='Period 2'),
    Patch(facecolor='red', label='Period 3'),
    Patch(facecolor='gray', label='Method 1'),
    Patch(facecolor='gray', hatch='//', label='Method 2')
]

ax.legend(handles=legend_elements, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

group_centers = []
current = 0
for _ in model_ids:
    group_centers.append(current + (spacing * 3))
    current += spacing * 6 + group_spacing

ax.set_xticks(group_centers)
ax.set_xticklabels(model_ids, rotation=45, ha='right')

ax.set_xlabel('Model ID')
ax.set_ylabel('CRPS')
ax.set_title('Comparing CRPS of Methods (UF: {})'.format(np.unique(df_long['uf'])[0]))
plt.tight_layout()
plt.show()

# Making the plot for a period only
df_period1 = df_long[df_long['valid_test'] == 1].copy()

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

positions = []
box_data = []
colors = []
hatch_styles = []
labels = []

box_width = 0.3
spacing = 0.4
group_spacing = 1.0
position_counter = 0

unique_models = df_period1['model_id'].unique()

pos_counter = 0
for model in model_ids:
    for method in methods:
        subset = df_long[
            (df_long['model_id'] == model) &
            (df_long['method'] == method)
        ]
        if not subset.empty:
            box = ax.boxplot(
                subset['CRPS'],
                positions=[pos_counter],
                widths=box_width,
                patch_artist=True,
                medianprops=dict(linewidth=2.5)
            )
            for patch in box['boxes']:
                patch.set_facecolor('blue')
                if method == 2:
                    patch.set_hatch('//')
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)
            labels.append(f'{model}\nP{vt}\nM{method}')
            positions.append(pos_counter)
            pos_counter += spacing
    pos_counter += group_spacing

# Legend
legend_elements = [
    Patch(facecolor='blue', hatch='', label='Method 1'),
    Patch(facecolor='blue', hatch='//', label='Method 2')
]
ax.legend(handles=legend_elements, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

group_centers = []
current = 0
for _ in model_ids:
    group_centers.append(current + (spacing * 1))
    current += spacing * 2 + group_spacing

ax.set_xticks(group_centers)
ax.set_xticklabels(model_ids, rotation=45, ha='right')

ax.set_xlabel('Model ID')
ax.set_ylabel('CRPS')
ax.set_title('Comparing CRPS of Methods for Period 1 (UF: {})'.format(np.unique(df_long['uf'])[0]))
plt.tight_layout()
plt.show()


def plot_crps_timeseries(model_id, valid_test):
    # Filter the dataframe for the selected model and period
    filtered_df = df_long[
        (df_long['model_id'] == model_id) &
        (df_long['valid_test'] == valid_test)
        ]

    filtered_df.loc[:, 'date'] = pd.to_datetime(filtered_df['date'])

    if filtered_df.empty:
        print("No data available for the specified model_id and valid_test period.")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot each method
    for method in [1, 2]:
        method_df = filtered_df[filtered_df['method'] == method].sort_values('date')
        linestyle = '-' if method == 1 else '--'
        plt.plot(
            method_df['date'],
            method_df['CRPS'],
            label=f"Method {method}",
            linestyle=linestyle
        )

    all_dates = sorted(filtered_df['date'].unique())
    step = max(1, 5)
    selected_dates = all_dates[::step]
    plt.xticks(selected_dates, rotation=10)

    plt.xlabel("Date")
    plt.ylabel("CRPS")
    aux_uf = np.unique(df_long['uf'])[0]
    plt.title(f"CRPS Time-Series for Model {model_id}, Period {valid_test} (UF: {aux_uf})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_models_for_period(df_long, valid_test_period):
    # Filter for the selected period
    df_period = df_long[df_long['valid_test'] == valid_test_period]

    if df_period.empty:
        print(f"No data found for period {valid_test_period}")
        return

    df_period.loc[:, 'date'] = pd.to_datetime(df_period['date'])

    # Get unique models
    model_ids = sorted(df_period['model_id'].unique())
    n_models = len(model_ids)

    # Define subplot grid
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2 * n_rows), sharex=False)
    axes = axes.flatten()

    # Plot each model
    for i, model_id in enumerate(model_ids):
        ax = axes[i]
        model_data = df_period[df_period['model_id'] == model_id]

        for method in [1, 2]:
            method_data = model_data[model_data['method'] == method].sort_values('date')
            linestyle = '-' if method == 1 else '--'
            ax.plot(method_data['date'], method_data['CRPS'], linestyle=linestyle, label=f"Method {method}")

        all_dates = sorted(method_data['date'].unique())
        step = max(1, 13)
        selected_dates = all_dates[::step]
        ax.set_xticks(selected_dates)
        #ax.tick_params(axis='x', rotation=0)

        ax.set_title(f"Model: {model_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("CRPS")
        ax.legend()

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    aux_uf = np.unique(df_long['uf'])[0]
    plt.suptitle(f"CRPS Time-Series by Model – Period {valid_test_period} (UF: {aux_uf})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


plot_all_models_for_period(df_long, valid_test_period=1)
plot_all_models_for_period(df_long, valid_test_period=2)
plot_all_models_for_period(df_long, valid_test_period=3)