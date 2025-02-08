""" Code to create Figure 9.1 """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import calendar

style.use('ggplot')

# Read the data
rainfall = pd.read_csv('ptsf-Python/Data/MelbourneRainfall.csv', parse_dates=['Date'], index_col='Date')
rainfall.index = pd.to_datetime(rainfall.index, format='%d/%m/%Y')

# Create additional columns
rainfall['rainy'] = np.where(rainfall['RainfallAmount_millimetres'] > 0, 1, 0)
rainfall['Year'] = rainfall.index.year
rainfall['Month'] = rainfall.index.month.map(lambda x: calendar.month_abbr[x])

# Calculate monthly and "month-year" rain percentages
monthly_rain = pd.DataFrame({'pct': rainfall.groupby('Month')['rainy'].mean() * 100})
monthly_yearly_rain = pd.DataFrame({'pct': rainfall.groupby(['Year', 'Month'])['rainy'].mean() * 100})

# Define the correct order for the months
month_order = list(calendar.month_abbr)[1:]

# Pivot the DataFrame and use month names as the index
pivot_df = monthly_yearly_rain.reset_index().pivot(index='Month', columns='Year', values='pct')
pivot_df.index = pd.CategoricalIndex(pivot_df.index, categories=month_order, ordered=True)
pivot_df = pivot_df.sort_index()

# Plotting using pivot_df.plot()
ax = pivot_df.plot(figsize=(10, 6), linestyle='-', marker='')

# Plot the average line
monthly_rain.index = pd.CategoricalIndex(monthly_rain.index, categories=month_order, ordered=True)
monthly_rain = monthly_rain.sort_index()
monthly_rain.plot(ax=ax, color='black', linestyle='--', linewidth=2, label='Average', marker='')

# Add labels and legend
ax.set_xlabel('Month')
ax.set_ylabel('Percent of rainy days per month')
ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

# Ensure all months are displayed on the x-axis
ax.set_xticks(range(len(month_order)))
ax.set_xticklabels(month_order)

plt.tight_layout()

# Save and show the plot
plt.savefig('Ch09_Fig_9_1_MelbourneRainfall.pdf', format='pdf', bbox_inches='tight')
plt.show()