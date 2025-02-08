""" Code to create Figure 9.1 """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns # the seaborn package makes it easier to produce a seasonal plot

style.use('ggplot')

# Read the data
rain = pd.read_csv('ptsf-Python/Data/MelbourneRainfall.csv', parse_dates=['Date'], index_col='Date')
rain.index = pd.to_datetime(rain.index, format='%d/%m/%Y')

# Create additional columns: 'rainy', 'Year', 'Month'
rain['rainy'] = np.where(rain['RainfallAmount_millimetres'] > 0, 1, 0)
rain['Year'] = rain.index.year
rain['Month'] = rain.index.month

# Calculate monthly and "month-year" rain percentages
month_year_rain = pd.DataFrame({'pct': rain.groupby(['Year', 'Month'])['rainy'].mean() * 100}).reset_index()
month_rain = pd.DataFrame({'pct': rain.groupby('Month')['rainy'].mean() * 100}).reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=month_year_rain, x='Month', y='pct', hue='Year', marker='', legend='full')

# Overlay the monthly average rain
sns.lineplot(data=month_rain, x='Month', y='pct', color='black', linestyle='--', linewidth=2, label='Average')

# Add labels and legend
plt.xlabel('')
plt.ylabel('Percent of rainy days per month')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(ticks=range(1, 13), labels=[str(i) for i in range(1, 13)])
plt.tight_layout()

# Save and show the plot
plt.savefig('Ch09_Fig_9_1_MelbourneRainfall.pdf', format='pdf', bbox_inches='tight')
plt.show()