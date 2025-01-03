""" Code to create Figure 9.1 """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('ggplot')

rainfall = pd.read_csv('ptsf-Python/Data/MelbourneRainfall.csv', parse_dates=['Date'], index_col='Date')
rainfall.index = pd.to_datetime(rainfall.index, format='%d/%m/%Y')
rainfall['rainy'] = np.where(rainfall['RainfallAmount_millimetres'] > 0, 1, 0)
rainfall['Year'] = rainfall.index.year
rainfall['Month'] = rainfall.index.month

monthly_rain = pd.DataFrame({'pct': rainfall.groupby('Month')['rainy'].mean() * 100})
monthly_yearly_rain = pd.DataFrame({'pct': rainfall.groupby(['Year', 'Month'])['rainy'].mean() * 100})

pivot_df = monthly_yearly_rain.reset_index().pivot(index='Month', columns='Year', values='pct')

plt.figure(figsize=(10, 6))
lstyle = ['-', '--', '-.', ':']
for i, year in enumerate(pivot_df.columns):
    plt.plot(pivot_df.index, pivot_df[year], label=str(year), linestyle=lstyle[i % len(lstyle)])

plt.plot(monthly_rain.index, monthly_rain['pct'], color='black', linestyle='--',
         linewidth=2, label='Average')

plt.xlabel('Month')
plt.ylabel('Percent of rainy days per month')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('Ch09_Fig_9_1_MelbourneRainfall.pdf', format='pdf', bbox_inches='tight')
plt.show()
