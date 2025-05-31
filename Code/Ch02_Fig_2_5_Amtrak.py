""" Code to create Figure 2.5 """
# Assume ridership is already loaded with a DatetimeIndex named 'Month'
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns # the seaborn package makes it easier to produce a seasonal plot

style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership['Year'] = ridership.index.year
ridership['Mon'] = ridership.index.month
ridership_reset = ridership.reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=ridership_reset,
    x='Mon',
    y='Ridership',
    hue='Year',
    marker='',
    palette='tab10'
)
plt.xlabel('Month')
plt.ylabel('Ridership')
plt.title('Seasonal Plot: Ridership by Month and Year')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(
    ticks=range(1, 13),
    labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
)
plt.tight_layout()
plt.savefig('Ch02_Fig_2_5_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()

