""" Code to create Figure 2.5 """
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership['Year'] = ridership.index.year
ridership['Mon'] = ridership.index.month

ax = ridership.pivot_table(index='Mon', columns='Year', values='Ridership').plot(figsize=(10,6))

ax.set_xlabel('Month')
ax.set_ylabel('Ridership')
ax.set_title('Seasonal Plot: Amtrak Ridership by Month and Year')
ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig('Ch02_Fig_2_5_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
