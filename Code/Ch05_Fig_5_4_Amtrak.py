""" Code to create Figure 5.4 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from ptsf_setup import ptsf_theme

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
b = ridership.copy()
b['Lag-12 Difference'] = ridership['Ridership'].diff(12)
b['Lag-1 Difference'] = ridership['Ridership'].diff(1)
b['Twice Differenced (Lag-12, Lag-1)'] = ridership['Ridership'].diff(12).diff(1)

fig, axes = plt.subplots(nrows=2, ncols=2, tight_layout=True, figsize=(7,4))
for i, column in enumerate(b.columns):
    row = i // 2  # Calculate the subplot row index
    col = i % 2   # Calculate the subplot column index
    ax = axes[row, col]  # Get the current subplot
    ax.plot(b.index, b[column])
    ax.get_lines()[0].set_color('black')
    ax.get_lines()[0].set_linewidth(1)
    ax.set_title(column, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('Ridership')
    years = [pd.to_datetime(str(year)) for year in [1992, 1996, 2000, 2004]] 
    ax.set_xticks(years) 
    ax.set_xticklabels([year.year for year in years])
    ptsf_theme(ax, colors=['black'],idx=[0], lty=['-'])

plt.savefig('Ch05_Fig_5_4_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
