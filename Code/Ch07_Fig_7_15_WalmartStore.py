""" Code to create Figure 7.15 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series

warnings.filterwarnings('ignore', category=FutureWarning)

sales = pd.read_csv('ptsf-Python/Data/WalmartStore1Dept72.csv', index_col='Date')
sales.index = pd.to_datetime(sales.index, format='%d/%m/%Y').to_period('W').to_timestamp()
sales = sales['Weekly_Sales'].to_frame()

style.use('ggplot')
fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(sales, colors=['black'], markers=[''], ax=ax, y_label='Weekly Sales')

# Rotate xticklabels by 45 degrees
ax.tick_params(axis='x', rotation=45)

plt.savefig('Ch07_Fig_7_15_WalmartStore.pdf', format='pdf', bbox_inches='tight')
plt.show()
