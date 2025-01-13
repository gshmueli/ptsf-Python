""" Code to create Figure 5.2  """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M').to_timestamp()

a = ridership.copy() ## use shorter name
WIDTH = 12
a['Centered'] = a['Ridership'].rolling(window=WIDTH, min_periods=WIDTH, center=True).mean()
a['Trailing'] = a['Ridership'].rolling(window=WIDTH, min_periods=WIDTH, center=False).mean()

fig, ax = plt.subplots(figsize=(5.5,3.5))
ax = plot_series(a['Ridership'], a['Centered'], a['Trailing'], markers=['']*3,
                 labels=['Ridership', 'Centered Moving Average', 'Trailing Moving Average'],
                 x_label="", y_label="Ridership", ax=ax)

ptsf_theme(ax, colors=['black','blue','red'], idx=[0,1,2], lty=['-','--','--'])
plt.legend(loc='upper left', fontsize=7)
plt.savefig('Ch05_Fig_5_2_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()
