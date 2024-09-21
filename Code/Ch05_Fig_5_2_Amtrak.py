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

def centered_moving_average(v, width):
    if width % 2 == 0: ## width is even
        left = v.rolling(window=width, min_periods=width, center=True).mean()
        right = v.rolling(window=width, min_periods=width, center=True).mean().shift(-1)
        return (left + right) / 2
    else:
        return v.rolling(window=width, min_periods=width, center=True).mean()

a = ridership.copy() ## use shorter name
WIDTH = 12
a['Centered'] = centered_moving_average(a['Ridership'], width=WIDTH)
a['Trailing'] = a['Ridership'].rolling(window=WIDTH, min_periods=WIDTH).mean()

fig, ax = plt.subplots(figsize=(5.5,3.5))
ax = plot_series(a['Ridership'], a['Centered'], a['Trailing'], markers=['']*3,
                 labels=['Ridership', 'Centered Moving Average', 'Trailing Moving Average'],
                 x_label="Time", y_label="Ridership", ax=ax)

ptsf_theme(ax, colors=['black','blue','red'], idx=[0,1,2], lty=['-','--','--'])
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, fontsize=7)
plt.subplots_adjust(bottom=0.25)

plt.savefig('Ch05_Fig_5_2_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()
