""" Code to create Figure 3-6 """

import matplotlib.style as style
from ptsf_setup import *
   
tumblr = pd.read_csv('ptsf-Python/Data/Tumblr.csv', parse_dates=True, index_col=0)
tumblr.index = tumblr.index.to_period('M')
views = (tumblr['People Worldwide'] / 1e6).to_frame()

style.use('ggplot')
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 5.2))
plt.subplots_adjust(wspace=0.6)
def customize_plot(ax, title):
    ax.set_title(title)
    ax.set_ylim([0, 1000])
    ax.set_xticks([pd.Period(year, freq='M') for year in ['2010', '2015', '2020']])
    ax.set_xticklabels(['2010', '2015', '2020'])
    ax.get_legend().remove()

def do_one_analysis(forecaster, train, fh, coverage, ax, xlab, ylab):
    forecaster.fit(train)
    pred = forecaster.predict(fh)
    pred_interval = forecaster.predict_interval(fh=fh, coverage=coverage)
    plot_series(train, pred, ax=ax, markers=['',''], pred_interval=pred_interval, x_label=xlab, y_label=ylab)
    return ax

fh = ForecastingHorizon(np.arange(1,116), is_relative=True)
coverage = np.array([.8,.6,.4,.2])
models = ["AAN","MMN","MMdN"]
for i, z in enumerate([("add", "add", False), ("mul", "mul", False), ("mul", "mul", True)]):
    forecaster = AutoETS(error=z[0], trend=z[1], damped_trend=z[2])
    axs[i] = do_one_analysis(forecaster, views, fh, coverage, axs[i], "", "People (in millions)")
    customize_plot(axs[i], f"ETS({models[i]})")

plt.savefig('Ch03_Fig_3_6_tumblr.pdf', format='pdf', bbox_inches='tight')
