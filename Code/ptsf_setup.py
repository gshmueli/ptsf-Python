"""Custom helper functions for the book Practical Time Series Forecasting with Python"""
from datetime import datetime
import pandas as pd
import numpy as np
#from scipy.stats import t as t_dist
#import matplotlib.style as style
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#import arrow
import seaborn as sns
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.performance_metrics.forecasting import mean_absolute_error, \
        mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm






def ptsf_train_test(ax, train_index, test_index, ylim=None, arrow_y=None, text_y = None ):
    """Many charts have a vertical line at the train/test boundary and label the 2 periods."""
    grey37 = '0.37'

    # Add vertical line
    ax.axvline(x=train_index[-1], color=grey37, linewidth=1.3)

    if ylim is None:
        ylim = ax.get_ylim()

    delta_mult = .05 ## this may need to be tweaked
    if arrow_y is None:
        delta = ylim[1] - ylim[0]
        arrow_y = ylim[1] + delta_mult * delta

    if text_y is None:
        text_y = arrow_y + delta_mult * delta

    # Add segments
    ax.annotate('', xy=(train_index[0], arrow_y), xytext=(train_index[-1], arrow_y),
                arrowprops=dict(arrowstyle='<->', color=grey37, mutation_scale=20))
    ax.annotate('', xy=(train_index[-1], arrow_y), xytext=(test_index[-1], arrow_y),
                arrowprops=dict(arrowstyle='<->', color=grey37, mutation_scale=20))

    # Add text
    if isinstance(train_index[0], pd.Timestamp) and isinstance(test_index[0], pd.Timestamp):
        # Use the commented-out code if train_index and test_index are Timestamps
        mid_train = train_index[0] + (train_index[-1] - train_index[0]) / 2
        mid_test = train_index[-1] + (test_index[-1] - train_index[-1]) / 2
    else:
        # Use the existing code if train_index and test_index are not Timestamps
        mid_train = train_index[0].to_timestamp() + \
            (train_index[-1].to_timestamp() - train_index[0].to_timestamp()) / 2
        mid_test = train_index[-1].to_timestamp() + \
            (test_index[-1].to_timestamp() - train_index[-1].to_timestamp()) / 2

    ax.text(mid_train, text_y, 'Training', color=grey37, ha='center', fontsize=12)
    ax.text(mid_test, text_y, 'Test', color=grey37, ha='center', fontsize=12)

    ax.set_ylim(ylim[0], text_y + 4 * delta_mult * delta)

    return ax


def insert_colors(sns_palette, colors, positions):
    """Helper function for some plotting"""
    col_lst = list(sns_palette)

    # Insert colors at specific positions
    for color, pos in zip(colors, positions):
        col_lst.insert(pos, color)

    # Convert the list back to a seaborn color palette
    col_pal = sns.color_palette(col_lst)
    return col_pal

            
def ptsf_theme(ax, colors, idx, lty=None, labels=None, do_legend=False, legend_loc='best'):
    """Keep a consistent graphics theme; emulates theme in R book, which is ggplot style"""
    n_series = len(ax.get_lines())
    pal = sns.color_palette("colorblind", n_colors=n_series)
    pal = insert_colors(pal, colors, idx)
    for i in range(n_series):
        ax.get_lines()[i].set_color(pal[i])
    if lty is not None:
        for i in range(len(lty)):
            ax.get_lines()[i].set_linestyle(lty[i])
    # Remove markers from all lines - we might want to revisit this
    for line in ax.lines:
        line.set_marker("")
    # if the plot has a title, move it to the left and reduce its size
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=12, ha='left', x=0)
    # if do_legend is True, create a legend with the provided labels
    if do_legend and labels is not None:
        if legend_loc == 'below':
            legend = ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                               ncol=len(labels))
        else:
            legend = ax.legend(labels, loc=legend_loc)
        # Set the colors and linestyles for the legend lines to match those of the plot lines
        for legend_line, plot_line in zip(legend.get_lines(), ax.get_lines()):
            legend_line.set_color(plot_line.get_color())
            legend_line.set_linestyle(plot_line.get_linestyle())

def ptsf_summary(fc, y):
    """Calculates and prints a summary of a fitted forecaster"""
    # Check that fc is a PolynomialTrendForecaster or TrendForecaster
    if not isinstance(fc, (PolynomialTrendForecaster, TrendForecaster)):
        raise TypeError("fc must be an instance of PolynomialTrendForecaster or TrendForecaster")

    if isinstance(y.index, pd.DatetimeIndex):
        days_from_epoch = (y.index - pd.Timestamp("1970-01-01")).days.values.reshape(-1, 1)
        terms_from_epoch = days_from_epoch
    elif isinstance(y.index, pd.PeriodIndex):
        epoch = pd.Period("1970-01-01", freq=y.index.freq)
        periods_from_epoch = [(period - epoch).n for period in y.index]
        terms_from_epoch = np.array(periods_from_epoch).reshape(-1, 1)
    else:
        raise TypeError("y.index must be a DatetimeIndex or PeriodIndex")

    if isinstance(fc, PolynomialTrendForecaster):
        # get the coefficients from the fitted forecaster (fc should have been fitted already)
        final_estimator = fc.regressor_.named_steps['linearregression']
        #intercept = final_estimator.intercept_
        coefficients = final_estimator.coef_

        X = fc.regressor_.named_steps['polynomialfeatures'].transform(terms_from_epoch)
        X = sm.add_constant(X)  # Add intercept term
        model = sm.OLS(y.values, X).fit()  # Fit the model using statsmodels
        sm_coefficients = model.params

        #if not np.allclose(np.append(intercept, coefficients), sm_coefficients):
        if not np.allclose(coefficients, sm_coefficients):
            raise ValueError("Coefficients from final estimator do not match those from sm.OLS")

    elif isinstance(fc, TrendForecaster):
        # get the coefficients from the fitted forecaster (fc should have been fitted already)
        intercept = fc.regressor_.intercept_
        coefficients = fc.regressor_.coef_

        X = sm.add_constant(terms_from_epoch)  # Add intercept term
        model = sm.OLS(y.values, X).fit()  # Fit the model using statsmodels
        sm_coefficients = model.params

        if not np.allclose(np.append(intercept, coefficients), sm_coefficients):
            raise ValueError("Coefficients from final estimator do not match those from sm.OLS")

    # We can use the OLS summary as the basis of the summary
    summary = model.summary2()
    summary.tables[0].title = "TrendForecaster Results" if isinstance(fc, TrendForecaster) else "PolynomialTrendForecaster Results"
    summary.tables[0].iloc[0, 1] = "TrendForecaster" if isinstance(fc, TrendForecaster) else "PolynomialTrendForecaster"
    summary.tables[0].iloc[1, 0] = "Dep. Variable:"
    summary.tables[0].iloc[2, 0] = "# Obs:"
    summary.tables[0].iloc[2, 1] = summary.tables[0].iloc[3, 1]
    summary.tables[0].iloc[3, 0] = "Time:"
    summary.tables[0].iloc[3, 1] = datetime.now().strftime("%H:%M")
    summary.tables[1].columns = ['coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']
    feature_names = ['const'] + [f't**{i}' for i in range(1, len(coefficients))] if isinstance(fc, PolynomialTrendForecaster) else ['const', 't']
    summary.tables[1].index = feature_names

    # Format the values in scientific notation with 3 significant digits
    formatted_table = summary.tables[1].map(lambda x: f"{x:.2e}")

    # Convert the formatted table to a string and add separators
    formatted_table_str = formatted_table.to_string()
    separator = "=" * (len(formatted_table.columns) * 12)
    dash_separator = "-" * (len(formatted_table.columns) * 12)
    formatted_table_str = f"{separator}\n{formatted_table_str}\n{separator}"

    # Print the modified summary
    print(summary.tables[0].title.center(len(separator)))
    print(f"{separator}")
    print(summary.tables[0].to_string(index=False, header=False))
    print(formatted_table_str)

    return None

def ptsf_get_prediction_interval(model, X, alpha, colname, a_index):  # assumes model is statsmodels.OLS
    pi = model.get_prediction(X).summary_frame(alpha=alpha) 
    p_i = pd.DataFrame({'lower': pi['obs_ci_lower'], 'upper': pi['obs_ci_upper']}, index=a_index)
    p_i.columns = pd.MultiIndex.from_product([[colname],[1-alpha], ['lower', 'upper']])
    return p_i
