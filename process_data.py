import os
from datetime import datetime

from ml.utils.csv import open_csv_as_data_frame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time


def date_fom_str(x):
    return datetime.fromtimestamp(time.mktime(time.strptime(x, "%m/%d/%Y")))


def to_float(x):
    return float(x) if x != '' else 0


def inverse(x):
    return 1 / x if x != 0 else x


def noop(x):
    return x


csv = open_csv_as_data_frame(os.path.abspath('./prepared_data/balances/balance_Type_12_Currency_3.csv'))

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

fig, ax = plt.subplots()

ax2 = ax.twinx()

ax.plot(csv['Date'].values, csv['Balance'].values)

ax2.plot(csv['Date'].values, csv['Avg Rate (for Balance)'].values, 'r')

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

ax2.xaxis.set_major_locator(years)
ax2.xaxis.set_major_formatter(yearsFmt)
ax2.xaxis.set_minor_locator(months)

# round to nearest years...
datemin = np.datetime64(csv['Date'].values[0], 'Y')
datemax = np.datetime64(csv['Date'].values[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)
ax2.set_xlim(datemin, datemax)


# format the coords message box
def price(x):
    return '$%1.2f' % x


ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()
