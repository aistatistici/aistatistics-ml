import os
from datetime import datetime

from ml.utils.csv import open_csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time


def date_fom_str(x):
    return datetime.fromtimestamp(time.mktime(time.strptime(x, "%m/%d/%Y")))


def to_float(x):
    return float(x) if x != '' else 0


csv_init = open_csv(os.path.abspath("./data/balances.csv"))[1:]

for i in range(1, 6):
    csv = csv_init[csv_init[:, 3] == '12']
    csv = csv[csv[:, 4] == str(i)]
    # csv = csv[csv[:, 1].astype(float) > 0]
    csv = csv[sorted(np.unique(csv[:, 0], return_index=True)[1])]

    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots()

    ax2 = ax.twinx()

    dates = np.vectorize(date_fom_str)(csv[:, 0])

    ax.plot(dates, np.vectorize(to_float)(csv[:, 1]))


    ax2.plot(dates, np.vectorize(to_float)(csv[:, 2]))

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)


    ax2.xaxis.set_major_locator(years)
    ax2.xaxis.set_major_formatter(yearsFmt)
    ax2.xaxis.set_minor_locator(months)

    # round to nearest years...
    datemin = np.datetime64(date_fom_str(csv[0][0]), 'Y')
    datemax = np.datetime64(date_fom_str(csv[-1][0]), 'Y') + np.timedelta64(1, 'Y')
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
