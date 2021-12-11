import time
import sched
import pandas as pd
import logging
import requests
from io import StringIO

import utils
from database import upsert_voltage


vol_url1 = "https://transmission.bpa.gov/Business/Operations/Charts/ashe.txt"
vol_url2 = "https://transmission.bpa.gov/Business/Operations/Charts/triCities.txt"
MAX_DOWNLOAD_ATTEMPT = 5
DOWNLOAD_PERIOD = 10         # second
logger = logging.Logger(__name__)
utils.setup_logger(logger, 'data.log')


def download_voltage(url, retries=MAX_DOWNLOAD_ATTEMPT):
    """
    download the data from the txt file
    return None if network failed
    """
    text = None
    for i in range(retries):
        try:
            req = requests.get(url, timeout=0.5)
            req.raise_for_status()
            text = req.text
        except requests.exceptions.HTTPError as e:
            logger.warning("Retry on HTTP Error: {}".format(e))
    if text is None:
        logger.error('download_voltage too many FAILED attempts')
    return text


def filter_voltage(text):
    # convert each txt to dataframe, removes empty lines and descriptions
    # use StringIO to convert string to a readable buffer
    df = pd.read_csv(StringIO(text), skiprows=6, delimiter='\t')
    df.columns = df.columns.str.strip()             # remove space in columns name
    df['Datetime'] = pd.to_datetime(df['Date/Time'])
    df.drop(columns=['Date/Time'], axis=1, inplace=True)
    df.dropna(inplace=True)             # drop rows with empty cells
    return df


def update_once():
    # join two datasets, and update the final df
    t1 = download_voltage(url=vol_url1)
    t2 = download_voltage(url=vol_url2)
    df1 = filter_voltage(t1)
    df2 = filter_voltage(t2)
    df1.columns = ['vol_value', 'Datetime']
    df2.drop(['Datetime'], axis=1, inplace=True)
    df = pd.concat([df1, df2], axis=1)
    # change the order for the columns
    df = df[['Datetime', 'vol_value', 'Import', 'Load', 'Generation']]
    upsert_voltage(df)


def main_loop(timeout=DOWNLOAD_PERIOD):
    scheduler = sched.scheduler(time.time, time.sleep)

    def _worker():
        try:
            update_once()
        except Exception as e:
            logger.warning("main loop worker ignores exception and continues: {}".format(e))
        scheduler.enter(timeout, 1, _worker)    # schedule the next event

    scheduler.enter(0, 1, _worker)              # start the first event
    scheduler.run(blocking=True)


if __name__ == '__main__':
    main_loop()