import logging
import pymongo
import pandas as pds
import expiringdict

import utils

client = pymongo.MongoClient()
logger = logging.Logger(__name__)
utils.setup_logger(logger, 'db.log')
RESULT_CACHE_EXPIRATION = 10             # seconds


# Update MongoDB database "voltage" with the dataframe we get.
def upsert_voltage(df):
    db = client.get_database("voltage")
    collection = db.get_collection("voltage")
    update_count = 0
    for record in df.to_dict('records'):
        result = collection.replace_one(
            filter={'Datetime': record['Datetime']},    # locate the document if exists
            replacement=record,                         # latest document
            upsert=True)                                # update if exists, insert if not
        if result.matched_count > 0:
            update_count += 1
    logger.info("rows={}, update={}, ".format(df.shape[0], update_count) +
                "insert={}".format(df.shape[0]-update_count))


def fetch_all_voltage():
    db = client.get_database("voltage")
    collection = db.get_collection("voltage")
    ret = list(collection.find())
    logger.info(str(len(ret)) + ' documents read from the db')
    return ret


_fetch_all_voltage_as_df_cache = expiringdict.ExpiringDict(max_len=1,
                                                       max_age_seconds=RESULT_CACHE_EXPIRATION)

# convert the output from 'fetch_all_voltage' to dataframe, and drop the string column id
# retrieve the cashe if allowed.
def fetch_all_voltage_as_df(allow_cached=False):
    def _work():
        data = fetch_all_voltage()
        if len(data) == 0:
            return None
        df = pds.DataFrame.from_records(data)
        df.drop('_id', axis=1, inplace=True)
        return df

    if allow_cached:
        try:
            return _fetch_all_voltage_as_df_cache['cache']
        except KeyError:
            pass
    ret = _work()
    _fetch_all_voltage_as_df_cache['cache'] = ret
    return ret


if __name__ == '__main__':
    print(fetch_all_voltage_as_df())
