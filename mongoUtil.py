import pandas as pd
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from logger import Logger
from configparser import SafeConfigParser
from dateutil.relativedelta import relativedelta
import csv
import time
import sys
log = Logger('mongo.log')
config = SafeConfigParser()
config.read('config.ini')
mongo_host = config.get('Mongo','HOST')
mongo_port = config.getint('Mongo','PORT')

class mongoUtil:
    def __init__(self):
        print("Initialized.")

    def get_client(self):
        return MongoClient(mongo_host, mongo_port)

    def index_to_db(self,filename):
        path='train_data/'+str(filename)
        with open(path) as f:
            reader = csv.reader(f)
            i = next(reader)

        client = self.get_client()
        db = client.otc
        csvfile = open(path)
        reader = csv.DictReader(csvfile)
        for each in reader:
            row={}
            for field in i:
                row[field] = each[field]
                row['timestamp'] = datetime.now()
            db.train_data.insert_one(row)
        log.info('Indexing complete.')

    def aggregate_db_data(self,timestamp_from_model):
        client = self.get_client()
        db = client.otc
        train_data = db.train_data
        result = train_data.aggregate([{'$addFields': {'datestr': {'$dateToString': {'format': "%Y-%m-%d", 'date': \
            "$timestamp"}}}}, {'$group': {'_id': "$datestr", 'n': {'$sum' : 1}}}])

        # columns = ['Date', 'Count']
        date_array = []
        count_array = []
        for document in result:
            date_array.append(document['_id'])
            count_array.append(document['n'])
        d = {'Date' : date_array, 'Count' : count_array}
        df = pd.DataFrame(data=d)
        timestamp = timestamp_from_model
        filename_path = "db_summary/" + "summary_" + timestamp + ".csv"
        filename = "summary_" + timestamp + ".csv"

        df.to_csv(filename_path)
        return filename_path, filename

    def purge_model(self, startDate, endDate):

        mongo_host = config.get('Mongo','HOST')
        mongo_port = config.getint('Mongo','PORT')
        client = MongoClient(mongo_host, mongo_port)
        db = client.otc
        train_data = db.train_data

        start = time.time()
        log.info('Purging model started.')

        train_data.remove({"timestamp":{'$gte':startDate, '$lte':endDate}})

        log.info('Purging model complete.')

        end = time.time()
        log.info("Purging time: " + str(end - start))




