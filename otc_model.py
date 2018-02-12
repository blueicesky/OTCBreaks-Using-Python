from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from pymongo import MongoClient
from imblearn.over_sampling import SMOTE
from datetime import datetime
from collections import Counter
from math import factorial
from sklearn.externals import joblib
from configparser import SafeConfigParser
from logger import Logger
import pandas as pd
import numpy as np
import csv
import time
import sys
from mongoUtil import mongoUtil

config = SafeConfigParser()
log = Logger('model.log')
config.read('config.ini')
sys.setrecursionlimit(10000)
mongoUtil = mongoUtil()


hidden_1 = config.getint('NeuralNetConfig','HIDDEN_1')
hidden_2 = config.getint('NeuralNetConfig','HIDDEN_2')
hidden_3 = config.getint('NeuralNetConfig','HIDDEN_3')
hidden_4 = config.getint('NeuralNetConfig','HIDDEN_4')

t_size = config.getfloat('Misc','TEST_SIZE_1')
t_size_2 = config.getfloat('Misc','TEST_SIZE_2')
rstate_2 = config.getint('Misc','RSTATE_1')
digit_num = config.getint('Misc','DIGIT_NUM')

rstate = config.getint('NeuralNetConfig','RSTATE')
activationFunc = config.get('NeuralNetConfig','ACTIVATION')
m_iter =config.getint('NeuralNetConfig','MAX_ITER')
l_rate = config.getfloat('NeuralNetConfig','LEARNING_RATE')
ver_bose = config.getboolean('NeuralNetConfig','VERBOSE')
class otc_model:

    def __init__(self,model_type=1,model_params={'n_estimators':200,'max_depth':10,'max_features':0.4}):
        self.break_reason_mapping = {
            1:1,
            2:2,
            3:3,
            4:4,
            5:2,
            6:4,
            7:5,
            8:2,
            9:4,
            10:5,
            11:4,
            12:2,
            13:2,
            14:6,
            15:2,
            16:2,
            17:4,
            18:4,
            19:7,
            20:8,
            21:4
        }

        #get config values
        #get config values
        log.info('Initializing OTC Model.')
        self.isActive = False
        # self.training_data  = None
        self.unique_list = None
        self.training_frame = None
        self.date_trained = None
        self.current_report = None
        if model_type == 1:
            log.info('Initializing Neural Network.')
            self.model =  MLPClassifier(solver='adam', \
                alpha=config.getfloat('NeuralNetConfig','ALPHA'),\
                hidden_layer_sizes=(hidden_1, hidden_2, hidden_3, hidden_4), \
                random_state=rstate, activation=activationFunc, max_iter=m_iter, learning_rate_init=l_rate, verbose=ver_bose)
        else:
            log.info('No model selected.')

    def setActive(self):
        self.isActive = True
        log.info('------------------ Changed Active Model -------------------')    

    def get_info(self):
        print(self.training_frame)
        log.info('INFO requested.')


    def fit_to_data(self):
        # Grab data and target from database
        log.info('Fitting model via manual fit function call.')
        self.model.fit(self.data,self.target)
        # Update date

    def get_break_owner(self,break_reason):
        log.info('Break owner mapping requested.')
        return self.break_reason_mapping[break_reason]

    def flag(self, df):

        df_new = df
        df_new['FLAG'] = None

        client = mongoUtil.get_client()
        db = client.otc

        train_data = pd.DataFrame(list(db.train_data.find({})))

        d = {}
        for col in train_data.columns:
            d[col] = dict(enumerate(train_data[col].unique().flatten(), 1))

        # starttime = time()
        for index, row in df_new.iterrows():
            for col in train_data.columns.values:
                if (row[col]) not in set(d[col].values()) and col not in \
                        ['NOTIONAL', 'NOTIONAL_2', 'STRIKE_PRICE', 'DIFF_AGE', '_id', 'timestamp']:
                    df_new.loc[index, 'FLAG'] = 'x'
                    break

        return df_new

    def one_hot(self, dataframe):
        df_in = dataframe
        log.info('One hot function performed.')
        to_drop=open('drop.txt')
        lst=[]
        for line in to_drop:
            lst.append([(x) for x in line.split()])
        flat=[item for sublist in lst for item in sublist]
        to_drop=[x for x in df_in if x in flat] # list of labels to drop

        df_in.drop(labels=to_drop, axis=1, inplace=True) # drop labels from df
        dum_lst=list(df_in.columns) # list of all columns left
        dum_lst.remove('NOTIONAL') # remove continous variables from list to specify dummy variables
        dum_lst.remove('NOTIONAL2')
        dum_lst.remove('STRIKE_PRICE')
        dum_lst.remove('DIFF_AGE')


        df_one_hot=pd.get_dummies(data=df_in, columns=dum_lst) # transform dataframe
        #Drop duplicates
        df_one_hot = df_one_hot.T.drop_duplicates(keep='first').T
        log.info('One hot complete.')
        return df_one_hot

        # For this method please pass in the one_hot encoded version of the data without break owner
    def initial_train(self):
        log.info('Performing initial training of the model.')
        client = mongoUtil.get_client()
        db = client.otc
        train_data = db.train_data
        if_trained = False
        log.info('Established connection to Mongo Database.')
        old_data_x_frame = pd.DataFrame(list(train_data.find()))
        if not old_data_x_frame.empty:
            if_trained = True
            log.info('Initial training sequence started.')
            old_data_y_frame = old_data_x_frame['Break Owner'].astype(str)
            old_data_x_frame.drop('_id', axis=1, inplace=True)
            old_data_x_frame.drop('timestamp', axis=1, inplace=True)
            old_data_x_frame.drop('Break Owner', axis=1, inplace=True)
            old_data_x_frame.drop('Break Reason', axis=1, inplace=True)
            # self.training_data = old_data_x_frame
            train_x = self.one_hot(old_data_x_frame)
            self.training_frame = train_x.columns.values
            train_y = list(old_data_y_frame)
            self.current_report = self.execute_train_report(train_x, train_y)
        else:
            log.info('Cannot find training data in Mongo Database. Abort initial training sequence.')

        return if_trained

    def set_model(self,model_param):
        self.model = model_param
        log.info('Setting model via function call.')

    def get_model(self):
        log.info('Getting model instance via function call.')
        return self.model

    def set_report(self,report_param):
        self.current_report = report_param
        log.info('Setting classification report via function call.')

    def get_report(self):
        log.info('Getting current classification report.')
        return self.current_report

    def set_training_frame(self,training_frame):
        self.training_frame = training_frame
        # self.training_data = training_data
        log.info('Setting training frame via function call.')

    def get_training_frame(self):
        log.info('Getting training frame via function call.')
        return self.training_frame


    def reset_model(self):
        log.info('Resetting current model via function call.')
        self.model =  MLPClassifier(solver='adam', \
                alpha=config.getfloat('NeuralNetConfig','ALPHA'),\
                hidden_layer_sizes=(hidden_1, hidden_2, hidden_3, hidden_4), \
                random_state=rstate, activation=activationFunc, max_iter=m_iter, learning_rate_init=l_rate, verbose=ver_bose)

    def re_train(self,filename):
        log.info('Training started.')
        start = time.time()
        # Grab new data from data folder
        csv_path = 'train_data/'+str(filename)
        df = pd.read_csv(csv_path)
        new_data_x_frame = df.drop('Break Owner', axis=1).drop('Break Reason', axis = 1)

        new_data_y_frame = df['Break Owner'].astype(str)
        new_data_x = new_data_x_frame.values
        new_data_y = df[['Break Owner']].values

        # Grab the original training data from the database
        log.info('Establishing connection to Mongo Database.')
        client = mongoUtil.get_client()
        db = client.otc
        train_data = db.train_data

        log.info('Connection established.')
        old_data_x_frame = pd.DataFrame(list(train_data.find()))
        # print old_data_x.columns
        classification_report_var = None
        return_json = None

        if not old_data_x_frame.empty:
            # old_data_x = self.one_hot(old_data_x_frame)
            log.info('Previous training data was found.')
            old_data_y_frame = old_data_x_frame['Break Owner'].astype(str)
            old_data_x_frame.drop('_id', axis=1, inplace=True)
            old_data_x_frame.drop('timestamp', axis=1, inplace=True)
            old_data_x_frame.drop('Break Owner', axis=1, inplace=True)
            old_data_x_frame.drop('Break Reason', axis=1, inplace=True)

            a = new_data_x_frame.columns
            b = old_data_x_frame.columns
            log.info('Performing data clean up.')
            # This set is the columns that the unanalyzed data doesn't have
            new_set = set(b) - set(a)
            # This set is the columns that the training data doesn't have
            new_set_2 = set(a) - set(b)

            # Remove the columns that the training data doesn't have
            for col in new_set_2:
                new_data_x_frame = new_data_x_frame.drop(col,axis=1)

            # Adding columns of 0 for the columns the unanalyzed data doesn't have but the training data does
            for col in new_set:
                new_data_x_frame[col] = 0
            train_x = [old_data_x_frame,new_data_x_frame]
            log.info('Merging previous data with new data.')
            # Merge them together

            train_x = pd.concat(train_x, ignore_index=True)
            # Grab the original target data from the database
            train_y = [old_data_y_frame,new_data_y_frame]

            # Merge them together
            train_y = pd.concat(train_y, ignore_index=True)
            train_y = list(train_y)

            # self.training_data = train_x
            train_x = self.one_hot(train_x)
            self.training_frame = train_x.columns.values
            self.current_report = self.execute_train_report(train_x, train_y)
        else:
            log.info('No prevous training data found.')
            # self.training_data = new_data_x_frame
            new_data_x_frame = self.one_hot(new_data_x_frame)
            self.training_frame = new_data_x_frame.columns.values

            new_data_x = new_data_x_frame.values 
            self.current_report = self.execute_train_report(new_data_x, new_data_y, test_size=t_size_2,random_state=rstate_2)
        end = time.time()
        log.info('Total training time: ' + str(end - start))
        return self.current_report

    def sample_data(self, X_,y_, numNeighbours = 2, samplingRatio = 1):
        '''Method to oversample data based on maximum
        number of selections possible from available samples
        samplingRatio: ratio to maximum class count to be considered for oversampling
        '''
        unique, counts = np.unique(y_, return_counts=True)
        classCount = dict(zip(unique, counts))
        log.info('Data oversampling sequence initialized.')
        #classCount = Counter(y_)
        maxCount = max(classCount.values())
        samplingMap = dict()

        for c, cnt in classCount.items():
            #For each class, estimate additional samples
            if cnt < (samplingRatio * maxCount) and cnt >= numNeighbours:
                potentialSamples = max(cnt,int(factorial(cnt)/factorial(cnt - numNeighbours)/factorial(numNeighbours))) #Number of ways to select number of beighbours from existing data
                samplingMap[c] = min(potentialSamples, maxCount)

        sampler = SMOTE(ratio = samplingMap, k_neighbors= numNeighbours)
        X_sampled, y_sampled = sampler.fit_sample(X_, y_)

        return X_sampled, y_sampled

    def predict(self,filename):
        log.info('Prediction sequence started.')
        start = time.time()
        csv_path = 'temp_predict/'+str(filename)
        df = pd.DataFrame.from_csv(csv_path, index_col=None)
        if 'Break Owner' in df:
            df = df.drop('Break Owner', axis=1)
        if 'Break Reason' in df:
            df = df.drop('Break Reason', axis=1)
        log.info('Performing one hot function.')
        df = self.one_hot(df)


        predict_list_columns = df.columns.values
        training_list_columns = self.training_frame

        # This set is the columns that the unanalyzed data doesn't have
        new_set = set(training_list_columns) - set(predict_list_columns)


        # Adding columns of 0 for the columns the unanalyzed data doesn't have but the training data does
        for col in new_set:
            df[col] = 0

        # Remove the columns that the training data doesn't have

        df = df[self.training_frame]

        final_frame = self.model.predict_proba(df)

        # process predictions in the right format
        prediction = self.get_top_probabilities(final_frame)

        # get original file and append predictions
        original_frame = pd.DataFrame.from_csv(csv_path, index_col=None)
        results_df = pd.merge(prediction,original_frame, left_index=True, right_index=True)
        results_df = self.flag(results_df)

        # set filename according to timestamp and return filename
        timestamp = self.get_timestamp()
        filename_path = "temp_predicted/"+"predicted_" + timestamp + ".csv"
        filename = "predicted_" + timestamp + ".csv"
        results_df.to_csv(filename_path)
        log.info('Prediction done.')

        end = time.time()
        log.info('Prediction time: ' + str(end - start))
        return filename
        

    def get_timestamp(self):
        log.info('Getting time stamp.')
        # generate timestamp and convert it to string
        ts = time.time()
        ts = int(ts)
        ts = str(ts)
        return ts

    def get_top_probabilities(self, result_array):
        log.info('Getting top probabilities called.')
        numpy_arr = np.array(result_array)
        final_array_index = []
        final_array_value = []

        i=0
        for item in numpy_arr:
            temp_index = item.argsort()[-3:][::-1]
            final_array_index.append(temp_index)
            final_array_value.append(numpy_arr[i][temp_index])
            i += 1

        index_num = np.array(final_array_index)
        value_num = np.array(final_array_value)
        index_df = pd.DataFrame(index_num)
        value_df = pd.DataFrame(value_num)
        index_df.columns = ['Break Owner 1', "Break Owner 2", "Break Owner 3"]
        value_df.columns = ['Break Owner Probability 1', 'Break Owner Probability 2', 'Break Owner Probability 3']

        results_df = pd.merge(index_df,value_df, left_index=True, right_index=True)
        results_df['Break Owner 1'] = results_df['Break Owner 1'] + 1
        results_df['Break Owner 2'] = results_df['Break Owner 2'] + 1
        results_df['Break Owner 3'] = results_df['Break Owner 3'] + 1
        results_df = results_df[['Break Owner 1', 'Break Owner Probability 1', 'Break Owner 2', 'Break Owner Probability 2', 'Break Owner 3', 'Break Owner Probability 3']]
        return results_df

    def get_info_data(self):
        log.info('Getting info.')
        #Reach into the database for this information
        return self.data.info()

    def get_info_target(self):
        log.info('Getting target info.')
        #Reach into the database for this information
        return self.target.info()

    def change_training_data(self,data,target):
        log.info('Changing training data.')
        # Reach into the database for modifications
        self.data = data
        self.target = target

    def make_classification_json(self,report):
        log.info('Creating classification report in JSON.')
        classes = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split()
            row['class'] = float(row_data[0])
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            classes.append(row)
        temp_row_data = lines[-2].split()
        temp_row  = {'precision': temp_row_data[3], 'recall': temp_row_data[4], 'f1_score': temp_row_data[5],
                    'support': temp_row_data[6]}
        classes.append(temp_row)
        return classes

    def execute_train_report(self, X, y, test_size=t_size,random_state=rstate_2):
        #Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)
        #Oversample classes in data
        x_train_sample, y_train_sample = self.sample_data(X_train, y_train)

        # Fitting the model
        log.info('Fitting model with entire training data set.')
        self.model.fit(x_train_sample,y_train_sample)
        
        # Making predictions with X_test, the test data set.
        pred = self.model.predict(X_test)
        # print(classification_report(y_test,pred,digits=5))
        classification_report_var = classification_report(y_test,pred,digits=digit_num)
        current_report = self.make_classification_json(classification_report_var)
        x_final, y_final = self.sample_data(X,y)
        self.model.fit(x_final, y_final)
        log.info('Model training complete.')
        return current_report
