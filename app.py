from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, send_from_directory, send_file, jsonify
from errors import BadRequestException, NoOutputException, NoFieldsFoundException
from otc_model import otc_model
from datetime import datetime
from werkzeug import secure_filename
import pickle as pkl
import os
from configparser import SafeConfigParser
from logger import Logger
from mongoUtil import mongoUtil

#The Utilities class for Mongo Database
mongoUtil = mongoUtil()

#Logger for logging capabilities
log = Logger('main_app.log')
log.info('Starting Application')
config = SafeConfigParser()
log.info('Loading configurations from config.ini')
config.read('config.ini')
#Getting password and usernames from the config file.
passAdmin = config.get('Passwords','ADMIN_PASS')
passUser = config.get('Passwords','USER_PASS')
nameAdmin = config.get('Usernames','OTC_ADMIN')
nameUser = config.get('Usernames','OTC_USER')
#Getting information from config file
UPLOAD_FOLDER = config.get('UploadFolders','UPLOAD_FOLDER_PRED')
UPLOAD_FOLDER_TRAIN = config.get('UploadFolders','UPLOAD_FOLDER_TRAIN')
DOWNLOAD_FOLDER = config.get('DownloadFolders','DOWNLOAD_FOLDER')
DOWNLOAD_Summary_FOLDER = config.get('DownloadFolders','DOWNLOAD_Summary_FOLDER')
ALLOWED_EXT = set(['csv'])

log.info('Setting initial variables')
#Initializing flask variable here
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#Secret key is MANDATORY!
app.secret_key = os.urandom(12)
#Declaring model
global model
global temp_model
temp_model = otc_model()
accept = True
fname = 'pickled_2017.pkl'

#Checking to see if the pickle file exists
if os.path.isfile(fname):
    #Load the pickle file if it exists
    log.info('Pickle file detected, initializing loading sequence.')
    try:
        file_opened = open(fname, "rb")
        model = pkl.load(file_opened)
        file_opened.close()
    except:
        log.error('File cannot be pickled.')
    log.info('Pickle file loaded successfully')
else:
    #No pickle file found
    log.info('No pickle file found. Creating new otc_model class.')
    model = otc_model()
    trained = model.initial_train()
    try:
        #If data is found in the Mongo Database, then pickle
        if trained:
            file_pickle = open(fname,'wb')
            log.info('Saving model as a pickle file.')
            pickled = pkl.dump(model, file_pickle)
            file_pickle.close()
    except:
        log.error('File cannot be pickled.')
    log.info('Initial stage complete.')

#Error handlers for flask
@app.errorhandler(BadRequestException)
def _handle_bad_request(exception):
    response = jsonify({'message': exception.message})
    response.status_code = 400
    log.error('Error occurred: ' + response)
    return response

#Error handlers for flask
@app.errorhandler(NoFieldsFoundException)
@app.errorhandler(NoOutputException)
def _handle_no_results(exception):
    response = jsonify({'message': exception.message})
    response.status_code = 200
    log.error('Error occurred: ' + response)
    return response

#This will check to see if the file uploaded is a CSV
"""
Checks if the file uploaded by the user is CSV file.
"""
def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.',1)[1].lower() in ALLOWED_EXT # Splits the filename to check the extension of the file (csv)


#The login procedure UI and logic
@app.route('/login', methods=['GET','POST'])
def do_admin_login():
    try:
        log.info('Performing login sequence.')
        log.info(request.method + " detected.")
        if not session.get('logged_in') and request.method == 'POST':
            log.info('No user currently is logged in.')
            if request.form['password'] == passAdmin and request.form['username'] == nameAdmin:
                #Setting session variables to check which use is logged in
                session['logged_in'] = True
                session['admin'] = True
                session['predict'] = False
                log.info('Admin logged in. Redirecting to admin console page.')
                return redirect(url_for('index_train'))
            elif request.form['password'] == passUser and request.form['username'] == nameUser:
                #Setting session variables to check which use is logged in
                session['logged_in'] = True
                session['admin'] = False
                session['predict'] = True
                log.info('User logged in. Redirecting to use prediction page. Access limited.')
                return redirect(url_for('predict'))
            else:
                log.info('No user logged in, redirecting to logout page. Please log in.')
                return redirect(url_for('logout'))
        log.info('Rendering HTML template.')
    except:
        #Creating an error pop up alert in front end.
        errorString = 'Unknown error occurred.'
        log.error(errorString)
        session['errorOc'] = True
        errorString = session.get('errorInfo')
        session['errorInfo'] = errorString
        return redirect(url_for('logout'))
    return render_template('login.html')

#Log out procedure
@app.route("/logout")
def logout():
    log.info('User logged out.')
    session['logged_in'] = False
    session['admin'] = False
    if not session.get('logged_in'):
        log.info('Redirecting to login page.')
        return render_template('login.html')
    elif session.get('errorOc'):
        session['errorOc'] = False
        errorString = session.get('errorInfo')
        session['errorInfo'] = ''
        return render_template('login.html', errorOcurred='True', errorInfo=errorString)

#The root. It renders the login page
@app.route('/')
def home():
    log.info('Initial redirect')
    return redirect(url_for('do_admin_login'))

#The training loading page
@app.route('/training_in_progress', methods=['GET', 'POST'])
def load_Train(filename_load="nothing"):
    #Global variables are needed
    global data, newReport, currentReport
    try:
        log.info('Executing the loading screen.')
        #If the method is GET then we render the loading screen.
        if request.method != 'GET':
            filename_load = session['trainingFile']
            log.info('File name of the file being loaded.')
        if request.method == 'GET':
            log.info('Rendering loading screen.')
            return render_template('loading.html')
        #If the request method is POST then we proceed to training
        elif request.method == 'POST' and filename_load != 'nothing':
            log.info('Executing training sequence.')
            latest_report = ''
            try:
                #This is where training occurres.
                latest_report = temp_model.re_train(session['trainingFile'])
            except:
                #Error occurred
                errorString = 'Unable to generate a report based on the training data. Training data is invalid. Please upload another set of training data.'
                log.error(errorString)
                session['errorInfo'] = errorString
                session['errorOc'] = True
                return redirect(url_for('index_train'))
            previous_report = model.get_report()
            data = [previous_report,latest_report]

            #If the data is not empty then theres a previous model, if it is empty then we display no model trained  d
            if data[0] == None:
                currentReport = [{"No Model has been trained before": "This is the first training of the model."}]
            else:
                currentReport = data[0]
            #Setting new report to show on UI
            newReport = data[1]
            session['fileChosen'] = filename_load
            #This is changing the upload folder back to the temporary data folder
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
            log.info('Training complete.')
            return redirect(url_for('train_res'))
        else:
            return render_template('loading.html')
    except:
        errorString = 'Unknown error occurred.'
        log.error(errorString)
        return render_template('loading.html', errorOcurred='True', errorInfo=errorString)

#This renders the training page
@app.route('/admin', methods=['GET'])
def admin():
    log.info('Redirecting to admin console.')
    if not session.get('logged_in'):
        return render_template('login.html')
    elif session.get('admin'):
        return render_template('index_Train.html')

#This method will process the predictions
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    log.info('Currently on prediction page.')
    try:
        if not session.get('logged_in'):
            log.error('You are not logged in, please login again. Redirecting to login page.')
            return render_template('login.html')
        elif (session.get('predict') or session.get('admin')) and request.method == 'POST' and request.files['file'] and allowed_file(request.files['file'].filename):
            log.info('File acquired.')
            #This gets the file from the request.
            file_to_process = request.files['file']
            #Parses the file to a secure file for backend
            filename = secure_filename(file_to_process.filename)
            log.info('File being processed: ' + filename)
            log.info('Saving file to location: ' + app.config['UPLOAD_FOLDER'])
            #Saves the file to the upload folder for processing
            file_to_process.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            log.info('File saved successfully.')
            #This will download the file
            return download(model.predict(file_to_process.filename))
        return render_template('index_Predict.html')
    except:
        errorString = 'Unable to process uploaded file. Please try again later or contact your administrator.'
        log.error(errorString)
        return render_template('index_Predict.html', errorOcurred='True', errorInfo=errorString)

#This method will download the file from the predicted folder once the backend is finished
@app.route('/download', methods=['GET'])
def download(filename_1):
    log.info('Downloading processed file.')
    try:
        if not session.get('logged_in'):
            log.error('You are not logged in, please login again. Redirecting to login page.')
            return render_template('login.html')
        else:
            abs_path = app.root_path + DOWNLOAD_FOLDER
            log.info('Acquiring processed file from ' + abs_path)
            #Try just using send from directory
            #This line gets the file
            return send_from_directory(abs_path, filename=filename_1, as_attachment=True)
    except:
        errorString = 'Unknown error occurred.'
        log.error(errorString)
        return render_template('predict_submission.html', errorOcurred='True', errorInfo=errorString)
    return render_template('predict_submission.html')


#This method process the file for training
@app.route("/train", methods=['GET', 'POST'])
def index_train():
    log.info('Training process method called.')
    try:
        if not session.get('logged_in'):
            log.info('You are not logged in, please login again. Redirecting to login page.')
            return render_template('login.html')
        elif session.get('errorOc'):
            session['errorOc'] = False
            errorString = session.get('errorInfo')
            session['errorInfo'] = ''
            return render_template('index_Train.html', errorOcurred='True', errorInfo=errorString)
        elif session.get('admin') and request.method == 'POST':
            file_to_train = request.files['file']
            #This checks if the file exists and it is the correct format
            if file_to_train and allowed_file(file_to_train.filename):
                log.info('File acquired. Beginning training sequence.')
                #This will secure the file for training
                filename = secure_filename(file_to_train.filename)
                app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_TRAIN
                #This saves the file to the path and prepares the file for the backend model
                file_to_train.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                log.info('Save file to local repository.')
                session['trainingFile'] = file_to_train.filename
                return redirect(url_for('load_Train'))
    except:
        errorString = 'Unable to process training data. Please try again later.'
        log.error(errorString)
        return render_template('index_Train.html', errorOcurred='True', errorInfo=errorString)
    return render_template('index_Train.html')

#This is a loading screen and this also indexs the data to Mongo
@app.route('/loading', methods=['GET', 'POST'])
def load_accept(filename_load="nothing"):
    log.info('Indexing to Mongo database.')
    try:
        if request.method != 'GET':
            filename_load = session['trainingFile']
        if request.method == 'GET':
            return render_template('accepted.html')
        elif request.method == 'POST' and filename_load != 'nothing':
            log.info('Executing saving sequence.')
            #This calls the utility class to index the data to database
            mongoUtil.index_to_db(session['fileChosen'])
            #this sets the current model to the newly trained model
            model.set_model(temp_model.get_model())
            #This sets the classification report of the latest model
            model.set_report(temp_model.get_report())
            model.set_training_frame(temp_model.get_training_frame())
            temp_model.reset_model()
            log.info('Saving updated model as a pickle file.')
            #This changes the log file to active model
            model.setActive()
            #These lines will pickle the newly trained model
            file_pickle = open(fname,'wb')
            pickled = pkl.dump(model,file_pickle)
            file_pickle.close()
            log.info('Saving complete. Redirecting to admin console.')
            return redirect(url_for('index_train'))
    except:
        errorString = 'Unable to complete model update process. Returning to training page.'
        log.error(errorString)
        session['errorOc'] = True
        session['errorInfo'] = errorString
        return redirect(url_for('index_train'))
    return render_template('accepted.html')


#This page displays the result of training
@app.route('/train_result', methods=['GET','POST'])
def train_res():
    log.info('Training results rendered.')
    try:
        if session.get('errorOc'):
            errorString = session.get('errorInfo')
            session['errorInfo'] = ''
            session['errorOc'] = False
            return render_template('Train_result.html', newjsonTable=newReport, currentjsonTable=currentReport, errorOcurred='True', errorInfo=errorString)
        if request.method == 'POST':
            #These statements check to see the decision of the user
            if request.form['accept'] == 'Accept Changes':
                log.info('Changes accepted.')
                return redirect(url_for('load_accept'))
            elif request.form['accept'] == 'Discard Changes':
                #If the user discards the changes, this line will reset the temporary model
                temp_model.reset_model()
                log.info('Changes rejected. Discarding new model.')
                return redirect(url_for('index_train'))
            else:
                return render_template('Train_result.html', newjsonTable=newReport, currentjsonTable=currentReport)
    except:
        errorString = 'Unable to gather model training results. Returning to training page.'
        log.error(errorString)
        session['errorOc'] = True
        session['errorInfo'] = errorString
        return render_template('Train_result.html', newjsonTable=newReport, currentjsonTable=currentReport, errorOcurred='True', errorInfo=errorString)
    return render_template('Train_result.html', newjsonTable=newReport, currentjsonTable=currentReport)


#This will download the information of the mongo db into a csv
@app.route("/download_db_info", methods=['GET'])
def download_db_info():
    if session.get('logged_in') and session.get('admin'):
        log.info('Downloading database information.')
        #This is the Mongo query
        abs_path, filename_2 = mongoUtil.aggregate_db_data(model.get_timestamp())
        log.info('Database information downloaded successfully.')
        log.info('Acquiring processed file from ' + abs_path)
        abs_path = app.root_path + DOWNLOAD_Summary_FOLDER
        return send_from_directory(abs_path, filename=filename_2, as_attachment=True)
    return render_template('index_Train.html')


@app.route("/modify_model")
def modify_model():
    if session.get('logged_in') and session.get('admin'):
        log.info('Rendering purging interface.')
        return redirect(url_for('purge_model'))

#This method purges the data from the data base
@app.route("/purge_model", methods=['GET','POST'])
def purge_model():
    try:
        log.info('Model data is being purged.')
        #These lines get the training data from the database
        client = mongoUtil.get_client()
        db = client.otc
        train_data = db.train_data
        if request.method == 'POST':
            log.info('POST: waiting for purge time frame.')
            start_date = request.values["Start Date"]
            end_date = request.values["End Date"]
            #This handles the incorrect data error
            if (end_date < start_date) or ((start_date == "") or (end_date == "")):
                errorString = 'Invalid dates detected. Please select both start and end date.'
                log.error(errorString)
                session['errorOc'] = True
                session['errorInfo'] = errorString
                return redirect(url_for('purge_result'))
            else:
                #These lines parse the date strings to ISO date format
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                session['start'] = start_date
                session['end'] = end_date
            log.info('Purge start date: ' + str(start_date) + '. Purge end date: ' + str(end_date))
            #These checks if the data is all purged. If it is we do not need to train the model again
            if train_data.count({"timestamp": {'$gte': start_date, '$lte': end_date}}) == 0:
                return redirect(url_for('purge_result'))
            if train_data.count() != 0:
                return redirect(url_for('load_purging'))
        return render_template('modify_model.html')
    except:
        errorString = 'Unknown error occurred.'
        log.error(errorString)

#This is the method that invokes backend for purging
@app.route('/purging_data', methods=['GET', 'POST'])
def load_purging():
    if request.method == 'POST':
        mongoUtil.purge_model(session['start'],session['end'])
        #This line trains the model on the new dataset without the purged data.
        model.initial_train()
        log.info('Model training on purged model begun.')
        client = mongoUtil.get_client()
        db = client.otc
        train_data = db.train_data
        if train_data.count() == 0:
            model.current_report = None
            model.reset_model()
        log.info('Model training on purged model completed.')
        return redirect(url_for('purge_result'))
    return render_template('loading_purge.html')

#this method displays the purge result
@app.route("/purge_result", methods=['GET','POST'])
def purge_result():
    if session.get('errorOc'):
            errorString = session.get('errorInfo')
            session['errorInfo'] = ''
            session['errorOc'] = False
            return render_template('index_train.html', errorOcurred='True', errorInfo=errorString)
    if session.get('logged_in') and session.get('admin'):
        log.info('Purge classification report generating.')
        currentReport = model.get_report()
        if currentReport is None:
            currentReport = [{"Database is empty after the purge": "Please train the model again."}]
        log.info('Purge classification report generated and redirecting to result page.')
        return render_template('Purge_result.html', currentjsonTable=currentReport)
