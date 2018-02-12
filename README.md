# OTC-Breaks

Web interface with model that identifies break owners in OTC Reconciliations.


## Setup

### Installation
Access CIBC Data Studio - otc-breaks Git repository to pull all the relevent application codes and files.

First, create a folder to clone the repository.

Then run this:

1) Clone the git repository
```shell
Not there
```

2) Check status to see if the repo have succesfully been cloned.
```shell
git status
```

3) Go into the otc-breaks folder
```shell
cd otc-breaks
```

#### Install dependencies

To install Python dependencies, run:

```shell
pip install -r requirements.txt
```


#### Running the application

Inside the otc-breaks folder from the git repo, run the following:
```shell
python ./run.py
```


### Running the server

Go to http://40.84.53.8:5000/ for OTC Application.

## Website Users

The website has two setup users admin and normal user. The admin user is allowed to train new data and add it to the database. Also the admin can purge old data and access the database summary.


## Code Setup

### Config File Description
<p>[UploadFolders]: These are the names of the temporary folders that handles the uploaded files from the website.<br>

[DownloadFolders]: These are the names of the temporary folders that handles the downloaded files from the website.<br>

[PickleFileName]: The name of the otc-model class pickle file.<br>

[Usernames]: This tag holds the user names for the website. (Both admin and user access)<br>

[Passwords]: This tag holds the passwords for the website. (Both admin and user access)<br>

[Mongo]: This tag shows the host IP and PORT of access.<br>

[NeuralNetConfig]: This tag sets the parameters of the neural network structure.<br>

[Misc]: This tag sets the additional parameters used in the machine learning model.<br>

### Classes used
<ul>
<li> otc_model - Holds all methods used by the machine learning model class and the model.
<li> mongoUtil - Holds methods which interacts with the database including indexing new data and purging data.
</ul>

### Machine Learning Model Used

The machine learning model deployed is a 4-layer neural network with hidden layer sizes 150,300,300,50 resepectively. The hidden layer size can be changed by editing the config.ini file.

### Database 

The database used to store data is MongoDB. Its IP and Port configuration can be seen on the config file. <br>
The db name is otc. In otc there is a collection called train_data where all the current training data is stored.

