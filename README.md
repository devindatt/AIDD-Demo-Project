# Project: Figure8 Disaster Response Pipeline
![N|Solid](https://awsmp-logos.s3.amazonaws.com/8cb4fa6a-a4d1-4f05-a305-18ca1bc32a53/2fc3991a7f3442b2761adf59a702e25f.png)
![Build Status](https://sites.uci.edu/emergencymanagement/files/2017/02/disastercollage.png)

Machine Learning ETL pipeline for Figure8 disaster response classification modelling and web app
## Project Introduction

In this project we are assigned to analyze disaster data from Figure Eight and build a model that classifies disaster messages. The project objective is to combine these three steps (ETL Pipeline, Machine learning Pipeline, and web application) that uses a trained model and classify any messages.

This project is designed using a real messages that were sent during disaster events, so this model can categorize these events to send the messages to an appropriate disaster relief agency.
## Installation

Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk:

- punkt
- wordnet
- stopwords


1) Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
- To run ML pipeline that trains classifier and saves ```python    models/train_classifier.py data/DisasterResponse1.db models/model_classifier.pkl```
2) Run the following command in the app's directory to run your web app. ``` run.py```

3) Go to http://0.0.0.0:3001/ to view the web application, use 'env|grep WORK' command to find the exact space ID to add to your , as anexample that would be: 
"https://viewa7a4999b-3001.udacity-student-workspaces.com/"
![grep_image](https://github.com/devindatt/Disaster-Response-Pipeline/blob/master/assets/spaceID_grep_command.png)


## Steps To Complete:
1) Choose the source and destination filepaths
2) Run each step of the pipeline (ETL and then ML)


## File Descriptions:
| File | Description |
| ------ | ------ |
|process_data.py | ETL Pipeline In a Python script (Load datasets, Merge datasets, Clean the data, and Store the data in a SQLite database)|
|train_classifier.py | Machine Learning Pipeline In a Python script (Load data from SQLite database, Splits data into training and test sets, Builds a text processing and machine learning pipeline, Trains and tunes a model using GridSearchCV, Outputs results on the test set, and Exports the final model as a pickle file)|
|Run.py | Flask Web App (display visualization from the datasets, the app accept messages from users and returns classification results for 36 categories of disaster events)|

## Images of Application:

###### Training Model Run:
![Training Model](https://github.com/devindatt/Disaster-Response-Pipeline/blob/master/assets/model_training_snapshot.png)

###### Accuracy Score Run:
![Accuracy Score](https://github.com/devindatt/Disaster-Response-Pipeline/blob/master/assets/accuracy_score_run1.png)

###### Classification Report Run:
![Classification Report](https://github.com/devindatt/Disaster-Response-Pipeline/blob/master/assets/classification_report_score1.png)

###### Flask Application:
![Flask Image1](https://github.com/devindatt/Disaster-Response-Pipeline/blob/master/assets/flask_disaster-response_pic4.png)

![Flask Image2](https://github.com/devindatt/Disaster-Response-Pipeline/blob/master/assets/flask_disaster-response_pic3.png)


## Resources
- [Python Flask Tutorial](https://www.youtube.com/watch?v=MwZwr5Tvyxo)
- [Sklearn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Sklearn Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
# AIDD-Demo-Project
