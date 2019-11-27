# import libraries
from sqlalchemy import create_engine
import sqlite3
import pandas as pd
import re 


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


#nltk.download('punkt', 'wordnet', 'stopwords')
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import sys

# function to load data from database as dataframe
def load_data(database_filepath):
    '''    
    Input: database_filepath: File path of sql database
    Output:
        X: Message data (features)
        y: Categories (target)
        col_names: Labels for 36 categories
    '''
    #Load data from database as dataframe
    engine = create_engine('sqlite:///'+database_filepath)
    #df = pd.DataFrame(pd.read_sql('SELECT * FROM InsertTableName', engine))
    df = pd.read_sql_table('disaster_texts', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    col_names = list(df.columns[4:])
    
    return X, y, col_names


#  function to tokenize and clean text
def tokenize(text):
     ''' 
    Input: text: the original disaster message text from dataframe
    Output: lemmed: list of text that has been Tokenized, cleaned, and lemmatized
    '''

    #Lower case normalization
    text = text.lower()     

    #Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)      

    #splits text into list of words
    words = word_tokenize(text)                  
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    #Chain lemmatization of nouns then verbs
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='n') for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    return lemmed

# function to build the ML pipeline using counterVector, ifidf, random forest, and gridsearch
def build_model():
    '''
    Input: None
    Output: cv_grid of the results of GridSearchCV
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']}

    cv_grid = GridSearchCV(pipeline, verbose=2, param_grid=parameters, cv=2, n_jobs=-1)

    return cv_grid


# function to evaluate model performance using test data
def evaluate_model(model, X_test, y_test, category_names):
    '''
    Input: 
        model: The model to be evaluated
        X_test: The test data of the features
        y_test: The true labels for Test data from split dataset
        category_names: The labels for all 36 categories
    Output:
        Print of accuracy score and classfication report for each category
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('Accuracy of {:<25}: {:>2}%'.format(category_names[i], 
                                                 round(accuracy_score(y_test.iloc[:, i].values, y_pred[:,i]),3)))

    for i in range(len(category_names)):
        print("Category: {:<25} \n {}".format(category_names[i], 
                                              classification_report(y_test.iloc[:, i].values, y_pred[:, i])))    
        
# function to save model as a pickle file         
def save_model(model, model_filepath):
    '''
    Input: 
        model: The model to save
        model_filepath: path of the output pick file
    Output: A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db model_classifier.pkl')


if __name__ == '__main__':
    main()