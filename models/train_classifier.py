import sys
import pandas as pd
import re
from sqlalchemy import create_engine
from sqlalchemy.engine import reflection
import time
import pickle

#NLTK

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


#Sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, recall_score, \
    precision_score, accuracy_score

# uncomment below to download nltk data. Can be done only once.
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def elapsed_time(func):
    '''Decorator that computes the running time of the input function `func`.
    '''
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"Elapsed time: {elapsed:.2f} seconds")
        return result
    return wrapper

def load_data(database_filepath):
    '''Loads data from the sqlite database `database_filepath`.
    Inputs:
        - `database_filepath`: sqlite database file path
    Outputs:
        - X: messages (pandas series)
        - Y: categories (pandas dataframe)
        - category_names: list of category names
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('categorized_messages',con=engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original','genre'],axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    ''' Tokenizes text by replacing urls with placehoders, tokenizing, normalizing and lemmatizing text,
    and removing stopwords.
    Inputs:
        - text: input text (list of strings)
    Outputs:
        - clean_tokens: tokenized text (list of strings)

    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens: 
        if tok not in stopwords.words("english"): #do not include stopwords
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    ''' Build 3 step machine learning pipeline (1- Vectorize, 2-TF-IDF transform, 3-MultiOutput classifier
    with LinearSVC as base classifier.)
    Outputs:
        - model: machine learning model
    '''
    base_classifier = LinearSVC(random_state=54, class_weight='balanced')
    #metrics of above model: mean_f1: 0.4462, mean_accuracy: 0.931 
    #class_weight='balanced' is not a default parameter of LinearSVC, but was
    #found to give higher f1-score through gridsearch.

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(base_classifier))
    ])
    ######### Use gridsearch (optional)
    #Tune base classifier parameters to obtain better performing model.
    #The below commented parameters have already been tested
    parameters = {'mclf__estimator__C': [0.1, 1, 10]
     }
     # 'mclf__estimator__C': [0.1, 1, 10]  #C=1 is better     
     # 'mclf__estimator__dual': [True,False] #dual = True is better
     # 'mclf__estimator__class_weight': [None, 'balanced']  #'balanced' is better
    #use scoring f1 with the averaging "weighted", which calculates f1 for each label,
    # and returns the average, weighted by support (the number of true instances for each label)
    grid_search = GridSearchCV(pipeline, param_grid=parameters, scoring = 'f1_weighted')
    #########
    model = pipeline #use model = grid_search if you are using gridsearch
    return model

@elapsed_time
def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluate model by outputing evaluation metrics for each category, and mean evaluation metrics.
    Inputs:
        - model: machine learning model
        - X_test: message test dataset
        - Y_test: categories test dataset
        - category_names: category names
    '''
    y_pred = model.predict(X_test) #predict categories
    metrics_df = pd.DataFrame(columns = ['f1', 'recall', 'precision','accuracy']) #instantiate empty df
    #iterate over categories
    for i, (col_name, col_values) in enumerate(Y_test.items()):
        #Compute and print different evaluation metrics
        report = classification_report(Y_test[col_name], y_pred[:,i], zero_division=0)                                       
        metrics_df.loc[col_name,'f1'] = f1_score(Y_test[col_name], y_pred[:,i], zero_division = 0)
        metrics_df.loc[col_name,'recall'] = recall_score(Y_test[col_name], y_pred[:,i],zero_division=0)
        metrics_df.loc[col_name,'precision'] = precision_score(Y_test[col_name], y_pred[:,i], zero_division = 0)
        metrics_df.loc[col_name,'accuracy'] = accuracy_score(Y_test[col_name], y_pred[:,i])
        print(col_name)
        print(report)
        print()
    print(f'mean_f1: {metrics_df["f1"].mean()}, mean_accuracy: {metrics_df["accuracy"].mean()}')
    return 


def save_model(model, model_filepath):
    '''Save the model to a pickle file.'''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        #wrap function with elapsed_time decorator to print time
        elapsed_time(model.fit)(X_train, Y_train)

        # Uncomment below when using gridsearch
        # print('best params: ',model.best_params_)
        # print('best score: ',model.best_score_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()