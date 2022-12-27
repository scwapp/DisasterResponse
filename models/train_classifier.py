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

# import warnings
# warnings.filterwarnings("error")

def elapsed_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"Elapsed time: {elapsed:.2f} seconds")
        return result
    return wrapper

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('categorized_messages',con=engine)
    # df = df.iloc[:int(df.shape[0]*0.1)]
    X = df['message']#,'genre'
    Y = df.drop(['message', 'id', 'original','genre'],axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
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
    base_classifier = LinearSVC(random_state=54, class_weight='balanced')
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(base_classifier))
    ])
    parameters = {'mclf__estimator__C': [0.1, 1, 10]
     }
     # 'mclf__estimator__C': [0.1, 1, 10]  #C=1 is better     
     # 'mclf__estimator__dual': [True,False] #dual = True is better
     # 'mclf__estimator__class_weight': [None, 'balanced']  #'balanced' is better
    #use scoring f1 with the averaging "weighted", which calculates f1 for each label,
    # and returns the average, weighted by support (the number of true instances for each label)
    grid_search = GridSearchCV(pipeline, param_grid=parameters, scoring = 'f1_weighted')
    model = pipeline
    return model

@elapsed_time
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    metrics_df = pd.DataFrame(columns = ['f1', 'recall', 'precision','accuracy'])
    for i, (col_name, col_values) in enumerate(Y_test.items()):
        report = classification_report(Y_test[col_name], y_pred[:,i], zero_division=0)
                                       #output_dict = True)
        metrics_df.loc[col_name,'f1'] = f1_score(Y_test[col_name], y_pred[:,i], zero_division = 0)
        metrics_df.loc[col_name,'recall'] = recall_score(Y_test[col_name], y_pred[:,i],zero_division=0)
        metrics_df.loc[col_name,'precision'] = precision_score(Y_test[col_name], y_pred[:,i], zero_division = 0)
        metrics_df.loc[col_name,'accuracy'] = accuracy_score(Y_test[col_name], y_pred[:,i])
    print(f'mean_f1: {metrics_df["f1"].mean()}, mean_accuracy: {metrics_df["accuracy"].mean()}')
    return metrics_df


def save_model(model, model_filepath):
    '''Save the model to a file'''
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
        # Print the best parameters and score
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