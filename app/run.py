import json
import plotly
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')#create_engine('sqlite:///data/DisasterResponse.db') #
df = pd.read_sql_table('categorized_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")#joblib.load("models/classifier.pkl") #

Y = df.drop(['message', 'id', 'original','genre'],axis=1)
category_names = list(Y.columns)




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    #prepare data for genres bar plot
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #prepare data for categories bar plot
    Y = df.drop(['message', 'id', 'original','genre'],axis=1) #select category subset
    category_names = list(Y.columns)
    category_counts = []
    for cat_name in category_names:
        category_counts.append(Y[cat_name].sum())
    #save category counts into a pandas series
    cat_s = pd.Series(index = pd.Index(category_names, name = 'category'), data = category_counts)
    cat_s = cat_s.sort_values(ascending = False) #order series descending
    cat_s.index = cat_s.index.str.replace('_', ' ') #replace _ with white space

    #prepare data for pairing categories heatmap
    df_groupcounts = pd.DataFrame(columns = category_names, index = category_names) #instantiate empty dataframe
    for cat_name in category_names: #iterate over categories
        Y_cat = Y[Y[cat_name] == 1] #select subset labelled with category
        #count other categories and normalize it with categorie's length
        df_groupcounts[cat_name] = Y_cat.sum() / Y_cat.shape[0] 
        df_groupcounts.loc[cat_name,cat_name] = 0 #category can not be paired with self --> set to 0
    #replace _ with white space
    df_groupcounts.index = df_groupcounts.index.str.replace('_', ' ')
    df_groupcounts.columns = df_groupcounts.columns.str.replace('_', ' ')
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_s.index,
                    y=cat_s.values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=df_groupcounts.values,
                    x=df_groupcounts.columns,
                    y =df_groupcounts.index
                )
            ],

            'layout': {
                'width': 1100,
                'height': 700,
                'title': 'Probability of X-Category to be with Y-Category',
                'yaxis': {
                    'title': "Y-Category",
                    'tickangle': -45
                },
                'xaxis': {
                    'title': "X-Category",
                    'tickangle': 20
                },
                'colorbar': {
                    'title': "Count"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()