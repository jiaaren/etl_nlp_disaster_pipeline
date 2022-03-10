import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
 
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
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

    # category labels
    cat_labels = df.columns[4:]
    
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(cat_labels, classification_labels))
    classification_proba = model.predict_proba([query])
    proba_arr = []
    for x in classification_proba:
        try:
            proba_arr.append(x[0][1])
        except:
            proba_arr.append(0)
    proba_dict = dict(zip(cat_labels, proba_arr))
    
    # create visuals
    graphs = [ 
        {
            'data': [
                Bar(
                    x=list(proba_dict.keys()),
                    y=list(proba_dict.values())
                ) 
            ],
            'layout': {
                'title': 'Probability of each disaster message category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html', 
        query=query,
        classification_result=classification_results,
        ids=ids,
        graphJSON=graphJSON
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()