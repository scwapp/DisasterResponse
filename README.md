# Disaster Response Pipeline Project

This project aims to classify emergency messages into 36 different categories using a machine learning
algorithm to facilitate the professional response in a disaster situation. The training dataset was provided by
[Figure Eight](http://s33181.p1443.sites.pressdns.com/).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. The message `Running on http://192.168.0.168:3000/ (Press CTRL+C to quit)` will appear on the terminal.
    CTRL+Click on the http address to preview the website.
<!-- Click the `PREVIEW` button to open the homepage -->

### Files
- `data/process_data.py`: ETL pipeline that reads message and categories datasets, cleans data and saves it
    to an sqlite database. It takes 3 command line arguments \{ __input path to message dataset_, 
    __input path to categories dataset_, _output path to sqlite database_ \} .
- `models/train_classifier.py`: machine learning pipeline that uses the message column to classify for 
    36 categories. It takes 2 command line arguments \{ _input path to sqlite database_, 
    _output path to  model_ \} .
- `app/run.py`: Flask web app.
- `templates/*`: html files for each web page.

### Comments
The dataset we are working with is imbalanced, which means that there are some categoreies that
have fewer samples than others. This means that it is more difficult for the model to learn
about these under-represented categories, and it will "under predict" them. In disaster response classification 
it is better to emphasize higher recall (predicting all positives) over higher precision 
(all the predicted positives are true). That is because we prefer to have a false positive (assigning
a category to a message that turned out to be incorrect) than a false negative (not assigning a true
category to a message), since the latter might make us miss crucial information.
