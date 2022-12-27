# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Comments
The dataset we are working with is imbalanced, which means that there are some categoreies that
have fewer samples than others. This means that it is more difficult for the model to learn
about these under-represented categories, and it will "under predict" them. In disaster response classification 
it is better to emphasize higher recall (predicting all positives) over higher precision 
(all the predicted positives are true). That is because we prefer to have a false positive (assigning
a category to a message that turned out to be incorrect) than a false negative (not assigning a true
category to a message), since the latter might make us miss crucial information.
