# Analyze Disaster Response Messages

## Installations
 - NumPy
 - Pandas
 - Seaborn
 - Matplotlib
 - Sklearn
 - Flask

## Project Motivation
To analyze disaster messages data from [Appen](https://appen.com/) for quick classification of messages into categories, resulting in efficient responses from appropriate disaster relief agencies.

Examples of disaster categories include shelter, food, safety, etc.

## Files
### Dataset
- `data/disaster_messages.csv` - Dataset containing messages in string format
- `data/disaster_categories.csv` - Dataset which corresponds to `disaster_messages.csv` via `id`, and which contains labels for all 36 disaster categories

### Project Components
1. `data/process_data.py` - Extract Transfer Load (ETL) Pipeline
   - Loads the `messages` and `categories` datasets
   - Merges the two datasets
   - Cleans the data
   - Stores it in a SQLite database
2. `models/train_classifier.py` - Machine Learning (ML) Pipeline
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file

3. `app/run.py` - Flask app
   - Accepts message as an input and obtains classification of message into disaster categories
   - Provides data visualisation of probabilities of each individual disaster message

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Acknowledgements and references
- 