# Disaster Response Pipeline Project
The goal of this project is to predict whether a message is related to disaster response and which categories of disasters.
Trainded data is given by Figure 8 corp which has about 20,000 messages and labels of 36 categories.
The script builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. 

# Files in the repository.
- disaster_reponse_pipeline_project
    - app
        - templates
            - go.html
            - master.html
    - data
        - disaster_categories.csv
        - disaster_messages.csv
        - DisasterResponse.db
        - process_data.py
    - models
        - train_classifier.py
        - classifier.pkl
    - README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
