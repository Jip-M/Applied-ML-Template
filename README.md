# 🧛🏽‍♂️🩸 Bat Audio Classifier 🦇🔊

**Welcome to our project** 
This repository is a working pipeline that gets audio files, it transforms them conveniently and then are inserted in a CNN for the classification.

**Prerequisites**
In order to have a full functioning pipeline, there is a requirements.txt file that has all the needed dependencies. They have to be installed before running anything, otherwise this will result in errors. To install all the dependencies, you just have to write in the terminal: 
```
pip install -r requirements.txt
```
This script will let you automatically download all the dependencies.

**Run the code**
***Important!***
Before getting started and running the pipeline, it is important to mention that the pipeline will automatically download the data locally. This implies that you need quite an amount of free space (up to 10 GB) to be able to run the pipeline without problems.

To **run** the pipeline, you have to run this script on the APPLIED-ML-TEMPLATE integrated terminal:
'''
python -m pipeline
'''

As a result, you will see multiple checkpoints (print statements) throughout the run that highlight when the data is getting prepared, saved, start to train the models, each epoch update of the accuracy and the loss of the main model (CNN), confustion matrix and the average accuracy after 10 (set as default in the pipeline, in the initialize_CNN() function) epochs.


**How to run the API app**
In order to run the FastAPI, you have to open the integrated terminal from the folder called "api" and run this script:
'''
uvicorn api.main:app --reload
'''
After that, you have to enter this site in order to see it work:

http://127.0.0.1:8000/docs#/


On the site, you will see a **Prediction** section that will let you try the code out. To do this, you press on the section and press on the "Try it out" button. This will let you upload a ".wav" file. If you do not upload a file, there will be a built-in sample that you can use to judge the model's performance. When you selected your choice, you press on the "Execute" button. As a result, you will see the prediction of the model below, in the *responses* section.

```bash
├───data  # Stores .csv
├───models  # Stores .pkl
├───notebooks  # Contains experimental .ipynbs
├───project_name
│   ├───data  # For data processing, not storing .csv
│   ├───features
│   └───models  # For model creation, not storing .pkl
├───reports
├───tests
│   ├───data
│   ├───features
│   └───models
├───.gitignore
├───.pre-commit-config.yaml
├───pipeline.py
├───README.md
```