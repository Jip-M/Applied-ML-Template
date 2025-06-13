# 🧛🏽‍♂️🩸 Bat Audio Classifier 🦇🔊

**Welcome to our project**  
This repository is a working pipeline that gets audio files, it transforms them conveniently and then are inserted in a CNN for the classification.

**Introduction**  
As is commonly known, bats emit high-frequency sounds to detect their surroundings — a method
known as echolocation. In addition to navigation, their sounds can serve other purposes, such
as feeding buzzes, which are quick sequences of calls emitted while targeting prey to pinpoint its
exact location, or social calls, which are used to communicate with or attract other bats. The
exact properties of these different sounds can be used to determine the species of bat, which is a
practice done by fieldworkers, experts, or researchers. Since bats are protected animals, ecological
research of bats is of great importance, especially in high-construction areas where they may be
negatively affected by the changes of their surroundings. One of our teammates started a summer
job for a Dutch ecological fieldwork company ‘Gaia ecologie’, from which we learnt that the currently
implemented AI software for bat classification is not being used, due to its unreliability and frequent
errors, especially for uncommon bat types or sounds other than echolocation. Therefore, we will
aim to create a successful bat classification model using sounds sourced from Xeno-Canto.

**Demo**  
For demo purposes we will (temporarily) host the app on a webserver. Feel free to try it out here:

http://batclassifer.westeurope.azurecontainer.io:8000/

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
```
python -m main
```

As a result, you will see multiple checkpoints (print statements) throughout the run that highlight when the data is getting prepared, saved, start to train the models, each epoch update of the accuracy and the loss of the main model (CNN), confustion matrix and the average accuracy after 10 (set as default in the pipeline, in the initialize_CNN() function) epochs. If you wish to not see the print statements, just set the verbose value to false in the fit function of the CNN. In addition to that, you will also see the explainable AI part explained below in this file.


**How to run the Streamlit app**  
First, open a terminal at the top of the project (applied-ml-template). When you have the terminal correctly opened, simple run the command **streamlit run streamlit_app/Home.py**. The first time you enter Streamlit might try to get you to try and sign up for a newsletter, just leave the field in the terminal blank and proceed. The Streamlit app will boot up, and further instructions on how to use it are on the Home page. This page will be the first one opened. 


**How to run the API app**  
In order to run the FastAPI, you have to open the integrated terminal from the folder called "api" and run this script:
```
uvicorn api.main:app --reload
```
After that, you have to enter this site in order to see it work:

http://127.0.0.1:8000/docs#/


On the site, you will see a **Prediction** section that will let you try the code out. To do this, you press on the section and press on the "Try it out" button. This will let you upload a ".wav" file. If you do not upload a file, there will be a built-in sample that you can use to judge the model's performance. When you selected your choice, you press on the "Execute" button. As a result, you will see the prediction of the model below, in the *responses* section.


**explainable.py**  
This function is run the main.py, after the run_pipeline(). After the function is run, it show two plots: one of a selected spectrogram and one with the patches of which the function suggests are important for the model prediction. The function occludes a part of the input image (spectrogram) and checks the model prediction. If the model’s confidence interval decreases, this means that the occluded part is important for the classification of the model, therefore it is highlighted respectively. We do this on the entire image to see the entire importance of each patch. Moreover, we decided that instead of having small patches to occlude the image, we need bigger patches that are similar to a bar that occludes the entire horizontal. We do this because if we occlude one of the sound spikes from the spectrogram, the model would still classify it as the bat and the importance matrix would be misguiding or useless. In this case, we iterate through all the pixels vertically and we just move the long patch over the image to highlight the important parts

**Docker**  
The project can also be built and run as a Docker container using the following command:
```
docker compose build
```
Followed by:
```
docker compose up
```
For this, it uses the settings in compose.yaml, which by default will also run the app on port 8000. So to access the app in the browser, navigate to:

http://127.0.0.1:8000

The Demo website is also hosted from this docker container.

**Overview**  
An overview of our git repo can be found below:

```bash
├───api
│   └───main.py
│
├───bat_classifier
│   │		├───data
│   │   		├───download_data.py
│   │   		└───preprocess.py
│   └───models
│       	├───__init__.py
│      	 	├───base_model.py
│      		├───CNN.py
│      		└───metrics.py
│
├───data
│   ├───cleaned
│  	├───raw
│ 	├───sample
│  	└───labels.csv
│
├───dataset
│   	└───metadata
│				└───grp_batsq_A
│           			├───page1.json
│           			├───page2.json
│           			└───page3.json
├───explained_ai_predictions_images
│   			├───bat_type 0 (bomba).png
│   			├───bat_type 0.png
│   			├───bat_type 1.png
│   			├───bat_type 2.png
│   			├───bat_type 3.png
│   			├───bat_type example.png
│   			└───BOOOMBA.jpg
├───notebooks
├───streamlit_app
│   │		├───pages
│   │   		└───about_bats_images
│   │       				├───Myotis albescens.jpg
│   │      					├───Nyctalus_noctula.jpg
│   │      					├───Pipistrellus pipistrellus.jpg
│   │      					└───Plecotus auritus.jpg
│   ├───About_Bats.py
│   ├───About.py
│   ├───Data_Exploration.py
│   ├───Logistic_Regression.py
│   ├───Model_Evaluation.py
│   ├───Model_Results_Info.py
│   ├───Home.py
│   └───utils.py
├───trained_model
│ 			├───cnn_metrics.csv
│ 			├───CNN.pt
│ 			├───coef.npy
│ 			├───intercept.npy
│ 			└───lr_metrics.csv
│ 
├───__init__.py
├───.dockerignore
├───.gitattributes
├───.gitignore
├───.pre-commit-config.yaml
├───compose.yaml
├───Dockerfile
├───explainable.py
├───main.py
├───pipeline.py
├───README.Docker.md
├───README.md
├───requirements.txt
```