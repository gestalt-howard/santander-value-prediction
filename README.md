# 2018 Santander Value Prediction Kaggle Challenge
***Project Complete: September 7, 2018***

![santander logo](https://github.com/gestalt-howard/santander-value-prediction/blob/master/images/santander_logo.jpg)

***Final Placement Info:***
* Bronze medalist
* Top 6%
* 243rd out of 4484 competitors

Welcome to my project repo for the 2018 Santander Value Prediction Kaggle Challenge! This repo contains the scripts, data files, and submission results that I accumulated while participating in the Kaggle challenge. From the period of June 18, 2018 to the project completion date, I created 10 unique machine learning models and conducted over half-a-dozen in-depth analysis into the Santander dataset. Collectively, the work demonstrated in this repo represents the most-involved data science / machine learning project I've undertaken in my budding career thus far. If you would like to look through my work, I ***strongly recommend*** using my project write-up (contained in the file **project_summary.pdf**) as a guide. In that document, you'll find a chronological account of my journey throughout this Kaggle challenge. Feel free to skip through sections as you desire.   

## Repo Directory Guide:
A high-level overview of the directories and subdirectories is listed below. Please note that the granular details (i.e. the purpose of each individual file) are not included in this directory guide since there are simply too many files to consider. That being said, rest assured that the most important files are neatly introduced and explained within my project write-up. Any file that is not directly mentioned within **project_summary.pdf** can be assumed to be an auxiliary file that is referenced by the more-important scripts featured within the write-up.

* **data**: Folder containing data files that are loaded for analysis or model training
  * *Please note that many of the required data files are missing due to Github's data upload size limits*
* **images**: Folder containing images used in this repo's various markdown files
* **scripts**: Folder containing all machine learning models and data analysis scripts
  * **autoencoder**: Folder containing scripts relating to the development and deployment of a stacked-autoencoder
  * **covariate_shift**: Folder that contains scripts used to develop, debug, and tune a KLIEP covariate-shift correction algorithm
  * **time_series**: Folder that contains all the scripts and analysis files relating to mining time-series structure from the Santander datasets
* **submissions**: Folder containing all my submission files
  * *Please note that not all of these submission files were actually submitted for a leaderboard score*
* **ideas.md**: Markdown file that served as a place to track ideas and progress throughout most of the project
* **project_summary.pdf**: PDF containing my report write-up (perhaps the most important file in this entire repo)
* **sample_submission.csv**: A file provided by the competition organizers to demonstrate the proper submission format

## Setting Up
***IMPORTANT: Due to Github's constraints on uploaded file sizes, I am unable to provide all the required files that it'll take to smoothly run through all my scripts. If you would like to replicate my results, please follow the below instructions for setting up and making sure that all the required data files are present.***

1. Download the training and test sets for the Santander Kaggle challenge (https://www.kaggle.com/c/santander-value-prediction-challenge/data) and place these unzipped files in the **data** folder
2. Run the file ```./scripts/debug_gen.py``` which will create debugging versions of the training and test sets
  * *This is required if you wish to access the debugging functionality built-in to many of my models*
3. Run the file ```./scripts/h5_converter.py``` which will export the training and test datasets from CSV format into h5 format
  * Many of my models load the training and test datasets from their h5 format since read speeds are dramatically faster in the h5 scheme
4. Have fun running through my models!
  * Remember to reference my project write-up to gain a high-level intuition of the model concepts before diving into the raw code

## One Last Comment...
As a final note, please feel free to reach out to me in case you are particularly curious about one of my processes or simply need help in getting some path dependencies sorted. Thank you for taking the time to review this repo! It is my proudest project to-date.

Also, I would be remiss to not give a massive, heartfelt thanks to the collective Kaggle community whose generous sharing of wisdom has been a steady guide throughout this challenge.
