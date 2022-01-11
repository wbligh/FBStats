
			Football Stats Project
			
			
Aim: Predict the scores/outcomes of football matches using machine learning techniques

Components:

	1)	Web scrapers to pull football statistical data from understat.com (to be expanded)
	2)	Google Cloud Function to trigger scrapers weekly (via Pub/Sub) and store in Big Query
	3)	Data wrangling scripts to condense and create important feature set for prediction
	4) 	Modelling scripts to ingest data and predict goals scored per match
		a) Random Forest Classifier
		b) K-Nearest Neighbours Classifier
		
Still to build

	1)	Ensemble approach to investigate combination of classifiers
	2)	Investigate regression performance on goals scored
	3) 	Publish final model results to Google Cloud Platform
	4)	Create live google dashboard with upcoming match predictions
	
	

Notes:

	understatScraper.py

There are 3 datasets to be pulled from understat.com, League table history, player history & match details history. This scraper file uses bs4 and pandas to scrape, parse and restructure this data.

For the match history, each match sits on it's own webpage. In order to collect this data in a complete way, the sitemap.xml webpage is used to find any missing values in the already collected dataset, and new pages and scraped and pushed.

This file uses the pandas_gbq module which encapsulates the Google Client objects in order to succinctly push the data into Google Big Query.


This file is currently set up on a Cloud Function which is triggered by a Pub/Sub topic which is published to weekly by a Cloud Scheduler module each week.


	data_wrangling.py
	
This file uses SQL queries to pull the relevant data from the Big Query database and then uses a self-join approach to find recent historic data for each fixture. This allows stats from previous matches to be used as the base data for future prediction


	modelling.py
	
This file reads in the data from the previous file and does some basic pre-processing routines including missing values, scaling and label encoding. This data is then split for training and testing and passed into Random Forest and K-Nearest Neighbours models which undergo hyperparamter optimisation via Random Search cross validation and grid search cross validation






