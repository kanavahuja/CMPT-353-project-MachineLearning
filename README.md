# CMPT 353 Final Project: Credit Card Fraud Detection

## How to Run

1. Run Cleaning-data.py to obtain the output.csv for the data modeling section from the original dataset creditcards.csv

   python3 Cleaning-data.py creditcard.csv output.csv
2. Run Outliers.py to obtain a cleaned.csv sample dataset for the data analysis section

   python3 Outliers.py
3. Run plots_and_statistical_tests.py to run our initial data analysis

   python3 plots_and_statistical_tests.py 
4. Run Model-Analysis.py to run our ML models that was applied to output.csv

   python3 Model-Analysis.py output.csv

## Miscellanous Section

1. plots_and_statistical_tests.ipynb available to showcase the plots that we produced but is not required to run for the project
2. NOTE: the data returned in Outliers.py was not used in the data modeling section due to reasons explained in the Report


## Libraries Used
- pandas
- scipy
- numpy
- matplotlib.pyplot
- seaborn
- sklearn
- statsmodels.stats.multicomp 
- statsmodels.api 
- imblearn