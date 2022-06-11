# cs267project
How to run the code:

baseline.ipynb:
Code for baseline linear classifier. Simply run the whole notebook and it will do a grid search on hyper-parameters and print out the results. Need to change the path to data.csv before running the notebook.

problog_model.ipynb:
Code for our problog model. Simply run the whole notebook and it will do the following: (Need to change the path to data.csv before running the notebook.)
1. generate the .pl file for our problog model
2. run query on all songs and evaluate using the metrics: root mean squared error (rmse) and mean absolute error (mae). 
Note: learning and inference are computationally intractable for large dataset. To run a toy example, we recommand choosing a dataset size < 30
4. investiage the growth of learning and inference time as dataset size increases and plot the results
