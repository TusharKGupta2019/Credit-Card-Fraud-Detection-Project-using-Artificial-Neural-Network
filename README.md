Credit Card Fraud Detection using Artificial Neural Networks

This project aims to develop an effective credit card fraud detection system using artificial neural networks (ANNs). The system is trained on a dataset of credit card transactions to learn patterns of fraudulent and legitimate transactions. Once trained, the ANN model can be used to classify new transactions as either fraudulent or legitimate, helping to prevent credit card fraud.

Features : 
Preprocesses and cleans the credit card transaction dataset, 
Uses SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced data, 
Trains an ANN model using the preprocessed and oversampled data, 
Evaluates the performance of the trained model using various metrics, 
Provides functionality to classify new transactions as fraudulent or legitimate, 

Requirements : 
Python 3.x, 
Pandas, 
Scikit-learn, 
Keras, 

Usage : 

Prepare the dataset: Ensure that the credit card transaction dataset is in a compatible format and located in the project directory.

Run the Jupyter notebook: Open the CC Fraud Detection using ANN.ipynb file in Jupyter Notebook and execute the cells sequentially.

Preprocess the data: The notebook will preprocess and clean the dataset, handling missing values and encoding categorical features.

Apply SMOTE: SMOTE will be used to oversample the minority class (fraudulent transactions) to balance the dataset.

Train the ANN model: The preprocessed and oversampled data will be used to train an artificial neural network model for credit card fraud detection.

Evaluate the model: The trained model will be evaluated using various performance metrics such as accuracy, precision, recall, and F1-score.

Classify new transactions: The trained model can be used to classify new credit card transactions as either fraudulent or legitimate.

Acknowledgments
This project was inspired by the growing need for effective credit card fraud detection systems and the potential of artificial neural networks in this domain. The dataset used in this project is publicly available and can be obtained from Kaggle.
link to dataset - https://www.kaggle.com/datasets/kartik2112/fraud-detection?resource=download
