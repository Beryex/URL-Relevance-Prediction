# [Stanford STATS 202 Prediction 2024](https://www.kaggle.com/competitions/stanford-stats-202-prediction-2024/overview): URL Relevance Prediction

Search engines have become an integral part of our daily lives, offering relevant documents and websites based on user input and browsing behavior, such as cookies. To determine the relevance of a document, search engines utilize hundreds of signals, ultimately returning a ranked list of documents based on these signals.

In this project, we are provided with a training dataset comprising 80,046 observations and 10 attributes, including "query\_length", "is\_homepaged", and eight unnamed signals. The dataset also includes a binary output indicating whether an observation is relevant, based on search engine query and URL data. Additionally, we have a test dataset containing 30,001 observations with the same 10 attributes, but without the relevance output.

Our objective is to develop a model using the training dataset to predict the relevance of each observation in the test dataset.

## Install
1. Clone the repository and navigate to the RLPruner working directory
```bash 
git clone https://github.com/Beryex/URL-Relevance-Prediction.git
cd URL-Relevance-Prediction
```
2. Set up environment
```bash 
pip install -r requirements.txt
```

## Usage
We implement multiple dataset preprocessing and data mining methods, play with them and find the true relationships between the 10 attributes and the relevance.

## Results
For traditional methods, boosting works the best and we record the effect of dataset preprocessing on boosting with optimal hyperparameter.

| Standardize | Remove Sig5 | Remove Outlier | Kernel Methods | Apply PCA | CV Average Accuracy  | Test Accuracy |
|-------------|-------------|----------------|----------------|-----------|----------------------|---------------|
| No          | No          | No             | No             | No        | 66.83                | 67.43         |
| No          | No          | No             | No             | Yes       | 58.83                | N/A           |
| Yes         | No          | No             | No             | Yes       | 65.97                | N/A           |
| Yes         | No          | Yes            | No             | No        | 66.73                | N/A           |
| Yes         | No          | No             | Yes            | No        | 66.73                | N/A           |
| Yes         | No          | No             | Yes            | Yes       | 65.87                | N/A           |
| Yes         | Yes         | No             | No             | No        | 66.73                | 68.25         |
| Yes         | Yes         | Yes            | No             | No        | 66.64                | 67.53         |
| Yes         | Yes         | No             | Yes            | No        | 66.58                | N/A           |

As observed, removing outliers (despite eliminating only 23 samples in total) resulted in performance declines across all models. We hypothesize that these outliers represent high-leverage extreme cases that are beneficial for the model's fitting. Additionally, using degree-2 kernel methods to add features also led to performance degradation, possibly due to the interference of redundant features with the model's fitting. Similarly, applying PCA did not improve model performance, which we attribute to the loss of original information when the attributes were linearly mapped into an orthogonal feature space.

Standardizing the data proved to be effective. While not standardizing the data might slightly improve the average cross-validation accuracy in some cases, standardizing clearly yielded better results on the test dataset. Removing sig5 was also beneficial; although it did not significantly affect the average cross-validation accuracy, it slightly improved the test accuracy.

For deep learning, we also test several models and record their results.

| Model    | Initial Learning Rate | Weight Decay | Total Epoch | Validation Accuracy |
|----------|-----------------------|--------------|-------------|---------------------|
| VGG11    | 0.1                   | 1e-4         | 10          | 66.66               |
| VGG19    | 0.05                  | 1e-2         | 20          | 66.72               |
| ResNet18 | 0.05                  | 1e-4         | 10          | 66.76               |
| ResNet50 | 0.05                  | 1e-2         | 20          | 66.23               |


The neural networks tended to overfit the dataset easily (with the highest validation accuracy reaching 68.34\%) and exhibited training instability, even when we reduced the learning rate. To mitigate overfitting, we increased the optimizer's weight decay, adjusted the validation set ratio, and applied dropout layers. Although neural networks have strong representational power, they are prone to overfitting or failing to capture the true relationships. The final results were not as good as those achieved with the traditional methods mentioned above.
