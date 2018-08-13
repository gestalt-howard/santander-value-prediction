# Project Ideas and Documentation:
This markdown file is a log of the various stages and ideas that were explored throughout the course of this Kaggle challenge. This is a "living" document in the sense that it will be frequently updated as new concepts are brainstormed, tested, and documented (at least, until the end of the competition / project).

![idea man](https://github.com/gestalt-howard/santander-value-prediction/blob/master/images/ideas.png)


## Stage 0: Initial Exploration and Public Resources
### Vanilla Preprocessing:
Preprocessing steps:
* Saved Stage 0 Vanilla training and test sets
  * Removal of train set's constant columns from train and test
  * Removing train set's duplicate columns from train and test
* Saved training and test sets' indexes and column names
* Found that 3000 features account for around 95% of variance using PCA
* Visualized distribution of training labels

### Stage 0v0 - Tuned Light GBM:
***Public leaderboard score: 1.44***

**NOTE: Could still implement Robust Scaler**

Model Highlights:
* Used the Stage 0 Vanilla training and test sets
* Extensive parameter tuning focusing on:
  * Number of boosting rounds with large learning rate
  * Number of boosting rounds with small learning rate
* Averaging results over many random seed initializations

### Stage 0v1 - Auto-Encoder / Dimensionality Reduction with CatBoost:
***Public leaderboard score: 1.40***

Model Highlights:
* Uses Stage 0 Vanilla training and test sets
  * Scales vanilla sets by:
    1. Concatenating training and test sets
    2. Dividing out the maximum value from each feature column
  * Scales target values by
    1. Subtracting out minimum target value
    2. Scaling by dividing all target values by (y_max - y_min)
* Uses data transformed by Keras-built stacked auto-encoder (5-layer stack)
* Dimensionality Reduction / Clustering methods including:
  * Kmeans clustering
  * PCA
  * Truncated SVD
  * Gaussian Random Projection
  * Sparse Random Projection
* CatBoost Regressor with cross-validation (5-fold)

### Stage 0v2 - XGBoost with Piped Feature Extraction:
***Public leaderboard score: 1.39***

Model Highlights:
* Feature processing and training pipeline:
  * Variance threshold to remove constant columns
  * Removes duplicate columns using numpy's *unique* function
  * Feature union with:
    * PCA down to 100 dimensions
    * Random Forest Classifier to transform dataset into predicted probabilities and binned target values
    * Statistical value transformations (total of 26 x 6 different variations)
  * XGBoost Regressor with 10-fold cross-validation

### Stage 0 Blending:
#### 0v1 and 0v2:
***Public leaderboard score: 1.38 (402th)*** (at the time)
#### 0v0, 0v1, and 0v2:
***Public leaderboard score: 1.38 (380th)*** (at the time)


## Stage 1: Covariate Shift Correction
Stage 1 deviates from the public kernels into uncharted waters. Specifically, Stage 1 is primarily distinguished by an exploration of covariate shift correction through the Kullback-Leibler Importance Estimation Procedure (KLIEP). I heavily referenced the paper **Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation** authored by Sugiyama, Nakajima, Kashima, von Bunau, and Kawanabe.

### KLIEP Importance Weights Training:
To determine importance weights for samples in the Santander training set, I undertook the following steps:
* Removed redundant features
* Applied PCA on dataset down to 100 components
* Scaled data using sklearn's StandardScaler library
* Tested combinations of Gaussian Width and number of kernels in a grid-search fashion to determine optimal KLIEP settings

There were a few important takeaways from my research into this algorithm:
* Higher number of kernels result in better performance (maximum number is the number of test samples)
* Lower gaussian width results in better performance but excessively small widths will yield extremely small values that will simply be expressed as zero

Ultimately, I proceed with the following models using a **Gaussian Width of 75** and the **Number of Kernels** as 1000. Graphics of my experimentation can be found in:

```
./scripts/covariate_shift/images/
```

### Stage 1v0 - XGBoost with Covariate Shift Correction:
***Public leaderboard score: 1.45***

Model Highlights:
* Uses same cross-validated XGBoost regressor structure as Model 0v2
* Uses same preprocessing as Model 0v2 except for a scaling operation using StandardScaler following all feature preprocessing
* Implements a custom objective and evaluation function for the XGBoost model
  * Custom objective is a **weighted ordinary least squares**
  * Custom evaluation function is a **root mean squared log error** function

### A Comment on Covariate Shift:
A relatively untuned implementation of covariate shift on the Santander dataset yielded a result of **1.45** on the public Kaggle leaderboard. While it is disappointing that my covariate shift implementation did not perform better, there were some obvious (in hindsight) reasons why this should have been expected:
1. Covariate shift is a strategy that relies heavily upon the careful tuning of hyperparameters (something that was not exhaustively explored in Model 1v0)
2. Since my implementation of covariate shift, it has been revealed that the Santander dataset is actually a scrambled time-series which completely negates the utility of covariate shift which understands dataset features to be independent of each other
3. The Santander train and test set vary so significantly from each other that (in conjunction with the time-series nature of the dataset features and the target variable) it is infeasible for covariate shift to correct for these deviations

In subsequent, I'm confident that there will be an application that is suitable for covariate shift correction. However, the nature of the problem in this Kaggle challenge does not readily facilitate the usage of this algorithm.

## Stage 2: Time-Series Unraveling
Stage 2 follows upon the heels of the failed covariate shift correction stage. In Stage 2, I begin my exploration with the knowledge of the Kaggle community (aka the public kernels). It was the Kaggle community who had initially revealed the time-series nature of the data and a significant portion of this Stage is dedicated to validating and replicating the results shown in the public leaderboard. With only 13 days left on the competition clock, my aim going forward is to catch up to the public leaderboard and try out one more original method.

Stage 2 can be broken-down into 2 phases: Exploration and Modeling. All efforts in the Exploration phase do not utilize machine learning whatsoever and are aimed at uncovering time-series information about the Santander dataset. In the Modeling phase, machine learning models will be reintroduced into the equation.

### Exploration and Validation:
The folder containing scripts for this segment of the project can be found in:
```
./scripts/time_series/
```
The intent of this segment was to validate the Kaggle community's findings through a mixture of my own analysis and replicating some public kernels' results.

#### Time-Series Reconstruction:
***Public leaderboard score: 0.69***
One of the replication efforts was focused on this public kernel: https://www.kaggle.com/johnfarrell/breaking-lb-fresh-start-with-lag-selection.

On a high-level, this public kernel sought to leverage the time-series nature of the Santander dataset to make predictions on the target variable. A more detailed explanation of the approaches and strategies used in this kernel can be found in my own replication:
```
./scripts/time_series/time_construct.ipynb
```
The result of this replication effort yielded my highest public leaderboard score yet (0.69). This result affirms that there is indeed a time-series aspect to the Santander dataset that can be used to obtain excellent predictions (independent of using machine learning models).

### Stage 2v0 - LightGBM with Data Leak:
***Public leaderboard score: 0.66***

**NOTE:** Model based on public kernel (https://www.kaggle.com/ogrellier/feature-scoring-vs-zeros)
Model Highlights:
* Improves upon the time-series reconstruction results (with public leaderboard score of 0.69) by adding a LightGBM component
  * The addition of a boosting procedure aims to improve upon the time-series prediction by filling in for unknown leak values (values that were predicted as 0 during the time-series reconstruction)
* Includes a feature-scoring function that analyzes the features to find each individual feature's predictive value (measured by a RMSE score)

### Stage 2v1 - Modified LightGBM with Blending:
***Public leaderboard score: 0.63***

**NOTE:** Model based on public kernel (https://www.kaggle.com/the1owl/love-is-the-answer/output?scriptVersionId=4733381)
Model Highlights:
* Incorporates an important-feature selection process based on:
  * Number of shared values between any given feature and the target variable
  * Whether or not most of the values in any given feature are within a 5% range of the target variable (thus implying that they are closely related in a sequence-like manner)
* Generates row-index features in hopes of capturing any sequence information contained in the row order
* Light GBM training set is a concatenation of the train and test datasets along with a target variable (the concatenated test set's target variable is the test predictions generated by Model 2v0)
* Incorporates other public kernel results for a final blended, better-generalized solution  
