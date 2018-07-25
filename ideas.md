# Project Ideas and Documentation:
This markdown file is a log of the various stages and ideas that were explored throughout the course of this Kaggle challenge. This is a "living" document in the sense that it will be frequently updated as new concepts are brainstormed, tested, and documented.

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

**NOTE: Still need to implement Robust Scaler**

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
***Public leaderboard score: TBA***

Model Highlights:
* Uses same cross-validated XGBoost regressor structure as Model 0v2
* Uses same preprocessing as Model 0v2 except for a scaling operation using StandardScaler following all feature preprocessing
* Implements a custom objective and evaluation function for the XGBoost model
  * Custom objective is a **weighted ordinary least squares**
  * Custom evaluation function is a **root mean squared log error** function


## Stage 2: Time-Series Unraveling


## Ongoing Ideas:
* ***MAIN PRIORITY***: Time-series exploration
* Try different dimensionality projection concepts with Stage 0 models
  * T-SNE
* Weight models based on validation accuracy
