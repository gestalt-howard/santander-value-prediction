# Project Ideas and Documentation:
This markdown file is a log of the various stages and ideas that were explored throughout the course of this Kaggle challenge. This is a "living" document in the sense that it will be frequently updated as new concepts are brainstormed, tested, and documented.

![idea man](https://github.com/gestalt-howard/santander-value-prediction/blob/master/images/ideas.png)

## Stage 0:
### Vanilla Preprocessing:
Preprocessing steps:
* Saved Stage 0 Vanilla training and test sets
  * Removal of train set's constant columns from train and test
  * Removing train set's duplicate columns from train and test
* Saved training and test sets' indexes and column names
* Found that 3000 features account for around 95% of variance using PCA
* Visualized distribution of training labels

### Tuned Light GBM:
***Public leaderboard score: 1.44***

**NOTE: Still need to implement Robust Scaler**

Model Highlights:
* Used the Stage 0 Vanilla training and test sets
* Extensive parameter tuning focusing on:
  * Number of boosting rounds with large learning rate
  * Number of boosting rounds with small learning rate
* Averaging results over many random seed initializations

### Auto-Encoder / Dimensionality Reduction with CatBoost:
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

### XGBoost with Piped Feature Extraction:
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

## Ongoing Ideas:
* Try different dimensionality projection concepts with Stage 0 models
  * T-SNE
* Find a way to align the train and test set
  * Covariance shift correction ([see here](http://blog.smola.org/post/4110255196/real-simple-covariate-shift-correction))
    * Find features that have highest deviance between training and test sets
    * Find weighting factors for the features that have high deviances to weight train and test sets
    * Find upsampling factors by training on whole feature sets
  * Exploring applications of Kolmogorov-Smirnov Test
  * Area under curve approach
  * Binary 0/1 matching
  * Feature clustering via correlation values
* Weight models based on validation accuracy
