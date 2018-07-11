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
***Public leaderboard score: TBA***

Model Highlights:
* Uses Stage 0 Vanilla training and test sets
  * Scales vanilla sets by:
    1. Concatenating training and test sets
    2. Dividing out the maximum value from each feature column
  * Scales target values by
    1. Subtracting out minimum target value
    2. Scaling by dividing all target values by (y_max - y_min)
* Keras-built stacked auto-encoder (5-layer stack)
* Dimensionality Reduction / Clustering methods including:
  * Kmeans clustering
  * PCA
  * Truncated SVD
  * Gaussian Random Projection
  * Sparse Random Projection
* CatBoost

## Ongoing Ideas:
* Find a way to align the train and test set
  * Covariance shift correction ([see here](http://blog.smola.org/post/4110255196/real-simple-covariate-shift-correction))
    * Find features that have highest deviance between training and test sets
    * Find weighting factors for the features that have high deviances to weight train and test sets
    * Find upsampling factors by training on whole feature sets
  * Exploring applications of Kolmogorov-Smirnov Test
  * Area under curve approach
  * Binary 0/1 matching
  * Feature clustering via correlation values
* Build XGBoost model
* Weight models based on validation accuracy
