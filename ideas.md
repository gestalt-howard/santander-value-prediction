# Project Ideas and Documentation:
This markdown file is a log of the various stages and ideas that were explored throughout the course of this Kaggle challenge. This is a "living" document in the sense that it will be frequently updated as new concepts are brainstormed, tested, and documented.

![idea man]()

## Stage 0:
### Tuned Light GBM:
***Public leaderboard score: 1.44***

Model Highlights:
* Extensive parameter tuning focusing on:
  * Number of boosting rounds with large learning rate
  * Number of boosting rounds with small learning rate
* Averaging results over many random seed initializations

### Auto-Encoder / Dimensionality Reduction with CatBoost:
***Public leaderboard score: TBA***

Model Highlights:
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
  * Area under curve approach
  * Binary 0/1 matching
  * Feature clustering via correlation values
* Build XGBoost model
* Weight models based on validation accuracy
