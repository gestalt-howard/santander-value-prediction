# Pseudocode for KLIEP Algorithm:

## KLIEP Function:
### Input:
* Kernel functions set at selected **test set points**
* Training set
* Test set

### Output:
Estimated importance weights for training set

### Algorithm:
1. Compute **A matrix**:
    * Number of rows = number of test samples
    * Number of cols = number of basis functions
    * Each element of **A** is the **l-th basis function** evaluated for the **j-th test sample**
2. Compute the **b vector**:
    * Number of elements = number of basis functions
    * Each element of **b** is the averaged sum of all training samples evaluated for the **l-th basis function**
3. Initialize **alpha vector** as values *greater than zero*
4. Initialize learning rate as a value *greater than zero* but *much less than one*
5. Repeat until convergence: (modified gradient descent)
    * Make slight update to **alpha vector** with **A matrix**
    * Apply training vector constraint with **b vector**
    * Apply greater than 0 constraint
    * Apply normalization constraint with **b vector**
6. Calculate **estimated importance weight** for given sample:
    * Sum over all basis functions *multiplied by* its corresponding **alpha** value

## Likelihood Cross Validation Model Selection Function:
### Input:
* Set of models:
  * Each model has a set of basis functions (kernels) initialized at *b number of points*
* Training set
* Test set

### Output:
Estimated importance weights for training set

### Algorithm:
1. Split the test set into R disjoint subsets
2. For each model **m** in the given set of models:
    1. For each split disjoint test set:
        * Calculate an **estimated importance weight** using the *KLIEP algorithm*
        * Calculate the Kullback-Leibler (KL) divergence term
    2. Average the KL divergence terms calculated for each split disjoint test set
3. Select the **best model** from the given set of models:
    * Best model is the one that results in the maximal KL divergence term (averaged across all split disjoint test sets)
4. Calculate the **estimated importance weight** for a given sample using the **best model**
