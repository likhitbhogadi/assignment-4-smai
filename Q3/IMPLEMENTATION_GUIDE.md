# Q3: Decision Trees and Random Forests - Implementation Guide

## Overview

This guide documents the implementation of decision trees and random forests for sentiment classification on Amazon product reviews.

## Part 1: From-Scratch Implementation

### File: `tree_utils.py`

#### Task A: LeafNode.predict()

**Implementation**:

- Returns the mean of training labels (`self.y_N`) for all input examples
- All examples at a leaf get the same prediction
- Minimizes squared error on the training set

```python
def predict(self, x_TF):
    T = x_TF.shape[0]
    yhat_T = np.mean(self.y_N) * np.ones(T)
    return yhat_T
```

#### Task B: InternalDecisionNode.predict()

**Implementation**:

- Splits input examples based on feature threshold
- Routes examples to left child (< threshold) or right child (>= threshold)
- Recursively gets predictions from children
- Aggregates results while preserving order

```python
def predict(self, x_TF):
    T, F = x_TF.shape

    # Determine left/right routing
    left_mask_T = x_TF[:, self.feat_id] < self.thresh_val
    right_mask_T = np.logical_not(left_mask_T)

    # Initialize output
    yhat_T = np.zeros(T, dtype=np.float64)

    # Get predictions from children
    if np.sum(left_mask_T) > 0:
        yhat_T[left_mask_T] = self.left_child.predict(x_TF[left_mask_T])
    if np.sum(right_mask_T) > 0:
        yhat_T[right_mask_T] = self.right_child.predict(x_TF[right_mask_T])

    return yhat_T
```

### File: `select_best_binary_split.py`

#### Task C: Best Binary Split Selection

**Implementation**:

- For each feature, evaluates all possible thresholds
- Computes squared error cost for left and right children
- Selects threshold minimizing total cost
- Returns best feature and threshold across all features

**Key Steps**:

1. Loop through all features
2. For each feature, get unique threshold values
3. For each threshold:
   - Split data into left (< threshold) and right (>= threshold)
   - Compute mean prediction for each child
   - Compute squared error cost for each child
4. Select threshold with minimum total cost
5. Return best feature, threshold, and split data

```python
for v in range(V):
    thresh = possib_xthresh_V[v]
    left_mask = x_NF[:, f] < thresh
    right_mask = np.logical_not(left_mask)

    y_left = y_N[left_mask]
    y_right = y_N[right_mask]

    left_yhat_V[v] = np.mean(y_left)
    right_yhat_V[v] = np.mean(y_right)

    left_cost_V[v] = np.sum(np.square(y_left - left_yhat_V[v]))
    right_cost_V[v] = np.sum(np.square(y_right - right_yhat_V[v]))

chosen_v_id = np.argmin(total_cost_V)
```

### File: `train_tree.py`

#### Task D: Recursive Tree Training

**Implementation**:

- Recursively builds tree using greedy algorithm
- Stops when max depth reached or insufficient samples
- Calls `select_best_binary_split()` to find optimal split
- Creates leaf if no beneficial split exists

```python
if feat_id is None:
    return LeafNode(x_NF, y_N)
else:
    left_child = train_tree_greedy(
        x_LF, y_L, depth + 1,
        MAX_DEPTH, MIN_SAMPLES_INTERNAL, MIN_SAMPLES_LEAF)
    right_child = train_tree_greedy(
        x_RF, y_R, depth + 1,
        MAX_DEPTH, MIN_SAMPLES_INTERNAL, MIN_SAMPLES_LEAF)
    return InternalDecisionNode(
        x_NF, y_N, feat_id, thresh_val, left_child, right_child)
```

## Part 2 & 3: Scikit-Learn Implementation

### Notebook Structure

1. **Data Loading**

   - Load train/valid/test sets from CSV files
   - Load vocabulary (7729 terms)
   - Verify data consistency

2. **Custom Train/Valid Split**

   - Use `PredefinedSplit` for GridSearchCV
   - Ensures validation set is fixed (not random)

3. **Simple Decision Tree** (Part 2.1)

   - `max_depth=3`, `criterion='gini'`
   - Visualize using `pretty_print_sklearn_tree()`
   - Analyze structure for same-class children

4. **Decision Tree Grid Search** (Part 2.2)

   - Hyperparameters: `max_depth` ∈ {2, 8, 32, 128}, `min_samples_leaf` ∈ {1, 3, 9}
   - Scoring: balanced accuracy
   - Train best tree on training set only

5. **Simple Random Forest** (Part 3.1)

   - 100 trees, `max_depth=3`
   - Extract feature importances
   - Identify top 10 important words and near-zero importance words

6. **Random Forest Grid Search** (Part 3.2)

   - Hyperparameters: `max_features` ∈ {3, 10, 33, 100, 333}, `max_depth` ∈ {16, 32}
   - `n_estimators=100`
   - Use `n_jobs=-1` for parallel processing

7. **Model Comparison** (Part 3.3)
   - Compare all 4 models (simple tree, best tree, simple RF, best RF)
   - Report balanced accuracy on train/valid/test
   - Visualizations with username watermark

## Key Design Decisions

### Reproducibility

- Fixed random seeds: `random_state=101` for models, `seed=42` for numpy operations
- Ensures identical results across runs

### Balanced Accuracy

- Used as primary metric (handles class imbalance)
- Formula: (sensitivity + specificity) / 2

### Visualization Requirements

- All plots include:
  - Title with username (`EMAIL_USERNAME`)
  - x-label and y-label
  - Legend (where applicable)
  - Username watermark

### Code Organization

- Separate functions for computation and visualization
- Docstrings for all functions
- Clear variable naming (following conventions: `x_NF`, `y_N`, etc.)

## Running the Notebook

1. **Install Dependencies**:

   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **Run All Cells**:

   - Execute cells sequentially from top to bottom
   - Grid searches may take 2-10 minutes

3. **Output**:
   - Tables printed in console
   - Figures saved as PNG files
   - All results in notebook

## Expected Results

- **Simple Tree**: Baseline performance, interpretable
- **Best Tree**: Improved via hyperparameter tuning
- **Simple RF**: Better than single trees due to ensemble
- **Best RF**: Best overall performance

## Files Generated

- `model_comparison.png`: Bar chart and scatter plot of model performance
- `feature_importance.png`: Top 20 important features visualization
- Trained models available in notebook memory

## Notes

- Data files are large (>50MB), loaded via pandas
- Grid search uses parallel processing for efficiency
- All implementations follow assignment guidelines for reproducibility and documentation
