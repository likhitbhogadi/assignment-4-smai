# Q3: Decision Trees and Random Forests for Sentiment Classification

## Overview

This directory contains the complete implementation for Assignment 4, Question 3: Decision Trees and Random Forests for sentiment classification on Amazon product reviews.

## Files Structure

```
Q3/
├── hw4_notebook.ipynb              # Main submission notebook (self-contained)
├── tree_utils.py                   # Part 1: LeafNode and InternalDecisionNode classes
├── select_best_binary_split.py    # Part 1: Binary split selection algorithm
├── train_tree.py                   # Part 1: Recursive tree training
├── pretty_print_sklearn_tree.py   # Helper for tree visualization (provided)
├── IMPLEMENTATION_GUIDE.md         # Detailed implementation notes
├── SUBMISSION_SUMMARY.md           # Completion checklist and results
├── README.md                       # This file
└── data_product_reviews/           # Dataset directory
    ├── x_train.csv                 # Training features (6346 x 7729)
    ├── y_train.csv                 # Training labels
    ├── x_valid.csv                 # Validation features (792 x 7729)
    ├── y_valid.csv                 # Validation labels
    ├── x_test.csv                  # Test features (793 x 7729)
    └── y_test.csv                  # Test labels
```

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or use the provided environment:

```bash
source ../smaia4env/bin/activate
```

### 2. Run the Notebook

Open `hw4_notebook.ipynb` in Jupyter/VS Code and run all cells sequentially.

**Note**: Grid search cells may take 2-10 minutes to complete.

## Implementation Details

### Part 1: From-Scratch Decision Tree (20 Marks)

#### Task A: LeafNode.predict()

- **File**: `tree_utils.py`, class `LeafNode`
- **Implementation**: Returns mean of training labels for all input examples
- **Tested**: Doctests pass ✅

#### Task B: InternalDecisionNode.predict()

- **File**: `tree_utils.py`, class `InternalDecisionNode`
- **Implementation**: Routes examples to children based on threshold, recursively aggregates predictions
- **Tested**: Doctests pass ✅

#### Task C: Binary Split Selection

- **File**: `select_best_binary_split.py`
- **Implementation**: Evaluates all feature-threshold combinations, returns optimal split
- **Tested**: Doctests pass, assertion validates cost computation ✅

#### Task D: Tree Training

- **File**: `train_tree.py`, function `train_tree_greedy()`
- **Implementation**: Recursive greedy algorithm with depth/sample constraints
- **Tested**: Integrated in notebook ✅

### Part 2: Scikit-Learn Decision Trees (25 Marks)

#### 2.1: Simple Tree Visualization (10 Marks)

- Trained tree with `max_depth=3`, `criterion='gini'`
- Visualized structure with `pretty_print_sklearn_tree()`
- Analyzed same-class children phenomenon
- Reported balanced accuracy on all sets

#### 2.2: Hyperparameter Tuning (15 Marks)

- Grid search over `max_depth` ∈ {2, 8, 32, 128}
- Grid search over `min_samples_leaf` ∈ {1, 3, 9}
- Used `PredefinedSplit` for custom train/valid split
- Identified and trained best tree

### Part 3: Random Forests (30 Marks)

#### 3.1: Feature Importance Analysis (10 Marks)

- Trained random forest with 100 trees, `max_depth=3`
- Extracted and ranked feature importances
- Created Table 2: Top 10 + near-zero importance words

#### 3.2: Random Forest Tuning (10 Marks)

- Grid search over `max_features` ∈ {3, 10, 33, 100, 333}
- Grid search over `max_depth` ∈ {16, 32}
- Answered theoretical questions about hyperparameters

#### 3.3: Model Comparison (10 Marks)

- Created Table 3 comparing all 4 models
- Reported balanced accuracy (3 decimal places) on train/valid/test
- Provided comprehensive analysis and conclusions

## Dataset Information

- **Task**: Binary sentiment classification (positive/negative)
- **Representation**: Bag-of-words with 7729 vocabulary terms
- **Features**: Binary (presence/absence of each word)
- **Train Set**: 6346 documents (3175 negative, 3171 positive)
- **Valid Set**: 792 documents (404 negative, 388 positive)
- **Test Set**: 793 documents (385 negative, 408 positive)

## Reproducibility

All results are 100% reproducible:

- Fixed random seeds: `random_state=101` (sklearn), `np.random.seed(42)` (numpy)
- Deterministic data loading
- No external randomness

Running the notebook multiple times produces identical:

- Model parameters
- Performance metrics
- Grid search results
- Feature importance rankings
- Visualizations

## Outputs Generated

### Tables

1. **Table 2**: Feature importances (top 10 + near-zero)
2. **Table 3**: Model comparison (4 models x 3 metrics x 3 sets)

### Figures

1. `model_comparison.png`: Performance comparison (bar + scatter plots)
2. `feature_importance.png`: Top 20 important features (horizontal bar chart)

All visualizations include:

- Title with username (likhitbhogadi)
- Axis labels
- Legend (where applicable)
- Username watermark

## Testing

All components tested:

- ✅ Doctests for `tree_utils.py` (pass)
- ✅ Doctests for `select_best_binary_split.py` (pass)
- ✅ Assertions for cost computation (pass)
- ✅ Manual verification of results (pass)

## Key Results

### Model Performance Summary

Performance increases progressively:

1. Simple Tree < Best Tree (hyperparameter tuning helps)
2. Simple RF > Simple Tree (ensemble advantage)
3. Best RF > all others (optimal hyperparameters + ensemble)

### Important Findings

- Random forests generalize better than single trees
- Hyperparameter tuning significantly improves performance
- Feature importances reveal sentiment-discriminative vocabulary
- No overfitting observed with increased `n_estimators`

## Compliance with Guidelines

✅ Object-oriented programming (sklearn + custom classes)  
✅ Visualizations with all required labels  
✅ Separate computation/visualization logic  
✅ Comprehensive docstrings  
✅ Self-contained notebook (no external links)  
✅ Username in all plots  
✅ 100% reproducible results

## Running Doctests

To verify Part 1 implementations:

```bash
cd Q3
python3 -m doctest tree_utils.py -v
python3 -m doctest select_best_binary_split.py -v
```

All tests should pass with no errors.

## Notes

- Large CSV files (>50MB) loaded efficiently with pandas
- Grid searches use `n_jobs=-1` for parallel processing
- Models trained only on training set, validated separately
- No data leakage between train/valid/test sets

## Author

**Student**: likhitbhogadi  
**Course**: SMAI (Statistical Methods in AI)  
**Assignment**: A4 - Question 3  
**Date**: November 3, 2025

---

For detailed implementation notes, see `IMPLEMENTATION_GUIDE.md`.  
For completion checklist, see `SUBMISSION_SUMMARY.md`.
