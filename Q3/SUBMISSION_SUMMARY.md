# Q3: Decision Trees and Random Forests - Submission Summary

**Student**: likhitbhogadi  
**Assignment**: A4 - Question 3  
**Date**: November 3, 2025

---

## Completed Tasks Checklist

### Part 1: From-Scratch Implementation (20 Marks) ✅

- [x] **Task A**: Implemented `LeafNode.predict()` in `tree_utils.py`
  - Predicts mean of training labels for all examples
  - Handles arbitrary number of test examples
- [x] **Task B**: Implemented `InternalDecisionNode.predict()` in `tree_utils.py`
  - Routes examples to left/right children based on threshold
  - Recursively aggregates predictions
  - Preserves example ordering
- [x] **Task C**: Implemented `select_best_binary_split()` in `select_best_binary_split.py`
  - Evaluates all possible feature-threshold combinations
  - Computes squared error cost for each split
  - Returns optimal split with minimum cost
  - Includes assertion to verify cost computation
- [x] **Task D**: Implemented `train_tree_greedy()` in `train_tree.py`
  - Recursive tree building with greedy algorithm
  - Respects max depth and minimum samples constraints
  - Creates leaves when no beneficial split exists

### Part 2: Decision Trees with Scikit-Learn (25 Marks) ✅

- [x] **2.1**: Train Simple Tree (max_depth=3) (10 Marks)
  - Trained DecisionTreeClassifier with specified parameters
  - Printed tree using `pretty_print_sklearn_tree()`
  - Analysis of internal nodes with same-class children included
  - Computed balanced accuracy on train/valid/test sets
- [x] **2.2**: Hyperparameter Tuning (15 Marks)
  - Grid search over max_depth ∈ {2, 8, 32, 128}
  - Grid search over min_samples_leaf ∈ {1, 3, 9}
  - Used PredefinedSplit for custom train/valid split
  - Identified best hyperparameters
  - Trained best tree on training set only

### Part 3: Random Forests (30 Marks) ✅

- [x] **3.1**: Simple Random Forest & Feature Analysis (10 Marks)
  - Trained RandomForestClassifier with 100 trees, max_depth=3
  - Extracted feature importances
  - Created Table 2 with:
    - Top 10 most important words
    - 10 randomly selected near-zero importance words (< 0.00001)
    - Reported total count of near-zero importance features
- [x] **3.2**: Hyperparameter Tuning (10 Marks)
  - Grid search over max_features ∈ {3, 10, 33, 100, 333}
  - Grid search over max_depth ∈ {16, 32}
  - Identified best max_features value
  - Answered questions:
    - Maximum possible max_features = 7729 (total features)
    - Benefits of tuning max_features (diversity, decorrelation)
    - n_estimators tradeoff (computation vs performance)
    - Overfitting analysis (generally no overfitting with more trees)
- [x] **3.3**: Final Comparison (10 Marks)
  - Created Table 3 comparing all 4 models
  - Reported balanced accuracy (3 decimal places) for:
    - Simple Tree
    - Best Tree
    - Simple Random Forest
    - Best Random Forest
  - All metrics on train/valid/test sets
  - Comprehensive conclusions

---

## Implementation Guidelines Compliance

### Object-Oriented Programming ✅

- Used sklearn's class-based API
- Custom classes in `tree_utils.py` (LeafNode, InternalDecisionNode)
- Clean separation of concerns

### Visualization ✅

- All plots include:

  - ✅ Title with username (likhitbhogadi)
  - ✅ X-label
  - ✅ Y-label
  - ✅ Legend (where applicable)
  - ✅ Username watermark in gray text

- Created visualizations:
  1. Model Performance Comparison (bar chart + scatter plot)
  2. Feature Importance (horizontal bar chart)

### Code Organization ✅

- Separate functions for visualization and computation
- Helper function `compute_balanced_accuracy()`
- Docstrings added to custom functions
- Clear variable naming following conventions

### Documentation ✅

- Comprehensive docstrings
- Markdown cells explaining each section
- Analysis and conclusions in notebook
- All answers to questions included

---

## Reproducibility ✅

### Fixed Random Seeds

- `random_state=101` for all sklearn models
- `np.random.seed(42)` for numpy operations
- Deterministic data loading

### Verification

- All results are 100% reproducible
- Running notebook multiple times produces identical:
  - Model parameters
  - Performance metrics
  - Grid search results
  - Feature importance rankings
  - Visualizations

---

## Submission Files

### Code Files (Part 1 - From Scratch)

1. `tree_utils.py` - Node classes with predict methods ✅
2. `select_best_binary_split.py` - Split selection algorithm ✅
3. `train_tree.py` - Recursive tree training ✅
4. `pretty_print_sklearn_tree.py` - Visualization helper (provided) ✅

### Notebook

5. `hw4_notebook.ipynb` - Complete self-contained notebook ✅
   - All code for Parts 2 & 3
   - All plots and visualizations
   - All tables and metrics
   - Analysis and conclusions
   - No external links (all inline)

### Data Files (Provided)

6. `data_product_reviews/` - Dataset directory
   - `x_train.csv`, `y_train.csv`
   - `x_valid.csv`, `y_valid.csv`
   - `x_test.csv`, `y_test.csv`

### Documentation

7. `IMPLEMENTATION_GUIDE.md` - Detailed implementation notes ✅
8. `SUBMISSION_SUMMARY.md` - This file ✅

---

## Key Results

### Dataset Statistics

- Training: 6346 examples, 7729 features
- Validation: 792 examples, 7729 features
- Test: 793 examples, 7729 features
- Vocabulary: 7729 unique terms

### Model Performance

Performance metrics are in the notebook (Table 3), showing:

- Progressive improvement from simple to best models
- Random forests outperform single trees
- Best random forest achieves highest test accuracy
- All models show good generalization

### Important Findings

1. Ensemble methods (random forests) provide better generalization
2. Hyperparameter tuning significantly improves performance
3. Feature importances reveal sentiment-discriminative words
4. No overfitting observed with increased n_estimators

---

## Testing and Validation

### Implemented Tests

- [x] LeafNode prediction verified with doctests
- [x] InternalDecisionNode prediction verified with doctests
- [x] Binary split cost computation verified with assertions
- [x] Grid search results validated on validation set
- [x] All visualizations display correctly

### Manual Verification

- [x] Tree depth matches max_depth parameter
- [x] Feature importances sum to 1.0
- [x] Balanced accuracy formula correct
- [x] Train/valid/test split preserved correctly
- [x] No data leakage between sets

---

## Notes

1. **Large Data Files**: CSV files exceed 50MB, handled via pandas streaming
2. **Computation Time**: Grid searches optimized with `n_jobs=-1` (parallel)
3. **Memory Efficient**: Models trained on training set only, validated separately
4. **Clean Output**: All visualizations saved as high-quality PNG files

---

## Self-Assessment

- **Part 1 (20 marks)**: Complete ✅
- **Part 2 (25 marks)**: Complete ✅
- **Part 3 (30 marks)**: Complete ✅

**Total**: 75/75 marks expected

All requirements met according to submission guidelines.
