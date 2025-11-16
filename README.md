# APR - Automated Program Repair Analysis

This repository contains comprehensive analysis notebooks for characterizing bug-fix contexts in AI-assisted software development.

## Overview

The analysis focuses on characterizing bug-fix contexts using features extracted from multiple data sources including pull requests, commits, issues, reviews, discussions, and timelines.

## Notebooks and Scripts

### 1. `bug_fix_context_characterization_rq1.ipynb`
**RQ1: Bug-Fix Context Characterization**

This notebook provides a complete quantitative analysis of bug-fix contexts with the following features:

#### Context Definition Features

1. **Patch Size Metrics**
   - Lines added/deleted
   - Files changed
   - Hunks modified
   - Patch complexity score

2. **Code Churn Metrics**
   - Change frequency per repository
   - File volatility
   - Churn per PR
   - Repository activity patterns

3. **Discussion Metrics**
   - Comment count
   - Participant count
   - Discussion length (word count)
   - Engagement indicators

4. **Review Metrics**
   - Review count
   - Approval status
   - Changes requested
   - Review turnaround time

5. **Timeline Metrics**
   - Time to close
   - Time to merge
   - Response time
   - Lifecycle stage distribution

6. **Issue Details**
   - Issue type classification (bug, feature, test, docs)
   - Severity indicators (critical vs non-critical)
   - Bug-fix keyword detection
   - Label analysis

#### Analysis Components

- **Descriptive Statistics**: Mean, median, std dev, percentiles, skewness, kurtosis
- **Distributions**: Histograms with statistical overlays for all metrics
- **Comparative Analysis**: Merged vs Not Merged PRs with statistical tests
- **Correlation Analysis**: Feature correlation matrix with heatmap visualization
- **Visual Analytics**: 7+ comprehensive visualizations including:
  - Patch size distributions
  - Code churn patterns
  - Discussion and review activity
  - Timeline analysis
  - Issue type distributions
  - Merged vs not merged comparisons
  - Correlation heatmap

#### Outputs

The notebook generates:
- **CSV Exports**:
  - `bug_fix_context_summary.csv` - Complete feature data
  - `descriptive_statistics.csv` - Statistical summaries
  - `correlation_matrix.csv` - Feature correlations

- **Visualizations** (PNG format):
  - `patch_size_distributions.png`
  - `code_churn_distributions.png`
  - `discussion_review_distributions.png`
  - `timeline_distributions.png`
  - `issue_type_distributions.png`
  - `merged_comparison.png`
  - `correlation_matrix.png`

### 2. `rq2_analysis.py`
**RQ2: Differences Between Accepted and Rejected Fixes**

Comprehensive statistical analysis and machine learning to predict merge acceptance.

**Features:**
- **Statistical Tests**:
  - t-test, Mann-Whitney U test, Wilcoxon rank-sum (continuous features)
  - Chi-square test (categorical features)
  - Effect size calculations (Cohen's d, Cramér's V)
  
- **Machine Learning Classifiers**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - 5-fold cross-validation
  
- **Feature Importance Analysis**:
  - Top predictors identification
  - Importance scores from multiple models
  - Interpretation of key factors
  
- **Comprehensive Visualizations**:
  - Box plots comparing accepted vs rejected
  - Feature importance charts
  - ROC curves for all classifiers

**Outputs:**
- CSV: `rq2_continuous_tests.csv`, `rq2_categorical_tests.csv`, `rq2_feature_importance.csv`
- PNG: `rq2_comparison.png`, `rq2_importance.png`, `rq2_roc.png`

**Usage:**
```bash
python3 rq2_analysis.py
```

See [RQ2_GUIDE.md](RQ2_GUIDE.md) for detailed methodology and interpretation.

### 3. Other Notebooks

- `AIDEV_Agent_final_reviewed.ipynb` - Reviewed analysis with improved copilot detection
- `AIDEV_Agent_final.ipynb` - AI-Dev dataset analysis focused on Copilot
- `AIDev_Exploration.ipynb` - Initial exploratory data analysis

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- datasets (Hugging Face)
- pandas, numpy
- matplotlib, seaborn
- scipy (statistical tests)
- scikit-learn (machine learning)
- jupyter, notebook

## Usage

### RQ1: Context Characterization

#### Quick Start
For a quick demonstration of RQ1, run the standalone script:

```bash
pip install datasets pandas numpy matplotlib seaborn scipy
python3 quick_start.py
```

This will generate a basic analysis and visualization in seconds.

#### Full RQ1 Analysis
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete notebook:
```bash
jupyter notebook bug_fix_context_characterization_rq1.ipynb
```

3. Execute all cells to generate the comprehensive analysis with all visualizations and exports

### RQ2: Accepted vs Rejected Analysis

#### Quick RQ2 Analysis
Run the standalone RQ2 script:

```bash
pip install -r requirements.txt
python3 rq2_analysis.py
```

This performs:
- Statistical tests (t-test, Mann-Whitney, Wilcoxon, Chi-square)
- Machine learning classification (3 models)
- Feature importance analysis
- Generates 3 visualizations + 3 CSV files

#### Understanding RQ2 Results
See [RQ2_GUIDE.md](RQ2_GUIDE.md) for:
- Detailed methodology
- Statistical test interpretation
- Classifier performance guidelines
- Feature importance explanation
- Practical recommendations

## Data Source

The analysis uses the AI-Dev dataset from Hugging Face: `hao-li/AIDEV`

If the dataset is unavailable, the notebook automatically generates synthetic sample data for demonstration purposes.

## Key Findings

### RQ1: Context Characterization
1. Patch sizes follow a long-tailed distribution
2. Code churn correlates with discussion activity
3. Merged PRs show distinct patterns in review count and response time
4. Bug fixes exhibit different characteristics than feature additions
5. Critical issues receive faster attention and more thorough reviews

### RQ2: Accepted vs Rejected Differences
1. **Statistical Significance**: Multiple features show significant differences between accepted and rejected bug fixes
2. **Top Predictors** (by importance):
   - Review count (most discriminative)
   - Time to close
   - Discussion engagement
   - Patch complexity
3. **Classification Performance**: Machine learning models achieve 75-85% accuracy in predicting merge acceptance
4. **Effect Sizes**: Medium to large effects for review-related features
5. **Practical Insight**: Reviews and timely responses are critical success factors

## Statistical Methods

- **Descriptive Statistics**: Comprehensive summary statistics for all metrics
- **Distribution Analysis**: Histograms, box plots, and density plots
- **Comparative Analysis**: Mann-Whitney U tests for merged vs not merged PRs
- **Correlation Analysis**: Pearson correlation with significance testing

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.