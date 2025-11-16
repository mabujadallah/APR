# APR - Automated Program Repair Analysis

This repository contains comprehensive analysis notebooks for characterizing bug-fix contexts in AI-assisted software development.

## Overview

The analysis focuses on characterizing bug-fix contexts using features extracted from multiple data sources including pull requests, commits, issues, reviews, discussions, and timelines.

## Notebooks

### 1. `bug_fix_context_characterization.ipynb`
**Comprehensive Bug-Fix Context Characterization**

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

### 2. Other Notebooks

- `AIDEV_Agent_final_reviewed.ipynb` - Reviewed analysis with improved copilot detection
- `AIDEV_Agent_final.ipynb` - AI-Dev dataset analysis focused on Copilot
- `AIDev_Exploration.ipynb` - Initial exploratory data analysis

## Requirements

```bash
pip install datasets pandas numpy matplotlib seaborn scipy jupyter notebook
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt  # if available
# or
pip install datasets pandas numpy matplotlib seaborn scipy jupyter notebook
```

2. Run the notebook:
```bash
jupyter notebook bug_fix_context_characterization.ipynb
```

3. Execute all cells to generate the complete analysis

## Data Source

The analysis uses the AI-Dev dataset from Hugging Face: `hao-li/AIDEV`

If the dataset is unavailable, the notebook automatically generates synthetic sample data for demonstration purposes.

## Key Findings

The analysis reveals:
1. Patch sizes follow a long-tailed distribution
2. Code churn correlates with discussion activity
3. Merged PRs show distinct patterns in review count and response time
4. Bug fixes exhibit different characteristics than feature additions
5. Critical issues receive faster attention and more thorough reviews

## Statistical Methods

- **Descriptive Statistics**: Comprehensive summary statistics for all metrics
- **Distribution Analysis**: Histograms, box plots, and density plots
- **Comparative Analysis**: Mann-Whitney U tests for merged vs not merged PRs
- **Correlation Analysis**: Pearson correlation with significance testing

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.