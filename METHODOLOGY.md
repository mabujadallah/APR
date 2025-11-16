# Bug-Fix Context Characterization Methodology

## Overview

This document describes the comprehensive approach to characterizing bug-fix contexts using features extracted from multiple data sources.

## Context Definition

We define "context" as a multi-dimensional characterization of bug-fix pull requests using features from:

1. **Pull Request Data** - Basic PR information
2. **Commit Data** - Code changes and patches
3. **Issue Data** - Bug reports and feature requests
4. **Review Data** - Code review information
5. **Discussion Data** - Comments and conversations
6. **Timeline Data** - Event chronology and timestamps

## Quantitative Metrics

### 1. Patch Size

**Definition**: Measures the magnitude of code changes in a PR.

**Metrics**:
- `lines_added_proxy`: Estimated lines added
- `lines_deleted_proxy`: Estimated lines deleted
- `total_lines_changed`: Sum of additions and deletions
- `files_changed_proxy`: Number of files modified
- `hunks_proxy`: Number of change hunks
- `patch_complexity`: Overall complexity score (character count)

**Statistical Measures**:
- Mean, median, standard deviation
- Percentiles (25th, 50th, 75th, 90th, 95th)
- Skewness and kurtosis

**Interpretation**:
- Small patches (<50 lines): Quick fixes, typos
- Medium patches (50-500 lines): Standard bug fixes
- Large patches (>500 lines): Major refactoring or multiple issues

### 2. Code Churn

**Definition**: Frequency and volatility of changes in the codebase.

**Metrics**:
- `churn_per_pr`: Average lines changed per PR in a repository
- `file_volatility`: Average files changed per PR
- `total_churn`: Cumulative changes in repository
- `total_files_touched`: Total files modified

**Statistical Measures**:
- Repository-level aggregations
- Temporal patterns
- Distribution analysis

**Interpretation**:
- High churn: Active development or unstable code
- Low churn: Stable codebase or maintenance mode
- Volatility indicates code hotspots

### 3. Discussion

**Definition**: Level of engagement and communication around a PR.

**Metrics**:
- `comment_count_proxy`: Number of comments
- `participants_proxy`: Number of unique participants
- `discussion_length`: Total word count
- `has_discussion`: Binary indicator (threshold: 20 words)

**Statistical Measures**:
- Mean and median comment counts
- Participant distribution
- Discussion length distribution

**Interpretation**:
- High discussion: Controversial or complex changes
- Low discussion: Straightforward fixes or automated PRs
- More participants suggests broader impact

### 4. Reviews

**Definition**: Code review activity and outcomes.

**Metrics**:
- `review_count_proxy`: Number of reviews
- `has_reviews`: Binary indicator
- `approved`: Approval status (1 if merged)
- `changes_requested_proxy`: Whether changes were requested

**Statistical Measures**:
- Review count distribution
- Approval rate
- Review turnaround time

**Interpretation**:
- More reviews: Higher quality assurance
- Quick approval: Trusted contributor or simple fix
- Changes requested: Initial implementation issues

### 5. Timeline

**Definition**: Temporal characteristics of PR lifecycle.

**Metrics**:
- `time_to_close_hours`: Duration from creation to closure
- `is_merged`: Merge status
- `is_closed`: Closure status
- `lifecycle_stage`: Current state (merged/closed/open)

**Statistical Measures**:
- Mean and median closure time
- Distribution by lifecycle stage
- Fast-track rates (<24 hours)

**Interpretation**:
- Quick closure: Urgent fix or simple change
- Slow closure: Complex issue or low priority
- Merged vs closed: Success rate indicator

### 6. Issue Details

**Definition**: Classification and severity of the issue being addressed.

**Metrics**:
- `is_bug_fix`: Bug-related keywords detected
- `is_feature`: Feature-related keywords detected
- `is_test`: Test-related keywords detected
- `is_docs`: Documentation-related keywords detected
- `is_critical`: Critical severity indicators
- `issue_type`: Categorical classification

**Keyword Patterns**:
- **Bug**: bug, fix, fixes, fixed, error, issue, debug, patch, fault, defect, crash
- **Feature**: feature, enhancement, add, implement, new, improve
- **Test**: test, testing, unit, integration, coverage, spec
- **Docs**: doc, docs, documentation, readme, comment
- **Critical**: critical, urgent, blocker, severe, security, vulnerability

**Statistical Measures**:
- Type distribution
- Critical issue rate
- Bug fix percentage

**Interpretation**:
- Bug fixes have different patterns than features
- Critical issues receive priority treatment
- Type affects review and discussion patterns

## Comparative Analysis

### Merged vs Not Merged PRs

We compare accepted (merged) and rejected (not merged) PRs across all metrics using:

1. **Descriptive Statistics**: Mean, median, standard deviation by group
2. **Visual Comparison**: Box plots for each metric
3. **Statistical Testing**: Mann-Whitney U test (non-parametric)
4. **Effect Size**: Practical significance of differences

**Hypotheses**:
- H1: Merged PRs have different patch sizes than rejected PRs
- H2: Merged PRs have more reviews than rejected PRs
- H3: Merged PRs are closed faster than rejected PRs
- H4: Bug fixes have higher merge rates than other types

## Correlation Analysis

**Purpose**: Identify relationships between features and merge success.

**Method**: Pearson correlation with significance testing

**Key Correlations to Examine**:
- Patch size vs merge success
- Review count vs merge success
- Discussion length vs merge success
- Time to close vs merge success
- Issue type vs merge success

## Distributions

All metrics are visualized using:

1. **Histograms**: Show frequency distribution with mean/median overlays
2. **Box Plots**: Show quartiles, outliers, and comparison groups
3. **Pie Charts**: Show categorical distributions
4. **Bar Charts**: Show count distributions by category
5. **Heatmaps**: Show correlation matrices

## Data Quality Considerations

### Proxies and Approximations

When actual data is unavailable, we use proxies:
- **Lines changed**: PR body line count (proxy for actual diff)
- **Files changed**: Estimated from body length
- **Comments**: Estimated from body word count
- **Reviews**: Estimated from merge status

### Handling Missing Data

- Fill missing string fields with empty strings
- Skip records with missing critical fields (merge/close status)
- Use median imputation for missing numeric values
- Document all assumptions

### Sample Size

- Minimum 100 PRs for meaningful analysis
- Stratified sampling maintains label ratios
- Report sample size in all summaries

## Interpretation Guidelines

### Small Dataset (<100 PRs)
- Focus on descriptive statistics
- Avoid strong causal claims
- Use visualizations for patterns
- Report confidence intervals

### Medium Dataset (100-1000 PRs)
- Perform statistical tests
- Examine correlations
- Build predictive models
- Identify significant patterns

### Large Dataset (>1000 PRs)
- Full statistical analysis
- Machine learning models
- Temporal trend analysis
- Subgroup analysis

## Validation

### Internal Validation
- Cross-check metrics for consistency
- Verify statistical assumptions
- Check for data anomalies
- Validate against known cases

### External Validation
- Compare with published research
- Benchmark against similar repositories
- Expert review of findings
- Replication on different datasets

## Reporting

### Summary Statistics Table
- Feature name
- Count, mean, std, min, 25%, 50%, 75%, max
- Skewness and kurtosis

### Comparative Table
- Group comparison (merged vs not merged)
- Statistical test results
- Effect sizes
- Significance levels

### Visualizations
- One plot per metric category
- Clear labels and legends
- Statistical overlays (mean, median)
- Consistent color scheme

### Narrative Summary
- Dataset overview
- Key findings (top 5)
- Interpretation of results
- Limitations and caveats
- Recommendations

## Tools and Technologies

- **Python 3.8+**: Core language
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **scipy**: Statistical tests
- **datasets**: Hugging Face dataset loading
- **jupyter**: Interactive notebooks

## References

1. Hugging Face AI-Dev Dataset: `hao-li/AIDEV`
2. Statistical methods from scipy documentation
3. Visualization best practices from matplotlib/seaborn
4. Software engineering metrics research literature

## Appendix: Code Examples

### Loading Data
```python
from datasets import load_dataset
ds = load_dataset("hao-li/AIDEV", split="train")
df = pd.DataFrame(ds)
```

### Feature Extraction
```python
def extract_patch_size_features(row):
    body = str(row.get('body', ''))
    lines = len(body.splitlines())
    return {
        'total_lines_changed': lines,
        'files_changed_proxy': max(1, lines // 10)
    }
```

### Statistical Testing
```python
from scipy import stats
merged = df[df['is_merged'] == 1]['total_lines_changed']
not_merged = df[df['is_merged'] == 0]['total_lines_changed']
stat, p_value = stats.mannwhitneyu(merged, not_merged)
```

### Visualization
```python
import matplotlib.pyplot as plt
plt.hist(df['total_lines_changed'], bins=50)
plt.axvline(df['total_lines_changed'].mean(), color='red', linestyle='--')
plt.show()
```

---

**Last Updated**: November 2024
**Version**: 1.0
