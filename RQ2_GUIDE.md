# RQ2: Differences Between Accepted and Rejected Fixes

## Research Question 2

**How do accepted (merged) and rejected (not merged) bug fixes differ across all RQ1 features?**

## Overview

RQ2 performs comprehensive statistical analysis and machine learning to understand what distinguishes accepted bug fixes from rejected ones. This analysis helps identify key factors that predict merge acceptance.

## Analysis Components

### 1. Statistical Tests

We apply multiple statistical tests to compare accepted vs rejected bug fixes:

#### Continuous Features (Parametric and Non-Parametric Tests)

**Tests Applied:**
- **Independent t-test** (parametric): Assumes normal distribution
- **Mann-Whitney U test** (non-parametric): No distribution assumption
- **Wilcoxon rank-sum test** (non-parametric): Alternative to Mann-Whitney

**Features Tested:**
- `total_lines_changed` - Patch size
- `files_changed_proxy` - Number of files modified
- `patch_complexity` - Overall complexity score
- `churn_per_pr` - Code churn per PR
- `file_volatility` - File change frequency
- `comment_count_proxy` - Discussion activity
- `participants_proxy` - Number of contributors
- `review_count_proxy` - Code review activity
- `time_to_close_hours` - Time until closure

**Output Metrics:**
- p-values for each test
- Effect sizes (Cohen's d)
- Significance markers: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant

**Effect Size Interpretation:**
- Small: |d| < 0.5
- Medium: 0.5 ≤ |d| < 0.8
- Large: |d| ≥ 0.8

#### Categorical Features (Chi-Square Test)

**Test Applied:**
- **Chi-square test**: Tests independence between categorical variables

**Features Tested:**
- `is_critical` - Critical severity indicator
- `has_discussion` - Presence of discussion
- `has_reviews` - Presence of reviews

**Output Metrics:**
- Chi-square statistic (χ²)
- p-value
- Degrees of freedom
- Cramér's V (effect size for chi-square)
- Contingency tables

### 2. Machine Learning Classifiers

We build predictive models to classify bug fixes as likely to be merged or rejected.

#### Models Trained

1. **Logistic Regression**
   - Linear model with probability output
   - Interpretable coefficients
   - Fast training and prediction
   - Best for: Understanding linear relationships

2. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Provides feature importance
   - Best for: Robust predictions, feature ranking

3. **Gradient Boosting**
   - Sequential ensemble method
   - Often highest accuracy
   - Complex but powerful
   - Best for: Maximum predictive performance

#### Performance Metrics

For each classifier, we compute:

- **Accuracy**: Overall correct prediction rate
- **Precision**: Of predicted merges, how many were actually merged
- **Recall**: Of actual merges, how many were predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (0.5=random, 1.0=perfect)

**Interpretation Guidelines:**
- ROC-AUC > 0.9: Excellent
- ROC-AUC > 0.8: Good
- ROC-AUC > 0.7: Acceptable
- ROC-AUC > 0.6: Poor
- ROC-AUC ≤ 0.5: No better than random

#### Cross-Validation

5-fold stratified cross-validation to ensure:
- Stable performance estimates
- No overfitting
- Preserve class distribution in folds

### 3. Feature Importance Analysis

#### Methods

1. **Random Forest Feature Importance**
   - Based on decrease in node impurity
   - Measures feature contribution to splits

2. **Gradient Boosting Feature Importance**
   - Based on gain from splits
   - Accounts for sequential boosting

3. **Logistic Regression Coefficients**
   - Absolute values of coefficients
   - Direct linear impact on log-odds

**Aggregation:**
- Average importance across all models
- Normalized to sum to 100%
- Ranked from highest to lowest

#### Top Predictors

The analysis identifies the most discriminative features:
- Top 5 features displayed prominently
- Importance scores (% contribution)
- Direction of effect (higher/lower for accepted)
- Statistical significance from tests

### 4. Visualizations

#### Comparison Plots
- **Box plots** for each continuous feature
- Side-by-side comparison: Accepted vs Rejected
- Statistical significance annotations
- Mean/median markers
- Outlier identification

#### Feature Importance Charts
- **Horizontal bar chart** of top 10 features
- **Grouped bar chart** comparing models
- Percentage contribution
- Clear ranking

#### ROC Curves
- **Curves for all classifiers**
- AUC scores displayed
- Random classifier baseline (diagonal line)
- Model comparison on single plot

## Output Files

### Data Files (CSV)

1. **rq2_continuous_tests.csv**
   - Statistical test results for continuous features
   - Columns: Feature, Accepted_Mean, Rejected_Mean, T_Stat, T_PValue, 
             MW_Stat, MW_PValue, WR_Stat, WR_PValue, Cohens_D

2. **rq2_categorical_tests.csv**
   - Chi-square test results for categorical features
   - Columns: Feature, Chi2, PValue, DOF, Cramers_V

3. **rq2_feature_importance.csv**
   - Feature importance from all models
   - Columns: Feature, Random_Forest, Gradient_Boosting, 
             Logistic_Regression, Average

### Visualizations (PNG)

1. **rq2_feature_comparison.png**
   - 3x3 grid of box plots
   - All continuous features compared
   - Statistical significance markers

2. **rq2_feature_importance.png**
   - Top 10 features by importance
   - Model-specific and average importance
   - Clear ranking visualization

3. **rq2_roc_curves.png**
   - ROC curves for all classifiers
   - AUC scores in legend
   - Performance comparison

## Usage

### Standalone Script

```bash
python3 rq2_analysis.py
```

This will:
1. Load data (or create sample data)
2. Extract features
3. Perform all statistical tests
4. Train all classifiers
5. Generate all visualizations
6. Save results to /tmp/

### As Part of Full Analysis

The RQ2 analysis is integrated into the main notebook:
```bash
jupyter notebook bug_fix_context_characterization.ipynb
```

Execute all cells including Section 9 (RQ2) for the complete analysis.

## Interpretation Guide

### Statistical Significance

**p-value < 0.05**: Statistically significant difference
- Feature shows reliable difference between accepted and rejected
- Not due to random chance
- Consider in decision-making

**p-value ≥ 0.05**: Not statistically significant
- No reliable difference detected
- Could be due to random variation
- Less useful for prediction

### Effect Size

Even if significant, effect size matters:
- **Large effect**: Practically important difference
- **Small effect**: Statistically significant but may not matter in practice

### Classifier Performance

**High accuracy but low precision:**
- Model predicts many merges (inclusive)
- Many false positives
- Use for early screening

**High precision but low recall:**
- Model is conservative
- Misses some true merges
- Use for quality assurance

**High F1-score:**
- Balanced performance
- Good all-around predictor
- Recommended for general use

### Feature Importance

**Top features** are most discriminative:
- Focus improvement efforts here
- Monitor these metrics closely
- Use as acceptance criteria

**Low importance features:**
- Less predictive of outcome
- May still be important for other reasons
- Don't ignore completely

## Example Results

### Typical Findings

Based on analysis of sample data:

1. **Time to close** is often the top predictor
   - Accepted fixes close faster
   - Quick turnaround indicates clear fixes

2. **Review count** strongly correlates with acceptance
   - More reviews → higher quality
   - Peer validation important

3. **Patch size** shows mixed results
   - Very large or very small patches may be rejected
   - Moderate size optimal

4. **Discussion length** can go both ways
   - Long discussion may indicate controversy
   - Or indicate thorough review

### Statistical Test Results Example

```
Feature              T-Test          Mann-Whitney    Wilcoxon        Effect Size
--------------------------------------------------------------------------------
lines_changed        p=0.0123 *      p=0.0089 **     p=0.0091 **     d=0.523 (med)
review_count         p=0.0001 ***    p=0.0001 ***    p=0.0001 ***    d=0.892 (large)
time_to_close        p=0.0450 *      p=0.0378 *      p=0.0382 *      d=-0.381 (small)
```

**Interpretation:**
- `review_count` shows large, highly significant difference
- Accepted PRs have more reviews (positive effect)
- Strong predictor of acceptance

### Classifier Performance Example

```
Classifier           Accuracy   Precision  Recall     F1         ROC-AUC
----------------------------------------------------------------------
Logistic Regression  0.725      0.780      0.820      0.800      0.785
Random Forest        0.768      0.801      0.848      0.824      0.823
Gradient Boosting    0.782      0.815      0.851      0.833      0.841
```

**Interpretation:**
- Gradient Boosting performs best (F1=0.833, AUC=0.841)
- All models show good predictive power (AUC > 0.78)
- Can reliably distinguish accepted from rejected

### Feature Importance Example

```
Feature                        Importance
----------------------------------------
review_count                   24.5%
time_to_close                  18.3%
has_reviews                    12.7%
comment_count                  11.2%
patch_complexity               9.8%
```

**Interpretation:**
- Reviews are the most important factor (24.5%)
- Timeline matters (18.3%)
- Top 5 features account for 76.5% of predictive power

## Recommendations

Based on RQ2 findings:

### For Contributors

1. **Seek reviews early**: High review count strongly predicts acceptance
2. **Respond quickly**: Time to close correlates with merge success
3. **Engage in discussion**: Shows commitment and addresses concerns
4. **Keep patches focused**: Moderate size works best

### For Reviewers

1. **Prioritize based on predictions**: Use classifier to identify promising PRs
2. **Monitor key metrics**: Focus on top predictors
3. **Provide timely feedback**: Reduce time to close for good PRs
4. **Request changes clearly**: Help contributors improve quickly

### For Project Managers

1. **Track acceptance metrics**: Monitor trends over time
2. **Set thresholds**: Use top predictors as quality gates
3. **Optimize review process**: Ensure timely, thorough reviews
4. **Provide feedback loops**: Help contributors learn from rejections

## Limitations

1. **Sample size**: Small datasets may show unstable patterns
2. **Proxy features**: Some metrics are approximations
3. **Causation vs correlation**: High correlation doesn't imply causation
4. **Context-specific**: Patterns may vary by project/domain
5. **Temporal effects**: Patterns may change over time

## Future Work

1. **Temporal analysis**: Track how patterns evolve
2. **Subgroup analysis**: Different patterns for different PR types
3. **Ensemble models**: Combine multiple classifiers
4. **Deep learning**: Neural networks for complex patterns
5. **Causal inference**: Establish causal relationships

## References

- Mann-Whitney U test: scipy.stats.mannwhitneyu
- Chi-square test: scipy.stats.chi2_contingency
- Wilcoxon rank-sum: scipy.stats.ranksums
- Scikit-learn classifiers: sklearn.ensemble, sklearn.linear_model
- Effect sizes: Cohen's d, Cramér's V

---

**Last Updated**: November 2024
**Version**: 1.0
