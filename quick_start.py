#!/usr/bin/env python3
"""
Quick Start Example for Bug-Fix Context Characterization

This script demonstrates how to use the bug_fix_context_characterization notebook
programmatically or as a standalone script.
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the AI-Dev dataset or create sample data"""
    try:
        print("📥 Loading AI-Dev dataset...")
        ds = load_dataset("hao-li/AIDEV", split="train")
        df = pd.DataFrame(ds)
        print(f"✅ Loaded {len(df)} records from AI-Dev dataset")
        return df
    except Exception as e:
        print(f"⚠️  Could not load dataset: {e}")
        print("📊 Creating sample data for demonstration...")
        
        # Create synthetic sample data
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'id': range(n_samples),
            'number': range(1, n_samples + 1),
            'title': [f'Fix bug #{i}' if i % 3 == 0 else f'Feature #{i}' for i in range(n_samples)],
            'body': [f'This PR fixes issue #{i}\n' * np.random.randint(1, 20) for i in range(n_samples)],
            'state': np.random.choice(['closed', 'open'], n_samples, p=[0.8, 0.2]),
            'merged_at': [pd.Timestamp.now() if np.random.random() > 0.3 else None for _ in range(n_samples)],
            'created_at': [pd.Timestamp.now() for _ in range(n_samples)],
            'agent': np.random.choice(['copilot', 'human', 'other'], n_samples, p=[0.3, 0.5, 0.2]),
        })
        print(f"✅ Created {len(df)} sample records")
        return df

def extract_basic_features(df):
    """Extract basic context features"""
    print("\n🔍 Extracting features...")
    
    # Patch size (using body length as proxy)
    df['lines_changed'] = df['body'].apply(lambda x: len(str(x).splitlines()))
    df['files_changed'] = df['lines_changed'].apply(lambda x: max(1, x // 10))
    
    # Issue type
    df['is_bug_fix'] = df['title'].str.contains('fix|bug', case=False, na=False).astype(int)
    
    # Merge status
    df['is_merged'] = df['merged_at'].notna().astype(int)
    
    print(f"✅ Features extracted")
    return df

def generate_summary(df):
    """Generate summary statistics"""
    print("\n📊 SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\n📈 Dataset Overview:")
    print(f"  Total PRs: {len(df)}")
    print(f"  Merged: {df['is_merged'].sum()} ({df['is_merged'].mean()*100:.1f}%)")
    print(f"  Bug fixes: {df['is_bug_fix'].sum()} ({df['is_bug_fix'].mean()*100:.1f}%)")
    
    print(f"\n📏 Patch Size:")
    print(f"  Mean lines changed: {df['lines_changed'].mean():.2f}")
    print(f"  Median lines changed: {df['lines_changed'].median():.2f}")
    print(f"  95th percentile: {df['lines_changed'].quantile(0.95):.2f}")
    
    print(f"\n🔀 Comparison - Merged vs Not Merged:")
    comparison = df.groupby('is_merged')['lines_changed'].agg(['mean', 'median', 'count'])
    comparison.index = ['Not Merged', 'Merged']
    print(comparison)

def create_visualization(df):
    """Create basic visualization"""
    print("\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution of lines changed
    axes[0].hist(df['lines_changed'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['lines_changed'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["lines_changed"].mean():.1f}')
    axes[0].set_title('Distribution of Lines Changed')
    axes[0].set_xlabel('Lines Changed')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Merged vs Not Merged
    merged_data = df[df['is_merged'] == 1]['lines_changed']
    not_merged_data = df[df['is_merged'] == 0]['lines_changed']
    axes[1].boxplot([merged_data, not_merged_data], labels=['Merged', 'Not Merged'])
    axes[1].set_title('Lines Changed: Merged vs Not Merged')
    axes[1].set_ylabel('Lines Changed')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/quick_start_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to: /tmp/quick_start_visualization.png")
    plt.close()

def main():
    """Main execution flow"""
    print("🚀 Bug-Fix Context Characterization - Quick Start")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Extract features
    df = extract_basic_features(df)
    
    # Generate summary
    generate_summary(df)
    
    # Create visualization
    create_visualization(df)
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("\n💡 For full analysis, run: jupyter notebook bug_fix_context_characterization.ipynb")
    print("=" * 60)

if __name__ == "__main__":
    main()
