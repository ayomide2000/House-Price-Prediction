"""
Exploratory Data Analysis (EDA) with Advanced Visualizations
Kaggle House Prices: Advanced Regression Techniques

This script provides comprehensive visualization-focused EDA including:
- Scatter plots for feature-price relationships
- Box plots for categorical distributions
- Violin plots for distribution shapes
- Multi-variate analysis (correlations, pair plots)
- Trivariate analysis (3D relationships)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

def load_data(train_path='train.csv', test_path='test.csv'):
    """Load and prepare training and test data"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print("="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"\nTrain set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"\nTarget variable (SalePrice) statistics:")
    print(train['SalePrice'].describe())
    
    return train, test

# ============================================================================
# 2. DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality(train):
    """Assess missing values and data types"""
    print("\n" + "="*80)
    print("DATA QUALITY ASSESSMENT")
    print("="*80)
    
    # Missing values
    missing = train.isnull().sum()
    missing_pct = (missing / len(train)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Missing_Count', ascending=False)
    
    print("\nMissing Values:")
    print(missing_df)
    
    # Data types
    print("\nData Types Distribution:")
    print(train.dtypes.value_counts())
    
    return missing_df

# ============================================================================
# 3. SCATTER PLOTS: Feature vs Price Relationships
# ============================================================================

def plot_scatter_plots(train):
    """Create scatter plots for numerical features vs SalePrice"""
    print("\n" + "="*80)
    print("SCATTER PLOTS: Feature vs SalePrice Relationships")
    print("="*80)
    
    # Select top numerical features with highest correlation
    numerical_cols = train.select_dtypes(include=[np.number]).columns
    correlations = train[numerical_cols].corr()['SalePrice'].sort_values(ascending=False)
    top_features = correlations[1:9].index  # Top 8 features (excluding SalePrice)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Scatter Plots: Top Features vs SalePrice', fontsize=16, y=1.00)
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features):
        axes[idx].scatter(train[feature], train['SalePrice'], alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('SalePrice', fontsize=10)
        axes[idx].set_title(f'Correlation: {correlations[feature]:.3f}', fontsize=9)
        
        # Add trend line
        z = np.polyfit(train[feature].dropna(), train.loc[train[feature].notna(), 'SalePrice'], 1)
        p = np.poly1d(z)
        axes[idx].plot(train[feature].sort_values(), p(train[feature].sort_values()), 
                      "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('01_scatter_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_scatter_plots.png")
    plt.show()

# ============================================================================
# 4. BOX PLOTS: Categorical Features vs Price
# ============================================================================

def plot_box_plots(train):
    """Create box plots for categorical features vs SalePrice"""
    print("\n" + "="*80)
    print("BOX PLOTS: Categorical Features vs SalePrice")
    print("="*80)
    
    # Select top categorical features
    categorical_cols = train.select_dtypes(include=['object']).columns
    selected_cats = ['Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 
                     'ExterQual', 'KitchenQual', 'FireplaceQu', 'GarageType']
    selected_cats = [col for col in selected_cats if col in train.columns][:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Box Plots: Categorical Features vs SalePrice', fontsize=16, y=1.00)
    axes = axes.ravel()
    
    for idx, feature in enumerate(selected_cats):
        if feature in train.columns:
            sns.boxplot(data=train, x=feature, y='SalePrice', ax=axes[idx], palette='Set2')
            axes[idx].set_xlabel(feature, fontsize=10)
            axes[idx].set_ylabel('SalePrice', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].set_title(f'{feature} Distribution', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('02_box_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_box_plots.png")
    plt.show()

# ============================================================================
# 5. VIOLIN PLOTS: Distribution Shapes with Categories
# ============================================================================

def plot_violin_plots(train):
    """Create violin plots to show distribution shapes"""
    print("\n" + "="*80)
    print("VIOLIN PLOTS: Distribution Shapes by Categories")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Violin Plots: Distribution Shapes by Categorical Features', fontsize=16, y=1.00)
    axes = axes.ravel()
    
    violin_features = ['BldgType', 'HouseStyle', 'OverallQual', 'ExterQual', 'KitchenQual', 'Neighborhood']
    
    for idx, feature in enumerate(violin_features):
        if feature in train.columns:
            sns.violinplot(data=train, x=feature, y='SalePrice', ax=axes[idx], palette='muted')
            axes[idx].set_xlabel(feature, fontsize=10)
            axes[idx].set_ylabel('SalePrice', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].set_title(f'{feature} Distribution', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('03_violin_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_violin_plots.png")
    plt.show()

# ============================================================================
# 6. MULTI-VARIATE ANALYSIS: Correlations and Relationships
# ============================================================================

def plot_correlation_heatmap(train):
    """Create correlation heatmap for numerical features"""
    print("\n" + "="*80)
    print("MULTI-VARIATE ANALYSIS: Correlation Heatmap")
    print("="*80)
    
    # Select top numerical features
    numerical_cols = train.select_dtypes(include=[np.number]).columns
    top_cols = train[numerical_cols].corr()['SalePrice'].abs().sort_values(ascending=False)[:15].index
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = train[top_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, ax=ax, cbar_kws={'shrink': 0.8}, linewidths=0.5)
    ax.set_title('Correlation Matrix: Top 15 Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_correlation_heatmap.png")
    plt.show()

def plot_pairplot(train):
    """Create pairplot for top features"""
    print("\nGenerating pairplot (this may take a moment)...")
    
    # Select top 5 features
    numerical_cols = train.select_dtypes(include=[np.number]).columns
    top_cols = train[numerical_cols].corr()['SalePrice'].abs().sort_values(ascending=False)[1:6].index.tolist()
    top_cols.append('SalePrice')
    
    pairplot = sns.pairplot(train[top_cols], diag_kind='kde', plot_kws={'alpha': 0.6}, 
                            diag_kws={'shade': True})
    pairplot.fig.suptitle('Pairplot: Top 5 Features Relationships', fontsize=14, y=1.00)
    plt.savefig('05_pairplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_pairplot.png")
    plt.show()

# ============================================================================
# 7. TRIVARIATE ANALYSIS: 3D Relationships
# ============================================================================

def plot_3d_scatter(train):
    """Create 3D scatter plot showing three feature interactions"""
    print("\n" + "="*80)
    print("TRIVARIATE ANALYSIS: 3D Feature Relationships")
    print("="*80)
    
    # Select top 3 features for 3D visualization
    numerical_cols = train.select_dtypes(include=[np.number]).columns
    top_3_features = train[numerical_cols].corr()['SalePrice'].abs().sort_values(ascending=False)[1:4].index
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D Scatter 1: Top 3 features
    ax1 = fig.add_subplot(221, projection='3d')
    scatter1 = ax1.scatter(train[top_3_features[0]], train[top_3_features[1]], 
                          train['SalePrice'], c=train[top_3_features[2]], 
                          cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax1.set_xlabel(top_3_features[0], fontsize=9)
    ax1.set_ylabel(top_3_features[1], fontsize=9)
    ax1.set_zlabel('SalePrice', fontsize=9)
    ax1.set_title(f'3D: {top_3_features[0]} vs {top_3_features[1]} vs SalePrice\n(colored by {top_3_features[2]})', 
                 fontsize=10)
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, label=top_3_features[2])
    
    # 3D Scatter 2: Different feature combination
    ax2 = fig.add_subplot(222, projection='3d')
    scatter2 = ax2.scatter(train[top_3_features[0]], train['SalePrice'], 
                          train[top_3_features[2]], c=train[top_3_features[1]], 
                          cmap='plasma', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.set_xlabel(top_3_features[0], fontsize=9)
    ax2.set_ylabel('SalePrice', fontsize=9)
    ax2.set_zlabel(top_3_features[2], fontsize=9)
    ax2.set_title(f'3D: {top_3_features[0]} vs SalePrice vs {top_3_features[2]}\n(colored by {top_3_features[1]})', 
                 fontsize=10)
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label=top_3_features[1])
    
    # 3D Scatter 3: Different angle
    ax3 = fig.add_subplot(223, projection='3d')
    scatter3 = ax3.scatter(train[top_3_features[1]], train[top_3_features[2]], 
                          train['SalePrice'], c=train[top_3_features[0]], 
                          cmap='coolwarm', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax3.set_xlabel(top_3_features[1], fontsize=9)
    ax3.set_ylabel(top_3_features[2], fontsize=9)
    ax3.set_zlabel('SalePrice', fontsize=9)
    ax3.set_title(f'3D: {top_3_features[1]} vs {top_3_features[2]} vs SalePrice\n(colored by {top_3_features[0]})', 
                 fontsize=10)
    plt.colorbar(scatter3, ax=ax3, shrink=0.5, label=top_3_features[0])
    
    # 2D projection showing all interactions
    ax4 = fig.add_subplot(224)
    scatter4 = ax4.scatter(train[top_3_features[0]], train[top_3_features[1]], 
                          c=train['SalePrice'], s=100, cmap='RdYlGn', alpha=0.6, 
                          edgecolors='k', linewidth=0.5)
    ax4.set_xlabel(top_3_features[0], fontsize=9)
    ax4.set_ylabel(top_3_features[1], fontsize=9)
    ax4.set_title(f'2D Projection: {top_3_features[0]} vs {top_3_features[1]}\n(colored by SalePrice)', fontsize=10)
    plt.colorbar(scatter4, ax=ax4, label='SalePrice')
    
    plt.tight_layout()
    plt.savefig('06_trivariate_3d_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_trivariate_3d_analysis.png")
    plt.show()

# ============================================================================
# 8. ADVANCED TRIVARIATE: Interaction Effects
# ============================================================================

def plot_interaction_analysis(train):
    """Analyze and visualize interaction effects between features"""
    print("\n" + "="*80)
    print("ADVANCED TRIVARIATE: Interaction Effects Analysis")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Interaction 1: Overall Quality + Year Built + Price
    if 'OverallQual' in train.columns and 'YearBuilt' in train.columns:
        ax = axes[0, 0]
        for quality in sorted(train['OverallQual'].unique()):
            subset = train[train['OverallQual'] == quality]
            ax.scatter(subset['YearBuilt'], subset['SalePrice'], alpha=0.6, 
                      label=f'Quality {quality}', s=50, edgecolors='k', linewidth=0.5)
        ax.set_xlabel('Year Built', fontsize=10)
        ax.set_ylabel('SalePrice', fontsize=10)
        ax.set_title('Interaction: OverallQual × YearBuilt → SalePrice', fontsize=11, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Interaction 2: Building Type + Neighborhood + Price
    if 'BldgType' in train.columns and 'Neighborhood' in train.columns:
        ax = axes[0, 1]
        top_neighborhoods = train['Neighborhood'].value_counts().head(4).index
        subset = train[train['Neighborhood'].isin(top_neighborhoods)]
        
        for bldg_type in subset['BldgType'].unique():
            bldg_subset = subset[subset['BldgType'] == bldg_type]
            ax.scatter(bldg_subset['Neighborhood'], bldg_subset['SalePrice'], 
                      alpha=0.6, label=bldg_type, s=50, edgecolors='k', linewidth=0.5)
        ax.set_xlabel('Neighborhood', fontsize=10)
        ax.set_ylabel('SalePrice', fontsize=10)
        ax.set_title('Interaction: BldgType × Neighborhood → SalePrice', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Interaction 3: Garage Area + Condition + Price
    if 'GarageArea' in train.columns and 'OverallCond' in train.columns:
        ax = axes[1, 0]
        for condition in sorted(train['OverallCond'].unique()):
            subset = train[train['OverallCond'] == condition]
            ax.scatter(subset['GarageArea'], subset['SalePrice'], alpha=0.6, 
                      label=f'Condition {condition}', s=50, edgecolors='k', linewidth=0.5)
        ax.set_xlabel('Garage Area', fontsize=10)
        ax.set_ylabel('SalePrice', fontsize=10)
        ax.set_title('Interaction: GarageArea × OverallCond → SalePrice', fontsize=11, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Interaction 4: Living Area + House Style + Price
    if 'GrLivArea' in train.columns and 'HouseStyle' in train.columns:
        ax = axes[1, 1]
        for style in train['HouseStyle'].unique()[:6]:  # Top 6 styles
            subset = train[train['HouseStyle'] == style]
            ax.scatter(subset['GrLivArea'], subset['SalePrice'], alpha=0.6, 
                      label=style, s=50, edgecolors='k', linewidth=0.5)
        ax.set_xlabel('Ground Living Area', fontsize=10)
        ax.set_ylabel('SalePrice', fontsize=10)
        ax.set_title('Interaction: GrLivArea × HouseStyle → SalePrice', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('07_interaction_effects.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_interaction_effects.png")
    plt.show()

# ============================================================================
# 9. DISTRIBUTION ANALYSIS
# ============================================================================

def plot_distributions(train):
    """Analyze distribution of target and key features"""
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS: Target and Key Features")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Distribution Analysis', fontsize=16, y=1.00)
    axes = axes.ravel()
    
    # SalePrice distribution
    axes[0].hist(train['SalePrice'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('SalePrice', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title('SalePrice Distribution', fontsize=11, fontweight='bold')
    axes[0].axvline(train['SalePrice'].mean(), color='r', linestyle='--', label=f'Mean: {train["SalePrice"].mean():.0f}')
    axes[0].legend()
    
    # Log of SalePrice
    axes[1].hist(np.log1p(train['SalePrice']), bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Log(SalePrice)', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].set_title('Log-Transformed SalePrice Distribution', fontsize=11, fontweight='bold')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(train['SalePrice'], dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot: SalePrice Normality', fontsize=11, fontweight='bold')
    
    # Top features distributions
    numerical_cols = train.select_dtypes(include=[np.number]).columns
    top_feature = train[numerical_cols].corr()['SalePrice'].abs().sort_values(ascending=False)[1]
    
    axes[3].hist(train[top_feature], bins=40, edgecolor='black', alpha=0.7, color='coral')
    axes[3].set_xlabel(top_feature, fontsize=10)
    axes[3].set_ylabel('Frequency', fontsize=10)
    axes[3].set_title(f'{top_feature} Distribution', fontsize=11, fontweight='bold')
    
    # Skewness visualization
    skew_features = train[numerical_cols].skew().sort_values(ascending=False).head(5).index
    skew_values = train[skew_features].skew()
    axes[4].barh(range(len(skew_values)), skew_values.values, color='skyblue', edgecolor='black')
    axes[4].set_yticks(range(len(skew_values)))
    axes[4].set_yticklabels(skew_values.index, fontsize=9)
    axes[4].set_xlabel('Skewness', fontsize=10)
    axes[4].set_title('Top 5 Skewed Features', fontsize=11, fontweight='bold')
    axes[4].axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Kurtosis visualization
    kurt_features = train[numerical_cols].kurtosis().sort_values(ascending=False).head(5).index
    kurt_values = train[kurt_features].kurtosis()
    axes[5].barh(range(len(kurt_values)), kurt_values.values, color='lightcoral', edgecolor='black')
    axes[5].set_yticks(range(len(kurt_values)))
    axes[5].set_yticklabels(kurt_values.index, fontsize=9)
    axes[5].set_xlabel('Kurtosis', fontsize=10)
    axes[5].set_title('Top 5 Features by Kurtosis', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('08_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 08_distributions.png")
    plt.show()

# ============================================================================
# 10. FEATURE INSIGHTS SUMMARY
# ============================================================================

def generate_feature_insights(train):
    """Generate comprehensive feature insights"""
    print("\n" + "="*80)
    print("FEATURE INSIGHTS SUMMARY")
    print("="*80)
    
    numerical_cols = train.select_dtypes(include=[np.number]).columns
    
    # Top correlated features
    print("\n🔝 TOP 10 FEATURES CORRELATED WITH SALEPRICE:")
    print("-" * 60)
    correlations = train[numerical_cols].corr()['SalePrice'].sort_values(ascending=False)
    for i, (feature, corr) in enumerate(correlations[:11].items(), 1):
        if feature != 'SalePrice':
            print(f"  {i:2d}. {feature:20s} → Correlation: {corr:7.4f}")
    
    # Feature statistics
    print("\n📊 NUMERICAL FEATURES STATISTICS:")
    print("-" * 60)
    stats_summary = train[numerical_cols].describe().round(2)
    print(stats_summary.to_string())
    
    # Outliers detection
    print("\n⚠️  OUTLIERS DETECTION (IQR Method):")
    print("-" * 60)
    for feature in correlations[1:6].index:  # Top 5 features
        Q1 = train[feature].quantile(0.25)
        Q3 = train[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = train[(train[feature] < Q1 - 1.5*IQR) | (train[feature] > Q3 + 1.5*IQR)]
        print(f"  {feature:20s} → Outliers: {len(outliers):4d} ({len(outliers)/len(train)*100:5.2f}%)")
    
    # Categorical insights
    print("\n🏷️  CATEGORICAL FEATURES:")
    print("-" * 60)
    categorical_cols = train.select_dtypes(include=['object']).columns
    for cat in categorical_cols[:8]:
        unique_count = train[cat].nunique()
        missing = train[cat].isnull().sum()
        print(f"  {cat:20s} → Unique: {unique_count:3d}, Missing: {missing:3d}")
    
    # Price insights
    print("\n💰 SALEPRICE INSIGHTS:")
    print("-" * 60)
    print(f"  Mean Price:        ${train['SalePrice'].mean():>12,.0f}")
    print(f"  Median Price:      ${train['SalePrice'].median():>12,.0f}")
    print(f"  Std Deviation:     ${train['SalePrice'].std():>12,.0f}")
    print(f"  Min Price:         ${train['SalePrice'].min():>12,.0f}")
    print(f"  Max Price:         ${train['SalePrice'].max():>12,.0f}")
    print(f"  Price Range:       ${train['SalePrice'].max() - train['SalePrice'].min():>12,.0f}")
    print(f"  Skewness:          {train['SalePrice'].skew():>16.4f}")
    print(f"  Kurtosis:          {train['SalePrice'].kurtosis():>16.4f}")
    
    print("\n" + "="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete EDA pipeline"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  HOUSE PRICES EDA: ADVANCED VISUALIZATION & FEATURE ANALYSIS".center(78) + "║")
    print("║" + "  Kaggle Dataset: Advanced Regression Techniques".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Load data
    train, test = load_data()
    
    # Data quality
    assess_data_quality(train)
    
    # Visualizations
    plot_scatter_plots(train)
    plot_box_plots(train)
    plot_violin_plots(train)
    plot_correlation_heatmap(train)
    plot_pairplot(train)
    plot_3d_scatter(train)
    plot_interaction_analysis(train)
    plot_distributions(train)
    
    # Feature insights
    generate_feature_insights(train)
    
    print("\n✅ EDA COMPLETE!")
    print("📁 Generated visualization files:")
    print("   - 01_scatter_plots.png")
    print("   - 02_box_plots.png")
    print("   - 03_violin_plots.png")
    print("   - 04_correlation_heatmap.png")
    print("   - 05_pairplot.png")
    print("   - 06_trivariate_3d_analysis.png")
    print("   - 07_interaction_effects.png")
    print("   - 08_distributions.png")
    print("\n")

if __name__ == '__main__':
    main()
