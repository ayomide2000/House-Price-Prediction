import pandas as pd
import numpy as np

def apply_intensive_engineering(df):
    df = df.copy()
    
    # Interactions: GrLivArea, HouseAge, Qual Density
    df['GrLivArea_Qual'] = df['GrLivArea'] * df.get('OverallQual', 5)
    df['HouseAge'] = df.get('YrSold', 2010) - df.get('YearBuilt', 1950)
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['Qual_Density'] = df['TotalSF'] * df.get('OverallQual', 5)
    df['Age_Qual_Density'] = df['HouseAge'] * df['Qual_Density']
    
    # MS Zoning / Lot Area Interaction
    # Note: In production, you'd usually use pre-calculated means to avoid drift
    df['Cond_LotArea'] = df.get('OverallCond', 5) * df.get('LotArea', 5000)
    
    # Fill remaining NaNs to satisfy Lasso/Linear models
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df
