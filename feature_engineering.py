"""
IPL Cricket Player Performance Prediction - Feature Engineering

This script performs comprehensive feature engineering on the IPL dataset to create 
predictive features for player performance.

Objectives:
1. Aggregate ball-by-ball data to player-match level
2. Engineer features:
   - Rolling averages (form)
   - Venue averages
   - Opponent-specific stats (PvT, PvP)
   - Career stats
3. Create training labels (runs/wickets in next match)
4. Time-series aware train-test split
5. Save feature pipeline and final dataset

Deliverables:
- dataset.csv: Final feature-engineered dataset
- feature_pipeline.pkl: Saved pre-processor
- train_data.csv: Training set
- test_data.csv: Test set
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load merged data and prepare for feature engineering"""
    print("=" * 60)
    print("1. LOADING AND PREPARING DATA")
    print("=" * 60)
    
    # Load merged data
    df = pd.read_csv('merged_data.csv')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Rename batsman to batter for consistency
    df['batter'] = df['batsman']
    
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique batters: {df['batter'].nunique()}")
    print(f"Unique venues: {df['venue'].nunique()}")
    print(f"Unique bowling teams: {df['bowling_team'].nunique()}")
    
    return df

def aggregate_to_player_match(df):
    """Aggregate ball-by-ball data to player-match level"""
    print("\n" + "=" * 60)
    print("2. AGGREGATING TO PLAYER-MATCH LEVEL")
    print("=" * 60)
    
    # Aggregate to player-match level
    player_match = df.groupby(['batter', 'match_id', 'date', 'venue', 'bowling_team']).agg({
        'batsman_runs': 'sum',
        'ball': 'count',
        'is_wicket': 'sum',
        'is_boundary': 'sum',
        'is_six': 'sum',
        'is_four': 'sum'
    }).reset_index()
    
    # Calculate strike rate
    player_match['strike_rate'] = (player_match['batsman_runs'] / player_match['ball'] * 100).round(2)
    
    # Sort by batter and date for time-series operations
    player_match = player_match.sort_values(['batter', 'date'])
    
    print(f"Player-match level data shape: {player_match.shape}")
    print(f"Sample data:")
    print(player_match.head())
    
    return player_match

def engineer_put_features(player_match):
    """Create Player vs Team (PUT) features"""
    print("\n" + "=" * 60)
    print("3.1 ENGINEERING PUT FEATURES")
    print("=" * 60)
    
    # PUT (Player vs Team) average - overall average against bowling team
    player_match['put_avg'] = player_match.groupby(['batter', 'bowling_team'])['batsman_runs'].transform('mean')
    
    # PUT average with expanding mean (cumulative performance)
    put_expanding = player_match.groupby(['batter', 'bowling_team'])['batsman_runs']\
        .expanding().mean().shift(1, fill_value=np.nan)
    # Reset index to match original dataframe
    put_expanding = put_expanding.reset_index(level=[0,1], drop=True)
    player_match['put_avg_expanding'] = put_expanding
    
    print("PUT features created:")
    print(player_match[['batter', 'bowling_team', 'batsman_runs', 'put_avg', 'put_avg_expanding']].head())
    
    return player_match

def engineer_rolling_features(player_match):
    """Create rolling form features (last 5 matches)"""
    print("\n" + "=" * 60)
    print("3.2 ENGINEERING ROLLING FORM FEATURES")
    print("=" * 60)
    
    # Rolling average of last 5 matches
    player_match['rolling_avg_5'] = player_match.groupby('batter')['batsman_runs']\
        .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Rolling strike rate of last 5 matches
    player_match['rolling_sr_5'] = player_match.groupby('batter')['strike_rate']\
        .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    print("Rolling form features created:")
    print(player_match[['batter', 'date', 'batsman_runs', 'rolling_avg_5', 'rolling_sr_5']].head(10))
    
    return player_match

def engineer_venue_features(player_match):
    """Create venue average features"""
    print("\n" + "=" * 60)
    print("3.3 ENGINEERING VENUE FEATURES")
    print("=" * 60)
    
    # Venue average for each batter
    player_match['venue_avg'] = player_match.groupby(['batter', 'venue'])['batsman_runs'].transform('mean')
    
    # Overall venue average
    player_match['venue_overall_avg'] = player_match.groupby('venue')['batsman_runs'].transform('mean')
    
    print("Venue features created:")
    print(player_match[['batter', 'venue', 'batsman_runs', 'venue_avg', 'venue_overall_avg']].head())
    
    return player_match

def engineer_pvp_features(player_match, df):
    """Create Player vs Player (PvP) features"""
    print("\n" + "=" * 60)
    print("3.4 ENGINEERING PVP FEATURES")
    print("=" * 60)
    
    # For PvP, we need to consider the specific bowlers faced
    # First, get ball-by-ball data with bowler information
    pvp_data = df.groupby(['batter', 'bowler', 'match_id'])['batsman_runs'].sum().reset_index()
    pvp_data = pvp_data.sort_values(['batter', 'match_id'])
    
    # Calculate PvP expanding average
    pvp_expanding = pvp_data.groupby(['batter', 'bowler'])['batsman_runs']\
        .expanding().mean().shift(1, fill_value=np.nan)
    pvp_expanding = pvp_expanding.reset_index(level=[0,1], drop=True)
    pvp_data['pvp_avg'] = pvp_expanding
    
    # Merge back to player-match level (take average of all bowlers faced in the match)
    pvp_match_avg = pvp_data.groupby(['batter', 'match_id'])['pvp_avg'].mean().reset_index()
    player_match = player_match.merge(pvp_match_avg, on=['batter', 'match_id'], how='left')
    
    print("PvP features created:")
    print(player_match[['batter', 'match_id', 'batsman_runs', 'pvp_avg']].head())
    
    return player_match

def engineer_career_features(player_match):
    """Create career average features"""
    print("\n" + "=" * 60)
    print("3.5 ENGINEERING CAREER FEATURES")
    print("=" * 60)
    
    # Career average using expanding mean
    career_avg_expanding = player_match.groupby('batter')['batsman_runs']\
        .expanding().mean().shift(1, fill_value=np.nan)
    career_avg_expanding = career_avg_expanding.reset_index(level=0, drop=True)
    player_match['career_avg'] = career_avg_expanding
    
    # Career strike rate using expanding mean
    career_sr_expanding = player_match.groupby('batter')['strike_rate']\
        .expanding().mean().shift(1, fill_value=np.nan)
    career_sr_expanding = career_sr_expanding.reset_index(level=0, drop=True)
    player_match['career_sr'] = career_sr_expanding
    
    # Career matches played
    player_match['career_matches'] = player_match.groupby('batter').cumcount()
    
    print("Career features created:")
    print(player_match[['batter', 'date', 'batsman_runs', 'career_avg', 'career_sr', 'career_matches']].head(10))
    
    return player_match

def create_target_labels(player_match):
    """Create target labels (next match performance)"""
    print("\n" + "=" * 60)
    print("4. CREATING TARGET LABELS")
    print("=" * 60)
    
    # Create target: runs in next match
    player_match = player_match.sort_values(['batter', 'date'])
    player_match['target_next_runs'] = player_match.groupby('batter')['batsman_runs'].shift(-1)
    
    # Create target: strike rate in next match
    player_match['target_next_sr'] = player_match.groupby('batter')['strike_rate'].shift(-1)
    
    # Remove rows where target is NaN (last match for each player)
    player_match_clean = player_match.dropna(subset=['target_next_runs'])
    
    print(f"Data after creating targets: {player_match_clean.shape}")
    print(f"Target statistics:")
    print(f"Next match runs - Mean: {player_match_clean['target_next_runs'].mean():.2f}, Std: {player_match_clean['target_next_runs'].std():.2f}")
    print(f"Next match SR - Mean: {player_match_clean['target_next_sr'].mean():.2f}, Std: {player_match_clean['target_next_sr'].std():.2f}")
    
    return player_match_clean

def feature_selection_and_preprocessing(player_match_clean):
    """Select features and handle preprocessing"""
    print("\n" + "=" * 60)
    print("5. FEATURE SELECTION AND PREPROCESSING")
    print("=" * 60)
    
    # Select features for modeling
    feature_columns = [
        'put_avg', 'put_avg_expanding',
        'rolling_avg_5', 'rolling_sr_5',
        'venue_avg', 'venue_overall_avg',
        'pvp_avg',
        'career_avg', 'career_sr', 'career_matches',
        'ball', 'is_boundary', 'is_six', 'is_four'
    ]
    
    # Handle missing values in features
    for col in feature_columns:
        if col in player_match_clean.columns:
            player_match_clean[col] = player_match_clean[col].fillna(player_match_clean[col].median())
    
    features = player_match_clean[feature_columns]
    labels_runs = player_match_clean['target_next_runs']
    labels_sr = player_match_clean['target_next_sr']
    
    print(f"Features shape: {features.shape}")
    print(f"Features selected: {feature_columns}")
    print(f"Feature statistics:")
    print(features.describe())
    
    return features, labels_runs, labels_sr, feature_columns

def time_series_split(features, labels_runs, labels_sr, player_match_clean):
    """Perform time-series aware train-test split"""
    print("\n" + "=" * 60)
    print("6. TIME-SERIES AWARE TRAIN-TEST SPLIT")
    print("=" * 60)
    
    # Sort by date for proper time-series split
    player_match_clean = player_match_clean.sort_values('date')
    
    # Time-series split (80% train, 20% test)
    split_idx = int(len(player_match_clean) * 0.8)
    
    X_train = features[:split_idx]
    X_test = features[split_idx:]
    y_train_runs = labels_runs[:split_idx]
    y_test_runs = labels_runs[split_idx:]
    y_train_sr = labels_sr[:split_idx]
    y_test_sr = labels_sr[split_idx:]
    
    print(f"Train set size: {len(X_train)} ({len(X_train)/len(features)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(features)*100:.1f}%)")
    print(f"Train date range: {player_match_clean.iloc[:split_idx]['date'].min()} to {player_match_clean.iloc[:split_idx]['date'].max()}")
    print(f"Test date range: {player_match_clean.iloc[split_idx:]['date'].min()} to {player_match_clean.iloc[split_idx:]['date'].max()}")
    
    return X_train, X_test, y_train_runs, y_test_runs, y_train_sr, y_test_sr

def create_feature_pipeline(X_train, X_test, feature_columns):
    """Create and apply feature preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("7. CREATING FEATURE PIPELINE")
    print("=" * 60)
    
    # Create preprocessing pipeline
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Fit pipeline on training data
    X_train_scaled = feature_pipeline.fit_transform(X_train)
    X_test_scaled = feature_pipeline.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    
    print("Feature pipeline created and applied:")
    print(f"Scaled training data shape: {X_train_scaled.shape}")
    print(f"Scaled test data shape: {X_test_scaled.shape}")
    print(f"Sample scaled features:")
    print(X_train_scaled.head())
    
    return feature_pipeline, X_train_scaled, X_test_scaled

def save_artifacts(feature_pipeline, final_dataset, X_train, X_test, y_train_runs, y_test_runs, y_train_sr, y_test_sr, feature_columns):
    """Save all artifacts"""
    print("\n" + "=" * 60)
    print("8. SAVING ARTIFACTS")
    print("=" * 60)
    
    # Save the feature pipeline
    joblib.dump(feature_pipeline, 'feature_pipeline.pkl')
    
    # Save the final dataset with all features and targets
    final_dataset.to_csv('dataset.csv', index=False)
    
    # Save train-test splits
    train_data = pd.concat([X_train, y_train_runs, y_train_sr], axis=1)
    train_data.columns = feature_columns + ['target_runs', 'target_sr']
    train_data.to_csv('train_data.csv', index=False)
    
    test_data = pd.concat([X_test, y_test_runs, y_test_sr], axis=1)
    test_data.columns = feature_columns + ['target_runs', 'target_sr']
    test_data.to_csv('test_data.csv', index=False)
    
    print("Artifacts saved successfully:")
    print("- feature_pipeline.pkl: Preprocessing pipeline")
    print("- dataset.csv: Complete feature-engineered dataset")
    print("- train_data.csv: Training set")
    print("- test_data.csv: Test set")

def print_summary(final_dataset, labels_runs, labels_sr, X_train, X_test, features, feature_columns):
    """Print final summary"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   - Final dataset shape: {final_dataset.shape}")
    print(f"   - Date range: {final_dataset['date'].min()} to {final_dataset['date'].max()}")
    print(f"   - Unique players: {final_dataset['batter'].nunique()}")
    print(f"   - Unique venues: {final_dataset['venue'].nunique()}")
    
    print(f"\n2. FEATURES ENGINEERED:")
    print(f"   - PUT (Player vs Team) averages: 2 features")
    print(f"   - Rolling form (last 5 matches): 2 features")
    print(f"   - Venue averages: 2 features")
    print(f"   - PvP (Player vs Player): 1 feature")
    print(f"   - Career statistics: 3 features")
    print(f"   - Match-specific: 4 features")
    print(f"   - Total features: {len(feature_columns)}")
    
    print(f"\n3. TARGET VARIABLES:")
    print(f"   - Next match runs: Mean={labels_runs.mean():.2f}, Std={labels_runs.std():.2f}")
    print(f"   - Next match strike rate: Mean={labels_sr.mean():.2f}, Std={labels_sr.std():.2f}")
    
    print(f"\n4. TRAIN-TEST SPLIT:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Split ratio: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")
    
    print(f"\n5. FEATURE DISTRIBUTIONS:")
    for col in feature_columns[:5]:  # Show first 5 features
        print(f"   - {col}: Mean={features[col].mean():.2f}, Std={features[col].std():.2f}")
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

def main():
    """Main function to execute the complete feature engineering pipeline"""
    print("IPL Cricket Player Performance Prediction - Feature Engineering")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    
    # Step 2: Aggregate to player-match level
    player_match = aggregate_to_player_match(df)
    
    # Step 3: Engineer features
    player_match = engineer_put_features(player_match)
    player_match = engineer_rolling_features(player_match)
    player_match = engineer_venue_features(player_match)
    player_match = engineer_pvp_features(player_match, df)
    player_match = engineer_career_features(player_match)
    
    # Step 4: Create target labels
    player_match_clean = create_target_labels(player_match)
    
    # Step 5: Feature selection and preprocessing
    features, labels_runs, labels_sr, feature_columns = feature_selection_and_preprocessing(player_match_clean)
    
    # Step 6: Time-series aware train-test split
    X_train, X_test, y_train_runs, y_test_runs, y_train_sr, y_test_sr = time_series_split(
        features, labels_runs, labels_sr, player_match_clean
    )
    
    # Step 7: Create feature pipeline
    feature_pipeline, X_train_scaled, X_test_scaled = create_feature_pipeline(X_train, X_test, feature_columns)
    
    # Step 8: Save artifacts
    save_artifacts(
        feature_pipeline, player_match_clean, X_train, X_test, 
        y_train_runs, y_test_runs, y_train_sr, y_test_sr, feature_columns
    )
    
    # Step 9: Print summary
    print_summary(player_match_clean, labels_runs, labels_sr, X_train, X_test, features, feature_columns)

if __name__ == "__main__":
    main()
