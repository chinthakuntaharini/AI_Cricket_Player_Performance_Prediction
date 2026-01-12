"""
Data Cleaning Script for IPL Dataset
This script handles data cleaning operations for matches and deliveries datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_data(matches_path='matches.csv', deliveries_path='deliveries.csv'):
    """
    Load the IPL datasets.
    
    Parameters:
    -----------
    matches_path : str
        Path to matches CSV file
    deliveries_path : str
        Path to deliveries CSV file
    
    Returns:
    --------
    tuple
        (matches_df, deliveries_df)
    """
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries


def clean_matches_data(matches):
    """
    Clean the matches dataset.
    
    Parameters:
    -----------
    matches : pd.DataFrame
        Raw matches dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned matches dataframe
    """
    matches_clean = matches.copy()
    
    # Convert date column to datetime
    matches_clean['date'] = pd.to_datetime(matches_clean['date'], errors='coerce')
    
    # Handle missing city values - fill with venue city if possible or 'Unknown'
    matches_clean['city'] = matches_clean['city'].fillna('Unknown')
    
    # Handle missing winner - these are likely no-result or abandoned matches
    # Keep as is for now, but mark them
    matches_clean['winner'] = matches_clean['winner'].fillna('No Result')
    
    # Handle missing player_of_match
    matches_clean['player_of_match'] = matches_clean['player_of_match'].fillna('Not Awarded')
    
    # Handle missing umpires - fill with 'Unknown' for umpire1 and umpire2
    matches_clean['umpire1'] = matches_clean['umpire1'].fillna('Unknown')
    matches_clean['umpire2'] = matches_clean['umpire2'].fillna('Unknown')
    
    # umpire3 is mostly empty (expected - only used in some matches)
    matches_clean['umpire3'] = matches_clean['umpire3'].fillna('Not Applicable')
    
    # Normalize team names (handle any inconsistencies)
    # Standardize common team name variations
    team_name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Delhi Capitals': 'Delhi Capitals',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Rising Pune Supergiants': 'Rising Pune Supergiants',
        'Kings XI Punjab': 'Punjab Kings',
        'Punjab Kings': 'Punjab Kings'
    }
    
    # Apply team name normalization if needed
    for old_name, new_name in team_name_mapping.items():
        matches_clean['team1'] = matches_clean['team1'].replace(old_name, new_name)
        matches_clean['team2'] = matches_clean['team2'].replace(old_name, new_name)
        matches_clean['toss_winner'] = matches_clean['toss_winner'].replace(old_name, new_name)
        matches_clean['winner'] = matches_clean['winner'].replace(old_name, new_name)
    
    # Ensure win_by_runs and win_by_wickets are integers
    matches_clean['win_by_runs'] = matches_clean['win_by_runs'].astype(int)
    matches_clean['win_by_wickets'] = matches_clean['win_by_wickets'].astype(int)
    
    # Create derived columns
    matches_clean['year'] = matches_clean['date'].dt.year
    matches_clean['month'] = matches_clean['date'].dt.month
    
    return matches_clean


def clean_deliveries_data(deliveries):
    """
    Clean the deliveries dataset.
    
    Parameters:
    -----------
    deliveries : pd.DataFrame
        Raw deliveries dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned deliveries dataframe
    """
    deliveries_clean = deliveries.copy()
    
    # Handle missing player_dismissed - these are balls without dismissals (normal)
    deliveries_clean['player_dismissed'] = deliveries_clean['player_dismissed'].fillna('Not Out')
    
    # Handle missing dismissal_kind - fill with 'not_dismissed' for non-dismissal balls
    deliveries_clean['dismissal_kind'] = deliveries_clean['dismissal_kind'].fillna('not_dismissed')
    
    # Handle missing fielder - fill with 'N/A' for non-fielding dismissals
    deliveries_clean['fielder'] = deliveries_clean['fielder'].fillna('N/A')
    
    # Ensure numeric columns are proper types
    numeric_cols = ['over', 'ball', 'wide_runs', 'bye_runs', 'legbye_runs', 
                    'noball_runs', 'penalty_runs', 'batsman_runs', 
                    'extra_runs', 'total_runs', 'is_super_over']
    for col in numeric_cols:
        deliveries_clean[col] = pd.to_numeric(deliveries_clean[col], errors='coerce').fillna(0).astype(int)
    
    # Normalize team names in deliveries (same as matches)
    team_name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Delhi Capitals': 'Delhi Capitals',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Rising Pune Supergiants': 'Rising Pune Supergiants',
        'Kings XI Punjab': 'Punjab Kings',
        'Punjab Kings': 'Punjab Kings'
    }
    
    for old_name, new_name in team_name_mapping.items():
        deliveries_clean['batting_team'] = deliveries_clean['batting_team'].replace(old_name, new_name)
        deliveries_clean['bowling_team'] = deliveries_clean['bowling_team'].replace(old_name, new_name)
    
    # Create derived columns
    deliveries_clean['is_wicket'] = (deliveries_clean['player_dismissed'] != 'Not Out').astype(int)
    deliveries_clean['is_boundary'] = (deliveries_clean['batsman_runs'] >= 4).astype(int)
    deliveries_clean['is_six'] = (deliveries_clean['batsman_runs'] == 6).astype(int)
    deliveries_clean['is_four'] = ((deliveries_clean['batsman_runs'] == 4) & 
                                    (deliveries_clean['batsman_runs'] != 6)).astype(int)
    
    return deliveries_clean


def merge_datasets(matches, deliveries):
    """
    Merge matches and deliveries datasets.
    
    Parameters:
    -----------
    matches : pd.DataFrame
        Cleaned matches dataframe
    deliveries : pd.DataFrame
        Cleaned deliveries dataframe
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    # Merge on match_id (matches.id = deliveries.match_id)
    merged = deliveries.merge(
        matches,
        left_on='match_id',
        right_on='id',
        how='left',
        suffixes=('', '_match')
    )
    
    return merged


def save_cleaned_data(matches_clean, deliveries_clean, merged=None, 
                     output_dir='.'):
    """
    Save cleaned datasets to CSV files.
    
    Parameters:
    -----------
    matches_clean : pd.DataFrame
        Cleaned matches dataframe
    deliveries_clean : pd.DataFrame
        Cleaned deliveries dataframe
    merged : pd.DataFrame, optional
        Merged dataframe
    output_dir : str
        Output directory path
    """
    matches_clean.to_csv(f'{output_dir}/matches_cleaned.csv', index=False)
    deliveries_clean.to_csv(f'{output_dir}/deliveries_cleaned.csv', index=False)
    if merged is not None:
        merged.to_csv(f'{output_dir}/merged_data.csv', index=False)
    print("Cleaned data saved successfully!")

def main():
    """
    Main function to execute data cleaning pipeline.
    """
    print("Loading data...")
    matches, deliveries = load_data()
    
    print("Cleaning matches data...")
    matches_clean = clean_matches_data(matches)
    
    print("Cleaning deliveries data...")
    deliveries_clean = clean_deliveries_data(deliveries)
    
    print("Merging datasets...")
    merged = merge_datasets(matches_clean, deliveries_clean)
    
    print("Saving cleaned data...")
    save_cleaned_data(
        matches_clean,
        deliveries_clean,
        merged,
        output_dir=r"C:\Users\Harini\Downloads\AI_CPP"
    )
    
    print("\nData cleaning completed!")
    print(f"Matches: {matches_clean.shape}")
    print(f"Deliveries: {deliveries_clean.shape}")
    print(f"Merged: {merged.shape}")
    
    return matches_clean, deliveries_clean, merged

if __name__ == "__main__":
    main()

