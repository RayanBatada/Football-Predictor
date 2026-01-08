"""
NFL Data Loader - Updated to get current 2024-2025 data using nfl_data_py

This pulls fresh data from the official nfl_data_py library.
"""

import nfl_data_py as nfl
import pandas as pd


def download_nfl_data():
    """
    Download NFL data using nfl_data_py library
    
    Returns:
        schedules_df: Game schedules with scores
        seasonal_stats_df: Season-level player stats (empty for now)
        weekly_stats_df: Week-level player stats (empty for now)
        injuries_df: Injury reports
    """
    
    print("Downloading NFL data using nfl_data_py...")
    print("This may take 30-60 seconds on first run (data is cached after)")
    
    # Define which seasons to download
    # Get last 4-5 seasons for good training data
    # Current date is January 7, 2026
    seasons = [2021, 2022, 2023, 2024, 2025]
    
    print(f"\nDownloading seasons: {seasons}")
    
    # 1. Download schedules (games with scores)
    print("  - Downloading schedules...")
    try:
        schedules_df = nfl.import_schedules(seasons)
        
        # Filter to only completed games (games that have been played)
        schedules_df = schedules_df[schedules_df['home_score'].notna()].copy()
        
        print(f"    ✓ Loaded {len(schedules_df)} completed games")
        print(f"    Date range: {schedules_df['gameday'].min()} to {schedules_df['gameday'].max()}")
    except Exception as e:
        print(f"    ✗ Error loading schedules: {e}")
        schedules_df = pd.DataFrame()
    
    # 2. Download seasonal stats (optional - for future enhancements)
    # NOTE: API has changed, stat_type parameter no longer supported
    print("  - Downloading seasonal player stats...")
    try:
        # FIXED: Removed stat_type parameter
        seasonal_stats_df = nfl.import_seasonal_data(seasons)
        print(f"    ✓ Loaded {len(seasonal_stats_df)} seasonal stat records")
    except Exception as e:
        print(f"    ⚠ Could not load seasonal stats: {e}")
        seasonal_stats_df = pd.DataFrame()
    
    # 3. Download weekly stats (optional - for future enhancements)
    print("  - Downloading weekly player stats...")
    try:
        # FIXED: Removed stat_type parameter
        weekly_stats_df = nfl.import_weekly_data(seasons)
        print(f"    ✓ Loaded {len(weekly_stats_df)} weekly stat records")
    except Exception as e:
        print(f"    ⚠ Could not load weekly stats: {e}")
        weekly_stats_df = pd.DataFrame()
    
    # 4. Download injuries (may have limited availability)
    print("  - Downloading injury data...")
    try:
        injuries_df = nfl.import_injuries(seasons)
        
        # Add date column from report_date if available
        if 'date' not in injuries_df.columns and 'report_date' in injuries_df.columns:
            injuries_df['date'] = injuries_df['report_date']
        
        print(f"    ✓ Loaded {len(injuries_df)} injury records")
        
        # Check what date columns we have
        date_cols = [col for col in injuries_df.columns if 'date' in col.lower()]
        if date_cols:
            print(f"    Date columns found: {date_cols}")
        else:
            print(f"    ⚠ Warning: No date columns found in injury data")
        
    except Exception as e:
        print(f"    ⚠ Could not load injuries: {e}")
        print(f"    Note: Injury data has limited API availability")
        injuries_df = pd.DataFrame()
    
    print("\n" + "="*60)
    print("DATA DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Schedules: {len(schedules_df)} games")
    print(f"Seasonal stats: {len(seasonal_stats_df)} records")
    print(f"Weekly stats: {len(weekly_stats_df)} records")
    print(f"Injuries: {len(injuries_df)} records")
    print("="*60 + "\n")
    
    return schedules_df, seasonal_stats_df, weekly_stats_df, injuries_df


def download_current_season_only(season=2024):
    """
    Download only a specific season (faster for testing)
    
    Args:
        season: Year to download (default: 2024)
    """
    print(f"Downloading {season} season only (quick mode)...")
    
    try:
        schedules_df = nfl.import_schedules([season])
        schedules_df = schedules_df[schedules_df['home_score'].notna()].copy()
        print(f"  ✓ Loaded {len(schedules_df)} games from {season}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        schedules_df = pd.DataFrame()
    
    try:
        injuries_df = nfl.import_injuries([season])
        if 'report_date' in injuries_df.columns:
            injuries_df['date'] = injuries_df['report_date']
        print(f"  ✓ Loaded {len(injuries_df)} injuries from {season}")
    except Exception as e:
        print(f"  ⚠ Could not load injuries: {e}")
        injuries_df = pd.DataFrame()
    
    # Return empty dataframes for unused data
    seasonal_stats_df = pd.DataFrame()
    weekly_stats_df = pd.DataFrame()
    
    return schedules_df, seasonal_stats_df, weekly_stats_df, injuries_df


if __name__ == "__main__":
    # Test the data loader
    print("Testing NFL Data Loader...\n")
    
    schedules, seasonal, weekly, injuries = download_nfl_data()
    
    print("\n" + "="*60)
    print("DATA SAMPLE")
    print("="*60)
    
    if not schedules.empty:
        print("\nSchedules columns:")
        print(schedules.columns.tolist())
        
        print("\nRecent games:")
        print(schedules[['gameday', 'home_team', 'away_team', 'home_score', 'away_score']].tail(10))
    
    if not injuries.empty:
        print("\nInjuries columns:")
        print(injuries.columns.tolist())
        
        print("\nRecent injuries (first 5):")
        print(injuries.head(5))
    else:
        print("\n⚠ No injury data available")
        print("Note: The model will still work without injury data")
        print("The injury_count feature will simply be 0 for all teams")