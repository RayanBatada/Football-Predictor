"""
Quick script to inspect what data you currently have
Run this to see what columns are in your injury data
"""

import sys
sys.path.append('src')

from data_loader import download_nfl_data
import pandas as pd

print("="*70)
print("INSPECTING CURRENT DATA")
print("="*70)

# Load your current data
schedules, seasonal, weekly, injuries = download_nfl_data()

print("\n" + "="*70)
print("SCHEDULES DATA")
print("="*70)
print(f"Total games: {len(schedules)}")
print(f"\nColumns ({len(schedules.columns)}):")
for i, col in enumerate(schedules.columns, 1):
    print(f"  {i}. {col}")

schedules['gameday'] = pd.to_datetime(schedules['gameday'])
print(f"\nDate range: {schedules['gameday'].min()} to {schedules['gameday'].max()}")
print(f"Days since last game: {(pd.Timestamp.now() - schedules['gameday'].max()).days}")

print("\n" + "="*70)
print("INJURIES DATA")
print("="*70)
print(f"Total records: {len(injuries)}")
print(f"\nColumns ({len(injuries.columns)}):")
for i, col in enumerate(injuries.columns, 1):
    dtype = injuries[col].dtype
    non_null = injuries[col].notna().sum()
    print(f"  {i}. {col:<30} (type: {dtype}, {non_null}/{len(injuries)} non-null)")

# Check for any date-like columns
print("\n" + "="*70)
print("DATE-RELATED COLUMNS IN INJURIES")
print("="*70)

date_keywords = ['date', 'day', 'week', 'time', 'report', 'updated']
date_cols = [col for col in injuries.columns if any(kw in col.lower() for kw in date_keywords)]

if date_cols:
    print(f"Found {len(date_cols)} date-related columns:")
    for col in date_cols:
        print(f"\n  Column: {col}")
        print(f"  Sample values:")
        print(injuries[col].head(10).to_list())
else:
    print("❌ NO DATE COLUMNS FOUND!")
    print("This is why your injury feature doesn't work.")

print("\n" + "="*70)
print("SAMPLE INJURIES")
print("="*70)
print(injuries.head(10))

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if date_cols:
    print(f"✓ You have date columns: {date_cols}")
    print(f"  Update features.py to use: {date_cols[0]}")
else:
    print("✗ No date columns in injury data")
    print("  SOLUTION: Update data_loader.py to use nfl_data_py")
    print("  This will get fresh data with proper date columns")