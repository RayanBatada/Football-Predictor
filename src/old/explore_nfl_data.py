"""
NFL_DATA_PY API PLAYGROUND & REFERENCE
======================================

This file shows you EVERYTHING available in the nfl_data_py library.
Run each section to explore what data you can access.

USAGE:
    python explore_nfl_api.py

Or import and run specific sections:
    from explore_nfl_api import explore_schedules
    explore_schedules()
"""

import nfl_data_py as nfl
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set display options for better viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)


def print_section(title):
    """Pretty print section headers"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def explore_schedules():
    """
    GAME SCHEDULES & SCORES
    
    Contains: All NFL games with scores, dates, locations
    Available: 1999-present
    Best for: Win/loss records, game outcomes, scores
    """
    print_section("1️⃣  GAME SCHEDULES & SCORES")
    
    # Download just 2023 season for example
    print("📥 Downloading 2023 season schedules...")
    schedules = nfl.import_schedules([2023])
    
    print(f"✅ Got {len(schedules)} games\n")
    
    # Show all available columns
    print("📋 AVAILABLE COLUMNS:")
    print("-" * 80)
    for i, col in enumerate(schedules.columns, 1):
        print(f"  {i:2d}. {col}")
    print()
    
    # Show key columns with sample data
    print("📊 KEY FIELDS (first 3 games):")
    print("-" * 80)
    key_cols = [
        'game_id',           # Unique game identifier
        'season',            # Year
        'game_type',         # REG, WC, DIV, CON, SB
        'week',              # Week number
        'gameday',           # Date
        'weekday',           # Day of week
        'gametime',          # Kickoff time
        'home_team',         # Home team abbreviation
        'away_team',         # Away team abbreviation
        'home_score',        # Home team score
        'away_score',        # Away team score
        'location',          # Stadium
        'roof',              # Dome, Outdoors, Open, Closed
        'surface',           # Grass, FieldTurf, etc.
        'temp',              # Temperature (if outdoor)
        'wind',              # Wind speed
        'home_rest',         # Days since home team's last game
        'away_rest',         # Days since away team's last game
        'home_moneyline',    # Betting odds
        'spread_line',       # Point spread
        'total_line',        # Over/under
        'overtime',          # 1 if game went to OT
    ]
    
    available_key_cols = [col for col in key_cols if col in schedules.columns]
    print(schedules[available_key_cols].head(3).to_string())
    
    # Show derived info
    print("\n💡 DERIVED FIELDS YOU CAN CREATE:")
    print("-" * 80)
    schedules['home_win'] = (schedules['home_score'] > schedules['away_score']).astype(int)
    schedules['point_diff'] = schedules['home_score'] - schedules['away_score']
    schedules['total_points'] = schedules['home_score'] + schedules['away_score']
    schedules['close_game'] = (abs(schedules['point_diff']) <= 7).astype(int)
    
    print(schedules[['home_team', 'away_team', 'home_win', 'point_diff', 
                     'total_points', 'close_game']].head(3).to_string())
    
    # Show game types
    print("\n📈 GAME TYPE BREAKDOWN:")
    print("-" * 80)
    print(schedules['game_type'].value_counts().to_string())
    print("\nGame types: REG=Regular, WC=Wild Card, DIV=Divisional, CON=Conference, SB=Super Bowl")
    
    return schedules


def explore_weekly_data():
    """
    WEEKLY TEAM STATISTICS
    
    Contains: Team performance stats by week (offense, defense, special teams)
    Available: 1999-present
    Best for: Team strength, trends, season-long performance
    """
    print_section("2️⃣  WEEKLY TEAM STATISTICS")
    
    print("📥 Downloading 2023 weekly data...")
    weekly = nfl.import_weekly_data([2023], downcast=False)
    
    print(f"✅ Got {len(weekly)} team-week records\n")
    
    # FIRST: Check what columns we actually have
    print("📋 ACTUAL COLUMNS IN DATA:")
    print("-" * 80)
    for i, col in enumerate(weekly.columns[:20], 1):  # Show first 20
        print(f"  {i:2d}. {col}")
    print(f"  ... (total: {len(weekly.columns)} columns)\n")
    
    # Find the team column (it might be named differently)
    team_col = None
    for possible_name in ['team', 'recent_team', 'team_abbr', 'posteam']:
        if possible_name in weekly.columns:
            team_col = possible_name
            break
    
    if team_col is None:
        print("⚠️  Warning: Could not find team column!")
        print("Available columns:", list(weekly.columns[:10]))
        return weekly
    
    # Show structure
    print("📋 DATA STRUCTURE:")
    print(f"  → {weekly['season'].nunique()} season" if 'season' in weekly.columns else "  → season: N/A")
    print(f"  → {weekly['week'].nunique()} weeks" if 'week' in weekly.columns else "  → weeks: N/A")
    print(f"  → {weekly[team_col].nunique()} teams")
    print(f"  → {len(weekly)} total records (team × week)\n")
    
    # Show all columns
    print("📋 AVAILABLE COLUMNS (Total: {})".format(len(weekly.columns)))
    print("-" * 80)
    
    # Categorize columns
    basic_cols = ['season', 'week', team_col, 'opponent', 'game_type']
    offense_cols = [col for col in weekly.columns if any(x in col.lower() for x in ['pass', 'rush', 'offense', 'yards'])]
    defense_cols = [col for col in weekly.columns if 'defense' in col.lower() or col.startswith('opp_')]
    special_cols = [col for col in weekly.columns if any(x in col.lower() for x in ['punt', 'kick', 'return', 'fg'])]
    
    print("🏈 BASIC INFO:")
    for col in basic_cols:
        if col in weekly.columns:
            print(f"  • {col}")
    
    print("\n⚡ OFFENSIVE STATS (sample):")
    for col in offense_cols[:15]:
        print(f"  • {col}")
    if len(offense_cols) > 15:
        print(f"  ... and {len(offense_cols) - 15} more")
    
    print("\n🛡️  DEFENSIVE STATS (sample):")
    for col in defense_cols[:10]:
        print(f"  • {col}")
    if len(defense_cols) > 10:
        print(f"  ... and {len(defense_cols) - 10} more")
    
    print("\n🦶 SPECIAL TEAMS (sample):")
    for col in special_cols[:10]:
        print(f"  • {col}")
    
    # Show sample team performance over season
    print(f"\n📊 EXAMPLE: Kansas City Chiefs' 2023 Season Stats")
    print("-" * 80)
    kc_stats = weekly[weekly[team_col] == 'KC'].sort_values('week') if 'week' in weekly.columns else weekly[weekly[team_col] == 'KC']
    
    # Auto-detect interesting columns
    display_cols = ['week', 'opponent']
    for col in weekly.columns:
        if any(x in col.lower() for x in ['passing_yards', 'rushing_yards', 'points', 'turnovers']):
            display_cols.append(col)
            if len(display_cols) >= 8:  # Limit display
                break
    
    available_display = [col for col in display_cols if col in kc_stats.columns]
    
    if available_display and len(kc_stats) > 0:
        print(kc_stats[available_display].head(10).to_string(index=False))
    
    # Show aggregations you can do
    print("\n💡 EXAMPLE AGGREGATIONS:")
    print("-" * 80)
    
    # Find a scoring column
    scoring_col = None
    for possible in ['points_scored', 'points', 'pts']:
        if possible in weekly.columns:
            scoring_col = possible
            break
    
    if scoring_col:
        team_avg = weekly.groupby(team_col)[scoring_col].mean().sort_values(ascending=False)
        print(f"\n🏆 Top 5 Teams ({scoring_col} average):")
        print(team_avg.head().to_string())
    
    return weekly


def explore_player_stats():
    """
    PLAYER STATISTICS (WEEKLY)
    
    Contains: Individual player performance by week
    Available: 1999-present  
    Best for: Player trends, matchup analysis, fantasy football
    """
    print_section("3️⃣  PLAYER STATISTICS")
    
    print("📥 Downloading 2023 player stats (this may take a moment)...")
    # Just get a few weeks to keep it manageable
    players = nfl.import_seasonal_data([2023])
    
    print(f"✅ Got {len(players)} player records\n")
    
    # FIRST: Show what columns we actually have
    print(f"📋 FIRST 30 COLUMNS (Total: {len(players.columns)}):")
    print("-" * 80)
    for i, col in enumerate(players.columns[:30], 1):
        print(f"  {i:2d}. {col}")
    print(f"  ... ({len(players.columns)} total columns)\n")
    
    # Auto-detect key columns
    team_col = None
    for possible in ['team', 'recent_team', 'team_abbr']:
        if possible in players.columns:
            team_col = possible
            break
    
    name_col = None
    for possible in ['player_display_name', 'player_name', 'name']:
        if possible in players.columns:
            name_col = possible
            break
    
    # Categorize available columns
    categories = {
        'Basic Info': ['player_id', name_col, 'position', 'position_group', 
                      team_col, 'games', 'games_started'],
        'Passing': ['completions', 'attempts', 'passing_yards', 'passing_tds', 
                   'interceptions', 'sacks', 'sack_yards', 'passing_air_yards', 'passing_epa'],
        'Rushing': ['carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 
                   'rushing_first_downs', 'rushing_epa'],
        'Receiving': ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 
                     'receiving_fumbles', 'receiving_air_yards', 'receiving_epa'],
        'Fantasy': ['fantasy_points', 'fantasy_points_ppr'],
    }
    
    print("📋 AVAILABLE STAT CATEGORIES:")
    print("-" * 80)
    
    for category, cols in categories.items():
        available = [col for col in cols if col and col in players.columns]
        if available:
            print(f"\n{category}:")
            for col in available:
                print(f"  • {col}")
    
    # Show top players by position
    print("\n📊 TOP PERFORMERS 2023:")
    print("-" * 80)
    
    # Helper function to safely display top players
    def show_top_players(df, position, stat_col, label, n=5):
        if 'position' not in df.columns or stat_col not in df.columns:
            print(f"\n⚠️  {label}: Data not available")
            return
        
        pos_players = df[df['position'] == position].nlargest(n, stat_col)
        
        if len(pos_players) == 0:
            print(f"\n⚠️  {label}: No {position} players found")
            return
        
        # Build display columns from available columns
        display_cols = []
        if name_col and name_col in pos_players.columns:
            display_cols.append(name_col)
        if team_col and team_col in pos_players.columns:
            display_cols.append(team_col)
        
        # Add stat-specific columns
        stat_options = {
            'passing_yards': ['passing_yards', 'passing_tds', 'interceptions'],
            'rushing_yards': ['carries', 'rushing_yards', 'rushing_tds'],
            'receiving_yards': ['receptions', 'receiving_yards', 'receiving_tds']
        }
        
        if stat_col in stat_options:
            for col in stat_options[stat_col]:
                if col in pos_players.columns:
                    display_cols.append(col)
        else:
            display_cols.append(stat_col)
        
        # Filter to only available columns
        display_cols = [col for col in display_cols if col in pos_players.columns]
        
        if display_cols:
            print(f"\n{label}:")
            print(pos_players[display_cols].to_string(index=False))
        else:
            print(f"\n⚠️  {label}: No displayable columns found")
    
    # QBs by passing yards
    show_top_players(players, 'QB', 'passing_yards', '🎯 Top 5 QBs (Passing Yards)')
    
    # RBs by rushing yards
    show_top_players(players, 'RB', 'rushing_yards', '🏃 Top 5 RBs (Rushing Yards)')
    
    # WRs by receiving yards
    show_top_players(players, 'WR', 'receiving_yards', '🙌 Top 5 WRs (Receiving Yards)')
    
    return players


def explore_rosters():
    """
    TEAM ROSTERS
    
    Contains: Player roster information (names, positions, status)
    Available: Recent years
    Best for: Getting player info, position groups, depth charts
    """
    print_section("4️⃣  TEAM ROSTERS")
    
    print("📥 Downloading 2023 rosters...")
    rosters = nfl.import_rosters([2023])
    
    print(f"✅ Got {len(rosters)} players\n")
    
    # Show columns
    print("📋 AVAILABLE COLUMNS:")
    print("-" * 80)
    for col in rosters.columns:
        print(f"  • {col}")
    
    # Show sample roster
    print("\n📊 EXAMPLE: Kansas City Chiefs Roster (first 10 players)")
    print("-" * 80)
    kc_roster = rosters[rosters['team'] == 'KC'].head(10)
    
    display_cols = ['player_name', 'position', 'jersey_number', 'status', 
                   'height', 'weight', 'college', 'years_exp']
    available = [col for col in display_cols if col in kc_roster.columns]
    
    if available:
        print(kc_roster[available].to_string(index=False))
    
    # Position breakdown
    print("\n📈 POSITION DISTRIBUTION:")
    print("-" * 80)
    print(rosters['position'].value_counts().head(15).to_string())
    
    return rosters


def explore_injuries():
    """
    INJURY REPORTS
    
    Contains: Weekly injury reports (practice status, game status)
    Available: 2022-present (limited)
    Best for: Factoring in player availability
    
    NOTE: Coverage is limited - not all teams/weeks available
    """
    print_section("5️⃣  INJURY REPORTS")
    
    try:
        print("📥 Downloading injury reports...")
        injuries = nfl.import_injuries([2023])
        
        print(f"✅ Got {len(injuries)} injury records\n")
        
        # Show columns
        print("📋 AVAILABLE COLUMNS:")
        print("-" * 80)
        for col in injuries.columns:
            print(f"  • {col}")
        
        # Show sample injuries
        print("\n📊 SAMPLE INJURY REPORTS:")
        print("-" * 80)
        sample_cols = ['season', 'week', 'team', 'full_name', 'position', 
                      'report_status', 'practice_status']
        available = [col for col in sample_cols if col in injuries.columns]
        
        if available:
            print(injuries[available].head(10).to_string(index=False))
        
        # Show status breakdown
        print("\n📈 INJURY STATUS BREAKDOWN:")
        print("-" * 80)
        if 'report_status' in injuries.columns:
            print(injuries['report_status'].value_counts().to_string())
            print("\nStatuses: Out, Questionable, Doubtful, etc.")
        
        return injuries
        
    except Exception as e:
        print(f"⚠️  Injury data not available: {str(e)}")
        print("   Injury reports have limited coverage (2022+)")
        return pd.DataFrame()


def explore_team_descriptions():
    """
    TEAM INFORMATION
    
    Contains: Team metadata (names, abbreviations, colors, logos)
    Available: Current teams
    Best for: Team lookups, displaying team info
    """
    print_section("6️⃣  TEAM INFORMATION")
    
    print("📥 Downloading team descriptions...")
    teams = nfl.import_team_desc()
    
    print(f"✅ Got {len(teams)} teams\n")
    
    # Show columns
    print("📋 AVAILABLE COLUMNS:")
    print("-" * 80)
    for col in teams.columns:
        print(f"  • {col}")
    
    # Show all teams
    print("\n📊 ALL NFL TEAMS:")
    print("-" * 80)
    display_cols = ['team_abbr', 'team_name', 'team_conf', 'team_division', 'team_color']
    available = [col for col in display_cols if col in teams.columns]
    
    if available:
        print(teams[available].to_string(index=False))
    
    # Show by division
    if 'team_division' in teams.columns and 'team_abbr' in teams.columns:
        print("\n📈 TEAMS BY DIVISION:")
        print("-" * 80)
        for division in sorted(teams['team_division'].unique()):
            div_teams = teams[teams['team_division'] == division]['team_abbr'].tolist()
            print(f"  {division}: {', '.join(div_teams)}")
    
    return teams


def explore_draft_picks():
    """
    DRAFT PICKS
    
    Contains: NFL Draft history
    Available: 2000-present
    Best for: Player backgrounds, team building analysis
    """
    print_section("7️⃣  DRAFT PICKS")
    
    try:
        print("📥 Downloading 2023 draft picks...")
        draft = nfl.import_draft_picks([2023])
        
        print(f"✅ Got {len(draft)} draft picks\n")
        
        # Show columns
        print("📋 AVAILABLE COLUMNS:")
        print("-" * 80)
        for col in draft.columns:
            print(f"  • {col}")
        
        # Show first round
        print("\n📊 2023 FIRST ROUND PICKS:")
        print("-" * 80)
        first_round = draft[draft['round'] == 1].head(10)
        
        display_cols = ['pick', 'team', 'position', 'player_name', 'college']
        available = [col for col in display_cols if col in first_round.columns]
        
        if available:
            print(first_round[available].to_string(index=False))
        
        # Position breakdown
        print("\n📈 POSITIONS DRAFTED:")
        print("-" * 80)
        if 'position' in draft.columns:
            print(draft['position'].value_counts().head(10).to_string())
        
        return draft
        
    except Exception as e:
        print(f"⚠️  Draft data not available: {str(e)}")
        return pd.DataFrame()


def explore_pbp_data():
    """
    PLAY-BY-PLAY DATA
    
    Contains: Every single play in NFL games (DETAILED!)
    Available: 1999-present
    Best for: Advanced analytics, situational analysis, deep dives
    
    WARNING: This is HUGE data (100k+ plays per season)
    """
    print_section("8️⃣  PLAY-BY-PLAY DATA (Preview)")
    
    print("⚠️  WARNING: Play-by-play data is VERY large!")
    print("   Full season = 40k+ plays, many columns")
    print("   We'll just show what's available...\n")
    
    response = input("Download sample play-by-play data? (y/n): ")
    
    if response.lower() != 'y':
        print("Skipped. To explore later:")
        print("  pbp = nfl.import_pbp_data([2023])")
        return None
    
    print("\n📥 Downloading 2023 play-by-play (this will take a while)...")
    pbp = nfl.import_pbp_data([2023])
    
    print(f"✅ Got {len(pbp)} plays!\n")
    
    # Show structure
    print(f"📋 MASSIVE DATASET:")
    print(f"  → {len(pbp)} total plays")
    print(f"  → {len(pbp.columns)} columns!")
    print(f"  → {pbp.memory_usage(deep=True).sum() / 1024**2:.1f} MB in memory\n")
    
    # Show column categories
    print("📋 COLUMN CATEGORIES (sample):")
    print("-" * 80)
    
    categories = {
        'Game Info': ['game_id', 'home_team', 'away_team', 'season', 'week'],
        'Play Info': ['play_id', 'play_type', 'yards_gained', 'down', 'ydstogo', 
                     'yardline_100', 'quarter_seconds_remaining'],
        'Passing': ['pass_length', 'pass_location', 'air_yards', 'yards_after_catch', 
                   'complete_pass', 'incomplete_pass', 'interception'],
        'Rushing': ['rush_attempt', 'rushing_yards', 'rush_touchdown'],
        'Scoring': ['touchdown', 'field_goal_attempt', 'field_goal_result', 'extra_point_result'],
        'Expected Points': ['ep', 'epa', 'wp', 'wpa'],  # Advanced metrics
        'Players': ['passer_player_name', 'rusher_player_name', 'receiver_player_name'],
    }
    
    for category, cols in categories.items():
        print(f"\n{category}:")
        available = [col for col in cols if col in pbp.columns]
        for col in available[:5]:
            print(f"  • {col}")
        if len(available) > 5:
            print(f"  ... and {len(available) - 5} more")
    
    # Show sample plays
    print("\n📊 SAMPLE PLAYS (first 5):")
    print("-" * 80)
    sample_cols = ['game_id', 'play_type', 'down', 'ydstogo', 'yards_gained', 'desc']
    available = [col for col in sample_cols if col in pbp.columns]
    
    if available:
        print(pbp[available].head(5).to_string(index=False))
    
    return pbp


def explore_depth_charts():
    """
    DEPTH CHARTS
    
    Contains: Team depth charts by position
    Available: Recent seasons
    Best for: Starter identification, position rankings
    """
    print_section("9️⃣  DEPTH CHARTS")
    
    try:
        print("📥 Downloading depth charts...")
        depth = nfl.import_depth_charts([2023])
        
        print(f"✅ Got {len(depth)} depth chart entries\n")
        
        # Show columns
        print("📋 AVAILABLE COLUMNS:")
        print("-" * 80)
        for col in depth.columns:
            print(f"  • {col}")
        
        # Show sample depth chart
        print("\n📊 EXAMPLE: Kansas City Chiefs QB Depth Chart")
        print("-" * 80)
        
        kc_qbs = depth[(depth['team'] == 'KC') & (depth['position'] == 'QB')]
        display_cols = ['week', 'position', 'depth_team', 'player_name', 'jersey_number']
        available = [col for col in display_cols if col in kc_qbs.columns]
        
        if available:
            print(kc_qbs[available].head().to_string(index=False))
        
        return depth
        
    except Exception as e:
        print(f"⚠️  Depth chart data not available: {str(e)}")
        return pd.DataFrame()


def show_api_summary():
    """
    Summary of all available functions in nfl_data_py
    """
    print_section("📚 NFL_DATA_PY API SUMMARY")
    
    functions = [
        ("nfl.import_schedules(years)", "Game schedules & scores", "1999-present", "⭐⭐⭐"),
        ("nfl.import_weekly_data(years)", "Weekly team stats", "1999-present", "⭐⭐⭐"),
        ("nfl.import_seasonal_data(years)", "Season player stats", "1999-present", "⭐⭐⭐"),
        ("nfl.import_rosters(years)", "Team rosters", "Recent years", "⭐⭐"),
        ("nfl.import_team_desc()", "Team information", "Current", "⭐⭐"),
        ("nfl.import_injuries(years)", "Injury reports", "2022-present", "⭐"),
        ("nfl.import_draft_picks(years)", "Draft history", "2000-present", "⭐"),
        ("nfl.import_pbp_data(years)", "Play-by-play (HUGE!)", "1999-present", "⭐⭐⭐"),
        ("nfl.import_depth_charts(years)", "Depth charts", "Recent years", "⭐"),
        ("nfl.import_win_totals(years)", "Season win totals", "Recent years", "⭐"),
        ("nfl.import_officials(years)", "Game officials", "Recent years", "⭐"),
        ("nfl.import_snap_counts(years)", "Player snap counts", "Recent years", "⭐⭐"),
    ]
    
    print(f"{'Function':<40} {'Description':<25} {'Years':<15} {'Usefulness'}")
    print("-" * 100)
    for func, desc, years, stars in functions:
        print(f"{func:<40} {desc:<25} {years:<15} {stars}")
    
    print("\n⭐⭐⭐ = Essential for game prediction")
    print("⭐⭐   = Very useful")
    print("⭐     = Nice to have\n")
    
    print("💡 MOST USEFUL FOR GAME PREDICTION:")
    print("  1. import_schedules() - Historical game results")
    print("  2. import_weekly_data() - Team performance metrics")
    print("  3. import_seasonal_data() - Player performance")
    print("  4. import_injuries() - Player availability\n")


def interactive_explorer():
    """
    Interactive mode - let user choose what to explore
    """
    print("\n" + "=" * 80)
    print("  🏈 NFL DATA API INTERACTIVE EXPLORER")
    print("=" * 80)
    
    options = {
        '1': ('Game Schedules & Scores', explore_schedules),
        '2': ('Weekly Team Statistics', explore_weekly_data),
        '3': ('Player Statistics', explore_player_stats),
        '4': ('Team Rosters', explore_rosters),
        '5': ('Injury Reports', explore_injuries),
        '6': ('Team Information', explore_team_descriptions),
        '7': ('Draft Picks', explore_draft_picks),
        '8': ('Play-by-Play (Advanced)', explore_pbp_data),
        '9': ('Depth Charts', explore_depth_charts),
        '0': ('API Summary', show_api_summary),
        'q': ('Quit', None),
    }
    
    while True:
        print("\n📋 SELECT DATA TO EXPLORE:")
        print("-" * 80)
        for key, (name, _) in options.items():
            if key == 'q':
                print(f"  {key}. {name}")
            else:
                print(f"  {key}. {name}")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Happy coding!")
            break
        
        if choice in options and options[choice][1] is not None:
            options[choice][1]()
            input("\nPress ENTER to continue...")
        else:
            print("❌ Invalid choice!")


def quick_reference():
    """
    Print quick reference guide
    """
    print_section("⚡ QUICK REFERENCE")
    
    reference = """
# Import library
import nfl_data_py as nfl

# Get game schedules (most common)
schedules = nfl.import_schedules([2023])

# Get team stats by week
weekly = nfl.import_weekly_data([2023])

# Get player stats
players = nfl.import_seasonal_data([2023])

# Get rosters
rosters = nfl.import_rosters([2023])

# Get team info
teams = nfl.import_team_desc()

# Get injuries (limited coverage)
injuries = nfl.import_injuries([2023])

# Get multiple years
schedules = nfl.import_schedules([2021, 2022, 2023])

# Years as range
years = list(range(2018, 2024))
schedules = nfl.import_schedules(years)

# Save to CSV
schedules.to_csv('nfl_games.csv', index=False)

# Filter data
chiefs_games = schedules[(schedules['home_team'] == 'KC') | (schedules['away_team'] == 'KC')]
week1_games = schedules[schedules['week'] == 1]
playoff_games = schedules[schedules['game_type'] != 'REG']

# Aggregate stats
ppg = schedules.groupby('home_team')['home_score'].mean()
win_pct = schedules.groupby('home_team')['home_win'].mean()
"""
    
    print(reference)


def main():
    """
    Main execution - run all explorations
    """
    print("\n" + "=" * 80)
    print("  🏈 NFL_DATA_PY API EXPLORER & REFERENCE")
    print("=" * 80)
    print("\n  This script shows you ALL data available in nfl_data_py")
    print("  and how to use it for your game predictor.\n")
    
    mode = input("Choose mode:\n  1. Auto-explore all (recommended)\n  2. Interactive mode\n  3. Quick reference\n\nChoice: ").strip()
    
    if mode == '1':
        # Run all explorations
        show_api_summary()
        input("\nPress ENTER to start exploring...")
        
        explore_schedules()
        input("\nPress ENTER to continue...")
        
        explore_weekly_data()
        input("\nPress ENTER to continue...")
        
        explore_player_stats()
        input("\nPress ENTER to continue...")
        
        explore_rosters()
        input("\nPress ENTER to continue...")
        
        explore_team_descriptions()
        input("\nPress ENTER to continue...")
        
        explore_injuries()
        input("\nPress ENTER to continue...")
        
        print("\n✅ Exploration complete! Check the output above.")
        print("\n💡 TIP: Import specific functions to explore more:")
        print("   from explore_nfl_api import explore_schedules")
        print("   schedules = explore_schedules()")
        
    elif mode == '2':
        interactive_explorer()
        
    elif mode == '3':
        quick_reference()
        show_api_summary()
    
    else:
        print("Invalid choice. Run again and choose 1, 2, or 3.")


if __name__ == "__main__":
    main()