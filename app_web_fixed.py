import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="DK NBA Best Ball Draft Assistant",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .value-alert {
        background-color: #1f4d1f;
        padding: 5px;
        border-radius: 3px;
    }
    .stack-badge {
        background-color: #4a1a4a;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.8em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'drafted_players' not in st.session_state:
        st.session_state.drafted_players = set()
    if 'my_roster' not in st.session_state:
        st.session_state.my_roster = []
    if 'finals_emphasis' not in st.session_state:
        st.session_state.finals_emphasis = 1.0
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None

init_session_state()

def process_data(df):
    """Calculate derived columns for the dataframe"""
    if df is None:
        return None
    
    # Ensure ID column exists
    if 'ID' not in df.columns:
        df['ID'] = range(len(df))
    
    # Fill missing ADP values with high number (drafted last)
    if 'ADP' in df.columns:
        max_adp = df['ADP'].max()
        if pd.isna(max_adp):
            max_adp = 999
        df['ADP'] = df['ADP'].fillna(max_adp + 100)
    
    # Ensure required numeric columns exist with defaults
    required_cols = {
        'ShutdownRisk': 0.5,
        'R2Mult': 1.0,
        'R3Mult': 1.0,
        'FinalsMult': 1.0,
        'TeamRank': 15,
        'R2Games': 3,
        'R3Games': 3,
        'FinalsGames': 3,
        'FinalAdjGPP': 0,
        'ADP': 999
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            # Fill NA values
            df[col] = df[col].fillna(default_val)
    
    # Remove any rows where critical columns are still NA
    critical_cols = ['Name', 'Position', 'Team', 'FinalAdjGPP']
    df = df.dropna(subset=critical_cols)
    
    # Filter to only players with reasonable projections (remove outliers/bad data)
    df = df[df['FinalAdjGPP'] > 0]
    
    # Calculate rankings
    df['FinalAdjGPP_Rank'] = df['FinalAdjGPP'].rank(ascending=False, method='min').astype(int)
    df['ADP_Rank'] = df['ADP'].rank(ascending=True, method='min').astype(int)
    df['ValueScore'] = df['ADP_Rank'] - df['FinalAdjGPP_Rank']
    
    # Calculate z-scores (handle potential all-same-value cases)
    try:
        df['zFinal'] = stats.zscore(df['FinalAdjGPP'])
        df['zADP'] = stats.zscore(-df['ADP'])
        df['ValueZ'] = df['zFinal'] - df['zADP']
    except:
        df['zFinal'] = 0
        df['zADP'] = 0
        df['ValueZ'] = 0
    
    # Value Alert flag
    df['ValueAlert'] = ((df['ValueScore'] >= 12) | (df['ValueZ'] >= 0.75))
    
    return df

def calculate_live_score(df, emphasis):
    """Calculate LiveScore based on finals emphasis"""
    df['LiveScore'] = df['FinalAdjGPP'] * emphasis
    return df

def calculate_stack_score(player_row, my_roster_df):
    """Calculate stack score for a potential pick"""
    score = 0
    
    if my_roster_df.empty:
        return score
    
    player_team = player_row['Team']
    team_count = len(my_roster_df[my_roster_df['Team'] == player_team])
    
    # Stack bonuses
    if team_count == 1:
        score += 6
    elif team_count == 2:
        score += 10
    
    # Team quality bonuses
    try:
        if float(player_row['TeamRank']) <= 10:
            score += 3
    except:
        pass
    
    try:
        if int(player_row['FinalsGames']) == 4:
            score += 4
    except:
        pass
    
    try:
        if float(player_row['FinalsMult']) >= 1.04:
            score += 2
    except:
        pass
    
    return score

def calculate_equity(roster_df):
    """Calculate AdvanceEquity and WinEquity for roster"""
    if roster_df.empty:
        return 0, 0
    
    try:
        advance_equity = (roster_df['R2Games'] * roster_df['R2Mult'] * 1.0 + 
                         roster_df['R3Games'] * roster_df['R3Mult'] * 1.35).sum()
        
        win_equity = (roster_df['FinalsGames'] * roster_df['FinalsMult'] * 1.75).sum()
        
        return advance_equity, win_equity
    except:
        return 0, 0

def display_player_table(df, show_drafted=False, show_available=True, show_my_roster=False):
    """Display the main player table with filtering"""
    
    # Filter based on view mode
    if show_my_roster:
        display_df = df[df['ID'].isin(st.session_state.my_roster)]
    elif show_drafted:
        display_df = df[df['ID'].isin(st.session_state.drafted_players)]
    elif show_available:
        display_df = df[~df['ID'].isin(st.session_state.drafted_players)]
    else:
        display_df = df.copy()
    
    if display_df.empty:
        st.info("No players match the current filters.")
        return None
    
    # Select columns to display
    base_cols = ['Name', 'Position', 'Team', 'ADP', 'TeamRank', 
                 'R2Games', 'R3Games', 'FinalsGames',
                 'FinalAdjGPP', 'FinalAdjGPP_Rank', 'ADP_Rank', 
                 'ValueScore', 'ValueZ', 'LiveScore', 'ValueAlert']
    
    # Only include columns that exist
    display_cols = [col for col in base_cols if col in display_df.columns]
    
    # Format the dataframe
    styled_df = display_df[display_cols].copy()
    
    # Format numeric columns safely
    for col in ['ADP', 'R2Mult', 'R3Mult', 'FinalsMult', 'FinalAdjGPP', 'LiveScore']:
        if col in styled_df.columns:
            styled_df[col] = pd.to_numeric(styled_df[col], errors='coerce').round(1)
    
    if 'ValueScore' in styled_df.columns:
        styled_df['ValueScore'] = pd.to_numeric(styled_df['ValueScore'], errors='coerce').fillna(0).astype(int)
    
    if 'ValueZ' in styled_df.columns:
        styled_df['ValueZ'] = pd.to_numeric(styled_df['ValueZ'], errors='coerce').round(2)
    
    # Display with clickable rows
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            "ValueAlert": st.column_config.CheckboxColumn("🚨 Value"),
            "ValueScore": st.column_config.NumberColumn("Value Score", help="ADP_Rank - FinalAdjGPP_Rank"),
            "ValueZ": st.column_config.NumberColumn("Value Z", help="Z-score based value metric"),
        }
    )
    
    return display_df

def main():
    st.title("🏀 DraftKings NBA Best Ball Draft Assistant")
    st.subheader("Shootaround Tournament Optimizer")
    
    # File upload section
    if st.session_state.df is None:
        st.info("👆 Upload your draft data to get started!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Excel file (with 'Draft Board (values)' sheet) or CSV",
                type=['xlsx', 'csv'],
                help="Upload your DK Shootaround projections file"
            )
        
        with col2:
            st.markdown("### Required Columns:")
            st.markdown("""
            - Name
            - Position  
            - Team
            - FinalAdjGPP
            - ADP (can have blanks)
            - R2Games, R3Games, FinalsGames
            - R2Mult, R3Mult, FinalsMult
            - TeamRank
            """)
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        # Try to load Excel with required sheets
                        df = pd.read_excel(uploaded_file, sheet_name='Draft Board (values)')
                        
                        # Try to load multipliers
                        try:
                            env_df = pd.read_excel(uploaded_file, sheet_name='Week Environment (actual)')
                            mult_cols = ['Team', 'R2Mult', 'R3Mult', 'FinalsMult']
                            existing_mult_cols = [col for col in mult_cols if col in env_df.columns]
                            if existing_mult_cols:
                                df = df.merge(env_df[existing_mult_cols], on='Team', how='left', suffixes=('', '_env'))
                                # Use env values if they exist
                                for col in ['R2Mult', 'R3Mult', 'FinalsMult']:
                                    if f'{col}_env' in df.columns:
                                        df[col] = df[f'{col}_env'].fillna(df.get(col, 1.0))
                                        df = df.drop(columns=[f'{col}_env'])
                        except Exception as e:
                            st.warning(f"Could not load 'Week Environment' sheet: {e}. Using default multipliers.")
                        
                        # Try to load team rankings
                        try:
                            team_df = pd.read_excel(uploaded_file, sheet_name='Team Schedule (actual)')
                            if 'TeamRank' in team_df.columns:
                                if 'TeamRank' not in df.columns or df['TeamRank'].isna().all():
                                    df = df.drop(columns=['TeamRank'], errors='ignore')
                                    df = df.merge(team_df[['Team', 'TeamRank']], on='Team', how='left')
                        except Exception as e:
                            st.warning(f"Could not load 'Team Schedule' sheet: {e}. Using default rankings.")
                    
                    # Process and store data
                    df = process_data(df)
                    
                    if df is None or len(df) == 0:
                        st.error("No valid data found after processing. Please check your file.")
                        return
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded successfully! {len(df)} players ready.")
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.info("""
                **Troubleshooting Tips:**
                
                1. **Missing ADP values:** This is OK! Players without ADP will be ranked last.
                
                2. **For Excel files:** Make sure you have a sheet named "Draft Board (values)"
                
                3. **For CSV files:** Export the "Draft Board (values)" sheet as CSV
                
                4. **Required columns:** Name, Position, Team, FinalAdjGPP
                
                5. **Optional columns:** ADP, TeamRank, R2Games, R3Games, FinalsGames, R2Mult, R3Mult, FinalsMult
                """)
        return
    
    # Data is loaded, show the app
    df = st.session_state.df
    df = calculate_live_score(df, st.session_state.finals_emphasis)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Data management
        if st.button("📤 Upload New Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.data_loaded = False
            st.session_state.drafted_players = set()
            st.session_state.my_roster = []
            st.rerun()
        
        st.divider()
        
        # Finals Emphasis Slider
        st.subheader("Finals Bias")
        finals_emphasis = st.slider(
            "Finals Emphasis",
            min_value=0.8,
            max_value=1.4,
            value=st.session_state.finals_emphasis,
            step=0.05,
            help="Adjust how much to weight Finals games in recommendations"
        )
        st.session_state.finals_emphasis = finals_emphasis
        
        st.divider()
        
        # View Mode
        st.subheader("📋 View Mode")
        view_mode = st.radio(
            "Show players:",
            ["Available Only", "Drafted Only", "My Roster", "All Players"],
            index=0
        )
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filters")
        
        # Position filter
        positions = ['All'] + sorted(df['Position'].unique().tolist())
        position_filter = st.multiselect("Position", positions, default=['All'])
        
        # Team filter
        teams = ['All'] + sorted(df['Team'].unique().tolist())
        team_filter = st.multiselect("Team", teams, default=['All'])
        
        # TeamRank tier filter
        rank_tier = st.selectbox(
            "Team Rank Tier",
            ["All", "1-5 (Elite)", "6-10 (Good)", "11-20 (Mid)", "21-30 (Weak)"]
        )
        
        # Finals games filter
        unique_finals = sorted(df['FinalsGames'].unique().tolist())
        finals_games_filter = st.multiselect(
            "Finals Games",
            unique_finals,
            default=unique_finals
        )
        
        # Search box
        search_term = st.text_input("🔎 Search Player Name", "")
        
        st.divider()
        
        # Draft controls
        st.subheader("🎯 Draft Actions")
        
        if st.button("🔄 Reset Draft", use_container_width=True):
            st.session_state.drafted_players = set()
            st.session_state.my_roster = []
            st.rerun()
        
        if st.button("↩️ Undo Last Pick", use_container_width=True):
            if st.session_state.my_roster:
                last_pick = st.session_state.my_roster.pop()
                st.session_state.drafted_players.discard(last_pick)
                st.rerun()
        
        st.divider()
        
        # Roster summary
        st.subheader("👥 My Roster")
        st.metric("Players Drafted", len(st.session_state.my_roster))
        
        if st.session_state.my_roster:
            my_roster_df = df[df['ID'].isin(st.session_state.my_roster)]
            advance_eq, win_eq = calculate_equity(my_roster_df)
            
            st.metric("Advance Equity", f"{advance_eq:.1f}")
            st.metric("Win Equity", f"{win_eq:.1f}")
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'All' not in position_filter and position_filter:
        filtered_df = filtered_df[filtered_df['Position'].isin(position_filter)]
    
    if 'All' not in team_filter and team_filter:
        filtered_df = filtered_df[filtered_df['Team'].isin(team_filter)]
    
    if rank_tier != "All":
        if rank_tier == "1-5 (Elite)":
            filtered_df = filtered_df[filtered_df['TeamRank'] <= 5]
        elif rank_tier == "6-10 (Good)":
            filtered_df = filtered_df[(filtered_df['TeamRank'] >= 6) & (filtered_df['TeamRank'] <= 10)]
        elif rank_tier == "11-20 (Mid)":
            filtered_df = filtered_df[(filtered_df['TeamRank'] >= 11) & (filtered_df['TeamRank'] <= 20)]
        elif rank_tier == "21-30 (Weak)":
            filtered_df = filtered_df[filtered_df['TeamRank'] >= 21]
    
    if finals_games_filter:
        filtered_df = filtered_df[filtered_df['FinalsGames'].isin(finals_games_filter)]
    
    if search_term:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(search_term, case=False, na=False)]
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Draft Board", "⏰ On The Clock", "📈 Analytics"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Draft Board")
        
        with col2:
            st.metric("Total Players", len(filtered_df))
        
        # Display appropriate view
        if view_mode == "Available Only":
            display_df = display_player_table(filtered_df, show_available=True, show_drafted=False)
        elif view_mode == "Drafted Only":
            display_df = display_player_table(filtered_df, show_available=False, show_drafted=True)
        elif view_mode == "My Roster":
            display_df = display_player_table(filtered_df, show_my_roster=True)
        else:
            display_df = display_player_table(filtered_df, show_available=False, show_drafted=False)
        
        # Player selection interface
        if display_df is not None and not display_df.empty:
            st.divider()
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                available_players = display_df[~display_df['ID'].isin(st.session_state.drafted_players)]
                player_options = {f"{row['Name']} ({row['Position']}, {row['Team']})": row['ID'] 
                                for _, row in available_players.iterrows()}
                
                if player_options:
                    selected_player = st.selectbox(
                        "Select player to draft:",
                        options=list(player_options.keys()),
                        key="player_select"
                    )
            
            with col2:
                if player_options and st.button("Draft to My Team", use_container_width=True):
                    player_id = player_options[selected_player]
                    st.session_state.drafted_players.add(player_id)
                    st.session_state.my_roster.append(player_id)
                    st.success(f"✅ Drafted {selected_player}")
                    st.rerun()
            
            with col3:
                if player_options and st.button("Mark as Drafted", use_container_width=True):
                    player_id = player_options[selected_player]
                    st.session_state.drafted_players.add(player_id)
                    st.info(f"Marked {selected_player} as drafted")
                    st.rerun()
    
    with tab2:
        st.subheader("⏰ On The Clock - Recommendations")
        
        available_df = filtered_df[~filtered_df['ID'].isin(st.session_state.drafted_players)].copy()
        
        if available_df.empty:
            st.warning("No players available matching current filters.")
        else:
            my_roster_df = df[df['ID'].isin(st.session_state.my_roster)]
            available_df['StackScore'] = available_df.apply(
                lambda row: calculate_stack_score(row, my_roster_df), axis=1
            )
            
            available_df['SafetyScore'] = (
                available_df['FinalAdjGPP'] * 0.6 +
                (31 - available_df['TeamRank']) * 10 +
                available_df['FinalsGames'] * 20 +
                (1 - available_df['ShutdownRisk']) * 100
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🎯 Best Value Picks")
                value_picks = available_df.nlargest(5, 'ValueZ')[
                    ['Name', 'Position', 'Team', 'ValueZ', 'ValueScore', 'LiveScore', 'ADP']
                ].head(5)
                
                for idx, row in value_picks.iterrows():
                    with st.container():
                        st.markdown(f"**{row['Name']}** ({row['Position']}, {row['Team']})")
                        st.write(f"ValueZ: {row['ValueZ']:.2f} | Value: {row['ValueScore']} | ADP: {row['ADP']:.1f}")
                        st.write(f"LiveScore: {row['LiveScore']:.1f}")
                        st.divider()
            
            with col2:
                st.markdown("### 🔗 Best Stack Picks")
                stack_picks = available_df.nlargest(3, 'StackScore')[
                    ['Name', 'Position', 'Team', 'StackScore', 'TeamRank', 'FinalsGames', 'LiveScore']
                ].head(3)
                
                for idx, row in stack_picks.iterrows():
                    with st.container():
                        st.markdown(f"**{row['Name']}** ({row['Position']}, {row['Team']})")
                        st.write(f"Stack Score: {row['StackScore']:.0f} | TeamRank: {row['TeamRank']:.0f}")
                        st.write(f"Finals Games: {row['FinalsGames']:.0f} | LiveScore: {row['LiveScore']:.1f}")
                        
                        team_count = len(my_roster_df[my_roster_df['Team'] == row['Team']])
                        if team_count > 0:
                            st.markdown(f"<span class='stack-badge'>Creates {team_count+1}-man stack!</span>", 
                                      unsafe_allow_html=True)
                        st.divider()
            
            with col3:
                st.markdown("### 🛡️ Safest Picks")
                safe_picks = available_df.nlargest(3, 'SafetyScore')[
                    ['Name', 'Position', 'Team', 'SafetyScore', 'TeamRank', 'ShutdownRisk', 'LiveScore']
                ].head(3)
                
                for idx, row in safe_picks.iterrows():
                    with st.container():
                        st.markdown(f"**{row['Name']}** ({row['Position']}, {row['Team']})")
                        st.write(f"Safety: {row['SafetyScore']:.1f} | TeamRank: {row['TeamRank']:.0f}")
                        st.write(f"Shutdown Risk: {row['ShutdownRisk']:.2%}")
                        st.write(f"LiveScore: {row['LiveScore']:.1f}")
                        st.divider()
    
    with tab3:
        st.subheader("📈 Team Exposure & Analytics")
        
        if st.session_state.my_roster:
            my_roster_df = df[df['ID'].isin(st.session_state.my_roster)]
            
            st.markdown("#### Team Breakdown")
            team_summary = my_roster_df.groupby('Team').agg({
                'Name': 'count',
                'TeamRank': 'first',
                'R2Games': 'first',
                'R3Games': 'first',
                'FinalsGames': 'first'
            }).reset_index()
            team_summary.columns = ['Team', 'Count', 'TeamRank', 'R2Games', 'R3Games', 'FinalsGames']
            team_summary = team_summary.sort_values('Count', ascending=False)
            
            st.dataframe(team_summary, use_container_width=True, hide_index=True)
            
            st.markdown("#### 🏆 Tournament Equity Meters")
            
            advance_eq, win_eq = calculate_equity(my_roster_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Advance Equity", f"{advance_eq:.1f}")
                st.progress(min(advance_eq / 500, 1.0))
                st.caption("Target: ~400-500 for competitive advance chances")
            
            with col2:
                st.metric("Win Equity", f"{win_eq:.1f}")
                st.progress(min(win_eq / 300, 1.0))
                st.caption("Target: ~250-300 for championship equity")
            
            st.markdown("#### Position Breakdown")
            pos_breakdown = my_roster_df['Position'].value_counts().reset_index()
            pos_breakdown.columns = ['Position', 'Count']
            
            fig = go.Figure(data=[
                go.Bar(x=pos_breakdown['Position'], y=pos_breakdown['Count'],
                      marker_color='#1f77b4')
            ])
            fig.update_layout(
                title="Players by Position",
                xaxis_title="Position",
                yaxis_title="Count",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Draft some players to see analytics!")

if __name__ == "__main__":
    main()
