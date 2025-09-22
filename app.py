import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Agent Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Color Palette ---
COLOR_LOGGED_IN = '#0d6efd'         # Primary Blue
COLOR_REMAINING = '#e9ecef'         # Light Gray
COLOR_VIOLATION_HEADER = '#f8d7da'  # Soft Red for alerts
COLOR_VIOLATION_CELL = '#fdf3f4'    # Lighter Red
COLOR_SUMMARY_HEADER = '#dee2e6'    # Neutral Gray
COLOR_SUMMARY_CELL = '#f8f9fa'      # Off-White
COLOR_TEXT_ON_DARK = '#FFFFFF'      # White
COLOR_TEXT_ON_LIGHT = '#212529'     # Dark Gray (almost black)

# --- Data Processing Functions ---

def load_and_clean_data(uploaded_file):
    """
    Loads data from the uploaded Excel file, performs initial validation and cleaning.
    Returns a cleaned DataFrame or None if errors occur.
    """
    try:
        df = pd.read_excel(uploaded_file, header=0)
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Failed to read the Excel file. Error: {e}")
        return None

    required_cols = ['Day', 'Agent', 'Logged In', 'Logged Out', 'Total Logged In per day']
    if not all(col in df.columns for col in required_cols):
        st.error(f"The uploaded file is missing one or more required columns: {required_cols}")
        return None

    df.dropna(subset=required_cols, inplace=True)
    df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
    df.dropna(subset=['Day'], inplace=True)
    
    if df.empty:
        st.warning("The file contains no valid data rows after initial cleaning.")
        return None
        
    df['Logged In'] = pd.to_datetime(df['Logged In'], errors='coerce')
    df['Logged Out'] = pd.to_datetime(df['Logged Out'], errors='coerce')
    df['Logged_in_seconds'] = pd.to_timedelta(df['Total Logged In per day'].astype(str), errors='coerce').dt.total_seconds()
    df.dropna(subset=['Logged In', 'Logged Out', 'Logged_in_seconds'], inplace=True)

    if df.empty:
        st.warning("No valid time entries found. Please check 'Logged In', 'Logged Out', and 'Total Logged In per day' columns.")
        return None

    return df

def calculate_metrics_and_violations(df):
    """
    Calculates all business logic: violations, targets, and aggregated stats.
    Returns three DataFrames: aggregated data, violations data, and donut chart data.
    """
    # --- Violation Logic ---
    def get_violation_reason(dt):
        if pd.isna(dt): return ""
        day_of_week = dt.dayofweek
        time_in_minutes = dt.hour * 60 + dt.minute
        if day_of_week == 5: return "Activity on a Saturday (Holiday)"
        if day_of_week == 6 and not (420 <= time_in_minutes <= 900): return "Sunday - Outside 7:00 AM - 3:00 PM"
        if 0 <= day_of_week <= 3 and not (480 <= time_in_minutes <= 1260): return "Mon-Thu - Outside 8:00 AM - 9:00 PM"
        if day_of_week == 4:
            is_morning = 450 <= time_in_minutes <= 720
            is_evening = 840 <= time_in_minutes <= 1230
            if not (is_morning or is_evening): return "Friday - Outside official shifts"
        return ""

    df['Login Violation'] = df['Logged In'].apply(get_violation_reason)
    df['Logout Violation'] = df['Logged Out'].apply(get_violation_reason)
    violations_df = df[(df['Login Violation'] != "") | (df['Logout Violation'] != "")].copy()
    violations_df['Violation Reason'] = violations_df.apply(lambda row: " | ".join(filter(None, [row['Login Violation'], row['Logout Violation']])), axis=1)

    # --- Target and Aggregation Logic ---
    def get_target_seconds(row):
        day_of_week = row['Day'].dayofweek
        login_time = row['Logged In'].time()
        if 0 <= day_of_week <= 3: return (7 * 3600) + (45 * 60)
        if day_of_week == 4:
            if time(7, 0) <= login_time < time(13, 0): return 4 * 3600
            if time(13, 30) <= login_time < time(21, 0): return 6 * 3600
        if day_of_week == 6: return (7 * 3600) + (30 * 60)
        return 0
        
    df['Target_seconds'] = df.apply(get_target_seconds, axis=1)
    df.sort_values(by=['Agent', 'Logged In'], inplace=True)
    df_agg = df.groupby(['Agent', 'Day']).agg(
        Logged_in_seconds=('Logged_in_seconds', 'sum'),
        Target_seconds=('Target_seconds', 'first')
    ).reset_index()

    df_agg['Remaining_seconds'] = (df_agg['Target_seconds'] - df_agg['Logged_in_seconds']).clip(lower=0)
    
    donut_data = df_agg.groupby('Agent').agg(
        Total_Logged_In=('Logged_in_seconds', 'sum'),
        Total_Remaining=('Remaining_seconds', 'sum')
    ).reset_index()
    
    return df_agg, violations_df, donut_data

# --- Visualization Function ---

def create_figure(df_agg, violations_df, donut_data):
    """
    Creates the complete Plotly figure with all subplots.
    """
    def seconds_to_hms_str(seconds):
        if pd.isna(seconds) or seconds < 0: return "00:00:00"
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    text_threshold = 30 * 60
    df_agg['Logged_in_text'] = df_agg.apply(lambda r: seconds_to_hms_str(r['Logged_in_seconds']) if r['Logged_in_seconds'] > text_threshold else '', axis=1)
    df_agg['Remaining_text'] = df_agg.apply(lambda r: seconds_to_hms_str(r['Remaining_seconds']) if r['Remaining_seconds'] > text_threshold else '', axis=1)

    fig = make_subplots(
        rows=3, cols=2, vertical_spacing=0.1,
        specs=[[{"type": "bar", "colspan": 2}, None], [{"type": "table", "colspan": 2}, None], [{"type": "table"}, {"type": "pie"}]],
        row_heights=[0.5, 0.2, 0.3], column_widths=[0.6, 0.4]
    )
    agents = df_agg['Agent'].unique()
    
    for agent in agents:
        agent_df = df_agg[df_agg['Agent'] == agent]
        is_visible = (agent == agents[0])
        # Bar Traces
        fig.add_trace(go.Bar(x=agent_df['Day'], y=agent_df['Logged_in_seconds'], name='Logged-in Time', marker_color=COLOR_LOGGED_IN, text=agent_df['Logged_in_text'], textposition='inside', textfont=dict(color=COLOR_TEXT_ON_DARK, size=12), visible=is_visible), row=1, col=1)
        fig.add_trace(go.Bar(x=agent_df['Day'], y=agent_df['Remaining_seconds'], name='Remaining Time', marker_color=COLOR_REMAINING, text=agent_df['Remaining_text'], textposition='inside', textfont=dict(color=COLOR_TEXT_ON_LIGHT, size=12), visible=is_visible), row=1, col=1)

        # Violation Table
        agent_violations = violations_df[violations_df['Agent'] == agent]
        if agent_violations.empty:
            header, cells = ["Violation Status"], [["No violations found"]]
        else:
            tbl = agent_violations[['Day', 'Logged In', 'Logged Out', 'Violation Reason']].copy()
            tbl['Day'] = tbl['Day'].dt.strftime('%Y-%m-%d')
            tbl['Logged In'] = tbl['Logged In'].dt.strftime('%H:%M:%S')
            tbl['Logged Out'] = tbl['Logged Out'].dt.strftime('%H:%M:%S')
            header, cells = list(tbl.columns), [tbl[c] for c in tbl.columns]
        fig.add_trace(go.Table(header=dict(values=header, fill_color=COLOR_VIOLATION_HEADER, font=dict(color=COLOR_TEXT_ON_LIGHT, size=12), align='left'), cells=dict(values=cells, fill_color=COLOR_VIOLATION_CELL, align='left', font=dict(size=11)), visible=is_visible), row=2, col=1)

        # Daily Summary Table
        summary_tbl = agent_df[['Day', 'Logged_in_seconds', 'Remaining_seconds']]
        summary_tbl.columns = ['Date', 'Total Logged-in', 'Total Remaining']
        summary_tbl['Date'] = summary_tbl['Date'].dt.strftime('%Y-%m-%d')
        summary_tbl['Total Logged-in'] = summary_tbl['Total Logged-in'].apply(seconds_to_hms_str)
        summary_tbl['Total Remaining'] = summary_tbl['Total Remaining'].apply(seconds_to_hms_str)
        fig.add_trace(go.Table(header=dict(values=list(summary_tbl.columns), fill_color=COLOR_SUMMARY_HEADER, align='left', font=dict(size=12, color=COLOR_TEXT_ON_LIGHT)), cells=dict(values=[summary_tbl[c] for c in summary_tbl.columns], fill_color=COLOR_SUMMARY_CELL, align='left', font=dict(size=11)), visible=is_visible), row=3, col=1)
        
        # Donut Chart
        donut = donut_data[donut_data['Agent'] == agent]
        labels, values = ['Logged-in', 'Remaining'], [donut['Total_Logged_In'].iloc[0], donut['Total_Remaining'].iloc[0]]
        fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4, textinfo='percent+label', customdata=[seconds_to_hms_str(s) for s in values], hovertemplate='%{label}: %{customdata}<extra></extra>', marker_colors=[COLOR_LOGGED_IN, COLOR_REMAINING], visible=is_visible, title=dict(text='Overall Time Distribution', font_size=14)), row=3, col=2)

    buttons = []
    for i, agent in enumerate(agents):
        visibility = [False] * (len(agents) * 5)
        visibility[i*2] = visibility[i*2 + 1] = True  # Bars
        visibility[len(agents)*2 + i] = True           # Violation Table
        visibility[len(agents)*3 + i] = True           # Summary Table
        visibility[len(agents)*4 + i] = True           # Donut
        buttons.append(dict(label=agent, method='update', args=[{'visible': visibility}, {'title': f'Performance Dashboard: {agent}'}]))

    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right", active=0, buttons=buttons, pad={"r": 10, "t": 10}, showactive=True, x=0.5, xanchor="center", y=1.15, yanchor="top")],
        title_text=f'Performance Dashboard: {agents[0]}', title_x=0.5, barmode='stack', height=1000, showlegend=False, margin=dict(l=40, r=40, t=120, b=40)
    )
    fig.update_yaxes(title_text="Total Time", tickvals=[i * 3600 for i in range(12)], ticktext=[f'{i:02d}:00' for i in range(12)], row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    
    return fig

# --- Streamlit UI ---
st.title("📊 Agent Performance and Violations Dashboard")
st.markdown("Upload your agent activity Excel file to visualize daily performance, violations, and time metrics.")

with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    with st.spinner('Processing data and generating visuals... This may take a moment.'):
        raw_df = load_and_clean_data(uploaded_file)
        if raw_df is not None:
            df_agg, violations_df, donut_data = calculate_metrics_and_violations(raw_df)
            if df_agg is not None and not df_agg.empty:
                fig = create_figure(df_agg, violations_df, donut_data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate dashboard. The processed data is empty.")
else:
    st.info("Please upload an agent activity file using the sidebar to begin.")

