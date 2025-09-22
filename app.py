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

# --- Main Dashboard Function ---
def create_dashboard(df):
    """
    Processes the dataframe and generates the Plotly figure.
    """
    # --- Step 2: Data Cleaning and Type Conversion ---
    if 'Logged Out' not in df.columns:
        st.error("The 'Logged Out' column is missing from the uploaded file.")
        return go.Figure()

    df = df.dropna(subset=['Day', 'Logged In', 'Logged Out', 'Total Logged In per day'])
    df = df[pd.to_datetime(df['Day'], errors='coerce').notna()]
    if df.empty:
        st.warning("No valid date entries found in the 'Day' column after cleaning.")
        return go.Figure()

    df['Day'] = pd.to_datetime(df['Day'])
    df['Logged In'] = pd.to_datetime(df['Logged In'], errors='coerce')
    df['Logged Out'] = pd.to_datetime(df['Logged Out'], errors='coerce')
    df['Logged_in_seconds'] = pd.to_timedelta(df['Total Logged In per day'].astype(str), errors='coerce').dt.total_seconds()
    df = df.dropna(subset=['Logged In', 'Logged Out', 'Logged_in_seconds'])


    # --- Step 3: Outside Working Hours (OWH) Logic ---
    def get_violation_reason(dt):
        day_of_week = dt.dayofweek
        time_in_minutes = dt.hour * 60 + dt.minute
        if day_of_week == 5: return "Activity on a Saturday (Holiday)"
        elif day_of_week == 6:
            if not (420 <= time_in_minutes <= 900): return f"Sunday - Outside 7:00 AM - 3:00 PM"
        elif 0 <= day_of_week <= 3:
            if not (480 <= time_in_minutes <= 1260): return f"Mon-Thu - Outside 8:00 AM - 9:00 PM"
        elif day_of_week == 4:
            is_in_morning_shift = (450 <= time_in_minutes <= 720)
            is_in_evening_shift = (840 <= time_in_minutes <= 1230)
            if not (is_in_morning_shift or is_in_evening_shift): return "Friday - Outside official shifts"
        return ""

    df['Login Violation'] = df['Logged In'].apply(get_violation_reason)
    df['Logout Violation'] = df['Logged Out'].apply(get_violation_reason)
    violations_df = df[(df['Login Violation'] != "") | (df['Logout Violation'] != "")].copy()
    violations_df['Violation Reason'] = violations_df.apply(lambda row: " | ".join(filter(None, [row['Login Violation'], row['Logout Violation']])), axis=1)

    # --- Step 4: Chart-Specific Data Aggregation ---
    def get_target_seconds(row):
        day_of_week = row['Day'].dayofweek
        login_time = row['Logged In'].time()
        if 0 <= day_of_week <= 3: return (7 * 3600) + (45 * 60)
        elif day_of_week == 4:
            if time(7, 0) <= login_time < time(13, 0): return 4 * 3600
            elif time(13, 30) <= login_time < time(21, 0): return 6 * 3600
            else: return 0
        elif day_of_week == 6: return (7 * 3600) + (30 * 60)
        else: return 0
    df['Target_seconds'] = df.apply(get_target_seconds, axis=1)
    df = df.sort_values(by=['Agent', 'Logged In'])
    df_agg = df.groupby(['Agent', 'Day']).agg(Logged_in_seconds=('Logged_in_seconds', 'sum'), Target_seconds=('Target_seconds', 'first')).reset_index()
    df_agg['Remaining_seconds'] = df_agg['Target_seconds'] - df_agg['Logged_in_seconds']
    df_agg['Remaining_seconds'] = df_agg['Remaining_seconds'].apply(lambda x: max(x, 0))

    def seconds_to_hms_str(seconds):
        if pd.isna(seconds) or seconds < 0: seconds = 0
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    text_threshold = 30 * 60
    df_agg['Logged_in_text'] = df_agg.apply(lambda r: seconds_to_hms_str(r['Logged_in_seconds']) if r['Logged_in_seconds'] > text_threshold else '', axis=1)
    df_agg['Remaining_text'] = df_agg.apply(lambda r: seconds_to_hms_str(r['Remaining_seconds']) if r['Remaining_seconds'] > text_threshold else '', axis=1)

    donut_data = df_agg.groupby('Agent').agg(
        Total_Logged_In=('Logged_in_seconds', 'sum'),
        Total_Remaining=('Remaining_seconds', 'sum')
    ).reset_index()

    # --- Step 5: Create the Interactive Figure ---
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.08,
        specs=[
            [{"type": "bar", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None],
            [{"type": "table"}, {"type": "pie"}]
        ],
        row_heights=[0.5, 0.2, 0.3],
        column_widths=[0.6, 0.4]
    )

    agents = df_agg['Agent'].unique()
    if not agents.any():
        st.warning("No agent data to display after processing.")
        return go.Figure()

    # --- Add Traces for Each Agent ---
    for agent in agents:
        agent_df = df_agg[df_agg['Agent'] == agent]
        is_visible = (agent == agents[0])
        # Bar Traces
        fig.add_trace(go.Bar(x=agent_df['Day'], y=agent_df['Logged_in_seconds'], name='Logged-in Time', marker_color='#1f77b4', text=agent_df['Logged_in_text'], textposition='inside', textfont=dict(color='white', size=12), visible=is_visible), row=1, col=1)
        fig.add_trace(go.Bar(x=agent_df['Day'], y=agent_df['Remaining_seconds'], name='Remaining Time', marker_color='#e0e0e0', text=agent_df['Remaining_text'], textposition='inside', textfont=dict(color='black', size=12), visible=is_visible), row=1, col=1)

        # Violation Table
        agent_violations = violations_df[violations_df['Agent'] == agent]
        if agent_violations.empty:
            header_values, cell_values = ["Violation Status"], [["No violations found"]]
        else:
            table_df = agent_violations[['Day', 'Logged In', 'Logged Out', 'Violation Reason']].copy()
            table_df['Day'] = table_df['Day'].dt.strftime('%Y-%m-%d')
            table_df['Logged In'] = table_df['Logged In'].dt.strftime('%H:%M:%S')
            table_df['Logged Out'] = table_df['Logged Out'].dt.strftime('%H:%M:%S')
            header_values, cell_values = list(table_df.columns), [table_df[col] for col in table_df.columns]
        fig.add_trace(go.Table(header=dict(values=header_values, fill_color='paleturquoise', align='left', font=dict(size=12)), cells=dict(values=cell_values, fill_color='lavender', align='left', font=dict(size=11)), visible=is_visible), row=2, col=1)

        # Daily Summary Table
        agent_summary_df = agent_df[['Day', 'Logged_in_seconds', 'Remaining_seconds']].copy()
        agent_summary_df.rename(columns={'Day': 'Date', 'Logged_in_seconds': 'Total Logged-in', 'Remaining_seconds': 'Total Remaining'}, inplace=True)
        agent_summary_df['Date'] = agent_summary_df['Date'].dt.strftime('%Y-%m-%d')
        agent_summary_df['Total Logged-in'] = agent_summary_df['Total Logged-in'].apply(seconds_to_hms_str)
        agent_summary_df['Total Remaining'] = agent_summary_df['Total Remaining'].apply(seconds_to_hms_str)
        fig.add_trace(go.Table(header=dict(values=list(agent_summary_df.columns), fill_color='#D3D3D3', align='left', font=dict(size=12, color='black')), cells=dict(values=[agent_summary_df[col] for col in agent_summary_df.columns], fill_color='#F5F5F5', align='left', font=dict(size=11)), visible=is_visible), row=3, col=1)

        # Donut Chart
        agent_donut_data = donut_data[donut_data['Agent'] == agent]
        total_logged_in = agent_donut_data['Total_Logged_In'].iloc[0]
        total_remaining = agent_donut_data['Total_Remaining'].iloc[0]
        labels, values = ['Total Logged-in Time', 'Total Remaining Time'], [total_logged_in, total_remaining]
        fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4, textinfo='percent+label', texttemplate='%{label}<br>%{percent}', hoverinfo='label+value', hovertemplate='%{label}: %{customdata}<extra></extra>', customdata=[seconds_to_hms_str(s) for s in values], marker_colors=['#1f77b4', '#e0e0e0'], visible=is_visible, title=dict(text='Overall Time Distribution', position='top center')), row=3, col=2)

    # --- Step 6: Create Buttons ---
    buttons = []
    for i, agent in enumerate(agents):
        visibility = [False] * (len(agents) * 5)
        visibility[i*2] = True; visibility[i*2 + 1] = True  # Bars
        visibility[len(agents)*2 + i] = True  # Violation Table
        visibility[len(agents)*3 + i] = True  # Summary Table
        visibility[len(agents)*4 + i] = True  # Donut
        buttons.append(dict(label=agent, method='update', args=[{'visible': visibility}, {'title': f'Daily Performance for: {agent}'}]))

    # --- Step 7: Finalize Layout ---
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right", active=0, buttons=buttons, pad={"r": 10, "t": 10}, showactive=True, x=0.5, xanchor="center", y=1.15, yanchor="top")],
        title_text=f'Daily Performance for: {agents[0]}', title_x=0.5, barmode='stack', height=1000, showlegend=False, margin=dict(l=40, r=40, t=120, b=40)
    )
    fig.update_yaxes(title_text="Total Time", tickvals=[i * 3600 for i in range(12)], ticktext=[f'{i:02d}:00:00' for i in range(12)], row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)

    return fig

# --- Streamlit UI ---
st.title("📊 Agent Performance and Violations Dashboard")
st.markdown("""
Welcome to the interactive performance dashboard.
Upload your agent activity Excel file to visualize daily performance, violations, and time management metrics.
""")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    st.info("If no file is uploaded, the dashboard will load with sample data.")

# --- Data Loading Logic ---
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, header=0)
        df.columns = df.columns.str.strip()
        st.sidebar.success("File uploaded and loaded successfully!")
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        df = None
else:
    # Sample Data as fallback
    df = pd.DataFrame({
        'Day': ['2025-09-22', '2025-09-23', '2025-09-24', '2025-09-25', '2025-09-26', '2025-09-22', '2025-09-23'],
        'Agent': ['Agent A', 'Agent A', 'Agent A', 'Agent A', 'Agent B', 'Agent B', 'Agent B'],
        'Logged In': ['2025-09-22T07:55:00', '2025-09-23T21:30:00', '2025-09-24T08:10:00', '2025-09-25T14:00:00', '2025-09-26T12:30:00', '2025-09-22T09:05:00', '2025-09-23T10:00:00'],
        'Logged Out': ['2025-09-22T17:00:00', '2025-09-23T22:00:00', '2025-09-24T16:00:00', '2025-09-25T18:00:00', '2025-09-26T21:00:00', '2025-09-22T18:00:00', '2025-09-23T15:00:00'],
        'Total Logged In per day': ['08:05:00', '00:30:00', '07:50:00', '04:00:00', '08:30:00', '08:55:00', '05:00:00']
    })

# --- Display Dashboard ---
if df is not None:
    with st.spinner('Processing data and generating visuals...'):
        fig = create_dashboard(df)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please upload a valid Excel file to proceed.")
