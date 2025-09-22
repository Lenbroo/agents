import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import time

# --- Page Configuration ---
# Set the layout to wide mode for a better dashboard experience
st.set_page_config(layout="wide", page_title="Agent Performance Dashboard")

# --- Main Title ---
st.title("📊 Agent Performance Dashboard")
st.markdown("Upload an Excel file with agent login data to generate an interactive report.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# --- Helper Functions (from original script) ---
def get_violation_reason(dt):
    """Checks if a datetime object is outside defined working hours."""
    if not isinstance(dt, pd.Timestamp):
        return ""
    day_of_week = dt.dayofweek
    time_in_minutes = dt.hour * 60 + dt.minute
    # Saturday
    if day_of_week == 5:
        return "Activity on a Saturday (Holiday)"
    # Sunday
    elif day_of_week == 6:
        if not (420 <= time_in_minutes <= 900): # 7:00 AM - 3:00 PM
            return f"Sunday - Outside 7:00 AM - 3:00 PM"
    # Monday to Thursday
    elif 0 <= day_of_week <= 3:
        if not (480 <= time_in_minutes <= 1260): # 8:00 AM - 9:00 PM
            return f"Mon-Thu - Outside 8:00 AM - 9:00 PM"
    # Friday
    elif day_of_week == 4:
        is_in_morning_shift = (450 <= time_in_minutes <= 720)   # 7:30 AM - 12:00 PM
        is_in_evening_shift = (840 <= time_in_minutes <= 1230) # 2:00 PM - 8:30 PM
        if not (is_in_morning_shift or is_in_evening_shift):
            return "Friday - Outside official shifts"
    return ""

def get_target_seconds(row):
    """Calculates the target logged-in seconds based on the day of the week and login time."""
    day_of_week = row['Day'].dayofweek
    login_time = row['Logged In'].time()
    # Monday to Thursday: 7 hours 45 mins
    if 0 <= day_of_week <= 3:
        return (7 * 3600) + (45 * 60)
    # Friday: Two possible shifts
    elif day_of_week == 4:
        if time(7, 0) <= login_time < time(13, 0): return 4 * 3600 # Morning shift: 4 hours
        elif time(13, 30) <= login_time < time(21, 0): return 6 * 3600 # Evening shift: 6 hours
        else: return 0
    # Sunday: 7 hours 30 mins
    elif day_of_week == 6:
        return (7 * 3600) + (30 * 60)
    # Saturday or other cases
    else:
        return 0

def seconds_to_hms_str(seconds):
    """Converts seconds into a HH:MM:SS string format."""
    if seconds < 0: seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

# --- Main Application Logic ---
# This block runs only if a file has been uploaded.
if uploaded_file is not None:
    try:
        # --- Step 1: Load and Prepare the Data ---
        df = pd.read_excel(uploaded_file, header=0)
        df.columns = df.columns.str.strip()

        # --- Step 2: Data Cleaning and Type Conversion ---
        required_columns = ['Day', 'Agent', 'Logged In', 'Logged Out', 'Total Logged In per day']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Error: The uploaded file is missing one or more required columns. Please ensure it contains: {', '.join(required_columns)}")
        else:
            df = df[pd.to_datetime(df['Day'], errors='coerce').notna()]
            df['Day'] = pd.to_datetime(df['Day'])
            df['Logged In'] = pd.to_datetime(df['Logged In'], errors='coerce')
            df['Logged Out'] = pd.to_datetime(df['Logged Out'], errors='coerce')
            df['Logged_in_seconds'] = pd.to_timedelta(df['Total Logged In per day'].astype(str), errors='coerce').dt.total_seconds().fillna(0)
            df.dropna(subset=['Logged In', 'Logged Out'], inplace=True)

            # --- Step 3: Outside Working Hours (OWH) Logic ---
            df['Login Violation'] = df['Logged In'].apply(get_violation_reason)
            df['Logout Violation'] = df['Logged Out'].apply(get_violation_reason)
            violations_df = df[(df['Login Violation'] != "") | (df['Logout Violation'] != "")].copy()
            violations_df['Violation Reason'] = violations_df.apply(lambda row: " | ".join(filter(None, [row['Login Violation'], row['Logout Violation']])), axis=1)

            # --- Step 4: Chart-Specific Data Aggregation ---
            df['Target_seconds'] = df.apply(get_target_seconds, axis=1)
            df = df.sort_values(by=['Agent', 'Logged In'])
            df_agg = df.groupby(['Agent', 'Day']).agg(
                Logged_in_seconds=('Logged_in_seconds', 'sum'),
                Target_seconds=('Target_seconds', 'first')
            ).reset_index()

            df_agg['Remaining_seconds'] = df_agg['Target_seconds'] - df_agg['Logged_in_seconds']
            df_agg['Remaining_seconds'] = df_agg['Remaining_seconds'].apply(lambda x: max(x, 0))

            text_threshold = 30 * 60
            df_agg['Logged_in_text'] = df_agg.apply(lambda r: seconds_to_hms_str(r['Logged_in_seconds']) if r['Logged_in_seconds'] > text_threshold else '', axis=1)
            df_agg['Remaining_text'] = df_agg.apply(lambda r: seconds_to_hms_str(r['Remaining_seconds']) if r['Remaining_seconds'] > text_threshold else '', axis=1)

            donut_data = df_agg.groupby('Agent').agg(
                Total_Logged_In=('Logged_in_seconds', 'sum'),
                Total_Remaining=('Remaining_seconds', 'sum')
            ).reset_index()
            
            # --- Check if there is data to plot ---
            if df_agg.empty or 'Agent' not in df_agg.columns:
                 st.warning("No valid agent data found in the uploaded file to generate a dashboard.")
            else:
                agents = sorted(df_agg['Agent'].unique())

                # --- Step 5: Create the Interactive Figure with Subplots ---
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
                
                # Each agent has 5 traces: 2 bars, 1 violation table, 1 summary table, 1 pie chart.
                num_traces_per_agent = 5 

                # --- Add all traces (initially visible for the first agent) ---
                for agent in agents:
                    agent_df = df_agg[df_agg['Agent'] == agent]
                    is_visible = (agent == agents[0])
                    # Bar Traces
                    fig.add_trace(go.Bar(x=agent_df['Day'], y=agent_df['Logged_in_seconds'], name='Logged-in Time', marker_color='#1f77b4', text=agent_df['Logged_in_text'], textposition='inside', textfont=dict(color='white'), visible=is_visible), row=1, col=1)
                    fig.add_trace(go.Bar(x=agent_df['Day'], y=agent_df['Remaining_seconds'], name='Remaining Time', marker_color='#e0e0e0', text=agent_df['Remaining_text'], textposition='inside', textfont=dict(color='black'), visible=is_visible), row=1, col=1)

                    # Violation Table Trace
                    agent_violations = violations_df[violations_df['Agent'] == agent]
                    if agent_violations.empty:
                        header_values, cell_values = ["Violation Status"], [["No violations found"]]
                    else:
                        table_df = agent_violations[['Day', 'Logged In', 'Logged Out', 'Violation Reason']].copy()
                        table_df['Day'] = table_df['Day'].dt.strftime('%Y-%m-%d')
                        table_df['Logged In'] = table_df['Logged In'].dt.strftime('%H:%M:%S')
                        table_df['Logged Out'] = table_df['Logged Out'].dt.strftime('%H:%M:%S')
                        header_values = list(table_df.columns)
                        cell_values = [table_df[col] for col in table_df.columns]
                    fig.add_trace(go.Table(header=dict(values=header_values, fill_color='paleturquoise', align='left'), cells=dict(values=cell_values, fill_color='lavender', align='left'), visible=is_visible), row=2, col=1)

                    # Daily Summary Table Trace
                    agent_summary_df = agent_df[['Day', 'Logged_in_seconds', 'Remaining_seconds']].rename(columns={'Day': 'Date', 'Logged_in_seconds': 'Total Logged-in', 'Remaining_seconds': 'Total Remaining'})
                    agent_summary_df['Date'] = agent_summary_df['Date'].dt.strftime('%Y-%m-%d')
                    agent_summary_df['Total Logged-in'] = agent_summary_df['Total Logged-in'].apply(seconds_to_hms_str)
                    agent_summary_df['Total Remaining'] = agent_summary_df['Total Remaining'].apply(seconds_to_hms_str)
                    fig.add_trace(go.Table(header=dict(values=list(agent_summary_df.columns), fill_color='#D3D3D3', align='left', font=dict(color='black')), cells=dict(values=[agent_summary_df[col] for col in agent_summary_df.columns], fill_color='#F5F5F5', align='left'), visible=is_visible), row=3, col=1)

                    # Donut Chart Trace
                    agent_donut_data = donut_data[donut_data['Agent'] == agent]
                    total_logged_in = agent_donut_data['Total_Logged_In'].iloc[0] if not agent_donut_data.empty else 0
                    total_remaining = agent_donut_data['Total_Remaining'].iloc[0] if not agent_donut_data.empty else 0
                    pie_values = [total_logged_in, total_remaining]
                    fig.add_trace(go.Pie(labels=['Total Logged-in Time', 'Total Remaining Time'], values=pie_values, hole=0.4, textinfo='percent+label', texttemplate='%{label}<br>%{percent}', hoverinfo='label+value', hovertemplate='%{label}: %{customdata}<extra></extra>', customdata=[seconds_to_hms_str(s) for s in pie_values], marker_colors=['#1f77b4', '#e0e0e0'], visible=is_visible, title=dict(text='Overall Time Distribution', position='top center')), row=3, col=2)
                
                # --- Step 6: Create Buttons with CORRECTED Logic ---
                buttons = []
                total_traces = len(agents) * num_traces_per_agent
                
                for i, agent in enumerate(agents):
                    # Create a visibility list that is False for all traces
                    visibility = [False] * total_traces
                    
                    # Set the 5 traces for the current agent to True
                    start_index = i * num_traces_per_agent
                    for j in range(num_traces_per_agent):
                        visibility[start_index + j] = True
                    
                    buttons.append(dict(
                        label=agent,
                        method='update',
                        args=[{'visible': visibility}, {'title': f'Daily Performance for: {agent}'}]
                    ))

                # --- Step 7: Finalize Layout ---
                fig.update_layout(
                    updatemenus=[dict(
                        type="buttons", direction="right", active=0, buttons=buttons,
                        pad={"r": 10, "t": 10}, showactive=True, x=0.5, xanchor="center", y=1.15, yanchor="top"
                    )],
                    title_text=f'Daily Performance for: {agents[0]}',
                    title_x=0.5,
                    barmode='stack',
                    height=1000,
                    showlegend=False,
                    margin=dict(l=40, r=40, t=120, b=40)
                )
                fig.update_yaxes(
                    title_text="Total Time",
                    tickvals=[i * 3600 for i in range(12)],
                    ticktext=[f'{i:02d}:00:00' for i in range(12)],
                    row=1, col=1
                )
                fig.update_xaxes(title_text="Date", row=1, col=1)

                # --- Display the chart in Streamlit ---
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please ensure your Excel file is correctly formatted and not corrupted.")

# --- Initial Message when no file is uploaded ---
else:
    st.info("Upload an Excel file to get started.")
