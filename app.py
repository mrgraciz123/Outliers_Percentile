import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config for a premium wide layout
st.set_page_config(
    page_title="NYC Airbnb Outlier Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject premium fonts, styles, and card hover effects
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Metrics Card Grid Styling */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        flex: 1;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-container {
        display: flex;
        align-items: baseline;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .metric-value-raw {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ef4444; /* red for raw/distorted */
        text-decoration: line-through;
        opacity: 0.7;
    }
    
    .metric-value-clean {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981; /* green for cleaned */
    }
    
    .metric-value-single {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-color);
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Info box styling */
    .info-box {
        background-color: rgba(59, 130, 246, 0.05);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get correct filepath
@st.cache_data
def get_csv_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "AB_NYC_2019.csv")

# Load and cache raw dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    # Fill reviews per month NaN with 0
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)
    return df

csv_file = get_csv_path()

try:
    df_raw = load_data(csv_file)
except Exception as e:
    st.error(f"Error loading AB_NYC_2019.csv: {e}")
    st.stop()

# Header Area
st.markdown('<div class="main-title">🏙️ NYC Airbnb Price Outliers Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive percentile-based outlier detection, treatment, and analysis</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### ⚙️ Outlier Parameters")

# Percentile sliders
low_pct = st.sidebar.slider(
    "Lower Percentile Bound (%)", 
    min_value=0.0, 
    max_value=5.0, 
    value=1.0, 
    step=0.1,
    help="Listings below this percentile of price will be treated as lower outliers (e.g. abnormally low or $0 prices)."
)

high_pct = st.sidebar.slider(
    "Upper Percentile Bound (%)", 
    min_value=95.0, 
    max_value=100.0, 
    value=99.9, 
    step=0.1,
    help="Listings above this percentile of price will be treated as upper outliers (extremely expensive listings)."
)

# Sidebar Filter Section
st.sidebar.markdown("### 🔍 Categorical Filters")

# Neighborhood Groups Multi-select
all_neighborhood_groups = sorted(df_raw['neighbourhood_group'].unique())
selected_groups = st.sidebar.multiselect(
    "Neighbourhood Groups", 
    options=all_neighborhood_groups, 
    default=all_neighborhood_groups
)

# Room Type Multi-select
all_room_types = sorted(df_raw['room_type'].unique())
selected_rooms = st.sidebar.multiselect(
    "Room Types", 
    options=all_room_types, 
    default=all_room_types
)

# Map Configuration Sidebar Section
st.sidebar.markdown("### 🗺️ Map Settings")
map_style = st.sidebar.selectbox(
    "Map Style",
    options=["carto-positron", "open-street-map", "carto-darkmatter"],
    index=0
)
sample_size = st.sidebar.slider(
    "Geospatial Sample Size",
    min_value=500,
    max_value=len(df_raw),
    value=min(5000, len(df_raw)),
    step=500,
    help="For smoother browser performance, we sample the data points displayed on the map."
)

# ----------------- COMPUTATION -----------------
# 1. Calculate Outlier Thresholds based on entire raw dataset price column (matching exercise style)
min_threshold = df_raw['price'].quantile(low_pct / 100.0)
max_threshold = df_raw['price'].quantile(high_pct / 100.0)

# Ensure thresholds are valid
if min_threshold > max_threshold:
    min_threshold, max_threshold = max_threshold, min_threshold

# 2. Segment Raw Data into Cleaned and Outliers
df_cleaned_all = df_raw[(df_raw['price'] >= min_threshold) & (df_raw['price'] <= max_threshold)]
df_outliers_low_all = df_raw[df_raw['price'] < min_threshold]
df_outliers_high_all = df_raw[df_raw['price'] > max_threshold]
df_outliers_all = pd.concat([df_outliers_low_all, df_outliers_high_all])

# 3. Apply Sidebar Categorical Filters to the segments
df_raw_filtered = df_raw[
    df_raw['neighbourhood_group'].isin(selected_groups) & 
    df_raw['room_type'].isin(selected_rooms)
]
df_cleaned_filtered = df_cleaned_all[
    df_cleaned_all['neighbourhood_group'].isin(selected_groups) & 
    df_cleaned_all['room_type'].isin(selected_rooms)
]
df_outliers_filtered = df_outliers_all[
    df_outliers_all['neighbourhood_group'].isin(selected_groups) & 
    df_outliers_all['room_type'].isin(selected_rooms)
]
df_outliers_low_filtered = df_outliers_low_all[
    df_outliers_low_all['neighbourhood_group'].isin(selected_groups) & 
    df_outliers_low_all['room_type'].isin(selected_rooms)
]
df_outliers_high_filtered = df_outliers_high_all[
    df_outliers_high_all['neighbourhood_group'].isin(selected_groups) & 
    df_outliers_high_all['room_type'].isin(selected_rooms)
]

# Total counts for calculations
total_raw_cnt = len(df_raw_filtered)
total_clean_cnt = len(df_cleaned_filtered)
total_outlier_cnt = len(df_outliers_filtered)

# ----------------- TABS SETUP -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "📈 Price Distributions", 
    "🗺️ Geospatial Map", 
    "🔍 Data Inspector & Export"
])

# ----------------- TAB 1: EXECUTIVE SUMMARY -----------------
with tab1:
    st.markdown('<div class="section-header">📈 Core Metrics: Before vs. After Treatment</div>', unsafe_allow_html=True)
    
    # Custom Styled Metrics Layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Listings</div>
            <div class="metric-value-container">
                <span class="metric-value-raw">{total_raw_cnt:,}</span>
                <span class="metric-value-clean">{total_clean_cnt:,}</span>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #ef4444;">
                Removed: {total_outlier_cnt:,} ({ (total_outlier_cnt/max(1, total_raw_cnt))*100 :.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        raw_mean = df_raw_filtered['price'].mean() if total_raw_cnt > 0 else 0
        clean_mean = df_cleaned_filtered['price'].mean() if total_clean_cnt > 0 else 0
        mean_diff = ((clean_mean - raw_mean) / max(1, raw_mean)) * 100
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Price</div>
            <div class="metric-value-container">
                <span class="metric-value-raw">${raw_mean:.1f}</span>
                <span class="metric-value-clean">${clean_mean:.1f}</span>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: {'#10b981' if mean_diff < 0 else '#ef4444'};">
                Change: {mean_diff:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        raw_median = df_raw_filtered['price'].median() if total_raw_cnt > 0 else 0
        clean_median = df_cleaned_filtered['price'].median() if total_clean_cnt > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median Price</div>
            <div class="metric-value-container">
                <span class="metric-value-raw">${raw_median:.1f}</span>
                <span class="metric-value-clean">${clean_median:.1f}</span>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #64748b;">
                Robust indicator
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        raw_max = df_raw_filtered['price'].max() if total_raw_cnt > 0 else 0
        clean_max = df_cleaned_filtered['price'].max() if total_clean_cnt > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Price Limit</div>
            <div class="metric-value-container">
                <span class="metric-value-raw">${raw_max:,}</span>
                <span class="metric-value-clean">${clean_max:,}</span>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #10b981;">
                Outliers capped
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Info banner detailing current thresholds
    st.markdown(f"""
    <div class="info-box">
        💡 <b>Outlier Criteria Details:</b> Based on your percentile thresholds, prices below <b>${min_threshold:,.2f}</b> 
        (representing the bottom {low_pct}%) and above <b>${max_threshold:,.2f}</b> (representing the top {100-high_pct:.2f}%) 
        are isolated. 
        <br>• Lower bound outlier count: <b>{len(df_outliers_low_filtered)}</b> listings.
        <br>• Upper bound outlier count: <b>{len(df_outliers_high_filtered)}</b> listings.
    </div>
    """, unsafe_allow_html=True)

    # Side-by-side plots for statistics
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("##### Price Volatility (Standard Deviation)")
        raw_std = df_raw_filtered['price'].std() if total_raw_cnt > 1 else 0
        clean_std = df_cleaned_filtered['price'].std() if total_clean_cnt > 1 else 0
        
        fig_std = go.Figure()
        fig_std.add_trace(go.Bar(
            x=['Raw Dataset', 'Cleaned Dataset'],
            y=[raw_std, clean_std],
            marker_color=['#ef4444', '#10b981'],
            text=[f"${raw_std:.1f}", f"${clean_std:.1f}"],
            textposition='auto',
            width=0.4
        ))
        fig_std.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis_title="Standard Deviation ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_std, use_container_width=True)
        
    with right_col:
        st.markdown("##### Outlier Distribution by Room Type")
        if total_outlier_cnt > 0:
            room_outliers = df_outliers_filtered['room_type'].value_counts().reset_index()
            room_outliers.columns = ['Room Type', 'Outliers Count']
            fig_room = px.pie(
                room_outliers, 
                values='Outliers Count', 
                names='Room Type',
                color_discrete_sequence=px.colors.qualitative.Safe,
                hole=0.4
            )
            fig_room.update_layout(
                height=300, 
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_room, use_container_width=True)
        else:
            st.info("No outliers detected with current bounds.")

# ----------------- TAB 2: PRICE DISTRIBUTIONS -----------------
with tab2:
    st.markdown('<div class="section-header">📈 Histogram & Distribution Comparisons</div>', unsafe_allow_html=True)
    
    dist_toggle = st.radio(
        "Compare Dataset distributions:",
        options=["Cleaned Dataset (Outliers Pruned)", "Raw Dataset (Including Outliers)"],
        horizontal=True
    )
    
    if dist_toggle == "Cleaned Dataset (Outliers Pruned)":
        if total_clean_cnt > 0:
            fig_hist = px.histogram(
                df_cleaned_filtered, 
                x="price", 
                nbins=50,
                color="room_type",
                marginal="box",
                title=f"Price Distribution of Cleaned Listings (${min_threshold:.1f} to ${max_threshold:.1f})",
                labels={"price": "Price ($)", "room_type": "Room Type"},
                color_discrete_sequence=px.colors.qualitative.Prism,
                opacity=0.8
            )
            fig_hist.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Price ($)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Cleaned dataset is empty. Readjust percentile values in the sidebar.")
            
    else:
        log_scale = st.checkbox("Apply Log Scale to X-Axis (Recommended to view highly skewed raw prices)", value=True)
        
        if total_raw_cnt > 0:
            fig_hist_raw = px.histogram(
                df_raw_filtered, 
                x="price", 
                nbins=100,
                color="room_type",
                marginal="box",
                title="Skewed Price Distribution of Raw Listings (All data points)",
                labels={"price": "Price ($)", "room_type": "Room Type"},
                color_discrete_sequence=px.colors.qualitative.Safe,
                log_x=log_scale,
                opacity=0.8
            )
            fig_hist_raw.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Price ($) [Log Scaled]" if log_scale else "Price ($)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_hist_raw, use_container_width=True)
        else:
            st.warning("Dataset is empty. Readjust sidebar filters.")

    st.markdown('<div class="section-header">📦 Boxplot Analysis by Neighbourhood Group</div>', unsafe_allow_html=True)
    
    col_box1, col_box2 = st.columns(2)
    
    with col_box1:
        st.markdown("##### Cleaned Prices Box Plot")
        if total_clean_cnt > 0:
            fig_box_clean = px.box(
                df_cleaned_filtered,
                x="neighbourhood_group",
                y="price",
                color="neighbourhood_group",
                points=False, # hide points to keep visual premium
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"price": "Price ($)", "neighbourhood_group": "Borough"}
            )
            fig_box_clean.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_box_clean, use_container_width=True)
        else:
            st.info("No cleaned data available.")
            
    with col_box2:
        st.markdown("##### Raw Prices Box Plot (Compressed due to outliers)")
        if total_raw_cnt > 0:
            fig_box_raw = px.box(
                df_raw_filtered,
                x="neighbourhood_group",
                y="price",
                color="neighbourhood_group",
                points=False,
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"price": "Price ($)", "neighbourhood_group": "Borough"}
            )
            fig_box_raw.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_box_raw, use_container_width=True)
        else:
            st.info("No raw data available.")

# ----------------- TAB 3: GEOSPATIAL MAP -----------------
with tab3:
    st.markdown('<div class="section-header">🗺️ Interactive Listings Scatter Map</div>', unsafe_allow_html=True)
    
    map_dataset_choice = st.radio(
        "Display listings mapping:",
        options=["Cleaned Data only", "Outliers only", "Raw Data (Whole Dataset)"],
        horizontal=True
    )
    
    if map_dataset_choice == "Cleaned Data only":
        map_df = df_cleaned_filtered
        title_map = f"Map of Cleaned Airbnb Listings (Sampled to {sample_size:,} points)"
    elif map_dataset_choice == "Outliers only":
        map_df = df_outliers_filtered
        title_map = f"Map of Outliers only (Total Outliers: {len(map_df):,})"
    else:
        map_df = df_raw_filtered
        title_map = f"Map of All Raw Listings (Sampled to {sample_size:,} points)"
        
    if len(map_df) == 0:
        st.warning("No data points fit the current selections for the map.")
    else:
        # Sample points to keep it premium and responsive
        if len(map_df) > sample_size and map_dataset_choice != "Outliers only":
            map_df_sample = map_df.sample(sample_size, random_state=42)
        else:
            map_df_sample = map_df
            
        # Draw Map
        fig_map = px.scatter_mapbox(
            map_df_sample, 
            lat="latitude", 
            lon="longitude", 
            color="price",
            size="price" if map_dataset_choice != "Outliers only" else None,
            color_continuous_scale=px.colors.sequential.Viridis, 
            range_color=[map_df_sample['price'].min(), map_df_sample['price'].max()],
            hover_name="name",
            hover_data={
                "price": ":$.2f",
                "neighbourhood": True,
                "room_type": True,
                "minimum_nights": True
            },
            zoom=10, 
            height=600,
            mapbox_style=map_style,
            title=title_map
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ----------------- TAB 4: DATA INSPECTOR & EXPORT -----------------
with tab4:
    st.markdown('<div class="section-header">🔍 Explore Segmented Datasets</div>', unsafe_allow_html=True)
    
    insp_col1, insp_col2 = st.columns(2)
    with insp_col1:
        data_show = st.selectbox(
            "Select dataset segment to view:",
            options=["Cleaned Data", "Raw Data", "Outliers Only (Combined)", "Low Price Outliers", "High Price Outliers"]
        )
    with insp_col2:
        st.write("") # alignments spacing
        st.write("")
        # Determine CSV download properties
        if data_show == "Cleaned Data":
            export_df = df_cleaned_filtered
            file_name = "cleaned_nyc_airbnb_listings.csv"
        elif data_show == "Raw Data":
            export_df = df_raw_filtered
            file_name = "raw_nyc_airbnb_listings.csv"
        elif data_show == "Outliers Only (Combined)":
            export_df = df_outliers_filtered
            file_name = "nyc_airbnb_all_outliers.csv"
        elif data_show == "Low Price Outliers":
            export_df = df_outliers_low_filtered
            file_name = "nyc_airbnb_low_outliers.csv"
        else:
            export_df = df_outliers_high_filtered
            file_name = "nyc_airbnb_high_outliers.csv"
            
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {data_show} as CSV ({len(export_df):,} rows)",
            data=csv_data,
            file_name=file_name,
            mime='text/csv',
            use_container_width=True
        )
        
    st.markdown(f"##### Showing sample of {data_show} (Displaying up to 100 rows)")
    if len(export_df) > 0:
        st.dataframe(export_df.head(100), use_container_width=True)
    else:
        st.info("This dataset segment contains 0 listings for the current filters.")
