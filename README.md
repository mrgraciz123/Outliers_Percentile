# NYC Airbnb Price Outliers Dashboard 🏙️

An interactive, premium Streamlit dashboard built to visualize, analyze, and treat price outliers in the 2019 New York City Airbnb dataset using the percentile (quantile) method.

## 📊 Overview

In data preprocessing, outliers can heavily distort statistical analyses and machine learning models. This project demonstrates how to identify and prune outliers using percentile thresholds (e.g., bottom 1.0% and top 99.9%), comparing the data distributions and key metrics before and after treatment.

### Key Features
* **Dynamic Sidebar Controls**: Adjust lower and upper percentile bounds on the fly. Filter by Neighborhood Groups (Boroughs) and Room Types.
* **Before vs. After Comparison**: Side-by-side KPI cards showing changes in listing counts, average prices, median prices, and price ranges.
* **Statistical Visualizations**:
  - Comparative histograms with marginal box plots showing the impact of removing extreme skewness.
  - Interactive box plots showing price distribution per borough.
  - Pie charts of outlier distributions by room types.
* **Geospatial Scatter Map**: An interactive map showing listing locations colored and sized by price, with options to view cleaned listings, outliers only, or all raw listings.
* **Data Exporter**: Filtered datasets (Cleaned or Outliers) can be inspected and downloaded directly as CSV files.

---

## 🚀 How to Run the App

### Prerequisites
Make sure you have Python installed, then install the required dependencies:
```bash
pip install streamlit pandas plotly numpy
```

### Running the App
1. Clone this repository:
   ```bash
   git clone https://github.com/mrgraciz123/Outliers_Percentile.git
   cd Outliers_Percentile
   ```
2. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to `http://localhost:8501`.

---

## 📁 Repository Structure
* `app.py`: The main Streamlit dashboard application source code.
* `1_outliers_percentile_exercise.ipynb`: The Jupyter Notebook containing the initial outlier detection exercise.
* `AB_NYC_2019.csv`: The raw New York City Airbnb dataset (2019).
