import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="App For Data Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load and preprocess data
def load_data():
    st.title("App For Data Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["data"] = df
            st.success("File uploaded and loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return False
    return False

# Function to handle missing values
def handle_missing_values(df, method):
    if method == "Drop rows":
        return df.dropna()
    elif method == "Mean":
        return df.fillna(df.mean())
    elif method == "Median":
        return df.fillna(df.median())
    elif method == "Mode":
        return df.fillna(df.mode().iloc[0])
    return df

# Advanced EDA Functions
def show_advanced_eda(data):
    st.header("Advanced EDA")
    
    # Data Overview
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Column Information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Type': data.dtypes,
        'Non-Null Count': data.count(),
        'Null Count': data.isnull().sum(),
        'Null Percentage': (data.isnull().sum() / len(data) * 100).round(2),
        'Unique Values': data.nunique()
    })
    st.dataframe(col_info)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       title="Correlation Heatmap")
        st.plotly_chart(fig)
    
    # Distribution Analysis
    st.subheader("Distribution Analysis")
    selected_col = st.selectbox("Select column for distribution analysis", numeric_cols)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[selected_col], nbinsx=30))
    fig.update_layout(title=f"Distribution of {selected_col}")
    st.plotly_chart(fig)

# Advanced Visualization Functions
def show_advanced_visualization(data):
    st.header("Advanced Visualization")
    
    viz_type = st.selectbox("Select Visualization Type", 
                           ["Statistical Plots", "Interactive Plots", "Multi-dimensional Analysis"])
    
    if viz_type == "Statistical Plots":
        plot_type = st.selectbox("Select Plot Type", 
                                ["Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot"])
        cols = st.multiselect("Select columns", data.select_dtypes(include=['int64', 'float64']).columns)
        if cols:
            if plot_type == "Box Plot":
                fig = px.box(data, y=cols)
            elif plot_type == "Violin Plot":
                fig = px.violin(data, y=cols)
            elif plot_type in ["Strip Plot", "Swarm Plot"]:
                fig = px.strip(data, y=cols)
            st.plotly_chart(fig)
    
    elif viz_type == "Interactive Plots":
        plot_type = st.selectbox("Select Plot Type", 
                                ["Scatter 3D", "Bubble Chart", "Animated Scatter"])
        
        if plot_type == "Scatter 3D":
            cols = st.multiselect("Select 3 columns", 
                                data.select_dtypes(include=['int64', 'float64']).columns,
                                max_selections=3)
            if len(cols) == 3:
                fig = px.scatter_3d(data, x=cols[0], y=cols[1], z=cols[2])
                st.plotly_chart(fig)
        
        elif plot_type == "Bubble Chart":
            x_col = st.selectbox("Select X axis", data.columns)
            y_col = st.selectbox("Select Y axis", data.columns)
            size_col = st.selectbox("Select Size variable", data.select_dtypes(include=['int64', 'float64']).columns)
            fig = px.scatter(data, x=x_col, y=y_col, size=size_col)
            st.plotly_chart(fig)
            
        elif plot_type == "Animated Scatter":
            x_col = st.selectbox("Select X axis", data.columns)
            y_col = st.selectbox("Select Y axis", data.columns)
            animation_col = st.selectbox("Select Animation frame", data.columns)
            fig = px.scatter(data, x=x_col, y=y_col, animation_frame=animation_col)
            st.plotly_chart(fig)
    
    elif viz_type == "Multi-dimensional Analysis":
        if st.checkbox("Perform PCA"):
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 2:
                X = StandardScaler().fit_transform(data[numeric_cols])
                pca = PCA(n_components=2)
                components = pca.fit_transform(X)
                
                fig = px.scatter(components, x=0, y=1,
                               labels={'0': 'First Principal Component', 
                                     '1': 'Second Principal Component'},
                               title='PCA Results')
                st.plotly_chart(fig)
                
                # Explained variance ratio
                st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Advanced Modeling Functions
def show_advanced_modeling(data):
    st.header("Advanced Modeling")
    
    # Data Preprocessing
    st.subheader("Data Preprocessing")
    
    # Handle missing values
    missing_method = st.selectbox("Handle Missing Values", 
                                ["None", "Drop rows", "Mean", "Median", "Mode"])
    if missing_method != "None":
        data = handle_missing_values(data, missing_method)
    
    # Feature Selection
    target = st.selectbox("Select Target Variable", data.columns)
    features = st.multiselect("Select Feature Variables", 
                            [col for col in data.columns if col != target])
    
    if target and features:
        # Prepare data
        X = data[features]
        y = data[target]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
        
        # Scale features
        if st.checkbox("Scale Features"):
            X = StandardScaler().fit_transform(X)
        
        # Split data
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                           test_size=test_size, 
                                                           random_state=42)
        
        # Model Selection
        model_type = st.selectbox("Select Model", 
                                ["Linear Regression", "Ridge Regression", "Lasso Regression",
                                 "Decision Tree", "Random Forest", "Gradient Boosting",
                                 "SVR", "KNN"])
        
        # Model parameters
        with st.expander("Model Parameters"):
            if model_type == "Ridge Regression":
                alpha = st.slider("Alpha", 0.0, 10.0, 1.0)
                model = Ridge(alpha=alpha)
            elif model_type == "Lasso Regression":
                alpha = st.slider("Alpha", 0.0, 10.0, 1.0)
                model = Lasso(alpha=alpha)
            elif model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 500, 100)
                max_depth = st.slider("Max Depth", 1, 20, 5)
                model = RandomForestRegressor(n_estimators=n_estimators, 
                                           max_depth=max_depth)
            elif model_type == "Gradient Boosting":
                n_estimators = st.slider("Number of Trees", 10, 500, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                model = GradientBoostingRegressor(n_estimators=n_estimators,
                                                learning_rate=learning_rate)
            elif model_type == "KNN":
                n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
            else:
                model = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "SVR": SVR()
                }[model_type]
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Model Evaluation
                st.subheader("Model Evaluation")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", 
                             round(r2_score(y_test, predictions), 3))
                with col2:
                    st.metric("Mean Squared Error", 
                             round(mean_squared_error(y_test, predictions), 3))
                with col3:
                    st.metric("Mean Absolute Error", 
                             round(mean_absolute_error(y_test, predictions), 3))
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5)
                st.write(f"Cross-validation scores (5-fold): {cv_scores}")
                st.write(f"Average CV score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
                
                # Feature Importance (for applicable models)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                title='Feature Importance Plot')
                    st.plotly_chart(fig)

# Main App
def main():
    if "data" not in st.session_state or st.session_state["data"] is None:
        file_loaded = load_data()
    else:
        file_loaded = True

    if file_loaded and "data" in st.session_state:
        data = st.session_state["data"]
        
        # Tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["EDA", "Visualization", "Modeling"])
        
        with tab1:
            show_advanced_eda(data)
        
        with tab2:
            show_advanced_visualization(data)
        
        with tab3:
            show_advanced_modeling(data)
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
