# model.py

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer

# --- 1. Define Helper Functions ---

def load_data(path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    return df

def create_pipeline(num_cols: list, cat_cols: list, n_components: int, n_clusters: int) -> Pipeline:
    """
    Creates the full preprocessing and clustering pipeline
    using a ColumnTransformer.
    """
    print("Creating pipeline...")
    
    # Create a transformer for numerical features
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Create a transformer for categorical features
    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create the 'preprocessor' by combining the two transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='passthrough' # Leaves other columns (if any) alone
    )
    
    # Now, chain the preprocessor with your other steps
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components, random_state=45)),
        ('clusterer', KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42
        ))
    ])
    
    return model_pipeline

def plot_clusters(pipeline: Pipeline, data: pd.DataFrame):
    """
        Plots the first two principal components (PC1 vs PC2) and colors
        them by their assigned cluster. Also plots the cluster centers.
    
        Args:
            pipeline: The fitted scikit-learn pipeline object.
            data: The original, raw DataFrame used to fit the pipeline.
    """
    print("Generating cluster plot...")
    
    # --- 1. Get data from the pipeline ---
    # Access the different steps of your pipeline by name
    preprocessor = pipeline.named_steps['preprocessor']
    pca_transformer = pipeline.named_steps['pca']
    kmeans = pipeline.named_steps['clusterer']
    
    # Get the data in PCA space
    # First, preprocess the raw data
    preprocessed_data = preprocessor.transform(data)
    # Second, transform preprocessed data into PCA space
    pca_data = pca_transformer.transform(preprocessed_data)
    
    # Get the cluster labels and centers
    # The labels_ are for the data points
    labels = kmeans.labels_
    # The cluster_centers_ are already in PCA space
    centers_pca = kmeans.cluster_centers_
    
    # --- 2. Create a DataFrame for plotting ---
    df_plot = pd.DataFrame()
    df_plot['PC1'] = pca_data[:, 0]
    df_plot['PC2'] = pca_data[:, 1]
    df_plot['cluster'] = labels
    
    # --- 3. Plot the data ---
    plt.figure(figsize=(10, 7))
    
    # Plot the clustered data points
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='cluster',
        data=df_plot,
        palette='viridis',
        s=100,
        alpha=0.7,
        legend='full'
    )
    
    # Plot the cluster centers
    plt.scatter(
        x=centers_pca[:, 0],
        y=centers_pca[:, 1],
        marker='X',           # Use a distinct marker
        c='red',              # Use a distinct color
        s=250,                # Make them larger
        label='Centroids'     # Add a label for the legend
    )
    
    plt.title('Customer Segments (PC1 vs PC2)')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig('models/cluster_plot.png')
    print(f"Cluster plot saved to: models/cluster_plot.png")
    
    # Optionally, display the plot
    plt.show()
        

# --- 2. Main Execution Function ---

def main():
    """
    Main function to run the clustering pipeline.
    """
    print("Running clustering-with-dimensionality-reduction...")
    
    # --- Configuration ---
    OPTIMAL_N_COMPONENTS = 4
    OPTIMAL_N_CLUSTERS = 4
    
    # Define your feature columns
    NUM_FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    CAT_FEATURES = ['Gender']
    
    # --- File Paths ---
    INPUT_DATA_PATH = "data/raw/Mall_Customers.csv"
    OUTPUT_DATA_PATH = "data/clustered_results.csv"
    MODEL_SAVE_PATH = "models/clustering_pipeline.joblib"
    
    
    # 1. Load Data
    data = load_data(INPUT_DATA_PATH)
    
    # 2. Create Pipeline
    pipeline = create_pipeline(
        num_cols=NUM_FEATURES,
        cat_cols=CAT_FEATURES,
        n_components=OPTIMAL_N_COMPONENTS,
        n_clusters=OPTIMAL_N_CLUSTERS
    )
    
    # 3. Fit pipeline and get labels
    print("Fitting pipeline and predicting clusters...")
    cluster_labels = pipeline.fit_predict(data)
    
    # 4. Save results
    print("Saving results...")
   
    # Create the output directories if they don't exist
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = cluster_labels
    data_with_clusters.to_csv(OUTPUT_DATA_PATH, index=False)
    
    # 5. Save the trained pipeline
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    
    print(f"\n✅ Success!")
    print(f"Clustered data saved to: {OUTPUT_DATA_PATH}")
    print(f"Trained pipeline saved to: {MODEL_SAVE_PATH}")

    # 6. PLOT THE RESULTS
    plot_clusters(pipeline, data)


# --- 3. Entry Point ---
# This is the only if __name__ == "__main__" block.
# It should be at the very end of the file.
if __name__ == "__main__":
    main()