from app.models import fetch_product_data, fetch_supplier_data, fetch_countries_data
from app.clustering import perform_dbscan


product_features = ['avg_price', 'sales_frequency', 'avg_quantity_per_order', 'unique_customers']
suppliers_features = ['num_products','total_sales_quantity', 'avg_unit_price', 'unique_customers']
countries_features = ["total_orders", "avg_order_amount", "avg_items_per_order"]

def cluster_products(min_samples=3):
    df = fetch_product_data()
    clustered_df = perform_dbscan(df, product_features, min_samples)
    return clustered_df

def outliers_products(min_samples=3):
    df = fetch_product_data()
    clustered_df = perform_dbscan(df, product_features, min_samples)
    outliers = clustered_df[clustered_df["cluster"] == -1]
    return outliers

def cluster_suppliers(min_samples=3):
    df = fetch_supplier_data()
    clustered_df = perform_dbscan(df,suppliers_features, min_samples)
    return clustered_df


def outliers_suppliers(min_samples=3):
    df = fetch_supplier_data()
    clustered_df = perform_dbscan(df, suppliers_features, min_samples)
    outliers = clustered_df[clustered_df["cluster"] == -1]
    return outliers

def cluster_countries(min_samples=3):
    df = fetch_countries_data()
    clustered_df = perform_dbscan(df,countries_features, min_samples)
    return clustered_df


def outliers_countries(min_samples=3):
    df = fetch_countries_data()
    clustered_df = perform_dbscan(df, countries_features, min_samples)
    outliers = clustered_df[clustered_df["cluster"] == -1]
    return outliers
