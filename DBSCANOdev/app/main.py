from fastapi import FastAPI
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
from pydantic import BaseModel
from app.clustering import find_optimal_eps_varying_min_samples
from app.services import cluster_products, cluster_suppliers, cluster_countries
from sklearn.datasets import make_blobs

app = FastAPI()

class EpsRequest(BaseModel):
    min_samples_list: list = [3, 4, 5, 6]

@app.post("/find-optimal-eps")
def find_eps(request: EpsRequest):

    X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.6, random_state=42)
    
    X_scaled = X

    results = find_optimal_eps_varying_min_samples(X_scaled, request.min_samples_list)
    
    return JSONResponse(content={"results": results})

@app.get("/products/clusters")
def get_clusters(min_samples: int = 3):
    df = cluster_products(min_samples)
    clusters = df.to_dict(orient="records")
    return JSONResponse(content=clusters)

@app.get("/products/outliers")
def get_clusters(min_samples: int = 3):
    df = cluster_products(min_samples)
    outliers = df.to_dict(orient="records")
    return JSONResponse(content=outliers)

@app.get("/products/plot")
def plot_clusters(min_samples: int = 3):
    df = cluster_products(min_samples)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sales_frequency'], df['unique_customers'], c=df['cluster'], cmap='plasma', s=60)
    plt.xlabel("Satış Sıklığı")
    plt.ylabel("Farklı Müşteri Sayısı")
    plt.title("Ürün Kümeleri (DBSCAN)")
    plt.grid(True)
    plt.savefig("products_cluster_plot.png")
    plt.colorbar(label='Küme No')
    plt.show()

    return JSONResponse(content={"message": "Plot saved as cluster_plot.png"})

#Problem 3
@app.get("/suppliers/clusters")
def get_supplier_clusters(min_samples: int = 3):
    df = cluster_suppliers(min_samples)
    clusters = df.to_dict(orient="records")
    return JSONResponse(content=clusters)

@app.get("/suppliers/outliers")
def get_clusters(min_samples: int = 3):
    df = cluster_suppliers(min_samples)
    outliers = df.to_dict(orient="records")
    return JSONResponse(content=outliers)

@app.get("/suppliers/plot")
def plot_supplier_clusters(min_samples: int = 3):
    df = cluster_suppliers(min_samples)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['num_products'], df['total_sales_quantity'], c=df['cluster'], cmap='plasma', s=60)
    plt.xlabel("Ürün Sayısı")
    plt.ylabel("Toplam Satış Miktarı")
    plt.title("Tedarikçi Kümeleri (DBSCAN)")
    plt.grid(True)
    plt.savefig("supplier_cluster_plot.png")
    plt.colorbar(label='Küme No')
    plt.show()

    return JSONResponse(content={"message": "Plot saved as supplier_cluster_plot.png"})

@app.post("/countries/clusters")
def get_supplier_clusters(min_samples: int = 3):
    df = cluster_countries(min_samples)
    clusters = df.to_dict(orient="records")
    return JSONResponse(content=clusters)

@app.get("/countries/outliers")
def get_clusters(min_samples: int = 3):
    df = cluster_countries(min_samples)
    outliers = df.to_dict(orient="records")
    return JSONResponse(content=outliers)

@app.get("/countries/plot")
def plot(min_samples : int = 3):
    df = cluster_countries(min_samples)

    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_orders'], df['avg_order_amount'], c=df['cluster'], cmap='plasma', s=60)
    plt.xlabel("Toplam satış")
    plt.ylabel("Ortalama Satış Miktarı")
    plt.title("Ürün Segmentasyonu (DBSCAN)")
    plt.grid(True)
    plt.savefig("countries_cluster_plot.png")
    plt.colorbar(label='Küme No')
    plt.show()

    return JSONResponse(content="success")
