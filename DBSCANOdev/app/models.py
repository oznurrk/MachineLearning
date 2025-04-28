import pandas as pd
from app.db import engine

#problem2
def fetch_product_data():
    query = """
        SELECT 
            p.product_id,
            AVG(od.unit_price) as avg_price,
            COUNT(od.order_id) as sales_frequency,
            AVG(od.quantity) as avg_quantity_per_order,
            COUNT(DISTINCT o.customer_id) as unique_customers
        FROM products p
        JOIN order_details od ON p.product_id = od.product_id
        JOIN orders o ON o.order_id = od.order_id
        GROUP BY p.product_id
    """
    df = pd.read_sql_query(query, engine)
    return df

#problem 3
def fetch_supplier_data():
    query = """
        SELECT 
            s.supplier_id,
            COUNT(DISTINCT p.product_id) as num_products,
            SUM(od.quantity) as total_sales_quantity,
            AVG(od.unit_price) as avg_unit_price,
            COUNT(DISTINCT o.customer_id) as unique_customers
        FROM suppliers s
        JOIN products p ON s.supplier_id = p.supplier_id
        JOIN order_details od ON p.product_id = od.product_id
        JOIN orders o ON od.order_id = o.order_id
        GROUP BY s.supplier_id
    """
    df = pd.read_sql_query(query, engine)
    return df

#problem 4
def fetch_countries_data():
    query = """
    SELECT c.country,
           COUNT(DISTINCT o.order_id) AS total_orders,
           AVG(od.unit_price * od.quantity) AS avg_order_amount,
           AVG(od.quantity) AS avg_items_per_order
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    GROUP BY c.country
    """
    df = pd.read_sql_query(query, engine)
    return df