import pandas as pd
import os
import psycopg2
from psycopg2 import sql

DB_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "postgres"),
    "port": os.getenv("DATABASE_PORT", "5432"),
    "dbname": os.getenv("DATABASE_NAME", "llm_data"),
    "user": os.getenv("DATABASE_USER", "admin"),
    "password": os.getenv("DATABASE_PASSWORD", os.getenv("POSTGRES_PASSWORD", "admin"))
}

def populate_table():
    """Populate the table with data from the saved metadata CSV."""
    # Define the path for the source CSV
    data_dir = os.path.join("/", "data")
    source_csv_path = os.path.join(data_dir, "source.csv")

    # Check if the CSV file exists
    if not os.path.exists(source_csv_path):
        print(f"Error: {source_csv_path} does not exist.")
        return

    print("Loading metadata CSV into pandas DataFrame...")
    source = pd.read_csv(source_csv_path)
    print(f"Loaded metadata from {source_csv_path}")

    # Convert all column names to lowercase
    headers = [header.lower() for header in source.columns.tolist()]

    # Establish connection to the database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("Inserting data into the PostgreSQL table...")
        # Iterate over rows in the DataFrame
        for _, row in source.iterrows():
            values = tuple(row.tolist())
            insert_query = sql.SQL("INSERT INTO source ({}) VALUES ({})").format(
                sql.SQL(", ").join(map(sql.Identifier, headers)),
                sql.SQL(", ").join(sql.Placeholder() for _ in headers)
            )
            cursor.execute(insert_query, values)

        conn.commit()
        print("Data inserted successfully into the source table!")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
