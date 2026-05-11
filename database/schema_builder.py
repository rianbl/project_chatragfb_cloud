import os
import psycopg2
import pandas as pd

DB_CONFIG = {
    "host": "postgres", 
    "port": "5432",
    "dbname": "llm_data",
    "user": "admin",
    "password": "admin"
}

def load_data(file_path):
    """Load customer data from a local CSV file."""
    data = pd.read_csv(file_path)
    # Clean column names by replacing spaces with underscores
    print("Cleaning column names...")
    data.columns = data.columns.str.replace(' ', '_')

    return data

def generate_sql_table_create(meta):
    """Generate SQL CREATE TABLE statement from data"""
    columns = ', '.join([f"{col} VARCHAR" for col in meta.columns])
    sql_command = f"DROP TABLE IF EXISTS source; CREATE TABLE source ({columns});"
    return sql_command

def create_table(filepath):
    """Create table schema and save SQL command to file and database"""
    try:
        # Load data from the file
        data = load_data(filepath)

        # Print cleaned column names for reference
        print("Cleaned column names:", data.columns)

        # Generate the SQL schema based on the cleaned source
        sql_command = generate_sql_table_create(data)

        # Connect to PostgreSQL database
        connection = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["dbname"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = connection.cursor()

        # Execute the SQL command to create the table
        cursor.execute(sql_command)
        connection.commit()

        print(f"Table 'source' created in the database successfully.")

        # Save the SQL command to a file
        os.makedirs("data", exist_ok=True)
        sql_file_path = os.path.join(os.getcwd(), "data", "init.sql")

        try:
            with open(sql_file_path, "w") as sql_file:
                sql_file.write(sql_command)
            print(f"SQL command to create table saved to {sql_file_path}")
        except Exception as e:
            print(f"Error while saving the SQL file: {e}")

        # Save the source data as a CSV file for future data insertion
        data_csv_path = os.path.join(os.getcwd(), "data", "source.csv")
        try:
            data.to_csv(data_csv_path, index=False)
            print(f"Source data saved to {data_csv_path}")
        except Exception as e:
            print(f"Error while saving source CSV: {e}")

        # Close the database connection
        cursor.close()
        connection.close()

    except Exception as e:
        print(f"Error: {e}")

