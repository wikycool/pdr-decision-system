import sqlalchemy
import pandas as pd
from typing import Optional, Dict, Any
import logging
from pathlib import Path

def load_parts_to_database(csv_file: str, db_url: Optional[str] = None) -> None:
    """
    Load parts data from CSV file into a database.
    
    Args:
        csv_file: Path to the CSV file containing parts data
        db_url: Database URL (if None, uses SQLite)
    """
    try:
        # Read CSV file
        if not Path(csv_file).exists():
            print(f"Error: CSV file {csv_file} not found")
            return
        
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} parts from {csv_file}")
        
        # Set default database URL if not provided
        if db_url is None:
            db_url = "sqlite:///parts.db"
        
        # Create engine
        engine = sqlalchemy.create_engine(db_url)
        
        # Load data to database
        table_name = 'parts'
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"Successfully loaded {len(df)} parts to database table '{table_name}'")
        
        # Verify data
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.fetchone()[0]
            print(f"Verified: {count} records in database")
            
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
    except Exception as e:
        print(f"Error loading parts to database: {e}")

def load_parts_to_sqlite(csv_file: str, db_path: str = "parts.db") -> None:
    """
    Load parts data from CSV file into SQLite database.
    
    Args:
        csv_file: Path to the CSV file containing parts data
        db_path: Path to SQLite database file
    """
    try:
        # Read CSV file
        if not Path(csv_file).exists():
            print(f"Error: CSV file {csv_file} not found")
            return
        
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} parts from {csv_file}")
        
        # Create SQLite database URL
        db_url = f"sqlite:///{db_path}"
        
        # Load to database
        load_parts_to_database(csv_file, db_url)
        
    except Exception as e:
        print(f"Error loading parts to SQLite: {e}")

def get_parts_from_database(db_url: str, table_name: str = "parts") -> pd.DataFrame:
    """
    Retrieve parts data from database.
    
    Args:
        db_url: Database URL
        table_name: Name of the table containing parts data
        
    Returns:
        DataFrame containing parts data
    """
    try:
        engine = sqlalchemy.create_engine(db_url)
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        
        print(f"Retrieved {len(df)} parts from database")
        return df
        
    except Exception as e:
        print(f"Error retrieving parts from database: {e}")
        return pd.DataFrame()

def create_parts_table(db_url: str, table_name: str = "parts") -> None:
    """
    Create parts table in database with proper schema.
    
    Args:
        db_url: Database URL
        table_name: Name of the table to create
    """
    try:
        engine = sqlalchemy.create_engine(db_url)
        
        # Define table schema
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            part_id TEXT UNIQUE NOT NULL,
            name TEXT,
            category TEXT,
            price REAL,
            lead_time INTEGER,
            supplier TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text(create_table_sql))
            conn.commit()
        
        print(f"Created table '{table_name}' in database")
        
    except Exception as e:
        print(f"Error creating parts table: {e}")

def main():
    """Example usage of parts loading functions."""
    # Example CSV file path
    csv_file = "data/parts_raw.csv"
    
    # Load to SQLite
    if Path(csv_file).exists():
        load_parts_to_sqlite(csv_file)
        
        # Retrieve and display sample data
        df = get_parts_from_database("sqlite:///parts.db")
        if not df.empty:
            print("\nSample parts data:")
            print(df.head())
    else:
        print(f"Sample CSV file {csv_file} not found")

if __name__ == "__main__":
    main()