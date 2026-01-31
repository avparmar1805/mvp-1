"""
Database setup script for Data Product Registry

Creates PostgreSQL database and tables.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.registry.models import Base
from dotenv import load_dotenv

load_dotenv()


def get_database_url():
    """Get database URL from environment or use default"""
    return os.getenv(
        'REGISTRY_DATABASE_URL',
        'postgresql://postgres:test@localhost:5432/data_product_registry'
    )


def create_database():
    """Create database and tables"""
    database_url = get_database_url()
    
    print(f"Connecting to database: {database_url}")
    
    try:
        # Create engine
        engine = create_engine(database_url, echo=True)
        
        # Create all tables
        print("\nCreating tables...")
        Base.metadata.create_all(engine)
        
        print("\n✅ Database setup completed successfully!")
        print(f"   - data_products table created")
        print(f"   - data_product_versions table created")
        print(f"   - execution_history table created")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating database: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker run -d --name registry-db \\")
        print("    -p 5432:5432 \\")
        print("    -e POSTGRES_DB=data_product_registry \\")
        print("    -e POSTGRES_PASSWORD=test \\")
        print("    postgres:14")
        return False


def get_session():
    """Get database session"""
    database_url = get_database_url()
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


if __name__ == "__main__":
    success = create_database()
    sys.exit(0 if success else 1)
