"""
Unit tests for Data Product Registry
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.registry.models import Base, DataProduct, DataProductVersion, DataProductStatus
from src.registry.service import RegistryService
from src.registry.schemas import DataProductCreate, DataProductUpdate, ExecutionCreate


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def registry_service(db_session):
    """Create registry service instance"""
    return RegistryService(db_session)


@pytest.fixture
def sample_spec():
    """Sample data product specification"""
    return {
        "metadata": {
            "name": "daily_sales_analytics",
            "version": "1.0.0",
            "description": "Daily sales analytics by region and category"
        },
        "data_model": {
            "target_table": "gold.daily_sales_analytics",
            "grain": "Daily, by region and category",
            "schema": [
                {"name": "date", "type": "DATE", "nullable": False},
                {"name": "region", "type": "VARCHAR(50)", "nullable": False},
                {"name": "total_revenue", "type": "DECIMAL(18,2)", "nullable": False}
            ]
        },
        "transformations": {
            "language": "SQL",
            "code": "SELECT DATE(order_date) AS date..."
        }
    }


class TestRegistryService:
    """Test suite for RegistryService"""
    
    def test_create_data_product(self, registry_service, sample_spec):
        """Test creating a new data product"""
        data = DataProductCreate(
            name="daily_sales_analytics",
            description="Daily sales analytics",
            owner="test_user",
            specification=sample_spec,
            tags=["sales", "analytics"]
        )
        
        dp = registry_service.create_data_product(data, created_by="test_user")
        
        assert dp is not None
        assert dp.name == "daily_sales_analytics"
        assert dp.owner == "test_user"
        assert dp.current_version == "1.0.0"
        assert dp.status == DataProductStatus.DRAFT
        assert len(dp.tags) == 2
        assert dp.id.startswith("dp_")
    
    def test_get_data_product(self, registry_service, sample_spec):
        """Test retrieving a data product by ID"""
        # Create
        data = DataProductCreate(
            name="test_product",
            owner="test_user",
            specification=sample_spec
        )
        created = registry_service.create_data_product(data)
        
        # Retrieve
        retrieved = registry_service.get_data_product(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "test_product"
        assert retrieved.specification is not None
    
    def test_get_data_product_by_name(self, registry_service, sample_spec):
        """Test retrieving a data product by name"""
        data = DataProductCreate(
            name="unique_product",
            owner="test_user",
            specification=sample_spec
        )
        registry_service.create_data_product(data)
        
        retrieved = registry_service.get_data_product_by_name("unique_product")
        
        assert retrieved is not None
        assert retrieved.name == "unique_product"
    
    def test_list_data_products(self, registry_service, sample_spec):
        """Test listing data products"""
        # Create multiple products
        for i in range(3):
            data = DataProductCreate(
                name=f"product_{i}",
                owner="test_user",
                specification=sample_spec,
                tags=["test"]
            )
            registry_service.create_data_product(data)
        
        # List all
        products = registry_service.list_data_products()
        assert len(products) == 3
        
        # List with filter
        products = registry_service.list_data_products(owner="test_user")
        assert len(products) == 3
    
    def test_update_data_product(self, registry_service, sample_spec):
        """Test updating a data product"""
        # Create
        data = DataProductCreate(
            name="update_test",
            owner="test_user",
            specification=sample_spec
        )
        created = registry_service.create_data_product(data)
        
        # Update description
        update_data = DataProductUpdate(
            description="Updated description",
            status="active"  # Use string, Pydantic will convert
        )
        updated = registry_service.update_data_product(created.id, update_data)
        
        assert updated.description == "Updated description"
        assert updated.status.value == "active"  # Compare enum value
    
    def test_version_increment_on_spec_change(self, registry_service, sample_spec):
        """Test that version increments when specification changes"""
        # Create
        data = DataProductCreate(
            name="version_test",
            owner="test_user",
            specification=sample_spec
        )
        created = registry_service.create_data_product(data)
        assert created.current_version == "1.0.0"
        
        # Update specification
        new_spec = sample_spec.copy()
        new_spec["metadata"]["version"] = "1.1.0"
        
        update_data = DataProductUpdate(
            specification=new_spec,
            change_description="Added new column"
        )
        updated = registry_service.update_data_product(created.id, update_data)
        
        assert updated.current_version == "1.1.0"
    
    def test_get_version_history(self, registry_service, sample_spec):
        """Test retrieving version history"""
        # Create
        data = DataProductCreate(
            name="history_test",
            owner="test_user",
            specification=sample_spec
        )
        created = registry_service.create_data_product(data)
        
        # Update spec twice
        for i in range(2):
            update_data = DataProductUpdate(
                specification=sample_spec,
                change_description=f"Update {i+1}"
            )
            registry_service.update_data_product(created.id, update_data)
        
        # Get history
        history = registry_service.get_version_history(created.id)
        
        assert len(history) == 3  # Initial + 2 updates
        assert history[0].version == "1.2.0"  # Latest first
        assert history[1].version == "1.1.0"
        assert history[2].version == "1.0.0"
    
    def test_delete_data_product(self, registry_service, sample_spec):
        """Test deleting (archiving) a data product"""
        # Create
        data = DataProductCreate(
            name="delete_test",
            owner="test_user",
            specification=sample_spec
        )
        created = registry_service.create_data_product(data)
        
        # Delete
        success = registry_service.delete_data_product(created.id)
        assert success is True
        
        # Verify archived
        retrieved = registry_service.get_data_product(created.id, include_spec=False)
        assert retrieved.status == DataProductStatus.ARCHIVED
    
    def test_create_execution(self, registry_service, sample_spec):
        """Test recording execution history"""
        # Create data product
        data = DataProductCreate(
            name="execution_test",
            owner="test_user",
            specification=sample_spec
        )
        dp = registry_service.create_data_product(data)
        
        # Record execution
        exec_data = ExecutionCreate(
            version="1.0.0",
            status="success",
            duration_ms=1500,
            rows_processed=10000,
            rows_output=5000
        )
        execution = registry_service.create_execution(dp.id, exec_data)
        
        assert execution is not None
        assert execution.status == "success"
        assert execution.duration_ms == 1500
    
    def test_get_execution_history(self, registry_service, sample_spec):
        """Test retrieving execution history"""
        # Create data product
        data = DataProductCreate(
            name="exec_history_test",
            owner="test_user",
            specification=sample_spec
        )
        dp = registry_service.create_data_product(data)
        
        # Record multiple executions
        for i in range(3):
            exec_data = ExecutionCreate(
                version="1.0.0",
                status="success" if i % 2 == 0 else "failed",
                duration_ms=1000 + i * 100
            )
            registry_service.create_execution(dp.id, exec_data)
        
        # Get history
        history = registry_service.get_execution_history(dp.id)
        
        assert len(history) == 3
        assert history[0].duration_ms == 1200  # Latest first


class TestVersionIncrement:
    """Test version increment logic"""
    
    def test_patch_increment(self):
        """Test patch version increment"""
        result = RegistryService._increment_version("1.2.3")
        assert result == "1.2.4"
    
    def test_minor_increment(self):
        """Test minor version increment"""
        result = RegistryService._increment_version("1.2.3", minor=True)
        assert result == "1.3.0"
    
    def test_major_increment(self):
        """Test major version increment"""
        result = RegistryService._increment_version("1.2.3", major=True)
        assert result == "2.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
