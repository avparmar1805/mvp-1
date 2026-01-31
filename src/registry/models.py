"""
SQLAlchemy models for Data Product Registry
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey, Enum, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class DataProductStatus(enum.Enum):
    """Data product lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class DataProduct(Base):
    """Main data product registry table"""
    __tablename__ = 'data_products'
    
    id = Column(String(255), primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    owner = Column(String(255), nullable=False)
    status = Column(Enum(DataProductStatus), default=DataProductStatus.DRAFT, nullable=False)
    
    # Current version info
    current_version = Column(String(50), nullable=False, default="1.0.0")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Metadata
    tags = Column(JSON, default=list)
    extra_metadata = Column(JSON, default=dict)
    
    # Relationships
    versions = relationship("DataProductVersion", back_populates="data_product", cascade="all, delete-orphan")
    executions = relationship("ExecutionHistory", back_populates="data_product", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DataProduct(id={self.id}, name={self.name}, version={self.current_version})>"


class DataProductVersion(Base):
    """Version history for data products"""
    __tablename__ = 'data_product_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    data_product_id = Column(String(255), ForeignKey('data_products.id'), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Full specification (YAML/JSON)
    specification = Column(JSON, nullable=False)
    
    # Lineage graph
    lineage = Column(JSON, default=dict)
    
    # Quality metrics
    quality_metrics = Column(JSON, default=dict)
    
    # Change description
    change_description = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    data_product = relationship("DataProduct", back_populates="versions")
    
    def __repr__(self):
        return f"<DataProductVersion(id={self.id}, version={self.version})>"


class ExecutionHistory(Base):
    """Execution history for data products"""
    __tablename__ = 'execution_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    data_product_id = Column(String(255), ForeignKey('data_products.id'), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Execution details
    execution_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(50), nullable=False)  # success, failed, running
    duration_ms = Column(Integer)
    
    # Metrics
    rows_processed = Column(Integer)
    rows_output = Column(Integer)
    quality_check_results = Column(JSON, default=dict)
    
    # Error info
    error_message = Column(Text)
    error_traceback = Column(Text)
    
    # Relationships
    data_product = relationship("DataProduct", back_populates="executions")
    
    def __repr__(self):
        return f"<ExecutionHistory(id={self.id}, status={self.status})>"
