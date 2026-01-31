"""
Pydantic schemas for Data Product Registry API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DataProductStatus(str, Enum):
    """Data product lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class DataProductCreate(BaseModel):
    """Schema for creating a new data product"""
    name: str = Field(..., description="Unique name for the data product")
    description: Optional[str] = Field(None, description="Description of the data product")
    owner: str = Field(..., description="Owner/creator of the data product")
    specification: Dict[str, Any] = Field(..., description="Full data product specification")
    lineage: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data lineage graph")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DataProductUpdate(BaseModel):
    """Schema for updating an existing data product"""
    description: Optional[str] = None
    status: Optional[DataProductStatus] = None
    specification: Optional[Dict[str, Any]] = None
    lineage: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    change_description: Optional[str] = None


class VersionResponse(BaseModel):
    """Schema for version information"""
    id: int
    version: str
    created_at: datetime
    created_by: str
    change_description: Optional[str]
    
    class Config:
        from_attributes = True


class DataProductResponse(BaseModel):
    """Schema for data product response"""
    id: str
    name: str
    description: Optional[str]
    owner: str
    status: DataProductStatus
    current_version: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    specification: Optional[Dict[str, Any]] = None
    lineage: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class DataProductListResponse(BaseModel):
    """Schema for listing data products"""
    id: str
    name: str
    description: Optional[str]
    owner: str
    status: DataProductStatus
    current_version: str
    updated_at: datetime
    tags: List[str]
    
    class Config:
        from_attributes = True


class ExecutionCreate(BaseModel):
    """Schema for creating execution history"""
    version: str
    status: str
    duration_ms: Optional[int] = None
    rows_processed: Optional[int] = None
    rows_output: Optional[int] = None
    quality_check_results: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class ExecutionResponse(BaseModel):
    """Schema for execution history response"""
    id: int
    data_product_id: str
    version: str
    execution_date: datetime
    status: str
    duration_ms: Optional[int]
    rows_processed: Optional[int]
    rows_output: Optional[int]
    quality_check_results: Dict[str, Any]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True
