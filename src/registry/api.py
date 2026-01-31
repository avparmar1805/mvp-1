"""
FastAPI endpoints for Data Product Registry
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from .service import RegistryService
from .schemas import (
    DataProductCreate,
    DataProductResponse,
    DataProductListResponse,
    DataProductUpdate,
    VersionResponse,
    ExecutionCreate,
    ExecutionResponse,
    DataProductStatus
)

router = APIRouter(prefix="/api/v1/registry", tags=["registry"])


def get_db():
    """Dependency to get database session"""
    from scripts.setup_registry_db import get_session
    db = get_session()
    try:
        yield db
    finally:
        db.close()


def get_registry_service(db: Session = Depends(get_db)) -> RegistryService:
    """Dependency to get registry service"""
    return RegistryService(db)


@router.post("/data-products", response_model=DataProductResponse, status_code=201)
def create_data_product(
    data: DataProductCreate,
    service: RegistryService = Depends(get_registry_service)
):
    """Create a new data product"""
    try:
        # Check if name already exists
        existing = service.get_data_product_by_name(data.name, include_spec=False)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Data product with name '{data.name}' already exists"
            )
        
        dp = service.create_data_product(data, created_by=data.owner)
        
        # Attach specification
        latest_version = service.get_latest_version(dp.id)
        if latest_version:
            dp.specification = latest_version.specification
            dp.lineage = latest_version.lineage
        
        return dp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-products/{data_product_id}", response_model=DataProductResponse)
def get_data_product(
    data_product_id: str,
    service: RegistryService = Depends(get_registry_service)
):
    """Get data product by ID"""
    dp = service.get_data_product(data_product_id, include_spec=True)
    if not dp:
        raise HTTPException(status_code=404, detail="Data product not found")
    return dp


@router.get("/data-products", response_model=List[DataProductListResponse])
def list_data_products(
    status: Optional[DataProductStatus] = Query(None),
    owner: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: RegistryService = Depends(get_registry_service)
):
    """List data products with optional filters"""
    return service.list_data_products(
        status=status,
        owner=owner,
        tags=tags,
        limit=limit,
        offset=offset
    )


@router.put("/data-products/{data_product_id}", response_model=DataProductResponse)
def update_data_product(
    data_product_id: str,
    data: DataProductUpdate,
    service: RegistryService = Depends(get_registry_service)
):
    """Update data product"""
    dp = service.update_data_product(data_product_id, data, updated_by="api_user")
    if not dp:
        raise HTTPException(status_code=404, detail="Data product not found")
    
    # Attach specification
    latest_version = service.get_latest_version(dp.id)
    if latest_version:
        dp.specification = latest_version.specification
        dp.lineage = latest_version.lineage
    
    return dp


@router.delete("/data-products/{data_product_id}", status_code=204)
def delete_data_product(
    data_product_id: str,
    service: RegistryService = Depends(get_registry_service)
):
    """Delete (archive) data product"""
    success = service.delete_data_product(data_product_id)
    if not success:
        raise HTTPException(status_code=404, detail="Data product not found")
    return None


@router.get("/data-products/{data_product_id}/versions", response_model=List[VersionResponse])
def get_version_history(
    data_product_id: str,
    limit: int = Query(10, ge=1, le=100),
    service: RegistryService = Depends(get_registry_service)
):
    """Get version history for a data product"""
    # Check if data product exists
    dp = service.get_data_product(data_product_id, include_spec=False)
    if not dp:
        raise HTTPException(status_code=404, detail="Data product not found")
    
    return service.get_version_history(data_product_id, limit=limit)


@router.post("/data-products/{data_product_id}/executions", response_model=ExecutionResponse, status_code=201)
def create_execution(
    data_product_id: str,
    data: ExecutionCreate,
    service: RegistryService = Depends(get_registry_service)
):
    """Record execution history"""
    # Check if data product exists
    dp = service.get_data_product(data_product_id, include_spec=False)
    if not dp:
        raise HTTPException(status_code=404, detail="Data product not found")
    
    return service.create_execution(data_product_id, data)


@router.get("/data-products/{data_product_id}/executions", response_model=List[ExecutionResponse])
def get_execution_history(
    data_product_id: str,
    limit: int = Query(20, ge=1, le=100),
    service: RegistryService = Depends(get_registry_service)
):
    """Get execution history for a data product"""
    # Check if data product exists
    dp = service.get_data_product(data_product_id, include_spec=False)
    if not dp:
        raise HTTPException(status_code=404, detail="Data product not found")
    
    return service.get_execution_history(data_product_id, limit=limit)


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "data_product_registry"}
