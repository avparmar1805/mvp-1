"""
Registry Service - Business logic for Data Product Registry
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .models import DataProduct, DataProductVersion, ExecutionHistory, DataProductStatus
from .schemas import DataProductCreate, DataProductUpdate, ExecutionCreate


class RegistryService:
    """Service class for data product registry operations"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_data_product(
        self, 
        data: DataProductCreate,
        created_by: str = "system"
    ) -> DataProduct:
        """Create a new data product with initial version"""
        
        # Generate unique ID
        dp_id = f"dp_{uuid.uuid4().hex[:12]}"
        
        # Create data product
        data_product = DataProduct(
            id=dp_id,
            name=data.name,
            description=data.description,
            owner=data.owner,
            status=DataProductStatus.DRAFT,
            current_version="1.0.0",
            tags=data.tags or [],
            extra_metadata=data.metadata or {}
        )
        
        # Create initial version
        version = DataProductVersion(
            data_product_id=dp_id,
            version="1.0.0",
            specification=data.specification,
            lineage=data.lineage or {},
            created_by=created_by,
            change_description="Initial version"
        )
        
        self.db.add(data_product)
        self.db.add(version)
        self.db.commit()
        self.db.refresh(data_product)
        
        return data_product
    
    def get_data_product(
        self, 
        data_product_id: str,
        include_spec: bool = True
    ) -> Optional[DataProduct]:
        """Get data product by ID"""
        dp = self.db.query(DataProduct).filter(
            DataProduct.id == data_product_id
        ).first()
        
        if dp and include_spec:
            # Attach latest specification
            latest_version = self.get_latest_version(data_product_id)
            if latest_version:
                dp.specification = latest_version.specification
                dp.lineage = latest_version.lineage
        
        return dp
    
    def get_data_product_by_name(
        self, 
        name: str,
        include_spec: bool = True
    ) -> Optional[DataProduct]:
        """Get data product by name"""
        dp = self.db.query(DataProduct).filter(
            DataProduct.name == name
        ).first()
        
        if dp and include_spec:
            latest_version = self.get_latest_version(dp.id)
            if latest_version:
                dp.specification = latest_version.specification
                dp.lineage = latest_version.lineage
        
        return dp
    
    def list_data_products(
        self,
        status: Optional[DataProductStatus] = None,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DataProduct]:
        """List data products with optional filters"""
        query = self.db.query(DataProduct)
        
        if status:
            query = query.filter(DataProduct.status == status)
        
        if owner:
            query = query.filter(DataProduct.owner == owner)
        
        if tags:
            # Filter by tags (PostgreSQL JSON contains)
            for tag in tags:
                query = query.filter(DataProduct.tags.contains([tag]))
        
        query = query.order_by(desc(DataProduct.updated_at))
        query = query.limit(limit).offset(offset)
        
        return query.all()
    
    def update_data_product(
        self,
        data_product_id: str,
        data: DataProductUpdate,
        updated_by: str = "system"
    ) -> Optional[DataProduct]:
        """Update data product and create new version if spec changed"""
        dp = self.db.query(DataProduct).filter(
            DataProduct.id == data_product_id
        ).first()
        
        if not dp:
            return None
        
        # Update basic fields
        if data.description is not None:
            dp.description = data.description
        if data.status is not None:
            dp.status = data.status
        if data.tags is not None:
            dp.tags = data.tags
        if data.metadata is not None:
            dp.extra_metadata = data.metadata
        
        # If specification changed, create new version
        if data.specification is not None:
            new_version = self._increment_version(dp.current_version, minor=True)
            
            version = DataProductVersion(
                data_product_id=data_product_id,
                version=new_version,
                specification=data.specification,
                lineage=data.lineage or {},
                created_by=updated_by,
                change_description=data.change_description or "Updated specification"
            )
            
            dp.current_version = new_version
            self.db.add(version)
        
        dp.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(dp)
        
        return dp
    
    def delete_data_product(self, data_product_id: str) -> bool:
        """Delete data product (soft delete by archiving)"""
        dp = self.db.query(DataProduct).filter(
            DataProduct.id == data_product_id
        ).first()
        
        if not dp:
            return False
        
        dp.status = DataProductStatus.ARCHIVED
        dp.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True
    
    def get_latest_version(self, data_product_id: str) -> Optional[DataProductVersion]:
        """Get latest version for a data product"""
        return self.db.query(DataProductVersion).filter(
            DataProductVersion.data_product_id == data_product_id
        ).order_by(desc(DataProductVersion.created_at)).first()
    
    def get_version_history(
        self, 
        data_product_id: str,
        limit: int = 10
    ) -> List[DataProductVersion]:
        """Get version history for a data product"""
        return self.db.query(DataProductVersion).filter(
            DataProductVersion.data_product_id == data_product_id
        ).order_by(desc(DataProductVersion.created_at)).limit(limit).all()
    
    def create_execution(
        self,
        data_product_id: str,
        data: ExecutionCreate
    ) -> ExecutionHistory:
        """Record execution history"""
        execution = ExecutionHistory(
            data_product_id=data_product_id,
            version=data.version,
            status=data.status,
            duration_ms=data.duration_ms,
            rows_processed=data.rows_processed,
            rows_output=data.rows_output,
            quality_check_results=data.quality_check_results or {},
            error_message=data.error_message,
            error_traceback=data.error_traceback
        )
        
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        
        return execution
    
    def get_execution_history(
        self,
        data_product_id: str,
        limit: int = 20
    ) -> List[ExecutionHistory]:
        """Get execution history for a data product"""
        return self.db.query(ExecutionHistory).filter(
            ExecutionHistory.data_product_id == data_product_id
        ).order_by(desc(ExecutionHistory.execution_date)).limit(limit).all()
    
    @staticmethod
    def _increment_version(current: str, major: bool = False, minor: bool = False) -> str:
        """Increment semantic version"""
        parts = current.split('.')
        major_v, minor_v, patch_v = int(parts[0]), int(parts[1]), int(parts[2])
        
        if major:
            return f"{major_v + 1}.0.0"
        elif minor:
            return f"{major_v}.{minor_v + 1}.0"
        else:
            return f"{major_v}.{minor_v}.{patch_v + 1}"
