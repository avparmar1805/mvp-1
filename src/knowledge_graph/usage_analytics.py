"""
Usage Analytics Service

Tracks usage patterns for Knowledge Graph learning:
- Dataset selections
- Query patterns
- User feedback
- Co-occurrence data
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    """Represents a single usage event"""
    event_id: str
    timestamp: str
    event_type: str  # "dataset_selection", "query", "feedback"
    user_id: str
    data_product_id: Optional[str]
    datasets: List[str]
    query_text: Optional[str]
    extracted_terms: List[str]
    feedback_score: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UsageAnalyticsService:
    """
    Service for tracking and analyzing usage patterns
    """
    
    def __init__(self, db_path: str = "data/usage_analytics.db"):
        """
        Initialize usage analytics service
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Usage Analytics Service initialized with DB: {db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Usage events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                data_product_id TEXT,
                datasets TEXT,  -- JSON array
                query_text TEXT,
                extracted_terms TEXT,  -- JSON array
                feedback_score REAL
            )
        """)
        
        # Dataset co-occurrence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_cooccurrence (
                dataset_a TEXT NOT NULL,
                dataset_b TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                score REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (dataset_a, dataset_b)
            )
        """)
        
        # Discovered business terms table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_terms (
                term TEXT PRIMARY KEY,
                discovery_date TEXT NOT NULL,
                usage_count INTEGER DEFAULT 1,
                related_datasets TEXT,  -- JSON array
                confidence REAL DEFAULT 0.5,
                auto_discovered INTEGER DEFAULT 1
            )
        """)
        
        # Dataset relevance scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_relevance (
                dataset_name TEXT PRIMARY KEY,
                relevance_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_used TEXT,
                avg_feedback_score REAL DEFAULT 0.0,
                feedback_count INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON usage_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON usage_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cooccur_count ON dataset_cooccurrence(count DESC)")
        
        conn.commit()
        conn.close()
        logger.info("Database schema initialized")
    
    def record_event(self, event: Dict[str, Any]) -> str:
        """
        Record a usage event
        
        Args:
            event: Event data dictionary
            
        Returns:
            Event ID
        """
        import uuid
        
        event_id = event.get("event_id", str(uuid.uuid4()))
        timestamp = event.get("timestamp", datetime.now().isoformat())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO usage_events (
                event_id, timestamp, event_type, user_id, data_product_id,
                datasets, query_text, extracted_terms, feedback_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id,
            timestamp,
            event.get("event_type", "unknown"),
            event.get("user_id", "anonymous"),
            event.get("data_product_id"),
            json.dumps(event.get("datasets", [])),
            event.get("query_text"),
            json.dumps(event.get("extracted_terms", [])),
            event.get("feedback_score")
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded event: {event_id} ({event.get('event_type')})")
        return event_id
    
    def record_dataset_selection(
        self,
        datasets: List[str],
        query: str,
        data_product_id: Optional[str] = None,
        user_id: str = "anonymous"
    ) -> str:
        """
        Record dataset selection event
        
        Args:
            datasets: List of selected datasets
            query: User query
            data_product_id: Optional data product ID
            user_id: User identifier
            
        Returns:
            Event ID
        """
        event = {
            "event_type": "dataset_selection",
            "user_id": user_id,
            "data_product_id": data_product_id,
            "datasets": datasets,
            "query_text": query,
            "extracted_terms": [],
            "timestamp": datetime.now().isoformat()
        }
        
        event_id = self.record_event(event)
        
        # Update co-occurrence matrix
        self._update_cooccurrence(datasets)
        
        # Update dataset usage counts
        self._update_dataset_usage(datasets)
        
        return event_id
    
    def record_feedback(
        self,
        dataset: str,
        score: float,
        query: Optional[str] = None,
        user_id: str = "anonymous"
    ) -> str:
        """
        Record user feedback on dataset relevance
        
        Args:
            dataset: Dataset name
            score: Feedback score (0-5)
            query: Optional query context
            user_id: User identifier
            
        Returns:
            Event ID
        """
        event = {
            "event_type": "feedback",
            "user_id": user_id,
            "datasets": [dataset],
            "query_text": query,
            "feedback_score": score,
            "timestamp": datetime.now().isoformat()
        }
        
        event_id = self.record_event(event)
        
        # Update relevance score
        self._update_relevance_score(dataset, score)
        
        return event_id
    
    def _update_cooccurrence(self, datasets: List[str]):
        """Update co-occurrence counts for dataset pairs"""
        if len(datasets) < 2:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate all pairs
        for i, ds_a in enumerate(datasets):
            for ds_b in datasets[i+1:]:
                # Ensure consistent ordering
                if ds_a > ds_b:
                    ds_a, ds_b = ds_b, ds_a
                
                # Upsert co-occurrence
                cursor.execute("""
                    INSERT INTO dataset_cooccurrence (dataset_a, dataset_b, count, last_updated)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(dataset_a, dataset_b) DO UPDATE SET
                        count = count + 1,
                        last_updated = ?
                """, (ds_a, ds_b, datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _update_dataset_usage(self, datasets: List[str]):
        """Update dataset usage counts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for dataset in datasets:
            cursor.execute("""
                INSERT INTO dataset_relevance (dataset_name, usage_count, last_used)
                VALUES (?, 1, ?)
                ON CONFLICT(dataset_name) DO UPDATE SET
                    usage_count = usage_count + 1,
                    last_used = ?
            """, (dataset, datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _update_relevance_score(self, dataset: str, feedback_score: float):
        """Update dataset relevance score based on feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current scores
        cursor.execute("""
            SELECT avg_feedback_score, feedback_count
            FROM dataset_relevance
            WHERE dataset_name = ?
        """, (dataset,))
        
        row = cursor.fetchone()
        
        if row:
            current_avg, count = row
            # Calculate new average
            new_avg = (current_avg * count + feedback_score) / (count + 1)
            new_count = count + 1
        else:
            new_avg = feedback_score
            new_count = 1
        
        # Update
        cursor.execute("""
            INSERT INTO dataset_relevance (dataset_name, avg_feedback_score, feedback_count)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset_name) DO UPDATE SET
                avg_feedback_score = ?,
                feedback_count = ?
        """, (dataset, new_avg, new_count, new_avg, new_count))
        
        # Update relevance score (weighted combination of usage and feedback)
        cursor.execute("""
            UPDATE dataset_relevance
            SET relevance_score = 
                0.7 * MIN(1.0, usage_count / 100.0) + 
                0.3 * (avg_feedback_score / 5.0)
            WHERE dataset_name = ?
        """, (dataset,))
        
        conn.commit()
        conn.close()
    
    def get_cooccurrence_score(self, dataset_a: str, dataset_b: str) -> float:
        """
        Get co-occurrence score between two datasets
        
        Args:
            dataset_a: First dataset
            dataset_b: Second dataset
            
        Returns:
            Co-occurrence score (0-1)
        """
        # Ensure consistent ordering
        if dataset_a > dataset_b:
            dataset_a, dataset_b = dataset_b, dataset_a
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT count FROM dataset_cooccurrence
            WHERE dataset_a = ? AND dataset_b = ?
        """, (dataset_a, dataset_b))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return 0.0
        
        count = row[0]
        
        # Simple normalization (can be improved)
        return min(1.0, count / 10.0)
    
    def get_related_datasets(self, dataset: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get datasets frequently used with the given dataset
        
        Args:
            dataset: Dataset name
            top_k: Number of related datasets to return
            
        Returns:
            List of (dataset_name, score) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN dataset_a = ? THEN dataset_b
                    ELSE dataset_a
                END as related_dataset,
                count,
                score
            FROM dataset_cooccurrence
            WHERE dataset_a = ? OR dataset_b = ?
            ORDER BY count DESC
            LIMIT ?
        """, (dataset, dataset, dataset, top_k))
        
        results = cursor.fetchall()
        conn.close()
        
        # Calculate scores
        return [(ds, min(1.0, count / 10.0)) for ds, count, _ in results]
    
    def get_relevance_score(self, dataset: str) -> float:
        """
        Get current relevance score for a dataset
        
        Args:
            dataset: Dataset name
            
        Returns:
            Relevance score (0-1)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT relevance_score FROM dataset_relevance
            WHERE dataset_name = ?
        """, (dataset,))
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else 0.5  # Default score
    
    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get usage statistics for the last N days
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics dictionary
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total events
        cursor.execute("""
            SELECT COUNT(*) FROM usage_events
            WHERE timestamp >= ?
        """, (cutoff,))
        total_events = cursor.fetchone()[0]
        
        # Events by type
        cursor.execute("""
            SELECT event_type, COUNT(*) FROM usage_events
            WHERE timestamp >= ?
            GROUP BY event_type
        """, (cutoff,))
        events_by_type = dict(cursor.fetchall())
        
        # Most used datasets
        cursor.execute("""
            SELECT dataset_name, usage_count
            FROM dataset_relevance
            ORDER BY usage_count DESC
            LIMIT 10
        """)
        top_datasets = cursor.fetchall()
        
        # Top co-occurrences
        cursor.execute("""
            SELECT dataset_a, dataset_b, count
            FROM dataset_cooccurrence
            ORDER BY count DESC
            LIMIT 10
        """)
        top_cooccurrences = cursor.fetchall()
        
        conn.close()
        
        return {
            "period_days": days,
            "total_events": total_events,
            "events_by_type": events_by_type,
            "top_datasets": [{"name": ds, "count": count} for ds, count in top_datasets],
            "top_cooccurrences": [
                {"datasets": [a, b], "count": count}
                for a, b, count in top_cooccurrences
            ]
        }
