"""
Configuration management for the Agentic Data Product Builder.

This module handles loading environment variables and providing
configuration settings across the application.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class LLMConfig(BaseModel):
    """LLM API configuration."""
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key")
    default_model: str = Field(default="gpt-4", description="Default LLM model to use")
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if at least one LLM API key is configured."""
        return bool(self.openai_api_key or self.anthropic_api_key or self.gemini_api_key)


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: Optional[str] = Field(default=None, description="Neo4j password")
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if Neo4j is properly configured."""
        return bool(self.password)


class AppConfig(BaseModel):
    """Application configuration."""
    environment: str = Field(default="development", description="Current environment")
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="Data directory path")
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


class Settings(BaseModel):
    """Main settings container."""
    llm: LLMConfig
    neo4j: Neo4jConfig
    app: AppConfig
    
    @classmethod
    def load(cls) -> "Settings":
        """Load all settings from environment."""
        return cls(
            llm=LLMConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            app=AppConfig.from_env(),
        )


# Global settings instance
settings = Settings.load()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def check_configuration() -> dict:
    """
    Check the current configuration status.
    
    Returns:
        dict: Configuration status for each component
    """
    return {
        "llm_configured": settings.llm.is_configured,
        "neo4j_configured": settings.neo4j.is_configured,
        "environment": settings.app.environment,
        "log_level": settings.app.log_level,
    }


if __name__ == "__main__":
    # Quick configuration check
    print("Configuration Status:")
    print("-" * 40)
    status = check_configuration()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    if not status["llm_configured"]:
        print("\n⚠️  Warning: No LLM API key configured.")
        print("   Please add OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file")

