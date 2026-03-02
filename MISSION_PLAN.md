# Neuro-Guided Reinforcement Learning Trading Network

## Objective
A framework where neural networks guide reinforcement learning (RL) agents to build and navigate a knowledge graph of market dynamics, enabling autonomous decision-making.

## Strategy
Develop a knowledge graph capturing market behaviors, use neural networks for insights, implement RL agents, and test in simulations and real markets.

## Execution Output
SUMMARY: I've architected a comprehensive Neuro-Guided Reinforcement Learning Trading Network that integrates neural networks with RL agents and a Firebase-backed knowledge graph. The system features robust error handling, type hints, logging, and edge case management across multiple components.

OUTPUT:
### FILE: config.py
```python
"""
Configuration and environment management for the Neuro-Guided RL Trading Network.
Centralizes all settings, paths, and environment variables with validation.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation"""
    project_id: str
    database_url: str
    credentials_path: str
    
    def validate(self) -> None:
        """Validate Firebase configuration"""
        if not self.project_id:
            raise ValueError("Firebase project_id is required")
        if not Path(self.credentials_path).exists():
            raise FileNotFoundError(f"Firebase credentials not found at {self.credentials_path}")

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    symbols: list[str]
    timeframe: str
    initial_balance: float
    max_position_size: float
    transaction_fee: float
    
    def validate(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if self.max_position_size <= 0:
            raise ValueError("Max position size must be positive")

class Config:
    """Main configuration manager with singleton pattern"""
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.firebase = FirebaseConfig(
                project_id=os.getenv("FIREBASE_PROJECT_ID", ""),
                database_url=os.getenv("FIREBASE_DATABASE_URL", ""),
                credentials_path=os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
            )
            
            self.trading = TradingConfig(
                symbols=os.getenv("TRADING_SYMBOLS", "BTC/USDT,ETH/USDT").split(","),
                timeframe=os.getenv("TRADING_TIMEFRAME", "1h"),
                initial_balance=float(os.getenv("INITIAL_BALANCE", "10000.0")),
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
                transaction_fee=float(os.getenv("TRANSACTION_FEE", "0.001"))
            )
            
            self.log_level = os.getenv("LOG_LEVEL", "INFO")
            self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
            self.models_dir = Path(os.getenv("MODELS_DIR", "./models"))
            
            self._validate()
            self._setup_directories()
            self._initialized = True
    
    def _validate(self) -> None:
        """Validate all configurations"""
        self.firebase.validate()
        self.trading.validate()
        
        if not self.data_dir:
            raise ValueError("DATA_DIR must be specified")
    
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "market").mkdir(exist_ok=True)
        (self.data_dir / "embeddings").mkdir(exist_ok=True)
        (self.models_dir / "rl").mkdir(exist_ok=True)
        (self.models_dir / "nn").mkdir(exist_ok=True)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                    'level': self.log_level,
                    'stream': sys.stdout
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'standard',
                    'level': self.log_level,
                    'filename': self.data_dir / 'system.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': self