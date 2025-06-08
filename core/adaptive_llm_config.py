"""
Configuration settings for the WitsV3 Adaptive LLM System.

This module defines the configuration settings for the Adaptive LLM System,
including settings for the complexity analyzer, dynamic module loader,
semantic cache, and adaptive LLM interface.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class ComplexityAnalyzerSettings:
    """Settings for the ComplexityAnalyzer component."""
    
    # Model settings
    analyzer_model: str = "microsoft/deberta-v3-small"
    embedding_model: str = "all-MiniLM-L6-v2"
    use_embeddings: bool = True
    
    # Complexity factors
    complexity_factors: Dict[str, float] = field(default_factory=lambda: {
        "length": 0.2,
        "vocabulary": 0.3,
        "structure": 0.3,
        "reasoning": 0.4,
    })
    
    # Individual weights for direct access
    length_weight: float = 0.2
    vocab_weight: float = 0.3
    structure_weight: float = 0.3
    reasoning_weight: float = 0.4
    
    # Domain classification
    domains: List[str] = field(default_factory=lambda: [
        "general",
        "python",
        "math",
        "creative",
        "chat",
        "science",
        "history",
        "philosophy",
        "business",
        "technology",
    ])
    
    # Routing thresholds
    complexity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.3,
        "medium": 0.7,
    })
    
    # Module routing rules
    routing_rules: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "chat": {
            "complexity_max": 0.3,
            "domains": ["chat", "general"],
        },
        "python": {
            "complexity_min": 0.3,
            "domains": ["python", "technology"],
        },
        "math": {
            "complexity_min": 0.5,
            "domains": ["math", "science"],
        },
        "creative": {
            "complexity_min": 0.3,
            "domains": ["creative", "philosophy", "history"],
        },
        "base": {
            "default": True,
        },
    })
    
    # Performance settings
    enable_batching: bool = True
    cache_embeddings: bool = True
    
    # Fallback settings
    fallback_module: str = "base"


@dataclass
class DynamicModuleSettings:
    """Settings for the DynamicModuleLoader component."""
    
    # Resource budgets (in bytes)
    vram_budget: float = 7.5e9  # 7.5 GB
    ram_budget: float = 30e9    # 30 GB
    
    # Module directory
    module_dir: str = "models"
    
    # Preload modules
    preload_modules: List[str] = field(default_factory=lambda: ["base"])
    
    # Quantization settings
    enable_quantization: bool = True
    quantization_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "low": 4,     # 4-bit quantization for low complexity
        "medium": 8,  # 8-bit quantization for medium complexity
    })
    
    # Module eviction policy
    eviction_policy: str = "lru"  # Least Recently Used
    
    # Module loading timeout (seconds)
    loading_timeout: float = 30.0


@dataclass
class SemanticCacheSettings:
    """Settings for the SemanticCache component."""
    
    # Cache settings
    enable_caching: bool = True
    cache_size: int = 1000000
    embedding_dim: int = 768
    
    # Cache directory
    cache_dir: str = "cache"
    cache_file_path: str = field(default_factory=lambda: os.path.join("cache", "semantic_cache.json"))
    
    # Similarity settings
    similarity_metric: str = "cosine"
    cache_similarity_threshold: float = 0.85
    
    # Persistence settings
    enable_persistence: bool = True
    persistence_interval: int = 100  # Save every 100 patterns
    
    # User pattern tracking
    track_user_patterns: bool = True
    max_user_patterns: int = 1000
    
    # Compression settings
    enable_compression: bool = True
    compression_threshold: int = 100000  # Compress when cache size exceeds this
    compression_ratio: float = 0.5  # Compress to 50% of original size
    
    # TTL settings
    ttl_seconds: int = 86400  # 24 hours


@dataclass
class AdaptiveLLMSettings:
    """Settings for the AdaptiveLLMInterface component."""
    
    # Component settings
    complexity_analyzer_settings: ComplexityAnalyzerSettings = field(default_factory=ComplexityAnalyzerSettings)
    module_loader_settings: DynamicModuleSettings = field(default_factory=DynamicModuleSettings)
    semantic_cache_settings: SemanticCacheSettings = field(default_factory=SemanticCacheSettings)
    
    # Caching settings
    enable_caching: bool = True
    cache_similarity_threshold: float = 0.85
    
    # Performance tracking
    enable_performance_tracking: bool = True
    performance_log_path: str = os.path.join("logs", "adaptive_llm_performance.json")
    
    # Fallback settings
    fallback_to_base_on_error: bool = True
    max_fallback_attempts: int = 3
    
    # Generation settings
    default_temperature: float = 0.7
    default_max_tokens: int = 1024
