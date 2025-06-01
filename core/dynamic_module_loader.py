"""
DynamicModuleLoader for WitsV3 Adaptive LLM System.

This module implements a dynamic module loader that manages specialized LLM
modules with VRAM/RAM budgeting and quantization.
"""

import logging
import asyncio
import time
import json
import os
import gc
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import numpy as np

from .config import WitsV3Config
from .adaptive_llm_config import DynamicModuleSettings

class DynamicModuleLoader:
    """
    Manages specialized LLM modules with VRAM/RAM budgeting.
    
    The DynamicModuleLoader manages specialized LLM modules with VRAM/RAM
    budgeting and quantization for optimal resource usage.
    """
    
    def __init__(self, config: WitsV3Config):
        """
        Initialize the DynamicModuleLoader.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.settings = DynamicModuleSettings()  # Use defaults
        self.logger = logging.getLogger("WitsV3.DynamicModuleLoader")
        
        # LRU cache for loaded modules
        self.loaded_modules = OrderedDict()
        self.ram_cache = OrderedDict()
        
        # Module registry
        self.module_registry = {
            'base': {'size': 2e9, 'path': os.path.join(self.settings.module_dir, 'base_module.safetensors')},
            'python': {'size': 2.5e9, 'path': os.path.join(self.settings.module_dir, 'python_expert.safetensors')},
            'creative': {'size': 2.2e9, 'path': os.path.join(self.settings.module_dir, 'creative_expert.safetensors')},
            'math': {'size': 2.8e9, 'path': os.path.join(self.settings.module_dir, 'math_expert.safetensors')},
            'chat': {'size': 1.8e9, 'path': os.path.join(self.settings.module_dir, 'chat_expert.safetensors')},
        }
        
        # Resource tracking
        self.current_vram_usage = 0
        self.current_ram_usage = 0
        
        # Preload modules if specified
        if self.settings.preload_modules:
            asyncio.create_task(self._preload_modules())
        
        self.logger.info("DynamicModuleLoader initialized")
    
    async def _preload_modules(self) -> None:
        """Preload specified modules."""
        for module_name in self.settings.preload_modules:
            if module_name in self.module_registry:
                self.logger.info(f"Preloading module: {module_name}")
                await self.load_module(module_name)
    
    async def load_module(self, module_name: str, complexity_score: float = 0.5) -> Any:
        """
        Load a module by name.
        
        Args:
            module_name: The name of the module to load
            complexity_score: The complexity score (0.0 to 1.0)
            
        Returns:
            The loaded module
        """
        start_time = time.time()
        
        # Check if module exists
        if module_name not in self.module_registry:
            self.logger.error(f"Module not found: {module_name}")
            raise ValueError(f"Module not found: {module_name}")
        
        # Check if already loaded
        if module_name in self.loaded_modules:
            # Move to end (most recently used)
            self.loaded_modules.move_to_end(module_name)
            self.logger.debug(f"Module already loaded: {module_name}")
            return self.loaded_modules[module_name]
            
        # Check RAM cache
        if module_name in self.ram_cache:
            self.logger.debug(f"Module found in RAM cache: {module_name}")
            module = self.ram_cache.pop(module_name)
            return await self._move_to_vram(module_name, module)
            
        # Load from disk
        self.logger.info(f"Loading module from disk: {module_name}")
        return await self._load_from_disk(module_name, complexity_score)
    
    async def _move_to_vram(self, module_name: str, module: Any) -> Any:
        """
        Move a module from RAM to VRAM.
        
        Args:
            module_name: The name of the module
            module: The module to move
            
        Returns:
            The module on GPU
        """
        module_size = self.module_registry[module_name]['size']
        
        # Make room if needed
        while self.current_vram_usage + module_size > self.settings.vram_budget:
            await self._evict_oldest_vram_module()
            
        # Move to GPU
        try:
            module = module.cuda()
            self.loaded_modules[module_name] = module
            self.current_vram_usage += module_size
            
            self.logger.debug(f"Module moved to VRAM: {module_name}")
            self.logger.debug(f"Current VRAM usage: {self.current_vram_usage / 1e9:.2f} GB")
            
            return module
            
        except Exception as e:
            self.logger.error(f"Error moving module to VRAM: {e}")
            raise
    
    async def _evict_oldest_vram_module(self) -> None:
        """Evict the oldest module from VRAM to RAM."""
        if not self.loaded_modules:
            self.logger.error("Cannot evict: no modules loaded in VRAM")
            raise RuntimeError("Cannot evict: no modules loaded in VRAM")
            
        # Get oldest (least recently used)
        old_name, old_module = self.loaded_modules.popitem(last=False)
        old_size = self.module_registry[old_name]['size']
        
        self.logger.info(f"Evicting module from VRAM: {old_name}")
        
        try:
            # Move to RAM cache
            old_module = old_module.cpu()
            
            # Make room in RAM if needed
            while self.current_ram_usage + old_size > self.settings.ram_budget:
                await self._evict_oldest_ram_module()
                
            self.ram_cache[old_name] = old_module
            self.current_vram_usage -= old_size
            self.current_ram_usage += old_size
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            
            self.logger.debug(f"Module evicted to RAM: {old_name}")
            self.logger.debug(f"Current VRAM usage: {self.current_vram_usage / 1e9:.2f} GB")
            self.logger.debug(f"Current RAM usage: {self.current_ram_usage / 1e9:.2f} GB")
            
        except Exception as e:
            self.logger.error(f"Error evicting module from VRAM: {e}")
            raise
    
    async def _evict_oldest_ram_module(self) -> None:
        """Evict the oldest module from RAM."""
        if not self.ram_cache:
            self.logger.error("Cannot evict: no modules loaded in RAM")
            raise RuntimeError("Cannot evict: no modules loaded in RAM")
            
        # Get oldest (least recently used)
        old_name, _ = self.ram_cache.popitem(last=False)
        old_size = self.module_registry[old_name]['size']
        
        self.logger.info(f"Evicting module from RAM: {old_name}")
        
        # Update RAM usage
        self.current_ram_usage -= old_size
        
        # Force garbage collection
        gc.collect()
        
        self.logger.debug(f"Module evicted from RAM: {old_name}")
        self.logger.debug(f"Current RAM usage: {self.current_ram_usage / 1e9:.2f} GB")
    
    async def _load_from_disk(self, module_name: str, complexity_score: float) -> Any:
        """
        Load a module from disk.
        
        Args:
            module_name: The name of the module
            complexity_score: The complexity score (0.0 to 1.0)
            
        Returns:
            The loaded module
        """
        module_info = self.module_registry[module_name]
        module_path = module_info['path']
        
        self.logger.info(f"Loading module from disk: {module_name} (path: {module_path})")
        
        # Check if file exists
        if not os.path.exists(module_path):
            self.logger.error(f"Module file not found: {module_path}")
            raise FileNotFoundError(f"Module file not found: {module_path}")
        
        try:
            # Determine quantization level based on complexity
            bits = self._determine_quantization_bits(complexity_score)
            
            # Load module with appropriate quantization
            if bits == 4:
                self.logger.debug(f"Loading module with 4-bit quantization: {module_name}")
                module = await self._load_quantized(module_path, bits=4)
            elif bits == 8:
                self.logger.debug(f"Loading module with 8-bit quantization: {module_name}")
                module = await self._load_quantized(module_path, bits=8)
            else:
                self.logger.debug(f"Loading module with full precision: {module_name}")
                module = await self._load_full_precision(module_path)
                
            # Move to VRAM
            return await self._move_to_vram(module_name, module)
            
        except Exception as e:
            self.logger.error(f"Error loading module from disk: {e}")
            raise
    
    def _determine_quantization_bits(self, complexity_score: float) -> int:
        """
        Determine quantization bits based on complexity score.
        
        Args:
            complexity_score: The complexity score (0.0 to 1.0)
            
        Returns:
            Quantization bits (4, 8, or 16)
        """
        if not self.settings.enable_quantization:
            return 16  # Full precision
            
        if complexity_score < self.settings.quantization_thresholds.get('low', 0.3):
            return 4  # 4-bit quantization
        elif complexity_score < self.settings.quantization_thresholds.get('medium', 0.7):
            return 8  # 8-bit quantization
        else:
            return 16  # Full precision
    
    async def _load_quantized(self, module_path: str, bits: int = 8) -> Any:
        """
        Load a module with quantization.
        
        Args:
            module_path: The path to the module file
            bits: The quantization bits (4 or 8)
            
        Returns:
            The loaded module
        """
        # This is a placeholder for actual quantized loading
        # In a real implementation, this would use libraries like bitsandbytes
        
        self.logger.debug(f"Loading module with {bits}-bit quantization: {module_path}")
        
        # Simulate loading delay
        await asyncio.sleep(0.5)
        
        # Create a dummy module for demonstration
        class DummyModule:
            def __init__(self, name, bits):
                self.name = name
                self.bits = bits
                
            def to(self, device):
                self.device = device
                return self
                
            def cuda(self):
                return self.to('cuda')
                
            def cpu(self):
                return self.to('cpu')
                
            def generate(self, input_ids, **kwargs):
                # Simulate generation
                return torch.tensor([[101, 102, 103, 104, 105]])
        
        return DummyModule(os.path.basename(module_path), bits)
    
    async def _load_full_precision(self, module_path: str) -> Any:
        """
        Load a module with full precision.
        
        Args:
            module_path: The path to the module file
            
        Returns:
            The loaded module
        """
        self.logger.debug(f"Loading module with full precision: {module_path}")
        
        # Simulate loading delay
        await asyncio.sleep(1.0)
        
        # Create a dummy module for demonstration
        class DummyModule:
            def __init__(self, name):
                self.name = name
                self.bits = 16
                
            def to(self, device):
                self.device = device
                return self
                
            def cuda(self):
                return self.to('cuda')
                
            def cpu(self):
                return self.to('cpu')
                
            def generate(self, input_ids, **kwargs):
                # Simulate generation
                return torch.tensor([[101, 102, 103, 104, 105]])
        
        return DummyModule(os.path.basename(module_path))
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """
        Get information about a module.
        
        Args:
            module_name: The name of the module
            
        Returns:
            Dictionary with module information
        """
        if module_name not in self.module_registry:
            self.logger.error(f"Module not found: {module_name}")
            raise ValueError(f"Module not found: {module_name}")
            
        module_info = self.module_registry[module_name].copy()
        
        # Add status information
        if module_name in self.loaded_modules:
            module_info['status'] = 'loaded_vram'
        elif module_name in self.ram_cache:
            module_info['status'] = 'loaded_ram'
        else:
            module_info['status'] = 'not_loaded'
            
        return module_info
    
    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with resource usage information
        """
        return {
            'vram_usage': self.current_vram_usage,
            'vram_budget': self.settings.vram_budget,
            'vram_percent': self.current_vram_usage / self.settings.vram_budget * 100,
            'ram_usage': self.current_ram_usage,
            'ram_budget': self.settings.ram_budget,
            'ram_percent': self.current_ram_usage / self.settings.ram_budget * 100,
        }
    
    async def unload_all(self) -> None:
        """Unload all modules."""
        self.logger.info("Unloading all modules")
        
        # Clear VRAM modules
        self.loaded_modules.clear()
        
        # Clear RAM cache
        self.ram_cache.clear()
        
        # Reset resource usage
        self.current_vram_usage = 0
        self.current_ram_usage = 0
        
        # Force garbage collection
        torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.debug("All modules unloaded")


# Test function
async def test_dynamic_module_loader():
    """Test the DynamicModuleLoader functionality."""
    from .config import WitsV3Config
    import os
    
    print("Testing DynamicModuleLoader...")
    
    # Load config
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    config_file_path = os.path.join(project_root_dir, "config.yaml")
    
    config = WitsV3Config.from_yaml(config_file_path)
    
    # Create module loader
    loader = DynamicModuleLoader(config)
    
    # Test loading modules
    print("\nLoading modules...")
    
    # Load base module
    print("\nLoading base module...")
    base_module = await loader.load_module('base', 0.2)
    print(f"Base module loaded: {base_module.name}, {base_module.bits}-bit")
    
    # Load python module
    print("\nLoading python module...")
    python_module = await loader.load_module('python', 0.6)
    print(f"Python module loaded: {python_module.name}, {python_module.bits}-bit")
    
    # Load math module
    print("\nLoading math module...")
    math_module = await loader.load_module('math', 0.9)
    print(f"Math module loaded: {math_module.name}, {math_module.bits}-bit")
    
    # Test resource usage
    print("\nResource usage:")
    usage = loader.get_resource_usage()
    print(f"VRAM usage: {usage['vram_usage'] / 1e9:.2f} GB / {usage['vram_budget'] / 1e9:.2f} GB ({usage['vram_percent']:.1f}%)")
    print(f"RAM usage: {usage['ram_usage'] / 1e9:.2f} GB / {usage['ram_budget'] / 1e9:.2f} GB ({usage['ram_percent']:.1f}%)")
    
    # Test module info
    print("\nModule info:")
    for module_name in loader.module_registry:
        info = loader.get_module_info(module_name)
        print(f"{module_name}: {info['status']}, size: {info['size'] / 1e9:.2f} GB")
    
    # Test unloading
    print("\nUnloading all modules...")
    await loader.unload_all()
    
    print("\nDynamicModuleLoader tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_dynamic_module_loader())
