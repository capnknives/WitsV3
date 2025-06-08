# core/content_fallback.py
"""
Content Fallback System for WitsV3
Automatically switches to uncensored models when content is refused
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ContentRefusal:
    """Represents a detected content refusal"""
    original_prompt: str
    refusal_response: str
    model_used: str
    timestamp: datetime
    reason_detected: str
    retry_successful: bool = False
    fallback_model: Optional[str] = None


class ContentFallbackManager:
    """
    Manages automatic fallback to uncensored models when content is refused
    """
    
    def __init__(self, llm_interface, config):
        self.llm_interface = llm_interface
        self.config = config
        
        # Fallback model hierarchy
        self.uncensored_models = [
            "llama2-uncensored",
            "openhermes",  # Generally less restrictive
            "llama3"       # Final fallback
        ]
        
        # Content refusal detection patterns
        self.refusal_patterns = [
            # Direct refusals
            r"I can't|I cannot|I'm not able to|I'm unable to",
            r"I don't feel comfortable|I'm not comfortable",
            r"I cannot provide|I can't provide|I cannot create|I can't create",
            r"I cannot assist|I can't assist|I cannot help|I can't help",
            
            # Policy-based refusals
            r"against my guidelines|violates guidelines|against guidelines",
            r"inappropriate|harmful|offensive|controversial",
            r"ethical concerns|safety concerns|content policy",
            r"not appropriate|not suitable|not allowed",
            
            # Creative writing specific refusals
            r"I cannot write|I can't write.*that",
            r"cannot generate.*content|can't generate.*content",
            r"not comfortable writing|uncomfortable writing",
            r"violates content policy|against content policy",
            
            # Apology patterns that often indicate refusal
            r"I apologize.*but I cannot|Sorry.*but I cannot",
            r"I understand.*but I cannot|I appreciate.*but I cannot"
        ]
        
        # Content categories that often get refused
        self.sensitive_categories = [
            "violence", "mature_themes", "political", "controversial",
            "dark_fiction", "horror", "adult_content", "crime_fiction"
        ]
        
        # Track refusals for learning
        self.refusal_history: List[ContentRefusal] = []
        self.successful_fallbacks = {}
        
        logger.info("Content Fallback Manager initialized")
    
    async def generate_with_fallback(
        self, 
        prompt: str, 
        primary_model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        content_category: Optional[str] = None,
        force_uncensored: bool = False
    ) -> Dict[str, Any]:
        """
        Generate content with automatic fallback to uncensored models
        
        Args:
            prompt: The prompt to generate content for
            primary_model: Primary model to try first
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            content_category: Category of content (for tracking)
            force_uncensored: Skip primary model and go straight to uncensored
            
        Returns:
            Dict with generated content and metadata
        """
        
        if force_uncensored:
            logger.info("Forcing uncensored model due to user request")
            return await self._generate_uncensored(prompt, temperature, max_tokens)
        
        # Try primary model first
        try:
            logger.debug(f"Attempting generation with primary model: {primary_model}")
            
            response = await self.llm_interface.generate_text(
                prompt=prompt,
                model=primary_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Check if response indicates refusal
            if self._is_content_refused(response):
                logger.info(f"Content refused by {primary_model}, attempting fallback")
                
                # Record the refusal
                refusal = ContentRefusal(
                    original_prompt=prompt,
                    refusal_response=response,
                    model_used=primary_model,
                    timestamp=datetime.now(),
                    reason_detected=self._detect_refusal_reason(response)
                )
                
                # Try fallback
                fallback_result = await self._attempt_fallback(
                    prompt, temperature, max_tokens, refusal
                )
                
                if fallback_result['success']:
                    refusal.retry_successful = True
                    refusal.fallback_model = fallback_result['model_used']
                
                self.refusal_history.append(refusal)
                return fallback_result
            
            else:
                # Successful generation with primary model
                return {
                    'success': True,
                    'content': response,
                    'model_used': primary_model,
                    'fallback_used': False,
                    'refusal_detected': False
                }
                
        except Exception as e:
            logger.error(f"Error with primary model {primary_model}: {e}")
            # Try fallback on error
            return await self._attempt_fallback(
                prompt, temperature, max_tokens, 
                error_fallback=True, original_error=str(e)
            )
    
    async def generate_streaming_with_fallback(
        self,
        prompt: str,
        primary_model: str,
        temperature: float = 0.7,
        force_uncensored: bool = False,
        refusal_check_interval: int = 50  # Check for refusal every N characters
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming content with fallback detection
        """
        
        if force_uncensored:
            async for chunk in self._stream_uncensored(prompt, temperature):
                yield chunk
            return
        
        accumulated_content = ""
        refusal_detected = False
        
        try:
            async for chunk in self.llm_interface.stream_text(
                prompt=prompt,
                model=primary_model,
                temperature=temperature
            ):
                accumulated_content += chunk
                
                # Check for refusal patterns periodically
                if len(accumulated_content) % refusal_check_interval < len(chunk):
                    if self._is_content_refused(accumulated_content):
                        refusal_detected = True
                        yield {
                            'type': 'refusal_detected',
                            'content': '⚠️ Content refusal detected, switching to uncensored model...',
                            'model_switch': True
                        }
                        break
                
                yield {
                    'type': 'content',
                    'content': chunk,
                    'model': primary_model,
                    'fallback': False
                }
            
            # If refusal was detected, continue with uncensored model
            if refusal_detected:
                async for chunk in self._stream_uncensored(prompt, temperature):
                    yield chunk
                    
        except Exception as e:
            yield {
                'type': 'error',
                'content': f'Error with {primary_model}: {str(e)}',
                'fallback_starting': True
            }
            
            async for chunk in self._stream_uncensored(prompt, temperature):
                yield chunk
    
    def _is_content_refused(self, response: str) -> bool:
        """Check if the response indicates content refusal"""
        
        response_lower = response.lower()
        
        # Check against refusal patterns
        for pattern in self.refusal_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                return True
        
        # Additional heuristics
        # Short responses that seem like refusals
        if len(response) < 100 and any(word in response_lower for word in 
                                      ['cannot', 'unable', 'sorry', 'apologize']):
            return True
        
        # Responses that immediately pivot to alternatives
        if response_lower.startswith(('instead', 'however', 'alternatively')) and \
           len(response) < 200:
            return True
        
        return False
    
    def _detect_refusal_reason(self, response: str) -> str:
        """Detect the likely reason for content refusal"""
        
        response_lower = response.lower()
        
        if 'guideline' in response_lower or 'policy' in response_lower:
            return "policy_violation"
        elif 'appropriate' in response_lower or 'suitable' in response_lower:
            return "inappropriate_content"
        elif 'harmful' in response_lower or 'offensive' in response_lower:
            return "harmful_content"
        elif 'comfortable' in response_lower:
            return "comfort_boundary"
        elif 'ethical' in response_lower or 'safety' in response_lower:
            return "ethical_concern"
        else:
            return "general_refusal"
    
    async def _attempt_fallback(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: Optional[int],
        refusal: Optional[ContentRefusal] = None,
        error_fallback: bool = False,
        original_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Attempt fallback to uncensored models"""
        
        for fallback_model in self.uncensored_models:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                
                # Modify prompt if needed for uncensored model
                modified_prompt = self._modify_prompt_for_uncensored(prompt, refusal)
                
                response = await self.llm_interface.generate_text(
                    prompt=modified_prompt,
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Check if this model also refuses
                if not self._is_content_refused(response):
                    logger.info(f"Successful fallback with {fallback_model}")
                    
                    # Track successful fallback
                    self.successful_fallbacks[fallback_model] = \
                        self.successful_fallbacks.get(fallback_model, 0) + 1
                    
                    return {
                        'success': True,
                        'content': response,
                        'model_used': fallback_model,
                        'fallback_used': True,
                        'refusal_detected': refusal is not None,
                        'original_model': refusal.model_used if refusal else None,
                        'error_fallback': error_fallback,
                        'original_error': original_error
                    }
                else:
                    logger.warning(f"Fallback model {fallback_model} also refused content")
                    continue
                    
            except Exception as e:
                logger.error(f"Error with fallback model {fallback_model}: {e}")
                continue
        
        # All models failed
        return {
            'success': False,
            'content': "All available models refused to generate the requested content.",
            'model_used': None,
            'fallback_used': True,
            'refusal_detected': True,
            'all_models_failed': True
        }
    
    async def _generate_uncensored(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Generate content directly with uncensored model"""
        
        uncensored_model = self.uncensored_models[0]  # Primary uncensored model
        
        try:
            response = await self.llm_interface.generate_text(
                prompt=prompt,
                model=uncensored_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                'success': True,
                'content': response,
                'model_used': uncensored_model,
                'fallback_used': False,
                'forced_uncensored': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'content': f"Error with uncensored model: {str(e)}",
                'model_used': uncensored_model,
                'error': str(e)
            }
    
    async def _stream_uncensored(
        self, 
        prompt: str, 
        temperature: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream content from uncensored model"""
        
        uncensored_model = self.uncensored_models[0]
        
        try:
            async for chunk in self.llm_interface.stream_text(
                prompt=prompt,
                model=uncensored_model,
                temperature=temperature
            ):
                yield {
                    'type': 'content',
                    'content': chunk,
                    'model': uncensored_model,
                    'fallback': True
                }
        except Exception as e:
            yield {
                'type': 'error',
                'content': f'Error with uncensored model: {str(e)}',
                'model': uncensored_model
            }
    
    def _modify_prompt_for_uncensored(
        self, 
        prompt: str, 
        refusal: Optional[ContentRefusal] = None
    ) -> str:
        """Modify prompt to work better with uncensored models"""
        
        # Add context that this is for creative/fictional purposes
        if any(word in prompt.lower() for word in ['story', 'character', 'fiction', 'novel', 'book']):
            prompt = f"For a fictional creative writing project:\n\n{prompt}"
        
        # If we know why the original model refused, address it
        if refusal and refusal.reason_detected == "comfort_boundary":
            prompt = f"Please complete this creative writing task:\n\n{prompt}"
        
        return prompt
    
    def should_preempt_with_uncensored(self, prompt: str, content_category: Optional[str] = None) -> bool:
        """
        Determine if we should skip the primary model and go straight to uncensored
        """
        
        # Check for obvious sensitive content indicators
        sensitive_keywords = [
            'violence', 'death', 'murder', 'killing', 'blood', 'gore',
            'adult', 'mature', 'explicit', 'nsfw',
            'controversial', 'political', 'sensitive',
            'dark', 'horror', 'thriller', 'crime'
        ]
        
        prompt_lower = prompt.lower()
        
        # If multiple sensitive keywords, likely to be refused
        keyword_count = sum(1 for keyword in sensitive_keywords if keyword in prompt_lower)
        if keyword_count >= 2:
            return True
        
        # Category-based preemption
        if content_category in self.sensitive_categories:
            return True
        
        # Historical refusal patterns
        for past_refusal in self.refusal_history[-10:]:  # Check last 10 refusals
            if any(word in prompt_lower for word in past_refusal.original_prompt.lower().split()[:5]):
                return True
        
        return False
    
    def get_refusal_statistics(self) -> Dict[str, Any]:
        """Get statistics about content refusals and fallbacks"""
        
        total_refusals = len(self.refusal_history)
        successful_fallbacks = len([r for r in self.refusal_history if r.retry_successful])
        
        # Refusal reasons breakdown
        reason_counts = {}
        for refusal in self.refusal_history:
            reason_counts[refusal.reason_detected] = \
                reason_counts.get(refusal.reason_detected, 0) + 1
        
        return {
            'total_refusals': total_refusals,
            'successful_fallbacks': successful_fallbacks,
            'fallback_success_rate': successful_fallbacks / max(total_refusals, 1),
            'refusal_reasons': reason_counts,
            'top_fallback_models': self.successful_fallbacks,
            'recent_refusals': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'model': r.model_used,
                    'reason': r.reason_detected,
                    'fallback_successful': r.retry_successful
                }
                for r in self.refusal_history[-5:]
            ]
        }


# Integration with existing agents
class ContentAwareAgent:
    """Mixin for agents that need content fallback capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_fallback = ContentFallbackManager(
            self.llm_interface, 
            self.config
        )
    
    async def generate_response_with_fallback(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        content_category: Optional[str] = None,
        force_uncensored: bool = False
    ) -> str:
        """Generate response with automatic uncensored fallback"""
        
        primary_model = model_name or self.get_model_name()
        temp = temperature or self.temperature
        
        # Check if we should preemptively use uncensored model
        if not force_uncensored:
            force_uncensored = self.content_fallback.should_preempt_with_uncensored(
                prompt, content_category
            )
        
        result = await self.content_fallback.generate_with_fallback(
            prompt=prompt,
            primary_model=primary_model,
            temperature=temp,
            max_tokens=max_tokens,
            content_category=content_category,
            force_uncensored=force_uncensored
        )
        
        if result['success']:
            # Log fallback usage for transparency
            if result.get('fallback_used'):
                self.logger.info(
                    f"Used fallback model {result['model_used']} "
                    f"after refusal from {result.get('original_model', 'primary')}"
                )
            return result['content']
        else:
            return "I was unable to generate the requested content due to model limitations."
    
    async def generate_streaming_response_with_fallback(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        force_uncensored: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with fallback"""
        
        primary_model = model_name or self.get_model_name()
        temp = temperature or self.temperature
        
        async for chunk_data in self.content_fallback.generate_streaming_with_fallback(
            prompt=prompt,
            primary_model=primary_model,
            temperature=temp,
            force_uncensored=force_uncensored
        ):
            if chunk_data['type'] == 'content':
                yield chunk_data['content']
            elif chunk_data['type'] == 'refusal_detected':
                yield f"\n\n{chunk_data['content']}\n\n"
            elif chunk_data['type'] == 'error':
                yield f"\n\n⚠️ {chunk_data['content']}\n\n"


# Test function
async def test_content_fallback():
    """Test the content fallback system"""
    from core.config import load_config
    from core.llm_interface import OllamaInterface
    
    config = load_config("config.yaml")
    llm_interface = OllamaInterface(config=config)
    
    fallback_manager = ContentFallbackManager(llm_interface, config)
    
    # Test prompts that might get refused
    test_prompts = [
        "Write a dark thriller scene with violence",
        "Create a character who breaks moral boundaries",
        "Write about controversial political topics",
        "Tell a story with mature themes"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting: {prompt[:50]}...")
        
        result = await fallback_manager.generate_with_fallback(
            prompt=prompt,
            primary_model="hf.co/google/gemma-3-4b-it-qat-q4_0-gguf",
            content_category="dark_fiction"
        )
        
        print(f"Success: {result['success']}")
        print(f"Model used: {result['model_used']}")
        print(f"Fallback used: {result.get('fallback_used', False)}")
        
    # Show statistics
    stats = fallback_manager.get_refusal_statistics()
    print(f"\nRefusal statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(test_content_fallback())