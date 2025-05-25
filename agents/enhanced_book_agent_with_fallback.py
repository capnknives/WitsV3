# agents/enhanced_book_writing_agent.py
"""
Enhanced Book Writing Agent with Content Fallback System
Automatically switches to uncensored models when content is refused
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator

from agents.book_writing_agent import BookWritingAgent, BookStructure, Chapter
from core.content_fallback import ContentFallbackManager, ContentAwareAgent
from core.schemas import StreamData


class EnhancedBookWritingAgent(BookWritingAgent, ContentAwareAgent):
    """
    Enhanced Book Writing Agent with intelligent content fallback
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Content categorization for smart model selection
        self.content_categories = {
            'mature_themes': ['mature', 'adult', 'explicit', 'sexual', 'intimate'],
            'dark_fiction': ['dark', 'noir', 'gritty', 'disturbing', 'twisted'],
            'violence': ['violence', 'murder', 'killing', 'blood', 'death', 'fight', 'battle'],
            'horror': ['horror', 'scary', 'terrifying', 'nightmare', 'fear', 'supernatural'],
            'thriller': ['thriller', 'suspense', 'danger', 'chase', 'escape'],
            'controversial': ['controversial', 'political', 'sensitive', 'taboo'],
            'crime': ['crime', 'criminal', 'illegal', 'law', 'police', 'detective'],
            'psychological': ['psychological', 'mental', 'mind', 'psyche', 'trauma']
        }
        
        # Writing style preferences for different content
        self.style_preferences = {
            'mature_themes': {'temperature': 0.85, 'force_uncensored': True},
            'dark_fiction': {'temperature': 0.8, 'force_uncensored': True},
            'violence': {'temperature': 0.75, 'force_uncensored': True},
            'horror': {'temperature': 0.8, 'force_uncensored': True},
            'crime': {'temperature': 0.75, 'force_uncensored': False},
            'general': {'temperature': 0.7, 'force_uncensored': False}
        }
        
        self.logger.info("Enhanced Book Writing Agent with content fallback initialized")
    
    async def _categorize_content(self, content_description: str) -> List[str]:
        """Categorize content to determine appropriate model and settings"""
        
        content_lower = content_description.lower()
        detected_categories = []
        
        for category, keywords in self.content_categories.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_categories.append(category)
        
        return detected_categories
    
    async def _get_writing_preferences(self, content_categories: List[str]) -> Dict[str, Any]:
        """Get writing preferences based on content categories"""
        
        # Default preferences
        preferences = self.style_preferences['general'].copy()
        
        # Override based on detected categories
        for category in content_categories:
            if category in self.style_preferences:
                preferences.update(self.style_preferences[category])
                break  # Use first matching category
        
        return preferences
    
    async def _handle_chapter_writing(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Enhanced chapter writing with content fallback"""
        
        yield self.stream_action("Writing chapter content with intelligent model selection...")
        
        # Analyze content for appropriate model selection
        content_description = f"{task_analysis['topic']} {task_analysis['genre']} {task_analysis.get('style', '')}"
        content_categories = await self._categorize_content(content_description)
        writing_prefs = await self._get_writing_preferences(content_categories)
        
        if content_categories:
            yield self.stream_thinking(f"Detected content categories: {', '.join(content_categories)}")
            
            if writing_prefs.get('force_uncensored'):
                yield self.stream_thinking("Using uncensored model for sensitive content")
        
        chapter_id = str(uuid.uuid4())
        
        writing_prompt = f"""
        Write a compelling chapter about {task_analysis['topic']} in {task_analysis['style']} style.
        
        Requirements:
        - Genre: {task_analysis['genre']}
        - Style: {task_analysis['style']}
        - Length: approximately {task_analysis.get('length', 2000)} words
        - Maintain consistent tone and voice
        - Include vivid details and engaging narrative
        - Don't hold back on creative expression
        """
        
        # Add content-specific instructions
        if 'dark_fiction' in content_categories:
            writing_prompt += "\n- Embrace dark themes and complex moral situations"
        if 'mature_themes' in content_categories:
            writing_prompt += "\n- Handle mature themes with appropriate depth and sensitivity"
        if 'violence' in content_categories:
            writing_prompt += "\n- Depict action and conflict realistically when necessary for the story"
        
        yield self.stream_thinking("Crafting chapter content...")
        
        try:
            # Use enhanced generation with fallback
            chapter_content = await self.generate_response_with_fallback(
                prompt=writing_prompt,
                temperature=writing_prefs['temperature'],
                content_category=content_categories[0] if content_categories else None,
                force_uncensored=writing_prefs.get('force_uncensored', False)
            )
            
            # Create chapter object
            chapter = Chapter(
                id=chapter_id,
                title=f"Chapter: {task_analysis['topic']}",
                outline="",
                content=chapter_content,
                word_count=len(chapter_content.split()),
                status="drafted"
            )
            
            # Store with content categories for future reference
            await self.store_memory(
                content=f"Wrote chapter: {chapter.title} ({chapter.word_count} words)",
                segment_type="CHAPTER_CONTENT",
                importance=0.8,
                metadata={
                    "chapter_id": chapter_id,
                    "content_categories": content_categories,
                    "writing_preferences": writing_prefs,
                    "session_id": session_id
                }
            )
            
            yield self.stream_result(f"Completed chapter: {chapter.word_count} words written")
            
            # Show content categories if detected
            if content_categories:
                yield self.stream_observation(f"Content categories: {', '.join(content_categories)}")
            
        except Exception as e:
            yield self.stream_error(f"Error writing chapter: {str(e)}")
    
    async def write_book_chapter_enhanced(
        self,
        book_id: str,
        chapter_outline: str,
        content_guidelines: Optional[Dict[str, Any]] = None,
        force_uncensored: bool = False
    ) -> AsyncGenerator[StreamData, None]:
        """
        Write a book chapter with enhanced content handling
        """
        
        yield self.stream_action("Starting enhanced chapter writing...")
        
        # Analyze chapter outline for content categories
        content_categories = await self._categorize_content(chapter_outline)
        writing_prefs = await self._get_writing_preferences(content_categories)
        
        # Override with user preferences
        if force_uncensored:
            writing_prefs['force_uncensored'] = True
            yield self.stream_thinking("Forcing uncensored model per user request")
        
        if content_guidelines:
            writing_prefs.update(content_guidelines)
        
        # Enhanced writing prompt
        enhanced_prompt = f"""
        Write a compelling book chapter based on this outline:
        
        {chapter_outline}
        
        Writing Guidelines:
        - Create engaging, well-paced narrative
        - Develop characters naturally through action and dialogue
        - Use vivid, immersive descriptions
        - Maintain consistent voice and style
        - Don't self-censor - write what the story demands
        - Focus on emotional truth and authentic character reactions
        
        Write freely and creatively without unnecessary restrictions.
        """
        
        yield self.stream_thinking("Generating chapter content...")
        
        # Use streaming generation with fallback
        chapter_content = ""
        async for chunk in self.generate_streaming_response_with_fallback(
            prompt=enhanced_prompt,
            temperature=writing_prefs['temperature'],
            force_uncensored=writing_prefs.get('force_uncensored', False)
        ):
            chapter_content += chunk
            
            # Show progress periodically
            if len(chapter_content) % 500 < 50:
                word_count = len(chapter_content.split())
                yield self.stream_action(f"Written {word_count} words...")
        
        final_word_count = len(chapter_content.split())
        yield self.stream_result(f"Chapter completed: {final_word_count} words")
        
        # Store the chapter
        await self.store_memory(
            content=f"Enhanced chapter: {chapter_content}",
            segment_type="ENHANCED_CHAPTER",
            importance=0.9,
            metadata={
                "word_count": final_word_count,
                "content_categories": content_categories,
                "writing_preferences": writing_prefs,
                "book_id": book_id
            }
        )
        
        yield self.stream_result(chapter_content)
    
    async def handle_content_request_with_context(
        self,
        request: str,
        content_type: str = "general",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamData, None]:
        """
        Handle content requests with full context awareness
        """
        
        yield self.stream_thinking("Analyzing content request with context...")
        
        # Determine if this should use uncensored model
        content_categories = await self._categorize_content(request)
        writing_prefs = await self._get_writing_preferences(content_categories)
        
        # User preferences override
        if user_preferences:
            if user_preferences.get('creative_freedom_level') == 'high':
                writing_prefs['force_uncensored'] = True
            if user_preferences.get('allow_auto_uncensored'):
                # Keep automatic detection
                pass
            else:
                writing_prefs['force_uncensored'] = False
        
        # Check if we should proactively use uncensored model
        should_preempt = self.content_fallback.should_preempt_with_uncensored(
            request, content_categories[0] if content_categories else None
        )
        
        if should_preempt or writing_prefs.get('force_uncensored'):
            yield self.stream_thinking("Using uncensored model based on content analysis")
        
        # Generate content with intelligent model selection
        try:
            result = await self.content_fallback.generate_with_fallback(
                prompt=request,
                primary_model=self.get_model_name(),
                temperature=writing_prefs['temperature'],
                content_category=content_categories[0] if content_categories else None,
                force_uncensored=writing_prefs.get('force_uncensored', False)
            )
            
            if result['success']:
                # Inform user about model used if fallback occurred
                if result.get('fallback_used'):
                    yield self.stream_observation(
                        f"Switched to {result['model_used']} for unrestricted creative expression"
                    )
                
                yield self.stream_result(result['content'])
            else:
                yield self.stream_error("Unable to generate content with available models")
                
        except Exception as e:
            yield self.stream_error(f"Error generating content: {str(e)}")
    
    async def get_content_statistics(self) -> Dict[str, Any]:
        """Get statistics about content generation and fallbacks"""
        
        # Get fallback statistics
        fallback_stats = self.content_fallback.get_refusal_statistics()
        
        # Get writing statistics
        writing_stats = await self.get_writing_statistics()
        
        return {
            'writing_stats': writing_stats,
            'fallback_stats': fallback_stats,
            'content_categories_used': len(set().union(
                *[memory.metadata.get('content_categories', []) 
                  for memory in await self.search_memory("CHAPTER_CONTENT")]
            )),
            'uncensored_usage_rate': fallback_stats.get('fallback_success_rate', 0),
            'most_common_refusal_reason': max(
                fallback_stats.get('refusal_reasons', {}).items(),
                key=lambda x: x[1],
                default=('none', 0)
            )[0]
        }


# Usage examples and integration
async def example_enhanced_writing():
    """Example of using the enhanced book writing agent"""
    
    from core.config import load_config
    from core.llm_interface import OllamaInterface
    from core.memory_manager import MemoryManager
    
    # Setup
    config = load_config("config.yaml")
    llm_interface = OllamaInterface(config=config)
    memory_manager = MemoryManager(config, llm_interface)
    await memory_manager.initialize()
    
    # Create enhanced agent
    book_agent = EnhancedBookWritingAgent(
        "EnhancedBookWriter",
        config,
        llm_interface,
        memory_manager
    )
    
    # Example 1: Regular content (will use primary model)
    print("=== Regular Content Example ===")
    async for stream in book_agent.handle_content_request_with_context(
        "Write a heartwarming story about friendship",
        content_type="general"
    ):
        if stream.type == 'result':
            print(f"Generated with primary model: {stream.content[:100]}...")
    
    # Example 2: Mature content (will automatically use uncensored model)
    print("\n=== Mature Content Example ===")
    async for stream in book_agent.handle_content_request_with_context(
        "Write a dark psychological thriller scene with complex moral dilemmas",
        content_type="dark_fiction"
    ):
        if stream.type in ['thinking', 'observation', 'result']:
            print(f"[{stream.type.upper()}] {stream.content[:100]}...")
    
    # Example 3: Force uncensored for any content
    print("\n=== Forced Uncensored Example ===")
    async for stream in book_agent.write_book_chapter_enhanced(
        book_id="test_book",
        chapter_outline="A chapter about overcoming personal demons",
        force_uncensored=True
    ):
        if stream.type in ['action', 'result']:
            print(f"[{stream.type.upper()}] {stream.content[:100]}...")
    
    # Get statistics
    stats = await book_agent.get_content_statistics()
    print(f"\nContent Statistics: {stats}")


# Command-line interface enhancements
class EnhancedCLI:
    """Enhanced CLI with content fallback controls"""
    
    def __init__(self, book_agent: EnhancedBookWritingAgent):
        self.book_agent = book_agent
    
    async def interactive_writing(self):
        """Interactive writing session with content controls"""
        
        print("Enhanced Book Writing CLI")
        print("Commands:")
        print("  /uncensored - Force uncensored model for next request")
        print("  /stats - Show content generation statistics")
        print("  /categories - Show detected content categories")
        print("  /quit - Exit")
        print()
        
        force_next_uncensored = False
        
        while True:
            user_input = input("\nWhat would you like me to write? ").strip()
            
            if user_input == "/quit":
                break
            elif user_input == "/uncensored":
                force_next_uncensored = True
                print("Next request will use uncensored model")
                continue
            elif user_input == "/stats":
                stats = await self.book_agent.get_content_statistics()
                print(f"Content Statistics: {json.dumps(stats, indent=2)}")
                continue
            elif user_input == "/categories":
                categories = list(self.book_agent.content_categories.keys())
                print(f"Available content categories: {categories}")
                continue
            
            if not user_input:
                continue
            
            print("\nGenerating content...")
            
            # Use enhanced generation
            user_prefs = {'creative_freedom_level': 'high'} if force_next_uncensored else None
            
            async for stream in self.book_agent.handle_content_request_with_context(
                user_input,
                user_preferences=user_prefs
            ):
                if stream.type == 'thinking':
                    print(f"💭 {stream.content}")
                elif stream.type == 'observation':
                    print(f"👁️ {stream.content}")
                elif stream.type == 'result':
                    print(f"\n📝 Generated Content:\n{stream.content}\n")
            
            force_next_uncensored = False


# Test function
async def test_enhanced_book_agent():
    """Test the enhanced book writing agent"""
    
    try:
        await example_enhanced_writing()
        print("✅ Enhanced book writing agent test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_book_agent())