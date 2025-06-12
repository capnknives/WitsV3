"""
NLP Tools for Neural Web Concept Extraction

Provides specialized Natural Language Processing tools for extracting concepts,
relationships, and domain knowledge from text for the Neural Web system.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import json

from core.base_tool import BaseTool, ToolResult
from core.config import WitsV3Config
from core.neural_web_core import NeuralWeb
from core.llm_interface import BaseLLMInterface, LLMMessage

logger = logging.getLogger(__name__)


@dataclass
class ExtractedConcept:
    """Represents a concept extracted from text."""
    text: str
    concept_type: str
    confidence: float
    context: str
    domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExtractedRelationship:
    """Represents a relationship extracted from text."""
    source_concept: str
    target_concept: str
    relationship_type: str
    confidence: float
    context: str
    strength: float = 1.0


@dataclass
class DomainClassification:
    """Represents domain classification results."""
    domain: str
    confidence: float
    supporting_concepts: List[str]
    reasoning: str


class ConceptExtractor:
    """Advanced concept extraction using NLP and LLM techniques."""

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        self.config = config
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)

        # Pattern-based concept indicators
        self.concept_patterns = {
            'technology': [
                r'\b(?:technology|software|algorithm|system|platform|framework|tool|application)\b',
                r'\b(?:AI|ML|machine learning|artificial intelligence|neural network|deep learning)\b',
                r'\b(?:programming|coding|development|engineering|architecture)\b'
            ],
            'science': [
                r'\b(?:theory|hypothesis|experiment|research|study|analysis|discovery)\b',
                r'\b(?:physics|chemistry|biology|mathematics|psychology|neuroscience)\b',
                r'\b(?:scientific|empirical|evidence|data|observation|measurement)\b'
            ],
            'business': [
                r'\b(?:strategy|market|customer|revenue|profit|growth|investment)\b',
                r'\b(?:management|leadership|organization|company|business|enterprise)\b',
                r'\b(?:sales|marketing|finance|operations|HR|human resources)\b'
            ],
            'philosophy': [
                r'\b(?:ethics|morality|consciousness|existence|reality|truth|knowledge)\b',
                r'\b(?:philosophy|philosophical|metaphysics|epistemology|ontology)\b',
                r'\b(?:wisdom|virtue|justice|freedom|democracy|rights|responsibility)\b'
            ],
            'arts': [
                r'\b(?:art|artistic|creativity|creative|design|aesthetic|beauty)\b',
                r'\b(?:music|painting|sculpture|literature|poetry|theater|dance)\b',
                r'\b(?:expression|inspiration|imagination|innovation|culture)\b'
            ]
        }

        # Relationship pattern indicators
        self.relationship_patterns = {
            'causes': [
                r'\b(?:causes?|leads? to|results? in|produces?|generates?|creates?)\b',
                r'\b(?:because of|due to|as a result of|consequently)\b'
            ],
            'enables': [
                r'\b(?:enables?|allows?|facilitates?|supports?|helps?|assists?)\b',
                r'\b(?:makes possible|empowers?|provides? the means)\b'
            ],
            'includes': [
                r'\b(?:includes?|contains?|encompasses?|comprises?|consists? of)\b',
                r'\b(?:is part of|belongs to|is a type of|is a kind of)\b'
            ],
            'relates_to': [
                r'\b(?:relates? to|connected to|associated with|linked to)\b',
                r'\b(?:similar to|comparable to|analogous to|corresponding to)\b'
            ],
            'contradicts': [
                r'\b(?:contradicts?|opposes?|conflicts? with|disagrees? with)\b',
                r'\b(?:contrary to|opposite of|against|versus|but|however)\b'
            ]
        }

    async def extract_concepts_from_text(self, text: str,
                                       domain_hint: Optional[str] = None) -> List[ExtractedConcept]:
        """
        Extract concepts from text using hybrid NLP and LLM approach.

        Args:
            text: Input text to analyze
            domain_hint: Optional domain hint to guide extraction

        Returns:
            List of extracted concepts
        """
        try:
            # Pattern-based extraction
            pattern_concepts = self._extract_concepts_by_patterns(text, domain_hint)

            # LLM-based extraction
            llm_concepts = await self._extract_concepts_with_llm(text, domain_hint)

            # Merge and deduplicate concepts
            merged_concepts = self._merge_concept_lists(pattern_concepts, llm_concepts)

            # Filter and rank by confidence
            filtered_concepts = [c for c in merged_concepts if c.confidence > 0.3]
            filtered_concepts.sort(key=lambda x: x.confidence, reverse=True)

            self.logger.info(f"Extracted {len(filtered_concepts)} concepts from text")
            return filtered_concepts[:20]  # Limit to top 20 concepts

        except Exception as e:
            self.logger.error(f"Error extracting concepts: {e}")
            return []

    def _extract_concepts_by_patterns(self, text: str,
                                    domain_hint: Optional[str] = None) -> List[ExtractedConcept]:
        """Extract concepts using pattern matching."""
        concepts = []
        text_lower = text.lower()

        # Extract noun phrases (simplified)
        noun_phrases = self._extract_noun_phrases(text)

        for phrase in noun_phrases:
            if len(phrase.split()) > 1 and len(phrase) > 3:
                # Determine concept type and domain
                concept_type = self._classify_concept_type(phrase, text_lower)
                domain = self._classify_domain_by_patterns(phrase, text_lower, domain_hint)

                # Calculate confidence based on context and patterns
                confidence = self._calculate_pattern_confidence(phrase, text_lower, domain)

                concepts.append(ExtractedConcept(
                    text=phrase,
                    concept_type=concept_type,
                    confidence=confidence,
                    context=self._extract_context(phrase, text),
                    domain=domain,
                    metadata={'extraction_method': 'pattern'}
                ))

        return concepts

    async def _extract_concepts_with_llm(self, text: str,
                                       domain_hint: Optional[str] = None) -> List[ExtractedConcept]:
        """Extract concepts using LLM analysis."""
        try:
            domain_instruction = f"Focus on the {domain_hint} domain. " if domain_hint else ""

            prompt = f"""
            {domain_instruction}Analyze the following text and extract key concepts, ideas, and important terms.

            For each concept, provide:
            1. The concept text (phrase or term)
            2. Concept type (technology, process, entity, principle, etc.)
            3. Confidence score (0.0-1.0)
            4. Domain classification (if applicable)

            Text to analyze:
            {text}

            Return your analysis as a JSON array with this structure:
            [
                {{
                    "text": "concept name",
                    "concept_type": "technology|process|entity|principle|other",
                    "confidence": 0.0-1.0,
                    "domain": "domain name or null",
                    "reasoning": "brief explanation"
                }}
            ]

            Focus on concepts that are meaningful, specific, and would be valuable in a knowledge network.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)

            # Parse LLM response
            concepts = self._parse_llm_concept_response(response.content, text)
            return concepts

        except Exception as e:
            self.logger.error(f"Error with LLM concept extraction: {e}")
            return []

    def _parse_llm_concept_response(self, response_text: str, original_text: str) -> List[ExtractedConcept]:
        """Parse LLM response into ExtractedConcept objects."""
        concepts = []

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                concept_data = json.loads(json_text)

                for item in concept_data:
                    if isinstance(item, dict) and 'text' in item:
                        concepts.append(ExtractedConcept(
                            text=item.get('text', '').strip(),
                            concept_type=item.get('concept_type', 'other'),
                            confidence=float(item.get('confidence', 0.5)),
                            context=self._extract_context(item.get('text', ''), original_text),
                            domain=item.get('domain'),
                            metadata={
                                'extraction_method': 'llm',
                                'reasoning': item.get('reasoning', '')
                            }
                        ))

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Could not parse LLM concept response as JSON: {e}")
            # Fallback to text parsing
            concepts = self._parse_text_concept_response(response_text, original_text)

        return concepts

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using simple pattern matching."""
        # Simple noun phrase patterns
        patterns = [
            r'\b(?:[A-Z][a-z]+ )+[A-Z][a-z]+\b',  # Title Case Phrases
            r'\b[a-z]+ [a-z]+(?:ing|tion|ment|ness|ity|ism)\b',  # Noun suffixes
            r'\b(?:the |a |an )?[a-z]+ [a-z]+\b',  # Simple two-word phrases
        ]

        phrases = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.update([match.strip().lower() for match in matches])

        # Filter out common stop phrases
        stop_phrases = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_phrases = [p for p in phrases if not any(stop in p.split() for stop in stop_phrases)]

        return list(filtered_phrases)

    def _classify_concept_type(self, concept: str, text: str) -> str:
        """Classify the type of a concept."""
        concept_lower = concept.lower()

        # Technology indicators
        if any(word in concept_lower for word in ['system', 'software', 'algorithm', 'technology', 'tool', 'platform']):
            return 'technology'

        # Process indicators
        if any(word in concept_lower for word in ['process', 'method', 'approach', 'procedure', 'technique']):
            return 'process'

        # Entity indicators
        if any(word in concept_lower for word in ['organization', 'company', 'institution', 'group', 'team']):
            return 'entity'

        # Principle indicators
        if any(word in concept_lower for word in ['principle', 'law', 'rule', 'theory', 'concept', 'idea']):
            return 'principle'

        return 'other'

    def _classify_domain_by_patterns(self, concept: str, text: str, domain_hint: Optional[str] = None) -> Optional[str]:
        """Classify the domain of a concept using patterns."""
        if domain_hint:
            return domain_hint

        concept_text = f"{concept} {text}".lower()

        domain_scores = {}
        for domain, patterns in self.concept_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, concept_text, re.IGNORECASE))
                score += matches

            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            return best_domain if domain_scores[best_domain] > 0 else None

        return None

    def _calculate_pattern_confidence(self, concept: str, text: str, domain: Optional[str]) -> float:
        """Calculate confidence score for pattern-based extraction."""
        base_confidence = 0.5

        # Length bonus (longer phrases often more specific)
        length_bonus = min(0.2, len(concept.split()) * 0.05)

        # Domain match bonus
        domain_bonus = 0.2 if domain else 0.0

        # Frequency penalty (very common words get lower confidence)
        word_count = len(text.lower().split(concept.lower()))
        frequency_penalty = min(0.2, (word_count - 1) * 0.05)

        confidence = base_confidence + length_bonus + domain_bonus - frequency_penalty
        return max(0.0, min(1.0, confidence))

    def _extract_context(self, concept: str, text: str, window_size: int = 50) -> str:
        """Extract context around a concept in text."""
        concept_lower = concept.lower()
        text_lower = text.lower()

        # Find the concept in text
        index = text_lower.find(concept_lower)
        if index == -1:
            return ""

        # Extract context window
        start = max(0, index - window_size)
        end = min(len(text), index + len(concept) + window_size)

        context = text[start:end].strip()
        return context

    def _merge_concept_lists(self, list1: List[ExtractedConcept],
                           list2: List[ExtractedConcept]) -> List[ExtractedConcept]:
        """Merge and deduplicate concept lists."""
        merged = {}

        # Add all concepts, keeping highest confidence for duplicates
        for concept in list1 + list2:
            key = concept.text.lower()
            if key not in merged or concept.confidence > merged[key].confidence:
                merged[key] = concept

        return list(merged.values())

    def _parse_text_concept_response(self, response_text: str, original_text: str) -> List[ExtractedConcept]:
        """Fallback parser for non-JSON LLM responses."""
        concepts = []
        lines = response_text.split('\n')

        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 3:
                # Simple heuristic parsing
                confidence = 0.6  # Default confidence for text parsing
                concept_type = 'other'

                # Remove common prefixes
                for prefix in ['- ', '* ', '• ', '1. ', '2. ', '3. ']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break

                if line:
                    concepts.append(ExtractedConcept(
                        text=line,
                        concept_type=concept_type,
                        confidence=confidence,
                        context=self._extract_context(line, original_text),
                        metadata={'extraction_method': 'llm_text_parse'}
                    ))

        return concepts


class RelationshipExtractor:
    """Extracts relationships between concepts."""

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        self.config = config
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)

    async def extract_relationships(self, text: str,
                                  concepts: List[ExtractedConcept]) -> List[ExtractedRelationship]:
        """Extract relationships between concepts in text."""
        try:
            # Pattern-based relationship extraction
            pattern_relationships = self._extract_relationships_by_patterns(text, concepts)

            # LLM-based relationship extraction
            llm_relationships = await self._extract_relationships_with_llm(text, concepts)

            # Merge and filter relationships
            all_relationships = pattern_relationships + llm_relationships
            filtered_relationships = [r for r in all_relationships if r.confidence > 0.4]

            self.logger.info(f"Extracted {len(filtered_relationships)} relationships")
            return filtered_relationships

        except Exception as e:
            self.logger.error(f"Error extracting relationships: {e}")
            return []

    def _extract_relationships_by_patterns(self, text: str,
                                         concepts: List[ExtractedConcept]) -> List[ExtractedRelationship]:
        """Extract relationships using pattern matching."""
        relationships = []
        text_lower = text.lower()
        concept_texts = [c.text.lower() for c in concepts]

        for i, source_concept in enumerate(concepts):
            for j, target_concept in enumerate(concepts):
                if i >= j:  # Avoid duplicates and self-relationships
                    continue

                # Look for relationship patterns between concepts
                for rel_type, patterns in ConceptExtractor(self.config, self.llm_interface).relationship_patterns.items():
                    for pattern in patterns:
                        # Create search text with both concept orders
                        search_text1 = f"{source_concept.text.lower()}.*{pattern}.*{target_concept.text.lower()}"
                        search_text2 = f"{target_concept.text.lower()}.*{pattern}.*{source_concept.text.lower()}"

                        if re.search(search_text1, text_lower, re.DOTALL) or re.search(search_text2, text_lower, re.DOTALL):
                            confidence = 0.6  # Base confidence for pattern matching

                            relationships.append(ExtractedRelationship(
                                source_concept=source_concept.text,
                                target_concept=target_concept.text,
                                relationship_type=rel_type,
                                confidence=confidence,
                                context=self._extract_relationship_context(
                                    source_concept.text, target_concept.text, text),
                                strength=confidence
                            ))
                            break  # Found a relationship, don't check other patterns

        return relationships

    async def _extract_relationships_with_llm(self, text: str,
                                            concepts: List[ExtractedConcept]) -> List[ExtractedRelationship]:
        """Extract relationships using LLM analysis."""
        try:
            concept_list = [c.text for c in concepts[:10]]  # Limit to avoid token overflow

            prompt = f"""
            Analyze the relationships between these concepts in the given text:

            Concepts: {', '.join(concept_list)}

            Text:
            {text}

            Identify relationships between the concepts. For each relationship, provide:
            1. Source concept
            2. Target concept
            3. Relationship type (causes, enables, includes, relates_to, contradicts, or other)
            4. Confidence score (0.0-1.0)
            5. Brief explanation

            Return as JSON array:
            [
                {{
                    "source": "concept1",
                    "target": "concept2",
                    "relationship": "causes|enables|includes|relates_to|contradicts|other",
                    "confidence": 0.0-1.0,
                    "explanation": "brief explanation"
                }}
            ]
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)

            relationships = self._parse_llm_relationship_response(response.content, text)
            return relationships

        except Exception as e:
            self.logger.error(f"Error with LLM relationship extraction: {e}")
            return []

    def _parse_llm_relationship_response(self, response_text: str, original_text: str) -> List[ExtractedRelationship]:
        """Parse LLM response into ExtractedRelationship objects."""
        relationships = []

        try:
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                relationship_data = json.loads(json_text)

                for item in relationship_data:
                    if isinstance(item, dict) and all(k in item for k in ['source', 'target', 'relationship']):
                        relationships.append(ExtractedRelationship(
                            source_concept=item['source'].strip(),
                            target_concept=item['target'].strip(),
                            relationship_type=item['relationship'],
                            confidence=float(item.get('confidence', 0.5)),
                            context=self._extract_relationship_context(
                                item['source'], item['target'], original_text),
                            strength=float(item.get('confidence', 0.5))
                        ))

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Could not parse LLM relationship response as JSON: {e}")

        return relationships

    def _extract_relationship_context(self, source: str, target: str, text: str, window_size: int = 100) -> str:
        """Extract context around a relationship in text."""
        source_lower = source.lower()
        target_lower = target.lower()
        text_lower = text.lower()

        # Find both concepts in text
        source_index = text_lower.find(source_lower)
        target_index = text_lower.find(target_lower)

        if source_index == -1 or target_index == -1:
            return ""

        # Get the span covering both concepts plus context
        start_index = min(source_index, target_index)
        end_index = max(source_index + len(source), target_index + len(target))

        context_start = max(0, start_index - window_size)
        context_end = min(len(text), end_index + window_size)

        context = text[context_start:context_end].strip()
        return context


class NeuralWebNLPTool(BaseTool):
    """Tool for NLP-based concept and relationship extraction for Neural Web."""

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        super().__init__(config)
        self.llm_interface = llm_interface
        self.name = "neural_web_nlp_extract"
        self.description = "Extract concepts and relationships from text for Neural Web integration"

        self.concept_extractor = ConceptExtractor(config, llm_interface)
        self.relationship_extractor = RelationshipExtractor(config, llm_interface)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze for concepts and relationships"
                        },
                        "domain_hint": {
                            "type": "string",
                            "description": "Optional domain hint to guide extraction (e.g., 'technology', 'science')"
                        },
                        "extract_relationships": {
                            "type": "boolean",
                            "description": "Whether to extract relationships between concepts"
                        },
                        "add_to_neural_web": {
                            "type": "boolean",
                            "description": "Whether to add extracted concepts to the Neural Web"
                        }
                    },
                    "required": ["text"]
                }
            }
        }

    async def execute(self, **kwargs) -> ToolResult:
        try:
            text = kwargs.get("text", "")
            domain_hint = kwargs.get("domain_hint")
            extract_relationships = kwargs.get("extract_relationships", True)
            add_to_neural_web = kwargs.get("add_to_neural_web", False)

            if not text.strip():
                return ToolResult(
                    success=False,
                    error="No text provided for analysis"
                )

            # Extract concepts
            concepts = await self.concept_extractor.extract_concepts_from_text(text, domain_hint)

            # Extract relationships if requested
            relationships = []
            if extract_relationships and concepts:
                relationships = await self.relationship_extractor.extract_relationships(text, concepts)

            # Add to Neural Web if requested
            if add_to_neural_web:
                await self._add_to_neural_web(concepts, relationships)

            # Prepare results
            result_data = {
                "concepts": [
                    {
                        "text": c.text,
                        "type": c.concept_type,
                        "confidence": round(c.confidence, 3),
                        "domain": c.domain,
                        "context": c.context[:100] + "..." if len(c.context) > 100 else c.context
                    }
                    for c in concepts
                ],
                "relationships": [
                    {
                        "source": r.source_concept,
                        "target": r.target_concept,
                        "type": r.relationship_type,
                        "confidence": round(r.confidence, 3),
                        "strength": round(r.strength, 3)
                    }
                    for r in relationships
                ],
                "statistics": {
                    "concept_count": len(concepts),
                    "relationship_count": len(relationships),
                    "domains_found": len(set(c.domain for c in concepts if c.domain)),
                    "avg_concept_confidence": round(sum(c.confidence for c in concepts) / len(concepts), 3) if concepts else 0
                }
            }

            return ToolResult(
                success=True,
                result=f"Extracted {len(concepts)} concepts and {len(relationships)} relationships from text",
                metadata=result_data
            )

        except Exception as e:
            logger.error(f"Error in NLP extraction: {e}")
            return ToolResult(
                success=False,
                error=f"Error during NLP extraction: {str(e)}"
            )

    async def _add_to_neural_web(self, concepts: List[ExtractedConcept],
                               relationships: List[ExtractedRelationship]):
        """Add extracted concepts and relationships to the Neural Web."""
        try:
            # For now, this is a placeholder
            # In production, this would integrate with the actual Neural Web instance
            logger.info(f"Would add {len(concepts)} concepts and {len(relationships)} relationships to Neural Web")

        except Exception as e:
            logger.error(f"Error adding to Neural Web: {e}")
