"""
Narrative analysis module for book writing
Analyzes story structures, character arcs, and narrative patterns
"""

import logging
from typing import Dict, Any, List, Optional


class NarrativeAnalyzer:
    """Analyzes narrative structures and patterns in writing"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Define narrative pattern templates
        self.narrative_patterns = {
            "hero_journey": self.analyze_hero_journey,
            "three_act": self.analyze_three_act_structure,
            "spiral": self.analyze_spiral_narrative,
            "kishoten": self.analyze_kishoten_structure,
            "five_act": self.analyze_five_act_structure
        }
    
    async def analyze_narrative_structure(self, content: str, pattern_type: str = "auto") -> Dict[str, Any]:
        """
        Analyze the narrative structure of given content
        
        Args:
            content: Text content to analyze
            pattern_type: Type of narrative pattern to analyze for, or "auto" to detect
            
        Returns:
            Analysis results with identified patterns and elements
        """
        if pattern_type == "auto":
            # Detect the most likely narrative pattern
            pattern_type = await self._detect_narrative_pattern(content)
        
        if pattern_type in self.narrative_patterns:
            return await self.narrative_patterns[pattern_type](content)
        else:
            return await self._generic_narrative_analysis(content)
    
    async def analyze_hero_journey(self, content: str) -> Dict[str, Any]:
        """Analyze content for Hero's Journey structure (Campbell's monomyth)"""
        stages = [
            "ordinary_world",
            "call_to_adventure",
            "refusal_of_call",
            "meeting_mentor",
            "crossing_threshold",
            "tests_allies_enemies",
            "approach_innermost_cave",
            "ordeal",
            "reward",
            "road_back",
            "resurrection",
            "return_with_elixir"
        ]
        
        analysis = {
            "pattern": "hero_journey",
            "stages_identified": [],
            "missing_stages": [],
            "strength": 0.0,
            "recommendations": []
        }
        
        # Analyze which stages are present
        content_lower = content.lower()
        for stage in stages:
            # Simple keyword matching - in a real implementation this would use NLP
            stage_keywords = self._get_stage_keywords(stage)
            if any(keyword in content_lower for keyword in stage_keywords):
                analysis["stages_identified"].append(stage)
            else:
                analysis["missing_stages"].append(stage)
        
        # Calculate pattern strength
        analysis["strength"] = len(analysis["stages_identified"]) / len(stages)
        
        # Generate recommendations
        if analysis["strength"] < 0.5:
            analysis["recommendations"].append(
                "Consider strengthening the hero's journey arc by developing missing stages"
            )
        
        return analysis
    
    async def analyze_three_act_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content for three-act structure"""
        acts = {
            "setup": {
                "elements": ["exposition", "inciting_incident", "first_plot_point"],
                "proportion": 0.25
            },
            "confrontation": {
                "elements": ["rising_action", "midpoint", "complications"],
                "proportion": 0.50
            },
            "resolution": {
                "elements": ["climax", "falling_action", "denouement"],
                "proportion": 0.25
            }
        }
        
        analysis = {
            "pattern": "three_act",
            "acts_identified": {},
            "pacing_analysis": {},
            "strength": 0.0,
            "recommendations": []
        }
        
        # Analyze each act
        for act_name, act_data in acts.items():
            act_analysis = {
                "elements_found": [],
                "elements_missing": [],
                "proportion_estimate": 0.0
            }
            
            # Check for act elements (simplified version)
            content_lower = content.lower()
            for element in act_data["elements"]:
                element_keywords = self._get_element_keywords(element)
                if any(keyword in content_lower for keyword in element_keywords):
                    act_analysis["elements_found"].append(element)
                else:
                    act_analysis["elements_missing"].append(element)
            
            analysis["acts_identified"][act_name] = act_analysis
        
        # Calculate overall strength
        total_elements = sum(len(acts[act]["elements"]) for act in acts)
        found_elements = sum(
            len(analysis["acts_identified"][act]["elements_found"]) 
            for act in acts
        )
        analysis["strength"] = found_elements / total_elements if total_elements > 0 else 0
        
        # Pacing recommendations
        if analysis["strength"] < 0.7:
            analysis["recommendations"].append(
                "Strengthen act structure by ensuring all key elements are present"
            )
        
        return analysis
    
    async def analyze_spiral_narrative(self, content: str) -> Dict[str, Any]:
        """Analyze content for spiral/circular narrative structure"""
        analysis = {
            "pattern": "spiral",
            "recurring_themes": [],
            "circular_elements": [],
            "depth_progression": 0.0,
            "strength": 0.0,
            "recommendations": []
        }
        
        # Identify recurring themes and motifs
        # This is a simplified analysis - real implementation would use NLP
        themes = ["beginning", "return", "cycle", "repeat", "mirror", "echo"]
        content_lower = content.lower()
        
        for theme in themes:
            if content_lower.count(theme) > 1:
                analysis["recurring_themes"].append(theme)
        
        # Check for circular structure markers
        if content_lower[:100] in content_lower[-200:]:
            analysis["circular_elements"].append("ending_mirrors_beginning")
        
        # Calculate pattern strength
        if analysis["recurring_themes"] or analysis["circular_elements"]:
            analysis["strength"] = min(
                (len(analysis["recurring_themes"]) + len(analysis["circular_elements"])) / 5,
                1.0
            )
        
        # Recommendations
        if analysis["strength"] < 0.5:
            analysis["recommendations"].append(
                "Consider adding more recurring motifs or circular references"
            )
        
        return analysis
    
    async def analyze_kishoten_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content for Kishōtenketsu (4-act Asian narrative structure)"""
        acts = {
            "ki": "introduction",
            "sho": "development", 
            "ten": "twist",
            "ketsu": "conclusion"
        }
        
        analysis = {
            "pattern": "kishoten",
            "acts_identified": {},
            "twist_strength": 0.0,
            "harmony_score": 0.0,
            "recommendations": []
        }
        
        # Simplified analysis for each act
        for act_code, act_name in acts.items():
            analysis["acts_identified"][act_code] = {
                "present": False,
                "description": act_name
            }
        
        # Check for twist/turn (ten) - key element
        twist_keywords = ["however", "but then", "suddenly", "unexpectedly", "twist"]
        content_lower = content.lower()
        twist_count = sum(1 for keyword in twist_keywords if keyword in content_lower)
        
        if twist_count > 0:
            analysis["acts_identified"]["ten"]["present"] = True
            analysis["twist_strength"] = min(twist_count / 3, 1.0)
        
        return analysis
    
    async def analyze_five_act_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content for five-act dramatic structure (Freytag's pyramid)"""
        acts = [
            "exposition",
            "rising_action",
            "climax",
            "falling_action",
            "denouement"
        ]
        
        analysis = {
            "pattern": "five_act",
            "acts_identified": [],
            "dramatic_arc": 0.0,
            "tension_progression": [],
            "recommendations": []
        }
        
        # Simplified act identification
        for act in acts:
            act_keywords = self._get_act_keywords(act)
            if any(keyword in content.lower() for keyword in act_keywords):
                analysis["acts_identified"].append(act)
        
        # Calculate dramatic arc strength
        analysis["dramatic_arc"] = len(analysis["acts_identified"]) / len(acts)
        
        return analysis
    
    async def _detect_narrative_pattern(self, content: str) -> str:
        """Detect the most likely narrative pattern in the content"""
        # Simple heuristic-based detection
        content_lower = content.lower()
        
        # Check for hero's journey markers
        hero_markers = ["hero", "journey", "mentor", "threshold", "ordeal", "return"]
        hero_score = sum(1 for marker in hero_markers if marker in content_lower)
        
        # Check for three-act markers
        three_act_markers = ["setup", "confrontation", "resolution", "climax", "inciting"]
        three_act_score = sum(1 for marker in three_act_markers if marker in content_lower)
        
        # Return the pattern with highest score
        scores = {
            "hero_journey": hero_score,
            "three_act": three_act_score,
            "spiral": 1 if "return" in content_lower and "beginning" in content_lower else 0
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "three_act"
    
    async def _generic_narrative_analysis(self, content: str) -> Dict[str, Any]:
        """Perform generic narrative analysis when specific pattern unknown"""
        return {
            "pattern": "generic",
            "elements_found": {
                "has_beginning": bool(content[:100]),
                "has_middle": len(content) > 500,
                "has_end": bool(content[-100:]),
                "word_count": len(content.split())
            },
            "recommendations": ["Consider using a specific narrative structure"]
        }
    
    def _get_stage_keywords(self, stage: str) -> List[str]:
        """Get keywords associated with a hero's journey stage"""
        keywords_map = {
            "ordinary_world": ["normal", "everyday", "routine", "before"],
            "call_to_adventure": ["call", "summon", "quest", "mission"],
            "meeting_mentor": ["mentor", "guide", "teacher", "wise"],
            "crossing_threshold": ["threshold", "enter", "begin", "cross"],
            "ordeal": ["ordeal", "crisis", "death", "fear"],
            "return_with_elixir": ["return", "home", "changed", "wisdom"]
        }
        return keywords_map.get(stage, [stage.replace("_", " ")])
    
    def _get_element_keywords(self, element: str) -> List[str]:
        """Get keywords for three-act structure elements"""
        keywords_map = {
            "exposition": ["introduce", "establish", "setting", "character"],
            "inciting_incident": ["catalyst", "trigger", "problem", "conflict"],
            "rising_action": ["tension", "escalate", "complicate", "develop"],
            "climax": ["climax", "peak", "turning point", "confrontation"],
            "denouement": ["resolve", "conclude", "ending", "aftermath"]
        }
        return keywords_map.get(element, [element.replace("_", " ")])
    
    def _get_act_keywords(self, act: str) -> List[str]:
        """Get keywords for five-act structure"""
        return self._get_element_keywords(act)  # Reuse element keywords
    
    async def analyze_character_arcs(self, content: str, characters: List[str]) -> Dict[str, Any]:
        """Analyze character development arcs in the narrative"""
        arcs = {}
        
        for character in characters:
            arc_analysis = {
                "character": character,
                "appearances": content.lower().count(character.lower()),
                "development_indicators": [],
                "arc_type": "unknown",
                "strength": 0.0
            }
            
            # Check for development indicators
            dev_keywords = ["changed", "learned", "realized", "grew", "transformed"]
            char_sentences = [s for s in content.split('.') if character.lower() in s.lower()]
            
            for sentence in char_sentences:
                for keyword in dev_keywords:
                    if keyword in sentence.lower():
                        arc_analysis["development_indicators"].append(keyword)
            
            # Determine arc type
            if arc_analysis["development_indicators"]:
                arc_analysis["arc_type"] = "dynamic"
                arc_analysis["strength"] = min(len(arc_analysis["development_indicators"]) / 5, 1.0)
            else:
                arc_analysis["arc_type"] = "static"
            
            arcs[character] = arc_analysis
        
        return arcs