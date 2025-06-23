"""
Enhanced JSON parsing utilities for LLM responses
Addresses the JSON parsing issues in the LLM evaluation system
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

class RobustJSONParser:
    """Enhanced JSON parser for LLM responses with multiple fallback strategies"""
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Extract JSON from LLM response using multiple strategies
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed JSON object/array or None if parsing fails
        """
        if not response or not response.strip():
            logger.warning("Empty response provided to JSON parser")
            return None
        
        # Strategy 1: Try direct JSON parsing (for clean responses)
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code blocks
        json_from_markdown = RobustJSONParser._extract_from_markdown(response)
        if json_from_markdown is not None:
            return json_from_markdown
        
        # Strategy 3: Find JSON boundaries with bracket/brace matching
        json_from_boundaries = RobustJSONParser._extract_with_boundary_matching(response)
        if json_from_boundaries is not None:
            return json_from_boundaries
        
        # Strategy 4: Regex-based extraction with multiple patterns
        json_from_regex = RobustJSONParser._extract_with_regex_patterns(response)
        if json_from_regex is not None:
            return json_from_regex
        
        # Strategy 5: Clean and retry with common fixes
        json_from_cleaning = RobustJSONParser._extract_with_cleaning(response)
        if json_from_cleaning is not None:
            return json_from_cleaning
        
        logger.error(f"Failed to extract JSON from response: {response[:200]}...")
        return None
    
    @staticmethod
    def _extract_from_markdown(response: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Extract JSON from markdown code blocks"""
        try:
            # Look for ```json ... ``` blocks
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response, re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # Look for ``` ... ``` blocks (without json specifier)
            code_match = re.search(r'```\s*([\s\S]*?)\s*```', response)
            if code_match:
                json_str = code_match.group(1).strip()
                # Try to parse as JSON
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.debug(f"Markdown extraction failed: {str(e)}")
        
        return None
    
    @staticmethod
    def _extract_with_boundary_matching(response: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Extract JSON using bracket/brace boundary matching"""
        try:
            # Find all potential JSON start positions
            start_positions = []
            for i, char in enumerate(response):
                if char in ['{', '[']:
                    start_positions.append((i, char))
            
            # Try each start position
            for start_pos, start_char in start_positions:
                end_char = '}' if start_char == '{' else ']'
                
                # Find matching end position
                level = 0
                for i in range(start_pos, len(response)):
                    if response[i] == start_char:
                        level += 1
                    elif response[i] == end_char:
                        level -= 1
                        if level == 0:
                            # Found complete JSON structure
                            json_str = response[start_pos:i+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                continue
                            break
        
        except Exception as e:
            logger.debug(f"Boundary matching failed: {str(e)}")
        
        return None
    
    @staticmethod
    def _extract_with_regex_patterns(response: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Extract JSON using multiple regex patterns"""
        patterns = [
            # Array patterns
            r'\[\s*\{[\s\S]*?\}\s*\]',  # Array of objects
            r'\[\s*"[\s\S]*?"\s*\]',    # Array of strings
            r'\[\s*[\d\s,]*\s*\]',      # Array of numbers
            
            # Object patterns
            r'\{\s*"[\s\S]*?\}\s*\}',   # Nested objects
            r'\{\s*"[^"]*"\s*:\s*"[^"]*"[\s\S]*?\}',  # Simple objects
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, response)
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.debug(f"Regex pattern {pattern} failed: {str(e)}")
                continue
        
        return None
    
    @staticmethod
    def _extract_with_cleaning(response: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Extract JSON after applying common cleaning operations"""
        try:
            # Remove common prefixes/suffixes
            cleaned = response.strip()
            
            # Remove thinking tags and reasoning content (for thinking models)
            thinking_patterns = [
                r'<think>.*?</think>',
                r'<thinking>.*?</thinking>',
                r'<reasoning>.*?</reasoning>',
                r'<analysis>.*?</analysis>',
            ]
            
            for pattern in thinking_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove common LLM response patterns
            patterns_to_remove = [
                r'^Here is the JSON:?\s*',
                r'^The JSON is:?\s*',
                r'^JSON:?\s*',
                r'^Response:?\s*',
                r'^Output:?\s*',
                r'^Okay,.*?JSON.*?:?\s*',  # Handle "Okay, let's tackle this..." patterns
                r'^Looking.*?document.*?:?\s*',  # Handle analysis text
                r'\s*That\'s the JSON\.?$',
                r'\s*Hope this helps\.?$',
            ]
            
            for pattern in patterns_to_remove:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Try parsing the cleaned response
            try:
                return json.loads(cleaned.strip())
            except json.JSONDecodeError:
                pass
            
            # Try finding JSON after removing everything before first { or [
            for start_char in ['{', '[']:
                start_idx = cleaned.find(start_char)
                if start_idx != -1:
                    try:
                        return json.loads(cleaned[start_idx:])
                    except json.JSONDecodeError:
                        continue
            
            # Try finding JSON before last } or ]
            for end_char in ['}', ']']:
                end_idx = cleaned.rfind(end_char)
                if end_idx != -1:
                    try:
                        return json.loads(cleaned[:end_idx+1])
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            logger.debug(f"Cleaning extraction failed: {str(e)}")
        
        return None
    
    @staticmethod
    def validate_extracted_facts(data: Union[Dict, List]) -> bool:
        """Validate that extracted data matches expected fact structure"""
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not isinstance(item, dict):
                return False
            
            required_fields = ['content', 'context', 'fact_type', 'confidence']
            if not all(field in item for field in required_fields):
                return False
            
            # Validate confidence field
            confidence = item.get('confidence')
            if isinstance(confidence, str):
                if confidence.lower() not in ['high', 'medium', 'low']:
                    return False
            elif isinstance(confidence, (int, float)):
                if not 0 <= confidence <= 1:
                    return False
        
        return True
    
    @staticmethod
    def validate_extracted_concepts(data: Union[Dict, List]) -> bool:
        """Validate that extracted data matches expected concept structure"""
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not isinstance(item, dict):
                return False
            
            required_fields = ['name', 'definition', 'domain', 'confidence']
            if not all(field in item for field in required_fields):
                return False
            
            # Validate optional fields
            if 'examples' in item and not isinstance(item['examples'], list):
                return False
            
            if 'relationships' in item and not isinstance(item['relationships'], list):
                return False
        
        return True
