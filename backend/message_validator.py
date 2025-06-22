"""
Message validation utilities for Ollama API integration
Prevents "list index out of range" errors in litellm transformations
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MessageValidator:
    """Validates and sanitizes messages for Ollama API compatibility"""
    
    @staticmethod
    def validate_messages_for_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and sanitize messages for Ollama API
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of sanitized message dictionaries
        """
        if not messages or len(messages) == 0:
            logger.warning("Empty messages array, using default message")
            return [{"role": "user", "content": "Hello, please respond."}]
        
        sanitized_messages = []
        
        for i, msg in enumerate(messages):
            try:
                if not isinstance(msg, dict):
                    logger.warning(f"Message at index {i} is not a dictionary, skipping")
                    continue
                
                if "role" not in msg or "content" not in msg:
                    logger.warning(f"Message at index {i} missing required fields (role/content), skipping")
                    continue
                
                # Create clean message without tool_calls or other unsupported fields
                clean_msg = {
                    "role": msg["role"],
                    "content": str(msg["content"]) if msg["content"] is not None else ""
                }
                
                # Validate role
                if clean_msg["role"] not in ["user", "assistant", "system"]:
                    logger.warning(f"Invalid role '{clean_msg['role']}' at index {i}, defaulting to 'user'")
                    clean_msg["role"] = "user"
                
                # Ensure content is not empty
                if not clean_msg["content"].strip():
                    logger.warning(f"Empty content at index {i}, using placeholder")
                    clean_msg["content"] = "Please respond."
                
                sanitized_messages.append(clean_msg)
                
            except Exception as e:
                logger.error(f"Error processing message at index {i}: {str(e)}")
                continue
        
        # Ensure we have at least one valid message
        if not sanitized_messages:
            logger.warning("No valid messages after sanitization, using default")
            return [{"role": "user", "content": "Hello, please respond."}]
        
        # Ensure the conversation starts with a user message
        if sanitized_messages[0]["role"] != "user":
            logger.info("Adding user message at the beginning")
            sanitized_messages.insert(0, {"role": "user", "content": "Hello"})
        
        logger.info(f"Sanitized {len(messages)} messages to {len(sanitized_messages)} valid messages")
        return sanitized_messages
    
    @staticmethod
    def create_simple_message(content: str, role: str = "user") -> List[Dict[str, Any]]:
        """
        Create a simple message array for basic interactions
        
        Args:
            content: Message content
            role: Message role (user, assistant, system)
            
        Returns:
            List containing a single message dictionary
        """
        return [{
            "role": role if role in ["user", "assistant", "system"] else "user",
            "content": str(content) if content else "Hello, please respond."
        }]
    
    @staticmethod
    def validate_single_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate a single message dictionary
        
        Args:
            message: Message dictionary to validate
            
        Returns:
            Validated message dictionary or None if invalid
        """
        try:
            if not isinstance(message, dict):
                return None
            
            if "role" not in message or "content" not in message:
                return None
            
            return {
                "role": message["role"] if message["role"] in ["user", "assistant", "system"] else "user",
                "content": str(message["content"]) if message["content"] is not None else "Please respond."
            }
        except Exception as e:
            logger.error(f"Error validating message: {str(e)}")
            return None
    
    @staticmethod
    def ensure_conversation_flow(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure proper conversation flow (alternating user/assistant messages)
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of messages with proper conversation flow
        """
        if not messages:
            return [{"role": "user", "content": "Hello, please respond."}]
        
        fixed_messages = []
        last_role = None
        
        for msg in messages:
            current_role = msg.get("role", "user")
            
            # Avoid consecutive messages from the same role
            if last_role == current_role:
                if current_role == "user":
                    # Add a simple assistant response
                    fixed_messages.append({
                        "role": "assistant", 
                        "content": "I understand. Please continue."
                    })
                else:
                    # Add a simple user prompt
                    fixed_messages.append({
                        "role": "user", 
                        "content": "Please continue."
                    })
            
            fixed_messages.append(msg)
            last_role = current_role
        
        return fixed_messages
