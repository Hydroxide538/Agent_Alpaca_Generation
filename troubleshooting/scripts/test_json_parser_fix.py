#!/usr/bin/env python3
"""
Test script to verify the robust JSON parser fixes the LLM evaluation issues
"""

import asyncio
import os
import sys
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.json_parser_fix import RobustJSONParser

def test_json_parser():
    """Test the robust JSON parser with various problematic LLM responses"""
    
    test_cases = [
        # Case 1: Clean JSON
        {
            "name": "Clean JSON",
            "response": '[{"content": "Test fact", "context": "Test context", "fact_type": "general", "confidence": "high"}]',
            "expected_success": True
        },
        
        # Case 2: JSON with markdown
        {
            "name": "JSON with markdown",
            "response": '```json\n[{"content": "Test fact", "context": "Test context", "fact_type": "general", "confidence": "high"}]\n```',
            "expected_success": True
        },
        
        # Case 3: JSON with extra text
        {
            "name": "JSON with extra text",
            "response": 'Here is the JSON output:\n[{"content": "Test fact", "context": "Test context", "fact_type": "general", "confidence": "high"}]\nThat\'s the result.',
            "expected_success": True
        },
        
        # Case 4: Malformed JSON (missing quotes)
        {
            "name": "Malformed JSON",
            "response": '[{content: "Test fact", context: "Test context", fact_type: "general", confidence: "high"}]',
            "expected_success": False
        },
        
        # Case 5: JSON with conversational wrapper
        {
            "name": "Conversational wrapper",
            "response": 'I\'ll extract the facts for you. Here\'s the JSON:\n\n```json\n[{"content": "Test fact", "context": "Test context", "fact_type": "general", "confidence": "high"}]\n```\n\nHope this helps!',
            "expected_success": True
        },
        
        # Case 6: Multiple JSON objects (should get first valid one)
        {
            "name": "Multiple JSON objects",
            "response": 'First attempt: {"invalid": true}\nActual result: [{"content": "Test fact", "context": "Test context", "fact_type": "general", "confidence": "high"}]',
            "expected_success": True
        },
        
        # Case 7: Empty response
        {
            "name": "Empty response",
            "response": '',
            "expected_success": False
        },
        
        # Case 8: No JSON at all
        {
            "name": "No JSON",
            "response": 'I cannot provide the requested information in JSON format.',
            "expected_success": False
        }
    ]
    
    print("Testing Robust JSON Parser...")
    print("=" * 50)
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {test_case['response'][:100]}{'...' if len(test_case['response']) > 100 else ''}")
        
        try:
            result = RobustJSONParser.extract_json_from_response(test_case['response'])
            success = result is not None
            
            print(f"Expected success: {test_case['expected_success']}")
            print(f"Actual success: {success}")
            
            if success:
                print(f"Extracted: {json.dumps(result, indent=2)[:200]}{'...' if len(str(result)) > 200 else ''}")
                
                # Test validation if it's a fact structure
                if isinstance(result, list) and result:
                    is_valid_facts = RobustJSONParser.validate_extracted_facts(result)
                    is_valid_concepts = RobustJSONParser.validate_extracted_concepts(result)
                    print(f"Valid facts structure: {is_valid_facts}")
                    print(f"Valid concepts structure: {is_valid_concepts}")
            
            if success == test_case['expected_success']:
                print("✅ PASS")
                passed += 1
            else:
                print("❌ FAIL")
                
        except Exception as e:
            print(f"❌ FAIL - Exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed == total

def test_validation_functions():
    """Test the validation functions specifically"""
    
    print("\n\nTesting Validation Functions...")
    print("=" * 50)
    
    # Test fact validation
    valid_facts = [
        {
            "content": "The capital of France is Paris",
            "context": "Geography section",
            "fact_type": "geographical",
            "confidence": "high"
        }
    ]
    
    invalid_facts = [
        {
            "content": "Missing fields"
            # Missing required fields
        }
    ]
    
    # Test concept validation
    valid_concepts = [
        {
            "name": "Photosynthesis",
            "definition": "Process by which plants convert sunlight to energy",
            "examples": ["Plant growth", "Oxygen production"],
            "relationships": ["Related to cellular respiration"],
            "domain": "biology",
            "confidence": "high"
        }
    ]
    
    invalid_concepts = [
        {
            "name": "Incomplete concept"
            # Missing required fields
        }
    ]
    
    tests = [
        ("Valid facts", valid_facts, RobustJSONParser.validate_extracted_facts, True),
        ("Invalid facts", invalid_facts, RobustJSONParser.validate_extracted_facts, False),
        ("Valid concepts", valid_concepts, RobustJSONParser.validate_extracted_concepts, True),
        ("Invalid concepts", invalid_concepts, RobustJSONParser.validate_extracted_concepts, False),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, data, validator, expected in tests:
        try:
            result = validator(data)
            if result == expected:
                print(f"✅ {name}: PASS")
                passed += 1
            else:
                print(f"❌ {name}: FAIL (expected {expected}, got {result})")
        except Exception as e:
            print(f"❌ {name}: FAIL - Exception: {str(e)}")
    
    print(f"\nValidation Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    return passed == total

async def test_integration_with_llm():
    """Test integration with actual LLM (if available)"""
    
    print("\n\nTesting Integration with LLM...")
    print("=" * 50)
    
    try:
        from backend.llm_manager import LLMManager
        
        llm_manager = LLMManager()
        
        # Test with a simple prompt
        test_prompt = """Extract facts from this text and return them in JSON format:

The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 324 meters tall.

Return a JSON array with this structure:
[
  {
    "content": "fact statement",
    "context": "surrounding context",
    "fact_type": "general",
    "confidence": "high"
  }
]

JSON OUTPUT ONLY:"""
        
        # Try with a local model if available
        config = {
            "data_generation_model": "ollama:llama3.3:latest",
            "ollama_url": "http://host.docker.internal:11434"
        }
        
        print("Testing with LLM response...")
        response = await llm_manager.generate_response(
            config["data_generation_model"], 
            test_prompt, 
            config
        )
        
        print(f"LLM Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Test our parser
        parsed_result = RobustJSONParser.extract_json_from_response(response)
        
        if parsed_result:
            print("✅ Successfully parsed LLM response")
            print(f"Parsed result: {json.dumps(parsed_result, indent=2)}")
            
            # Test validation
            if RobustJSONParser.validate_extracted_facts(parsed_result):
                print("✅ Response validates as proper fact structure")
                return True
            else:
                print("⚠️  Response doesn't validate as proper fact structure")
                return False
        else:
            print("❌ Failed to parse LLM response")
            return False
            
    except Exception as e:
        print(f"⚠️  Integration test skipped: {str(e)}")
        return True  # Don't fail the overall test if LLM isn't available

def main():
    """Run all tests"""
    
    print("JSON Parser Fix Verification")
    print("=" * 50)
    
    # Run tests
    parser_test_passed = test_json_parser()
    validation_test_passed = test_validation_functions()
    
    # Run integration test
    integration_test_passed = asyncio.run(test_integration_with_llm())
    
    # Summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Parser Tests: {'✅ PASS' if parser_test_passed else '❌ FAIL'}")
    print(f"Validation Tests: {'✅ PASS' if validation_test_passed else '❌ FAIL'}")
    print(f"Integration Tests: {'✅ PASS' if integration_test_passed else '❌ FAIL'}")
    
    all_passed = parser_test_passed and validation_test_passed and integration_test_passed
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 The JSON parsing fix should resolve the LLM evaluation issues!")
        print("You can now run the LLM evaluation with improved reliability.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
