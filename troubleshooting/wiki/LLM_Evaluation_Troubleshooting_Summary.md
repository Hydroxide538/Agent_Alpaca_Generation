# LLM Evaluation Troubleshooting Summary

## Problem Description
The `evaluate_llms.py` script consistently scores 0.00 for all evaluation tasks (fact extraction, concept extraction, analytical Q&A). The primary cause appears to be LLMs failing to return valid JSON, leading to `json.JSONDecodeError` exceptions. Additionally, a `ExtractedFact.__init__() missing 1 required positional argument: 'source_location'` error was observed for fact extraction.

## Attempted Solutions & Analysis

### Attempt 1: Initial Prompt Refinement and `_clean_json_response`
- **Action:** Modified prompts in `backend/improved_alpaca_generator.py` (`fact_extraction_prompt`, `concept_extraction_prompt`) to explicitly instruct LLMs to "Respond ONLY with the JSON array. Do NOT include any conversational text, explanations, markdown outside the JSON, or any other extraneous characters."
- **Action:** Introduced `_clean_json_response` method in `ImprovedAlpacaGenerator` to attempt to strip markdown code blocks (```json) and find the first/last braces/brackets to isolate JSON.
- **Result:** The `json.JSONDecodeError` persisted, indicating LLMs were still not adhering to the strict JSON output, and `_clean_json_response` was not sufficiently robust. The `source_location` error also appeared.

### Attempt 2: Correcting `source_location` Passing
- **Action:** Identified that `_extract_structured_facts` and `_extract_structured_concepts` were extending `chunk_facts` directly instead of calling `_parse_structured_facts` and `_parse_structured_concepts` with the `chunk_index`. Modified these methods to correctly pass `i` (chunk_index) to `_parse_structured_facts` and `_parse_structured_concepts` as the `source_location`.
- **Result:** The `source_location` error was expected to be resolved, but it reappeared in subsequent runs, suggesting either the `replace_in_file` operation was not fully effective or there was a misunderstanding of the call flow. The JSON parsing errors continued.

### Attempt 3: Further Enhancing `_clean_json_response` and Re-verifying `source_location`
- **Action:** Further enhanced `_clean_json_response` to be more aggressive. Added a list of common conversational prefixes and thinking tags to be removed from the response before attempting JSON parsing. The method now also explicitly extracts content between the first `[` or `{` and the last `]` or `}`.
- **Action:** Re-verified the call flow and confirmed that `_extract_structured_facts` and `_extract_structured_concepts` were indeed the ones responsible for calling `_parse_structured_facts` and `_parse_structured_concepts` with the `chunk_index`. The previous `replace_in_file` for this was confirmed to be correct in the file content. The issue was that the `facts.extend(chunk_facts)` line was overriding the intended behavior. This was corrected.

## Current Status
- The `source_location` error should now be resolved due to the direct correction of the `facts.extend(chunk_facts)` line.
- The `_clean_json_response` method has been significantly improved to better handle extraneous text from LLMs.
- The next step is to re-run the `evaluate_llms.py` script to confirm that both the `source_location` error and the JSON parsing errors are resolved, and that the LLMs can now be properly evaluated.
