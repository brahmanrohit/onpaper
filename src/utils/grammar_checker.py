import os
import json
from typing import Dict, List, Tuple
import re
from .gemini_helper import generate_text

class GrammarChecker:
    """Grammar checking utility using Gemini AI API"""
    
    def __init__(self):
        pass
        
    def check_grammar(self, text: str) -> Dict:
        """
        Check grammar using Gemini AI API
        Returns: Dict with corrected_text, changes, and statistics
        """
        if not text.strip():
            return {
                'corrected_text': text,
                'changes': [],
                'statistics': {
                    'total_errors': 0,
                    'grammar_errors': 0,
                    'spelling_errors': 0,
                    'style_errors': 0
                },
                'message': 'No text provided for checking.'
            }
        
        try:
            # Create prompt for Gemini AI
            prompt = self._create_grammar_check_prompt(text)
            
            # Get response from Gemini
            response = generate_text(prompt)
            
            # Parse the response
            result = self._parse_gemini_response(text, response)
            
            return result
            
        except Exception as e:
            # Fallback to basic spell checking if API fails
            return self._fallback_check(text, str(e))
    
    def _create_grammar_check_prompt(self, text: str) -> str:
        """Create a prompt for Gemini AI to check grammar"""
        prompt = f"""
You are a professional grammar checker. Please analyze the following text and provide corrections in JSON format.

Text to check:
"{text}"

Please respond with a JSON object in this exact format:
{{
    "corrected_text": "the corrected version of the text",
    "changes": [
        {{
            "type": "Grammar|Spelling|Style|Punctuation",
            "message": "brief description of the correction",
            "original": "original text",
            "corrected": "corrected text",
            "position": 0,
            "context": "context around the error",
            "rule_id": "gemini_grammar",
            "severity": "error|warning|info"
        }}
    ],
    "statistics": {{
        "total_errors": 0,
        "grammar_errors": 0,
        "spelling_errors": 0,
        "style_errors": 0
    }}
}}

Rules:
1. Only include actual corrections that improve the text
2. Be specific about what was changed and why
3. Provide context around each correction
4. Categorize errors appropriately (Grammar, Spelling, Style, Punctuation)
5. If no errors found, return empty changes array and 0 statistics
6. Ensure the corrected_text is the final, corrected version
7. Position should be the character position where the error starts
8. Context should be ~50 characters before and after the error

Focus on:
- Grammar errors (subject-verb agreement, sentence structure)
- Spelling mistakes
- Punctuation errors
- Style improvements
- Capitalization issues
"""
        return prompt
    
    def _parse_gemini_response(self, original_text: str, response: str) -> Dict:
        """Parse Gemini AI response and extract corrections"""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # If no JSON found, use fallback
                return self._fallback_check(original_text, "No valid JSON response from AI")
            
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            # Validate the response structure
            if not isinstance(result, dict):
                return self._fallback_check(original_text, "Invalid response format")
            
            # Ensure required fields exist
            corrected_text = result.get('corrected_text', original_text)
            changes = result.get('changes', [])
            statistics = result.get('statistics', {
                'total_errors': len(changes),
                'grammar_errors': 0,
                'spelling_errors': 0,
                'style_errors': 0
            })
            
            # Calculate statistics if not provided
            if statistics['total_errors'] == 0 and len(changes) > 0:
                statistics['total_errors'] = len(changes)
                for change in changes:
                    change_type = change.get('type', '').lower()
                    if 'grammar' in change_type:
                        statistics['grammar_errors'] += 1
                    elif 'spelling' in change_type:
                        statistics['spelling_errors'] += 1
                    elif 'style' in change_type:
                        statistics['style_errors'] += 1
                    else:
                        statistics['grammar_errors'] += 1
            
            return {
                'corrected_text': corrected_text,
                'changes': changes,
                'statistics': statistics,
                'message': self._generate_message(statistics)
            }
            
        except json.JSONDecodeError as e:
            return self._fallback_check(original_text, f"JSON parsing error: {str(e)}")
        except Exception as e:
            return self._fallback_check(original_text, f"Response parsing error: {str(e)}")
    
    def _get_context(self, text: str, offset: int, length: int, context_length: int = 50) -> str:
        """Get context around the error"""
        start = max(0, offset - context_length)
        end = min(len(text), offset + length + context_length)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    def _generate_message(self, statistics: Dict) -> str:
        """Generate a user-friendly message based on statistics"""
        total = statistics['total_errors']
        
        if total == 0:
            return "✅ No grammar or spelling errors found!"
        elif total == 1:
            return f"⚠️ Found 1 error that has been corrected."
        else:
            return f"⚠️ Found {total} errors that have been corrected."
    
    def _fallback_check(self, text: str, error_msg: str) -> Dict:
        """Fallback grammar checking when API is unavailable"""
        # Enhanced basic spell checking using common patterns
        changes = []
        corrected_text = text
        
        # Common corrections with better categorization
        common_fixes = [
            # Capitalization fixes
            (r'\b(i)\b', 'I', 'Capitalization', 'Capitalized "i" to "I"'),
            (r'\b(im)\b', 'I\'m', 'Grammar', 'Corrected "im" to "I\'m"'),
            
            # Contraction fixes
            (r'\b(its)\s+', 'it\'s ', 'Grammar', 'Corrected "its" to "it\'s"'),
            (r'\b(youre)\b', 'you\'re', 'Grammar', 'Corrected "youre" to "you\'re"'),
            (r'\b(theyre)\b', 'they\'re', 'Grammar', 'Corrected "theyre" to "they\'re"'),
            (r'\b(weve)\b', 'we\'ve', 'Grammar', 'Corrected "weve" to "we\'ve"'),
            (r'\b(ive)\b', 'I\'ve', 'Grammar', 'Corrected "ive" to "I\'ve"'),
            (r'\b(cant)\b', 'can\'t', 'Grammar', 'Corrected "cant" to "can\'t"'),
            (r'\b(wont)\b', 'won\'t', 'Grammar', 'Corrected "wont" to "won\'t"'),
            (r'\b(dont)\b', 'don\'t', 'Grammar', 'Corrected "dont" to "don\'t"'),
            (r'\b(doesnt)\b', 'doesn\'t', 'Grammar', 'Corrected "doesnt" to "doesn\'t"'),
            (r'\b(havent)\b', 'haven\'t', 'Grammar', 'Corrected "havent" to "haven\'t"'),
            (r'\b(hadnt)\b', 'hadn\'t', 'Grammar', 'Corrected "hadnt" to "hadn\'t"'),
            (r'\b(isnt)\b', 'isn\'t', 'Grammar', 'Corrected "isnt" to "isn\'t"'),
            (r'\b(arent)\b', 'aren\'t', 'Grammar', 'Corrected "arent" to "aren\'t"'),
            
            # Common spelling mistakes
            (r'\b(recieve)\b', 'receive', 'Spelling', 'Corrected "recieve" to "receive"'),
            (r'\b(seperate)\b', 'separate', 'Spelling', 'Corrected "seperate" to "separate"'),
            (r'\b(definately)\b', 'definitely', 'Spelling', 'Corrected "definately" to "definitely"'),
            (r'\b(occured)\b', 'occurred', 'Spelling', 'Corrected "occured" to "occurred"'),
            (r'\b(accomodate)\b', 'accommodate', 'Spelling', 'Corrected "accomodate" to "accommodate"'),
            (r'\b(neccessary)\b', 'necessary', 'Spelling', 'Corrected "neccessary" to "necessary"'),
            (r'\b(priviledge)\b', 'privilege', 'Spelling', 'Corrected "priviledge" to "privilege"'),
            (r'\b(occassion)\b', 'occasion', 'Spelling', 'Corrected "occassion" to "occasion"'),
            
            # Punctuation fixes
            (r'\s+([.!?])\s*([A-Z])', r'\1 \2', 'Punctuation', 'Added space after punctuation'),
            (r'\s+([,;:])\s*([A-Z])', r'\1 \2', 'Punctuation', 'Added space after punctuation'),
        ]
        
        for pattern, replacement, error_type, message in common_fixes:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                original = match.group(0)
                corrected = replacement
                
                if original.lower() != corrected.lower():
                    corrected_text = corrected_text[:match.start()] + corrected + corrected_text[match.end():]
                    changes.append({
                        'type': error_type,
                        'message': message,
                        'original': original,
                        'corrected': corrected,
                        'position': match.start(),
                        'context': self._get_context(text, match.start(), len(original)),
                        'rule_id': 'fallback_basic',
                        'severity': 'info'
                    })
        
        # Calculate statistics
        grammar_errors = sum(1 for c in changes if c['type'] == 'Grammar')
        spelling_errors = sum(1 for c in changes if c['type'] == 'Spelling')
        style_errors = sum(1 for c in changes if c['type'] == 'Style')
        
        return {
            'corrected_text': corrected_text,
            'changes': changes,
            'statistics': {
                'total_errors': len(changes),
                'grammar_errors': grammar_errors,
                'spelling_errors': spelling_errors,
                'style_errors': style_errors
            },
            'message': f'Basic corrections applied using fallback mode. (AI unavailable: {error_msg})'
        }

# Global instance
grammar_checker = GrammarChecker()

def check_grammar_text(text: str) -> Dict:
    """Main function to check grammar - maintains backward compatibility"""
    return grammar_checker.check_grammar(text)
