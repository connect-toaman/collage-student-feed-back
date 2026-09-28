import json
import os
import re

class AspectExtractor:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'aspects.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)
            
        # Pre-compile regex for aspect matching to be faster
        self.compiled_taxonomy = {}
        for aspect, synonyms in self.taxonomy.items():
            # Sort by length descending to match longest phrases first (e.g. "lab equipment" before "lab")
            sorted_syns = sorted(synonyms, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(map(re.escape, sorted_syns)) + r')\b'
            self.compiled_taxonomy[aspect] = re.compile(pattern, re.IGNORECASE)

    def split_into_clauses(self, text):
        # Split on punctuation and conjunctions that usually change sentiment context
        # "and" can sometimes change context ("teacher is good and lab is bad")
        split_pattern = r'(\.|,|;|!|\?|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhile\b|\byet\b|\band\b)'
        parts = re.split(split_pattern, text, flags=re.IGNORECASE)
        
        clauses = []
        current_clause = ""
        for part in parts:
            if re.match(split_pattern, part, re.IGNORECASE):
                if current_clause.strip():
                    clauses.append(current_clause.strip())
                current_clause = ""
            else:
                current_clause += part
        if current_clause.strip():
            clauses.append(current_clause.strip())
            
        return clauses

    def extract_aspects(self, text):
        clauses = self.split_into_clauses(text)
        
        extracted = []
        found_aspects = set()
        
        for clause in clauses:
            for aspect, regex in self.compiled_taxonomy.items():
                if regex.search(clause):
                    extracted.append({
                        "aspect": aspect,
                        "context": clause.strip()
                    })
                    found_aspects.add(aspect)
                    
        # Filter down if a clause has multiple matched aspects? Let's keep it simple: one clause can yield multiple aspects.
        # But we only want unique aspects. If an aspect appears multiple times, we just take the first matching clause, or join them?
        # Let's keep the first context for simplicity, or we could join contexts.
        # Actually, let's just group contexts by aspect.
        
        grouped_aspects = []
        seen = set()
        for item in extracted:
            aspect = item['aspect']
            if aspect not in seen:
                seen.add(aspect)
                # Combine all clauses mentioning this aspect? Or just take the first one?
                # The user requirement shows "The teachers are good" as context. First is fine.
                contexts = [x['context'] for x in extracted if x['aspect'] == aspect]
                grouped_aspects.append({
                    "aspect": aspect,
                    "context": " ... ".join(contexts)
                })
                
        return grouped_aspects
