#!/usr/bin/env python3
"""
Field-Level Real-Time Validation API v2
Follows policy_validator.py's dynamic approach - simpler, less hardcoding
Uses Gemini's analysis of actual search results to guide the search
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import subprocess
import os
import re
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configuration
GEMINI_API_KEY = "AIzaSyAgWfY5zft6IV00Y2HwPc3JHQva38zWEDQ"
RULES_FILE = "/home/hutech/Documents/docupolicy/rules.txt"
OUTPUT_DIR = "/home/hutech/Documents/docupolicy"
LOG_FILE = f"{OUTPUT_DIR}/validation_api.log"
CHROMA_DB_PATH = f"{OUTPUT_DIR}/chroma_embeddings"

app = Flask(__name__)
CORS(app)


class RuleIndexer:
    """Generic rule indexing for ANY policy structure"""
    
    def __init__(self, rules_content):
        self.rules_content = rules_content
        self.rules = []  # List of parsed rules
        self._build_index()
    
    def _build_index(self):
        """Dynamically extract and index ALL rules from content"""
        import re
        
        lines = self.rules_content.split('\n')
        current_rule = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Start of a new rule
            if line_stripped.startswith('rule:'):
                # Save previous rule if exists
                if current_rule:
                    self.rules.append(current_rule)
                
                # Extract rule name
                rule_name = line_stripped.replace('rule:', '').strip()
                current_rule = {
                    'name': rule_name,
                    'conditions': [],
                    'source': None,
                    'raw_text': line_stripped
                }
            
            # Extract conditions and THEN clauses (IF/THEN statements)
            elif line_stripped.startswith('- IF') and current_rule:
                condition_text = line_stripped.replace('- IF', '').strip()
                current_rule['conditions'].append(condition_text)
                
                # Also extract THEN clause if on same line or next line
                if 'THEN' in condition_text:
                    then_part = condition_text.split('THEN', 1)[1].strip()
                    if 'then_clause' not in current_rule:
                        current_rule['then_clause'] = []
                    current_rule['then_clause'].append(then_part)
            
            # Extract THEN clauses (can be on continuation lines)
            elif line_stripped.startswith('THEN') and current_rule:
                then_text = line_stripped.replace('THEN', '').strip()
                if 'then_clause' not in current_rule:
                    current_rule['then_clause'] = []
                current_rule['then_clause'].append(then_text)
            
            # Extract source
            elif line_stripped.startswith('SOURCE:') and current_rule:
                current_rule['source'] = line_stripped.replace('SOURCE:', '').strip()
            
            # Also parse numeric constants (e.g., allowance.lodging.tour.M3.A: 2000)
            elif ':' in line_stripped and not any(x in line_stripped for x in ['=', 'rule:', 'SOURCE:']):
                # Parse as constant/value definition
                match = re.match(r'([\w.]+):\s*(.+?)(?:\s*$)', line_stripped)
                if match:
                    key, value = match.groups()
                    self.rules.append({
                        'name': key,
                        'value': value.strip().strip('"\''),
                        'type': 'constant',
                        'raw_text': line_stripped
                    })
        
        # Add last rule
        if current_rule:
            self.rules.append(current_rule)
    
    def _generate_field_pattern(self, context_normalized):
        """
        Dynamically generate regex pattern for extracting entity.field references
        Uses Gemini to analyze the context and rules to determine valid entity types
        """
        import re
        
        # Extract all entity.field patterns from rules to understand structure
        all_entities = set()
        for rule in self.rules:
            if rule.get('type') != 'constant' and rule.get('conditions', []):
                combined = ' '.join(rule.get('conditions', []) + rule.get('then_clause', [])).lower()
                # Find any entity.field pattern (broad match)
                entities = re.findall(r'(\w+)\.[\w_]+', combined)
                all_entities.update(entities)
        
        # Get context keys as potential entities
        context_entities = set(context_normalized.keys())
        
        # Combine all possible entities
        possible_entities = list(all_entities | context_entities)
        possible_entities = sorted(set([e for e in possible_entities if len(e) > 1]))  # Filter short names
        
        # If we have entities, use them; otherwise use broader pattern
        if possible_entities:
            # Create pattern from found entities: (entity1|entity2|entity3)\.(\w+)
            entity_pattern = '|'.join(possible_entities)
            field_pattern = rf'({entity_pattern})\.([\w_]+)'
        else:
            # Fallback to broader pattern that matches any entity.field
            field_pattern = r'([\w_]+)\.([\w_]+)'
        
        return field_pattern
    
    def lookup(self, field_name, context):
        """
        Generically lookup rules for any field given context.
        Works with ANY policy structure - numeric, boolean, conditional, etc.
        """
        if not self.rules or not context:
            return None
        
        # Normalize context
        context_normalized = {}
        context_values_flat = []
        
        if isinstance(context, list):
            # Handle list of context items
            for item in context:
                if isinstance(item, dict):
                    for k, v in item.items():
                        normalized_key = k.lower().replace('.', '_')
                        context_normalized[normalized_key] = str(v)
                        context_values_flat.append(str(v).lower())
        else:
            # Handle dict context
            for k, v in context.items():
                normalized_key = k.lower().replace('.', '_')
                context_normalized[normalized_key] = str(v)
                context_values_flat.append(str(v).lower())
        
        matching_rules = []
        
        # Dynamically generate field pattern based on actual entities in rules
        field_pattern = self._generate_field_pattern(context_normalized)
        
        # Search through ALL rules - stricter matching
        for rule in self.rules:
            rule_name = rule.get('name', '').lower()
            conditions = rule.get('conditions', [])
            then_clauses = rule.get('then_clause', [])
            
            # Only match if rule is a conditional rule (not a metadata line)
            if rule.get('type') != 'constant' and conditions:
                combined_text = ' '.join(conditions + then_clauses).lower()
                
                # Try to find field_name mentioned in conditions or THEN clause
                # Extract actual field references from conditions using dynamic pattern
                field_mentions = []
                import re
                matches = re.findall(field_pattern, combined_text)
                # Extract just the field name (second group if tuple, else use as-is)
                field_mentions = [m[1] if isinstance(m, tuple) else m for m in matches]
                
                # Also handle multi-level fields like patient.incapacity.duration
                # Extract all dot-separated words and add the deepest levels
                deep_fields = re.findall(r'(?:[\w_]+\.)+(\w+)', combined_text)
                field_mentions.extend(deep_fields)
                
                # Also check for plain field names (without entity prefix) in THEN clause
                then_text = ' '.join(rule.get('then_clause', [])).lower()
                plain_fields = re.findall(r'([a-z_]+)\s*=', then_text)
                field_mentions.extend(plain_fields)
                
                # For value-based matching (e.g., "pregnancy" in condition value)
                # Extract values from conditions that might match field names
                condition_text = ' '.join(conditions).lower()
                quoted_values = re.findall(r'"([^"]+)"', condition_text)
                field_mentions.extend(quoted_values)
                
                # Check if our field_name or similar matches any mentioned field
                field_lower = field_name.lower().replace('_', '').replace('-', '')
                field_mentions_normalized = [fm.lower().replace('_', '').replace('-', '') for fm in field_mentions]
                
                # Direct match
                field_matches = [fm for fm in field_mentions if fm.lower().replace('_', '').replace('-', '') == field_lower]
                
                # Also allow suffix matches (e.g., "duration" matches "incapacity_duration")
                if not field_matches:
                    for fm_norm, fm_orig in zip(field_mentions_normalized, field_mentions):
                        if field_lower.endswith(fm_norm) and len(fm_norm) > 2:  # Suffix match with min length
                            field_matches.append(fm_orig)
                            break
                
                if field_matches:
                    # Direct field match - check context values for additional confirmation
                    match_score = 0
                    for ctx_val in context_values_flat:
                        if ctx_val and ctx_val in combined_text:
                            match_score += 1
                    
                    # Reward for direct field match
                    matching_rules.append((rule, 200 + match_score * 10))
                else:
                    # No direct field match, but check if context values appear in quoted values (for prohibition rules)
                    condition_text = ' '.join(conditions).lower()
                    quoted_values = re.findall(r'"([^"]+)"', condition_text)
                    
                    # Check if any context value matches quoted value in prohibition
                    for ctx_val in context_values_flat:
                        if ctx_val and any(ctx_val in qv.lower() for qv in quoted_values if qv):
                            # This is likely a value-based match (e.g., "genetic services" in context matches quoted value)
                            if 'prohibited' in rule_name or 'denied' in ' '.join(conditions + rule.get('then_clause', [])).lower():
                                matching_rules.append((rule, 180))  # High score for value match on prohibition
                                break
            
            # Also match numeric constants by field name
            elif rule.get('type') == 'constant':
                rule_name_key = rule.get('name', '').lower()
                if field_name.lower() in rule_name_key:
                    # Check if rule key contains any context value
                    for ctx_val in context_values_flat:
                        if ctx_val and ctx_val in rule_name_key:
                            matching_rules.append((rule, 80))
                            break
        
        # Try semantic inference if no direct match found
        best_rule = None
        if not matching_rules:
            # Check if field_name suggests it's measuring time/deadline related
            field_lower = field_name.lower()
            if any(word in field_lower for word in ['days', 'since', 'deadline', 'duration', 'within']):
                # This looks like a numeric temporal constraint field
                # Try to find rules with numeric constraints
                for rule in self.rules:
                    if rule.get('type') != 'constant' and rule.get('conditions', []):
                        combined_text = ' '.join(rule.get('conditions', []) + rule.get('then_clause', [])).lower()
                        # Look for time-related keywords
                        if any(word in combined_text for word in ['within', 'days', 'deadline', 'duration', 'calendar']):
                            # This rule has temporal constraints
                            matching_rules.append((rule, 150))  # Lower score than direct match
        
        # Return best matching rule
        if matching_rules:
            # Sort by score and return highest
            matching_rules.sort(key=lambda x: x[1], reverse=True)
            best_rule = matching_rules[0][0]
        else:
            best_rule = None
        
        if best_rule:
             # Convert to lookup format for compatibility
            if best_rule.get('type') == 'constant':
                return {
                    'rule': best_rule.get('raw_text', ''),
                    'value': best_rule.get('value'),
                    'name': best_rule.get('name')
                }
            else:
                return {
                    'rule': best_rule.get('name', ''),
                    'conditions': best_rule.get('conditions', []),
                    'then_clause': best_rule.get('then_clause', []),
                    'source': best_rule.get('source'),
                    'name': best_rule.get('name')
                }
        
        return None


class FieldValidator:
    """Dynamic field validation following policy_validator.py approach"""

    def __init__(self, rules_file, max_attempts=3):
        self.rules_file = rules_file
        self.max_attempts = max_attempts
        self.log = []
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=40
        )
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        # self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Cache for fast lookups
        self.field_classification_cache = {}
        self.field_patterns_cache = {}
        self.rules_content = None
        self.rules_lines = []
        self.rule_indexer = None
        
        # Initialize ChromaDB for embedding storage (will be initialized in startup)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.chroma_collection = None
        self.embedding_function = None
        
        # Initialize at startup (only once)
        self._startup_initialization()
        
    def _startup_initialization(self):
        """Initialize everything once at startup"""
        try:
            self.log_entry("STARTUP", "Initializing validator at startup...")
            
            # Load rules
            self._load_rules()
            
            # Initialize rule indexer (generic, works with any policy)
            self.rule_indexer = RuleIndexer(self.rules_content)
            self.log_entry("STARTUP", f"✓ Built generic rule index with {len(self.rule_indexer.rules)} rules")
            
            # Initialize ChromaDB with embeddings (keep for complex queries)
            self._init_chroma_collection()
            
            # Populate ChromaDB
            self._populate_chroma_db()
            
            self.log_entry("STARTUP", "✓ Validator initialization complete - ready for requests")
        except Exception as e:
            self.log_entry("ERROR", f"Startup initialization failed: {e}")
    
    def _load_rules(self):
        """Load rules file"""
        if self.rules_content is None:
            try:
                with open(self.rules_file, 'r') as f:
                    self.rules_content = f.read()
                # Split into lines and filter empty ones
                self.rules_lines = [line.strip() for line in self.rules_content.split('\n') 
                                   if line.strip() and not line.strip().startswith('=')]
                self.log_entry("INFO", f"Loaded {len(self.rules_lines)} rule lines")
            except Exception as e:
                self.log_entry("ERROR", f"Failed to load rules file: {e}")
                self.rules_content = ""
                self.rules_lines = []
        return self.rules_content
    
    def _init_chroma_collection(self):
        """Initialize ChromaDB collection with local sentence-transformers embeddings"""
        try:
            if self.chroma_collection is not None:
                return self.chroma_collection
            
            # Load sentence-transformer model (local, no API calls) - cache it
            if self.embedding_function is None:
                self.log_entry("INFO", "Loading sentence-transformers model...")
                self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"  # Fast, lightweight model
                )
            
            # Try to get existing collection, create if not exists
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name="policy_rules",
                    embedding_function=self.embedding_function
                )
                self.log_entry("INFO", f"Using existing ChromaDB collection with {self.chroma_collection.count()} rules")
            except:
                # Collection doesn't exist, create it
                self.chroma_collection = self.chroma_client.create_collection(
                    name="policy_rules",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_function
                )
                self.log_entry("INFO", "Created new ChromaDB collection with local embeddings")
            
            return self.chroma_collection
        except Exception as e:
            self.log_entry("ERROR", f"Failed to initialize ChromaDB: {e}")
            return None
    
    def _populate_chroma_db(self):
        """Populate ChromaDB with rule embeddings"""
        try:
            collection = self._init_chroma_collection()
            if collection is None or not self.rules_lines:
                return False
            
            # Check if already populated
            count = collection.count()
            if count > 0:
                self.log_entry("INFO", f"ChromaDB already has {count} rules, skipping population")
                return True
            
            self.log_entry("INFO", f"Populating ChromaDB with {len(self.rules_lines)} rules...")
            
            # Add rules to ChromaDB in batches
            batch_size = 10
            for i in range(0, len(self.rules_lines), batch_size):
                batch = self.rules_lines[i:i+batch_size]
                ids = [f"rule_{i+j}" for j in range(len(batch))]
                
                collection.add(
                    ids=ids,
                    documents=batch,
                    metadatas=[{"index": i+j, "length": len(text)} for j, text in enumerate(batch)]
                )
                
                if i % (batch_size * 5) == 0:
                    self.log_entry("DEBUG", f"Added {i+len(batch)}/{len(self.rules_lines)} rules to ChromaDB")
            
            self.log_entry("INFO", f"Successfully populated ChromaDB with {len(self.rules_lines)} rules")
            return True
        except Exception as e:
            self.log_entry("ERROR", f"Failed to populate ChromaDB: {e}")
            return False

    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)

    def is_general_non_policy_field(self, field_name):
        """Fast regex-based check for common non-policy fields"""
        cache_key = field_name.lower()
        if cache_key in self.field_classification_cache:
            return self.field_classification_cache[cache_key]
        
        general_patterns = [
            r'^(first_?name|last_?name|full_?name|employee_?name|user_?name|name)$',
            r'^(email|e.?mail|email_?address)$',
            r'^(phone|mobile|cell|telephone|phone_?number)$',
            r'^(id|employee_?id|user_?id|ssn)$',
            r'^(address|street|city|state|zip|postal)$',
            r'^(notes|comments|description|remarks|message|reason|purpose|justification)$',
            r'^(status|approval_?status|approval|submitted|draft|pending)$',
            r'^(company|organization|department|team|title|position|job_?title)$',
            r'^(date|timestamp|created|modified|updated_?at|reference|ticket)$',
        ]
        
        field_lower = field_name.lower()
        is_general = any(re.match(pattern, field_lower) for pattern in general_patterns)
        
        self.field_classification_cache[cache_key] = is_general
        return is_general

    def semantic_search(self, query, top_k=10):
        """Search rules using ChromaDB semantic similarity"""
        try:
            # Check if initialized (already done at startup)
            if not self.rules_lines or self.chroma_collection is None:
                self.log_entry("ERROR", "Validator not initialized - rules or ChromaDB missing")
                return []
            
            # Query ChromaDB
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=None  # No metadata filtering
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                docs = results['documents'][0]  # First query result
                distances = results['distances'][0] if 'distances' in results else []
                
                for i, doc in enumerate(docs):
                    # ChromaDB uses distances (lower = more similar), convert to similarity score
                    distance = distances[i] if i < len(distances) else 1.0
                    similarity = 1 - (distance / 2)  # Normalize to 0-1
                    
                    if similarity > 0.3:  # Only include if reasonable similarity
                        formatted_results.append({
                            'line': doc,
                            'score': float(similarity),
                            'distance': float(distance)
                        })
            
            self.log_entry("SEMANTIC_SEARCH", f"Query '{query}': Found {len(formatted_results)} relevant rules")
            return formatted_results
        except Exception as e:
            self.log_entry("ERROR", f"Semantic search failed: {e}")
            return []
    
    def semantic_search_multiple(self, queries):
        """Search for multiple query patterns using semantic similarity"""
        try:
            if not queries:
                return []
            
            # Combine queries into single search query
            combined_query = " ".join(queries)
            results = self.semantic_search(combined_query, top_k=15)
            
            self.log_entry("SEMANTIC_SEARCH_MULTI", f"Queries {queries}: Found {len(results)} rules")
            return results
        except Exception as e:
            self.log_entry("ERROR", f"Multi-query semantic search failed: {e}")
            return []
    
    def semantic_search_by_field(self, field_name, field_value, context=None):
        """Specialized semantic search for a field with intelligent query generation"""
        try:
            # Generate contextual query for the field
            query = self._generate_search_query(field_name, field_value, context)
            self.log_entry("FIELD_QUERY", f"{field_name}: {query}")
            
            # Perform semantic search
            results = self.semantic_search(query, top_k=10)
            return results
        except Exception as e:
            self.log_entry("ERROR", f"Field-based semantic search failed: {e}")
            return []
    
    def _generate_search_query(self, field_name, field_value, context=None):
        """Generate an intelligent search query for a field"""
        context_str = ""
        if context:
            context_items = [f"{k}={v}" for k, v in context.items()]
            context_str = f"\nContext: {', '.join(context_items)}"
        
        prompt = f"""Create a semantic search query to find policy constraints for this field.

Field Name: {field_name}
Field Value: {field_value}{context_str}

TASK: Generate a SHORT search query (1-2 sentences) that will find the SPECIFIC policy rule/constraint for this field.
- Include what the field represents
- Include what constraint or requirement you're looking for
- Be specific about what type of limit or rule to find
- If context is provided, include relevant context keywords (like grade, city, travel type)

Examples:
- For group_size=6: "how many employees can travel together limit employees travelling"
- For room_type=suite: "room upgrades policies standard room requirements"
- For flight_hours=8: "flight duration class of service business class first class"
- For lodging with context: "lodging allowance M3 city A tour"

RESPOND WITH ONLY THE QUERY (no quotes):
search_query_here"""

        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            query = response.text.strip().strip('"\'')
            return query
        except Exception as e:
            self.log_entry("ERROR", f"Query generation failed: {e}")
            return f"{field_name} {field_value}"

    def validate_field(self, field_name, field_value, previous_context=None):
        """
        Main validation using semantic search + Gemini analysis
        """
        self.log_entry("VALIDATION", f"Starting for {field_name}={field_value}")
        
        # Skip non-policy fields instantly
        if self.is_general_non_policy_field(field_name):
            self.log_entry("SKIP_REGEX", f"General field - auto-approved")
            return {
                'field_name': field_name,
                'field_value': field_value,
                'status': 'valid',
                'message': '✓ Field accepted - general information field',
                'rules': [],
                'validation_details': {
                    'field_meaning': 'General/non-policy field',
                    'skipped': True,
                    'reason': 'No policy constraints'
                }
            }

        # Step 1: Use dynamic rule indexing (no semantic search, no hardcoding)
        self.log_entry("INDEX_LOOKUP", f"Looking up rule for {field_name} with context {previous_context}")
        
        indexed_rule = None
        if self.rule_indexer:
            # Try lookup with field_name as-is, and also try extracting base field name
            indexed_rule = self.rule_indexer.lookup(field_name, previous_context)
            
            # If not found, try to extract base field name (e.g., "lodging" from "lodging_requested")
            if not indexed_rule and '_' in field_name:
                base_field = field_name.split('_')[0]
                self.log_entry("FIELD_EXTRACT", f"Trying base field: {base_field}")
                indexed_rule = self.rule_indexer.lookup(base_field, previous_context)
        
        if indexed_rule:
            # Found via indexing - use it directly
            self.log_entry("INDEX_HIT", f"Found indexed rule: {indexed_rule.get('rule', indexed_rule.get('name', ''))}")
            
            # Handle numeric rules (constants)
            if indexed_rule.get('type') == 'constant' or 'value' in indexed_rule:
                try:
                    field_value_num = float(field_value)
                    limit_value = float(indexed_rule.get('value', 0))
                    self.log_entry("NUMERIC_VALIDATION", f"Direct validation: {field_value_num} vs limit {limit_value}")
                    is_valid = field_value_num <= limit_value
                    
                    return {
                        'field_name': field_name,
                        'field_value': field_value,
                        'status': 'valid' if is_valid else 'error',
                        'message': f"{'✓' if is_valid else '✗'} {field_name} ({field_value}) {'is within' if is_valid else 'exceeds'} allowance ({limit_value})",
                        'rules': [indexed_rule.get('rule', indexed_rule.get('name', ''))],
                        'validation_details': {
                            'search_method': 'indexed_lookup',
                            'indexed_rule': indexed_rule.get('rule'),
                            'decision_reasoning': f"Direct rule match: {field_name} allowance is {limit_value}. Value {field_value} {'complies' if is_valid else 'violates'} this limit."
                        }
                    }
                except (ValueError, TypeError):
                    pass  # Not numeric, proceed to conditional logic
            
            # Handle conditional rules (FMLA, complex policies)
            conditions = indexed_rule.get('conditions', [])
            if conditions:
                self.log_entry("CONDITIONAL_RULE", f"Evaluating conditions for {indexed_rule.get('name', '')}")
                
                # Parse conditions for numeric constraints and denied status
                # Look for patterns like "MUST be within 15", "<= 15", "> 5", etc.
                import re
                constraint_match = None
                constraint_value = None
                constraint_operator = None
                
                # Extract numeric constraints from conditions AND THEN clauses
                then_clauses = indexed_rule.get('then_clause', [])
                combined_conditions = ' '.join(conditions + then_clauses).lower()
                
                # Check if rule explicitly denies the request
                is_denied = re.search(r'=\s*DENIED|request\s*=\s*DENIED', combined_conditions, re.IGNORECASE)
                if is_denied:
                    self.log_entry("DENIED_RULE", f"Rule {indexed_rule.get('name')} explicitly denies this request")
                    return {
                        'field_name': field_name,
                        'field_value': field_value,
                        'status': 'error',
                        'message': f"✗ {field_name} is DENIED by rule: {indexed_rule.get('name', '')}",
                        'rules': conditions,
                        'validation_details': {
                            'search_method': 'indexed_lookup',
                            'rule_name': indexed_rule.get('name'),
                            'conditions': conditions,
                            'source': indexed_rule.get('source'),
                            'decision_reasoning': f"Rule {indexed_rule.get('name')} specifies: {' '.join(then_clauses)}"
                        }
                    }
                
                # First, resolve variable references like "fmla.certification_deadline" → "15"
                for rule in self.rule_indexer.rules:
                    if rule.get('type') == 'constant':
                        const_name = rule.get('name', '').lower()
                        const_value = rule.get('value', '')
                        # Extract just the number from values like "15 calendar days"
                        number_match = re.search(r'(\d+)', const_value)
                        if number_match:
                            const_value = number_match.group(1)
                        # Replace constant names with their values in combined conditions
                        combined_conditions = combined_conditions.replace(const_name, const_value)
                
                # Pattern: "within 15", "15 days", "<= 15", "> 5", "maximum 15", "minimum 3"
                patterns = [
                    (r'within\s+(\d+)', '<='),  # "within 15" → <= 15
                    (r'<=\s*(\d+)', '<='),      # "<= 15"
                    (r'>=\s*(\d+)', '>='),      # ">= 15"
                    (r'<\s*(\d+)', '<'),        # "< 15"
                    (r'>\s*(\d+)', '>'),        # "> 5"
                    (r'maximum\s+(\d+)', '<='), # "maximum 15" → <= 15
                    (r'minimum\s+(\d+)', '>='), # "minimum 3" → >= 3
                    (r'exactly\s+(\d+)', '=='), # "exactly 3" → == 3
                ]
                
                for pattern, operator in patterns:
                    match = re.search(pattern, combined_conditions)
                    if match:
                        constraint_value = int(match.group(1))
                        constraint_operator = operator
                        constraint_match = match
                        break
                
                # Validate against constraint if found
                is_valid = True
                reasoning = f"Field {field_name} matches conditional rule."
                
                if constraint_value is not None and constraint_operator:
                    try:
                        field_val_num = float(field_value)
                        
                        if constraint_operator == '<=':
                            is_valid = field_val_num <= constraint_value
                        elif constraint_operator == '>=':
                            is_valid = field_val_num >= constraint_value
                        elif constraint_operator == '<':
                            is_valid = field_val_num < constraint_value
                        elif constraint_operator == '>':
                            is_valid = field_val_num > constraint_value
                        elif constraint_operator == '==':
                            is_valid = field_val_num == constraint_value
                        
                        reasoning = f"Constraint: {field_name} {constraint_operator} {constraint_value}. Value: {field_value}. {'Complies' if is_valid else 'Violates'} constraint."
                        
                        self.log_entry("CONSTRAINT_CHECK", f"{field_name}={field_value} vs {constraint_operator} {constraint_value}: {'PASS' if is_valid else 'FAIL'}")
                    except (ValueError, TypeError):
                        # Field value not numeric but constraint expects numeric
                        is_valid = False
                        reasoning = f"ERROR: {field_name} expects numeric value for constraint {constraint_operator} {constraint_value}, but got non-numeric value: {field_value}"
                        self.log_entry("TYPE_ERROR", f"{field_name} type mismatch: expected numeric, got '{field_value}'")
                
                return {
                    'field_name': field_name,
                    'field_value': field_value,
                    'status': 'valid' if is_valid else 'error',
                    'message': f"{'✓' if is_valid else '✗'} {field_name} {'matches' if is_valid else 'violates'} rule: {indexed_rule.get('name', '')}",
                    'rules': conditions,
                    'validation_details': {
                        'search_method': 'indexed_lookup',
                        'rule_name': indexed_rule.get('name'),
                        'conditions': conditions,
                        'source': indexed_rule.get('source'),
                        'constraint': f"{constraint_operator} {constraint_value}" if constraint_value else None,
                        'decision_reasoning': reasoning
                    }
                }
        else:
            # No indexed rule found - strict mode: context doesn't match any rule
            # Don't fall back to semantic search; return error or "no matching rule"
            self.log_entry("NO_INDEXED_RULE", f"No matching rule found in index for {field_name} with context {previous_context}")
            return {
                'field_name': field_name,
                'field_value': field_value,
                'status': 'error',
                'message': f'✗ No matching policy rule found for {field_name} with given context',
                'rules': [],
                'validation_details': {
                    'search_method': 'indexed_lookup',
                    'rules_found': 0,
                    'decision_reasoning': f'The provided context ({previous_context}) does not match any policy rule. This context combination is not defined in the policy.'
                }
            }
        
        # Step 2: Semantic search only used if indexed rule exists but needs Gemini analysis
        self.log_entry("SEMANTIC_ANALYSIS", f"Using semantic search for non-numeric fields")
        search_results = self.semantic_search_by_field(field_name, field_value, previous_context)
        
        if not search_results:
            self.log_entry("NO_RULES", f"No relevant rules found for {field_name}")
            return {
                'field_name': field_name,
                'field_value': field_value,
                'status': 'valid',
                'message': f'✓ {field_name} validated - no policy constraints found',
                'rules': [],
                'validation_details': {
                    'search_method': 'semantic',
                    'rules_found': 0,
                    'decision_reasoning': 'No applicable policy rules discovered'
                }
            }

        # Step 3: Filter and format rules - prioritize rules matching context
        formatted_rules = [result['line'] for result in search_results[:10]]
        
        # Try to extract the most relevant rule for the given context
        most_relevant_rule = self._extract_most_relevant_rule(formatted_rules, field_name, previous_context)
        if most_relevant_rule:
            # Put most relevant rule first
            formatted_rules = [most_relevant_rule] + [r for r in formatted_rules if r != most_relevant_rule]
        
        rules_text = "\n".join(formatted_rules)
        
        self.log_entry("RULES_FOUND", f"Found {len(search_results)} relevant rules, using top {len(formatted_rules)}")

        # Step 4: Generate validation decision from found rules
        # Use only the top rules (prioritized with most relevant first)
        rules_for_decision = formatted_rules[:5]
        decision = self._generate_validation_decision(
            field_name, field_value, rules_for_decision, previous_context
        )

        return {
            'field_name': field_name,
            'field_value': field_value,
            'status': decision['status'],
            'message': decision['message'],
            'rules': formatted_rules,
            'validation_details': {
                'search_method': 'semantic_similarity',
                'top_matches': [
                    {
                        'rule': result['line'],
                        'similarity_score': result['score']
                    } for result in search_results[:5]
                ],
                'total_rules_found': len(search_results),
                'decision_reasoning': decision.get('reasoning', '')
            }
        }



    def _extract_most_relevant_rule(self, rules, field_name, context):
        """Extract the most relevant rule based on context"""
        if not context or not rules:
            return None
        
        import re
        
        # Look for rules that match all context fields
        if isinstance(context, list):
            context_values = [str(item.get('field_value', '')).lower() for item in context if isinstance(item, dict)]
        else:
            context_values = [str(v).lower() for v in context.values()]
        
        for rule in rules:
            rule_lower = rule.lower()
            # Count how many context values appear in the rule
            matches = sum(1 for val in context_values if val in rule_lower)
            if matches >= 2:  # At least 2 context fields should match
                return rule
        
        return None
    
    def _generate_validation_decision(self, field_name, field_value, found_rules, previous_context):
        """Generate validation decision from found rules"""
        
        if not found_rules:
            return {
                'status': 'valid',
                'message': f'✓ {field_name} validated - no policy constraints found',
                'reasoning': 'Field accepted by default'
            }
        
        rules_text = "\n".join(found_rules[:10])
        
        # Format context for the prompt
        context_text = ""
        if previous_context:
            if isinstance(previous_context, list):
                context_items = [f"- {item.get('field_name')}: {item.get('field_value')}" 
                                for item in previous_context if isinstance(item, dict)]
            else:
                context_items = [f"- {k}: {v}" for k, v in previous_context.items()]
            context_text = "\n".join(context_items)
        
        # Extract numeric limits from rules if field_value is numeric
        numeric_limits = ""
        try:
            field_value_num = float(field_value)
            # Look for allowance values in rules
            import re
            allowances = re.findall(r'(allowance\.[\w.]+):\s*(\d+)', rules_text)
            if allowances:
                numeric_limits = "\nExtracted numeric limits from rules:\n"
                for allowance_name, value in allowances:
                    numeric_limits += f"- {allowance_name}: {value}\n"
        except:
            pass
        
        prompt = f"""Based on these policy rules, validate the field value.

Field: {field_name} = {field_value}

Context Information:
{context_text if context_text else "No additional context provided"}

Policy Rules:
{rules_text}
{numeric_limits}

TASK: Determine if the value complies with the policy rules by:
1. Identifying which specific rule applies to this context
2. Checking if the field value complies with that rule
3. Returning: valid (complies), warning (needs attention), or error (violates)

Provide clear reasoning explaining which rule applies and whether the value meets the requirement.

RESPOND IN JSON:
{{
  "status": "valid/warning/error",
  "message": "user-friendly message",
  "reasoning": "explanation of which rule applies and whether value complies"
}}"""

        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            self.log_entry("GEMINI_RAW", f"Response object: {response}")
            
            # Check if response has content
            if not hasattr(response, 'text') or not response.text:
                self.log_entry("ERROR", "Gemini response empty or no text attribute")
                # Return default valid decision
                return {
                    'status': 'valid',
                    'message': '✓ Field validated (no gemini response)',
                    'reasoning': 'Default approval due to empty response'
                }
            
            response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            
            self.log_entry("GEMINI_RESPONSE", f"Raw response: {response_text}")
            
            decision = json.loads(response_text.strip())
            self.log_entry("GEMINI_DECISION", f"Parsed: status={decision.get('status')}, reasoning={decision.get('reasoning')[:100]}")
            
            # Verify numeric comparison if the field value is numeric
            try:
                field_value_num = float(field_value)
                # Extract allowance values from the rules
                import re
                allowances = re.findall(r':\s*(\d+)(?:\s|$)', ' '.join(found_rules))
                if allowances:
                    allowances_nums = [int(a) for a in allowances]
                    max_allowance = max(allowances_nums)
                    # If the reasoning mentions a specific allowance, try to extract it
                    reasoning = decision.get('reasoning', '').lower()
                    for allowance_val in allowances_nums:
                        if str(allowance_val) in reasoning:
                            max_allowance = allowance_val
                            break
                    
                    # Override decision if numeric comparison is clear
                    if field_value_num <= max_allowance and decision['status'] != 'valid':
                        self.log_entry("OVERRIDE", f"Overriding gemini decision: {field_value_num} <= {max_allowance}, setting to valid")
                        decision['status'] = 'valid'
                        decision['message'] = f"✓ {field_name} ({field_value}) is within allowance ({max_allowance})"
                    elif field_value_num > max_allowance and decision['status'] != 'error':
                        self.log_entry("OVERRIDE", f"Overriding gemini decision: {field_value_num} > {max_allowance}, setting to error")
                        decision['status'] = 'error'
                        decision['message'] = f"✗ {field_name} ({field_value}) exceeds allowance ({max_allowance})"
            except:
                pass
            
            return decision
        except Exception as e:
            self.log_entry("ERROR", f"Decision generation failed: {e}")
            return {
                'status': 'valid',
                'message': '✓ Field validated',
                'reasoning': 'Default approval'
            }



    def save_log(self):
        """Save logs"""
        with open(LOG_FILE, 'a') as f:
            for entry in self.log:
                f.write(entry + '\n')


# API Endpoints
validator = FieldValidator(RULES_FILE)


@app.route('/api/validate-field', methods=['POST'])
def validate_field():
    """Field validation endpoint"""
    try:
        data = request.json
        field_name = data.get('field_name', '').strip()
        field_value = data.get('field_value', '').strip()
        previous_context = data.get('previous_context')

        if not field_name or not field_value:
            return jsonify({
                'status': 'error',
                'message': 'field_name and field_value required'
            }), 400

        result = validator.validate_field(field_name, field_value, previous_context)
        validator.save_log()

        return jsonify(result), 200

    except Exception as e:
        import traceback
        validator.log_entry("ERROR", f"API error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'Validation failed: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if validator.chroma_collection is not None else 'initializing',
        'rules_loaded': len(validator.rules_lines),
        'chroma_initialized': validator.chroma_collection is not None,
        'chroma_collection_count': validator.chroma_collection.count() if validator.chroma_collection else 0,
        'rules_file': RULES_FILE,
        'rules_file_exists': os.path.exists(RULES_FILE)
    }), 200


@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        'rules_loaded': len(validator.rules_lines),
        'chroma_ready': validator.chroma_collection is not None,
        'embeddings_count': validator.chroma_collection.count() if validator.chroma_collection else 0,
        'recent_logs': validator.log[-20:]  # Last 20 log entries
    }), 200


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Field Validation API v2 (Dynamic approach with startup init)...")
    print("="*80)
    print(f"Rules file: {RULES_FILE}")
    print(f"ChromaDB path: {CHROMA_DB_PATH}")
    print("="*80 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
