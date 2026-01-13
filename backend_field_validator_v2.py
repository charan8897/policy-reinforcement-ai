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


class FieldValidator:
    """Dynamic field validation following policy_validator.py approach"""

    def __init__(self, rules_file, max_attempts=3):
        self.rules_file = rules_file
        self.max_attempts = max_attempts
        self.log = []
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        # self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Cache for fast lookups
        self.field_classification_cache = {}
        self.field_patterns_cache = {}
        self.rules_content = None
        self.rules_lines = []
        
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
            
            # Initialize ChromaDB with embeddings
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
    
    def semantic_search_by_field(self, field_name, field_value):
        """Specialized semantic search for a field with intelligent query generation"""
        try:
            # Generate contextual query for the field
            query = self._generate_search_query(field_name, field_value)
            self.log_entry("FIELD_QUERY", f"{field_name}: {query}")
            
            # Perform semantic search
            results = self.semantic_search(query, top_k=10)
            return results
        except Exception as e:
            self.log_entry("ERROR", f"Field-based semantic search failed: {e}")
            return []
    
    def _generate_search_query(self, field_name, field_value):
        """Generate an intelligent search query for a field"""
        prompt = f"""Create a semantic search query to find policy constraints for this field.

Field Name: {field_name}
Field Value: {field_value}

TASK: Generate a SHORT search query (1-2 sentences) that will find the SPECIFIC policy rule/constraint for this field.
- Include what the field represents
- Include what constraint or requirement you're looking for
- Be specific about what type of limit or rule to find

Examples:
- For group_size=6: "how many employees can travel together limit employees travelling"
- For room_type=suite: "room upgrades policies standard room requirements"
- For flight_hours=8: "flight duration class of service business class first class"

RESPOND WITH ONLY THE QUERY (no quotes):
search_query_here"""

        try:
            response = self.model.generate_content(prompt)
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

        # Step 1: Semantic search for relevant rules
        self.log_entry("SEARCH", f"Performing semantic search for {field_name}")
        search_results = self.semantic_search_by_field(field_name, field_value)
        
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

        # Step 2: Filter and format rules
        formatted_rules = [result['line'] for result in search_results[:10]]
        rules_text = "\n".join(formatted_rules)
        
        self.log_entry("RULES_FOUND", f"Found {len(search_results)} relevant rules, using top {len(formatted_rules)}")

        # Step 3: Generate validation decision from found rules
        decision = self._generate_validation_decision(
            field_name, field_value, formatted_rules, previous_context
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



    def _generate_validation_decision(self, field_name, field_value, found_rules, previous_context):
        """Generate validation decision from found rules"""
        
        if not found_rules:
            return {
                'status': 'valid',
                'message': f'✓ {field_name} validated - no policy constraints found',
                'reasoning': 'Field accepted by default'
            }
        
        rules_text = "\n".join(found_rules[:10])
        
        prompt = f"""Based on these policy rules, validate the field value.

Field: {field_name} = {field_value}

Policy Rules:
{rules_text}

TASK: Determine if the value is:
1. Valid - complies with policy
2. Warning - acceptable but needs attention
3. Error - violates policy

RESPOND IN JSON:
{{
  "status": "valid/warning/error",
  "message": "user-friendly message",
  "reasoning": "brief explanation"
}}"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            
            decision = json.loads(response_text.strip())
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
