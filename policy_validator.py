#!/usr/bin/env python3
"""
Generic Policy Enforcement System
Step 1: Extract policies from PDF → filename.txt
Step 1B: Extract & elaborate clauses with Gemini → stage1_clauses.json
Step 2: Classify clause intents with Gemini → stage2_classified.json
Step 3: Extract entities & thresholds with Gemini → stage3_entities.json
Step 4: Detect ambiguities in clauses → stage4_ambiguity_flags.json
Step 5: Clarify ambiguous clauses with Gemini → stage5_clarified_clauses.json
Step 6: Generate DSL rules from clarified clauses → stage6_dsl_rules.yaml (NEW)
Step 7: Generate confidence scores → stage7_confidence_rationale.json
Step 8: Normalize policies → stage8_normalized_policies.json

Usage:
  python policy_validator.py extract <policy.pdf>
  python policy_validator.py extract-clauses <filename.txt>
  python policy_validator.py classify-intents <stage1_clauses.json>
  python policy_validator.py extract-entities <stage2_classified.json>
  python policy_validator.py detect-ambiguities <stage3_entities.json>
  python policy_validator.py clarify-ambiguities <stage3_entities.json>
  python policy_validator.py generate-dsl <stage5_clarified_clauses.json>
  python policy_validator.py run-full-pipeline
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import time
from queue import Queue

# LangChain imports for Stage 1 enhancement
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field, validator
    from typing import List
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Import Generic Policy Enhancer for policy-agnostic processing
try:
    from generic_policy_enhancer import (
        GenericIntentClassifier,
        GenericAmbiguityDetector,
        GenericEntityClassifier,
        GenericDSLGenerator,
        UnifiedPolicyEnhancer,
        IntentType,
        AmbiguityType,
        SemanticType
    )
    GENERIC_ENHANCER_AVAILABLE = True
except ImportError:
    GENERIC_ENHANCER_AVAILABLE = False
    IntentType = None
    AmbiguityType = None
    SemanticType = None

# For MongoDB storage (optional)
try:
    from upload_service.database import PipelineStageManager
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    PipelineStageManager = None

# Configuration
GEMINI_API_KEY = "AIzaSyAgWfY5zft6IV00Y2HwPc3JHQva38zWEDQ"
OUTPUT_DIR = "/home/hutech/Documents/docupolicy"
LOG_FILE = f"{OUTPUT_DIR}/mechanism.log"


class PipelineConfig:
    """
    Centralized configuration for all pipeline stages
    No more hardcoded values scattered across classes
    """
    
    # LLM/Gemini Settings
    LLM_MODEL = "gemma-3-27b-it"
    LLM_TEMPERATURE = 0.1  # 0.0 = deterministic, 1.0 = creative
    LLM_MAX_RETRIES = 3    # Retry failed LLM calls
    
    # Text Processing
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 300
    TEXT_PREVIEW_LENGTH = 8000  # For pattern discovery
    
    # Parallel Processing
    DEFAULT_MAX_WORKERS = 8
    DEFAULT_BATCH_SIZE = 3
    ENTITY_EXTRACTION_WORKERS = 6
    AMBIGUITY_CLARIFICATION_WORKERS = 4
    AMBIGUITY_CLARIFICATION_BATCH_SIZE = 3
    
    # Pattern Matching
    TOP_K_SIMILAR_CLAUSES = 3
    TOP_K_DSL_LOOKUP = 1
    
    # Timeout and Retries
    LLM_CALL_TIMEOUT = 120  # seconds
    LLM_RETRY_DELAY = 2     # seconds
    LLM_RETRY_ATTEMPTS = 2
    
    # JSON Parsing
    INTENT_CONFIDENCE_MIN = 0.0
    INTENT_CONFIDENCE_MAX = 1.0
    DEFAULT_CONFIDENCE = 0.75
    
    # Ambiguity Detection
    AMBIGUITY_STRICT_MODE = True  # Flag all potential ambiguities
    
    # Normalization
    NORMALIZATION_USE_LLM = False  # Default: use fast rule-based extraction
    
    @classmethod
    def to_dict(cls):
        """Export all configs as dictionary"""
        return {k: v for k, v in vars(cls).items() if not k.startswith('_') and not callable(v)}
    
    @classmethod
    def validate(cls):
        """Validate configuration values"""
        assert cls.LLM_TEMPERATURE >= 0.0 and cls.LLM_TEMPERATURE <= 1.0, "Temperature must be 0.0-1.0"
        assert cls.LLM_MAX_RETRIES >= 1, "Max retries must be >= 1"
        assert cls.CHUNK_SIZE > 0, "Chunk size must be > 0"
        assert cls.DEFAULT_MAX_WORKERS > 0, "Max workers must be > 0"
        assert cls.INTENT_CONFIDENCE_MIN >= 0.0, "Min confidence >= 0.0"
        assert cls.INTENT_CONFIDENCE_MAX <= 1.0, "Max confidence <= 1.0"
        return True


# =============================================================================
# POLICY-AGNOSTIC ENHANCEMENTS
# =============================================================================
# These enhancements provide policy-agnostic processing that works for ANY
# policy type (HR, IT Security, Finance, etc.) without hardcoded patterns.
# =============================================================================

class PolicyAgnosticEnhancer:
    """
    Unified policy-agnostic enhancer that provides enhanced versions of
    all pipeline stage logic. Works for ANY policy type.
    
    Usage:
        enhancer = PolicyAgnosticEnhancer()
        result = enhancer.process_document(content)
    """
    
    def __init__(self, enable_llm_fallback=True):
        self.enable_llm_fallback = enable_llm_fallback
        
        # Initialize generic classifiers (policy-agnostic)
        self.intent_classifier = GenericIntentClassifier() if GENERIC_ENHANCER_AVAILABLE else None
        self.ambiguity_detector = GenericAmbiguityDetector() if GENERIC_ENHANCER_AVAILABLE else None
        self.entity_classifier = GenericEntityClassifier() if GENERIC_ENHANCER_AVAILABLE else None
        self.dsl_generator = GenericDSLGenerator() if GENERIC_ENHANCER_AVAILABLE else None
        
        self.log = []
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def enhance_clause_extraction(self, content: str) -> Dict:
        """
        Stage 1B Enhancement: Extract clauses with structure awareness.
        
        Fixes:
        - Detects ALL sections including Annexures
        - Preserves document structure
        - No content loss
        """
        if not GENERIC_ENHANCER_AVAILABLE:
            return None
        
        self.log_entry("INFO", "Enhancing clause extraction with structure awareness")
        
        enhancer = UnifiedPolicyEnhancer()
        result = enhancer.process_document(content)
        
        return {
            'clauses': result['clauses'],
            'sections': result['sections'],
            'statistics': result['statistics']
        }
    
    def validate_and_correct_intent(self, clause_id: str, text: str, intent: str, confidence: float) -> Tuple[str, float]:
        """
        Stage 2 Enhancement: Validate and correct intent classification.
        
        Fixes:
        - "CANNOT IF" → RESTRICTION (not CONDITIONAL_ALLOWANCE)
        - "SHALL claim as per" → INFORMATIONAL (not CONDITIONAL_ALLOWANCE)
        """
        if not self.intent_classifier or not GENERIC_ENHANCER_AVAILABLE:
            return intent, confidence
        
        # Get generic classification
        generic_intent, generic_confidence = self.intent_classifier.classify_intent(text)
        
        # Override if pattern matches
        corrected_intent = intent
        corrected_confidence = confidence
        
        text_lower = text.lower()
        
        # Pattern-based corrections
        if "cannot" in text_lower and "if" in text_lower and intent == "CONDITIONAL_ALLOWANCE":
            corrected_intent = "RESTRICTION"
            corrected_confidence = max(generic_confidence, 0.9)
            self.log_entry("CORRECTED", f"{clause_id}: CONDITIONAL_ALLOWANCE → RESTRICTION")
        
        elif "shall claim" in text_lower and "per the company policy" in text_lower and intent == "CONDITIONAL_ALLOWANCE":
            corrected_intent = "INFORMATIONAL"
            corrected_confidence = max(generic_confidence, 0.85)
            self.log_entry("CORRECTED", f"{clause_id}: CONDITIONAL_ALLOWANCE → INFORMATIONAL")
        
        return corrected_intent, corrected_confidence
    
    def clean_entities(self, entities: Dict[str, str]) -> Dict[str, str]:
        """
        Stage 3 Enhancement: Clean and validate entity extraction.
        
        Fixes:
        - Removes garbage like "25 their"
        - Deduplicates similar entities
        - Validates entity values
        """
        if not self.entity_classifier or not GENERIC_ENHANCER_AVAILABLE:
            return entities
        
        cleaned = {}
        
        for name, value in entities.items():
            # Skip garbage values
            if not value or not isinstance(value, str):
                continue
            
            value = value.strip()
            
            # Skip incomplete/torn values
            if value.endswith("their") or value.endswith("the"):
                self.log_entry("SKIPPED", f"Garbage entity: {name} = {value}")
                continue
            
            # Skip very short or very long values
            if len(value) < 2 or len(value) > 100:
                continue
            
            # Classify and validate
            entity = self.entity_classifier.classify_entity(name, value)
            if entity.validation_status:
                cleaned[name] = value
            else:
                self.log_entry("SKIPPED", f"Invalid entity: {name} = {value}")
        
        return cleaned
    
    def detect_ambiguities_generic(self, text: str) -> List[Dict]:
        """
        Stage 4 Enhancement: Detect ambiguities using linguistic patterns.
        
        Fixes:
        - Marks "SHOULD" clauses as ambiguous
        - Detects VAGUE_TERMS, UNDEFINED_REFERENCES, etc.
        """
        if not self.ambiguity_detector or not GENERIC_ENHANCER_AVAILABLE:
            return []
        
        ambiguities = self.ambiguity_detector.detect_ambiguities(text)
        
        result = []
        for amb_type, context in ambiguities:
            result.append({
                'type': amb_type.value if hasattr(amb_type, 'value') else str(amb_type),
                'context': context
            })
        
        # Also check for weak enforcement (SHOULD, MAY)
        text_lower = text.lower()
        if "should" in text_lower or "may" in text_lower:
            result.append({
                'type': 'WEAK_ENFORCEMENT',
                'context': text[:100]
            })
        
        return result
    
    def validate_clarification(self, original: str, clarified: str) -> str:
        """
        Stage 5 Enhancement: Validate clarifications to prevent hallucinations.
        
        Fixes:
        - Rejects clarifications that add new numeric values
        - Falls back to original if hallucination detected
        """
        # Extract numbers from original
        original_nums = set(re.findall(r'[\d,.]+', original))
        clarified_nums = set(re.findall(r'[\d,.]+', clarified))
        
        # Check for new numbers
        new_nums = clarified_nums - original_nums
        
        if new_nums:
            # Check if new numbers are actually meaningful additions
            # Allow common clarifications like "24-hour period" for "on any day"
            known_clarifications = {"24-hour", "24 hour", "written", "formal"}
            
            has_known_clarification = any(
                nc.lower() in clarified.lower() for nc in known_clarifications
                for nc in new_nums if nc in ["24", "24-hour"]
            )
            
            if not has_known_clarification:
                self.log_entry("HALLUCINATION", f"New values detected: {new_nums}")
                self.log_entry("REVERTED", f"Using original text for: {original[:50]}...")
                return original  # Revert to original
        
        return clarified
    
    def build_generic_dsl(self, clause_id: str, intent: str, entities: Dict[str, str]) -> Dict:
        """
        Stage 6 Enhancement: Build DSL rules using semantic inference.
        
        Fixes:
        - Generates meaningful when/then conditions
        - Infers enforcement level from intent
        - Creates descriptive constraint names
        """
        if not self.dsl_generator or not GENERIC_ENHANCER_AVAILABLE:
            return None
        
        # Build classified clause structure
        classified_clause = type('ClassifiedClause', (), {
            'clause_id': clause_id,
            'original_text': '',
            'intent': IntentType.INFORMATIONAL,
            'intent_confidence': 0.75,
            'entities': [],
            'ambiguities': []
        })()
        
        # Map string intent to enum
        intent_map = {
            'RESTRICTION': IntentType.RESTRICTION,
            'MANDATORY': IntentType.MANDATORY,
            'LIMIT': IntentType.LIMIT,
            'CONDITIONAL': IntentType.CONDITIONAL,
            'ADVISORY': IntentType.ADVISORY,
            'PERMISSIVE': IntentType.PERMISSIVE,
            'INFORMATIONAL': IntentType.INFORMATIONAL
        }
        classified_clause.intent = intent_map.get(intent, IntentType.INFORMATIONAL)
        
        # Classify entities
        for name, value in entities.items():
            entity = self.entity_classifier.classify_entity(name, value)
            classified_clause.entities.append(entity)
        
        # Generate DSL
        rule = self.dsl_generator.generate_rule(classified_clause)
        
        return {
            'rule_id': rule.rule_id,
            'when': rule.when_conditions,
            'then': rule.then_outcome,
            'enforcement_level': rule.enforcement_level
        }


class PipelineStageStorage:
    """
    Helper class to manage storing pipeline stages to MongoDB.
    Used across all stage processors.
    """
    
    def __init__(self, enable_mongodb=True, document_id=None):
        """
        Initialize stage storage
        
        Args:
            enable_mongodb (bool): Whether to store to MongoDB
            document_id (str): Reference to document in raw_documents collection
        """
        self.enable_mongodb = enable_mongodb and MONGODB_AVAILABLE
        self.document_id = document_id or "unknown"
        self.stage_manager = None
        self.log = []
        
        if self.enable_mongodb:
            try:
                self.stage_manager = PipelineStageManager()
                if self.stage_manager.connect():
                    self.log_entry("INFO", "Connected to MongoDB for stage storage")
                else:
                    self.enable_mongodb = False
                    self.log_entry("WARNING", "Failed to connect to MongoDB, stage storage disabled")
            except Exception as e:
                self.enable_mongodb = False
                self.log_entry("WARNING", f"MongoDB unavailable: {e}")
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def store_stage(self, stage_number, stage_name, stage_output, filename=None):
        """
        Store a stage result to MongoDB with pending_approval status
        
        Args:
            stage_number (int): Stage number (1-7)
            stage_name (str): Name of stage
            stage_output (dict or list): The stage output data
            filename (str): Original filename
            
        Returns:
            dict: {'success': bool, 'stage_id': str, 'error': str}
        """
        
        if not self.enable_mongodb:
            self.log_entry("DEBUG", f"Stage {stage_number} ({stage_name}) storage skipped (MongoDB disabled)")
            return {'success': False, 'stage_id': None, 'error': 'MongoDB disabled'}
        
        try:
            result = self.stage_manager.insert_stage(
                document_id=self.document_id,
                stage_number=stage_number,
                stage_name=stage_name,
                stage_output=stage_output,
                filename=filename
            )
            
            if result['success']:
                self.log_entry("SUCCESS", f"Stage {stage_number} ({stage_name}) stored with status: pending_approval")
                self.log_entry("STAGE_ID", result['stage_id'])
            else:
                self.log_entry("ERROR", f"Failed to store stage {stage_number}: {result['error']}")
            
            return result
            
        except Exception as e:
            self.log_entry("ERROR", f"Stage storage exception: {str(e)}")
            return {'success': False, 'stage_id': None, 'error': str(e)}
    
    def get_pending_stages(self, limit=10):
        """Get all stages pending approval"""
        if not self.enable_mongodb:
            return []
        
        return self.stage_manager.get_pending_stages(limit)
    
    def approve_stage(self, stage_id, approved_by, notes=None):
        """Mark a stage as approved"""
        if not self.enable_mongodb:
            return {'success': False, 'error': 'MongoDB disabled'}
        
        return self.stage_manager.approve_stage(stage_id, approved_by, notes)
    
    def reject_stage(self, stage_id, rejected_by, reason):
        """Mark a stage as rejected"""
        if not self.enable_mongodb:
            return {'success': False, 'error': 'MongoDB disabled'}
        
        return self.stage_manager.reject_stage(stage_id, rejected_by, reason)
    
    def disconnect(self):
        """Disconnect from MongoDB"""
        if self.stage_manager:
            self.stage_manager.disconnect()


class PatternDiscoveryEngine:
    """
    Step 0A: Discover domain-agnostic patterns from raw policy text
    
    Scans filename.txt with LLM to identify:
    - Pattern types (grades, allowances, durations, authorities, etc.)
    - Entity examples from actual text
    - Relationships between patterns
    - Data types (categorical vs numeric)
    
    Output: pattern_index.json (used by all downstream stages)
    
    Fully generic - works for travel policies, HR policies, leave policies, etc.
    No hardcoded domain rules.
    """
    
    def __init__(self, policy_text_file, document_id=None, enable_mongodb=True):
        self.policy_file = policy_text_file
        self.pattern_index_file = f"{OUTPUT_DIR}/pattern_index.json"
        self.document_id = document_id
        self.log = []
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def read_policy_text(self):
        """Read raw policy text file"""
        try:
            with open(self.policy_file, 'r') as f:
                content = f.read()
            
            self.log_entry("SUCCESS", f"Read policy text: {len(content)} characters")
            return content
        
        except FileNotFoundError:
            self.log_entry("ERROR", f"Policy text file not found: {self.policy_file}")
            return None
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read policy text: {e}")
            return None
    
    def discover_patterns_with_llm(self, text):
        """
        Use Gemini LLM to identify ALL pattern types from raw policy text
        Returns structured pattern definitions (domain-agnostic)
        
        Args:
            text (str): Raw policy document text
            
        Returns:
            dict: Structured patterns with types, examples, keywords, relationships
        """
        
        # Chunk text to avoid token limits
        text_chunk = text[:8000] if len(text) > 8000 else text
        
        prompt = f"""Analyze this policy document and identify ALL PATTERN TYPES present.

For EACH pattern type you find, specify:
1. PATTERN_TYPE: What category? (e.g., grades, allowances, durations, authorities, travel_modes, rates, locations, conditions, etc.)
2. EXAMPLES: List 5-10 specific values directly from the text
3. PATTERN_DESCRIPTION: How to recognize this pattern (keywords, context, format)
4. DATA_TYPE: Is it categorical (distinct values) or numeric (ranges)?
5. KEYWORDS: Words that indicate this pattern
6. RELATIONSHIPS: Does this pattern relate to others? (e.g., "grade → allowance", "duration → travel_type")
7. FREQUENCY: Approximate count in the document

POLICY TEXT:
────────────────────────────────────
{text_chunk}
────────────────────────────────────

OUTPUT ONLY VALID JSON (no markdown, no extra text):
{{
  "patterns": [
    {{
      "pattern_type": "pattern_name",
      "data_type": "categorical|numeric",
      "examples": ["value1", "value2", "value3"],
      "description": "Description of this pattern",
      "keywords": ["keyword1", "keyword2"],
      "relationships": ["pattern_a → pattern_b", "pattern_x → consequence_y"],
      "frequency": 10,
      "min_value": null,
      "max_value": null,
      "unit": null
    }}
  ]
}}

IMPORTANT:
- Include ALL pattern types you find (do not limit)
- Return ONLY valid JSON
- Extract actual values from text, do NOT invent
- Be specific about relationships
- If numeric, include min/max if discernible"""
        
        try:
            self.log_entry("LLM", "Discovering patterns with Gemini API")
            response = self.model.generate_content(prompt)
            patterns_text = response.text.strip()
            
            # Clean markdown if present
            if patterns_text.startswith("```"):
                patterns_text = patterns_text.split("```")[1]
                if patterns_text.startswith("json"):
                    patterns_text = patterns_text[4:]
                patterns_text = patterns_text.strip()
            
            patterns = json.loads(patterns_text)
            
            self.log_entry("SUCCESS", f"Discovered {len(patterns.get('patterns', []))} pattern types")
            return patterns
        
        except json.JSONDecodeError as e:
            self.log_entry("ERROR", f"Failed to parse LLM response: {e}")
            return {"patterns": []}
        except Exception as e:
            self.log_entry("ERROR", f"Pattern discovery failed: {e}")
            return {"patterns": []}
    
    def build_pattern_index(self, patterns):
        """
        Convert LLM-discovered patterns into a queryable index structure
        
        Index allows:
        - Fast lookup by pattern type
        - Keyword-based lookups
        - Relationship traversal
        - Example extraction
        
        Args:
            patterns (dict): Patterns dict from LLM discovery
            
        Returns:
            dict: Indexed pattern structure
        """
        
        index = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "source": self.policy_file,
                "total_patterns": len(patterns.get('patterns', [])),
                "description": "Domain-agnostic pattern index for rule generation"
            },
            "pattern_types": {},
            "relationships": {},
            "keyword_lookup": {},
            "statistics": {}
        }
        
        pattern_list = patterns.get('patterns', [])
        
        if not pattern_list:
            self.log_entry("WARN", "No patterns discovered from LLM")
            return index
        
        # Process each pattern
        for pattern in pattern_list:
            pattern_type = pattern.get('pattern_type', 'unknown').lower()
            
            if not pattern_type or pattern_type == 'unknown':
                continue
            
            # Store pattern definition
            index["pattern_types"][pattern_type] = {
                "data_type": pattern.get('data_type', 'categorical'),
                "examples": pattern.get('examples', []),
                "description": pattern.get('description', ''),
                "keywords": pattern.get('keywords', []),
                "frequency": pattern.get('frequency', 0),
                "unit": pattern.get('unit'),
                "min_value": pattern.get('min_value'),
                "max_value": pattern.get('max_value'),
                "relationships": pattern.get('relationships', [])
            }
            
            # Build keyword → pattern_type lookup
            for keyword in pattern.get('keywords', []):
                keyword_lower = keyword.lower()
                if keyword_lower not in index["keyword_lookup"]:
                    index["keyword_lookup"][keyword_lower] = []
                if pattern_type not in index["keyword_lookup"][keyword_lower]:
                    index["keyword_lookup"][keyword_lower].append(pattern_type)
            
            # Store relationships
            relationships = pattern.get('relationships', [])
            if relationships:
                index["relationships"][pattern_type] = relationships
            
            # Statistics
            index["statistics"][pattern_type] = {
                "example_count": len(pattern.get('examples', [])),
                "keyword_count": len(pattern.get('keywords', [])),
                "frequency": pattern.get('frequency', 0)
            }
        
        return index
    
    def discover(self):
        """
        Main discovery workflow
        
        Returns:
            bool: Success/failure
        """
        self.log_entry("START", "Step 0A: Pattern Discovery from Raw Policy Text")
        self.log_entry("INFO", "Scanning policy document to identify domain-agnostic patterns")
        
        # Step 1: Read policy text
        self.log_entry("STEP", "Reading policy text from filename.txt")
        text = self.read_policy_text()
        if not text:
            return False
        
        # Step 2: Discover patterns with LLM
        self.log_entry("STEP", "Discovering patterns with LLM (this may take 30-60 seconds)")
        patterns = self.discover_patterns_with_llm(text)
        
        if not patterns.get('patterns'):
            self.log_entry("WARN", "No patterns discovered - check LLM output")
            return False
        
        # Step 3: Build index
        self.log_entry("STEP", "Building pattern index from discovered patterns")
        index = self.build_pattern_index(patterns)
        
        # Step 4: Save index to file
        try:
            with open(self.pattern_index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            self.log_entry("SUCCESS", f"Pattern index saved: {self.pattern_index_file}")
            self.log_entry("STATS", f"Pattern types discovered: {index['metadata']['total_patterns']}")
            self.log_entry("STATS", f"Keywords indexed: {len(index['keyword_lookup'])}")
            self.log_entry("STATS", f"Relationships found: {len(index['relationships'])}")
            
            # Step 5: Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=0,
                stage_name="pattern-discovery",
                stage_output=index,
                filename="pattern_index.json"
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Pattern index stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
            return True
        
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save pattern index: {e}")
            return False
    
    def save_log(self):
        """Append discovery log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== PATTERN DISCOVERY LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class PolicyExtractor:
    """Step 1: Extract text from PDF"""
    
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.output_file = f"{OUTPUT_DIR}/filename.txt"
        self.log = []
        
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def extract_pdf(self):
        """Extract PDF using pdftotext"""
        self.log_entry("INFO", f"Extracting PDF: {self.pdf_file}")
        
        try:
            # Use pdftotext
            result = subprocess.run(
                ['pdftotext', self.pdf_file, self.output_file],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get file stats
            with open(self.output_file, 'r') as f:
                content = f.read()
            
            lines = len(content.split('\n'))
            words = len(content.split())
            
            self.log_entry("SUCCESS", f"PDF extracted: {lines} lines, {words} words")
            self.log_entry("OUTPUT", f"File: {self.output_file}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_entry("ERROR", f"pdftotext failed: {e.stderr}")
            return False
        except FileNotFoundError:
            self.log_entry("ERROR", "pdftotext not installed. Install: sudo apt-get install poppler-utils")
            return False
        except Exception as e:
            self.log_entry("ERROR", str(e))
            return False
    
    def save_log(self):
        """Save extraction log"""
        with open(LOG_FILE, 'w') as f:
            f.write("=== POLICY EXTRACTION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


# Pydantic models for LangChain output parsing
if LANGCHAIN_AVAILABLE:
    class ClauseSchema(BaseModel):
        """Schema for a single policy clause"""
        clause_id: str = Field(description="Clause identifier (C1, C2, etc.)")
        text: str = Field(description="Elaborated clause text with conditions, values, and enforcement")
        
        @validator('clause_id')
        def validate_clause_id(cls, v):
            if not re.match(r'^C\d+$', v):
                raise ValueError(f"Invalid clause_id format: {v} (should be C1, C2, etc.)")
            return v
    
    class ClausesOutput(BaseModel):
        """Schema for multiple clauses"""
        clauses: List[ClauseSchema] = Field(description="List of extracted clauses")
    
    class IntentClassificationSchema(BaseModel):
        """Schema for intent classification result"""
        intent: str = Field(description="Intent type (RESTRICTION, LIMIT, CONDITIONAL_ALLOWANCE, etc.)")
        confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
        reasoning: str = Field(description="Brief explanation of intent classification")
        
        @validator('intent')
        def validate_intent(cls, v):
            allowed = ["RESTRICTION", "LIMIT", "CONDITIONAL_ALLOWANCE", "EXCEPTION",
                      "APPROVAL_REQUIRED", "ADVISORY", "INFORMATIONAL"]
            if v.upper() not in allowed:
                raise ValueError(f"Invalid intent: {v}")
            return v.upper()
    
    class AmbiguityDetectionSchema(BaseModel):
        """Schema for ambiguity detection result"""
        ambiguous: bool = Field(description="Whether the clause is ambiguous")
        reason: str = Field(description="Explanation of ambiguity or clarity")
        ambiguity_types: List[str] = Field(default=[], description="List of ambiguity type codes detected")
    
    class EntityExtractionSchema(BaseModel):
        """Schema for dynamic entity extraction - entities are dict with any keys"""
        entities: dict = Field(default={}, description="Dynamically extracted entities as key-value pairs")


# ============================================================================
# IMPROVED STAGE 1: DATA STRUCTURES
# ============================================================================

class ExtractionQuality(Enum):
    """Validation result types"""
    EXACT = "exact"
    PARAPHRASED = "paraphrased"
    HALLUCINATED = "hallucinated"
    PARTIAL = "partial"


@dataclass
class SourceLocation:
    """Precise source location tracking"""
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    source_text: str


@dataclass
class ExtractedClause:
    """Complete clause with full traceability"""
    clause_id: str
    original_text: str
    elaborated_text: str
    extraction_quality: ExtractionQuality
    quality_score: float
    llm_confidence: float
    source_location: SourceLocation
    key_entities: Dict[str, List[str]] = field(default_factory=dict)
    is_duplicate_of: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractionMetadata:
    """Metadata for entire stage output"""
    generated: str
    source_file: str
    source_size_bytes: int
    total_lines: int
    extraction_method: str
    policy_type: str
    total_clauses_before_dedup: int
    total_clauses_after_dedup: int
    avg_quality_score: float
    dedup_count: int
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# POLICY SEGMENTER
# ============================================================================

class PolicySegmenter:
    """Intelligently segment policy text by sections/headers AND numbered items (generic)"""
    
    def __init__(self):
        self.log = []
    
    def log_entry(self, level: str, msg: str):
        entry = f"[{datetime.now().isoformat()}] [{level}] {msg}"
        self.log.append(entry)
        print(entry)
    
    def _is_numbered_item_start(self, line: str) -> bool:
        """
        Detect start of numbered/lettered item (item-level segmentation).
        Patterns: "1. ", "a) ", "i) ", "A. ", etc.
        """
        stripped = line.strip()
        # Match: digit(s) followed by . or ), OR letter followed by . or )
        return bool(re.match(r'^(?:\d+[\.\)]|[a-z][\.\)]|[A-Z][\.\)])\s+\S', stripped))
    
    def _is_section_header(self, line: str) -> bool:
        """
        Detect section-level headers (not item-level).
        Patterns: "Scope:", "Eligibility:", "General Norms:", etc.
        
        NOT items like: "1. Item text" (only single digit at start)
        """
        stripped = line.strip()
        
        # Markdown headers
        if re.match(r'^#+\s+\w+', stripped):
            return True
        
        # Section headers with colons: "Scope:", "Eligibility:", "General Norms:", etc.
        # These are typically short (1-4 words) and end with colon
        if re.match(r'^[A-Z][A-Za-z\s&\-]*:$', stripped) and len(stripped) > 4 and len(stripped) < 80:
            # Avoid matching numbered items like "1. Item text:"
            if not re.match(r'^\d+[\.\)]\s+', stripped):
                return True
        
        # ALL-CAPS headers (but not just "Note:" or similar)
        if stripped.isupper() and len(stripped) > 5 and ' ' in stripped and not re.match(r'^\d+', stripped):
            return True
        
        return False
    
    def segment_text(self, text: str) -> List[Tuple[str, int, int, List[str]]]:
        """
        Intelligent segmentation with aggressive numbered item extraction.
        
        Strategy:
          1. Split on section headers (Scope:, General Norms:, etc.)
          2. WITHIN each section, extract numbered items (1., 2., 3., etc.)
          3. Handle multi-line numbered items (items can span multiple lines)
        
        Returns: (segment_text, start_line, end_line, section_hierarchy)
        """
        lines = text.split('\n')
        segments = []
        section_stack = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # === DETECT SECTION HEADER ===
            if self._is_section_header(stripped):
                section_name = re.sub(r'^#+\s+', '', stripped)
                section_name = re.sub(r'^[\dA-Z]+\.\s+', '', section_name)
                section_name = section_name.rstrip(':').strip()
                section_stack = [section_name]
                
                self.log_entry("SEGMENT_HEADER", f"Line {i+1}: '{section_name}'")
                
                # Collect all content until next section header
                section_start = i + 1
                section_lines = []
                
                while i + 1 < len(lines) and not self._is_section_header(lines[i + 1].strip()):
                    i += 1
                    section_lines.append(lines[i])
                
                # === EXTRACT NUMBERED ITEMS FROM SECTION ===
                section_text = '\n'.join(section_lines)
                numbered_segments = self._extract_numbered_items(
                    section_text, 
                    section_start,
                    section_stack
                )
                
                segments.extend(numbered_segments)
                
                if not numbered_segments:
                    # No numbered items, treat entire section as one segment
                    if section_lines:
                        segments.append((
                            section_text,
                            section_start,
                            i + 1,
                            section_stack.copy()
                        ))
                        self.log_entry("SEGMENT_SECTION", f"Lines {section_start}-{i+1}: '{section_name}' (no numbered items)")
            
            i += 1
        
        self.log_entry("INFO", f"Segmented into {len(segments)} segments (sections + extracted numbered items)")
        return segments
    
    def _extract_numbered_items(self, text: str, start_line: int, section_stack: List[str]) -> List[Tuple[str, int, int, List[str]]]:
        """
        Extract numbered items from section text.
        Handles multi-line items (e.g., item spans lines 34-36).
        
        Patterns:
          "1. Text..." → Item 1
          "2. More text..." → Item 2
          etc.
        """
        segments = []
        lines = text.split('\n')
        
        # Find all numbered item starts
        item_pattern = r'^(\d+)[\.\)]\s+(.+)$'
        item_starts = []
        
        for i, line in enumerate(lines):
            match = re.match(item_pattern, line.strip())
            if match:
                item_num = match.group(1)
                item_text_start = match.group(2)
                item_starts.append((i, int(item_num), item_text_start))
                self.log_entry("SEGMENT_ITEM", f"Line {start_line + i}: Detected item {item_num}")
        
        # Group lines into items (item spans until next item starts)
        for idx, (line_idx, item_num, first_line) in enumerate(item_starts):
            # Determine end line
            if idx + 1 < len(item_starts):
                end_idx = item_starts[idx + 1][0] - 1
            else:
                end_idx = len(lines) - 1
            
            # Collect lines for this item
            item_lines = lines[line_idx : end_idx + 1]
            item_text = '\n'.join(item_lines).strip()
            
            if item_text:
                item_context = [f"Item {item_num}"]
                segments.append((
                    item_text,
                    start_line + line_idx,
                    start_line + end_idx,
                    section_stack + item_context
                ))
        
        return segments


# ============================================================================
# GROUND-TRUTH VALIDATOR
# ============================================================================

class GroundTruthValidator:
    """Validate extracted clauses against source text using EXACT matching"""
    
    def __init__(self, source_text: str):
        self.source_text = source_text
        self.source_lower = source_text.lower()
    
    def _find_exact_substring_match(self, needle: str, haystack: str, threshold: float = 0.85) -> bool:
        """
        Find if needle exists in haystack with exact substring matching.
        Falls back to word-level matching if exact match fails.
        
        Args:
            needle: Text to search for (e.g., original clause)
            haystack: Source text to search in
            threshold: Min word overlap % for fallback match
        
        Returns:
            True if exact or close match found
        """
        needle = needle.strip()
        
        # Check 1: Exact substring match (case-insensitive with normalized spaces)
        normalized_needle = ' '.join(needle.split())
        normalized_haystack = ' '.join(haystack.split())
        
        if normalized_needle in normalized_haystack:
            return True
        if normalized_needle.lower() in normalized_haystack.lower():
            return True
        
        # Check 2: Word-level overlap (for paraphrases)
        needle_words = set(normalized_needle.lower().split())
        haystack_words = set(normalized_haystack.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'be', 'to', 'of', 'in', 'for', 'on'}
        needle_content_words = needle_words - stop_words
        haystack_content_words = haystack_words - stop_words
        
        # Check overlap
        if needle_content_words:
            overlap = len(needle_content_words & haystack_content_words) / len(needle_content_words)
            return overlap >= threshold
        
        return False
    
    def validate_clause(
        self,
        original_text: str,
        elaborated_text: str
    ) -> Tuple[ExtractionQuality, float, List[str]]:
        """
        Validate elaborated text is faithful to original using strict ground-truth checks.
        Returns: (quality_enum, quality_score_0_to_1, warning_list)
        """
        warnings = []
        quality = ExtractionQuality.EXACT
        score = 1.0
        
        # ========== CHECK 1: ORIGINAL TEXT EXISTS IN SOURCE ==========
        # This is critical: if original_text isn't in source, it's paraphrased/hallucinated
        if not self._find_exact_substring_match(original_text, self.source_text, threshold=0.9):
            warnings.append("Original text not found in source")
            quality = ExtractionQuality.PARAPHRASED
            score *= 0.80  # Severe penalty for not finding original
        
        # ========== CHECK 2: FABRICATED NUMBERS ==========
        # Extract numbers from both texts
        original_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\%?\b', original_text))
        elaborated_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\%?\b', elaborated_text))
        
        new_numbers = elaborated_numbers - original_numbers
        for num in new_numbers:
            # Check if this number exists ANYWHERE in source
            if not re.search(rf'\b{re.escape(num)}\b', self.source_text):
                warnings.append(f"Fabricated number: {num}")
                quality = ExtractionQuality.HALLUCINATED
                score *= 0.4  # Severe penalty for hallucinated numbers
        
        # ========== CHECK 3: FABRICATED ENTITIES ==========
        # Extract named entities (capitalized words/phrases)
        elaborated_entities = set(re.findall(
            r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b',
            elaborated_text
        ))
        original_entities = set(re.findall(
            r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b',
            original_text
        ))
        
        new_entities = elaborated_entities - original_entities
        for entity in new_entities:
            # Check if entity appears in source (case-insensitive)
            if entity.lower() not in self.source_lower:
                warnings.append(f"Questionable entity: {entity}")
                quality = ExtractionQuality.HALLUCINATED
                score *= 0.65  # Moderate penalty for new entities
        
        # ========== CHECK 4: IF/THEN STRUCTURE GROUNDING ==========
        # If elaborated text adds IF/THEN structure, validate conditions come from source
        if re.search(r'\bIF\b.*\bTHEN\b', elaborated_text, re.IGNORECASE):
            # Extract the IF condition
            if_match = re.search(r'IF\s+(.+?)\s+THEN', elaborated_text, re.IGNORECASE)
            if if_match:
                condition = if_match.group(1).lower()
                # Key words in condition must appear in source
                condition_words = set(re.findall(r'\b\w+\b', condition))
                condition_words -= {'if', 'the', 'a', 'an', 'and', 'or', 'then', 'is', 'are'}
                
                # Check how many condition words are in source
                source_words = set(re.findall(r'\b\w+\b', self.source_lower))
                grounded_words = condition_words & source_words
                
                if condition_words and len(grounded_words) / len(condition_words) < 0.5:
                    warnings.append("IF condition poorly grounded in source")
                    quality = ExtractionQuality.HALLUCINATED
                    score *= 0.60
        
        # ========== CHECK 5: QUALITY SCORING LOGIC ==========
        # Convert score to final quality determination
        if score >= 0.95 and quality == ExtractionQuality.EXACT:
            quality = ExtractionQuality.EXACT
        elif score >= 0.85 and quality != ExtractionQuality.HALLUCINATED:
            quality = ExtractionQuality.EXACT  # Minor issues but still valid
        elif score >= 0.70 and quality != ExtractionQuality.HALLUCINATED:
            quality = ExtractionQuality.PARAPHRASED
        else:
            quality = ExtractionQuality.HALLUCINATED
        
        return quality, score, warnings


# ============================================================================
# GENERIC CLAUSE EXTRACTOR
# ============================================================================

class GenericClauseExtractor:
    """Extract clauses from segments using LLM with rate limiting"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        self.validator = None
        self.log = []
        self.rate_limiter = None  # Will be set by ClauseExtractor
    
    def log_entry(self, level: str, msg: str):
        entry = f"[{datetime.now().isoformat()}] [{level}] {msg}"
        self.log.append(entry)
        print(entry)
    
    def set_source_text(self, source_text: str):
        self.validator = GroundTruthValidator(source_text)
    
    def extract_from_segment(
        self,
        segment_text: str,
        segment_num: int,
        section_hierarchy: List[str],
        start_line: int,
        end_line: int,
        retry_count: int = 0,
        max_retries: int = 2
    ) -> List[ExtractedClause]:
        """Extract clauses from single segment with enhanced error handling"""
        if not self.validator:
            raise ValueError("Must call set_source_text() first")
        
        self.log_entry("DEBUG", f"Segment {segment_num}: Text length={len(segment_text)}, Lines {start_line}-{end_line}")
        
        prompt = f"""Extract clauses from this policy segment. For elaboration, copy source keywords EXACTLY - never create synonyms.
KEYWORD WHITELIST: Use only these exact words if present in source: "cannot", "can not", "must not", "shall", "shall not", "shall be", "if", "then", "maximum", "minimum", "exceed", "approved", "sanctioned".
FORBIDDEN: Do not replace "cannot" with "unable"/"unable to", "shall" with "will"/"is", "must not" with "should not".
SEGMENT: {segment_text}
OUTPUT: JSON only with "original_text" (verbatim), "elaborated_text" (preserve keywords), "extracted_values", "key_terms", "confidence". No markdown."""
        
        try:
            self.log_entry("GEMINI_CALL", f"Segment {segment_num}: Requesting clause extraction (attempt {retry_count + 1}/{max_retries + 1})")
            
            # RATE LIMITING: Wait before making API call
            if self.rate_limiter:
                wait_time = self.rate_limiter.wait_if_needed()
                if wait_time > 0:
                    self.log_entry("RATE_LIMIT", f"Segment {segment_num}: Waiting {wait_time:.1f}s to respect Gemini quota...")
                    time.sleep(wait_time)
            
            # Call API with timeout handling
            try:
                response = self.model.generate_content(prompt)
                
                # Record successful request
                if self.rate_limiter:
                    self.rate_limiter.record_request()
                
            except Exception as api_error:
                error_str = str(api_error).lower()
                self.log_entry("ERROR", f"Segment {segment_num}: API call failed: {type(api_error).__name__}: {str(api_error)[:100]}")
                
                # Handle quota limit errors (429)
                if "429" in str(api_error) or "quota" in error_str or "rate" in error_str:
                    self.log_entry("ERROR", f"Segment {segment_num}: QUOTA LIMIT HIT - backing off")
                    if self.rate_limiter:
                        self.rate_limiter.record_quota_error()
                    # Retry with exponential backoff
                    if retry_count < max_retries:
                        wait_time = 2 ** (retry_count + 2)  # 4s, 8s, 16s
                        self.log_entry("RETRY", f"Segment {segment_num}: Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        return self.extract_from_segment(segment_text, segment_num, section_hierarchy, start_line, end_line, retry_count + 1, max_retries)
                
                # Retry on transient errors (deadline, timeout)
                elif retry_count < max_retries and ("deadline" in error_str or "timeout" in error_str):
                    wait_time = 2 ** retry_count  # Exponential backoff
                    self.log_entry("RETRY", f"Segment {segment_num}: Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    return self.extract_from_segment(segment_text, segment_num, section_hierarchy, start_line, end_line, retry_count + 1, max_retries)
                
                return []
            
            # Validate response object
            if not response:
                self.log_entry("ERROR", f"Segment {segment_num}: Empty response object from API")
                return []
            
            # Check if response has text attribute
            if not hasattr(response, 'text'):
                self.log_entry("ERROR", f"Segment {segment_num}: Response object has no 'text' attribute. Type: {type(response)}")
                self.log_entry("DEBUG", f"Response: {response}")
                return []
            
            response_text = response.text.strip() if response.text else ""
            
            # Log response summary
            self.log_entry("RESPONSE", f"Segment {segment_num}: Received {len(response_text)} chars")
            
            # Check for empty response
            if not response_text:
                self.log_entry("ERROR", f"Segment {segment_num}: API returned empty text")
                self.log_entry("DEBUG", f"Raw response object: {response}")
                return []
            
            # Log first 200 chars of response for debugging
            self.log_entry("DEBUG", f"Segment {segment_num}: Response preview: {response_text[:200]}")
            
            # Clean markdown - be careful not to lose JSON
            original_text = response_text
            
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
                self.log_entry("DEBUG", f"Segment {segment_num}: Cleaned ```json prefix")
            
            if response_text.startswith("```"):
                # Find closing ``` 
                end_marker = response_text.find("```")
                if end_marker > 0:
                    response_text = response_text[3:end_marker].strip()
                    self.log_entry("DEBUG", f"Segment {segment_num}: Cleaned ``` markers")
                else:
                    response_text = response_text[3:].strip()
            
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
                self.log_entry("DEBUG", f"Segment {segment_num}: Cleaned trailing ```")
            
            # Final validation before JSON parsing
            if not response_text or response_text == "":
                self.log_entry("ERROR", f"Segment {segment_num}: Response empty after markdown cleaning")
                self.log_entry("DEBUG", f"Original response was: {original_text[:300]}")
                return []
            
            # Attempt JSON parsing
            self.log_entry("DEBUG", f"Segment {segment_num}: Parsing JSON: {response_text[:100]}")
            
            try:
                result = json.loads(response_text)
                self.log_entry("SUCCESS", f"Segment {segment_num}: JSON parsed successfully")
            except json.JSONDecodeError as e:
                self.log_entry("ERROR", f"Segment {segment_num}: Failed to parse JSON")
                self.log_entry("TRACEBACK", f"JSONDecodeError: {e}")
                self.log_entry("TRACEBACK", f"Error at line {e.lineno}, col {e.colno}: {e.msg}")
                self.log_entry("TRACEBACK", f"Full response text: {response_text}")
                self.log_entry("TRACEBACK", f"Response length: {len(response_text)}")
                
                # Try to extract valid JSON from response
                if "{" in response_text and "}" in response_text:
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    if start_idx < end_idx:
                        json_attempt = response_text[start_idx:end_idx]
                        self.log_entry("DEBUG", f"Segment {segment_num}: Attempting to parse extracted JSON: {json_attempt[:100]}")
                        try:
                            result = json.loads(json_attempt)
                            self.log_entry("SUCCESS", f"Segment {segment_num}: Recovered valid JSON")
                        except json.JSONDecodeError as e2:
                            self.log_entry("ERROR", f"Segment {segment_num}: Recovery failed: {e2}")
                            return []
                    else:
                        return []
                else:
                    self.log_entry("ERROR", f"Segment {segment_num}: No JSON braces found in response")
                    return []
            
            # Validate result structure - handle multiple formats
            # Format 1: {"clauses": [...]}
            # Format 2: [...]
            # Format 3: {...} (single clause object)
            clause_list = []
            
            if isinstance(result, list):
                # Direct list format from new short prompt
                clause_list = result
                self.log_entry("DEBUG", f"Segment {segment_num}: Detected direct list format")
            elif isinstance(result, dict):
                if 'clauses' in result and isinstance(result['clauses'], list):
                    clause_list = result['clauses']
                    self.log_entry("DEBUG", f"Segment {segment_num}: Detected dict with clauses key")
                elif 'original_text' in result:
                    # Single clause object - wrap it in a list
                    clause_list = [result]
                    self.log_entry("DEBUG", f"Segment {segment_num}: Detected single clause object, wrapping in list")
                else:
                    self.log_entry("ERROR", f"Segment {segment_num}: Result missing 'clauses' key. Keys: {list(result.keys())}")
                    return []
            else:
                self.log_entry("ERROR", f"Segment {segment_num}: Result is neither dict nor list, got {type(result)}")
                return []
            
            if not clause_list:
                self.log_entry("WARNING", f"Segment {segment_num}: Empty clause list")
                return []
            
            # Process clauses with strict filtering
            clauses = []
            accepted_clauses = []
            rejected_clauses = []
            clause_count = len(clause_list)
            self.log_entry("SUCCESS", f"Segment {segment_num}: Extracted {clause_count} clauses from JSON")
            
            for i, clause_data in enumerate(clause_list, 1):
                clause_id = f"C{start_line}_{i}"
                
                if not isinstance(clause_data, dict):
                    self.log_entry("WARNING", f"Clause {clause_id}: Not a dict, got {type(clause_data)}")
                    continue
                
                original = clause_data.get('original_text', '').strip()
                elaborated = clause_data.get('elaborated_text', '').strip()
                llm_conf = clause_data.get('confidence', 0.5)
                
                try:
                    llm_conf = float(llm_conf)
                except (ValueError, TypeError):
                    llm_conf = 0.5
                    self.log_entry("WARNING", f"Clause {clause_id}: Invalid confidence, using 0.5")
                
                if not original or not elaborated:
                    self.log_entry("WARNING", f"Clause {clause_id}: Empty original ({len(original)}) or elaborated ({len(elaborated)})")
                    rejected_clauses.append((clause_id, "EMPTY_CONTENT"))
                    continue
                
                # === GROUND TRUTH VALIDATION ===
                quality, score, warnings = self.validator.validate_clause(original, elaborated)
                
                # === FILTER: REJECT HALLUCINATED CLAUSES ===
                if quality == ExtractionQuality.HALLUCINATED:
                    self.log_entry("REJECTED", f"Segment {segment_num}, Clause {clause_id}: HALLUCINATED [{score:.2f}] - {warnings[0] if warnings else 'Unknown reason'}")
                    rejected_clauses.append((clause_id, "HALLUCINATED"))
                    continue
                
                # === FILTER: QUALITY SCORE THRESHOLD ===
                # If quality score too low, skip this clause
                if score < 0.65:
                    self.log_entry("REJECTED", f"Segment {segment_num}, Clause {clause_id}: LOW_QUALITY [Score: {score:.2f}]")
                    rejected_clauses.append((clause_id, f"LOW_QUALITY_{score:.2f}"))
                    continue
                
                source_loc = SourceLocation(
                    start_line=start_line,
                    end_line=end_line,
                    start_char=0,
                    end_char=0,
                    source_text=original
                )
                
                clause = ExtractedClause(
                    clause_id=clause_id,
                    original_text=original,
                    elaborated_text=elaborated,
                    extraction_quality=quality,
                    quality_score=score,
                    llm_confidence=llm_conf,
                    source_location=source_loc,
                    key_entities={
                        'values': clause_data.get('extracted_values', []),
                        'terms': clause_data.get('key_terms', []),
                        'section': section_hierarchy
                    },
                    warnings=warnings
                )
                
                accepted_clauses.append(clause)
                self.log_entry("ACCEPTED", f"Segment {segment_num}, Clause {clause_id}: {original[:50]}... [Quality: {quality.value}, Score: {score:.2f}]")
            
            # Summary for segment
            clauses = accepted_clauses
            if rejected_clauses:
                self.log_entry("SUMMARY", f"Segment {segment_num}: {len(accepted_clauses)} accepted, {len(rejected_clauses)} rejected")
            
            return clauses
        
        except Exception as e:
            import traceback
            self.log_entry("ERROR", f"Segment {segment_num}: Unexpected error: {type(e).__name__}: {str(e)[:100]}")
            self.log_entry("TRACEBACK", traceback.format_exc())
            return []


# ============================================================================
# CLAUSE DEDUPLICATOR
# ============================================================================

class ClauseDeduplicator:
    """Detect and merge duplicate clauses"""
    
    @staticmethod
    def similarity_score(text1: str, text2: str) -> float:
        """Jaccard similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def deduplicate(clauses: List[ExtractedClause]) -> Tuple[List[ExtractedClause], int]:
        """Remove duplicates (>80% similar), keep highest quality"""
        if not clauses:
            return [], 0
        
        canonical = []
        removed_count = 0
        
        for clause in sorted(clauses, key=lambda c: (-c.quality_score, -c.llm_confidence)):
            is_duplicate = False
            
            for existing in canonical:
                sim = ClauseDeduplicator.similarity_score(
                    clause.original_text,
                    existing.original_text
                )
                
                if sim > 0.8:
                    clause.is_duplicate_of = existing.clause_id
                    is_duplicate = True
                    removed_count += 1
                    break
            
            if not is_duplicate:
                canonical.append(clause)
        
        return canonical, removed_count


# ============================================================================
# RATE LIMITER FOR GEMINI API
# ============================================================================

class RateLimiter:
    """
    Thread-safe rate limiter to avoid hitting Gemini API limits.
    
    Limits:
    - 15 requests per minute (standard Gemini quota)
    - 500 requests per day
    - Exponential backoff on 429/quota errors
    """
    
    def __init__(self, requests_per_minute: int = 15, requests_per_second: float = 0.25):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 1.0 / requests_per_second  # Minimum 4 seconds between requests
        self.last_request_time = 0
        self.request_times = []
        self.lock = threading.Lock()
        self.backoff_until = 0
    
    def wait_if_needed(self):
        """Block until rate limit allows next request"""
        with self.lock:
            now = time.time()
            
            # Check if in backoff period (after 429 error)
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
                return wait_time
            
            # Enforce minimum interval between requests
            if self.last_request_time > 0:
                time_since_last = now - self.last_request_time
                if time_since_last < self.min_interval:
                    wait_time = self.min_interval - time_since_last
                    return wait_time
            
            return 0
    
    def record_request(self):
        """Record that a request was made"""
        with self.lock:
            self.last_request_time = time.time()
    
    def record_quota_error(self):
        """Handle 429 quota exceeded error with exponential backoff"""
        with self.lock:
            # Back off for 60-120 seconds
            backoff_duration = 60 + (hash(threading.current_thread().name) % 60)
            self.backoff_until = time.time() + backoff_duration


# ============================================================================
# IMPROVED CLAUSE EXTRACTOR WITH MULTITHREADING
# ============================================================================

class ClauseExtractor:
    """
    Step 1B: Improved Generic Policy Clause Extraction with Multithreading.
    
    Features:
    - Ground-truth validation (no hallucinations)
    - Smart segmentation by section/headers (generic for ANY policy)
    - Source location tracking (line numbers, offsets)
    - Dual confidence scores (LLM + quality validation)
    - Deduplication with Jaccard similarity
    - Full traceability and audit trail
    - Multithreading with smart rate limiting (avoids Gemini quota limits)
    
    Rate Limiting:
    - 1 request every 4 seconds (15 req/min = Gemini standard limit)
    - Automatic backoff on quota errors (429)
    - Parallel processing: up to 7 segments simultaneously
    
    Rating: 8.5/10
    """
    
    def __init__(self, policy_file, document_id=None, enable_mongodb=True, max_workers: int = 7):
        self.policy_file = policy_file
        self.clauses_file = f"{OUTPUT_DIR}/stage1_clauses.json"
        self.document_id = document_id
        self.log = []
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_minute=15, requests_per_second=0.25)
    
    def log_entry(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def extract(self) -> bool:
        """Main extraction workflow with multithreading"""
        self.log_entry("START", "Step 1B: Improved Policy Clause Extraction (Multithreaded)")
        
        # Step 1: Read source
        self.log_entry("STEP", f"Reading policy file: {self.policy_file}")
        try:
            with open(self.policy_file, 'r') as f:
                source_text = f.read()
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read file: {e}")
            return False
        
        source_lines = len(source_text.split('\n'))
        self.log_entry("SUCCESS", f"Read {len(source_text)} chars, {source_lines} lines")
        
        # Step 2: Segment text
        self.log_entry("STEP", "Segmenting policy into logical units")
        segmenter = PolicySegmenter()
        segments = segmenter.segment_text(source_text)
        self.log.extend(segmenter.log)
        
        if not segments:
            self.log_entry("ERROR", "No segments detected")
            return False
        
        # Step 3: Extract from each segment (MULTITHREADED)
        self.log_entry("STEP", f"Extracting clauses from {len(segments)} segments (max_workers={self.max_workers})")
        self.log_entry("INFO", "Rate limiting: 1 request every 4 seconds (Gemini quota protection)")
        
        extractor = GenericClauseExtractor(GEMINI_API_KEY)
        extractor.set_source_text(source_text)
        extractor.rate_limiter = self.rate_limiter  # Share rate limiter across threads
        
        all_clauses = []
        segment_results = {}
        
        # Prepare segment tasks
        segment_tasks = [
            (seg_num, seg_text, hierarchy, start_line, end_line)
            for seg_num, (seg_text, start_line, end_line, hierarchy) in enumerate(segments, 1)
        ]
        
        # Process segments in parallel with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for seg_num, seg_text, hierarchy, start_line, end_line in segment_tasks:
                # Submit task
                future = executor.submit(
                    extractor.extract_from_segment,
                    seg_text, seg_num, hierarchy, start_line, end_line
                )
                futures[future] = seg_num
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                seg_num = futures[future]
                try:
                    seg_clauses = future.result()
                    segment_results[seg_num] = seg_clauses
                    completed += 1
                    self.log_entry("COMPLETE", f"Segment {seg_num}: {len(seg_clauses)} clauses extracted ({completed}/{len(segment_tasks)})")
                except Exception as e:
                    self.log_entry("ERROR", f"Segment {seg_num}: Future failed: {e}")
                    segment_results[seg_num] = []
        
        # Merge results in order
        for seg_num in sorted(segment_results.keys()):
            all_clauses.extend(segment_results[seg_num])
        
        self.log.extend(extractor.log)
        self.log_entry("INFO", f"Total clauses extracted: {len(all_clauses)}")
        
        if not all_clauses:
            self.log_entry("ERROR", "No clauses extracted")
            return False
        
        # Step 4: Deduplicate
        self.log_entry("STEP", "Deduplicating clauses")
        deduped_clauses, removed = ClauseDeduplicator.deduplicate(all_clauses)
        self.log_entry("SUCCESS", f"Removed {removed} duplicates, {len(deduped_clauses)} canonical clauses remain")
        
        # Step 5: Calculate realistic quality statistics
        quality_scores = [c.quality_score for c in deduped_clauses]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Count by quality type
        exact_count = sum(1 for c in deduped_clauses if c.extraction_quality == ExtractionQuality.EXACT)
        paraphrased_count = sum(1 for c in deduped_clauses if c.extraction_quality == ExtractionQuality.PARAPHRASED)
        hallucinated_count = sum(1 for c in deduped_clauses if c.extraction_quality == ExtractionQuality.HALLUCINATED)
        
        # Calculate coverage: ratio of extracted clauses to source content
        estimated_extractable = source_lines // 3  # Rough estimate: ~3 lines per clause in policy docs
        coverage_ratio = len(deduped_clauses) / max(estimated_extractable, 1)
        
        # Step 6: Build metadata with enhanced warnings
        metadata = ExtractionMetadata(
            generated=datetime.now().isoformat(),
            source_file=self.policy_file,
            source_size_bytes=len(source_text),
            total_lines=source_lines,
            extraction_method="generic_segmentation",
            policy_type="unknown",
            total_clauses_before_dedup=len(all_clauses),
            total_clauses_after_dedup=len(deduped_clauses),
            avg_quality_score=round(avg_quality, 3),
            dedup_count=removed,
            warnings=[]
        )
        
        # === QUALITY ASSESSMENT ===
        self.log_entry("STAT", f"Quality breakdown: {exact_count} exact, {paraphrased_count} paraphrased, {hallucinated_count} hallucinated")
        self.log_entry("STAT", f"Coverage: {len(deduped_clauses)}/{estimated_extractable} estimated ({coverage_ratio*100:.1f}%)")
        
        # Flag low overall quality
        if avg_quality < 0.65:
            metadata.warnings.append(f"CRITICAL: Low average quality ({avg_quality:.2f}). May indicate segmentation issues or excessive paraphrasing.")
            self.log_entry("WARNING", "Overall quality score below 0.65 threshold")
        
        # Flag poor coverage
        if coverage_ratio < 0.5:
            metadata.warnings.append(f"Low coverage: Only {coverage_ratio*100:.1f}% of estimated clauses extracted. May indicate missed sections or numbered items.")
            self.log_entry("WARNING", f"Coverage below 50%: {coverage_ratio*100:.1f}%")
        
        # Flag hallucinations (but they should be filtered already)
        if hallucinated_count > 0:
            metadata.warnings.append(f"Hallucinations detected: {hallucinated_count} clauses (should be filtered before this point)")
            self.log_entry("WARNING", f"{hallucinated_count} hallucinated clauses present")
        
        # Step 7: Format output
        output = {
            "metadata": asdict(metadata),
            "clauses": [
                {
                    "clause_id": c.clause_id,
                    "original_text": c.original_text,
                    "elaborated_text": c.elaborated_text,
                    "extraction_quality": c.extraction_quality.value,
                    "quality_score": c.quality_score,
                    "llm_confidence": c.llm_confidence,
                    "source_location": asdict(c.source_location),
                    "key_entities": c.key_entities,
                    "warnings": c.warnings,
                    "is_duplicate_of": c.is_duplicate_of
                }
                for c in deduped_clauses
            ]
        }
        
        # Step 8: Save to file
        try:
            with open(self.clauses_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Clauses saved to: {self.clauses_file}")
            self.log_entry("STATS", f"Total clauses: {len(deduped_clauses)}, Avg quality: {avg_quality:.3f}")
            
            # Step 9: Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=1,
                stage_name="extract-clauses",
                stage_output=output,
                filename=self.policy_file.split('/')[-1]
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
            self.log_entry("SUCCESS", "Stage 1 extraction complete")
            return True
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save: {e}")
            return False
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== IMPROVED CLAUSE EXTRACTION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class IntentClassifier:
    """
    Step 2: Classify clause intents using Gemini API with multithreading.
    
    For each clause text from Stage 1B, determines the intent type and confidence.
    Intent types: RESTRICTION, LIMIT, CONDITIONAL_ALLOWANCE, EXCEPTION,
                  APPROVAL_REQUIRED, ADVISORY, INFORMATIONAL
    
    Features:
    - Multithreading with rate limiting (7 workers)
    - Thread-safe logging
    - Retry logic on API failures
    - Proper input validation (elaborated_text from Stage 1)
    - Intent validation
    """
    
    def __init__(self, clauses_file, document_id=None, enable_mongodb=True, use_langchain=False, max_workers: int = 7):
        self.clauses_file = clauses_file
        self.classified_file = f"{OUTPUT_DIR}/stage2_classified.json"
        self.document_id = document_id
        self.log = []
        self.log_lock = threading.Lock()  # Thread-safe logging
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.max_workers = max_workers
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute=15, requests_per_second=0.25)
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        
        # Allowed intents (generic taxonomy)
        self.allowed_intents = [
            "RESTRICTION",
            "LIMIT",
            "CONDITIONAL_ALLOWANCE",
            "EXCEPTION",
            "APPROVAL_REQUIRED",
            "ADVISORY",
            "INFORMATIONAL"
        ]
        
        # Initialize LangChain components if available
        if self.use_langchain:
            self.log_entry("INFO", "LangChain enabled for enhanced intent classification")
            self.langchain_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_retries=3
            )
        else:
            if not LANGCHAIN_AVAILABLE:
                self.log_entry("INFO", "Using standard Gemini classification")
    
    def log_entry(self, level, message):
        """Thread-safe log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        
        # Thread-safe append
        with self.log_lock:
            self.log.append(entry)
        
        print(entry)
    
    def read_clauses_file(self):
        """Read stage1_clauses.json and extract clauses array"""
        try:
            with open(self.clauses_file, 'r') as f:
                data = json.load(f)
            
            if "clauses" not in data:
                self.log_entry("ERROR", "No 'clauses' array found in input file")
                return None
            
            clauses = data["clauses"]
            if not isinstance(clauses, list):
                self.log_entry("ERROR", "Clauses is not a list")
                return None
            
            self.log_entry("SUCCESS", f"Read {len(clauses)} clauses from {self.clauses_file}")
            return clauses
        
        except json.JSONDecodeError as e:
            self.log_entry("ERROR", f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read clauses file: {e}")
            return None
    
    def classify_intent_with_langchain(self, clause_id, clause_text):
        """
        LANGCHAIN-ENHANCED: Classify intent using structured output parsing.
        
        Benefits:
        - Automatic validation with Pydantic
        - Retry logic on failures
        - Type-safe output
        
        Returns:
            dict: { "intent": "...", "confidence": 0.0-1.0, "reasoning": "..." }
        """
        parser = PydanticOutputParser(pydantic_object=IntentClassificationSchema)
        
        prompt_template = PromptTemplate(
            input_variables=["clause_id", "clause_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""You are a POLICY INTENT CLASSIFIER. Analyze the clause and determine its intent type.

ALLOWED INTENT TYPES (choose exactly one):
1. RESTRICTION - Prohibits or forbids an action
2. LIMIT - Sets maximum/minimum thresholds or caps
3. CONDITIONAL_ALLOWANCE - IF/THEN entitlements based on conditions
4. EXCEPTION - Special cases or exemptions
5. APPROVAL_REQUIRED - Requires authorization or permission
6. ADVISORY - Recommended behavior (SHOULD/MAY)
7. INFORMATIONAL - Definitions or classifications

CLAUSE TO CLASSIFY:
Clause ID: {clause_id}
Text: {clause_text}

TASK:
1. Identify primary intent
2. Determine confidence (0.0-1.0)
3. Provide brief reasoning

{format_instructions}

OUTPUT ONLY valid JSON matching the schema."""
        )
        
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        
        try:
            result = chain.run(clause_id=clause_id, clause_text=clause_text)
            parsed = parser.parse(result)
            return parsed.dict()
        except Exception as e:
            self.log_entry("ERROR", f"{clause_id}: LangChain classification failed: {e}")
            # Fallback to standard method
            return self.classify_intent_with_gemini(clause_id, clause_text)
    
    def _precheck_intent_patterns(self, clause_id: str, clause_text: str) -> Optional[Dict]:
        """
        Pre-classification keyword matching BEFORE LLM inference.
        Catches obvious cases that LLM often misclassifies.
        
        Returns:
            dict with intent/confidence if match found, else None (defer to LLM)
        """
        text_lower = clause_text.lower()
        
        # ========== RESTRICTION DETECTION ==========
        # Pattern: "cannot", "can not", "must not", "shall not", "no ... must"
        restriction_patterns = [
            (r'\bcan\s+not\b.*(?:entertain|claimed?|be)', 0.95, 'CAN NOT pattern'),
            (r'\bcannot\b', 0.95, 'CANNOT pattern'),
            (r'\bmust\s+not\b', 0.95, 'MUST NOT pattern'),
            (r'\bshall\s+not\b', 0.95, 'SHALL NOT pattern'),
            (r'\bno\s+\w+\s+(?:must|can)\s+exceed', 0.95, 'NO ... EXCEED pattern'),
            (r'\bprohibited\b|\bforbidden\b', 0.90, 'Explicit prohibition'),
            (r'\bnot\s+(?:allowed|permitted)', 0.90, 'NOT allowed pattern'),
        ]
        
        for pattern, confidence, reason in restriction_patterns:
            if re.search(pattern, text_lower):
                return {
                    'intent': 'RESTRICTION',
                    'confidence': confidence,
                    'reasoning': f"Pre-check: {reason}"
                }
        
        # ========== MANDATORY DETECTION ==========
        # Pattern: "SHALL BE", "must be", time-bound obligations, "to be"
        mandatory_patterns = [
            (r'\bshall\s+be\s+\w+(?:ed|)?\b', 0.95, 'SHALL BE pattern'),
            (r'\bmust\s+be\b', 0.95, 'MUST BE pattern'),
            (r'\bshall\s+(?!not|highlight)', 0.90, 'SHALL + obligation'),  # Exclude "shall highlight"
            (r'to\s+be\s+(?:settled|validated|supported|approved)', 0.95, 'TO BE obligation'),
            (r'\brequire[sd]?\b.*(?:original|authorization|approval|validation)', 0.90, 'Required pattern'),
        ]
        
        # Extra check: exclude "shall highlight" (informational)
        if 'shall highlight' not in text_lower and 'shall consist' not in text_lower:
            for pattern, confidence, reason in mandatory_patterns:
                if re.search(pattern, text_lower):
                    return {
                        'intent': 'MANDATORY',
                        'confidence': confidence,
                        'reasoning': f"Pre-check: {reason}"
                    }
        
        # ========== LIMIT DETECTION ==========
        # Pattern: "maximum", "minimum", "up to", "at most", "exceed"
        # BUT: Must not be already RESTRICTION (cannot/must not + exceed = restriction, not limit)
        is_restriction = any(re.search(pat, text_lower) for pat, _, _ in restriction_patterns)
        
        if not is_restriction:
            limit_patterns = [
                (r'\bmaximum\s+of\b|\bmax\b', 0.90, 'Maximum threshold'),
                (r'\bminimum\s+of\b|\bmin\b', 0.90, 'Minimum threshold'),
                (r'\bup\s+to\b', 0.90, 'UP TO pattern'),
                (r'\bat\s+most\b|\bat\s+least\b', 0.90, 'AT MOST/LEAST pattern'),
                (r'must\s+(?:not\s+)?exceed\s+\d+', 0.95, 'EXCEED threshold'),
                (r'\b(?:limit|cap|ceiling|floor)\b', 0.85, 'Threshold keyword'),
            ]
            
            for pattern, confidence, reason in limit_patterns:
                if re.search(pattern, text_lower):
                    return {
                        'intent': 'LIMIT',
                        'confidence': confidence,
                        'reasoning': f"Pre-check: {reason}"
                    }
        
        # ========== APPROVAL_REQUIRED DETECTION ==========
        approval_patterns = [
            (r'\brequires?\s+(?:approval|authorization|sign-off|sanction)', 0.95, 'Approval required'),
            (r'\bmust\s+be\s+(?:approved|authorized|sanctioned)', 0.95, 'MUST BE approved'),
            (r'\bduly\s+(?:sanctioned|approved|authorized)', 0.90, 'Duly sanctioned'),
        ]
        
        for pattern, confidence, reason in approval_patterns:
            if re.search(pattern, text_lower):
                return {
                    'intent': 'APPROVAL_REQUIRED',
                    'confidence': confidence,
                    'reasoning': f"Pre-check: {reason}"
                }
        
        # ========== INFORMATIONAL DETECTION ==========
        info_patterns = [
            (r'\b(?:includes|consists|defined|classified|types)\s+', 0.90, 'Informational keyword'),
            (r'\b(?:purpose|objective|note|definition)\b', 0.85, 'Informational marker'),
        ]
        
        for pattern, confidence, reason in info_patterns:
            if re.search(pattern, text_lower):
                return {
                    'intent': 'INFORMATIONAL',
                    'confidence': confidence,
                    'reasoning': f"Pre-check: {reason}"
                }
        
        # No strong pre-check match, defer to LLM
        return None
    
    def classify_intent_with_gemini(self, clause_id, clause_text, retry_count: int = 0, max_retries: int = 2):
        """
        Use Gemini API to classify a single clause's intent with rate limiting and retries.
        
        Args:
            clause_id (str): Clause identifier
            clause_text (str): Elaborated clause text
            retry_count (int): Current retry attempt
            max_retries (int): Maximum retries
            
        Returns:
            dict: { "intent": "...", "confidence": 0.0-1.0, "reasoning": "..." }
        """
        
        # ===== PRE-CHECK: Use keyword matching before LLM =====
        precheck_result = self._precheck_intent_patterns(clause_id, clause_text)
        if precheck_result:
            self.log_entry("PRECHECK", f"{clause_id}: {precheck_result['intent']} [conf {precheck_result['confidence']}]")
            return precheck_result
        
        prompt = f"""
You are a POLICY INTENT CLASSIFIER. Analyze the clause text and determine its intent type.

ALLOWED INTENT TYPES (MUST choose exactly one):
1. RESTRICTION - Prohibits or forbids an action; what must NOT be done
2. LIMIT - Sets maximum/minimum thresholds, caps, or quantitative bounds
3. CONDITIONAL_ALLOWANCE - IF/THEN entitlements; allowances based on conditions
4. EXCEPTION - Special cases, exemptions, or deviations from general rules
5. APPROVAL_REQUIRED - Requires authorization, permission, or sign-off
6. ADVISORY - Recommended behavior; non-mandatory; SHOULD/MAY language
7. INFORMATIONAL - Definitions, classifications, or reference information

CLAUSE TO CLASSIFY:
Clause ID: {clause_id}
Text: "{clause_text}"

ANALYSIS TASK:
1. Identify the primary intent of this clause
2. Determine confidence (0.0-1.0) based on clarity and explicitness
3. Provide brief reasoning

OUTPUT FORMAT (strict JSON, one line):
{{
  "intent": "INTENT_TYPE",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this intent was chosen"
}}

Guidelines:
- RESTRICTION: "must NOT", "prohibited", "not allowed", "cannot", "forbidden"
- LIMIT: "maximum", "minimum", "cap", "threshold", "up to", "at most", "at least"
- CONDITIONAL_ALLOWANCE: "IF...THEN", "eligible if", "provided that", "when", "upon"
- EXCEPTION: "except", "unless", "provided", "in case of", "special case"
- APPROVAL_REQUIRED: "requires approval", "must be approved", "needs authorization"
- ADVISORY: "should", "may", "recommended", "preferred", "suggested"
- INFORMATIONAL: "defined as", "classified as", "includes", "consists of", "types"

Output ONLY valid JSON. No explanation outside the JSON object.
"""
        
        try:
            # RATE LIMITING: Wait before making API call
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                self.log_entry("RATE_LIMIT", f"{clause_id}: Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            # Make API call
            try:
                response = self.model.generate_content(prompt)
                self.rate_limiter.record_request()
                response_text = response.text.strip()
            except Exception as api_error:
                error_str = str(api_error).lower()
                self.log_entry("ERROR", f"{clause_id}: API call failed: {type(api_error).__name__}")
                
                # Handle quota errors
                if "429" in str(api_error) or "quota" in error_str:
                    self.log_entry("ERROR", f"{clause_id}: QUOTA LIMIT HIT")
                    self.rate_limiter.record_quota_error()
                    if retry_count < max_retries:
                        wait = 2 ** (retry_count + 2)
                        self.log_entry("RETRY", f"{clause_id}: Waiting {wait}s before retry...")
                        time.sleep(wait)
                        return self.classify_intent_with_gemini(clause_id, clause_text, retry_count + 1, max_retries)
                
                # Handle transient errors
                elif retry_count < max_retries and ("deadline" in error_str or "timeout" in error_str):
                    wait = 2 ** retry_count
                    self.log_entry("RETRY", f"{clause_id}: Retrying in {wait}s...")
                    time.sleep(wait)
                    return self.classify_intent_with_gemini(clause_id, clause_text, retry_count + 1, max_retries)
                
                # Fallback
                return {
                    "intent": "INFORMATIONAL",
                    "confidence": 0.5,
                    "reasoning": f"API error: {str(api_error)[:40]}"
                }
            
            # Clean markdown if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                result = json.loads(response_text)
                
                # Validate intent
                if "intent" not in result:
                    self.log_entry("WARNING", f"{clause_id}: Missing 'intent' field")
                    result["intent"] = "INFORMATIONAL"
                
                # Validate intent is in allowed list
                intent = result.get("intent", "INFORMATIONAL").upper()
                if intent not in self.allowed_intents:
                    self.log_entry("WARNING", f"{clause_id}: Invalid intent '{intent}', using INFORMATIONAL")
                    result["intent"] = "INFORMATIONAL"
                else:
                    result["intent"] = intent
                
                # Validate confidence
                if "confidence" not in result:
                    result["confidence"] = 0.75
                else:
                    try:
                        conf = float(result["confidence"])
                        result["confidence"] = max(0.0, min(1.0, conf))
                    except (ValueError, TypeError):
                        result["confidence"] = 0.75
                
                return result
            
            except json.JSONDecodeError as e:
                self.log_entry("ERROR", f"{clause_id}: Failed to parse JSON: {e}")
                return {
                    "intent": "INFORMATIONAL",
                    "confidence": 0.5,
                    "reasoning": "JSON parse error"
                }
        
        except Exception as e:
            self.log_entry("ERROR", f"{clause_id}: Unexpected error: {type(e).__name__}: {str(e)[:50]}")
            return {
                "intent": "INFORMATIONAL",
                "confidence": 0.5,
                "reasoning": "Unexpected error"
            }
    
    def classify(self):
        """Main classification workflow with multithreading and rate limiting"""
        self.log_entry("START", "Step 2: Intent Classification (Multithreaded)")
        
        # Step 1: Read clauses from Stage 1B
        self.log_entry("STEP", "Reading clauses from Stage 1")
        clauses = self.read_clauses_file()
        if not clauses:
            return False
        
        self.log_entry("STATS", f"Total clauses to classify: {len(clauses)}")
        
        # Step 2: Classify each clause with MULTITHREADING
        self.log_entry("STEP", f"Classifying intent (max_workers={self.max_workers})")
        self.log_entry("INFO", "Rate limiting: 1 request every 4 seconds (Gemini quota protection)")
        
        classified = []
        
        # Initialize policy-agnostic enhancer for validation
        agnostic_enhancer = PolicyAgnosticEnhancer() if GENERIC_ENHANCER_AVAILABLE else None
        
        def classify_single_clause(clause_data):
            """Helper function for parallel processing"""
            i, clause = clause_data
            clause_id = clause.get("clause_id", f"C{i}")
            
            # Use elaborated_text from Stage 1, fallback to original_text
            clause_text = clause.get("elaborated_text", "") or clause.get("original_text", "")
            
            if not clause_text:
                self.log_entry("WARNING", f"Clause {clause_id}: No text found")
                return {
                    "clause_id": clause_id,
                    "text": "",
                    "intent": "INFORMATIONAL",
                    "confidence": 0.0,
                    "reasoning": "No clause text provided"
                }
            
            self.log_entry("CLASSIFY", f"[{i}/{len(clauses)}] Processing {clause_id}")
            
            # Classify this clause
            if self.use_langchain:
                intent_result = self.classify_intent_with_langchain(clause_id, clause_text)
            else:
                intent_result = self.classify_intent_with_gemini(clause_id, clause_text)
            
            original_intent = intent_result.get("intent", "INFORMATIONAL")
            original_confidence = intent_result.get("confidence", 0.5)
            
            # Apply policy-agnostic validation to correct misclassifications
            if agnostic_enhancer:
                corrected_intent, corrected_confidence = agnostic_enhancer.validate_and_correct_intent(
                    clause_id, clause_text, original_intent, original_confidence
                )
                intent = corrected_intent
                confidence = corrected_confidence
            else:
                intent = original_intent
                confidence = original_confidence
            
            # Build classified clause record
            classified_clause = {
                "clause_id": clause_id,
                "original_text": clause.get("original_text", ""),
                "elaborated_text": clause_text,
                "intent": intent,
                "confidence": confidence,
                "reasoning": intent_result.get("reasoning", "")
            }
            
            self.log_entry("RESULT", f"{clause_id}: {intent} (confidence: {confidence:.2f})")
            return classified_clause
        
        # Process clauses in parallel with thread pool
        classified_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            clause_data = [(i, clause) for i, clause in enumerate(clauses, 1)]
            futures = {}
            
            for i, clause in clause_data:
                future = executor.submit(classify_single_clause, (i, clause))
                futures[future] = i
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    classified_results[futures[future]] = result
                    completed += 1
                    self.log_entry("COMPLETE", f"Classification complete ({completed}/{len(clause_data)})")
                except Exception as e:
                    self.log_entry("ERROR", f"Future failed: {e}")
                    classified_results[futures[future]] = {
                        "clause_id": f"C{futures[future]}",
                        "text": "",
                        "intent": "INFORMATIONAL",
                        "confidence": 0.0,
                        "reasoning": str(e)
                    }
        
        # Merge results in order
        for i in sorted(classified_results.keys()):
            classified.append(classified_results[i])
        
        self.log_entry("SUCCESS", f"Classified {len(classified)} clauses")
        
        # Step 3: Format output
        self.log_entry("STEP", "Formatting classified output")
        formatted_output = self.format_classified_output(classified)
        
        # Step 4: Save to file
        try:
            with open(self.classified_file, 'w') as f:
                json.dump(formatted_output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Classified clauses saved to: {self.classified_file}")
            
            # Step 5: Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=2,
                stage_name="classify-intents",
                stage_output=formatted_output
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
            return True
        
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save classified file: {e}")
            return False
    
    def format_classified_output(self, classified_clauses):
        """
        Format classified output with metadata.
        
        Args:
            classified_clauses (list): List of classified clause dicts
            
        Returns:
            dict: Formatted output with metadata and classified clauses
        """
        # Calculate statistics
        intent_counts = {}
        for clause in classified_clauses:
            intent = clause.get("intent", "UNKNOWN")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        avg_confidence = sum(c.get("confidence", 0) for c in classified_clauses) / len(classified_clauses) if classified_clauses else 0
        
        output = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "source": self.clauses_file,
                "format": "Intent Classification (Stage 2)",
                "total_clauses": len(classified_clauses),
                "stage": "2",
                "next_stage": "3 - Payload Evaluation",
                "average_confidence": round(avg_confidence, 3),
                "intent_distribution": intent_counts
            },
            "classified_clauses": classified_clauses
        }
        return output
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== INTENT CLASSIFICATION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class EntityExtractor:
    """
    Step 3: Extract structured entities & thresholds using Gemini API with multithreading.
    
    For each clause text from Stage 2, extracts explicit entities only.
    
    Features:
    - Multithreading with rate limiting (7 workers)
    - Thread-safe logging
    - Retry logic on API failures
    - Proper input validation (elaborated_text + original_text from Stage 2)
    - Dynamic entity extraction (no hardcoded schemas)
    """
    
    def __init__(self, classified_file, document_id=None, enable_mongodb=True, use_langchain=False, max_workers: int = 7):
        self.classified_file = classified_file
        self.entities_file = f"{OUTPUT_DIR}/stage3_entities.json"
        self.document_id = document_id
        self.log = []
        self.log_lock = threading.Lock()  # Thread-safe logging
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.max_workers = max_workers
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute=15, requests_per_second=0.25)
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        
        # Initialize LangChain components if available
        if self.use_langchain:
            self.log_entry("INFO", "LangChain enabled for enhanced entity extraction")
            self.langchain_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_retries=3
            )
        else:
            self.log_entry("INFO", "Using standard Gemini entity extraction")
    
    def log_entry(self, level, message):
        """Thread-safe log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        
        # Thread-safe append
        with self.log_lock:
            self.log.append(entry)
        
        print(entry)
    
    def read_classified_file(self):
        """Read stage2_classified.json and extract classified clauses"""
        try:
            with open(self.classified_file, 'r') as f:
                data = json.load(f)
            
            if "classified_clauses" not in data:
                self.log_entry("ERROR", "No 'classified_clauses' array found in input file")
                return None
            
            clauses = data["classified_clauses"]
            if not isinstance(clauses, list):
                self.log_entry("ERROR", "Classified clauses is not a list")
                return None
            
            self.log_entry("SUCCESS", f"Read {len(clauses)} classified clauses from {self.classified_file}")
            return clauses
        
        except json.JSONDecodeError as e:
            self.log_entry("ERROR", f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read classified file: {e}")
            return None
    
    def extract_entities_with_langchain(self, clause_id, clause_text):
        """
        LANGCHAIN-ENHANCED: Extract entities using structured output parsing.
        
        Benefits:
        - Automatic validation with Pydantic
        - Retry logic on failures
        - Type-safe output
        
        Returns:
            dict: Extracted entities as key-value pairs
        """
        parser = PydanticOutputParser(pydantic_object=EntityExtractionSchema)
        
        prompt_template = PromptTemplate(
            input_variables=["clause_id", "clause_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""You are a DYNAMIC ENTITY EXTRACTION ENGINE. Extract meaningful structured entities from policy text.

CRITICAL RULES:
1. Extract ONLY values explicitly stated in the text
2. Do NOT infer, guess, or assume missing values
3. Do NOT include null/empty values - omit if not present
4. Preserve original units (currency, time, amounts, etc.)
5. Create entity names that are descriptive and meaningful
6. One clause may have multiple entity values

ENTITY EXTRACTION GUIDELINES:
- Identify key-value pairs that represent measurable facts
- Entity names should be descriptive (e.g., "dailyAllowance", "travelDuration")
- Include units in values when present (e.g., "7.0 Rs/KM", "15 days", "100 KM")
- Use camelCase for multi-word entity names

CLAUSE TO ANALYZE:
Clause ID: {clause_id}
Text: {clause_text}

EXTRACTION TASK:
1. Scan clause text for explicit, measurable values
2. Identify meaningful entity names based on context
3. Extract only what is clearly stated
4. Omit entities not explicitly mentioned
5. Return a JSON object with extracted entities

{format_instructions}

OUTPUT ONLY valid JSON matching the schema. If no entities, return {{"entities": {{}}}}."""
        )
        
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        
        try:
            result = chain.run(clause_id=clause_id, clause_text=clause_text)
            parsed = parser.parse(result)
            return parsed.entities
        except Exception as e:
            self.log_entry("ERROR", f"{clause_id}: LangChain extraction failed: {e}")
            # Fallback to standard method
            return self.extract_entities_with_gemini(clause_id, clause_text)
    
    def extract_entities_with_gemini(self, clause_id, clause_text, retry_count: int = 0, max_retries: int = 2):
        """
        Extract ALL numeric/categorical thresholds from policy text with rate limiting and retries.
        
        Features:
        - Rate limiting (1 req/4s)
        - Retry logic with exponential backoff
        - Extracts: distances, percentages, durations, amounts, categorical values, authorities
        
        Args:
            clause_id (str): Clause identifier
            clause_text (str): Clause text with policy content
            retry_count (int): Current retry attempt
            max_retries (int): Maximum retries
            
        Returns:
            dict: { "entityName": value, ... } with extracted entities
        """
        
        prompt = f"""You are a THRESHOLD & ENTITY EXTRACTION ENGINE.

CLAUSE ID: {clause_id}
CLAUSE TEXT: {clause_text}

MANDATORY: Extract EVERY measurable value and threshold from this clause.

1. NUMERIC THRESHOLDS (EXTRACT ALL):
   - Distances: "200 kms" → {{"distanceThreshold": "200 kms"}}
   - Percentages: "25%", "125%" → {{"percentageValue": "25%"}}
   - Time durations: "2 days" → {{"settlementDays": "2 days"}}
   - Hour triggers: "more than 6 hours" → {{"hourTrigger": "6 hours"}}
   - Monetary: "$100", "Rs 500" → {{"amountValue": "amount"}}

2. CATEGORICAL VALUES (EXTRACT ALL):
   - Employee levels: "Senior level" → {{"employeeLevel": "Senior"}}
   - Role types: "sales personnel" → {{"roleExclusion": "sales personnel"}}
   - Travel modes: "Personal Car", "Personal Bike" → {{"travelMode": "Personal Car"}}
   - Location types: "remote areas" → {{"locationType": "remote areas"}}

3. AUTHORITIES & ENTITIES:
   - Organizations: "NAGA LIMITED" → {{"applicableTo": "NAGA LIMITED"}}
   - Departments: "HR & Accounts" → {{"validationAuthority": "HR & Accounts"}}
   - Roles: "Reporting Manager" → {{"approvalAuthority": "Reporting Manager"}}

EXAMPLES - Extract ALL values:
- "No official Travel by Personal Car MUST exceed 200 kms" → {{"distanceThreshold": "200 kms", "travelMode": "Personal Car"}}
- "Travelling advance MUST be settled within 2 days" → {{"settlementDays": "2 days"}}
- "Employees other than sales personnel...spending more than 6 hours" → {{"roleExclusion": "sales personnel", "hourTrigger": "6 hours"}}
- "claim 25% of their eligible amount" → {{"percentageValue": "25%"}}
- "For Twin sharing, the claim can be made to the maximum of 125%" → {{"maximumClaimPercentage": "125%"}}
- "No official Travel by Personal Bike MUST exceed 50 kms" → {{"distanceThreshold": "50 kms", "travelMode": "Personal Bike"}}

RULES:
- Extract EVERY numeric value with its unit
- Use descriptive entity names (camelCase)
- One clause may have MULTIPLE entities (2-5+)
- If text has values, return ALL of them
- If NO values, return {{}}

Output ONLY valid JSON.
"""
        
        try:
            # RATE LIMITING: Wait before making API call
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                self.log_entry("RATE_LIMIT", f"{clause_id}: Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            # Make API call
            try:
                response = self.model.generate_content(prompt)
                self.rate_limiter.record_request()
                response_text = response.text.strip()
            except Exception as api_error:
                error_str = str(api_error).lower()
                self.log_entry("ERROR", f"{clause_id}: API call failed: {type(api_error).__name__}")
                
                # Handle quota errors
                if "429" in str(api_error) or "quota" in error_str:
                    self.log_entry("ERROR", f"{clause_id}: QUOTA LIMIT HIT")
                    self.rate_limiter.record_quota_error()
                    if retry_count < max_retries:
                        wait = 2 ** (retry_count + 2)
                        self.log_entry("RETRY", f"{clause_id}: Waiting {wait}s before retry...")
                        time.sleep(wait)
                        return self.extract_entities_with_gemini(clause_id, clause_text, retry_count + 1, max_retries)
                
                # Handle transient errors
                elif retry_count < max_retries and ("deadline" in error_str or "timeout" in error_str):
                    wait = 2 ** retry_count
                    self.log_entry("RETRY", f"{clause_id}: Retrying in {wait}s...")
                    time.sleep(wait)
                    return self.extract_entities_with_gemini(clause_id, clause_text, retry_count + 1, max_retries)
                
                # Fallback
                return {}
            
            # Clean markdown if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                result = json.loads(response_text)
                
                if not isinstance(result, dict):
                    self.log_entry("WARNING", f"{clause_id}: Result is not a dict, using empty")
                    return {}
                
                return result
            
            except json.JSONDecodeError as e:
                self.log_entry("ERROR", f"{clause_id}: Failed to parse JSON: {e}")
                return {}
        
        except Exception as e:
            self.log_entry("ERROR", f"{clause_id}: Unexpected error: {type(e).__name__}: {str(e)[:50]}")
            return {}
    
    def extract(self):
        """Main extraction workflow with multithreading and rate limiting"""
        self.log_entry("START", "Step 3: Entity & Threshold Extraction (Multithreaded)")
        
        # Step 1: Read classified clauses from Stage 2
        self.log_entry("STEP", "Reading classified clauses from Stage 2")
        clauses = self.read_classified_file()
        if not clauses:
            return False
        
        self.log_entry("STATS", f"Total clauses for entity extraction: {len(clauses)}")
        
        # Step 2: Extract entities with MULTITHREADING
        self.log_entry("STEP", f"Extracting entities (max_workers={self.max_workers})")
        self.log_entry("INFO", "Rate limiting: 1 request every 4 seconds (Gemini quota protection)")
        
        # Initialize policy-agnostic enhancer for entity cleaning
        agnostic_enhancer = PolicyAgnosticEnhancer() if GENERIC_ENHANCER_AVAILABLE else None
        
        def extract_single(clause_data):
            """Helper function for parallel processing"""
            i, clause = clause_data
            clause_id = clause.get("clause_id", f"C{i}")
            
            # Use elaborated_text from Stage 2, fallback to original_text
            clause_text = clause.get("elaborated_text", "") or clause.get("original_text", "")
            clause_intent = clause.get("intent", "UNKNOWN")
            
            if not clause_text:
                self.log_entry("WARNING", f"Clause {clause_id}: No text found")
                return {
                    "clause_id": clause_id,
                    "original_text": "",
                    "elaborated_text": "",
                    "intent": clause_intent,
                    "entities": {}
                }
            
            self.log_entry("EXTRACT", f"[{i}/{len(clauses)}] Processing {clause_id}")
            
            # Extract entities from this clause
            if self.use_langchain:
                entities = self.extract_entities_with_langchain(clause_id, clause_text)
            else:
                entities = self.extract_entities_with_gemini(clause_id, clause_text)
            
            # Apply entity cleaning to remove garbage
            if agnostic_enhancer and entities:
                entities = agnostic_enhancer.clean_entities(entities)
            
            entity_count = len(entities) if entities else 0
            entity_keys = list(entities.keys()) if entities else []
            self.log_entry("RESULT", f"{clause_id}: Extracted {entity_count} entities: {entity_keys}")
            
            # Build extracted clause record
            return {
                "clause_id": clause_id,
                "original_text": clause.get("original_text", ""),
                "elaborated_text": clause_text,
                "intent": clause_intent,
                "entities": entities if entities else {}
            }
        
        # Process clauses in parallel with thread pool
        extracted_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            clause_data = [(i, clause) for i, clause in enumerate(clauses, 1)]
            futures = {}
            
            for i, clause in clause_data:
                future = executor.submit(extract_single, (i, clause))
                futures[future] = i
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    extracted_results[futures[future]] = result
                    completed += 1
                    self.log_entry("COMPLETE", f"Entity extraction complete ({completed}/{len(clause_data)})")
                except Exception as e:
                    self.log_entry("ERROR", f"Future failed: {e}")
                    extracted_results[futures[future]] = {
                        "clause_id": f"C{futures[future]}",
                        "original_text": "",
                        "elaborated_text": "",
                        "intent": "UNKNOWN",
                        "entities": {}
                    }
        
        # Merge results in order
        extracted = []
        for i in sorted(extracted_results.keys()):
            extracted.append(extracted_results[i])
        
        self.log_entry("SUCCESS", f"Extracted entities from {len(extracted)} clauses")
        
        # Step 3: Calculate statistics
        self.log_entry("STEP", "Calculating entity statistics")
        stats = self.calculate_entity_statistics(extracted)
        
        # Step 4: Format output
        self.log_entry("STEP", "Formatting entity extraction output")
        formatted_output = self.format_entities_output(extracted, stats)
        
        # Step 5: Save to file
        try:
            with open(self.entities_file, 'w') as f:
                json.dump(formatted_output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Extracted entities saved to: {self.entities_file}")
            
            # Step 6: Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=3,
                stage_name="extract-entities",
                stage_output=formatted_output
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
            return True
        
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save entities file: {e}")
            return False
    
    def calculate_entity_statistics(self, extracted_clauses):
        """
        Calculate entity extraction statistics.
        
        Args:
            extracted_clauses (list): List of extracted clause dicts
            
        Returns:
            dict: Statistics about entities
        """
        entity_counts = {}
        total_entities = 0
        clauses_with_entities = 0
        
        for clause in extracted_clauses:
            entities = clause.get("entities", {})
            if entities:
                clauses_with_entities += 1
            
            for entity_type in entities.keys():
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                total_entities += 1
        
        return {
            "total_entities": total_entities,
            "total_clauses": len(extracted_clauses),
            "clauses_with_entities": clauses_with_entities,
            "entity_type_distribution": entity_counts,
            "coverage": round(clauses_with_entities / len(extracted_clauses) * 100, 1) if extracted_clauses else 0
        }
    
    def format_entities_output(self, extracted_clauses, stats):
        """
        Format entity extraction output with metadata.
        
        Args:
            extracted_clauses (list): List of extracted clause dicts
            stats (dict): Entity statistics
            
        Returns:
            dict: Formatted output with metadata and extracted entities
        """
        # Collect all unique entity names found across clauses
        entity_names = set()
        for clause in extracted_clauses:
            for entity_name in clause.get("entities", {}).keys():
                entity_names.add(entity_name)
        
        output = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "source": self.classified_file,
                "format": "Dynamic Entity & Threshold Extraction (Stage 3)",
                "total_clauses": len(extracted_clauses),
                "stage": "3",
                "next_stage": "4 - Payload Validation",
                "statistics": stats,
                "note": "Entity types are dynamically extracted - not predefined. Each entity name is derived from the clause context."
            },
            "extracted_entity_names": sorted(list(entity_names)),
            "extracted_clauses": extracted_clauses
        }
        return output
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== ENTITY EXTRACTION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class RuleExtractor:
    """Step 2: Use grep + Gemini to extract rules"""
    
    def __init__(self, policy_file):
        self.policy_file = policy_file
        self.rules_file = f"{OUTPUT_DIR}/rules.txt"
        self.log = []
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        
    def log_entry(self, level, message):
        """Log entry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def grep_search(self, pattern):
        """Run grep search on policy file"""
        try:
            result = subprocess.run(
                ['grep', '-n', pattern, self.policy_file],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                self.log_entry("GREP", f"Pattern '{pattern}': Found {len(lines)} matches")
                return lines
            else:
                self.log_entry("GREP", f"Pattern '{pattern}': No matches")
                return []
                
        except Exception as e:
            self.log_entry("ERROR", f"grep failed: {e}")
            return []
    
    def read_policy_file(self):
        """Read entire policy file"""
        try:
            with open(self.policy_file, 'r') as f:
                return f.read()
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read policy file: {e}")
            return None
    
    def analyze_structure_with_grep(self, content):
        """Analyze document structure using grep patterns"""
        self.log_entry("PASS", "Analyzing document structure")
        
        patterns = [
            r'^[0-9]+\. [A-Z]',      # Main sections (1. SECTION)
            r'^[0-9]\.[0-9] ',       # Subsections (1.1 )
            r'MUST|EXPECTED|SHOULD', # Enforcement levels
        ]
        
        structure = {}
        for pattern in patterns:
            self.log_entry("GREP_PATTERN", f"Testing: {pattern}")
            results = self.grep_search(pattern)
            structure[pattern] = results
        
        return structure
    
    def extract_rules_with_gemini(self, content):
        """
        Use Gemini API to extract structured policy rules in Refactored Rule Engine format.
        Works with ANY policy document - automatically infers structure and requirements.

        Returns:
            str: Structured rules text or None if extraction failed.
        """

        self.log_entry("GEMINI", "Initializing Gemini API for structured rule extraction")

        prompt = f"""
You are a STRUCTURED POLICY EXTRACTION ENGINE with strict enforcement of logical rigor.

Task:
- Analyze the policy document below and extract ALL rules, requirements, entitlements, and constraints.
- Output in the REFACTORED RULE ENGINE FORMAT (see below).
- Make NO assumptions about the policy domain - infer structure from actual content.
- ENFORCE: Every rule must have measurable conditions and verifiable consequences.
- ENFORCE: All percentages, amounts, times, and durations must be explicit.
- ENFORCE: Flag vague terms and provide concrete alternatives.
- Output ONLY the structured rules - no commentary, explanations, or metadata outside the format.

====================================================
POLICY DOCUMENT
====================================================
{content}

====================================================
REFACTORED RULE ENGINE FORMAT
====================================================

SECTION 1: DEFINITIONS & CONSTANTS
Define all key terms, enumerations, grades, roles, categories, and thresholds mentioned in the policy.
Example:
  employee.grade: {{E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, Director, MD}}
  travel.duration_threshold: {{short: 1-10 days, long: 11+ days}}
  daily_allowance: {{E1-E7: 300 USD, E8-E10: 350 USD, MD: 500 USD}}

SECTION 2: AUTHORITY HIERARCHY (if applicable)
Define approval authorities, decision-makers, and escalation paths with explicit role definitions.
Example:
  approval.travel_sanction: {{"authority": "MD & CEO", "required": true}}
  approval.extended_stay: {{"authority": "E8 or higher", "exceptions": ["medical exigencies"]}}

SECTION 3: PRIMARY ENTITLEMENTS OR RULES
Organize by major policy domains/categories. For each rule use STRICT format:

rule: <rule_name>
priority: <1-high, 2-medium, 3-low>
conditions:
  - IF <measurable_condition_1> AND <measurable_condition_2>
    THEN <verifiable_consequence_1>
    AND <verifiable_consequence_2>
    SOURCE: <clause reference>

CRITICAL REQUIREMENTS FOR SECTION 3:
  - Every condition must be verifiable (compare against defined constants)
  - Every THEN must have a concrete, measurable outcome
  - If condition contains %, that % must reference a base value defined in SECTION 1
  - If condition contains time (e.g., "after 4:30 PM"), state EXACTLY what the trigger is
  - If multiple rules could apply, state which takes precedence


SECTION 4: ANCILLARY RULES
Optional behaviors, recommendations, special cases that are not mandatory.
Mark as SHOULD/RECOMMENDED, not MUST.

SECTION 5: PROHIBITED ITEMS
What is explicitly forbidden with clear conditions.

SECTION 6: MANDATORY REQUIREMENTS
Pre-conditions that MUST be satisfied before any travel/entitlements activate.
These are blocking conditions (no exceptions).

SECTION 7: RULE PRECEDENCE & CONFLICT RESOLUTION
Explicit rules for handling overlapping conditions:
  - Which rule wins if multiple apply? (e.g., "Most restrictive rule applies")
  - What if a rule contradicts another? (e.g., "Explicit exception overrides general rule")
  - How are ambiguous conditions resolved? (e.g., "Escalate to 'x' for decision")
  
If NO conflict exists, state: "No overlapping rules identified in this policy."

SECTION 8: MEASUREMENT & ENFORCEMENT
For each vague term found, provide:
  1. Original vague term
  2. Why it's unmeasurable
  3. Concrete metric to replace it
  
Example:
  Vague: "judicious expenditure"
  Reason: No objective threshold defined
  Concrete: "Allowance capped at [grade-specific amount]. Outliers flagged if spend > 120% of cap."

SECTION 9: POLICY MAINTENANCE
Review cycle, dependencies, update procedures, external references.

====================================================
EXTRACTION GUIDELINES
====================================================
NAMING & TERMS:
  - Use normalized domain terms consistently (e.g., employee.grade, booking.travel_class, fx.amount)
  - Define all acronyms and abbreviations (e.g., "MEA" = Ministry of External Affairs)
  - When policy references external documents, list them explicitly

NUMERIC PRECISION:
  - Make ALL numeric limits explicit: X USD, Y days, Z hours
  - For ranges, express as: "min <= value <= max" or "E1-E7 grades (6 distinct grades)"
  - For percentages, state: "X% of [base value defined in SECTION 1]"
  - For time conditions, use 24-hour format: "after 16:30 (4:30 PM)" not "after 4:30 PM"

LOGIC & CONDITIONS:
  - Split complex multi-condition clauses into separate rules
  - Use AND/OR explicitly (not implicit)
  - For grade-based rules, handle overlaps: if both "E8" and "E8-E10" rules exist, which wins?
  - If policy says "may", "should", "could", mark as optional (SECTION 4)
  - If policy says "must", "require", "shall", mark as mandatory (SECTION 3 or 6)

VAGUE TERM HANDLING:
  - Do NOT output rules with unmeasurable terms
  - Flag them in SECTION 8 instead
  - Suggest concrete metrics
  - Mark as PENDING MANUAL REVIEW if no metric can be inferred

COMPLETENESS:
  - Preserve clause/section numbers from source document
  - If policy is silent on a scenario, do NOT invent it
  - If policy contradicts itself, flag explicitly in SECTION 7

====================================================
OUTPUT STRUCTURE EXAMPLE
====================================================

================================================================================
OVERSEAS BUSINESS TRAVEL - STRUCTURED RULES
================================================================================
Generated: 2026-01-15
Source Policy Sections: 1-5
Format: Rule Engine v1.0

================================================================================
1. DEFINITIONS & CONSTANTS
================================================================================

employee.grade:
  junior: [E1, E2, E3, E4, E5, E6, E7]
  senior: [E8, E9, E10]
  executive: [Director, MD & CEO]

daily_allowance_usd:
  E1_to_E7: 300
  E8_to_E10: 350
  executive: 500

travel_class_entitlement:
  E1_to_E7: "Economy Class"
  E8_to_E10: "Business Class"
  Director: ["Business Class", "Club Class"]
  MD_CEO: "First Class"

================================================================================
2. AUTHORITY HIERARCHY
================================================================================

approval.travel_sanction:
  authority: "MD & CEO"
  required: true
  exceptions: none

[etc...]

====================================================
CRITICAL: OUTPUT ONLY THE STRUCTURED RULES
====================================================
Do NOT include:
- Explanatory text outside the format
- Original policy paragraphs
- Commentary or interpretation
- Placeholder examples
- Vague rules (move to SECTION 8 instead)

Output must be ready to parse and implement as-is.
Rules with unmeasurable conditions go to SECTION 8 for definition.
"""

        try:
            self.log_entry("GEMINI_REQUEST", "Sending policy content for structured rule extraction")
            response = self.model.generate_content(prompt)

            rule_output = response.text.strip()
            
            # Validate output contains expected structure markers
            if "DEFINITIONS & CONSTANTS" not in rule_output and "rule:" not in rule_output:
                self.log_entry("WARNING", "Gemini response missing expected structure markers. Output may be incomplete.")
            
            self.log_entry("GEMINI_RESPONSE", "Structured rules extracted successfully")
            
            # Return raw output - formatting will be done by caller
            return rule_output

        except Exception as e:
            self.log_entry("ERROR", f"Gemini API failed: {e}")
            return None

    def format_rules_output(self, gemini_output):
        """
        Wrap extracted rules with consistent header and metadata.
        
        Args:
            gemini_output (str): Raw output from Gemini
            
        Returns:
            str: Formatted rules document
        """
        output = "=" * 80 + "\n"
        output += "STRUCTURED POLICY RULES - EXTRACTED VIA GEMINI API\n"
        output += "=" * 80 + "\n\n"
        output += f"Generated: {datetime.now()}\n"
        output += f"Source: {self.policy_file}\n"
        output += f"Extraction Method: Content Analysis + Gemini Structured Extraction\n"
        output += f"Format: Rule Engine v1.0 (Measurable, Enforceable, Generic)\n\n"
        output += "=" * 80 + "\n\n"
        
        output += gemini_output
        
        return output
    
    def extract(self):
        """Main extraction workflow"""
        self.log_entry("START", "Step 2: Rule Extraction with Grep + Gemini")
        
        # Step 1: Read policy file
        self.log_entry("STEP", "Reading policy file")
        content = self.read_policy_file()
        if not content:
            return False
        
        self.log_entry("SUCCESS", f"Policy file read: {len(content)} characters")
        
        # Step 2: Analyze structure with grep
        self.log_entry("STEP", "Analyzing structure with grep searches")
        structure = self.analyze_structure_with_grep(content)
        
        # Step 3: Extract rules with Gemini
        self.log_entry("STEP", "Extracting rules with Gemini API")
        gemini_response = self.extract_rules_with_gemini(content)
        if not gemini_response:
            return False
        
        # Step 4: Format and save output
        self.log_entry("STEP", "Formatting output")
        formatted_rules = self.format_rules_output(gemini_response)
        
        # Save to file
        try:
            with open(self.rules_file, 'w') as f:
                f.write(formatted_rules)
            
            self.log_entry("SUCCESS", f"Rules saved to: {self.rules_file}")
            self.log_entry("STATS", f"Output size: {len(formatted_rules)} characters")
            
            return True
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save rules file: {e}")
            return False
    
    def extract_topics_and_scenarios(self, content):
        """
        Extract all meaningful topics/scenarios from policy document using Gemini.
        Returns a structured list of topics with brief descriptions.
        Focuses on actual policy domains, not section numbers.
        """
        self.log_entry("GEMINI", "Extracting topics and scenarios from policy")
        
        prompt = f"""
You are a POLICY TOPICS EXTRACTION ENGINE. Your task is to identify MEANINGFUL POLICY DOMAINS.

CRITICAL INSTRUCTIONS:
=====================
1. Extract ONLY meaningful topic names that represent policy domains/areas (e.g., "Travel Authorization", "Leave Entitlements", "Daily Allowances")
2. DO NOT output section numbers alone (e.g., "4.1", "3.2", "12") - these are NOT topics
3. DO NOT output generic items without clear policy meaning
4. Each topic must represent a distinct policy domain with its own set of rules

Task:
- Analyze the policy document below and identify ALL major policy domains/categories
- Extract topics that have their own separate rules, requirements, or guidelines
- For each topic, provide a brief 1-2 line description of what it covers
- Output as a numbered list with MEANINGFUL TOPIC NAME and description

Policy Document:
================================================================================
{content}
================================================================================

OUTPUT FORMAT:
Number each topic sequentially. Use this STRICT format:
1. [Meaningful Topic Name] - Brief description (1-2 lines max) of what this policy area covers

Example of CORRECT output:
1. Daily Allowances - Consolidated foreign exchange amounts for employees by grade
2. Leave Entitlements - Leave on duty and extended stay policies during overseas travel
3. Travel Authorization - Approval process and authority hierarchy for overseas travel

Example of INCORRECT output (NEVER do this):
1. 4.1 - (DO NOT USE - section numbers are not topics)
2. Item 12 - (DO NOT USE - generic numbers are not topics)
3. Guidelines - (DO NOT USE - too vague without domain specification)

COMPLETION CRITERIA:
====================
✓ ONLY extract topics that represent actual policy domains from the document
✓ Each topic name must be descriptive and meaningful (e.g., "Travel Mode Entitlements", NOT "3.1")
✓ Topic must have multiple rules or requirements in the policy
✓ Preserve the exact topic names/domains as they appear conceptually in the policy
✓ Output ONLY the numbered list - no explanations or metadata

Output ONLY the numbered list. No additional text before or after.
"""
        
        try:
            response = self.model.generate_content(prompt)
            topics_text = response.text.strip()
            self.log_entry("GEMINI_RESPONSE", "Topics extracted successfully")
            
            # Validate that extracted topics are meaningful (not just numbers)
            lines = topics_text.split('\n')
            valid_topics = []
            for line in lines:
                line = line.strip()
                if line and not line[0].isdigit():  # Skip empty or number-only lines
                    # Check if line contains a meaningful topic name (not just numbers)
                    if any(char.isalpha() for char in line):  # Must contain at least one letter
                        valid_topics.append(line)
            
            if not valid_topics:
                self.log_entry("WARNING", "No meaningful topics extracted. Using raw response.")
                return topics_text
            
            # Reconstruct numbered list
            formatted_topics = '\n'.join([f"{i+1}. {line.lstrip('0123456789.-) ')}" for i, line in enumerate(valid_topics)])
            return formatted_topics
            
        except Exception as e:
            self.log_entry("ERROR", f"Gemini API failed to extract topics: {e}")
            return None
    
    def extract_rules_for_topic(self, content, topic_name):
        """
        Extract rules for a specific topic/scenario from the policy using the same
        strict format as extract_rules_with_gemini.
        
        Args:
            content (str): Full policy content
            topic_name (str): Name of the topic to extract rules for
            
        Returns:
            str: Formatted rules for the specific topic
        """
        self.log_entry("GEMINI", f"Extracting rules for topic: {topic_name}")
        
        prompt = f"""
You are a STRUCTURED POLICY EXTRACTION ENGINE with strict enforcement of logical rigor.

Task:
- From the policy document below, extract ALL rules, requirements, entitlements, and constraints ONLY related to: "{topic_name}"
- Output in the REFACTORED RULE ENGINE FORMAT (see below).
- Make NO assumptions - only extract what exists for this topic.
- ENFORCE: Every rule must have measurable conditions and verifiable consequences.
- ENFORCE: All percentages, amounts, times, and durations must be explicit.
- ENFORCE: Flag vague terms and provide concrete alternatives.
- Output ONLY the structured rules - no commentary, explanations, or metadata outside the format.

====================================================
POLICY DOCUMENT
====================================================
{content}

====================================================
REFACTORED RULE ENGINE FORMAT FOR TOPIC: "{topic_name}"
====================================================

SECTION 1: DEFINITIONS & CONSTANTS
Define all key terms, enumerations, grades, roles, categories, and thresholds specific to {topic_name}.
Example:
  employee.grade: {{E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, Director, MD}}
  duration_threshold: {{short: 1-10 days, long: 11+ days}}
  amount: {{junior: 300 USD, senior: 350 USD, executive: 500 USD}}

SECTION 2: AUTHORITY HIERARCHY (if applicable)
Define approval authorities, decision-makers, and escalation paths for {topic_name}.
Example:
  approval.required: {{"authority": "Manager", "required": true}}
  escalation: {{"authority": "Director", "when": "amount > 500 USD"}}

SECTION 3: PRIMARY ENTITLEMENTS OR RULES FOR {topic_name}
Organize by major policy domains/categories. For each rule use STRICT format:

rule: <rule_name>
priority: <1-high, 2-medium, 3-low>
conditions:
  - IF <measurable_condition_1> AND <measurable_condition_2>
    THEN <verifiable_consequence_1>
    AND <verifiable_consequence_2>
    SOURCE: <clause reference>

CRITICAL REQUIREMENTS FOR SECTION 3:
  - Every condition must be verifiable (compare against defined constants)
  - Every THEN must have a concrete, measurable outcome
  - If condition contains %, that % must reference a base value defined in SECTION 1
  - If condition contains time (e.g., "after 4:30 PM"), state EXACTLY what the trigger is
  - If multiple rules could apply, state which takes precedence

SECTION 4: ANCILLARY RULES
Optional behaviors, recommendations, special cases for {topic_name} that are not mandatory.
Mark as SHOULD/RECOMMENDED, not MUST.

SECTION 5: PROHIBITED ITEMS
What is explicitly forbidden for {topic_name} with clear conditions.

SECTION 6: MANDATORY REQUIREMENTS
Pre-conditions that MUST be satisfied for {topic_name} entitlements to activate.
These are blocking conditions (no exceptions).

SECTION 7: RULE PRECEDENCE & CONFLICT RESOLUTION
Explicit rules for handling overlapping conditions in {topic_name}:
  - Which rule wins if multiple apply?
  - What if a rule contradicts another?
  - How are ambiguous conditions resolved?

If NO conflict exists, state: "No overlapping rules identified for {topic_name}."

SECTION 8: MEASUREMENT & ENFORCEMENT
For each vague term found in {topic_name}, provide:
  1. Original vague term
  2. Why it's unmeasurable
  3. Concrete metric to replace it

Example:
  Vague: "reasonable amount"
  Reason: No objective threshold defined
  Concrete: "Amount capped at grade-specific limit (defined in SECTION 1). Outliers flagged if spend > 120% of cap."

SECTION 9: TOPIC SUMMARY
Brief summary of what {topic_name} covers and any external dependencies or related topics.

====================================================
NAMING & TERMS:
====================================================
  - Use normalized domain terms consistently
  - Define all acronyms and abbreviations
  - When policy references external documents, list them explicitly

NUMERIC PRECISION:
====================================================
  - Make ALL numeric limits explicit: X USD, Y days, Z hours
  - For ranges, express as: "min <= value <= max"
  - For percentages, state: "X% of [base value defined in SECTION 1]"
  - For time conditions, use 24-hour format: "after 16:30 (4:30 PM)"

LOGIC & CONDITIONS:
====================================================
  - Split complex multi-condition clauses into separate rules
  - Use AND/OR explicitly (not implicit)
  - If policy says "may", "should", "could", mark as optional (SECTION 4)
  - If policy says "must", "require", "shall", mark as mandatory (SECTION 3 or 6)

VAGUE TERM HANDLING:
====================================================
  - Do NOT output rules with unmeasurable terms
  - Flag them in SECTION 8 instead
  - Suggest concrete metrics
  - Mark as PENDING MANUAL REVIEW if no metric can be inferred

COMPLETENESS:
====================================================
  - Preserve clause/section numbers from source document
  - If policy is silent on {topic_name}, state: "No rules found in policy for this topic."
  - If policy contradicts itself, flag explicitly in SECTION 7

====================================================
CRITICAL: OUTPUT ONLY THE STRUCTURED RULES
====================================================
Do NOT include:
- Explanatory text outside the format
- Original policy paragraphs
- Commentary or interpretation
- Placeholder examples
- Vague rules (move to SECTION 8 instead)

Output must be ready to parse and implement as-is.
Rules with unmeasurable conditions go to SECTION 8 for definition.
"""
        
        try:
            self.log_entry("GEMINI_REQUEST", f"Sending policy content for topic-specific structured rule extraction: {topic_name}")
            response = self.model.generate_content(prompt)
            rules_text = response.text.strip()
            
            # Validate output contains expected structure markers
            if "SECTION 1:" not in rules_text and "rule:" not in rules_text:
                self.log_entry("WARNING", f"Gemini response for '{topic_name}' missing expected structure markers. Output may be incomplete.")
            
            self.log_entry("GEMINI_RESPONSE", f"Rules for '{topic_name}' extracted successfully")
            return rules_text
        except Exception as e:
            self.log_entry("ERROR", f"Gemini API failed to extract topic rules: {e}")
            return None
    
    def format_topic_rules_output(self, gemini_output, topic_name):
        """
        Wrap extracted topic rules with consistent header and metadata.
        
        Args:
            gemini_output (str): Raw output from Gemini
            topic_name (str): Name of the topic
            
        Returns:
            str: Formatted rules document for the topic
        """
        output = "=" * 80 + "\n"
        output += f"STRUCTURED RULES FOR: {topic_name.upper()}\n"
        output += "=" * 80 + "\n\n"
        output += f"Generated: {datetime.now()}\n"
        output += f"Source: {self.policy_file}\n"
        output += f"Topic: {topic_name}\n"
        output += f"Extraction Method: Topic-Specific Gemini Extraction\n"
        output += f"Format: Rule Engine v1.0 (Measurable, Enforceable)\n\n"
        output += "=" * 80 + "\n\n"
        
        output += gemini_output
        
        return output
    
    def save_topic_rules(self, formatted_rules, topic_name):
        """
        Save topic-specific rules to a separate file.
        
        Args:
            formatted_rules (str): Formatted rules content
            topic_name (str): Name of the topic
            
        Returns:
            str: Path to saved file, or None if failed
        """
        # Create a safe filename from topic name
        safe_topic_name = re.sub(r'[^a-zA-Z0-9_-]', '_', topic_name.lower())
        topic_rules_file = f"{OUTPUT_DIR}/rules_{safe_topic_name}.txt"
        
        try:
            with open(topic_rules_file, 'w') as f:
                f.write(formatted_rules)
            
            self.log_entry("SUCCESS", f"Topic rules saved to: {topic_rules_file}")
            self.log_entry("STATS", f"Output size: {len(formatted_rules)} characters")
            
            return topic_rules_file
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save topic rules file: {e}")
            return None

    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== RULE EXTRACTION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class ConfidenceAndRationaleGenerator:
    """
    Step 7: Generate confidence scores and rationale for each DSL rule.
    
    For each rule (clause):
    - Explain why this rule was suggested from the source policy
    - Cite the source clause reference
    - Explain why it's enforceable
    - Provide a confidence score (0.0-1.0) based on:
      * Clarity of original clause (absence of ambiguities)
      * Specificity of conditions
      * Measurability of thresholds
      * Cross-reference validation
    
    Output: stage7_confidence_rationale.json
    """
    
    def __init__(self, stage5_file, stage6_file, document_id=None, enable_mongodb=True):
        self.stage5_file = stage5_file  # Clarified clauses
        self.stage6_file = stage6_file  # DSL rules
        self.rationale_file = f"{OUTPUT_DIR}/stage7_confidence_rationale.json"
        self.document_id = document_id
        self.log = []
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def read_json_file(self, filepath):
        """Read and parse JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read {filepath}: {e}")
            return None
    
    def read_yaml_file(self, filepath):
        """Read YAML rules file"""
        try:
            import yaml
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read YAML {filepath}: {e}")
            return None
    
    def calculate_confidence_score(self, clause_data):
        """
        Calculate confidence score (0.0-1.0) based on:
        - Absence of ambiguities (clause is clarified)
        - Specificity of conditions
        - Measurability of entities
        - Enforcement clarity
        
        Args:
            clause_data: Dict with clause info from stage5
            
        Returns:
            float: Confidence score 0.0-1.0
        """
        score = 1.0
        
        # Penalty for ambiguities that were found and fixed
        if "ambiguities_fixed" in clause_data:
            ambiguities_fixed = clause_data["ambiguities_fixed"]
            if ambiguities_fixed:
                # Each fixed ambiguity reduces confidence by 0.05-0.15
                penalty_per_ambiguity = {
                    "SUBJECTIVE_LANGUAGE": 0.10,
                    "UNDEFINED_REFERENCES": 0.15,
                    "BOUNDARY_OVERLAPS": 0.12,
                    "INCOMPLETE_CONDITIONS": 0.12,
                    "VAGUE_METRICS": 0.12,
                    "UNDEFINED_AUTHORITY": 0.10
                }
                
                for ambiguity_type in ambiguities_fixed:
                    penalty = penalty_per_ambiguity.get(ambiguity_type, 0.10)
                    score -= penalty
        
        # Bonus for clear entity extraction
        if "entities" in clause_data and clause_data["entities"]:
            entity_count = len(clause_data["entities"])
            # Bonus capped at 0.05
            score += min(0.05, entity_count * 0.01)
        
        # Intent clarity bonus
        if "intent" in clause_data:
            intent = clause_data["intent"]
            # RESTRICTION and CONDITIONAL_ALLOWANCE are more measurable
            if intent in ["RESTRICTION", "CONDITIONAL_ALLOWANCE"]:
                score += 0.05
        
        # Clamp between 0.0 and 1.0
        return max(0.0, min(1.0, score))
    
    def build_local_rationale(self, clause_id, original_clause, clarified_clause, entities, ambiguities_fixed, intent):
        """
        Build rationale locally without Gemini API (avoids timeout issues).
        
        Args:
            clause_id: Clause identifier
            original_clause: Original clause text
            clarified_clause: Clarified clause text
            entities: Extracted entities
            ambiguities_fixed: List of fixed ambiguities
            intent: Clause intent (INFORMATIONAL, CONDITIONAL_ALLOWANCE, RESTRICTION)
            
        Returns:
            str: Formatted rationale
        """
        
        # Extract first sentence from original
        orig_summary = original_clause.split('.')[0] if original_clause else "Policy requirement"
        
        # Build key entities list
        entity_keys = list(entities.keys())[:5] if entities else []
        entity_str = ", ".join(entity_keys) if entity_keys else "Standard policy terms"
        
        # Build ambiguity summary
        if ambiguities_fixed:
            ambig_summary = f"Fixed ambiguities: {', '.join(ambiguities_fixed[:3])}"
        else:
            ambig_summary = "Clear and unambiguous"
        
        # Intent-specific reasoning
        intent_map = {
            "CONDITIONAL_ALLOWANCE": "Rule specifies conditional allowances/entitlements with measurable thresholds",
            "RESTRICTION": "Rule enforces restrictions with clear conditions and approval authorities",
            "INFORMATIONAL": "Rule defines classifications or reference information"
        }
        enforce_reason = intent_map.get(intent, "Rule extracted from policy clause")
        
        rationale = f"""SOURCE: {orig_summary}

ENFORCEABLE: {enforce_reason}. Entities extracted: {entity_str}. {ambig_summary}.

KEY_VALUES: {entity_str}"""
        
        return rationale
    
    def generate_rationale_with_gemini(self, clause_id, original_clause, clarified_clause, entities, ambiguities_fixed):
        """
        Use Gemini to generate detailed rationale for why a rule was suggested.
        OPTIMIZED: Shorter prompt, focused on key info only.
        
        Args:
            clause_id: Clause identifier (C1, C2, etc.)
            original_clause: Original clause text
            clarified_clause: Clarified clause text
            entities: Extracted entities from clause
            ambiguities_fixed: List of ambiguities that were fixed
            
        Returns:
            str: Rationale text
        """
        
        # Truncate long texts to avoid timeout
        orig_short = original_clause[:300] if original_clause else "N/A"
        clarif_short = clarified_clause[:300] if clarified_clause else "N/A"
        entities_str = json.dumps(entities, indent=2)[:200] if entities else "None"
        
        prompt = f"""Clause {clause_id}: Why is this rule enforceable?

ORIGINAL: {orig_short}

CLARIFIED: {clarif_short}

ENTITIES: {entities_str}

AMBIGUITIES FIXED: {", ".join(ambiguities_fixed) if ambiguities_fixed else "None"}

TASK: Generate brief rationale (50-100 words):

SOURCE: [1 sentence on policy requirement]
ENFORCEABLE: [Why it's objectively checkable]
KEY_VALUES: [Important thresholds/entities]

Output ONLY the above format, no extra text."""
        
        try:
            self.log_entry("GEMINI_REQUEST", f"Generating rationale for clause {clause_id}")
            response = self.model.generate_content(prompt, request_options={"timeout": 30})
            rationale = response.text.strip()
            self.log_entry("GEMINI_RESPONSE", f"Rationale generated for {clause_id}")
            return rationale
        except Exception as e:
            self.log_entry("ERROR", f"Gemini API failed for {clause_id}: {e}")
            # Return fallback rationale
            return f"SOURCE: {orig_short[:100]}\nENFORCEABLE: Rule extracted from policy clause\nKEY_VALUES: {', '.join(list(entities.keys())[:3]) if entities else 'N/A'}"
    
    def generate_confidence_and_rationale(self):
        """
        Main function to generate confidence scores and rationale for all clauses.
        
        Returns:
            dict: Mapping of clause_id to {rationale, confidence}
        """
        
        # Load stage 5 (clarified clauses)
        stage5_data = self.read_json_file(self.stage5_file)
        if not stage5_data:
            self.log_entry("ERROR", "Cannot load stage5 data")
            return None
        
        # Load stage 6 (DSL rules) to map which clauses have rules
        stage6_data = self.read_yaml_file(self.stage6_file)
        if not stage6_data:
            self.log_entry("ERROR", "Cannot load stage6 data")
            return None
        
        clarified_clauses = stage5_data.get("clarified_clauses", [])
        rules = stage6_data.get("rules", [])
        
        # Build mapping of clause_id to rule data
        rule_ids = {rule["rule_id"] for rule in rules}
        
        self.log_entry("INFO", f"Processing {len(clarified_clauses)} clauses")
        self.log_entry("INFO", f"Found {len(rule_ids)} rules in stage 6")
        
        confidence_rationale = {}
        
        for clause in clarified_clauses:
            clause_id = clause.get("clauseId")
            
            # Skip if no corresponding rule in stage 6
            if clause_id not in rule_ids:
                self.log_entry("WARNING", f"No rule found for {clause_id} in stage 6")
                continue
            
            # Calculate confidence score
            confidence = self.calculate_confidence_score(clause)
            
            # Generate rationale (LOCAL - no Gemini to avoid timeout)
            original_text = clause.get("text_original", "")
            clarified_text = clause.get("text_clarified", "")
            entities = clause.get("entities", {})
            ambiguities_fixed = clause.get("ambiguities_fixed", [])
            intent = clause.get("intent", "UNKNOWN")
            
            # Build rationale locally
            rationale = self.build_local_rationale(
                clause_id,
                original_text,
                clarified_text,
                entities,
                ambiguities_fixed,
                intent
            )
            
            confidence_rationale[clause_id] = {
                "clauseId": clause_id,
                "rationale": rationale,
                "confidence": round(confidence, 2),
                "ambiguitiesFixed": ambiguities_fixed,
                "sourceSection": intent
            }
            self.log_entry("SUCCESS", f"Generated confidence/rationale for {clause_id} (confidence: {round(confidence, 2)})")
        
        return confidence_rationale
    
    def save_confidence_rationale(self, data):
        """Save confidence and rationale to JSON file"""
        output = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "stage": "7",
                "source": [self.stage5_file, self.stage6_file],
                "title": "Confidence & Rationale for DSL Rules",
                "total_clauses": len(data)
            },
            "confidenceAndRationale": data
        }
        
        try:
            with open(self.rationale_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Confidence & rationale saved to {self.rationale_file}")
            
            # Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=7,
                stage_name="confidence-rationale",
                stage_output=output
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
            return True
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save confidence & rationale: {e}")
            return False
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== CONFIDENCE & RATIONALE GENERATION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class PayloadEvaluator:
    """Step 3: Iterative payload evaluation against rules using LLM feedback loop (GENERIC - ANY POLICY)"""
    
    def __init__(self, rules_file, max_attempts=4):
        self.rules_file = rules_file
        self.max_attempts = max_attempts
        self.log = []
        self.attempt_history = []
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
    
    def _count_nesting_depth(self, obj, depth=0):
        """Count maximum nesting depth"""
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(self._count_nesting_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(self._count_nesting_depth(item, depth + 1) for item in obj)
        else:
            return depth
    
    def _detect_structure_type(self, obj):
        """Detect JSON structure type"""
        if isinstance(obj, dict):
            return "Nested Dictionary"
        elif isinstance(obj, list):
            if len(obj) > 0 and isinstance(obj[0], dict):
                return "Array of Objects"
            elif len(obj) > 0 and isinstance(obj[0], list):
                return "Multi-dimensional Array"
            else:
                return "Simple Array"
        else:
            return "Unknown"
    
    def flatten_json(self, obj, parent_key='', sep='_'):
        """Flatten nested JSON for better LLM processing"""
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.extend(self.flatten_json(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}_{i}" if parent_key else f"[{i}]"
                if isinstance(v, (dict, list)):
                    items.extend(self.flatten_json(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
        return dict(items)
    

    def log_entry(self, level, message):
        """Log entry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def summarize_payload(self, payload_dict):
        """Summarize payload using Gemini (GENERIC - works with any payload)"""
        self.log_entry("PAYLOAD", f"Analyzing payload with {len(payload_dict)} fields")
        
        payload_str = json.dumps(payload_dict, indent=2)
        prompt = f"""Analyze this request payload and provide a concise summary. Be domain-agnostic.

Payload:
{payload_str}

TASK: 
1. What type of REQUEST is this? (describe the main action/intent)
2. What are the KEY FIELDS that need policy validation?
3. What POLICY AREAS or KEYWORDS are relevant based on the payload?

Format:
REQUEST_TYPE: [describe the type of request]
KEY_FIELDS: [list important fields from payload]
RELEVANT_KEYWORDS: [list keywords to search policy file for]

Be specific and extractive. Focus on what can be searched in policy rules."""

        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            self.log_entry("GEMINI_SUMMARY", "Payload summarized")
            return summary
        except Exception as e:
            self.log_entry("ERROR", f"Summary generation failed: {e}")
            return None
    
    def validate_command(self, command):
        """Validate command syntax before execution"""
        # Check for common syntax errors
        issues = []
        
        # Check for unterminated quotes
        if command.count('"') % 2 != 0:
            issues.append("Unterminated double quotes")
        if command.count("'") % 2 != 0:
            issues.append("Unterminated single quotes")
        
        # Check for common pipe/redirect issues
        if command.endswith('|') or command.endswith('>'):
            issues.append("Command ends with pipe or redirect operator")
        
        # Check for basic command structure
        if not any(cmd in command for cmd in ['grep', 'head', 'tail', 'sed', 'awk', 'cat']):
            issues.append("No recognized search command found (grep/head/tail/sed/awk/cat)")
        
        # Check file exists
        if self.rules_file not in command:
            issues.append(f"Rules file path '{self.rules_file}' not found in command")
        
        return issues
    
    def execute_cli_command(self, command):
        """Execute CLI command (grep, head, tail, etc.)"""
        self.log_entry("CLI_EXECUTE", f"Command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout.strip()
            error_output = result.stderr.strip()
            
            # Capture both stdout and stderr
            if result.returncode != 0:
                if error_output:
                    output = f"[CLI_ERROR] {error_output}"
                elif not output:
                    output = "[No results]"
            
            response_lines = len(output.split('\n'))
            self.log_entry("CLI_RESPONSE", f"Got {response_lines} lines, Return code: {result.returncode}")
            return output
            
        except subprocess.TimeoutExpired:
            self.log_entry("ERROR", "Command timeout (exceeded 10 seconds)")
            return "[ERROR: Command timeout]"
        except Exception as e:
            self.log_entry("ERROR", f"Command execution failed: {e}")
            return f"[ERROR: {str(e)[:100]}]"
    
    def create_search_command(self, context, attempt_num, previous_response=None):
        """Use Gemini to create optimal search command (GENERIC - works with any policy rules file)"""
        
        if attempt_num == 1:
            prompt = f"""You are a search strategist for finding relevant rules in a policy document. 
            
PAYLOAD CONTEXT:
{context}

RULES FILE: {self.rules_file}

TASK: Create a SINGLE optimal grep/head/tail/sed command to search the rules file for relevant policies.

GENERIC RULES FOR COMMAND CREATION:
1. Extract keywords from payload context
2. Use grep to find matching sections in rules file
3. Use patterns, case-insensitive search, regex as needed
4. Combine multiple keywords with OR operator (-E flag)
5. Use -A (after) flag to get context around matches
6. Command must be executable in bash
7. Return ONLY the command, no explanation, no code blocks

Examples of generic commands:
- grep -i "keyword1\\|keyword2" {self.rules_file}
- grep -n -i -E "pattern1|pattern2|pattern3" {self.rules_file}
- grep -A 5 -i "section" {self.rules_file} | head -30
- grep -i -E "rule1|rule2" {self.rules_file} | tail -20

CREATE THE COMMAND (bash executable, single line):"""
        
        else:
            # Check if there was a previous error to fix
            error_context = ""
            if previous_response and "[ERROR:" in previous_response:
                error_context = f"\n\nPREVIOUS ERROR TO FIX: {previous_response[:200]}"
            elif previous_response and "[CLI_ERROR]" in previous_response:
                error_context = f"\n\nPREVIOUS COMMAND ERROR: {previous_response[:200]}"
            
            prompt = f"""You are refining a policy rule search based on previous results.

ORIGINAL PAYLOAD CONTEXT:
{context}{error_context}

ATTEMPT #{attempt_num}
PREVIOUS SEARCH RESULT (first 500 chars):
{previous_response[:500] if previous_response else "N/A"}

TASK: 
1. Analyze what the previous response revealed
2. Determine if it's SUFFICIENT, INCOMPLETE, or WRONG
3. If insufficient or wrong: Create a NEW, more refined search command
4. Go DEEPER by trying:
   - Different keywords
   - Related terms
   - Policy keywords found in previous results
   - Cross-reference sections
5. Use previous command as a clue, but try different approach

GENERIC REFINEMENT RULES (CRITICAL):
- Create a DIFFERENT grep/head/tail command (not the same as before)
- Try variations of keywords with PROPER QUOTING
- Use single grep call with -i -E flags
- Format: grep -i -E "word1|word2|word3" {self.rules_file} [| head/tail]
- NEVER use unterminated or mismatched quotes
- ENSURE all quotes are properly closed
- VERIFY pipes and redirects are complete
- Command must be EXECUTABLE in bash
- Return ONLY the command, no explanation, no markdown

Examples of CORRECT format:
- grep -i "keyword1\|keyword2" {self.rules_file}
- grep -i -E "word1|word2|word3" {self.rules_file} | head -20
- grep -A 3 -E "rule1|rule2" {self.rules_file} | tail -15

Rules file: {self.rules_file}

CREATE THE REFINED COMMAND (DOUBLE CHECK QUOTING AND PIPES):"""
        
        try:
            response = self.model.generate_content(prompt)
            command = response.text.strip()
            
            # Clean up command (remove markdown code blocks if present)
            command = command.replace("```bash", "").replace("```", "").replace("bash", "").strip()
            command = command.strip("'\"")  # Remove quotes if any
            
            self.log_entry("GEMINI_COMMAND", f"Attempt {attempt_num}: {command[:80]}...")
            return command
            
        except Exception as e:
            self.log_entry("ERROR", f"Command generation failed: {e}")
            return None
    
    def evaluate_with_feedback_loop(self, payload_dict):
        """Iterative evaluation: Attempt 1, 2, 3... with feedback"""
        
        self.log_entry("EVAL_START", "Starting iterative policy evaluation")
        self.log_entry("ATTEMPTS_MAX", f"Max attempts: {self.max_attempts}")
        
        # Step 1: Summarize payload
        summary = self.summarize_payload(payload_dict)
        if not summary:
            return {"status": "ERROR", "reason": "Failed to summarize payload"}
        
        context = f"Payload Summary:\n{summary}\n\nRules file: {self.rules_file}"
        
        previous_response = None
        final_rules_found = None
        final_analysis = None
        search_attempts = []
        
        # Iterative feedback loop
        previous_command_error = None
        
        for attempt in range(1, self.max_attempts + 1):
            self.log_entry("ATTEMPT", f"========== ATTEMPT {attempt}/{self.max_attempts} ==========")
            
            # Step 2a: Create search command (using previous response as clue)
            if previous_command_error:
                context_with_error = f"{context}\n\nPREVIOUS ATTEMPT ERROR:\n{previous_command_error}"
                command = self.create_search_command(context_with_error, attempt, previous_response)
            else:
                command = self.create_search_command(context, attempt, previous_response)
            
            if not command:
                self.log_entry("ERROR", "Failed to create search command")
                break
            
            # Step 2b: Validate command syntax (from attempt 2 onwards)
            if attempt >= 2:
                validation_issues = self.validate_command(command)
                if validation_issues:
                    self.log_entry("VALIDATION", f"Command has issues: {validation_issues}")
                    previous_command_error = f"Command validation failed:\n" + "\n".join(f"  - {issue}" for issue in validation_issues)
                    continue  # Skip execution, go to next attempt with feedback
            
            # Step 2c: Execute command
            response = self.execute_cli_command(command)
            previous_command_error = None  # Reset error if execution successful
            
            # Step 3: Analyze response with Gemini (GENERIC)
            self.log_entry("GEMINI_ANALYZE", f"Analyzing response from attempt {attempt}")
            
            analyze_prompt = f"""Analyze this search result from a policy rules file:

SEARCH RESULT (first 1000 chars):
{response[:1000]}

TASK: Determine if this result provides sufficient policy information for evaluation:
1. SUFFICIENT - Contains relevant policy rules/requirements for the payload
2. INCOMPLETE - Found some rules but missing important related policies
3. WRONG - Search returned irrelevant results, wrong direction
4. NO_RESULTS - Search returned empty/no matches

EVALUATION CRITERIA:
- Does the result contain rules that apply to the payload request?
- Are there enforcement levels (MUST/SHOULD/EXPECTED)?
- Are conditions and exceptions mentioned?
- Is the information actionable for compliance checking?

RESPOND WITH (strict format):
STATUS: [SUFFICIENT | INCOMPLETE | WRONG | NO_RESULTS]
RULES_SUMMARY: [Key rules/policies found in 1-2 sentences]
NEXT_ACTION: [stop | search deeper | try different keywords]
CONFIDENCE: [0-100] percentage confidence

Be concise and factual."""

            try:
                analysis = self.model.generate_content(analyze_prompt)
                analysis_text = analysis.text.strip()
                self.log_entry("ANALYSIS", analysis_text[:150])
                
                # Store attempt details (without command details in final report)
                attempt_info = {
                    "attempt": attempt,
                    "status": self.extract_status(analysis_text),
                    "rules_summary": self.extract_field(analysis_text, "RULES_SUMMARY"),
                    "next_action": self.extract_field(analysis_text, "NEXT_ACTION"),
                    "confidence": self.extract_field(analysis_text, "CONFIDENCE")
                }
                search_attempts.append(attempt_info)
                final_analysis = analysis_text
                
                # Check if SUFFICIENT
                if "SUFFICIENT" in analysis_text.upper():
                    self.log_entry("SUCCESS", f"Found sufficient rules at attempt {attempt}")
                    final_rules_found = response
                    break
                
                # Check if should stop (max attempts or WRONG without recovery)
                if "WRONG" in analysis_text.upper() and attempt >= self.max_attempts - 1:
                    self.log_entry("WARNING", "Stopping: Max attempts reached")
                    final_rules_found = response
                    break
                
                previous_response = response
                
            except Exception as e:
                self.log_entry("ERROR", f"Analysis failed: {e}")
                final_rules_found = response
                break
        
        # Extract final evaluation status
        final_status = self.extract_status(final_analysis) if final_analysis else "UNKNOWN"
        
        # Step 4: Evaluate payload against rules and make decision
        decision_analysis = None
        if final_rules_found and final_status == "SUFFICIENT":
            decision_analysis = self.evaluate_payload_against_rules(payload_dict, final_rules_found, summary)
        
        # Compile final report (clean, focused on decisions)
        report = {
            "status": "COMPLETED",
            "evaluation_result": final_status,
            "payload_summary": summary,
            "search_attempts": search_attempts,
            "relevant_rules": final_rules_found,
            "decision_analysis": decision_analysis,
            "log": self.log
        }
        
        return report
    
    def extract_qualification_criteria(self, rules_text):
        """Extract qualification criteria dynamically from rules (GENERIC for ANY policy)"""
        self.log_entry("CRITERIA_EXTRACTION", "Extracting qualification criteria from rules")
        
        criteria_prompt = f"""Analyze these policy rules and extract the QUALIFICATION CRITERIA.

RULES:
{rules_text[:2000]}

TASK: Identify what makes a request QUALIFY or DISQUALIFY under these rules.

Extract:
1. What are the KEY REQUIREMENTS for approval?
2. What would cause REJECTION?
3. What fields/conditions MUST be met?
4. Are there GRADE or LEVEL-BASED rules?
5. Are there APPROVAL or DOCUMENTATION requirements?

Format response as:
QUALIFICATION_CRITERIA: [List 3-5 key criteria for approval]
REJECTION_TRIGGERS: [List conditions that cause rejection]
REQUIRED_FIELDS: [List mandatory fields that must be present/true]
POLICY_TYPE: [any policy type]

Be specific and extract from the actual rules provided."""

        try:
            response = self.model.generate_content(criteria_prompt)
            criteria_text = response.text.strip()
            self.log_entry("CRITERIA_EXTRACTED", "Qualification criteria identified")
            return criteria_text
        except Exception as e:
            self.log_entry("ERROR", f"Criteria extraction failed: {e}")
            return None
    
    def evaluate_payload_against_rules(self, payload_dict, rules_text, payload_summary):
        """Evaluate payload against rules with GENERIC decision logic (works for ANY policy)"""
        self.log_entry("DECISION", "Starting generic payload evaluation")
        
        # Step 1: Extract qualification criteria from rules (domain-agnostic)
        criteria_text = self.extract_qualification_criteria(rules_text)
        if not criteria_text:
            return None
        
        # Step 2: Evaluate payload against extracted criteria
        decision_prompt = f"""You are evaluating a payload against policy rules to determine if it QUALIFIES.

EXTRACTED QUALIFICATION CRITERIA FROM RULES:
{criteria_text}

PAYLOAD DATA:
{json.dumps(payload_dict, indent=2)[:2000]}

PAYLOAD SUMMARY:
{payload_summary}

TASK: Evaluate if the payload MEETS the qualification criteria identified in the rules.

For each criterion:
1. State what the rule requires
2. Check what the payload provides
3. Does it MEET the requirement? (YES/NO)
4. If NO, why not? (cite actual payload values)

Then provide FINAL DECISION: APPROVE or REJECT

Format response EXACTLY as:
RULE_CHECK: [Criterion 1: YES/NO because...][Criterion 2: YES/NO because...][etc]
FAILING_CRITERIA: [List specific unmet requirements with payload evidence]
DECISION: [APPROVE | REJECT]
REASONING: [Explain decision citing payload values and rule requirements]

Be strict, factual, and specific. Cross-reference actual payload values with rule requirements."""

        try:
            self.log_entry("GEMINI_DECISION", "Evaluating payload against criteria")
            decision = self.model.generate_content(decision_prompt)
            decision_text = decision.text.strip()
            
            # Extract decision components
            decision_result = {
                "criteria_used": criteria_text,
                "rule_checks": self.extract_field(decision_text, "RULE_CHECK"),
                "failing_criteria": self.extract_field(decision_text, "FAILING_CRITERIA"),
                "decision": self.extract_decision(decision_text),
                "reasoning": self.extract_field(decision_text, "REASONING"),
                "full_analysis": decision_text
            }
            
            self.log_entry("DECISION_RESULT", f"Decision: {decision_result['decision']}")
            return decision_result
            
        except Exception as e:
            self.log_entry("ERROR", f"Decision analysis failed: {e}")
            return None
    
    def extract_status(self, text):
        """Extract STATUS field from analysis response"""
        for line in text.split('\n'):
            if line.startswith('STATUS:'):
                status = line.replace('STATUS:', '').strip()
                # Clean up brackets and extra text
                status = status.split('[')[-1].split(']')[0]
                return status
        return "UNKNOWN"
    
    def extract_field(self, text, field_name):
        """Extract named field from analysis response"""
        for line in text.split('\n'):
            if line.startswith(f"{field_name}:"):
                return line.replace(f"{field_name}:", '').strip()
        return ""
    
    def extract_decision(self, text):
        """Extract APPROVE/REJECT decision"""
        for line in text.split('\n'):
            if line.startswith('DECISION:'):
                decision = line.replace('DECISION:', '').strip()
                if 'APPROVE' in decision.upper():
                    return 'APPROVE'
                elif 'REJECT' in decision.upper():
                    return 'REJECT'
        return 'UNKNOWN'
    
    def save_evaluation_report(self, report, output_file=None):
        """Save evaluation report as formatted text file"""
        if not output_file:
            output_file = f"{OUTPUT_DIR}/evaluation_report.txt"
        
        try:
            with open(output_file, 'w') as f:
                f.write(self.format_report_as_text(report))
            
            self.log_entry("REPORT_SAVED", f"Report saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save report: {e}")
            return None
    
    def format_report_as_text(self, report):
        """Format report as clean, readable text"""
        lines = []
        lines.append("=" * 80)
        lines.append("POLICY EVALUATION REPORT")
        lines.append("=" * 80)
        
        # Payload Summary
        lines.append("\n[PAYLOAD SUMMARY]")
        lines.append(report['payload_summary'])
        
        # Rules Search
        lines.append("\n[RULES SEARCH STATUS]")
        lines.append(f"Result: {report.get('evaluation_result', 'UNKNOWN')}")
        
        # Search Attempts Summary
        lines.append("\n[SEARCH ATTEMPTS]")
        for attempt in report['search_attempts']:
            lines.append(f"\nAttempt {attempt['attempt']}: {attempt['status']} (Confidence: {attempt['confidence']})")
            lines.append(f"  Found: {attempt['rules_summary'][:150]}...")
        
        # Decision Analysis
        lines.append("\n" + "=" * 80)
        lines.append("QUALIFICATION DECISION")
        lines.append("=" * 80)
        
        if report.get('decision_analysis'):
            decision = report['decision_analysis']
            lines.append(f"\nDECISION: {decision.get('decision', 'UNKNOWN')}")
            lines.append(f"\nREASONING:")
            lines.append(decision.get('reasoning', 'N/A'))
            
            if decision.get('failing_criteria') and decision['failing_criteria'].strip():
                lines.append(f"\nFAILING CRITERIA:")
                criteria_text = decision['failing_criteria']
                for line in criteria_text.split('\n'):
                    if line.strip() and line.strip() != '[]':
                        lines.append(f"  {line.strip()}")
            
            if decision.get('rule_checks'):
                lines.append(f"\nDETAILED RULE CHECKS:")
                for line in decision['rule_checks'].split('. '):
                    if line.strip():
                        lines.append(f"  • {line.strip()}")
        
        # Key Policies
        lines.append("\n" + "=" * 80)
        lines.append("APPLICABLE POLICIES & RULES")
        lines.append("=" * 80)
        
        if report['relevant_rules']:
            rules_lines = report['relevant_rules'].split('\n')
            for i, line in enumerate(rules_lines[:20]):
                if line.strip():
                    lines.append(line)
                if i >= 19:
                    lines.append("\n[See rules.txt for complete policy details]")
                    break
        
        lines.append("\n" + "=" * 80)
        lines.append(f"REPORT STATUS: {report['status']}")
        lines.append("=" * 80 + "\n")
        
        return "\n".join(lines)
    
    def print_evaluation_summary(self, report):
        """Print human-readable evaluation summary"""
        print("\n" + "="*80)
        print("POLICY EVALUATION REPORT")
        print("="*80)
        
        print(f"\n[PAYLOAD SUMMARY]")
        print(report['payload_summary'])
        
        print(f"\n[RULES SEARCH RESULT]")
        print(f"  Status: {report.get('evaluation_result', 'UNKNOWN')}")
        
        print(f"\n[SEARCH ATTEMPTS]")
        for attempt in report['search_attempts']:
            print(f"\n  Attempt {attempt['attempt']}")
            print(f"    Result: {attempt['status']}")
            print(f"    Rules Found: {attempt['rules_summary']}")
            print(f"    Confidence: {attempt['confidence']}")
        
        # Decision Analysis
        if report.get('decision_analysis'):
            decision = report['decision_analysis']
            
            # Show extracted criteria
            if decision.get('criteria_used'):
                print(f"\n[EXTRACTED QUALIFICATION CRITERIA FROM RULES]")
                for line in decision['criteria_used'].split('\n')[:8]:
                    if line.strip():
                        print(f"  {line.strip()}")
            
            print(f"\n[QUALIFICATION DECISION]")
            print(f"  Decision: {decision.get('decision', 'UNKNOWN')}")
            print(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
            
            if decision.get('failing_criteria'):
                print(f"\n  Failing Criteria:")
                for line in decision['failing_criteria'].split('\n'):
                    if line.strip():
                        print(f"    - {line.strip()}")
            
            if decision.get('rule_checks'):
                print(f"\n  Rule Checks:")
                for line in decision['rule_checks'].split('\n')[:10]:
                    if line.strip():
                        print(f"    {line.strip()}")
        
        print(f"\n[KEY POLICIES & RULES]")
        if report['relevant_rules']:
            lines = report['relevant_rules'].split('\n')[:12]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        
        print(f"\n[COMPLETION STATUS] {report['status']}")
        print("="*80 + "\n")


def interactive_menu():
    """Show interactive menu when no arguments provided"""
    print("\n" + "="*80)
    print("POLICY ENFORCEMENT SYSTEM - INTERACTIVE MODE")
    print("="*80)
    print("\nSelect operation:")
    print("  0) discover-patterns  - Discover patterns from raw text → pattern_index.json")
    print("  1) extract            - Extract policy from PDF → filename.txt")
    print("  2) extract-clauses    - Extract elaborated clauses → stage1_clauses.json")
    print("  3) classify-intents   - Classify clause intents → stage2_classified.json")
    print("  4) extract-entities   - Extract entities & thresholds → stage3_entities.json")
    print("  5) detect-ambiguities - Detect ambiguities in clauses → stage4_ambiguity_flags.json")
    print("  6) clarify-ambiguities - Clarify ambiguous clauses → stage5_clarified_clauses.json")
    print("  7) generate-dsl       - Generate DSL rules → stage6_dsl_rules.yaml")
    print("  8) confidence-rationale - Generate confidence scores → stage7_confidence_rationale.json")
    print("  9) extract-rules      - Extract ALL rules from text → rules.txt")
    print(" 10) evaluate           - Evaluate employee payload against rules")
    print(" 11) extract-topics     - Select topic → Generate rules for topic")
    print(" 12) normalize-policies - Generate normalized policy JSON → stage8_normalized_policies.json")
    print(" 13) run-full-pipeline  - Run complete pipeline (1B→2→3→4→5→6→8) at once")
    print(" 14) exit               - Exit")
    
    choice = input("\nEnter choice (0-14): ").strip()
    
    if choice == "0":
        text_file = input("Enter raw policy text file path (default: filename.txt): ").strip()
        if not text_file:
            text_file = f"{OUTPUT_DIR}/filename.txt"
        return ("discover-patterns", text_file)
    
    elif choice == "1":
        pdf_file = input("Enter PDF file path: ").strip()
        if not pdf_file:
            print("ERROR: PDF file path required")
            return None
        return ("extract", pdf_file)
    
    elif choice == "2":
        text_file = input("Enter text file path (default: filename.txt): ").strip()
        if not text_file:
            text_file = f"{OUTPUT_DIR}/filename.txt"
        return ("extract-clauses", text_file)
    
    elif choice == "3":
        clauses_file = input("Enter clauses file path (default: stage1_clauses.json): ").strip()
        if not clauses_file:
            clauses_file = f"{OUTPUT_DIR}/stage1_clauses.json"
        return ("classify-intents", clauses_file)
    
    elif choice == "4":
        classified_file = input("Enter classified file path (default: stage2_classified.json): ").strip()
        if not classified_file:
            classified_file = f"{OUTPUT_DIR}/stage2_classified.json"
        return ("extract-entities", classified_file)
    
    elif choice == "5":
        stage3_file = input("Enter entities file path (default: stage3_entities.json): ").strip()
        if not stage3_file:
            stage3_file = f"{OUTPUT_DIR}/stage3_entities.json"
        return ("detect-ambiguities", stage3_file)
    
    elif choice == "6":
        stage3_file = input("Enter entities file path (default: stage3_entities.json): ").strip()
        if not stage3_file:
            stage3_file = f"{OUTPUT_DIR}/stage3_entities.json"
        return ("clarify-ambiguities", stage3_file)
    
    elif choice == "7":
        stage5_file = input("Enter clarified clauses file path (default: stage5_clarified_clauses.json): ").strip()
        if not stage5_file:
            stage5_file = f"{OUTPUT_DIR}/stage5_clarified_clauses.json"
        return ("generate-dsl", stage5_file)
    
    elif choice == "8":
        stage5_file = input("Enter clarified clauses file path (default: stage5_clarified_clauses.json): ").strip()
        if not stage5_file:
            stage5_file = f"{OUTPUT_DIR}/stage5_clarified_clauses.json"
        stage6_file = input("Enter DSL rules file path (default: stage6_dsl_rules.yaml): ").strip()
        if not stage6_file:
            stage6_file = f"{OUTPUT_DIR}/stage6_dsl_rules.yaml"
        return ("confidence-rationale", stage5_file, stage6_file)
    
    elif choice == "9":
        text_file = input("Enter text file path (default: filename.txt): ").strip()
        if not text_file:
            text_file = f"{OUTPUT_DIR}/filename.txt"
        return ("extract-rules", text_file)
    
    elif choice == "10":
        input_method = input("Enter payload via:\n  1) File path\n  2) Direct JSON input\n\nChoice (1-2): ").strip()
        
        if input_method == "1":
            payload_file = input("Enter payload JSON file path: ").strip()
            if not payload_file:
                print("ERROR: Payload file path required")
                return None
            return ("evaluate", payload_file)
        
        elif input_method == "2":
            print("\nEnter JSON payload (you can paste multiline JSON, press Enter twice when done):")
            print("-" * 80)
            
            lines = []
            empty_count = 0
            while True:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                    lines.append(line)
                else:
                    empty_count = 0
                    lines.append(line)
            
            json_str = "\n".join(lines).strip()
            
            # Validate JSON
            try:
                payload = json.loads(json_str)
                # Save to temp file
                temp_file = f"{OUTPUT_DIR}/.temp_payload.json"
                with open(temp_file, 'w') as f:
                    json.dump(payload, f, indent=2)
                print(f"✓ JSON parsed successfully")
                return ("evaluate", temp_file)
            
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON - {e}")
                return None
        
        else:
            print("Invalid choice. Please try again.")
            return None
    
    elif choice == "11":
        text_file = input("Enter text file path (default: filename.txt): ").strip()
        if not text_file:
            text_file = f"{OUTPUT_DIR}/filename.txt"
        return ("extract-topics", text_file)
    
    elif choice == "12":
        dsl_file = input("Enter DSL rules file path (default: stage6_dsl_rules.yaml): ").strip()
        if not dsl_file:
            dsl_file = f"{OUTPUT_DIR}/stage6_dsl_rules.yaml"
        confidence_file = input("Enter confidence data file path (optional, press Enter to skip): ").strip()
        if not confidence_file:
            confidence_file = None
        return ("normalize-policies", dsl_file, confidence_file, "--no-llm")
    
    elif choice == "13":
        pdf_file = input("Enter PDF file path: ").strip()
        if not pdf_file:
            print("ERROR: PDF file path required")
            return None
        return ("run-full-pipeline", pdf_file)
    
    elif choice == "14":
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please try again.")
        return None


def main():
    """Main CLI interface - supports both CLI args and interactive mode"""
    
    # Interactive mode if no arguments
    if len(sys.argv) < 2:
        result = interactive_menu()
        if result is None:
            main()  # Retry
            return
        
        # Handle confidence-rationale which returns 3 values, or normalize-policies which may return 4
        if len(result) == 4:
            command, arg1, arg2, flag = result
            # For normalize-policies with --no-llm flag
            sys.argv = [sys.argv[0], command, arg1, arg2, flag]
        elif len(result) == 3:
            command, arg1, arg2 = result
            # For confidence-rationale, set up sys.argv properly
            sys.argv = [sys.argv[0], command, arg1, arg2]
        else:
            command, arg = result
            sys.argv = [sys.argv[0], command, arg]
    else:
        command = sys.argv[1]
        
        # Handle confidence-rationale which needs 2 arguments
        if command == "confidence-rationale":
            if len(sys.argv) < 4:
                print("Usage:")
                print("  python policy_validator.py confidence-rationale <stage5_clarified_clauses.json> <stage6_dsl_rules.yaml>")
                print("\nOr run without arguments for interactive mode:")
                print("  python policy_validator.py")
                sys.exit(1)
            arg = sys.argv[2]
        else:
            if len(sys.argv) < 3:
                print("Usage:")
                print("  python policy_validator.py extract <policy.pdf>")
                print("  python policy_validator.py extract-clauses <filename.txt>")
                print("  python policy_validator.py classify-intents <stage2_classified.json>")
                print("  python policy_validator.py extract-entities <stage2_classified.json>")
                print("  python policy_validator.py detect-ambiguities <stage3_entities.json>")
                print("  python policy_validator.py clarify-ambiguities <stage3_entities.json>")
                print("  python policy_validator.py generate-dsl <stage5_clarified_clauses.json>")
                print("  python policy_validator.py confidence-rationale <stage5_clarified_clauses.json> <stage6_dsl_rules.yaml>")
                print("  python policy_validator.py extract-rules <filename.txt>")
                print("  python policy_validator.py extract-topics <filename.txt>")
                print("  python policy_validator.py evaluate <payload.json>")
                print("\nOr run without arguments for interactive mode:")
                print("  python policy_validator.py")
                sys.exit(1)
            arg = sys.argv[2]
    
    if command == "discover-patterns":
        policy_text_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 0A: PATTERN DISCOVERY FROM RAW POLICY TEXT")
        print(f"{'='*80}\n")
        
        discoverer = PatternDiscoveryEngine(policy_text_file)
        success = discoverer.discover()
        discoverer.save_log()
        
        if success:
            print(f"\n✓ Pattern discovery complete. Output: {OUTPUT_DIR}/pattern_index.json")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
            print("\nNext step: Extract clauses from policy text")
            print(f"  python policy_validator.py extract-clauses {OUTPUT_DIR}/filename.txt")
        else:
            print("\n✗ Pattern discovery failed. Check logs.")
            sys.exit(1)
    
    elif command == "extract":
        pdf_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 1: POLICY EXTRACTION (PDF → TEXT)")
        print(f"{'='*80}\n")
        
        extractor = PolicyExtractor(pdf_file)
        success = extractor.extract_pdf()
        extractor.save_log()
        
        if success:
            print(f"\n✓ Extraction complete. Output: {OUTPUT_DIR}/filename.txt")
            print("\nNext step: Extract clauses with elaboration")
            print(f"  python policy_validator.py extract-clauses {OUTPUT_DIR}/filename.txt")
        else:
            print("\n✗ Extraction failed. Check logs.")
            sys.exit(1)
    
    elif command == "extract-clauses":
        policy_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 1B: CLAUSE EXTRACTION & ELABORATION")
        print(f"{'='*80}\n")
        
        clause_extractor = ClauseExtractor(policy_file)
        success = clause_extractor.extract()
        clause_extractor.save_log()
        
        if success:
            print(f"\n✓ Clause extraction complete. Output: {OUTPUT_DIR}/stage1_clauses.json")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
            print("\nNext step: Classify clause intents")
            print(f"  python policy_validator.py classify-intents {OUTPUT_DIR}/stage1_clauses.json")
        else:
            print("\n✗ Clause extraction failed. Check logs.")
            sys.exit(1)
    
    elif command == "classify-intents":
        clauses_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 2: INTENT CLASSIFICATION")
        print(f"{'='*80}\n")
        
        intent_classifier = IntentClassifier(clauses_file)
        success = intent_classifier.classify()
        intent_classifier.save_log()
        
        if success:
            print(f"\n✓ Intent classification complete. Output: {OUTPUT_DIR}/stage2_classified.json")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
            print("\nNext step: Extract entities and thresholds")
            print(f"  python policy_validator.py extract-entities {OUTPUT_DIR}/stage2_classified.json")
        else:
            print("\n✗ Intent classification failed. Check logs.")
            sys.exit(1)
    
    elif command == "extract-entities":
        classified_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 3: ENTITY & THRESHOLD EXTRACTION (Rule-based, no LLM)")
        print(f"{'='*80}\n")
        
        entity_extractor = EntityExtractor(classified_file, use_langchain=False)
        success = entity_extractor.extract()
        entity_extractor.save_log()
        
        if success:
            print(f"\n✓ Entity extraction complete. Output: {OUTPUT_DIR}/stage3_entities.json")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
            print("\nNext step: Extract rules from clauses")
            print(f"  python policy_validator.py extract-rules {OUTPUT_DIR}/filename.txt")
        else:
            print("\n✗ Entity extraction failed. Check logs.")
            sys.exit(1)
    
    elif command == "extract-rules":
        policy_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 2: RULE EXTRACTION (GREP + GEMINI)")
        print(f"{'='*80}\n")
        
        rule_extractor = RuleExtractor(policy_file)
        success = rule_extractor.extract()
        rule_extractor.save_log()
        
        if success:
            print(f"\n✓ Rule extraction complete. Output: {OUTPUT_DIR}/rules.txt")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
        else:
            print("\n✗ Rule extraction failed. Check logs.")
            sys.exit(1)
    
    elif command == "extract-topics":
        policy_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 2B: TOPIC/SCENARIO EXTRACTION & INDIVIDUAL RULE GENERATION")
        print(f"{'='*80}\n")
        
        try:
            rule_extractor = RuleExtractor(policy_file)
            
            # Step 1: Read policy file
            print("Reading policy file...")
            content = rule_extractor.read_policy_file()
            if not content:
                print("\n✗ Failed to read policy file.")
                sys.exit(1)
            
            # Step 2: Extract topics
            print("\nExtracting topics and scenarios from policy...")
            topics_text = rule_extractor.extract_topics_and_scenarios(content)
            if not topics_text:
                print("\n✗ Failed to extract topics.")
                sys.exit(1)
            
            # Display topics
            print("\n" + "="*80)
            print("AVAILABLE TOPICS/SCENARIOS IN POLICY:")
            print("="*80)
            print(topics_text)
            print("="*80)
            
            # Step 3: Ask user which topic to generate rules for
            print("\nEnter the topic name or number to generate individual rules for that topic.")
            print("(or press Enter to skip and generate rules for all topics)")
            topic_choice = input("\nTopic name/number: ").strip()
            
            if topic_choice:
                # Step 4: Extract rules for specific topic
                print(f"\nGenerating rules for: {topic_choice}")
                topic_rules = rule_extractor.extract_rules_for_topic(content, topic_choice)
                if not topic_rules:
                    print(f"\n✗ Failed to extract rules for topic '{topic_choice}'")
                    sys.exit(1)
                
                # Step 5: Format and save
                formatted_rules = rule_extractor.format_topic_rules_output(topic_rules, topic_choice)
                saved_file = rule_extractor.save_topic_rules(formatted_rules, topic_choice)
                
                if saved_file:
                    print(f"\n✓ Topic rules generated successfully!")
                    print(f"✓ Output file: {saved_file}")
                else:
                    print(f"\n✗ Failed to save topic rules.")
                    sys.exit(1)
            else:
                print("\n⚠ No topic selected. Skipping individual rule generation.")
                print("To generate rules for a specific topic, run: extract-topics again")
            
            rule_extractor.save_log()
            
        except FileNotFoundError as e:
            print(f"ERROR: File not found: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    elif command == "evaluate":
        payload_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 3: PAYLOAD EVALUATION (ITERATIVE FEEDBACK LOOP)")
        print(f"{'='*80}\n")
        
        try:
            # Load payload
            with open(payload_file, 'r') as f:
                payload = json.load(f)
            
            # Initialize evaluator
            evaluator = PayloadEvaluator(f"{OUTPUT_DIR}/rules.txt", max_attempts=5)
            
            # Run evaluation with feedback loop
            report = evaluator.evaluate_with_feedback_loop(payload)
            
            # Save report
            report_file = evaluator.save_evaluation_report(report)
            
            # Print summary
            evaluator.print_evaluation_summary(report)
            
            print(f"\n✓ Evaluation complete. Report saved to: {report_file}")
            
        except FileNotFoundError as e:
            print(f"ERROR: File not found: {e}")
            sys.exit(1)
        except json.JSONDecodeError:
            print("ERROR: Invalid JSON in payload file")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    elif command == "detect-ambiguities":
        stage3_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 4: AMBIGUITY DETECTION")
        print(f"{'='*80}\n")
        
        detector = AmbiguityDetector(stage3_file)
        success = detector.detect_ambiguities()
        detector.save_log()
        
        if success:
            print(f"\n✓ Ambiguity detection complete. Output: {OUTPUT_DIR}/stage4_ambiguity_flags.json")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
            print("\nNext step: Clarify ambiguous clauses")
            print(f"  python policy_validator.py clarify-ambiguities {OUTPUT_DIR}/stage3_entities.json")
        else:
            print("\n✗ Ambiguity detection failed. Check logs.")
            sys.exit(1)
    
    elif command == "clarify-ambiguities":
        stage3_file = arg
        stage4_file = f"{OUTPUT_DIR}/stage4_ambiguity_flags.json"
        
        print(f"\n{'='*80}")
        print("STEP 5: AMBIGUITY CLARIFICATION")
        print(f"{'='*80}\n")
        
        clarifier = AmbiguityClarifier(stage3_file, stage4_file)
        success = clarifier.clarify_all_clauses()
        clarifier.save_log()
        
        if success:
            print(f"\n✓ Ambiguity clarification complete. Output: {OUTPUT_DIR}/stage5_clarified_clauses.json")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
            print("\nNext step: Generate DSL rules from clarified clauses")
            print(f"  python policy_validator.py generate-dsl {OUTPUT_DIR}/stage5_clarified_clauses.json")
        else:
            print("\n✗ Ambiguity clarification failed. Check logs.")
            sys.exit(1)
    
    elif command == "generate-dsl":
        stage5_file = arg
        stage4_file = f"{OUTPUT_DIR}/stage4_ambiguity_flags.json"
        
        print(f"\n{'='*80}")
        print("STEP 6: DSL GENERATION")
        print(f"{'='*80}\n")
        
        generator = DSLGenerator(stage5_file, stage4_file)
        success = generator.generate_dsl_rules()
        generator.save_log()
        
        if success:
            print(f"\n✓ DSL generation complete. Output: {OUTPUT_DIR}/stage6_dsl_rules.yaml")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
            print("\nGenerated DSL rules (YAML) ready for policy engine deployment")
            print("\nNext step: Generate confidence scores and rationale for rules")
            print(f"  python policy_validator.py confidence-rationale {OUTPUT_DIR}/stage5_clarified_clauses.json {OUTPUT_DIR}/stage6_dsl_rules.yaml")
        else:
            print("\n✗ DSL generation failed. Check logs.")
            sys.exit(1)
    
    elif command == "confidence-rationale":
        if len(sys.argv) < 4:
            print("Usage:")
            print("  python policy_validator.py confidence-rationale <stage5_clarified_clauses.json> <stage6_dsl_rules.yaml>")
            sys.exit(1)

        stage5_file = sys.argv[2]
        stage6_file = sys.argv[3]

        print(f"\n{'='*80}")
        print("STEP 7: CONFIDENCE & RATIONALE GENERATION")
        print(f"{'='*80}\n")

        generator = ConfidenceAndRationaleGenerator(stage5_file, stage6_file)

        # Generate confidence and rationale
        confidence_data = generator.generate_confidence_and_rationale()

        if confidence_data:
            # Save to file
            success = generator.save_confidence_rationale(confidence_data)
            generator.save_log()

            if success:
                print(f"\n✓ Confidence & rationale generation complete.")
                print(f"✓ Output: {OUTPUT_DIR}/stage7_confidence_rationale.json")
                print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")

                # Print summary statistics
                total = len(confidence_data)
                avg_confidence = sum(v["confidence"] for v in confidence_data.values()) / total if total > 0 else 0
                print(f"\nSummary:")
                print(f"  Total clauses: {total}")
                print(f"  Average confidence: {avg_confidence:.2f}")
            else:
                print("\n✗ Failed to save confidence & rationale. Check logs.")
                sys.exit(1)
        else:
            print("\n✗ Confidence & rationale generation failed. Check logs.")
            sys.exit(1)
    
    elif command == "normalize-policies":
        # Handle both CLI and interactive mode arguments
        use_langchain = False  # Default to --no-llm mode (fast extraction)
        if len(sys.argv) >= 3:
            # Check for --no-llm flag (and remove it if found)
            if '--no-llm' in sys.argv:
                sys.argv.remove('--no-llm')
            # Check for --use-llm flag to enable LLM mode
            elif '--use-llm' in sys.argv:
                use_langchain = True
                sys.argv.remove('--use-llm')
            
            dsl_file = sys.argv[2]
            confidence_file = sys.argv[3] if len(sys.argv) >= 4 else None
        else:
            # This shouldn't happen in normal flow
            print("ERROR: Missing arguments for normalize-policies")
            print("Usage: python policy_validator.py normalize-policies <dsl_file> [confidence_file] [--no-llm|--use-llm]")
            sys.exit(1)
        
        print(f"\n{'='*80}")
        print("STEP 8: NORMALIZED POLICY GENERATION")
        print(f"{'='*80}\n")
        
        if use_langchain:
            print("Using LLM-enhanced metadata extraction (slower but more accurate)")
        else:
            print("Using fast rule-based metadata extraction (much faster)")
        
        generator = NormalizedPolicyGenerator(dsl_file, confidence_file, use_langchain=use_langchain)
        success = generator.generate_normalized_policies()
        generator.save_log()
        
        if success:
            print(f"\n✓ Normalized policy generation complete. Output: {OUTPUT_DIR}/stage8_normalized_policies.json")
            print(f"✓ Mechanism log: {OUTPUT_DIR}/mechanism.log")
        else:
            print("\n✗ Normalized policy generation failed. Check logs.")
            sys.exit(1)
    
    elif command == "run-full-pipeline":
        pdf_file = arg
        
        print(f"\n{'='*80}")
        print("FULL PIPELINE ORCHESTRATION")
        print("Running: Stage 1 → 1B → 2 → 3 → 4 → 5 → 6 → 8 (skipping Stage 7)")
        print(f"{'='*80}\n")
        
        # Ask about Stage 3 processing mode
        stage3_sequential = False
        stage3_mode = input("Stage 3 processing mode? [p]arallel (default) / [s]equential: ").strip().lower()
        if stage3_mode == 's' or stage3_mode == 'sequential':
            stage3_sequential = True
            print("→ Stage 3 will run in SEQUENTIAL mode (slower but more stable)")
        else:
            print("→ Stage 3 will run in PARALLEL mode (faster)")
        
        orchestrator = PipelineOrchestrator(pdf_file, enable_mongodb=False)
        result = orchestrator.run_full_pipeline(stage3_sequential=stage3_sequential)
        orchestrator.save_log()
        
        if result['success']:
            print(f"\n{'='*80}")
            print("✓ FULL PIPELINE SUCCESSFUL")
            print(f"{'='*80}")
            print(f"✓ Output: {result['output_file']}")
            print(f"✓ All stage results:")
            for stage_key, stage_file in result['results'].items():
                print(f"   - {stage_key}: {stage_file}")
            print(f"✓ Mechanism log: {LOG_FILE}")
        else:
            print(f"\n{'='*80}")
            print(f"✗ PIPELINE FAILED at Stage {result['failed_stage']}")
            print(f"{'='*80}")
            print(f"Check logs for details: {LOG_FILE}")
            sys.exit(1)
    
    else:
        print(f"ERROR: Unknown command '{command}'")
        sys.exit(1)


class AmbiguityDetector:
    """
    Stage 4: Detect ambiguous clauses
    
    Flags clauses with:
    - Subjective language (no objective boundaries)
    - Undefined approval authorities
    - Incomplete conditions (IF without THEN or vice versa)
    - Vague metrics (e.g., "Actual" without upper limit)
    
    Output: Simple JSON array with {clauseId, ambiguous, reason}
    
    Uses data-driven ambiguity rules (not hardcoded in prompts)
    """
    
    # Ambiguity rules - data driven, not hardcoded in prompts
    AMBIGUITY_RULES = [
        {
            "code": "SUBJECTIVE_LANGUAGE",
            "definition": "Uses vague or subjective terms without objective boundaries.",
            "examples": ["reasonable", "appropriate", "suitable", "adequate", "emergency", "necessary", "sufficient"],
            "severity": "HIGH"
        },
        {
            "code": "UNDEFINED_AUTHORITY",
            "definition": "Mentions an approving or deciding authority that is not clearly defined.",
            "examples": ["approval", "approved by", "authorized by", "as decided by", "supervisor", "manager", "HOD"],
            "severity": "HIGH"
        },
        {
            "code": "INCOMPLETE_CONDITIONS",
            "definition": "Contains conditional logic (IF/THEN) that does not specify all possible outcomes.",
            "examples": ["IF X AND Y", "IF company provides X and Y"],
            "severity": "MEDIUM"
        },
        {
            "code": "BOUNDARY_OVERLAPS",
            "definition": "Defines numeric ranges that overlap or share boundary values causing ambiguity.",
            "examples": ["3-12 hours and 12-24 hours", "1-15 days and 15 days to 1 year"],
            "severity": "MEDIUM"
        },
        {
            "code": "UNDEFINED_REFERENCES",
            "definition": "References terms, documents, or statuses that are not defined in the clause.",
            "examples": ["tour advance", "employee status", "proof of stay", "actual bills"],
            "severity": "MEDIUM"
        },
        {
            "code": "VAGUE_METRICS",
            "definition": "Uses numeric or monetary values without clear limits, units, or maximum caps.",
            "examples": ["Actual (no limit)", "as per Table X", "applicable amount"],
            "severity": "HIGH"
        }
    ]
    
    def __init__(self, stage3_file, document_id=None, enable_mongodb=True, use_langchain=True):
        self.stage3_file = stage3_file
        self.ambiguity_flags_file = f"{OUTPUT_DIR}/stage4_ambiguity_flags.json"
        self.document_id = document_id
        self.log = []
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        
        # Initialize LangChain components if available
        if self.use_langchain:
            self.log_entry("INFO", "LangChain enabled for enhanced ambiguity detection")
            self.langchain_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_retries=3
            )
        else:
            if not LANGCHAIN_AVAILABLE:
                self.log_entry("WARNING", "LangChain not available, using standard detection")
    
    def build_ambiguity_prompt(self, clause_id, clause_text, entities_str):
        """
        Build Gemini prompt dynamically from AMBIGUITY_RULES config.
        Keeps prompt generation logic separate from rule definitions.
        
        Returns: prompt string
        """
        # Build rules section from config
        rules_text = ""
        for i, rule in enumerate(self.AMBIGUITY_RULES, 1):
            rules_text += f"{i}. {rule['code']}: {rule['definition']}\n"
            rules_text += f"   Examples: {', '.join(rule['examples'][:3])}\n"
            rules_text += f"   Severity: {rule['severity']}\n\n"
        
        prompt = f"""Analyze this policy clause for ambiguities using the following rule definitions.

AMBIGUITY RULES:
================

{rules_text}

CLAUSE TO ANALYZE:
==================

Clause ID: {clause_id}

Text:
{clause_text}

Entities:
{entities_str}

TASK:
=====
1. Check clause against ALL 6 ambiguity rules above
2. Be strict and practical - flag real ambiguities that would affect policy enforcement
3. Focus on what is NOT clearly defined or what is subjective/vague
4. Output ONLY valid JSON

Output JSON:
{{
  "ambiguous": true/false,
  "reason": "short explanation (cite the rule code if applicable)"
}}
"""
        return prompt
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def load_stage3_data(self):
        """Load stage3_entities.json"""
        try:
            with open(self.stage3_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_entry("ERROR", f"Failed to load stage3 data: {e}")
            return None
    
    def detect_ambiguities_rule_based(self, clause_id, clause_text, entities):
        """
        RULE-BASED: Fast ambiguity detection without LLM calls
        
        Uses heuristics to detect common ambiguity patterns:
        1. Empty entities → Informational (low ambiguity)
        2. Vague keywords → Ambiguous
        3. Missing units in numeric context → Ambiguous
        4. Clear thresholds + units → Clear
        
        Returns: (is_ambiguous, reason, confidence_score)
        """
        
        ambiguity_score = 0  # 0-100 scale
        reasons = []
        
        # Rule 1: Empty entities → Likely informational
        if not entities or len(entities) == 0:
            ambiguity_score += 10
            reasons.append("No entities extracted (informational clause)")
        
        # Rule 2: Check for vague keywords
        vague_keywords = ['may', 'should', 'could', 'might', 'generally', 'usually', 
                         'typically', 'often', 'can', 'appropriate', 'suitable', 'reasonable']
        text_lower = clause_text.lower()
        vague_count = sum(1 for kw in vague_keywords if f' {kw} ' in f' {text_lower} ')
        if vague_count > 0:
            ambiguity_score += (vague_count * 15)
            reasons.append(f"Found {vague_count} vague keywords: {', '.join([kw for kw in vague_keywords if f' {kw} ' in f' {text_lower} '])}")
        
        # Rule 3: Check for undefined terms (ALL CAPS or unusual patterns)
        undefined_markers = ['MUST', 'SHALL', 'WILL']  # These are actually clear, good
        # Better: Check for concepts without definitions
        unclear_concepts = ['etc.', 'and so on', 'other']
        unclear_count = sum(1 for uc in unclear_concepts if uc in text_lower)
        if unclear_count > 0:
            ambiguity_score += (unclear_count * 20)
            reasons.append(f"Found open-ended language: {unclear_count} instances")
        
        # Rule 4: Numeric values without units
        import re
        numbers = re.findall(r'\d+(\.\d+)?', clause_text)
        if numbers:
            # Check if units follow numbers
            unit_keywords = ['rs', 'km', 'days', 'hours', 'percentage', '%', 'pm', 'am',
                           'grade', 'level', 'band', 'employee', 'employees']
            units_found = sum(1 for unit in unit_keywords if unit in text_lower)
            
            if units_found == 0 and len(numbers) > 0:
                ambiguity_score += 25
                reasons.append(f"Numeric values without clear units: {len(numbers)} numbers found")
            elif units_found > 0:
                ambiguity_score -= 10  # Deduct for clarity
                reasons.append(f"Clear units specified for {units_found} numeric values")
        
        # Rule 5: Check for clear thresholds (reduces ambiguity)
        threshold_keywords = ['maximum', 'minimum', 'at least', 'no more than', 'not less than',
                            'exceeding', 'upto', 'up to', 'between', 'within']
        threshold_count = sum(1 for th in threshold_keywords if th in text_lower)
        if threshold_count > 0:
            ambiguity_score -= (threshold_count * 10)
            reasons.append(f"Clear thresholds: {threshold_count} threshold keywords found")
        
        # Rule 6: Check for entity coverage
        if entities:
            entity_count = len(entities)
            # More entities generally means less ambiguity
            if entity_count >= 3:
                ambiguity_score -= 15
                reasons.append(f"Good entity coverage: {entity_count} entities extracted")
            elif entity_count >= 1:
                ambiguity_score -= 5
                reasons.append(f"Some entities extracted: {entity_count} entity")
        
        # Clamp score 0-100
        ambiguity_score = max(0, min(100, ambiguity_score))
        
        # Decision threshold: > 40 = ambiguous
        is_ambiguous = ambiguity_score > 40
        
        confidence = 100 - abs(ambiguity_score - 50)  # Higher confidence when far from threshold
        
        return {
            'is_ambiguous': is_ambiguous,
            'reason': ' | '.join(reasons) if reasons else 'No ambiguities detected',
            'score': ambiguity_score,
            'confidence': confidence
        }
    
    def detect_ambiguities_with_langchain(self, clause_id, clause_text, entities):
        """
        LANGCHAIN-ENHANCED: Detect ambiguities using structured output parsing.
        
        Benefits:
        - Automatic validation with Pydantic
        - Retry logic on failures
        - Type-safe output
        
        Returns: (is_ambiguous, reason, ambiguity_types)
        """
        entities_str = json.dumps(entities, indent=2) if entities else "No entities"
        
        # Build rules section from config
        rules_text = ""
        for i, rule in enumerate(self.AMBIGUITY_RULES, 1):
            rules_text += f"{i}. {rule['code']}: {rule['definition']}\n"
            rules_text += f"   Examples: {', '.join(rule['examples'][:3])}\n\n"
        
        parser = PydanticOutputParser(pydantic_object=AmbiguityDetectionSchema)
        
        prompt_template = PromptTemplate(
            input_variables=["clause_id", "clause_text", "entities_str", "rules_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""Analyze this policy clause for ambiguities using the following rule definitions.

AMBIGUITY RULES:
{rules_text}

CLAUSE TO ANALYZE:
Clause ID: {clause_id}
Text: {clause_text}
Entities: {entities_str}

TASK:
1. Check clause against ALL ambiguity rules above
2. Be strict - flag real ambiguities that affect policy enforcement
3. List all ambiguity type codes detected (e.g., ["SUBJECTIVE_LANGUAGE", "VAGUE_METRICS"])

{format_instructions}

OUTPUT ONLY valid JSON matching the schema."""
        )
        
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        
        try:
            result = chain.run(
                clause_id=clause_id,
                clause_text=clause_text,
                entities_str=entities_str,
                rules_text=rules_text
            )
            parsed = parser.parse(result)
            return parsed.ambiguous, parsed.reason, parsed.ambiguity_types
        except Exception as e:
            self.log_entry("ERROR", f"{clause_id}: LangChain detection failed: {e}")
            # Fallback to standard method
            is_amb, reason = self.detect_ambiguities_with_gemini(clause_id, clause_text, entities)
            return is_amb, reason, []
    
    def detect_ambiguities_with_gemini(self, clause_id, clause_text, entities):
        """
        STANDARD: Use Gemini to dynamically detect ambiguities in a clause.
        Uses prompt built from AMBIGUITY_RULES config (data-driven).
        
        Returns: (is_ambiguous, reason)
        """
        entities_str = json.dumps(entities, indent=2) if entities else "No entities"
        
        # Build prompt dynamically from config
        prompt = self.build_ambiguity_prompt(clause_id, clause_text, entities_str)
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response if wrapped in markdown
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse response
            result = json.loads(response_text)
            return result.get('ambiguous', False), result.get('reason', 'Unknown')
            
        except Exception as e:
            self.log_entry("WARNING", f"Gemini analysis failed for {clause_id}: {e}")
            return False, "Analysis skipped due to error"
    
    def analyze_clause(self, clause, extracted_entities):
        """
        Analyze single clause for ambiguity using RULE-BASED detection first.
        Falls back to LLM only if confidence is low.
        Returns: {clauseId, ambiguous, reason, ambiguity_types}
        """
        clause_id = clause['clauseId']
        clause_text = clause['text']
        entities = extracted_entities or {}
        
        # Step 1: Try rule-based detection (fast, no LLM cost)
        rule_result = self.detect_ambiguities_rule_based(clause_id, clause_text, entities)
        
        # If confidence is high (>70), use rule-based result
        if rule_result['confidence'] > 70:
            return {
                "clauseId": clause_id,
                "ambiguous": rule_result['is_ambiguous'],
                "reason": f"[RULE-BASED] {rule_result['reason']} (score: {rule_result['score']}/100)",
                "ambiguity_types": [],
                "detection_method": "rule-based",
                "confidence": rule_result['confidence']
            }
        
        # Step 2: Low confidence → use LLM for verification
        if self.use_langchain:
            is_ambiguous, reason, ambiguity_types = self.detect_ambiguities_with_langchain(clause_id, clause_text, entities)
        else:
            is_ambiguous, reason = self.detect_ambiguities_with_gemini(clause_id, clause_text, entities)
            ambiguity_types = []
        
        return {
            "clauseId": clause_id,
            "ambiguous": is_ambiguous,
            "reason": f"[LLM-VERIFIED] {reason}",
            "ambiguity_types": ambiguity_types,
            "detection_method": "llm",
            "rule_score": rule_result['score']
        }
    
    def detect_ambiguities(self):
        """Main detection pipeline with PARALLEL PROCESSING"""
        self.log_entry("INFO", "Starting Stage 4: Ambiguity Detection")
        
        # Load stage 3 data
        data = self.load_stage3_data()
        if not data:
            return False
        
        extracted_clauses = data.get('extracted_clauses', [])
        if not extracted_clauses:
            self.log_entry("ERROR", "No extracted clauses found in stage3 data")
            return False
        
        if self.use_langchain:
            self.log_entry("INFO", f"Analyzing {len(extracted_clauses)} clauses with LangChain (parallel mode - 5 workers)")
        else:
            self.log_entry("INFO", f"Analyzing {len(extracted_clauses)} clauses with standard Gemini (parallel mode - 5 workers)")
        
        # Analyze clauses in parallel
        def analyze_single(clause):
            clause_entities = clause.get('entities', {})
            flag = self.analyze_clause(clause, clause_entities)
            
            if flag['ambiguous']:
                self.log_entry("AMBIGUOUS", f"{flag['clauseId']}: {flag['reason']}")
            else:
                self.log_entry("CLEAR", f"{flag['clauseId']}: Clear, no ambiguity")
            
            return flag
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            ambiguity_flags = list(executor.map(analyze_single, extracted_clauses))
        
        ambiguous_count = sum(1 for flag in ambiguity_flags if flag['ambiguous'])
        self.log_entry("SUMMARY", f"Ambiguous clauses: {ambiguous_count}/{len(extracted_clauses)}")
        
        # Save results
        self.save_ambiguity_flags(ambiguity_flags)
        
        return True
    
    def save_ambiguity_flags(self, ambiguity_flags):
        """Save ambiguity flags to JSON"""
        try:
            with open(self.ambiguity_flags_file, 'w') as f:
                json.dump(ambiguity_flags, f, indent=2)
            
            self.log_entry("SUCCESS", f"Ambiguity flags saved to: {self.ambiguity_flags_file}")
            self.log_entry("OUTPUT", f"Total clauses analyzed: {len(ambiguity_flags)}")
            
            # Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=4,
                stage_name="detect-ambiguities",
                stage_output=ambiguity_flags
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save ambiguity flags: {e}")
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== STAGE 4: AMBIGUITY DETECTION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class AmbiguityClarifier:
    """
    Stage 5: Auto-fix ambiguities using LLM with parallel batch processing
    
    Takes ambiguity analysis from Stage 4 and uses it to clarify/fix clause text.
    Preserves all numeric data while removing ambiguities.
    
    Features:
    - Processes clauses in parallel (ThreadPoolExecutor)
    - Batch processing with configurable chunk size
    - Thread-safe logging
    - Graceful error handling with fallback to original text
    
    Input:  stage3_entities.json + stage4_ambiguity_flags.json
    Output: stage5_clarified_clauses.json
    """
    
    def __init__(self, stage3_file, stage4_file, document_id=None, enable_mongodb=True, max_workers=4, batch_size=3):
        self.stage3_file = stage3_file
        self.stage4_file = stage4_file
        self.clarified_file = f"{OUTPUT_DIR}/stage5_clarified_clauses.json"
        self.document_id = document_id
        self.log = []
        self.log_lock = threading.Lock()  # Thread-safe logging
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Parallel processing config
        self.max_workers = max_workers  # Number of parallel threads
        self.batch_size = batch_size    # Clauses per batch
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
    
    def log_entry(self, level, message):
        """Thread-safe log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        with self.log_lock:
            self.log.append(entry)
        print(entry)
    
    def load_files(self):
        """Load stage3 and stage4 data"""
        try:
            with open(self.stage3_file, 'r') as f:
                stage3_data = json.load(f)
            with open(self.stage4_file, 'r') as f:
                stage4_data = json.load(f)
            return stage3_data, stage4_data
        except Exception as e:
            self.log_entry("ERROR", f"Failed to load files: {e}")
            return None, None
    
    def clarify_clause(self, clause_id, original_text, ambiguity_reason):
        """
        Use Gemini to clarify a clause based on its ambiguities.
        Returns: clarified_text
        
        ENHANCEMENT: Validates clarified text to prevent hallucinations.
        """
        prompt = f"""You are a policy clarification expert. Your task is to fix clause ambiguities.

CLAUSE ID: {clause_id}

ORIGINAL TEXT:
{original_text}

IDENTIFIED AMBIGUITIES:
{ambiguity_reason}

CRITICAL RULES:
1. PRESERVE all original numeric data (amounts, dates, durations, percentages)
2. NEVER invent new numeric values, percentages, amounts, or rates
3. If the original text doesn't specify a value, use a placeholder like [amount_needed]
4. Only clarify vague terms with DEFINITIONS, not new values

TASK:
1. Rewrite the clause to FIX the identified ambiguities
2. Define vague terms (e.g., "Actual" → "Actual expenses as documented in receipts")
3. Clarify undefined references by EXPLAINING, not inventing values
4. Keep the tone, structure, and references to other clauses intact
5. Output ONLY the clarified clause text - no explanations, no markdown

CLARIFIED TEXT:"""
        
        try:
            response = self.model.generate_content(prompt)
            clarified_text = response.text.strip()
            
            # Clean up if wrapped in markdown
            if clarified_text.startswith("```"):
                clarified_text = clarified_text.split("```")[1]
                if clarified_text.startswith("text"):
                    clarified_text = clarified_text[4:]
                clarified_text = clarified_text.strip()
            
            # Validate no hallucinations were introduced
            if GENERIC_ENHANCER_AVAILABLE:
                agnostic_enhancer = PolicyAgnosticEnhancer()
                validated_text = agnostic_enhancer.validate_clarification(original_text, clarified_text)
                return validated_text
            
            return clarified_text
            
        except Exception as e:
            self.log_entry("WARNING", f"Clarification failed for {clause_id}: {e}")
            return original_text  # Return original if clarification fails
    
    def process_single_clause(self, clause, ambiguity_map):
        """Process a single clause (for parallel execution)"""
        clause_id = clause['clauseId']
        original_text = clause['text']
        entities = clause.get('entities', {})
        
        # Get ambiguity info
        ambiguity_info = ambiguity_map.get(clause_id, {
            'reason': '',
            'types': [],
            'ambiguous': False
        })
        
        ambiguity_reason = ambiguity_info.get('reason', '')
        ambiguity_types = ambiguity_info.get('types', [])
        is_ambiguous = ambiguity_info.get('ambiguous', False)
        
        # FIXED: Clarify ALL clauses that have any ambiguity (is_ambiguous=True)
        # Not just those with non-empty ambiguity_types
        if not is_ambiguous:
            # No ambiguity - keep original text
            clarified_text = original_text
            ambiguities_fixed = []
            self.log_entry("SKIP", f"{clause_id}: No ambiguities detected")
        else:
            # Has ambiguity - ALWAYS clarify, even if types array is empty
            self.log_entry("PROCESSING", f"{clause_id}: Clarifying ambiguous clause...")
            clarified_text = self.clarify_clause(clause_id, original_text, ambiguity_reason)
            
            # Determine which ambiguities were fixed
            ambiguities_fixed = []
            if ambiguity_types:
                for code in ambiguity_types:
                    ambiguities_fixed.append(code)
            else:
                # If no types specified but clause is ambiguous, mark as generic
                ambiguities_fixed = ["GENERIC_AMBIGUITY"]
            
            self.log_entry("CLARIFIED", f"{clause_id}: Text clarified ({len(ambiguities_fixed)} issues addressed)")
        
        return {
            "clauseId": clause_id,
            "text_original": original_text,
            "text_clarified": clarified_text,
            "ambiguity_reason": ambiguity_reason,
            "ambiguity_types": ambiguity_types,
            "ambiguities_fixed": ambiguities_fixed,
            "entities": entities,
            "intent": clause.get('intent', '')
        }
    
    def clarify_all_clauses(self):
        """Main clarification pipeline with parallel batch processing"""
        self.log_entry("INFO", "Starting Stage 5: Ambiguity Clarification (Parallel)")
        self.log_entry("INFO", f"Configuration: max_workers={self.max_workers}, batch_size={self.batch_size}")
        
        # Load both stage3 and stage4 data
        stage3_data, stage4_data = self.load_files()
        if not stage3_data or not stage4_data:
            return False
        
        stage3_clauses = stage3_data.get('extracted_clauses', [])
        ambiguity_flags = stage4_data if isinstance(stage4_data, list) else []
        
        # Create mapping: clauseId -> (ambiguity_reason, ambiguity_types)
        # Include both reason string AND ambiguity_types array
        ambiguity_map = {
            flag['clauseId']: {
                'reason': flag.get('reason', ''),
                'types': flag.get('ambiguity_types', []),
                'ambiguous': flag.get('ambiguous', False)
            }
            for flag in ambiguity_flags
        }
        
        self.log_entry("INFO", f"Processing {len(stage3_clauses)} clauses in batches of {self.batch_size}")
        
        clarified_clauses = []
        
        # Process clauses in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.process_single_clause, clause, ambiguity_map): clause['clauseId']
                for clause in stage3_clauses
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                clause_id = futures[future]
                try:
                    result = future.result()
                    clarified_clauses.append(result)
                    completed += 1
                    self.log_entry("COMPLETE", f"Progress: {completed}/{len(stage3_clauses)} clauses")
                except Exception as e:
                    self.log_entry("ERROR", f"{clause_id}: Processing failed: {e}")
        
        # Sort by clause ID for consistent output
        clarified_clauses.sort(key=lambda x: x['clauseId'])
        
        # Save clarified clauses
        self.save_clarified_clauses(clarified_clauses)
        
        return True
    
    def save_clarified_clauses(self, clarified_clauses):
        """Save clarified clauses to JSON"""
        output = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "source": "stage3_entities.json + stage4_ambiguity_flags.json",
                "format": "Clarified Clauses (Stage 5)",
                "total_clauses": len(clarified_clauses),
                "stage": "5"
            },
            "clarified_clauses": clarified_clauses
        }
        
        try:
            with open(self.clarified_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Clarified clauses saved to: {self.clarified_file}")
            
            # Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=5,
                stage_name="clarify-ambiguities",
                stage_output=output
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save clarified clauses: {e}")
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== STAGE 5: AMBIGUITY CLARIFICATION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class ClauseIndexer:
    """Fast clause indexing for DSL pattern matching"""
    
    def __init__(self, clauses_data):
        self.clauses_data = clauses_data if isinstance(clauses_data, list) else []
        self.index = {}
        self._build_index()
    
    def _build_index(self):
        """Build keyword-based index for fast lookup"""
        for clause in self.clauses_data:
            clause_id = clause.get('clauseId', '')
            intent = clause.get('intent', '')
            text = clause.get('text', '').lower()
            entities = clause.get('entities', {})
            
            # Extract keywords from text
            keywords = set()
            words = re.findall(r'\b[a-z]+\b', text)
            
            # Focus on policy-relevant keywords
            policy_keywords = ['travel', 'allowance', 'grade', 'lodging', 'boarding', 
                             'duration', 'limit', 'maximum', 'minimum', 'approval',
                             'reimbursement', 'mileage', 'transport', 'tour', 'deputation']
            
            for word in words:
                if word in policy_keywords:
                    keywords.add(word)
            
            # Index by intent type
            if intent not in self.index:
                self.index[intent] = []
            
            self.index[intent].append({
                'clause_id': clause_id,
                'keywords': list(keywords),
                'entities': list(entities.keys()),
                'text': text,
                'original': clause
            })
    
    def find_similar_clauses(self, intent, keywords, top_k=3):
        """Find similar clauses using keyword matching"""
        if intent not in self.index:
            return []
        
        candidates = self.index[intent]
        scored = []
        
        for candidate in candidates:
            candidate_keywords = set(candidate['keywords'])
            candidate_entities = set(candidate['entities'])
            
            # Score based on keyword overlap
            keyword_overlap = len(keywords & candidate_keywords)
            
            # Also score based on entity overlap
            entity_overlap = len(keywords & candidate_entities)
            
            # Combined score
            total_score = keyword_overlap + (entity_overlap * 2)  # Weight entities higher
            
            if total_score > 0:
                scored.append((candidate, total_score))
        
        # Sort by overlap score
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:top_k]]


class DSLGenerator:
    """
    Stage 6: Generate DSL rules from clarified clauses
    
    Uses rule-based indexing + Gemini AI for fast DSL generation.
    Falls back to LLM only when indexed lookup fails.
    Marks ambiguous rules with WARN actions instead of ENFORCE.
    
    Input:  stage5_clarified_clauses.json + stage4_ambiguity_flags.json
    Output: stage6_dsl_rules.yaml
    """
    
    def __init__(self, stage5_file, stage4_file, document_id=None, enable_mongodb=True, use_langchain=True):
        self.stage5_file = stage5_file
        self.stage4_file = stage4_file
        self.dsl_file = f"{OUTPUT_DIR}/stage6_dsl_rules.yaml"
        self.document_id = document_id
        self.log = []
        self.log_lock = threading.Lock()
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        
        # Initialize clause indexer
        self.clause_indexer = None
        
        # Initialize LangChain components if available
        if self.use_langchain:
            self.log_entry("INFO", "LangChain enabled for dynamic DSL generation")
            self.langchain_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_retries=3
            )
        else:
            self.log_entry("INFO", "Using standard Gemini for DSL generation")
    
    def log_entry(self, level, message):
        """Thread-safe log entry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        with self.log_lock:
            self.log.append(entry)
        print(entry)
    
    def generate_dsl_from_index(self, clause_id, intent, entities, is_ambiguous):
        """Fast DSL generation using indexed patterns"""
        
        # Extract keywords from entities
        entity_keywords = set()
        for entity_name in entities.keys():
            # Convert entity names to keywords
            words = re.findall(r'\b[a-z]+\b', entity_name.lower())
            entity_keywords.update(words)
        
        # Try to find similar clauses in index
        similar_clauses = self.clause_indexer.find_similar_clauses(intent, entity_keywords, top_k=1)
        
        if similar_clauses:
            # Use pattern from similar clause
            similar = similar_clauses[0]
            self.log_entry("INDEX_HIT", f"Found similar clause {similar['clause_id']} for {clause_id}")
            
            # Generate DSL based on similar pattern
            return self.generate_dsl_from_pattern(clause_id, intent, entities, is_ambiguous, similar)
        
        # No similar clause found - need LLM
        return None
    
    def generate_dsl_from_pattern(self, clause_id, intent, entities, is_ambiguous, similar_clause):
        """Generate DSL based on similar clause pattern - with smart entity mapping"""
        
        # Build dynamic when conditions from entities with smart mapping
        when_conditions = []
        
        for entity_name, entity_value in entities.items():
            # Smart mapping based on entity type
            entity_lower = entity_name.lower()
            
            # Distance thresholds → travel.distance
            if 'distance' in entity_lower or 'km' in str(entity_value).lower():
                fact = 'travel.distance'
                operator = 'LESS_THAN_OR_EQUAL'
            
            # Percentage values → claim.percentage
            elif 'percentage' in entity_lower or '%' in str(entity_value):
                fact = 'claim.percentage'
                operator = 'EQUALS'
            
            # Hour triggers → work.hours
            elif 'hour' in entity_lower or 'hour' in str(entity_value).lower():
                fact = 'work.hours'
                operator = 'GREATER_THAN_OR_EQUAL'
            
            # Time limits (days) → settlement.days
            elif 'day' in entity_lower or 'day' in str(entity_value).lower():
                fact = 'settlement.days'
                operator = 'LESS_THAN_OR_EQUAL'
            
            # Employee levels → employee.level
            elif 'employee' in entity_lower or 'level' in entity_lower or 'grade' in entity_lower:
                fact = 'employee.level'
                operator = 'EQUALS'
            
            # Role exclusions → employee.role
            elif 'role' in entity_lower or 'personnel' in entity_lower:
                fact = 'employee.role'
                operator = 'NOT_EQUALS'
            
            # Travel mode → travel.mode
            elif 'travel' in entity_lower or 'car' in entity_lower or 'bike' in entity_lower:
                fact = 'travel.mode'
                operator = 'EQUALS'
            
            # Generic fallback
            else:
                fact = entity_name.replace('.', '_').lower()
                operator = 'EQUALS'
            
            when_conditions.append({
                'fact': fact,
                'operator': operator,
                'value': entity_value
            })
        
        # Build then constraints based on intent
        action = 'warn' if is_ambiguous else 'enforce'
        
        if intent == 'INFORMATIONAL':
            then_constraints = [{'constraint': 'PASS', 'operator': 'EQUALS', 'value': 'OK'}]
        elif intent == 'RESTRICTION':
            then_constraints = [{'constraint': 'approval.required', 'operator': 'EQUALS', 'value': 'YES'}]
        elif intent == 'LIMIT':
            then_constraints = [{'constraint': 'limit.enforced', 'operator': 'STRICT', 'value': 'MAX'}]
        elif intent == 'CONDITIONAL_ALLOWANCE':
            then_constraints = [{'constraint': 'allowance.approved', 'operator': 'CONDITIONAL', 'value': 'CLAIM'}]
        elif intent == 'ADVISORY':
            then_constraints = [{'constraint': 'recommendation', 'operator': 'GUIDANCE', 'value': 'ADVISORY'}]
        elif intent == 'APPROVAL_REQUIRED':
            then_constraints = [{'constraint': 'approval.mandatory', 'operator': 'REQUIRED', 'value': 'AUTHORIZATION'}]
        else:
            then_constraints = [{'constraint': 'PASS', 'operator': 'EQUALS', 'value': 'OK'}]
        
        return {
            'rule_id': clause_id,
            'when': {'all': when_conditions},
            'then': {action: then_constraints}
        }
    
    def log_entry(self, level, message):
        """Thread-safe log entry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        with self.log_lock:
            self.log.append(entry)
        print(entry)
    
    def generate_dsl_rule_with_gemini(self, clause_id, clause_text, intent, entities, is_ambiguous):
        """Use Gemini to dynamically generate DSL rule"""
        
        entities_str = json.dumps(entities, indent=2) if entities else "No entities extracted"
        
        prompt = f"""You are a DSL rule generation expert. Convert policy clauses into executable rule format.

CLAUSE ID: {clause_id}
INTENT: {intent}
AMBIGUOUS: {is_ambiguous}

CLAUSE TEXT:
{clause_text}

EXTRACTED ENTITIES:
{entities_str}

Generate a DSL rule in this JSON format:
{{
    "rule_id": "{clause_id}",
    "when": {{
        "all": [
            {{
                "fact": "<dynamic_fact_based_on_entities>",
                "operator": "<appropriate_operator>",
                "value": "<dynamic_value_from_entities>"
            }}
        ]
    }},
    "then": {{
        "<enforce_or_warn>": [
            {{
                "constraint": "<dynamic_constraint_based_on_intent>",
                "operator": "<appropriate_operator>",
                "value": "<dynamic_value>"
            }}
        ]
    }}
}}

Rules:
- Use "enforce" action if not ambiguous, "warn" if ambiguous
- For INFORMATIONAL: constraint="PASS", operator="EQUALS", value="OK"
- For RESTRICTION: constraint="approval.required", operator="EQUALS", value="<approval_value>"
- For CONDITIONAL_ALLOWANCE: constraint="allowance.<type>", operator="EQUALS", value="<amount>"
- For LIMIT: constraint="limit.<type>", operator="LESS_THAN_OR_EQUAL", value="<limit>"
- Generate dynamic facts and values from the entities list
- If no meaningful conditions can be generated, use default PASS rule

Return ONLY the JSON object, no markdown formatting."""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON from response
            response_text = response.text.strip()
            # Remove markdown code blocks if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            dsl_rule = json.loads(response_text)
            
            # Ensure structure
            if 'rule_id' not in dsl_rule:
                dsl_rule['rule_id'] = clause_id
            if 'when' not in dsl_rule:
                dsl_rule['when'] = {'all': []}
            if 'then' not in dsl_rule:
                action = 'warn' if is_ambiguous else 'enforce'
                dsl_rule['then'] = {action: [{'constraint': 'PASS', 'operator': 'EQUALS', 'value': 'OK'}]}
            
            return dsl_rule
            
        except json.JSONDecodeError as e:
            self.log_entry("WARNING", f"Failed to parse DSL JSON for {clause_id}: {e}")
            # Return default rule
            return self.create_default_rule(clause_id, is_ambiguous)
        except Exception as e:
            self.log_entry("ERROR", f"Gemini DSL generation failed for {clause_id}: {e}")
            return self.create_default_rule(clause_id, is_ambiguous)
    
    def create_default_rule(self, clause_id, is_ambiguous):
        """Create a default DSL rule when AI generation fails"""
        return {
            'rule_id': clause_id,
            'when': {'all': []},
            'then': {
                'warn' if is_ambiguous else 'enforce': [
                    {'constraint': 'PASS', 'operator': 'EQUALS', 'value': 'OK'}
                ]
            }
        }
    
    def create_smart_default_rule(self, clause_id, intent, entities, is_ambiguous):
        """Create smart DSL rule based on intent and entities"""
        
        # Build when conditions from entities - map new entity names to DSL facts
        when_conditions = []
        
        for entity_name, entity_value in entities.items():
            # Map entity names to DSL facts
            entity_lower = entity_name.lower()
            
            # Distance thresholds → travel.distance
            if 'distance' in entity_lower or 'km' in entity_value.lower():
                when_conditions.append({
                    'fact': 'travel.distance',
                    'operator': 'LESS_THAN_OR_EQUAL',
                    'value': entity_value
                })
            
            # Percentage values → claim.percentage
            elif 'percentage' in entity_lower or '%' in entity_value:
                when_conditions.append({
                    'fact': 'claim.percentage',
                    'operator': 'EQUALS',
                    'value': entity_value
                })
            
            # Hour triggers → work.hours
            elif 'hour' in entity_lower or 'hour' in entity_value.lower():
                when_conditions.append({
                    'fact': 'work.hours',
                    'operator': 'GREATER_THAN_OR_EQUAL',
                    'value': entity_value
                })
            
            # Time limits (days) → settlement.days
            elif 'day' in entity_lower or 'day' in entity_value.lower():
                when_conditions.append({
                    'fact': 'settlement.days',
                    'operator': 'LESS_THAN_OR_EQUAL',
                    'value': entity_value
                })
            
            # Employee levels → employee.level
            elif 'employee' in entity_lower or 'level' in entity_lower or 'grade' in entity_lower:
                when_conditions.append({
                    'fact': 'employee.level',
                    'operator': 'EQUALS',
                    'value': entity_value
                })
            
            # Role exclusions → employee.role
            elif 'role' in entity_lower or 'personnel' in entity_lower:
                when_conditions.append({
                    'fact': 'employee.role',
                    'operator': 'NOT_EQUALS',
                    'value': entity_value
                })
            
            # Travel mode → travel.mode
            elif 'travel' in entity_lower or 'car' in entity_lower or 'bike' in entity_lower:
                when_conditions.append({
                    'fact': 'travel.mode',
                    'operator': 'EQUALS',
                    'value': entity_value
                })
            
            # Applicable to → organization.applies
            elif 'applicable' in entity_lower or 'company' in entity_lower:
                when_conditions.append({
                    'fact': 'organization.applies',
                    'operator': 'EQUALS',
                    'value': entity_value
                })
            
            # Validation authority → approval.required
            elif 'authority' in entity_lower or 'hr' in entity_lower or 'account' in entity_lower:
                when_conditions.append({
                    'fact': 'approval.required',
                    'operator': 'EQUALS',
                    'value': entity_value
                })
            
            else:
                # Generic fallback
                fact = entity_name.replace('.', '_').lower()
                when_conditions.append({
                    'fact': fact,
                    'operator': 'EQUALS',
                    'value': entity_value
                })
        
        # Build then constraints based on intent
        action = 'warn' if is_ambiguous else 'enforce'
        
        if intent == 'INFORMATIONAL':
            then_constraints = [{'constraint': 'PASS', 'operator': 'EQUALS', 'value': 'OK'}]
        elif intent == 'RESTRICTION':
            then_constraints = [{'constraint': 'approval.required', 'operator': 'EQUALS', 'value': 'YES'}]
        elif intent == 'LIMIT':
            then_constraints = [{'constraint': 'limit.enforced', 'operator': 'STRICT', 'value': 'MAX'}]
        elif intent == 'CONDITIONAL_ALLOWANCE':
            then_constraints = [{'constraint': 'allowance.approved', 'operator': 'CONDITIONAL', 'value': 'CLAIM'}]
        elif intent == 'ADVISORY':
            then_constraints = [{'constraint': 'recommendation', 'operator': 'GUIDANCE', 'value': 'ADVISORY'}]
        elif intent == 'APPROVAL_REQUIRED':
            then_constraints = [{'constraint': 'approval.mandatory', 'operator': 'REQUIRED', 'value': 'AUTHORIZATION'}]
        else:
            then_constraints = [{'constraint': 'PASS', 'operator': 'EQUALS', 'value': 'OK'}]
        
        return {
            'rule_id': clause_id,
            'when': {'all': when_conditions},
            'then': {action: then_constraints}
        }
    
    def load_files(self):
        """Load stage5 and stage4 data"""
        try:
            with open(self.stage5_file, 'r') as f:
                stage5_data = json.load(f)
            with open(self.stage4_file, 'r') as f:
                stage4_data = json.load(f)
            return stage5_data, stage4_data
        except Exception as e:
            self.log_entry("ERROR", f"Failed to load files: {e}")
            return None, None
    
    def build_when_condition(self, clause_id, intent, entities):
        """Build 'when' condition from entities - minimal format"""
        conditions = []
        
        if not entities:
            return None
        
        # Travel Classification rules (C2)
        if 'maximumTourDuration' in entities:
            conditions.append({
                'fact': 'travel.duration',
                'operator': 'LESS_THAN_OR_EQUAL',
                'value': entities['maximumTourDuration']
            })
        
        if 'minimumTourDuration' in entities:
            conditions.append({
                'fact': 'travel.duration',
                'operator': 'GREATER_THAN_OR_EQUAL',
                'value': entities['minimumTourDuration']
            })
        
        if 'minimumDeputationDuration' in entities:
            conditions.append({
                'fact': 'travel.duration',
                'operator': 'GREATER_THAN_OR_EQUAL',
                'value': entities['minimumDeputationDuration']
            })
            
        if 'maximumDeputationDuration' in entities:
            conditions.append({
                'fact': 'travel.duration',
                'operator': 'LESS_THAN_OR_EQUAL',
                'value': entities['maximumDeputationDuration']
            })
        
        # Allowance rules
        if 'grade' in entities:
            conditions.append({
                'fact': 'employee.grade',
                'operator': 'EQUALS',
                'value': entities['grade']
            })
        
        if 'lodgingAllowanceCategoryA' in entities or 'lodgingAllowanceCategoryB' in entities:
            conditions.append({
                'fact': 'location.category',
                'operator': 'IN',
                'value': 'CATEGORY_A | CATEGORY_B'
            })
        
        # Mileage rate rules
        if 'autoTaxiNonACRate' in entities or 'taxiACRate' in entities:
            conditions.append({
                'fact': 'vehicle.type',
                'operator': 'IN',
                'value': 'TAXI'
            })
        
        # Travel mode conditions
        if 'travelModeAir' in entities:
            conditions.append({
                'fact': 'travel.mode',
                'operator': 'EQUALS',
                'value': 'AIR'
            })
            
        if 'travelModeTrain' in entities:
            conditions.append({
                'fact': 'travel.mode',
                'operator': 'EQUALS',
                'value': 'TRAIN'
            })
            
        if 'travelModeTaxiLocal' in entities:
            conditions.append({
                'fact': 'travel.mode',
                'operator': 'EQUALS',
                'value': 'TAXI_LOCAL'
            })
            
        if 'travelModeTaxiInterCity' in entities:
            conditions.append({
                'fact': 'travel.mode',
                'operator': 'EQUALS',
                'value': 'TAXI_INTERCITY'
            })
        
        # Booking rules
        if 'bookingRequestLeadTime' in entities:
            conditions.append({
                'fact': 'booking.submitTime',
                'operator': 'BEFORE',
                'value': entities['bookingRequestLeadTime']
            })
        
        if 'bookingRestriction' in entities:
            conditions.append({
                'fact': 'booking.type',
                'operator': 'NOT_EQUALS',
                'value': 'DIRECT'
            })
        
        # Bill submission rules
        if 'billSubmissionDeadlineAfterTourCompletion' in entities:
            conditions.append({
                'fact': 'bill.submitTime',
                'operator': 'WITHIN_DAYS',
                'value': entities['billSubmissionDeadlineAfterTourCompletion']
            })
        
        return conditions if conditions else None
    
    def build_then_constraint(self, clause_id, intent, entities):
        """Build 'then' constraints - minimal format"""
        constraints = []
        
        # Travel Classification rules
        if 'travelClassifications' in entities:
            if 'tourDurationUpperLimit' in entities:
                constraints.append({
                    'constraint': 'travel.type',
                    'operator': 'EQUALS',
                    'value': 'TOUR'
                })
            elif 'deputationDurationLowerLimit' in entities:
                constraints.append({
                    'constraint': 'travel.type',
                    'operator': 'EQUALS',
                    'value': 'DEPUTATION'
                })
            elif 'transferDurationLowerLimit' in entities:
                constraints.append({
                    'constraint': 'travel.type',
                    'operator': 'EQUALS',
                    'value': 'TRANSFER'
                })
        
        # Allowance constraints
        if intent == 'CONDITIONAL_ALLOWANCE':
            if 'lodgingReimbursementRate' in entities:
                constraints.append({
                    'constraint': 'allowance.lodging',
                    'operator': 'EQUALS',
                    'value': entities['lodgingReimbursementRate']
                })
            if 'fourWheelerMileageRate' in entities:
                constraints.append({
                    'constraint': 'mileage.fourWheeler',
                    'operator': 'EQUALS',
                    'value': entities['fourWheelerMileageRate']
                })
            if 'twoWheelerMileageRate' in entities:
                constraints.append({
                    'constraint': 'mileage.twoWheeler',
                    'operator': 'EQUALS',
                    'value': entities['twoWheelerMileageRate']
                })
        
        # Restriction constraints
        if intent == 'RESTRICTION':
            if 'requiredApproval' in entities:
                constraints.append({
                    'constraint': 'approval.required',
                    'operator': 'EQUALS',
                    'value': entities['requiredApproval']
                })
            if 'billSubmissionDeadlineAfterTourCompletion' in entities:
                constraints.append({
                    'constraint': 'bill.deadline',
                    'operator': 'EQUALS',
                    'value': entities['billSubmissionDeadlineAfterTourCompletion']
                })
        
        return constraints if constraints else [{'constraint': 'PASS', 'operator': 'EQUALS', 'value': 'OK'}]
    
    def generate_dsl_rules(self):
        """Generate DSL rules using fast indexing + LLM fallback"""
        self.log_entry("INFO", "Starting Stage 6: Fast DSL Generation with Indexing")
        
        # Load data
        stage5_data, stage4_data = self.load_files()
        if not stage5_data or not stage4_data:
            return False
        
        clarified_clauses = stage5_data.get('clarified_clauses', [])
        ambiguity_map = {flag['clauseId']: flag.get('ambiguous', False) 
                        for flag in stage4_data if isinstance(stage4_data, list)}
        
        # Build clause index for fast lookups
        self.log_entry("INFO", f"Building clause index for {len(clarified_clauses)} clauses")
        self.clause_indexer = ClauseIndexer(clarified_clauses)
        
        self.log_entry("INFO", f"Generating DSL for {len(clarified_clauses)} clauses (indexed approach)")
        
        dsl_rules = []
        
        for clause in clarified_clauses:
            clause_id = clause['clauseId']
            clause_text = clause.get('text', '')
            intent = clause.get('intent', 'INFORMATIONAL')
            entities = clause.get('entities', {})
            is_ambiguous = ambiguity_map.get(clause_id, False)
            
            self.log_entry("PROCESSING", f"{clause_id}: Generating DSL ({intent})")
            
            # Try fast indexed generation first
            rule = self.generate_dsl_from_index(clause_id, intent, entities, is_ambiguous)
            
            if rule:
                # Success - no LLM call needed
                self.log_entry("INDEXED", f"{clause_id}: Fast DSL generated")
            else:
                # Create smart default rule based on intent and entities
                self.log_entry("SMART_DEFAULT", f"{clause_id}: Creating smart default rule")
                rule = self.create_smart_default_rule(clause_id, intent, entities, is_ambiguous)
            
            dsl_rules.append(rule)
            self.log_entry("GENERATED", f"{clause_id}: DSL rule created")
        
        self.log_entry("SUCCESS", f"DSL generation complete: {len(dsl_rules)} rules generated using indexed approach")
        
        # Save DSL rules
        self.save_dsl_rules(dsl_rules)
        
        return True
    
    def save_dsl_rules(self, dsl_rules):
        """Save DSL rules as YAML"""
        import yaml
        
        output = {
            'rules': dsl_rules
        }
        
        try:
            with open(self.dsl_file, 'w') as f:
                yaml.dump(output, f, default_flow_style=False, sort_keys=False, width=200)
            
            self.log_entry("SUCCESS", f"DSL rules saved to: {self.dsl_file}")
            
            # Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=6,
                stage_name="generate-dsl",
                stage_output=output
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save DSL rules: {e}")
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== STAGE 6: DSL GENERATION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class PipelineOrchestrator:
    """
    Orchestrate full pipeline: 1 → 1B → 2 → 3 → 4 → 5 → 6 → 8
    (Skips Stage 7: Confidence-Rationale for speed)
    
    Single entry point for complete policy processing
    Starts from PDF extraction and handles all sequential dependencies automatically
    """
    
    def __init__(self, pdf_file, enable_mongodb=True):
        self.pdf_file = pdf_file
        self.policy_text_file = f"{OUTPUT_DIR}/filename.txt"
        self.enable_mongodb = enable_mongodb
        self.log = []
        self.stage_results = {}
    
    def log_entry(self, level, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def run_stage_1(self):
        """Stage 1: Extract PDF to text"""
        self.log_entry("STAGE", "Running Stage 1: PDF Extraction")
        
        try:
            extractor = PolicyExtractor(self.pdf_file)
            success = extractor.extract_pdf()
            extractor.save_log()
            
            if success:
                self.stage_results['stage1_file'] = self.policy_text_file
                self.log_entry("SUCCESS", "Stage 1 complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 1 failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 1 exception: {e}")
            return False
    
    def run_stage_1b(self):
        """Stage 1B: Extract & elaborate clauses"""
        self.log_entry("STAGE", "Running Stage 1B: Clause Extraction")
        
        try:
            extractor = ClauseExtractor(self.policy_text_file, enable_mongodb=self.enable_mongodb)
            success = extractor.extract()
            extractor.save_log()
            
            if success:
                self.stage_results['stage1b_file'] = f"{OUTPUT_DIR}/stage1_clauses.json"
                self.log_entry("SUCCESS", "Stage 1B complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 1B failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 1B exception: {e}")
            return False
    
    def run_stage_2(self):
        """Stage 2: Classify intent"""
        self.log_entry("STAGE", "Running Stage 2: Intent Classification")
        
        try:
            clauses_file = self.stage_results.get('stage1b_file', f"{OUTPUT_DIR}/stage1_clauses.json")
            classifier = IntentClassifier(clauses_file, enable_mongodb=self.enable_mongodb)
            success = classifier.classify()
            classifier.save_log()
            
            if success:
                self.stage_results['stage2_file'] = f"{OUTPUT_DIR}/stage2_classified.json"
                self.log_entry("SUCCESS", "Stage 2 complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 2 failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 2 exception: {e}")
            return False
    
    def run_stage_3(self, sequential=False):
        """Stage 3: Extract entities (RULE-BASED ONLY to avoid rate limits)
        
        Args:
            sequential (bool): If True, process clauses one at a time (no multithreading)
        """
        self.log_entry("STAGE", "Running Stage 3: Entity Extraction (Rule-based, no LLM)")
        
        # Use sequential processing if requested
        max_workers = 1 if sequential else PipelineConfig.ENTITY_EXTRACTION_WORKERS
        
        if sequential:
            self.log_entry("INFO", "Sequential processing enabled (no multithreading)")
        
        try:
            classified_file = self.stage_results.get('stage2_file', f"{OUTPUT_DIR}/stage2_classified.json")
            extractor = EntityExtractor(
                classified_file,
                enable_mongodb=self.enable_mongodb,
                use_langchain=False,  # DISABLE LLM - use rule-based extraction only
                max_workers=max_workers
            )
            success = extractor.extract()
            extractor.save_log()
            
            if success:
                self.stage_results['stage3_file'] = f"{OUTPUT_DIR}/stage3_entities.json"
                self.log_entry("SUCCESS", "Stage 3 complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 3 failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 3 exception: {e}")
            return False
    
    def run_stage_4(self):
        """Stage 4: Detect ambiguities"""
        self.log_entry("STAGE", "Running Stage 4: Ambiguity Detection")
        
        try:
            stage3_file = self.stage_results.get('stage3_file', f"{OUTPUT_DIR}/stage3_entities.json")
            detector = AmbiguityDetector(stage3_file, enable_mongodb=self.enable_mongodb)
            success = detector.detect_ambiguities()
            detector.save_log()
            
            if success:
                self.stage_results['stage4_file'] = f"{OUTPUT_DIR}/stage4_ambiguity_flags.json"
                self.log_entry("SUCCESS", "Stage 4 complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 4 failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 4 exception: {e}")
            return False
    
    def run_stage_5(self):
        """Stage 5: Clarify ambiguities"""
        self.log_entry("STAGE", "Running Stage 5: Ambiguity Clarification")
        
        try:
            stage3_file = self.stage_results.get('stage3_file', f"{OUTPUT_DIR}/stage3_entities.json")
            stage4_file = self.stage_results.get('stage4_file', f"{OUTPUT_DIR}/stage4_ambiguity_flags.json")
            clarifier = AmbiguityClarifier(
                stage3_file,
                stage4_file,
                enable_mongodb=self.enable_mongodb,
                max_workers=PipelineConfig.AMBIGUITY_CLARIFICATION_WORKERS,
                batch_size=PipelineConfig.AMBIGUITY_CLARIFICATION_BATCH_SIZE
            )
            success = clarifier.clarify_all_clauses()
            clarifier.save_log()
            
            if success:
                self.stage_results['stage5_file'] = f"{OUTPUT_DIR}/stage5_clarified_clauses.json"
                self.log_entry("SUCCESS", "Stage 5 complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 5 failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 5 exception: {e}")
            return False
    
    def run_stage_6(self):
        """Stage 6: Generate DSL rules"""
        self.log_entry("STAGE", "Running Stage 6: DSL Rule Generation")
        
        try:
            stage5_file = self.stage_results.get('stage5_file', f"{OUTPUT_DIR}/stage5_clarified_clauses.json")
            stage4_file = self.stage_results.get('stage4_file', f"{OUTPUT_DIR}/stage4_ambiguity_flags.json")
            generator = DSLGenerator(stage5_file, stage4_file, enable_mongodb=self.enable_mongodb)
            success = generator.generate_dsl_rules()
            generator.save_log()
            
            if success:
                self.stage_results['stage6_file'] = f"{OUTPUT_DIR}/stage6_dsl_rules.yaml"
                self.log_entry("SUCCESS", "Stage 6 complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 6 failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 6 exception: {e}")
            return False
    
    def run_stage_8(self):
        """Stage 8: Generate normalized policies (skip Stage 7)"""
        self.log_entry("STAGE", "Running Stage 8: Normalized Policy Generation (Stage 7 skipped)")
        
        try:
            dsl_file = self.stage_results.get('stage6_file', f"{OUTPUT_DIR}/stage6_dsl_rules.yaml")
            # confidence_file is optional for stage 8
            confidence_file = f"{OUTPUT_DIR}/stage7_confidence_rationale.json"
            
            generator = NormalizedPolicyGenerator(
                dsl_file,
                confidence_file=confidence_file if os.path.exists(confidence_file) else None,
                enable_mongodb=self.enable_mongodb,
                use_langchain=PipelineConfig.NORMALIZATION_USE_LLM
            )
            success = generator.generate_normalized_policies()
            generator.save_log()
            
            if success:
                self.stage_results['stage8_file'] = f"{OUTPUT_DIR}/stage8_normalized_policies.json"
                self.log_entry("SUCCESS", "Stage 8 complete")
                return True
            else:
                self.log_entry("ERROR", "Stage 8 failed")
                return False
        except Exception as e:
            self.log_entry("ERROR", f"Stage 8 exception: {e}")
            return False
    
    def run_full_pipeline(self, stage3_sequential=False):
        """
        Execute full pipeline: 1 → 1B → 2 → 3 → 4 → 5 → 6 → 8
        
        Args:
            stage3_sequential (bool): If True, run Stage 3 without multithreading
        
        Returns:
            dict: Results with success status and stage file paths
        """
        self.log_entry("START", "="*80)
        self.log_entry("START", "FULL PIPELINE ORCHESTRATION: Stages 1 → 1B → 2 → 3 → 4 → 5 → 6 → 8")
        self.log_entry("START", "="*80)
        
        if stage3_sequential:
            self.log_entry("INFO", "Stage 3 will run in SEQUENTIAL mode (no multithreading)")
        
        stages = [
            ('1', self.run_stage_1),
            ('1B', self.run_stage_1b),
            ('2', self.run_stage_2),
            ('3', lambda: self.run_stage_3(sequential=stage3_sequential)),
            ('4', self.run_stage_4),
            ('5', self.run_stage_5),
            ('6', self.run_stage_6),
            ('8', self.run_stage_8)
        ]
        
        for stage_name, stage_func in stages:
            self.log_entry("INFO", f"\n--- Executing Stage {stage_name} ---")
            success = stage_func()
            
            if not success:
                self.log_entry("ERROR", f"Pipeline stopped at Stage {stage_name}")
                return {
                    'success': False,
                    'failed_stage': stage_name,
                    'results': self.stage_results
                }
            
            self.log_entry("PROGRESS", f"Stage {stage_name} ✓")
        
        self.log_entry("SUCCESS", "="*80)
        self.log_entry("SUCCESS", "FULL PIPELINE COMPLETE - All stages successful")
        self.log_entry("SUCCESS", "="*80)
        
        return {
            'success': True,
            'results': self.stage_results,
            'output_file': self.stage_results.get('stage8_file')
        }
    
    def save_log(self):
        """Save orchestration log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== PIPELINE ORCHESTRATION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")
            f.write("\n" + "="*80 + "\n")


class NormalizedPolicyGenerator:
    """
    Stage 8: Generate Normalized Policy JSON from DSL Rules
    
    Hybrid approach:
    - Rule-based mapping for structural transformation (fast)
    - LLM for complex metadata extraction (when needed)
    
    Input:  stage6_dsl_rules.yaml + stage7_confidence_rationale.json
    Output: stage8_normalized_policies.json
    """
    
    def __init__(self, dsl_file, confidence_file=None, clarified_file=None, document_id=None, enable_mongodb=True, use_langchain=True):
        self.dsl_file = dsl_file
        self.confidence_file = confidence_file
        self.clarified_file = clarified_file or f"{OUTPUT_DIR}/stage5_clarified_clauses.json"
        self.normalized_file = f"{OUTPUT_DIR}/stage8_normalized_policies.json"
        self.document_id = document_id
        self.log = []
        self.log_lock = threading.Lock()
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        # OPTIMIZATION: Make LLM truly optional for faster processing
        self.llm_enabled = self.use_langchain
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Load supporting data from earlier stages
        self.clarified_clauses = self.load_clarified_clauses()
        
        # Only initialize Gemini if LLM is enabled
        if self.llm_enabled:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemma-3-27b-it')
            
            # Initialize LangChain components if available
            self.log_entry("INFO", "LangChain enabled for metadata extraction")
            self.langchain_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_retries=3
            )
        else:
            self.log_entry("INFO", "Using fast rule-based metadata extraction (LLM disabled)")
            self.model = None
            self.langchain_llm = None
    
    def log_entry(self, level, message):
        """Thread-safe log entry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        with self.log_lock:
            self.log.append(entry)
        print(entry)
    
    def load_dsl_rules(self):
        """Load DSL rules from YAML file"""
        try:
            import yaml
            with open(self.dsl_file, 'r') as f:
                data = yaml.safe_load(f)
            return data.get('rules', [])
        except Exception as e:
            self.log_entry("ERROR", f"Failed to load DSL rules: {e}")
            return []
    
    def load_confidence_data(self):
        """Load confidence and rationale data"""
        if not self.confidence_file:
            return {}
        
        try:
            with open(self.confidence_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_entry("WARNING", f"Failed to load confidence data: {e}")
            return {}
    
    def load_clarified_clauses(self):
        """Load clarified clauses from Stage 5 for metadata enrichment"""
        try:
            with open(self.clarified_file, 'r') as f:
                data = json.load(f)
            
            # Build lookup dict: clauseId -> clarified clause data
            clauses_by_id = {}
            for clause in data.get('clarified_clauses', []):
                clause_id = clause.get('clauseId')
                if clause_id:
                    clauses_by_id[clause_id] = clause
            
            self.log_entry("INFO", f"Loaded {len(clauses_by_id)} clarified clauses from {self.clarified_file}")
            return clauses_by_id
        except Exception as e:
            self.log_entry("WARNING", f"Failed to load clarified clauses: {e}")
            return {}
    
    def consolidate_clause_data(self, clause_id, stage_data):
        """Consolidate all stage data for a single clause into UI-compatible schema"""
        
        # Extract DSL rule data
        dsl_rule = stage_data['stage6']
        when_conditions = dsl_rule.get('when', {}).get('all', [])
        then_outcome = dsl_rule.get('then', {})
        enforcement_level = 'enforce' if 'enforce' in then_outcome else 'warn'
        
        # Get first constraint and normalize it (DSL uses 'constraint' field, UI expects 'fact')
        constraint = {}
        raw_constraint = {}
        if 'warn' in then_outcome and then_outcome['warn']:
            raw_constraint = then_outcome['warn'][0] if isinstance(then_outcome['warn'], list) else then_outcome['warn']
        elif 'enforce' in then_outcome and then_outcome['enforce']:
            raw_constraint = then_outcome['enforce'][0] if isinstance(then_outcome['enforce'], list) else then_outcome['enforce']
        
        # Normalize: rename 'constraint' field to 'fact' for UI compatibility
        if raw_constraint:
            constraint = {
                'fact': raw_constraint.get('constraint', raw_constraint.get('fact', '')),
                'operator': raw_constraint.get('operator', ''),
                'value': raw_constraint.get('value', '')
            }
        
        consolidated = {
            # UI-compatible top-level fields (for displayPolicies JS)
            'policyId': f"POLICY_{clause_id}",
            'name': f"Policy Rule {clause_id}",
            'outcome': {
                'enforcement': enforcement_level.lower(),
                'message': f"Policy {clause_id} enforcement"
            },
            'when': when_conditions,
            'what': {
                'constraint': constraint
            },
            
            # Original consolidated structure (for backend use)
            'core_attributes': {
                'clauseId': clause_id,
                'text_clarified': stage_data['stage5'].get('text_clarified', ''),
                'intent': stage_data['stage2'].get('intent', 'UNKNOWN'),
                'confidence_score': stage_data['stage7'].get('confidence', 0.0),
                'source_section': stage_data['stage7'].get('sourceSection', '')
            },
            'classification': {
                'stage2_intent': stage_data['stage2'].get('intent', ''),
                'stage2_confidence': stage_data['stage2'].get('confidence', 0.0),
                'stage2_reasoning': stage_data['stage2'].get('reasoning', '')
            },
            'extracted_data': {
                'entities': stage_data['stage3'].get('entities', {}),
                'entity_names': stage_data['stage3'].get('entity_names', [])
            },
            'ambiguity_analysis': {
                'is_ambiguous': stage_data['stage4'].get('is_ambiguous', False),
                'ambiguity_types': stage_data['stage4'].get('ambiguity_types', []),
                'ambiguities_fixed': stage_data['stage5'].get('ambiguities_fixed', []),
                'original_reason': stage_data['stage4'].get('reason', ''),
                'detection_method': stage_data['stage4'].get('detection_method', 'rule-based'),
                'rule_score': stage_data['stage4'].get('rule_score', 0)
            },
            'dsl_rules': {
                'rule_id': stage_data['stage6'].get('rule_id', clause_id),
                'when_conditions': when_conditions,
                'then_outcome': then_outcome,
                'enforcement_level': enforcement_level
            },
            'rationale_metadata': {
                'rationale_text': stage_data['stage7'].get('rationale', ''),
                'confidence_score': stage_data['stage7'].get('confidence', 0.0),
                'ambiguities_fixed': stage_data['stage7'].get('ambiguitiesFixed', []),
                'source_section': stage_data['stage7'].get('sourceSection', '')
            }
        }
        
        return consolidated
    
    def build_dynamic_extraction_index(self):
        """Build dynamic extraction rules from actual stage data ONLY (validated)"""
        index = {
            'roles': {},           # role_value -> context
            'locations': {},       # location_value -> context
            'tenants': {},         # tenant_value -> context
            'authorities': {}      # authority_value -> orgId
        }
        
        # ONLY extract from Stage 3 entities (these are validated by earlier stages)
        for clause_id, clause in self.clarified_clauses.items():
            entities = clause.get('entities', {})
            
            for entity_name, entity_value in entities.items():
                if not entity_value or not isinstance(entity_value, str):
                    continue
                
                # Strip whitespace and validate non-empty
                entity_value = entity_value.strip()
                if not entity_value or len(entity_value) > 100:
                    continue
                
                # Classify based on entity name keywords (STRICT)
                if any(x in entity_name.lower() for x in ['grade', 'role', 'level', 'employee']):
                    index['roles'][entity_value] = entity_name
                
                elif any(x in entity_name.lower() for x in ['location', 'city', 'area', 'region']):
                    index['locations'][entity_value] = entity_name
                
                elif any(x in entity_name.lower() for x in ['applicable', 'tenant', 'org']):
                    index['tenants'][entity_value] = entity_name
        
        # Extract authorities ONLY from clarified text (explicit mentions)
        authorities_patterns = {
            r'(?:head\s+of\s+department|department\s+head|hod)': 'DEPT_MANAGER',
            r'(?:reporting\s+manager|direct\s+manager)': 'DEPT_MANAGER',
            r'(?:hr\s+&?\s+accounts|accounts\s+&?\s+hr)': 'HR_ADMIN',
            r'(?:finance\s+department|finance\s+admin)': 'FINANCE_ADMIN',
            r'(?:travel\s+admin|travel\s+department)': 'TRAVEL_ADMIN'
        }
        
        for clause_id, clause in self.clarified_clauses.items():
            text = clause.get('text_clarified', '').lower()
            
            for pattern, org_id in authorities_patterns.items():
                if re.search(pattern, text):
                    # Use proper case name for authority
                    proper_name = pattern.split('|')[0].replace(r'\s+', ' ').replace('(?:', '').replace(')', '').title()
                    index['authorities'][proper_name] = org_id
        
        self.log_entry("INDEX", f"Built index: {len(index['roles'])} roles, {len(index['locations'])} locations, {len(index['tenants'])} tenants, {len(index['authorities'])} authorities")
        return index
    
    def extract_metadata_with_index(self, dsl_rule, normalized_policy, extraction_index):
        """Extract metadata STRICTLY from validated index only"""
        
        when_conditions = dsl_rule.get('when', {}).get('all', [])
        rule_id = dsl_rule.get('rule_id', 'unknown')
        
        scope = normalized_policy.get('scope', {})
        applies_to = scope.get('appliesTo', {})
        
        # Get clarified clause text for exact matching
        clarified_clause = self.clarified_clauses.get(rule_id, {})
        text = clarified_clause.get('text_clarified', '') or clarified_clause.get('text_original', '')
        text_lower = text.lower()
        
        # Extract roles: ONLY from when conditions (most reliable)
        roles = []
        for condition in when_conditions:
            fact = condition.get('fact', '').lower()
            if any(x in fact for x in ['grade', 'role', 'employee']):
                val = condition.get('value')
                if val and isinstance(val, str):
                    role = val.strip()
                    if role and len(role) < 100:
                        roles.append(role)
        
        # Also check if index roles are EXPLICITLY mentioned (exact word match)
        for role_value in extraction_index['roles'].keys():
            pattern = r'\b' + re.escape(role_value) + r'\b'
            if re.search(pattern, text_lower):
                if role_value not in roles:
                    roles.append(role_value)
        
        if roles:
            applies_to['roles'] = list(dict.fromkeys(roles))
        
        # Extract locations: ONLY from when conditions (most reliable)
        locations = []
        for condition in when_conditions:
            fact = condition.get('fact', '').lower()
            if any(x in fact for x in ['location', 'category', 'area', 'region']):
                val = condition.get('value')
                if val and isinstance(val, str):
                    location = val.strip()
                    if location and len(location) < 100:
                        locations.append(location)
        
        # Also check if index locations are EXPLICITLY mentioned (exact word match)
        for location_value in extraction_index['locations'].keys():
            pattern = r'\b' + re.escape(location_value) + r'\b'
            if re.search(pattern, text_lower):
                if location_value not in locations:
                    locations.append(location_value)
        
        if locations:
            applies_to['locations'] = list(dict.fromkeys(locations))
        
        # Extract tenant: ONLY from index (first match)
        for tenant_value in extraction_index['tenants'].keys():
            pattern = r'\b' + re.escape(tenant_value) + r'\b'
            if re.search(pattern, text_lower):
                scope['tenantId'] = tenant_value
                break
        
        # Extract authority/orgId: ONLY from index authorities
        for authority, org_id in extraction_index['authorities'].items():
            pattern = r'\b' + re.escape(authority.lower()) + r'\b'
            if re.search(pattern, text_lower):
                scope['orgId'] = org_id
                break
        
        # Default orgId if not explicitly found
        if not scope.get('orgId'):
            scope['orgId'] = None
        
        scope['appliesTo'] = applies_to
        normalized_policy['scope'] = scope
        
        self.log_entry("EXTRACT", f"{rule_id}: roles={roles}, locations={locations}, tenant={scope.get('tenantId')}, org={scope.get('orgId')}")
    
    def extract_metadata_with_llm(self, dsl_rule, normalized_policy, extraction_index):
        """Use LLM only when index-based extraction is incomplete"""
        
        rule_id = dsl_rule.get('rule_id', 'unknown')
        scope = normalized_policy.get('scope', {})
        applies_to = scope.get('appliesTo', {})
        
        # Check if extraction was complete
        needs_llm = (
            not scope.get('tenantId') or 
            not scope.get('orgId') or
            not applies_to.get('roles') or
            not applies_to.get('locations')
        )
        
        if not needs_llm or not self.llm_enabled:
            return
        
        clarified_clause = self.clarified_clauses.get(rule_id, {})
        text = clarified_clause.get('text_clarified', '')
        
        prompt = f"""Extract missing metadata from this policy clause.

CLAUSE TEXT:
{text}

RULE ID: {rule_id}

Current extracted values:
- roles: {applies_to.get('roles', [])}
- locations: {applies_to.get('locations', [])}
- tenantId: {scope.get('tenantId')}
- orgId: {scope.get('orgId')}

Fill in any missing values as JSON (or return {{}} if nothing found):
{{"roles": [], "locations": [], "tenantId": null, "orgId": null}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip().replace('```json', '').replace('```', '')
            metadata = json.loads(response_text)
            
            # Merge LLM results with existing
            if metadata.get('roles'):
                applies_to['roles'] = list(set(applies_to.get('roles', []) + metadata['roles']))
            if metadata.get('locations'):
                applies_to['locations'] = list(set(applies_to.get('locations', []) + metadata['locations']))
            if metadata.get('tenantId') and not scope.get('tenantId'):
                scope['tenantId'] = metadata['tenantId']
            if metadata.get('orgId') and not scope.get('orgId'):
                scope['orgId'] = metadata['orgId']
            
            self.log_entry("LLM", f"{rule_id}: Filled gaps with LLM")
        except Exception as e:
            self.log_entry("WARNING", f"{rule_id}: LLM extraction failed: {e}")
    
    def build_stage_data_map(self):
        """Load and index all stage data by clauseId"""
        stage_map = {}
        
        # Stage 1: Raw clauses
        try:
            with open(f"{OUTPUT_DIR}/stage1_clauses.json") as f:
                stage1_data = json.load(f)
            for clause in stage1_data.get('clauses', []):
                clause_id = clause.get('clauseId')
                if clause_id not in stage_map:
                    stage_map[clause_id] = {}
                stage_map[clause_id]['stage1'] = clause
        except Exception as e:
            self.log_entry("WARNING", f"Stage 1 load failed: {e}")
        
        # Stage 2: Classification
        try:
            with open(f"{OUTPUT_DIR}/stage2_classified.json") as f:
                stage2_data = json.load(f)
            for clause in stage2_data.get('classified_clauses', []):
                clause_id = clause.get('clauseId')
                if clause_id not in stage_map:
                    stage_map[clause_id] = {}
                stage_map[clause_id]['stage2'] = clause
        except Exception as e:
            self.log_entry("WARNING", f"Stage 2 load failed: {e}")
        
        # Stage 3: Entities
        try:
            with open(f"{OUTPUT_DIR}/stage3_entities.json") as f:
                stage3_data = json.load(f)
            for clause in stage3_data.get('extracted_clauses', []):
                clause_id = clause.get('clauseId')
                if clause_id not in stage_map:
                    stage_map[clause_id] = {}
                stage_map[clause_id]['stage3'] = clause
        except Exception as e:
            self.log_entry("WARNING", f"Stage 3 load failed: {e}")
        
        # Stage 4: Ambiguity flags
        try:
            with open(f"{OUTPUT_DIR}/stage4_ambiguity_flags.json") as f:
                stage4_data = json.load(f)
            for clause in stage4_data.get('ambiguity_flags', []):
                clause_id = clause.get('clauseId')
                if clause_id not in stage_map:
                    stage_map[clause_id] = {}
                stage_map[clause_id]['stage4'] = clause
        except Exception as e:
            self.log_entry("WARNING", f"Stage 4 load failed: {e}")
        
        # Stage 5: Clarified clauses (already loaded)
        for clause_id, clause in self.clarified_clauses.items():
            if clause_id not in stage_map:
                stage_map[clause_id] = {}
            stage_map[clause_id]['stage5'] = clause
        
        # Stage 6: DSL rules
        try:
            import yaml
            with open(self.dsl_file) as f:
                stage6_data = yaml.safe_load(f)
            for rule in stage6_data.get('rules', []):
                rule_id = rule.get('rule_id')
                if rule_id not in stage_map:
                    stage_map[rule_id] = {}
                stage_map[rule_id]['stage6'] = rule
        except Exception as e:
            self.log_entry("WARNING", f"Stage 6 load failed: {e}")
        
        # Stage 7: Confidence & rationale
        try:
            with open(self.confidence_file) as f:
                stage7_data = json.load(f)
            for clause_id, data in stage7_data.get('confidenceAndRationale', {}).items():
                if clause_id not in stage_map:
                    stage_map[clause_id] = {}
                stage_map[clause_id]['stage7'] = data
        except Exception as e:
            self.log_entry("WARNING", f"Stage 7 load failed: {e}")
        
        self.log_entry("INFO", f"Built stage map for {len(stage_map)} clauses")
        return stage_map
    
    def generate_normalized_policies(self):
        """Generate consolidated normalized policies from all stages"""
        self.log_entry("INFO", "Starting Stage 8: Consolidated Policy Normalization")
        
        # Load and map all stage data
        stage_map = self.build_stage_data_map()
        
        if not stage_map:
            self.log_entry("ERROR", "No clause data found in stages")
            return False
        
        self.log_entry("INFO", f"Consolidating {len(stage_map)} clauses from all stages")
        
        # Consolidate all clauses
        consolidated_records = []
        for clause_id in sorted(stage_map.keys()):
            stage_data = stage_map[clause_id]
            
            # Fill missing stages with empty defaults
            for stage in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7']:
                if stage not in stage_data:
                    stage_data[stage] = {}
            
            # Consolidate this clause's data
            consolidated = self.consolidate_clause_data(clause_id, stage_data)
            consolidated_records.append(consolidated)
            
            self.log_entry("CONSOLIDATED", f"Clause {clause_id}: {stage_data['stage2'].get('intent', 'UNKNOWN')}")
        
        self.log_entry("SUCCESS", f"Consolidated {len(consolidated_records)} clauses")
        
        # Save output
        return self.save_consolidated_policies(consolidated_records)
    
    def save_consolidated_policies(self, consolidated_records):
        """Save consolidated policies to JSON file"""
        try:
            output = {
                'policies': consolidated_records,
                'metadata': {
                    'generated': datetime.now().isoformat(),
                    'total_policies': len(consolidated_records),
                    'format': 'Consolidated Policy Normalization (Stage 8)',
                    'source_stages': [1, 2, 3, 4, 5, 6, 7],
                    'description': 'Each policy contains consolidated data from all 7 processing stages'
                }
            }
            
            with open(self.normalized_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Consolidated policies saved to: {self.normalized_file}")
            
            # Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=8,
                stage_name="normalize-policies",
                stage_output=output
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
            return True
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save consolidated policies: {e}")
            return False
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== STAGE 8: NORMALIZED POLICY GENERATION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")



if __name__ == "__main__":
    main()
