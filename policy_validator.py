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

Usage:
  python policy_validator.py extract <policy.pdf>
  python policy_validator.py extract-clauses <filename.txt>
  python policy_validator.py classify-intents <stage1_clauses.json>
  python policy_validator.py extract-entities <stage2_classified.json>
  python policy_validator.py detect-ambiguities <stage3_entities.json>
  python policy_validator.py clarify-ambiguities <stage3_entities.json>
  python policy_validator.py generate-dsl <stage5_clarified_clauses.json>
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    DEFAULT_MAX_WORKERS = 7
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


class ClauseExtractor:
    """
    Step 1B: Extract & elaborate clauses from policy text.
    
    ENHANCED WITH LANGCHAIN:
    - Document chunking for large policies
    - Parallel processing of chunks
    - Structured output parsing with Pydantic
    - Automatic retry logic
    - Memory management for clause numbering
    
    Output structure: Simple clause_id → text mapping for Stage 2 intent classification.
    Each clause contains conditional statements, numerical values, enforcement levels, etc.
    """
    
    def __init__(self, policy_file, document_id=None, enable_mongodb=True, use_langchain=True):
        self.policy_file = policy_file
        self.clauses_file = f"{OUTPUT_DIR}/stage1_clauses.json"
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
            self.log_entry("INFO", "LangChain enabled for enhanced clause extraction")
            self.langchain_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_retries=3
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            if not LANGCHAIN_AVAILABLE:
                self.log_entry("WARNING", "LangChain not available, using standard extraction")
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log.append(entry)
        print(entry)
    
    def read_policy_file(self):
        """Read policy file"""
        try:
            with open(self.policy_file, 'r') as f:
                return f.read()
        except Exception as e:
            self.log_entry("ERROR", f"Failed to read policy file: {e}")
            return None
    
    def extract_clauses_with_langchain(self, content):
        """
        LANGCHAIN-ENHANCED: Extract clauses using document chunking and parallel processing.
        
        Benefits:
        - Handles large documents (10k+ words)
        - Parallel chunk processing (30-40% faster)
        - Structured output validation
        - Automatic retry on failures
        
        Returns:
            list: Array of clause dicts with clause_id → text mapping
        """
        self.log_entry("LANGCHAIN", "Starting LangChain-enhanced clause extraction")
        
        # Step 1: Split document into chunks
        chunks = self.text_splitter.split_text(content)
        self.log_entry("CHUNKING", f"Split document into {len(chunks)} chunks")
        
        # Step 2: Create prompt template with output parser
        parser = PydanticOutputParser(pydantic_object=ClausesOutput)
        
        prompt_template = PromptTemplate(
            input_variables=["content", "start_clause_num"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""You are a POLICY CLAUSE EXTRACTION ENGINE. Extract ATOMIC, ELABORATED CLAUSES from this policy chunk.

CRITICAL REQUIREMENTS:
1. Each clause must be SELF-CONTAINED with elaborated details
2. Each clause text should be 4-5 lines MAXIMUM
3. MUST include:
   - Conditional statements (IF condition THEN consequence)
   - Numerical values (amounts, percentages, durations, thresholds)
   - Enforcement level (MUST/SHOULD/MAY)
   - Applicable roles or conditions
4. Start clause numbering from C{start_clause_num}
5. NO interpretation - extract exactly what the policy states
6. Preserve original section references

POLICY CHUNK:
{content}

{format_instructions}

OUTPUT ONLY valid JSON matching the schema. No explanations."""
        )
        
        # Step 3: Create chain
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        
        # Step 4: Process chunks sequentially (with clause counter)
        all_clauses = []
        clause_counter = 1
        
        for i, chunk in enumerate(chunks, 1):
            self.log_entry("PROCESSING", f"Chunk {i}/{len(chunks)} (starting at C{clause_counter})")
            
            try:
                # Run chain
                result = chain.run(content=chunk, start_clause_num=clause_counter)
                
                # Parse output
                parsed_output = parser.parse(result)
                chunk_clauses = [clause.dict() for clause in parsed_output.clauses]
                
                self.log_entry("SUCCESS", f"Extracted {len(chunk_clauses)} clauses from chunk {i}")
                all_clauses.extend(chunk_clauses)
                clause_counter += len(chunk_clauses)
                
            except Exception as e:
                self.log_entry("ERROR", f"Chunk {i} failed: {e}")
                # Fallback to standard extraction for this chunk
                self.log_entry("FALLBACK", f"Using standard extraction for chunk {i}")
                fallback_clauses = self._fallback_extraction(chunk, clause_counter)
                if fallback_clauses:
                    all_clauses.extend(fallback_clauses)
                    clause_counter += len(fallback_clauses)
        
        self.log_entry("COMPLETE", f"Total clauses extracted: {len(all_clauses)}")
        return all_clauses
    
    def _fallback_extraction(self, content, start_num):
        """Fallback to standard Gemini extraction if LangChain fails"""
        try:
            prompt = f"""Extract policy clauses from this text. Start numbering from C{start_num}.
Output JSON array: [{{"clause_id": "C{start_num}", "text": "..."}}]

CONTENT:
{content[:2000]}

OUTPUT ONLY JSON array."""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean markdown
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            clauses = json.loads(response_text.strip())
            return clauses if isinstance(clauses, list) else []
        except:
            return []
    
    def extract_clauses_with_gemini(self, content):
        """
        Use Gemini API to extract individual clauses from raw policy text.
        Each clause should contain:
        - clause_id (C1, C2, C3, etc.)
        - Elaborated text (4-5 lines max) with:
          * Conditional statements (IF/THEN logic)
          * Numerical values (amounts, percentages, thresholds)
          * Clear enforcement intent
          * Referenced source section
        
        Returns:
            list: Array of clause dicts with clause_id → text mapping (linked list style)
        """
        
        self.log_entry("GEMINI", "Extracting and elaborating clauses from policy")
        
        prompt = f"""
You are a POLICY CLAUSE EXTRACTION ENGINE. Your task is to break down a policy document into 
ATOMIC, ELABORATED CLAUSES that can be linked together (linked-list structure where each 
clause_id points to detailed text).

CRITICAL REQUIREMENTS:
====================
1. Each clause must be SELF-CONTAINED but reference other clauses if needed
2. Each clause text should be 4-5 lines MAXIMUM with elaborated details
3. MUST include:
   - Conditional statements (IF condition THEN consequence)
   - Numerical values (amounts, percentages, durations, thresholds)
   - Enforcement level (MUST/SHOULD/MAY)
   - Applicable roles or conditions
4. Clauses form a LINKED-LIST (later clauses can reference earlier ones)
5. NO interpretation - extract exactly what the policy states
6. Preserve original clause numbering/section references from source

POLICY DOCUMENT:
================================================================================
{content}
================================================================================

OUTPUT FORMAT (JSON array only - no other text):
[
  {{
    "clause_id": "C1",
    "text": "4-5 line elaborated text with all details. Include IF/THEN, amounts, thresholds, 
             roles, enforcement level. If references previous clause, cite it as (refs: C0). 
             Source: Section X, Paragraph Y."
  }},
  {{
    "clause_id": "C2",
    "text": "Next clause elaborated with conditional logic, numerical values, and enforcement intent..."
  }}
]

ELABORATION RULES:
==================
- Expand vague terms with inferred context
- Make implicit conditions explicit (e.g., "after 4:30 PM" → "IF time >= 16:30 THEN...")
- Convert amounts to explicit values (e.g., "$500 per diem" → "Daily allowance: 500 USD")
- State role/grade applicability explicitly
- Flag cross-references between clauses with (refs: C#)
- Mark enforcement level: MUST (mandatory), SHOULD (recommended), MAY (optional)

LINKED-LIST BEHAVIOR:
=====================
- First clause (C1) stands alone
- Subsequent clauses can reference earlier clauses like: (refs: C1, C3)
- Build a logical chain of policy enforcement
- Ensure each clause can be evaluated independently AND in sequence

OUTPUT ONLY the JSON array. No explanations, no markdown, no code blocks.
"""
        
        try:
            self.log_entry("GEMINI_REQUEST", "Sending policy content for clause extraction")
            response = self.model.generate_content(prompt)
            
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
            
            response_text = response_text.strip()
            
            # Parse JSON response
            try:
                clauses = json.loads(response_text)
                
                if not isinstance(clauses, list):
                    self.log_entry("ERROR", "Gemini response is not a JSON array")
                    return None
                
                self.log_entry("SUCCESS", f"Extracted {len(clauses)} clauses from policy")
                return clauses
                
            except json.JSONDecodeError as e:
                self.log_entry("ERROR", f"Failed to parse Gemini JSON response: {e}")
                self.log_entry("DEBUG", f"Response preview: {response_text[:200]}")
                return None
        
        except Exception as e:
            self.log_entry("ERROR", f"Gemini API failed: {e}")
            return None
    
    def validate_clauses(self, clauses):
        """Validate extracted clauses structure"""
        if not clauses:
            return False
        
        for i, clause in enumerate(clauses):
            # Check required fields
            if "clause_id" not in clause or "text" not in clause:
                self.log_entry("ERROR", f"Clause {i} missing required fields (clause_id, text)")
                return False
            
            # Check text length (4-5 lines recommendation, allow up to 10)
            text_lines = len(clause["text"].split('\n'))
            if text_lines > 15:
                self.log_entry("WARNING", f"Clause {clause['clause_id']} exceeds recommended length ({text_lines} lines)")
            
            # Validate clause_id format
            if not re.match(r'^C\d+$', clause['clause_id']):
                self.log_entry("ERROR", f"Invalid clause_id format: {clause['clause_id']} (should be C1, C2, etc.)")
                return False
        
        return True
    
    def build_clause_map(self, clauses):
        """
        Build simple clause_id → text mapping (no linked-list pointers).
        Each clause_id is directly linked to its elaborated text.
        Used as input for Stage 2 intent classification.
        """
        clause_map = []
        for clause in clauses:
            clause_map.append({
                "clauseId": clause["clause_id"],
                "text": clause["text"]
            })
        return clause_map
    
    def format_clauses_output(self, clauses):
        """
        Format clause output with metadata header for Stage 2 (intent classification).
        Simple structure: clauseId → text mapping.
        
        Args:
            clauses (list): List of clause dicts with clauseId and text
            
        Returns:
            dict: Formatted output with metadata and clauses array
        """
        output = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "source": self.policy_file,
                "format": "Clause → Text Mapping (Stage 1B → Stage 2 Intent Classification)",
                "total_clauses": len(clauses),
                "stage": "1B",
                "next_stage": "2 - Intent Classification"
            },
            "clauses": clauses
        }
        return output
    
    def extract(self):
        """Main extraction workflow with LangChain enhancement"""
        self.log_entry("START", "Step 1B: Clause Extraction with Elaboration")
        
        # Step 1: Read policy file
        self.log_entry("STEP", "Reading policy file")
        content = self.read_policy_file()
        if not content:
            return False
        
        self.log_entry("SUCCESS", f"Policy file read: {len(content)} characters")
        
        # Step 2: Extract and elaborate clauses (LangChain or standard)
        if self.use_langchain:
            self.log_entry("STEP", "Extracting clauses with LangChain (enhanced mode)")
            clauses = self.extract_clauses_with_langchain(content)
        else:
            self.log_entry("STEP", "Extracting clauses with standard Gemini")
            clauses = self.extract_clauses_with_gemini(content)
        
        if not clauses:
            return False
        
        # Step 3: Validate clauses
        self.log_entry("STEP", "Validating clause structure")
        if not self.validate_clauses(clauses):
            return False
        
        # Step 4: Build clause_id → text mapping (simple link, no pointers)
        self.log_entry("STEP", "Building clause_id → text mapping")
        clause_map = self.build_clause_map(clauses)
        
        # Step 5: Format output
        self.log_entry("STEP", "Formatting output")
        formatted_output = self.format_clauses_output(clause_map)
        
        # Step 6: Save to file
        try:
            with open(self.clauses_file, 'w') as f:
                json.dump(formatted_output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Clauses saved to: {self.clauses_file}")
            self.log_entry("STATS", f"Total clauses: {len(clause_map)}")
            
            # Step 7: Store to MongoDB
            db_result = self.storage.store_stage(
                stage_number=1,
                stage_name="extract-clauses",
                stage_output=formatted_output,
                filename=self.policy_file.split('/')[-1]
            )
            
            if db_result['success']:
                self.log_entry("MONGODB", f"Stage stored with ID: {db_result['stage_id']}")
            else:
                self.log_entry("WARNING", f"MongoDB storage failed: {db_result['error']}")
            
            return True
            
        except Exception as e:
            self.log_entry("ERROR", f"Failed to save clauses file: {e}")
            return False
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== CLAUSE EXTRACTION LOG ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")


class IntentClassifier:
    """
    Step 2: Classify clause intents using Gemini API.
    
    For each clause text from Stage 1B, determines the intent type and confidence.
    Intent types: RESTRICTION, LIMIT, CONDITIONAL_ALLOWANCE, EXCEPTION,
                  APPROVAL_REQUIRED, ADVISORY, INFORMATIONAL
    """
    
    def __init__(self, clauses_file, document_id=None, enable_mongodb=True, use_langchain=True):
        self.clauses_file = clauses_file
        self.classified_file = f"{OUTPUT_DIR}/stage2_classified.json"
        self.document_id = document_id
        self.log = []
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        
        # Allowed intents (hardcoded taxonomy)
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
                self.log_entry("WARNING", "LangChain not available, using standard classification")
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
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
    
    def classify_intent_with_gemini(self, clause_id, clause_text):
        """
        Use Gemini API to classify a single clause's intent.
        
        Args:
            clause_id (str): Clause identifier (C1, C2, etc.)
            clause_text (str): Elaborated clause text
            
        Returns:
            dict: { "intent": "...", "confidence": 0.0-1.0, "reasoning": "..." }
        """
        
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
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
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
                    result["intent"] = "INFORMATIONAL"  # Default fallback
                
                # Validate intent is in allowed list
                intent = result.get("intent", "INFORMATIONAL").upper()
                if intent not in self.allowed_intents:
                    self.log_entry("WARNING", f"{clause_id}: Invalid intent '{intent}', using INFORMATIONAL")
                    result["intent"] = "INFORMATIONAL"
                else:
                    result["intent"] = intent
                
                # Validate confidence
                if "confidence" not in result:
                    result["confidence"] = 0.75  # Default confidence
                else:
                    try:
                        conf = float(result["confidence"])
                        result["confidence"] = max(0.0, min(1.0, conf))  # Clamp 0-1
                    except (ValueError, TypeError):
                        result["confidence"] = 0.75
                
                return result
            
            except json.JSONDecodeError as e:
                self.log_entry("ERROR", f"{clause_id}: Failed to parse Gemini response: {e}")
                return {
                    "intent": "INFORMATIONAL",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse response"
                }
        
        except Exception as e:
            self.log_entry("ERROR", f"{clause_id}: Gemini API call failed: {e}")
            return {
                "intent": "INFORMATIONAL",
                "confidence": 0.5,
                "reasoning": f"API error: {str(e)[:50]}"
            }
    
    def classify(self):
        """Main classification workflow"""
        self.log_entry("START", "Step 2: Intent Classification")
        
        # Step 1: Read clauses from Stage 1B
        self.log_entry("STEP", "Reading clauses from Stage 1B")
        clauses = self.read_clauses_file()
        if not clauses:
            return False
        
        self.log_entry("STATS", f"Total clauses to classify: {len(clauses)}")
        
        # Step 2: Classify each clause (LangChain or standard) with PARALLEL PROCESSING
        if self.use_langchain:
            self.log_entry("STEP", "Classifying intent with LangChain (parallel mode - 5 workers)")
        else:
            self.log_entry("STEP", "Classifying intent with standard Gemini (parallel mode - 5 workers)")
        
        classified = []
        
        def classify_single_clause(clause_data):
            """Helper function for parallel processing"""
            i, clause = clause_data
            clause_id = clause.get("clauseId", f"C{i}")
            clause_text = clause.get("text", "")
            
            self.log_entry("CLASSIFY", f"[{i}/{len(clauses)}] Processing {clause_id}")
            
            # Classify this clause
            if self.use_langchain:
                intent_result = self.classify_intent_with_langchain(clause_id, clause_text)
            else:
                intent_result = self.classify_intent_with_gemini(clause_id, clause_text)
            
            # Build classified clause record
            classified_clause = {
                "clauseId": clause_id,
                "text": clause_text,
                "intent": intent_result.get("intent", "INFORMATIONAL"),
                "confidence": intent_result.get("confidence", 0.5),
                "reasoning": intent_result.get("reasoning", "")
            }
            
            self.log_entry("RESULT", f"{clause_id}: {classified_clause['intent']} (confidence: {classified_clause['confidence']})")
            return classified_clause
        
        # Process clauses in parallel (5 workers for optimal speed)
        with ThreadPoolExecutor(max_workers=7) as executor:
            clause_data = [(i, clause) for i, clause in enumerate(clauses, 1)]
            classified = list(executor.map(classify_single_clause, clause_data))
        
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
    Step 3: Extract structured entities & thresholds using Gemini API.
    
    For each clause text from Stage 2, extracts explicit entities only.
    Entity types: Type, Class, Category, amount, Currency, Hours, location, role, ApprovalAuthority
    """
    
    def __init__(self, classified_file, document_id=None, enable_mongodb=True, use_langchain=True, max_workers=7):
        self.classified_file = classified_file
        self.entities_file = f"{OUTPUT_DIR}/stage3_entities.json"
        self.document_id = document_id
        self.log = []
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        # Parallel processing config
        self.max_workers = max_workers  # Number of parallel threads
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
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
            if not LANGCHAIN_AVAILABLE:
                self.log_entry("WARNING", "LangChain not available, using standard extraction")
    
    def log_entry(self, level, message):
        """Log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
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
    
    def extract_entities_with_gemini(self, clause_id, clause_text):
        """
        STANDARD: Use Gemini API to extract entities dynamically from a single clause.
        Entities are NOT predefined - LLM identifies meaningful key-value pairs.
        
        Args:
            clause_id (str): Clause identifier (C1, C2, etc.)
            clause_text (str): Clause text with policy content
            
        Returns:
            dict: { "entityName": value, ... } with dynamically identified entities
        """
        
        prompt = f"""
You are a DYNAMIC ENTITY EXTRACTION ENGINE. Extract meaningful structured entities from policy text.

CRITICAL RULES:
1. Extract ONLY values explicitly stated in the text
2. Do NOT infer, guess, or assume missing values
3. Do NOT include null/empty values - omit if not present
4. Preserve original units (currency, time, amounts, etc.)
5. Create entity names that are descriptive and meaningful
6. One clause may have multiple entity values

ENTITY EXTRACTION GUIDELINES:
- Identify key-value pairs that represent measurable facts
- Entity names should be descriptive (e.g., "dailyAllowance", "travelDuration", "approvalAuthority")
- Include units in values when present (e.g., "7.0 Rs/KM", "15 days", "100 KM")
- Group related values logically
- Use camelCase for multi-word entity names

CLAUSE TO ANALYZE:
Clause ID: {clause_id}
Text: "{clause_text}"

EXTRACTION TASK:
1. Scan clause text for explicit, measurable values
2. Identify meaningful entity names based on context
3. Extract only what is clearly stated
4. Omit entities not explicitly mentioned
5. Return a JSON object with extracted entities

OUTPUT FORMAT (valid JSON object, empty if no entities):
{{
  "descriptiveEntityName1": "value1",
  "descriptiveEntityName2": "value2"
}}

EXAMPLES:
- Text: "Rs. 7.0 per KM for Four Wheeler or Rs. 2.5 per KM for Two Wheeler"
  Extract: {{"fourWheelerRate": "7.0 Rs/KM", "twoWheelerRate": "2.5 Rs/KM"}}

- Text: "Employee grade M1-M6 with corresponding lodging amounts"
  Extract: {{"employeeGrades": "M1-M6", "lodgingVariation": "grade-dependent"}}

- Text: "Bills MUST be submitted within 15 days"
  Extract: {{"billSubmissionDeadline": "15 days"}}

- Text: "Requires HOD approval for direct booking"
  Extract: {{"requiredApproval": "HOD"}}

- Text: "Travel to NCR, Mumbai, Delhi"
  Extract: {{"allowedCities": "NCR, Mumbai, Delhi"}}

- If text has no measurable values, return: {{}}

Output ONLY valid JSON. No explanation outside JSON.
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
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
                
                # Return all entities as-is (fully dynamic, no validation against predefined list)
                return result
            
            except json.JSONDecodeError as e:
                self.log_entry("ERROR", f"{clause_id}: Failed to parse Gemini response: {e}")
                return {}
        
        except Exception as e:
            self.log_entry("ERROR", f"{clause_id}: Gemini API call failed: {e}")
            return {}
    
    def extract(self):
        """Main extraction workflow with PARALLEL PROCESSING"""
        self.log_entry("START", "Step 3: Entity & Threshold Extraction")
        
        # Step 1: Read classified clauses from Stage 2
        self.log_entry("STEP", "Reading classified clauses from Stage 2")
        clauses = self.read_classified_file()
        if not clauses:
            return False
        
        self.log_entry("INFO", f"Starting Stage 3: Entity Extraction (Parallel)")
        self.log_entry("INFO", f"Configuration: max_workers={self.max_workers}")
        
        if self.use_langchain:
            self.log_entry("INFO", f"Extracting entities from {len(clauses)} clauses with LangChain (parallel mode)")
        else:
            self.log_entry("INFO", f"Extracting entities from {len(clauses)} clauses with standard Gemini (parallel mode)")
        
        # Step 2: Extract entities from each clause in parallel
        def extract_single(clause):
            i = 1  # Will be set in loop
            clause_id = clause.get("clauseId")
            clause_text = clause.get("text", "")
            clause_intent = clause.get("intent", "UNKNOWN")
            
            # Extract entities from this clause
            if self.use_langchain:
                entities = self.extract_entities_with_langchain(clause_id, clause_text)
            else:
                entities = self.extract_entities_with_gemini(clause_id, clause_text)
            
            entity_count = len(entities)
            self.log_entry("RESULT", f"{clause_id}: Extracted {entity_count} entities: {list(entities.keys())}")
            
            # Build extracted clause record
            return {
                "clauseId": clause_id,
                "text": clause_text,
                "intent": clause_intent,
                "entities": entities
            }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            extracted = list(executor.map(extract_single, clauses))
        
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
    
    def __init__(self, rules_file, max_attempts=5):
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
        print("STEP 3: ENTITY & THRESHOLD EXTRACTION")
        print(f"{'='*80}\n")
        
        entity_extractor = EntityExtractor(classified_file)
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
        
        orchestrator = PipelineOrchestrator(pdf_file, enable_mongodb=False)
        result = orchestrator.run_full_pipeline()
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
        Analyze single clause for ambiguity using LangChain or Gemini.
        Returns: {clauseId, ambiguous, reason, ambiguity_types}
        """
        clause_id = clause['clauseId']
        clause_text = clause['text']
        entities = extracted_entities or {}
        
        # Use LangChain or Gemini to detect ambiguities
        if self.use_langchain:
            is_ambiguous, reason, ambiguity_types = self.detect_ambiguities_with_langchain(clause_id, clause_text, entities)
        else:
            is_ambiguous, reason = self.detect_ambiguities_with_gemini(clause_id, clause_text, entities)
            ambiguity_types = []
        
        return {
            "clauseId": clause_id,
            "ambiguous": is_ambiguous,
            "reason": reason,
            "ambiguity_types": ambiguity_types
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
        
        with ThreadPoolExecutor(max_workers=7) as executor:
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
    
    def __init__(self, stage3_file, stage4_file, document_id=None, enable_mongodb=True, max_workers=7, batch_size=3):
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
        """
        prompt = f"""You are a policy clarification expert. Your task is to fix clause ambiguities.

CLAUSE ID: {clause_id}

ORIGINAL TEXT:
{original_text}

IDENTIFIED AMBIGUITIES:
{ambiguity_reason}

TASK:
1. Rewrite the clause to FIX the identified ambiguities
2. PRESERVE all original numeric data (amounts, dates, durations, percentages)
3. ADD clarifications where ambiguities exist:
   - Define vague terms (e.g., "Actual" → "Actual expenses capped at X amount")
   - Clarify undefined references (e.g., "tour advance" → explain when/how issued)
   - Fix boundary overlaps (e.g., "3-12 hours" and "12-24 hours" → "3 to <12 hours" and "≥12 to 24 hours")
   - Define undefined authorities (e.g., "HOD approval" → "Approval from Head of Department")
   - Complete incomplete conditions (e.g., "IF X AND Y provided" → "IF company provides BOTH X AND Y, THEN...")
4. Keep the tone, structure, and references to other clauses intact (C1, C2, refs:, etc.)
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
            
            return clarified_text
            
        except Exception as e:
            self.log_entry("WARNING", f"Clarification failed for {clause_id}: {e}")
            return original_text  # Return original if clarification fails
    
    def process_single_clause(self, clause, ambiguity_map):
        """Process a single clause (for parallel execution)"""
        clause_id = clause['clauseId']
        original_text = clause['text']
        entities = clause.get('entities', {})
        
        # Get ambiguity info (now includes both reason and types)
        ambiguity_info = ambiguity_map.get(clause_id, {
            'reason': '',
            'types': [],
            'ambiguous': False
        })
        
        ambiguity_reason = ambiguity_info.get('reason', '')
        ambiguity_types = ambiguity_info.get('types', [])
        is_ambiguous = ambiguity_info.get('ambiguous', False)
        
        # Extract which ambiguity codes exist (use types array, not substring matching)
        ambiguities_fixed = []
        if ambiguity_types:
            # Use the actual ambiguity_types from stage4
            for code in ambiguity_types:
                ambiguities_fixed.append(code)
        
        # Skip clarification if no ambiguity found
        if not is_ambiguous or not ambiguity_types:
            clarified_text = original_text
            ambiguities_fixed = []
            self.log_entry("SKIP", f"{clause_id}: No ambiguities to fix")
        else:
            # Clarify the clause
            self.log_entry("PROCESSING", f"{clause_id}: Generating clarified text ({len(ambiguity_types)} issues to fix)...")
            clarified_text = self.clarify_clause(clause_id, original_text, ambiguity_reason)
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
        """Generate DSL based on similar clause pattern"""
        
        # Build dynamic when conditions from entities
        when_conditions = []
        
        for entity_name, entity_value in entities.items():
            # Convert entity name to fact
            fact = entity_name.replace('.', '_').lower()
            
            # Determine operator based on entity type
            if 'upperLimit' in entity_name or 'maximum' in entity_name:
                operator = 'LESS_THAN_OR_EQUAL'
            elif 'lowerLimit' in entity_name or 'minimum' in entity_name:
                operator = 'GREATER_THAN_OR_EQUAL'
            elif 'rate' in entity_name or 'amount' in entity_name:
                operator = 'EQUALS'
            else:
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
            # Extract limit type from entities
            limit_type = 'general'
            for entity_name in entities.keys():
                if 'tour' in entity_name.lower():
                    limit_type = 'tour_duration'
                elif 'allowance' in entity_name.lower():
                    limit_type = 'allowance_limit'
                break
            
            then_constraints = [{'constraint': f'limit.{limit_type}', 'operator': 'ENFORCED', 'value': 'STRICT'}]
        else:  # CONDITIONAL_ALLOWANCE
            # Extract allowance type
            allowance_type = 'general'
            for entity_name in entities.keys():
                if 'lodging' in entity_name.lower():
                    allowance_type = 'lodging'
                elif 'boarding' in entity_name.lower():
                    allowance_type = 'boarding'
                elif 'mileage' in entity_name.lower():
                    allowance_type = 'mileage'
                break
            
            then_constraints = [{'constraint': f'allowance.{allowance_type}', 'operator': 'APPROVED', 'value': 'CONDITIONAL'}]
        
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
        
        # Build when conditions from entities
        when_conditions = []
        
        for entity_name, entity_value in entities.items():
            # Convert entity name to fact
            fact = entity_name.replace('.', '_').lower()
            
            # Determine operator based on entity type
            if 'upperLimit' in entity_name or 'maximum' in entity_name or 'limit' in entity_name.lower():
                operator = 'LESS_THAN_OR_EQUAL'
            elif 'lowerLimit' in entity_name or 'minimum' in entity_name:
                operator = 'GREATER_THAN_OR_EQUAL'
            elif 'rate' in entity_name or 'amount' in entity_name or 'allowance' in entity_name.lower():
                operator = 'EQUALS'
            elif 'deadline' in entity_name.lower() or 'duration' in entity_name.lower():
                operator = 'LESS_THAN_OR_EQUAL'
            else:
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
            # Extract limit type from entities
            limit_type = 'general'
            for entity_name in entities.keys():
                entity_lower = entity_name.lower()
                if 'tour' in entity_lower:
                    limit_type = 'tour_duration'
                elif 'lodging' in entity_lower:
                    limit_type = 'lodging'
                elif 'boarding' in entity_lower:
                    limit_type = 'boarding'
                elif 'mileage' in entity_lower:
                    limit_type = 'mileage'
                elif 'travel' in entity_lower:
                    limit_type = 'travel'
                break
            
            then_constraints = [{'constraint': f'limit.{limit_type}', 'operator': 'ENFORCED', 'value': 'STRICT'}]
        else:  # CONDITIONAL_ALLOWANCE
            # Extract allowance type
            allowance_type = 'general'
            for entity_name in entities.keys():
                entity_lower = entity_name.lower()
                if 'lodging' in entity_lower:
                    allowance_type = 'lodging'
                elif 'boarding' in entity_lower:
                    allowance_type = 'boarding'
                elif 'mileage' in entity_lower:
                    allowance_type = 'mileage'
                elif 'travel' in entity_lower:
                    allowance_type = 'travel'
                break
            
            then_constraints = [{'constraint': f'allowance.{allowance_type}', 'operator': 'APPROVED', 'value': 'CONDITIONAL'}]
        
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
    
    def run_stage_3(self):
        """Stage 3: Extract entities"""
        self.log_entry("STAGE", "Running Stage 3: Entity Extraction")
        
        try:
            classified_file = self.stage_results.get('stage2_file', f"{OUTPUT_DIR}/stage2_classified.json")
            extractor = EntityExtractor(
                classified_file,
                enable_mongodb=self.enable_mongodb,
                max_workers=PipelineConfig.ENTITY_EXTRACTION_WORKERS
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
            success = generator.generate()
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
    
    def run_full_pipeline(self):
        """
        Execute full pipeline: 1 → 1B → 2 → 3 → 4 → 5 → 6 → 8
        
        Returns:
            dict: Results with success status and stage file paths
        """
        self.log_entry("START", "="*80)
        self.log_entry("START", "FULL PIPELINE ORCHESTRATION: Stages 1 → 1B → 2 → 3 → 4 → 5 → 6 → 8")
        self.log_entry("START", "="*80)
        
        stages = [
            ('1', self.run_stage_1),
            ('1B', self.run_stage_1b),
            ('2', self.run_stage_2),
            ('3', self.run_stage_3),
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
    
    def __init__(self, dsl_file, confidence_file=None, document_id=None, enable_mongodb=True, use_langchain=True):
        self.dsl_file = dsl_file
        self.confidence_file = confidence_file
        self.normalized_file = f"{OUTPUT_DIR}/stage8_normalized_policies.json"
        self.document_id = document_id
        self.log = []
        self.log_lock = threading.Lock()
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        # OPTIMIZATION: Make LLM truly optional for faster processing
        self.llm_enabled = self.use_langchain
        
        # Initialize MongoDB storage
        self.storage = PipelineStageStorage(enable_mongodb=enable_mongodb, document_id=document_id or "unknown")
        
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
    
    def rule_based_mapping(self, dsl_rule):
        """Fast rule-based DSL → Normalized JSON mapping"""
        
        # Extract basic fields from DSL
        rule_id = dsl_rule.get('rule_id', 'unknown')
        when_conditions = dsl_rule.get('when', {}).get('all', [])
        then_actions = dsl_rule.get('then', {})
        
        # Determine enforcement type
        enforcement = 'enforce' if 'enforce' in then_actions else 'warn'
        constraints = then_actions.get(enforcement, [])
        
        # Build normalized structure
        normalized = {
            'policyId': f"POLICY_{rule_id}",
            'name': f"Policy Rule {rule_id}",
            'scope': {
                'tenantId': 'default-tenant',
                'orgId': 'default-org',
                'appliesTo': {
                    'roles': [],
                    'locations': []
                }
            },
            'when': [],
            'what': {
                'constraint': {
                    'fact': 'unknown',
                    'operator': 'EQUALS',
                    'value': 'unknown'
                }
            },
            'outcome': {
                'enforcement': enforcement,
                'message': f"Policy rule {rule_id} applied"
            },
            'metadata': {
                'source': f"DSL Rule {rule_id}",
                'approvedBy': 'system',
                'version': '1.0'
            }
        }
        
        # Map when conditions
        for condition in when_conditions:
            normalized['when'].append({
                'fact': condition.get('fact', 'unknown'),
                'operator': condition.get('operator', 'EQUALS'),
                'value': condition.get('value', 'unknown')
            })
        
        # Map constraints to 'what'
        if constraints:
            first_constraint = constraints[0]
            normalized['what']['constraint'] = {
                'fact': first_constraint.get('constraint', 'unknown'),
                'operator': first_constraint.get('operator', 'EQUALS'),
                'value': first_constraint.get('value', 'unknown')
            }
        
        return normalized
    
    def extract_metadata_with_llm(self, dsl_rule, normalized_policy):
        """Use LLM to extract complex metadata when rule-based mapping is insufficient"""
        
        # OPTIMIZATION: Skip LLM if disabled
        if not self.llm_enabled:
            self.apply_rule_based_metadata(dsl_rule, normalized_policy)
            return normalized_policy
        
        # Check if we need LLM (e.g., missing scope, roles, locations)
        scope = normalized_policy.get('scope', {})
        rule_id = dsl_rule.get('rule_id', 'unknown')
        
        if not scope.get('appliesTo', {}).get('roles') and not scope.get('appliesTo', {}).get('locations'):
            
            # Create a more specific prompt with examples
            prompt = f"""Extract policy metadata from this DSL rule.

DSL RULE:
{json.dumps(dsl_rule, indent=2)}

RULE ID: {rule_id}

TASK: Extract metadata as valid JSON:
1. roles: List of employee roles/grades (e.g., ["M1", "M2", "Directors", "CEO"])
2. locations: List of locations/regions (e.g., ["Category A cities", "India"])
3. tenantId: Use "default"
4. orgId: Use "default"

EXAMPLES:
- If rule mentions "grade M1" → roles: ["M1"]
- If rule mentions "Category A cities" → locations: ["Category A cities"]
- If no roles mentioned → roles: []
- If no locations mentioned → locations: []

OUTPUT THIS EXACT JSON (no markdown, no explanations):
{{"roles": [], "locations": [], "tenantId": "default", "orgId": "default"}}

CRITICAL: Output must be valid JSON that can be parsed by json.loads().
"""
            
            try:
                import time
                max_retries = 2
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        response = self.model.generate_content(prompt)
                        response_text = response.text.strip()
                        
                        # More aggressive cleanup
                        response_text = response_text.replace('```json', '').replace('```', '').strip()
                        response_text = '\n'.join(line.strip() for line in response_text.split('\n')).strip()
                        
                        # Check for empty response
                        if not response_text:
                            raise ValueError(f"Empty response from LLM on attempt {attempt + 1}")
                        
                        # Try to extract JSON with pattern matching
                        try:
                            metadata = json.loads(response_text)
                        except json.JSONDecodeError:
                            # Try to find JSON pattern in response
                            import re
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                metadata = json.loads(json_match.group())
                            else:
                                raise ValueError(f"No JSON found in response")
                        
                        # Validate metadata structure
                        if not isinstance(metadata, dict):
                            raise ValueError(f"Expected dict, got {type(metadata)}")
                        
                        # Update normalized policy with LLM-extracted metadata
                        if 'roles' in metadata and isinstance(metadata['roles'], list):
                            scope.setdefault('appliesTo', {})['roles'] = metadata['roles']
                        if 'locations' in metadata and isinstance(metadata['locations'], list):
                            scope.setdefault('appliesTo', {})['locations'] = metadata['locations']
                        if 'tenantId' in metadata:
                            scope['tenantId'] = metadata['tenantId']
                        if 'orgId' in metadata:
                            scope['orgId'] = metadata['orgId']
                        
                        normalized_policy['scope'] = scope
                        self.log_entry("LLM_METADATA", f"Successfully extracted metadata for {rule_id}")
                        return  # Success
                        
                    except Exception as inner_e:
                        if attempt < max_retries - 1:
                            self.log_entry("WARNING", f"LLM attempt {attempt + 1} failed for {rule_id}: {inner_e}")
                            time.sleep(retry_delay)
                        else:
                            raise inner_e
                
            except Exception as e:
                self.log_entry("WARNING", f"LLM metadata extraction failed for {rule_id}: {e}")
                # Use intelligent defaults based on DSL rule content
                self.apply_rule_based_metadata(dsl_rule, normalized_policy)
        
        return normalized_policy
    
    def apply_rule_based_metadata(self, dsl_rule, normalized_policy):
        """Apply metadata from DSL rule structure when LLM fails"""
        
        # Extract metadata from DSL rule conditions
        when_conditions = dsl_rule.get('when', {}).get('all', [])
        rule_id = dsl_rule.get('rule_id', 'unknown')
        original_text = dsl_rule.get('original_text', '')
        
        # Initialize scope if needed
        scope = normalized_policy.get('scope', {})
        applies_to = scope.get('appliesTo', {})
        
        # Extract roles from grade conditions
        roles = applies_to.get('roles', [])
        for condition in when_conditions:
            if condition.get('fact') in ['employee.grade', 'grade']:
                grade_value = condition.get('value')
                if grade_value and grade_value not in roles:
                    roles.append(grade_value)
        
        # Enhanced role extraction from original text
        if not roles and original_text:
            # Look for common role patterns
            import re
            grade_patterns = [r'Grade\s+([A-Z]\d+)', r'\b(M[1-6])\b', 
                            r'(Director|CEO|COO|President|VP|AVP|GM|DGM|RM|AGM|Sr\.\s*Manager|Manager)',
                            r'(Deputy\s*Manager|Assistant\s*Manager|Sr\.\s*Executive|Engineer|ASM|BM|RSM)']
            
            for pattern in grade_patterns:
                matches = re.findall(pattern, original_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]  # Handle grouped matches
                    if match and match not in roles:
                        roles.append(match.strip())
        
        # Apply extracted roles
        if roles:
            applies_to['roles'] = list(set(roles))  # Remove duplicates
        
        # Extract locations from location conditions
        locations = applies_to.get('locations', [])
        for condition in when_conditions:
            if condition.get('fact') in ['location.category', 'location']:
                location_value = condition.get('value')
                if location_value and location_value not in locations:
                    locations.append(location_value)
        
        # Enhanced location extraction from original text
        if not locations and original_text:
            import re
            location_patterns = [r'Category\s+([A-B])\s+cities', r'(Category\s+[A-B]\s+cities)',
                               r'(India|metro|urban|rural|international)', 
                               r'(city|state|region|zone)']
            
            for pattern in location_patterns:
                matches = re.findall(pattern, original_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                    if match and match.strip() and match.strip() not in locations:
                        locations.append(match.strip())
        
        # Apply extracted locations
        if locations:
            applies_to['locations'] = list(set(locations))
        
        # Set default organization values
        if 'tenantId' not in scope:
            scope['tenantId'] = 'default'
        if 'orgId' not in scope:
            scope['orgId'] = 'default'
        
        # Update scope
        scope['appliesTo'] = applies_to
        normalized_policy['scope'] = scope
        
        self.log_entry("RULE_METADATA", f"Applied enhanced rule-based metadata for {rule_id}: roles={roles}, locations={locations}")
    
    def generate_normalized_policies(self):
        """Generate normalized policies using hybrid approach"""
        self.log_entry("INFO", "Starting Stage 8: Normalized Policy Generation (Hybrid Approach)")
        
        # Load input data
        dsl_rules = self.load_dsl_rules()
        confidence_data = self.load_confidence_data()
        
        if not dsl_rules:
            self.log_entry("ERROR", "No DSL rules found")
            return False
        
        self.log_entry("INFO", f"Processing {len(dsl_rules)} DSL rules")
        
        normalized_policies = []
        llm_calls = 0
        
        for dsl_rule in dsl_rules:
            rule_id = dsl_rule.get('rule_id', 'unknown')
            self.log_entry("PROCESSING", f"Normalizing policy {rule_id}")
            
            # Step 1: Rule-based mapping (fast, no LLM)
            normalized = self.rule_based_mapping(dsl_rule)
            
            # Step 2: Apply confidence metadata if available
            if confidence_data and rule_id in confidence_data:
                confidence_info = confidence_data[rule_id]
                if 'confidence' in confidence_info:
                    normalized['metadata']['confidence_score'] = confidence_info['confidence']
                if 'rationale' in confidence_info:
                    normalized['outcome']['message'] = confidence_info['rationale']
            
            # Step 3: LLM for complex metadata (only when needed)
            if self.use_langchain:  # Enable LLM for metadata extraction only if enabled
                normalized = self.extract_metadata_with_llm(dsl_rule, normalized)
                llm_calls += 1
            else:
                # Use enhanced rule-based metadata extraction
                self.apply_rule_based_metadata(dsl_rule, normalized)
                self.log_entry("RULE_BASED", f"Using rule-based metadata for {rule_id}")
            
            normalized_policies.append(normalized)
            self.log_entry("GENERATED", f"Policy {rule_id} normalized")
        
        self.log_entry("SUCCESS", f"Generated {len(normalized_policies)} normalized policies ({llm_calls} LLM calls)")
        
        # Save output
        return self.save_normalized_policies(normalized_policies)
    
    def save_normalized_policies(self, policies):
        """Save normalized policies to JSON file"""
        try:
            output = {
                'policies': policies,
                'metadata': {
                    'generated': datetime.now().isoformat(),
                    'total_policies': len(policies),
                    'format': 'Normalized Policy JSON (Stage 8)'
                }
            }
            
            with open(self.normalized_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            self.log_entry("SUCCESS", f"Normalized policies saved to: {self.normalized_file}")
            
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
            self.log_entry("ERROR", f"Failed to save normalized policies: {e}")
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
