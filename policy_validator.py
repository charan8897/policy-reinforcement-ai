#!/usr/bin/env python3
"""
Generic Policy Enforcement System
Step 1: Extract policies from PDF → filename.txt
Step 2: Use grep + Gemini API to extract rules → rules.txt

Usage:
  python policy_validator.py extract <policy.pdf>
  python policy_validator.py extract-rules <filename.txt>
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import google.generativeai as genai

# Configuration
GEMINI_API_KEY = "AIzaSyBwRjRbssL6kxZPx55_I1yCONHdCZokM-c"
OUTPUT_DIR = "/home/hutech/Documents/docupolicy"
LOG_FILE = f"{OUTPUT_DIR}/mechanism.log"

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
       """Use Gemini API to extract structured rules in Drools format"""
       self.log_entry("GEMINI", "Initializing Gemini API for Drools rule extraction")
       
       # Build prompt for Drools rule extraction with advanced patterns
       prompt = f"""Analyze this policy document and convert ALL policies into Drools rule format.

       Policy Document:
       {content}

       TASK: Convert policies to Drools rules using this EXACT format:

       ```drools
       rule "Rule Name"
       when
       // Conditions (facts and constraints)
       FactType(attribute == value, attribute2 > value2)
       AnotherFact(condition)
       then
       // Actions based on enforcement level
       // For MUST: Set violations, errors, compliance status
       // For EXPECTED: Set warnings, approval flags  
       // For SHOULD: Set recommendations, best practice flags
       action1();
       action2();
       update(fact);
       end
       ```

       REQUIRED DROOLS ELEMENTS:
       1. rule "Clear Descriptive Name": Use the policy title or specific condition scenario
       2. when clause: Define conditions using facts (objects with attributes)
       - Use fact types like: Employee, TravelBooking, Flight, TravelExpense, Document, Policy, etc.
       - Use operators: ==, !=, <, >, <=, >=, contains, matches
       - Bind facts to variables using $: $employee : Employee(status == 'ACTIVE')
       - Support null checks and collection queries
       3. then clause: Define actions with validation logic
       - For MUST rules: setViolation(true), addError('message'), setComplianceStatus('NON_COMPLIANT')
       - For EXPECTED: setRequiresApproval(true), addWarning('message')
       - For SHOULD: addRecommendation('message'), setComplianceStatus('PENDING')
       - Include conditional actions (if/else) for validation logic
       - Always call update($fact) to persist changes
       4. end: Close each rule

       EXTRACTION GUIDELINES:
       1. Extract EVERY policy, requirement, and rule from the document
       2. Identify enforcement level: MUST (mandatory/hard stop), EXPECTED (should comply), SHOULD (guideline)
       3. Map policy attributes to Drools facts (create logical fact types as needed)
       4. Convert prose conditions into boolean logic with clear operators and thresholds
       5. Translate policy consequences into Drools actions with validation
       6. Create multiple specific rules if a policy has multiple conditions, thresholds, or exception paths
       7. Include exception handling rules for business justifications or special approvals

       PATTERN 1 - Specific Thresholds & Conditions:
       rule "Flight Class - International - Over 6 Hours - Business"
       when
       $flight : Flight(duration > 6, isInternational == true)
       $traveler : Employee(role != 'SET_MEMBER', role != 'BOARD_MEMBER')
       then
       $flight.setAllowedClass('BUSINESS');
       if ($flight.getBookedClass() != 'BUSINESS' && $flight.getBookedClass() != 'ECONOMY') {{
       $flight.addError('Invalid class selection for international flights over 6 hours');
       }}
       update($flight);
       end

       PATTERN 2 - Validation Logic with Compliance Status:
       rule "Corporate Card - Mandatory Payment - Validation"
       when
       $expense : TravelExpense(paymentMethod != 'CORPORATE_CARD', amount > 0)
       $traveler : Employee()
       then
       $expense.setViolation(true);
       $expense.addError('MUST use Corporate Card for all business travel expenses');
       $expense.setComplianceStatus('NON_COMPLIANT');
       update($expense);
       end

       PATTERN 3 - Exception Handling with Approval:
       rule "Advance Booking - Exception for Business Need"
       when
       $booking : TravelBooking(daysInAdvance < 14, hasBusinessJustification == true)
       $manager : Employee(role == 'LINE_MANAGER', approved == true)
       then
       $booking.setExceptionApproved(true);
       $booking.addNote('Exception approved due to business need');
       update($booking);
       end

       PATTERN 4 - Domestic vs International & Duration Rules:
       rule "Flight Class - Domestic - Any Duration - Economy"
       when
       $flight : Flight(isInternational == false)
       $traveler : Employee()
       then
       $flight.setAllowedClass('ECONOMY');
       if ($flight.getBookedClass() != 'ECONOMY') {{
       $flight.addWarning('Domestic flights must be booked in ECONOMY class');
       }}
       update($flight);
       end

       PATTERN 5 - Multi-Factor Approval Rules:
       rule "Premium Hotel - Requires Multi-Level Approval"
       when
       $booking : HotelBooking(starRating >= 4, costPerNight > 250)
       $traveler : Employee(approvalLevel < 3)
       then
       $booking.setRequiresApproval(true);
       $booking.setApprovalLevel(2);
       $booking.addWarning('Premium hotel bookings require director-level approval');
       update($booking);
       end

       ADVANCED FEATURES TO INCLUDE:
       - Separate rules for MUST vs EXPECTED vs SHOULD enforcement
       - Create threshold-based variations (e.g., < 14 days, 14-30 days, > 30 days)
       - Include both violation and warning rules
       - Add exception approval paths where applicable
       - Use validation logic (if statements) to check for invalid combinations
       - Set compliance status ('COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'EXCEPTION_APPROVED')
       - Include informative error/warning messages with specific guidance
       - Create rules for both positive and negative conditions

       Now extract ALL rules from the policy document above in Drools format, following all patterns and guidelines:"""

       try:
           self.log_entry("GEMINI_REQUEST", "Sending content for Drools rule generation")
           response = self.model.generate_content(prompt)
           
           self.log_entry("GEMINI_RESPONSE", f"Received Drools rules, processing...")
           return response.text
           
       except Exception as e:
           self.log_entry("ERROR", f"Gemini API failed: {e}")
           return None
    
    def format_rules_output(self, gemini_response):
        """Format Gemini response as structured rules.txt"""
        
        output = "=" * 80 + "\n"
        output += "POLICY EXTRACTION - COMPREHENSIVE RULES DATABASE\n"
        output += "=" * 80 + "\n\n"
        output += f"Generated: {datetime.now()}\n"
        output += f"Source: {self.policy_file}\n"
        output += f"Extraction Method: Grep + Gemini API\n\n"
        output += "=" * 80 + "\n"
        output += "EXTRACTED POLICIES AND RULES\n"
        output += "=" * 80 + "\n\n"
        
        output += gemini_response
        
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
    
    def save_log(self):
        """Append log to mechanism.log"""
        with open(LOG_FILE, 'a') as f:
            f.write("\n\n=== RULE EXTRACTION LOG ===\n")
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
    print("  1) extract      - Extract policy from PDF → filename.txt")
    print("  2) extract-rules - Extract rules from text → rules.txt")
    print("  3) evaluate     - Evaluate payload against rules")
    print("  4) exit         - Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        pdf_file = input("Enter PDF file path: ").strip()
        if not pdf_file:
            print("ERROR: PDF file path required")
            return None
        return ("extract", pdf_file)
    
    elif choice == "2":
        text_file = input("Enter text file path (default: filename.txt): ").strip()
        if not text_file:
            text_file = f"{OUTPUT_DIR}/filename.txt"
        return ("extract-rules", text_file)
    
    elif choice == "3":
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
    
    elif choice == "4":
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
        command, arg = result
    else:
        command = sys.argv[1]
        if len(sys.argv) < 3:
            print("Usage:")
            print("  python policy_validator.py extract <policy.pdf>")
            print("  python policy_validator.py extract-rules <filename.txt>")
            print("  python policy_validator.py evaluate <payload.json>")
            print("\nOr run without arguments for interactive mode:")
            print("  python policy_validator.py")
            sys.exit(1)
        arg = sys.argv[2]
    
    if command == "extract":
        pdf_file = arg
        
        print(f"\n{'='*80}")
        print("STEP 1: POLICY EXTRACTION (PDF → TEXT)")
        print(f"{'='*80}\n")
        
        extractor = PolicyExtractor(pdf_file)
        success = extractor.extract_pdf()
        extractor.save_log()
        
        if success:
            print(f"\n✓ Extraction complete. Output: {OUTPUT_DIR}/filename.txt")
            print("\nNext step: Extract rules with Gemini API")
            print(f"  python policy_validator.py extract-rules {OUTPUT_DIR}/filename.txt")
        else:
            print("\n✗ Extraction failed. Check logs.")
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
    
    else:
        print(f"ERROR: Unknown command '{command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
