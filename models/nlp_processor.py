import spacy
import re
from typing import List, Dict, Tuple, Optional
import streamlit as st

class NLPProcessor:
    def __init__(self, data_loader):
        # Store data loader reference
        self.data_loader = data_loader
        
        # Load a more comprehensive model if available
        try:
            # Try to load medical NLP model if installed
            self.nlp = spacy.load("en_core_sci_md")  # Scientific/medical model
        except:
            # Fallback to standard model
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load symptoms data from JSON
        self.symptoms_data = self.data_loader.load_symptoms()
        
        # Build dictionaries and patterns from symptoms data
        self._build_symptom_dictionaries()
        self.build_symptom_patterns()
    
    def _build_symptom_dictionaries(self):
        """Build dictionaries from symptoms data for quick lookups"""
        # Initialize dictionaries
        self.symptom_by_id = {}
        self.symptom_by_name = {}
        self.symptom_aliases = {}
        self.symptom_categories = {}
        self.severity_keywords = {}
        self.emergency_keywords = []
        
        # Process symptoms data
        if 'symptoms' in self.symptoms_data:
            for symptom in self.symptoms_data['symptoms']:
                # Store by ID and name
                self.symptom_by_id[symptom['id']] = symptom
                self.symptom_by_name[symptom['name'].lower()] = symptom
                
                # Store aliases
                if 'aliases' in symptom:
                    for alias in symptom['aliases']:
                        self.symptom_aliases[alias.lower()] = symptom['name'].lower()
                
                # Organize by category
                category = symptom.get('category', 'general')
                if category not in self.symptom_categories:
                    self.symptom_categories[category] = []
                self.symptom_categories[category].append(symptom['name'].lower())
                
                # Collect severity keywords
                if 'severity_indicators' in symptom:
                    for severity, indicators in symptom['severity_indicators'].items():
                        if severity not in self.severity_keywords:
                            self.severity_keywords[severity] = []
                        self.severity_keywords[severity].extend(indicators)
                
                # Collect emergency flags
                if 'emergency_flags' in symptom:
                    self.emergency_keywords.extend(symptom['emergency_flags'])
        
        # Ensure we have default severity keywords if none were loaded
        if not self.severity_keywords:
            self.severity_keywords = {
                'mild': ['slight', 'little', 'minor', 'light', 'mild', 'faint', 'gentle'],
                'moderate': ['moderate', 'regular', 'normal', 'medium', 'intermediate'],
                'severe': ['severe', 'intense', 'unbearable', 'extreme', 'terrible', 'excruciating', 'worst']
            }
        
        # Ensure we have default emergency keywords if none were loaded
        if not self.emergency_keywords:
            self.emergency_keywords = [
                'chest pain', 'difficulty breathing', 'shortness of breath', 'sudden severe',
                'unconscious', 'bleeding heavily', 'high fever', 'seizure', 'stroke',
                'heart attack', 'cannot move', 'paralysis', 'severe allergic reaction'
            ]
    
    def build_symptom_patterns(self):
        """Build comprehensive regex patterns from symptom categories"""
        self.symptom_patterns = []
        
        # Add patterns for symptom names
        all_symptoms = []
        for symptoms in self.symptom_categories.values():
            all_symptoms.extend(symptoms)
        
        # Add patterns for aliases
        all_symptoms.extend(self.symptom_aliases.keys())
        
        # Create patterns for each symptom with variations
        for symptom in all_symptoms:
            # Handle spaces and variations
            words = symptom.split()
            if len(words) > 1:
                # Multi-word symptoms with possible spacing variations
                pattern = r'(' + r'\s*'.join(words) + r')'
            else:
                # Single word symptoms
                pattern = r'(' + symptom + r')'
            
            self.symptom_patterns.append(pattern)
    
    def extract_symptoms(self, text: str) -> List[Dict]:
        """Extract symptoms from user input using enhanced methods"""
        doc = self.nlp(text.lower())
        symptoms = []
        
        # Method 1: Use spaCy's entity recognition for medical terms
        for ent in doc.ents:
            # Check if entity might be a symptom (based on label or custom rules)
            if self.is_potential_symptom(ent):
                symptom_name = self._normalize_symptom_name(ent.text)
                symptom = {
                    'name': symptom_name,
                    'severity': self._detect_severity(text, ent.start_char, ent.end_char),
                    'duration': self._extract_duration(text),
                    'context': self._extract_context(text, ent.start_char, ent.end_char),
                    'source': 'entity_recognition',
                    'span': (ent.start_char, ent.end_char)
                }
                
                # Add additional information if available in symptoms database
                self._enrich_symptom_data(symptom)
                symptoms.append(symptom)
        
        # Method 2: Pattern matching with comprehensive symptom list
        for pattern in self.symptom_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                # Check if this symptom overlaps with already detected ones
                if not self._is_overlapping(match.span(), symptoms):
                    symptom_name = self._normalize_symptom_name(match.group())
                    symptom = {
                        'name': symptom_name,
                        'severity': self._detect_severity(text, match.start(), match.end()),
                        'duration': self._extract_duration(text),
                        'context': self._extract_context(text, match.start(), match.end()),
                        'source': 'pattern_matching',
                        'span': match.span()
                    }
                    
                    # Add additional information if available in symptoms database
                    self._enrich_symptom_data(symptom)
                    symptoms.append(symptom)
        
        # Add related symptoms based on symptom relationships
        symptoms = self._add_related_symptoms(symptoms)
        
        return symptoms
    
    def _normalize_symptom_name(self, text: str) -> str:
        """Normalize symptom name using aliases dictionary"""
        text_lower = text.lower()
        
        # Check if this is an alias and return the main symptom name
        if text_lower in self.symptom_aliases:
            return self.symptom_aliases[text_lower]
        
        return text_lower
    
    def _enrich_symptom_data(self, symptom: Dict) -> None:
        """Add additional information to symptom from database"""
        symptom_name = symptom['name']
        
        if symptom_name in self.symptom_by_name:
            db_symptom = self.symptom_by_name[symptom_name]
            
            # Add symptom ID
            symptom['id'] = db_symptom['id']
            
            # Add category
            if 'category' in db_symptom:
                symptom['category'] = db_symptom['category']
            
            # Add description
            if 'description' in db_symptom:
                symptom['description'] = db_symptom['description']
            
            # Add suitable remedies
            if 'suitable_remedies' in db_symptom:
                symptom['suitable_remedies'] = db_symptom['suitable_remedies']
    
    def _add_related_symptoms(self, detected_symptoms: List[Dict]) -> List[Dict]:
        """Add related symptoms based on symptom relationships"""
        if 'symptom_relationships' not in self.symptoms_data:
            return detected_symptoms
        
        # Get IDs of detected symptoms
        detected_ids = set()
        for symptom in detected_symptoms:
            if 'id' in symptom:
                detected_ids.add(symptom['id'])
        
        # Find related symptoms
        related_ids = set()
        for relationship in self.symptoms_data['symptom_relationships']:
            primary_id = relationship['primary_symptom']
            if primary_id in detected_ids:
                # Add related symptoms with lower confidence
                for related_id in relationship['related_symptoms']:
                    if related_id not in detected_ids:
                        related_ids.add(related_id)
        
        # Add related symptoms to the list
        for related_id in related_ids:
            if related_id in self.symptom_by_id:
                db_symptom = self.symptom_by_id[related_id]
                symptom = {
                    'name': db_symptom['name'],
                    'id': db_symptom['id'],
                    'severity': 'moderate',  # Default severity
                    'duration': 'not specified',
                    'source': 'relationship_inference',
                    'confidence': 'low',
                    'inferred': True
                }
                
                # Add category and description
                if 'category' in db_symptom:
                    symptom['category'] = db_symptom['category']
                if 'description' in db_symptom:
                    symptom['description'] = db_symptom['description']
                
                detected_symptoms.append(symptom)
        
        return detected_symptoms
    
    def is_potential_symptom(self, entity) -> bool:
        """Determine if an entity might be a symptom"""
        # Check entity label if using a medical model
        if entity.label_ in ["SYMPTOM", "DISEASE", "PROBLEM", "FINDING"]:
            return True
        
        # Check against our symptom dictionary
        entity_text = entity.text.lower()
        
        # Check direct symptom names
        if entity_text in self.symptom_by_name:
            return True
        
        # Check aliases
        if entity_text in self.symptom_aliases:
            return True
        
        # Check against category lists
        for category, symptoms in self.symptom_categories.items():
            if any(symptom in entity_text or entity_text in symptom for symptom in symptoms):
                return True
        
        return False
    
    def _is_overlapping(self, span, symptoms) -> bool:
        """Check if a span overlaps with existing symptoms"""
        start, end = span
        for symptom in symptoms:
            if 'span' in symptom:
                s_start, s_end = symptom['span']
                if (start <= s_end and end >= s_start):
                    return True
        return False
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract the context around a symptom mention"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _detect_severity(self, text: str, start: int = None, end: int = None, window: int = 20) -> str:
        """Detect symptom severity from text with contextual window"""
        # If start and end are provided, look in a window around the symptom
        if start is not None and end is not None:
            context_start = max(0, start - window)
            context_end = min(len(text), end + window)
            text_to_check = text[context_start:context_end].lower()
        else:
            text_to_check = text.lower()
        
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text_to_check for keyword in keywords):
                return severity
        
        return 'moderate'  # default
    
    def _extract_duration(self, text: str) -> str:
        """Extract duration information"""
        duration_patterns = [
            r'(\d+)\s*(day|days|week|weeks|month|months|year|years)',
            r'(since\s+\w+)',
            r'(for\s+\d+\s+\w+)',
            r'(started|began)\s+(\w+|\d+)\s+(day|days|week|weeks|month|months|year|years)\s+(ago)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return 'not specified'
    
    def check_emergency_flags(self, text: str) -> Tuple[bool, List[str]]:
        """Check for emergency symptoms"""
        flags = []
        text_lower = text.lower()
        
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                flags.append(keyword)
        
        return len(flags) > 0, flags
    
    def get_symptom_info(self, symptom_name: str) -> Optional[Dict]:
        """Get detailed information about a symptom from the database"""
        symptom_name = self._normalize_symptom_name(symptom_name)
        
        if symptom_name in self.symptom_by_name:
            return self.symptom_by_name[symptom_name]
        
        return None