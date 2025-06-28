import spacy
import re
from typing import List, Dict, Tuple
import streamlit as st

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.severity_keywords = {
            'mild': ['slight', 'little', 'minor', 'light'],
            'moderate': ['moderate', 'regular', 'normal'],
            'severe': ['severe', 'intense', 'unbearable', 'extreme', 'terrible']
        }
        self.emergency_keywords = [
            'chest pain', 'difficulty breathing', 'sudden severe',
            'unconscious', 'bleeding heavily', 'high fever'
        ]
    
    def extract_symptoms(self, text: str) -> List[Dict]:
        """Extract symptoms from user input"""
        doc = self.nlp(text.lower())
        
        # Simple symptom extraction (can be enhanced)
        symptoms = []
        symptom_patterns = [
            r'(headache|head\s*ache|migraine)',
            r'(stomach\s*ache|stomach\s*pain|digestive\s*issue)',
            r'(cold|cough|fever)',
            r'(back\s*pain|backache)',
            r'(joint\s*pain|arthritis)'
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                symptom = {
                    'name': match if isinstance(match, str) else match[0],
                    'severity': self._detect_severity(text),
                    'duration': self._extract_duration(text),
                    'context': text
                }
                symptoms.append(symptom)
        
        return symptoms
    
    def _detect_severity(self, text: str) -> str:
        """Detect symptom severity from text"""
        text_lower = text.lower()
        
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return severity
        
        return 'moderate'  # default
    
    def _extract_duration(self, text: str) -> str:
        """Extract duration information"""
        duration_patterns = [
            r'(\d+)\s*(day|days|week|weeks|month|months)',
            r'(since\s+\w+)',
            r'(for\s+\d+\s+\w+)'
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