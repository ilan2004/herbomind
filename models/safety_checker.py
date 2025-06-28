from typing import List, Dict, Tuple
import streamlit as st

class SafetyChecker:
    def __init__(self):
        self.high_risk_conditions = [
            'pregnancy', 'breastfeeding', 'diabetes', 'hypertension',
            'heart disease', 'kidney disease', 'liver disease'
        ]
        self.medication_interactions = {
            'blood thinners': ['turmeric', 'ginger', 'garlic'],
            'diabetes medication': ['bitter melon', 'fenugreek'],
            'blood pressure medication': ['licorice', 'ginseng']
        }
    
    def check_safety(self, remedies: List[Dict], user_profile: Dict) -> List[Dict]:
        """Check safety of remedies for user"""
        safe_remedies = []
        
        for remedy in remedies:
            safety_info = self._assess_remedy_safety(remedy, user_profile)
            remedy['safety_assessment'] = safety_info
            
            if safety_info['is_safe']:
                safe_remedies.append(remedy)
        
        return safe_remedies
    
    def _assess_remedy_safety(self, remedy: Dict, user_profile: Dict) -> Dict:
        """Assess safety of individual remedy"""
        warnings = []
        is_safe = True
        
        # Check contraindications
        user_conditions = user_profile.get('conditions', [])
        for condition in user_conditions:
            if condition.lower() in [c.lower() for c in remedy.get('contraindications', [])]:
                warnings.append(f"Not recommended for {condition}")
                is_safe = False
        
        # Check medication interactions
        user_medications = user_profile.get('medications', [])
        for medication in user_medications:
            if self._check_interaction(remedy['name'], medication):
                warnings.append(f"May interact with {medication}")
                is_safe = False
        
        # Age-based warnings
        age = user_profile.get('age', 0)
        if age < 12:
            warnings.append("Consult pediatrician before use")
            is_safe = False
        elif age > 65:
            warnings.append("Consult doctor - elderly may need dose adjustment")
        
        return {
            'is_safe': is_safe,
            'warnings': warnings,
            'safety_level': remedy.get('safety_level', 'medium')
        }
    
    def _check_interaction(self, remedy_name: str, medication: str) -> bool:
        """Check for specific drug-herb interactions"""
        remedy_lower = remedy_name.lower()
        medication_lower = medication.lower()
        
        for med_type, herbs in self.medication_interactions.items():
            if med_type in medication_lower:
                if any(herb in remedy_lower for herb in herbs):
                    return True
        return False