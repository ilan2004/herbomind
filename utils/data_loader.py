import json
import pandas as pd
from typing import Dict, List
import streamlit as st

class DataLoader:
    def __init__(self):
        self.remedies_data = None
        self.symptoms_data = None
        self.interactions_data = None
    
    @st.cache_data
    def load_remedies(_self) -> Dict:
        with open('data/remedies.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @st.cache_data
    def load_symptoms(_self) -> Dict:
        with open('data/symptoms.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_remedy_by_id(self, remedy_id: str) -> Dict:
        remedies = self.load_remedies()
        for remedy in remedies['remedies']:
            if remedy['id'] == remedy_id:
                return remedy
        return None
    
    def search_remedies_by_indication(self, indication: str) -> List[Dict]:
        remedies = self.load_remedies()
        matching_remedies = []
        
        for remedy in remedies['remedies']:
            if any(indication.lower() in ind.lower() for ind in remedy['indications']):
                matching_remedies.append(remedy)
        
        return matching_remedies