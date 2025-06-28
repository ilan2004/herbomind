from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RemedyMatcher:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._build_remedy_vectors()
    
    def _build_remedy_vectors(self):
        """Build TF-IDF vectors for remedies"""
        remedies = self.data_loader.load_remedies()['remedies']
        
        # Create text representations of remedies
        remedy_texts = []
        for remedy in remedies:
            text = ' '.join(remedy['indications'] + remedy['properties'])
            remedy_texts.append(text)
        
        self.remedy_vectors = self.vectorizer.fit_transform(remedy_texts)
        self.remedies = remedies
    
    def find_matches(self, symptoms: List[Dict], top_k: int = 5) -> List[Dict]:
        """Find matching remedies for given symptoms"""
        if not symptoms:
            return []
        
        # Create query vector from symptoms
        symptom_text = ' '.join([s['name'] for s in symptoms])
        query_vector = self.vectorizer.transform([symptom_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.remedy_vectors)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        matches = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                remedy = self.remedies[idx].copy()
                remedy['confidence_score'] = float(similarities[idx])
                remedy['match_reasons'] = self._get_match_reasons(symptoms, remedy)
                matches.append(remedy)
        
        return matches
    
    def _get_match_reasons(self, symptoms: List[Dict], remedy: Dict) -> List[str]:
        """Explain why this remedy matches"""
        reasons = []
        symptom_names = [s['name'].lower() for s in symptoms]
        
        for indication in remedy['indications']:
            for symptom_name in symptom_names:
                if symptom_name in indication.lower():
                    reasons.append(f"Effective for {indication}")
        
        return reasons