from typing import List, Dict, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RemedyMatcher:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.symptom_vectorizer = TfidfVectorizer(stop_words='english')
        self._build_remedy_vectors()
        self._build_symptom_remedy_map()
    
    def _build_remedy_vectors(self):
        """Build TF-IDF vectors for remedies"""
        remedies = self.data_loader.load_remedies()['remedies']
        
        # Create text representations of remedies
        remedy_texts = []
        for remedy in remedies:
            # Combine indications and properties for better matching
            text = ' '.join(remedy['indications'] + remedy['properties'])
            remedy_texts.append(text)
        
        self.remedy_vectors = self.vectorizer.fit_transform(remedy_texts)
        self.remedies = remedies
        
        # Create a lookup dictionary for remedies by ID
        self.remedy_by_id = {remedy['id']: remedy for remedy in remedies}
    
    def _build_symptom_remedy_map(self):
        """Build mapping between symptoms and remedies from symptoms.json"""
        self.symptom_remedy_map = {}
        self.category_remedy_map = {}
        
        try:
            symptoms_data = self.data_loader.load_symptoms()
            
            # Process symptom-remedy connections if available
            if 'symptoms' in symptoms_data:
                for symptom in symptoms_data['symptoms']:
                    if 'suitable_remedies' in symptom:
                        self.symptom_remedy_map[symptom['name'].lower()] = symptom['suitable_remedies']
                    
                    # Also map by category
                    if 'category' in symptom:
                        category = symptom['category']
                        if category not in self.category_remedy_map:
                            self.category_remedy_map[category] = set()
                        
                        if 'suitable_remedies' in symptom:
                            self.category_remedy_map[category].update(symptom['suitable_remedies'])
        except Exception as e:
            # If there's an issue with symptoms.json, continue without the mapping
            print(f"Warning: Could not build symptom-remedy map: {e}")
            self.symptom_remedy_map = {}
            self.category_remedy_map = {}
    
    def find_matches(self, symptoms: List[Dict], top_k: int = 5) -> List[Dict]:
        """Find matching remedies for given symptoms using multiple methods"""
        if not symptoms:
            return []
        
        # Method 1: Direct mapping from symptoms.json
        direct_matches = self._find_direct_matches(symptoms)
        
        # Method 2: TF-IDF similarity matching
        similarity_matches = self._find_similarity_matches(symptoms, top_k)
        
        # Combine and rank results
        combined_matches = self._combine_and_rank_matches(direct_matches, similarity_matches, top_k)
        
        return combined_matches
    
    def _find_direct_matches(self, symptoms: List[Dict]) -> List[Dict]:
        """Find remedies directly mapped to symptoms in symptoms.json"""
        direct_matches = {}
        
        # Track categories for category-based matching
        categories = set()
        
        for symptom in symptoms:
            symptom_name = symptom['name'].lower()
            
            # Add direct symptom-remedy mappings
            if symptom_name in self.symptom_remedy_map:
                for remedy_id in self.symptom_remedy_map[symptom_name]:
                    if remedy_id in self.remedy_by_id:
                        if remedy_id not in direct_matches:
                            remedy = self.remedy_by_id[remedy_id].copy()
                            remedy['direct_match_count'] = 1
                            remedy['match_reasons'] = [f"Specifically recommended for {symptom_name}"]
                            direct_matches[remedy_id] = remedy
                        else:
                            direct_matches[remedy_id]['direct_match_count'] += 1
                            direct_matches[remedy_id]['match_reasons'].append(f"Specifically recommended for {symptom_name}")
            
            # Track symptom category for category-based matching
            if 'category' in symptom:
                categories.add(symptom['category'])
        
        # Add category-based matches
        for category in categories:
            if category in self.category_remedy_map:
                for remedy_id in self.category_remedy_map[category]:
                    if remedy_id in self.remedy_by_id and remedy_id not in direct_matches:
                        remedy = self.remedy_by_id[remedy_id].copy()
                        remedy['direct_match_count'] = 0.5  # Lower weight for category matches
                        remedy['match_reasons'] = [f"Effective for {category} symptoms"]
                        direct_matches[remedy_id] = remedy
        
        return list(direct_matches.values())
    
    def _find_similarity_matches(self, symptoms: List[Dict], top_k: int = 5) -> List[Dict]:
        """Find matching remedies using TF-IDF and cosine similarity"""
        # Create query vector from symptoms
        symptom_text = ' '.join([s['name'] for s in symptoms])
        query_vector = self.vectorizer.transform([symptom_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.remedy_vectors)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k*2]  # Get more candidates for combining
        
        matches = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                remedy = self.remedies[idx].copy()
                remedy['confidence_score'] = float(similarities[idx])
                remedy['match_reasons'] = self._get_match_reasons(symptoms, remedy)
                matches.append(remedy)
        
        return matches
    
    def _combine_and_rank_matches(self, direct_matches: List[Dict], similarity_matches: List[Dict], top_k: int = 5) -> List[Dict]:
        """Combine and rank matches from different methods"""
        # Create a dictionary to combine matches by remedy ID
        combined = {}
        
        # Add direct matches with high priority
        for remedy in direct_matches:
            remedy_id = remedy['id']
            combined[remedy_id] = remedy
            # Add a high base score for direct matches
            combined[remedy_id]['final_score'] = 0.5 + (remedy.get('direct_match_count', 0) * 0.2)
        
        # Add similarity matches, potentially updating direct matches
        for remedy in similarity_matches:
            remedy_id = remedy['id']
            if remedy_id in combined:
                # Update existing match
                combined[remedy_id]['confidence_score'] = remedy['confidence_score']
                combined[remedy_id]['final_score'] += remedy['confidence_score']
                # Combine match reasons without duplicates
                existing_reasons = set(combined[remedy_id]['match_reasons'])
                for reason in remedy['match_reasons']:
                    if reason not in existing_reasons:
                        combined[remedy_id]['match_reasons'].append(reason)
            else:
                # Add new match
                remedy['final_score'] = remedy['confidence_score']
                combined[remedy_id] = remedy
        
        # Convert to list and sort by final score
        result = list(combined.values())
        result.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Return top k results
        return result[:top_k]
    
    def _get_match_reasons(self, symptoms: List[Dict], remedy: Dict) -> List[str]:
        """Explain why this remedy matches based on text similarity"""
        reasons = []
        symptom_names = [s['name'].lower() for s in symptoms]
        
        # Check indications
        for indication in remedy['indications']:
            for symptom_name in symptom_names:
                if symptom_name in indication.lower():
                    reasons.append(f"Effective for {indication}")
        
        # Check properties if we don't have enough indication matches
        if len(reasons) < 2:
            for property in remedy['properties']:
                property_relevant = False
                for symptom in symptoms:
                    # Check if property is relevant to symptom category
                    if 'category' in symptom:
                        category = symptom['category']
                        if (category == 'respiratory' and property in ['expectorant', 'decongestant', 'anti-inflammatory']) or \
                           (category == 'digestive' and property in ['digestive', 'carminative', 'anti-nausea']) or \
                           (category == 'pain' and property in ['analgesic', 'anti-inflammatory']) or \
                           (category == 'skin' and property in ['antiseptic', 'anti-inflammatory', 'skin healing']):
                            property_relevant = True
                            break
                
                if property_relevant and f"Has {property} properties" not in reasons:
                    reasons.append(f"Has {property} properties")
        
        # Add safety level as a reason if it's high
        if 'safety_level' in remedy and remedy['safety_level'] == 'high':
            reasons.append("Has a high safety profile")
        
        # Add evidence level as a reason if it's strong
        if 'evidence_level' in remedy and remedy['evidence_level'] == 'strong':
            reasons.append("Has strong scientific evidence")
        
        return reasons
    
    def get_symptom_relationships(self, symptoms: List[Dict]) -> Dict:
        """Generate data for symptom relationship visualization"""
        nodes = []
        links = []
        node_ids = set()
        
        # Add symptom nodes
        for symptom in symptoms:
            symptom_id = f"symptom_{symptom['name'].lower().replace(' ', '_')}"
            
            # Only add if not already added
            if symptom_id not in node_ids:
                node_data = {
                    'id': symptom_id,
                    'name': symptom['name'],
                    'type': 'symptom',
                    'category': symptom.get('category', 'general'),
                    'severity': symptom.get('severity', 'moderate')
                }
                nodes.append(node_data)
                node_ids.add(symptom_id)
            
            # Add related symptoms if available
            if 'related_symptoms' in symptom:
                for related in symptom['related_symptoms']:
                    related_id = f"symptom_{related.lower().replace(' ', '_')}"
                    
                    # Add related symptom node if not already added
                    if related_id not in node_ids:
                        related_data = {
                            'id': related_id,
                            'name': related,
                            'type': 'symptom',
                            'category': 'general',  # Default category
                            'severity': 'moderate'  # Default severity
                        }
                        nodes.append(related_data)
                        node_ids.add(related_id)
                    
                    # Add relationship link
                    links.append({
                        'source': symptom_id,
                        'target': related_id,
                        'type': 'related'
                    })
        
        # Find matching remedies
        matches = self.find_matches(symptoms, top_k=5)
        
        # Add remedy nodes and links
        for remedy in matches:
            remedy_id = f"remedy_{remedy['id']}"
            
            # Only add if not already added
            if remedy_id not in node_ids:
                node_data = {
                    'id': remedy_id,
                    'name': remedy['name'],
                    'type': 'remedy',
                    'properties': remedy.get('properties', []),
                    'safety_level': remedy.get('safety_level', 'medium')
                }
                nodes.append(node_data)
                node_ids.add(remedy_id)
            
            # Add links between symptoms and remedies
            for symptom in symptoms:
                symptom_id = f"symptom_{symptom['name'].lower().replace(' ', '_')}"
                symptom_name = symptom['name'].lower()
                
                # Check if this remedy directly treats this symptom
                is_direct_match = False
                if symptom_name in self.symptom_remedy_map and remedy['id'] in self.symptom_remedy_map[symptom_name]:
                    is_direct_match = True
                
                # Add treatment link
                links.append({
                    'source': remedy_id,
                    'target': symptom_id,
                    'type': 'treats',
                    'strength': 'direct' if is_direct_match else 'indirect'
                })
        
        return {
            'nodes': nodes,
            'links': links
        }