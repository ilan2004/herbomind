import streamlit as st
import pandas as pd
from utils.data_loader import DataLoader
from models.nlp_processor import NLPProcessor
from models.remedy_matcher import RemedyMatcher
from models.safety_checker import SafetyChecker

# Configure page
st.set_page_config(
    page_title="HerboMind - AI Herbal Remedy Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    data_loader = DataLoader()
    nlp_processor = NLPProcessor()
    remedy_matcher = RemedyMatcher(data_loader)
    safety_checker = SafetyChecker()
    return data_loader, nlp_processor, remedy_matcher, safety_checker

def main():
    # Header
    st.title("🌿 HerboMind")
    st.subheader("AI-Powered Traditional Herbal Remedy Assistant")
    
    # Initialize components
    data_loader, nlp_processor, remedy_matcher, safety_checker = initialize_components()
    
    # Sidebar - User Profile
    with st.sidebar:
        st.header("👤 User Profile")
        
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        
        conditions = st.multiselect(
            "Existing Conditions",
            ["Diabetes", "Hypertension", "Heart Disease", "Pregnancy", "Kidney Disease"]
        )
        
        medications = st.multiselect(
            "Current Medications",
            ["Blood thinners", "Diabetes medication", "Blood pressure medication", "None"]
        )
        
        user_profile = {
            'age': age,
            'conditions': conditions,
            'medications': medications
        }
    
    # Main interface
    st.header("🔍 Symptom Analysis")
    
    # Symptom input
    symptom_input = st.text_area(
        "Describe your symptoms:",
        placeholder="e.g., I have a headache and feel nauseous. The pain started this morning and is getting worse.",
        height=100
    )
    
    if st.button("🔍 Analyze Symptoms", type="primary"):
        if symptom_input:
            with st.spinner("Analyzing symptoms..."):
                # Process symptoms
                symptoms = nlp_processor.extract_symptoms(symptom_input)
                
                # Check for emergency flags
                is_emergency, emergency_flags = nlp_processor.check_emergency_flags(symptom_input)
                
                if is_emergency:
                    st.error("⚠️ EMERGENCY: These symptoms may require immediate medical attention!")
                    st.error(f"Emergency indicators: {', '.join(emergency_flags)}")
                    st.error("Please consult a healthcare provider immediately.")
                    return
                
                # Find matching remedies
                matches = remedy_matcher.find_matches(symptoms)
                
                # Safety check
                safe_remedies = safety_checker.check_safety(matches, user_profile)
                
                # Display results
                if safe_remedies:
                    st.success(f"Found {len(safe_remedies)} safe remedies for your symptoms")
                    
                    for i, remedy in enumerate(safe_remedies):
                        with st.expander(f"🌿 {remedy['name']} (Confidence: {remedy['confidence_score']:.2f})"):
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Scientific Name:**", remedy['scientific_name'])
                                st.write("**Traditional System:**", remedy['traditional_system'])
                                st.write("**Properties:**", ", ".join(remedy['properties']))
                                
                                if remedy['match_reasons']:
                                    st.write("**Why this matches:**")
                                    for reason in remedy['match_reasons']:
                                        st.write(f"• {reason}")
                            
                            with col2:
                                st.write("**Preparation:**")
                                for prep in remedy['preparation']:
                                    st.write(f"• {prep}")
                                
                                st.write("**Dosage:**", remedy['dosage'])
                                
                                # Safety information
                                safety = remedy['safety_assessment']
                                if safety['warnings']:
                                    st.warning("⚠️ **Warnings:**")
                                    for warning in safety['warnings']:
                                        st.write(f"• {warning}")
                
                else:
                    st.warning("No safe remedies found. Please consult a healthcare provider.")
        
        else:
            st.warning("Please describe your symptoms.")
    
    # Additional sections
    st.header("📚 Additional Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Explore Remedies"):
            st.info("Browse our database of traditional remedies")
    
    with col2:
        if st.button("⚕️ Drug Interactions"):
            st.info("Check potential interactions")
    
    with col3:
        if st.button("📖 Learn More"):
            st.info("Traditional medicine education")
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Important Disclaimer:** 
    This tool is for educational purposes only and does not replace professional medical advice. 
    Always consult with qualified healthcare providers before using herbal remedies, especially if you have 
    existing medical conditions or are taking medications.
    """)

if __name__ == "__main__":
    main()