import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import json
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
    nlp_processor = NLPProcessor(data_loader)  # Pass data_loader here
    remedy_matcher = RemedyMatcher(data_loader)
    safety_checker = SafetyChecker()
    
    return data_loader, nlp_processor, remedy_matcher, safety_checker

def create_symptom_network_graph(relationship_data):
    """Create a network graph visualization of symptom relationships"""
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for node in relationship_data['nodes']:
        G.add_node(node['id'], **node)
    
    # Add edges
    for link in relationship_data['links']:
        G.add_edge(link['source'], link['target'], **link)
    
    # Create positions using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create node traces for different types
    symptom_nodes = [n for n in G.nodes if G.nodes[n]['type'] == 'symptom']
    remedy_nodes = [n for n in G.nodes if G.nodes[n]['type'] == 'remedy']
    
    # Create separate edge traces for different types and strengths
    edge_traces = []
    
    # Group edges by type and strength
    edge_groups = {
        'related': {'edges': [], 'color': 'rgba(150,150,150,0.5)', 'name': 'Related'},
        'treats_direct': {'edges': [], 'color': 'rgba(0,100,0,0.7)', 'name': 'Direct Treatment'},
        'treats_indirect': {'edges': [], 'color': 'rgba(0,100,0,0.3)', 'name': 'Indirect Treatment'},
        'default': {'edges': [], 'color': 'rgba(100,100,100,0.3)', 'name': 'Other'}
    }
    
    for edge in G.edges():
        edge_type = G.edges[edge].get('type', 'default')
        
        if edge_type == 'related':
            edge_groups['related']['edges'].append(edge)
        elif edge_type == 'treats':
            strength = G.edges[edge].get('strength', 'indirect')
            if strength == 'direct':
                edge_groups['treats_direct']['edges'].append(edge)
            else:
                edge_groups['treats_indirect']['edges'].append(edge)
        else:
            edge_groups['default']['edges'].append(edge)
    
    # Create traces for each edge group
    for group_key, group_data in edge_groups.items():
        if group_data['edges']:  # Only create trace if there are edges
            edge_x = []
            edge_y = []
            
            for edge in group_data['edges']:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2 if 'direct' in group_key else 1, color=group_data['color']),
                hoverinfo='none',
                mode='lines',
                name=group_data['name'],
                showlegend=False  # Hide edge legends to reduce clutter
            )
            edge_traces.append(edge_trace)
    
    # Create symptom node trace
    symptom_node_x = []
    symptom_node_y = []
    symptom_node_text = []
    symptom_node_colors = []
    symptom_node_sizes = []
    
    for node in symptom_nodes:
        x, y = pos[node]
        symptom_node_x.append(x)
        symptom_node_y.append(y)
        
        # Node text
        node_info = G.nodes[node]
        text = f"Symptom: {node_info['name']}<br>"
        if 'category' in node_info:
            text += f"Category: {node_info['category']}<br>"
        if 'severity' in node_info:
            text += f"Severity: {node_info['severity']}"
        symptom_node_text.append(text)
        
        # Node color based on category
        category = node_info.get('category', 'general')
        category_colors = {
            'respiratory': 'rgba(66, 135, 245, 0.8)',
            'digestive': 'rgba(245, 171, 66, 0.8)',
            'neurological': 'rgba(245, 66, 66, 0.8)',
            'musculoskeletal': 'rgba(66, 245, 108, 0.8)',
            'skin': 'rgba(245, 66, 233, 0.8)',
            'systemic': 'rgba(150, 150, 150, 0.8)',
            'general': 'rgba(100, 100, 100, 0.8)'
        }
        symptom_node_colors.append(category_colors.get(category, 'rgba(100, 100, 100, 0.8)'))
        
        # Node size based on severity
        severity = node_info.get('severity', 'moderate')
        severity_sizes = {
            'mild': 15,
            'moderate': 20,
            'severe': 25,
            'unknown': 15
        }
        symptom_node_sizes.append(severity_sizes.get(severity, 15))
    
    symptom_trace = go.Scatter(
        x=symptom_node_x, y=symptom_node_y,
        mode='markers',
        hoverinfo='text',
        text=symptom_node_text,
        marker=dict(
            color=symptom_node_colors,
            size=symptom_node_sizes,
            line=dict(width=1, color='rgba(50, 50, 50, 0.8)')
        ),
        name='Symptoms'
    )
    
    # Create remedy node trace
    remedy_node_x = []
    remedy_node_y = []
    remedy_node_text = []
    remedy_node_colors = []
    
    for node in remedy_nodes:
        x, y = pos[node]
        remedy_node_x.append(x)
        remedy_node_y.append(y)
        
        # Node text
        node_info = G.nodes[node]
        text = f"Remedy: {node_info['name']}<br>"
        if 'properties' in node_info:
            text += f"Properties: {', '.join(node_info['properties'][:3])}<br>"
        if 'safety_level' in node_info:
            text += f"Safety: {node_info['safety_level']}"
        remedy_node_text.append(text)
        
        # Node color based on safety level
        safety = node_info.get('safety_level', 'medium')
        safety_colors = {
            'high': 'rgba(0, 180, 0, 0.8)',
            'medium': 'rgba(180, 180, 0, 0.8)',
            'low': 'rgba(180, 0, 0, 0.8)'
        }
        remedy_node_colors.append(safety_colors.get(safety, 'rgba(100, 100, 100, 0.8)'))
    
    remedy_trace = go.Scatter(
        x=remedy_node_x, y=remedy_node_y,
        mode='markers',
        hoverinfo='text',
        text=remedy_node_text,
        marker=dict(
            symbol='square',
            color=remedy_node_colors,
            size=20,
            line=dict(width=1, color='rgba(50, 50, 50, 0.8)')
        ),
        name='Remedies'
    )
    
    # Combine all traces
    all_traces = edge_traces + [symptom_trace, remedy_trace]
    
    # Create figure
    fig = go.Figure(data=all_traces,
                   layout=go.Layout(
                       title='Symptom and Remedy Relationships',
                       titlefont=dict(size=16),
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       legend=dict(
                           x=0,
                           y=1,
                           traceorder="normal",
                           font=dict(family="sans-serif", size=12, color="black"),
                           bgcolor="rgba(255, 255, 255, 0.5)",
                           bordercolor="rgba(0, 0, 0, 0.5)",
                           borderwidth=1
                       )
                   ))
    
    return fig

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
                    
                    # Display symptom summary with severity
                    st.subheader("Symptom Summary")
                    symptom_df = pd.DataFrame([
                        {
                            "Symptom": s["name"].capitalize(),
                            "Severity": s.get("severity", "moderate").capitalize(),
                            "Duration": s.get("duration", "not specified").capitalize(),
                            "Category": s.get("category", "general").capitalize()
                        } for s in symptoms
                    ])
                    st.dataframe(symptom_df, use_container_width=True)
                    
                    # Get relationship data for visualization
                    relationship_data = remedy_matcher.get_symptom_relationships(symptoms)
                    
                    # Create tabs for different views
                    tabs = st.tabs(["Remedies", "Symptom Relationships"])
                    
                    with tabs[0]:
                        for i, remedy in enumerate(safe_remedies):
                            with st.expander(f"🌿 {remedy['name']} (Confidence: {remedy.get('final_score', 0):.2f})"):
                                
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
                                            
                                # Severity-specific recommendations
                                if any(s.get('severity') == 'severe' for s in symptoms):
                                    st.info("**Severe Symptom Recommendations:**")
                                    st.write("For severe symptoms, consider the following:")
                                    st.write("• Use the maximum recommended dosage (within safety limits)")
                                    st.write("• Consider combining with other remedies for enhanced effect")
                                    st.write("• Monitor symptoms closely and seek medical attention if not improving")
                    
                    with tabs[1]:
                        # Create and display the network visualization
                        if relationship_data['nodes']:
                            st.write("This visualization shows the relationships between your symptoms and recommended remedies.")
                            st.write("• **Circle nodes**: Symptoms (size indicates severity, color indicates category)")
                            st.write("• **Square nodes**: Remedies (color indicates safety level)")
                            st.write("• **Green lines**: Treatment relationships (darker lines indicate stronger connections)")
                            st.write("• **Gray lines**: Related symptoms")
                            
                            fig = create_symptom_network_graph(relationship_data)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough relationship data to create visualization.")
                
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