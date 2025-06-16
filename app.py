import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import time
import os
from nltk import ngrams  # For multi-word symptom extraction

# Streamlit app configuration
st.set_page_config(page_title="SympAI", page_icon="🩺", layout="wide")

# Custom CSS for professional look
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput>div>input, .stTextArea>div>textarea {border: 1px solid #4CAF50; border-radius: 5px;}
    .stMarkdown {font-size: 16px;}
</style>
""", unsafe_allow_html=True)

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = None

# SymptomChecker class for backend logic
class SymptomChecker:
    def __init__(self):
        # Debug: Print current working directory
        st.write(f"**Debug**: Current working directory: {os.getcwd()}")
        
        # Load real dataset using absolute path
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(script_dir, "data", "Symptom2Disease.csv")
            st.write(f"**Debug**: Attempting to load dataset from: {dataset_path}")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
            self.conditions_data = pd.read_csv(dataset_path)
            self.conditions_data = self.conditions_data[["text", "label"]].rename(
                columns={"text": "symptoms", "label": "condition"}
            )
            self.conditions_data["symptoms"] = self.conditions_data["symptoms"].str.lower().str.replace(
                r"i have |i am |been | experiencing |feeling ", "", regex=True
            )
            self.conditions_data["probability"] = 1.0 / len(self.conditions_data)
            st.info("Dataset loaded successfully.")
        except Exception as e:
            st.error(f"Error loading dataset: {e}. Using mock dataset.")
            self.conditions_data = pd.DataFrame({
                "symptoms": [
                    "fever cough congestion",
                    "fever sore throat fatigue",
                    "chest pain shortness breath",
                    "headache nausea vomiting",
                    "abdominal pain diarrhea",
                    "joint pain swelling fever",
                    "anxiety palpitations sweating"
                ],
                "condition": [
                    "Common Cold",
                    "Flu",
                    "Pneumonia",
                    "Migraine",
                    "Gastroenteritis",
                    "Arthritis",
                    "Anxiety Attack"
                ],
                "probability": [0.75, 0.20, 0.05, 0.60, 0.50, 0.30, 0.40]
            })
        
        # Initialize ML model
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
        self.model = MultinomialNB()
        try:
            X = self.vectorizer.fit_transform(self.conditions_data["symptoms"])
            y = self.conditions_data["condition"]
            self.model.fit(X, y)
            st.info("ML model trained successfully.")
        except Exception as e:
            st.error(f"Error training model: {e}")

    def get_outbreak_data(self, location):
        """Mock API to simulate fetching outbreak data based on location"""
        location_lower = location.lower()
        if not location:
            return "No location provided: Risk unknown"
        elif "india" in location_lower:
            return f"Mock outbreak data for {location}: Low risk (seasonal flu reported)"
        elif "usa" in location_lower:
            return f"Mock outbreak data for {location}: Medium risk (respiratory infections increasing)"
        elif "europe" in location_lower:
            return f"Mock outbreak data for {location}: High risk (influenza outbreak reported)"
        else:
            return f"Mock outbreak data for {location}: Low risk (no significant outbreaks)"

    def check_red_flags(self, symptoms):
        """Identify urgent symptoms"""
        red_flags = []
        symptoms_lower = symptoms.lower()
        if "fever" in symptoms_lower and "high" in symptoms_lower:
            red_flags.append("High fever > 103°F - Seek immediate care")
        if "chest pain" in symptoms_lower:
            red_flags.append("Chest pain - Seek immediate care")
        if "shortness of breath" in symptoms_lower:
            red_flags.append("Shortness of breath - Seek immediate care")
        return red_flags

    def calculate_risk_score(self, age, lifestyle, conditions):
        """Calculate personalized risk score"""
        score = 0.5
        if age > 60:
            score += 0.15
        if "Smoker" in lifestyle:
            score += 0.1
        if conditions:
            score += 0.05 * len(conditions.split(","))
        return min(score, 1.0)

    def extract_symptoms(self, symptoms):
        """Custom symptom extraction using n-grams"""
        # Expanded stop words to filter out non-symptom words
        stop_words = {
            "i", "have", "a", "and", "the", "with", "in", "on", "at", "of", "to", "for",
            "left", "right", "upper", "lower", "side", "my", "been", "experiencing", "feeling"
        }
        
        # Convert to lowercase and split into words
        words = symptoms.lower().split()
        words = [word for word in words if word not in stop_words]
        
        # Generate unigrams and bigrams
        unigrams = words
        bigrams = [" ".join(gram) for gram in ngrams(words, 2)]
        
        # Combine unigrams and bigrams
        symptoms_extracted = unigrams + bigrams
        
        # Filter for known symptoms by checking against dataset vocabulary
        known_symptoms = set()
        for symptom_str in self.conditions_data["symptoms"]:
            for s in symptom_str.split():
                known_symptoms.add(s)
            for bigram in ngrams(symptom_str.split(), 2):
                known_symptoms.add(" ".join(bigram))
        
        # Keep only symptoms that match known patterns
        symptoms_extracted = [s for s in symptoms_extracted if s in known_symptoms]
        
        # Remove duplicates while preserving order
        seen = set()
        symptoms_extracted = [s for s in symptoms_extracted if not (s in seen or seen.add(s))]
        
        return symptoms_extracted if symptoms_extracted else words

    def analyze_symptoms(self, symptoms, age, gender, location, conditions, lifestyle):
        """Analyze symptoms and return results"""
        # Extract symptoms using custom method
        symptoms_extracted = self.extract_symptoms(symptoms)
        symptoms_str = " ".join(symptoms_extracted)
        st.write(f"**Debug**: Processed symptoms: {symptoms_str}")

        # Predict conditions
        try:
            X_input = self.vectorizer.transform([symptoms_str])
            predicted_probs = self.model.predict_proba(X_input)[0]
            condition_probs = [
                {"name": cond, "probability": prob}
                for cond, prob in zip(self.model.classes_, predicted_probs)
            ]
            st.write(f"**Debug**: Predicted probabilities: {condition_probs}")
        except Exception as e:
            st.error(f"Error predicting conditions: {e}")
            condition_probs = [
                {"name": cond, "probability": prob}
                for cond, prob in zip(self.conditions_data["condition"], self.conditions_data["probability"])
            ]

        outbreak_data = self.get_outbreak_data(location)

        results = {
            "conditions": sorted(condition_probs, key=lambda x: x["probability"], reverse=True)[:3],
            "red_flags": self.check_red_flags(symptoms),
            "risk_score": self.calculate_risk_score(age, lifestyle, conditions),
            "recommendations": ["Rest, hydrate, consider OTC medications"],
            "teleconsult": "https://example-telehealth.com",
            "extracted_symptoms": symptoms_extracted,
            "outbreak_data": outbreak_data
        }
        return results

# Initialize model
try:
    model = SymptomChecker()
except Exception as e:
    st.error(f"Error initializing model: {e}")
    model = None

# Sidebar for user profile
st.sidebar.header("User Profile")
st.sidebar.markdown("Enter your details for personalized analysis")
age = st.sidebar.slider("Age", 0, 100, 30, help="Select your age")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"], help="Select your gender")
location = st.sidebar.text_input("Location (City, Country)", help="Enter your location for outbreak data")
conditions = st.sidebar.text_area("Pre-existing Conditions", help="List conditions, e.g., diabetes, asthma")
lifestyle = st.sidebar.multiselect("Lifestyle Factors", ["Smoker", "Sedentary", "Active", "Vegetarian"], help="Select all that apply")

# Main chat interface
st.title("🩺 SympAI: Your Personal Symptom Checker")
st.markdown("Describe your symptoms in natural language to get personalized health insights. Built for hackathon success!")
user_input = st.text_area("Enter your symptoms (e.g., 'I have a high fever and cough')", height=100)

if st.button("Analyze Symptoms", key="analyze"):
    if not user_input:
        st.error("Please enter symptoms to analyze.")
    elif not model:
        st.error("Model not initialized. Please check setup.")
    else:
        with st.spinner("Processing your symptoms..."):
            time.sleep(1)
            st.session_state.results = model.analyze_symptoms(user_input, age, gender, location, conditions, lifestyle)

        st.subheader("Analysis Results")
        
        if st.session_state.results["extracted_symptoms"]:
            st.write("**Extracted Symptoms**: " + ", ".join(st.session_state.results["extracted_symptoms"]))
        else:
            st.write("**Extracted Symptoms**: None (using raw input)")
        
        st.subheader("Possible Conditions")
        for condition in st.session_state.results["conditions"]:
            st.write(f"{condition['name']}: {condition['probability']*100:.1f}%")

        if st.session_state.results["red_flags"]:
            st.error("🚨 **Red Flag Alerts**: " + ", ".join(st.session_state.results["red_flags"]))
        
        st.subheader("Personalized Risk Score")
        st.write(f"Your risk score: {st.session_state.results['risk_score']*100:.1f}% (based on age, lifestyle, and conditions)")
        
        st.subheader("Outbreak Risk")
        st.write(f"{st.session_state.results['outbreak_data']}")
        
        st.subheader("Recommendations")
        st.write(", ".join(st.session_state.results["recommendations"]))
        
        st.subheader("Next Steps")
        st.markdown(f"[Book a Teleconsultation]({st.session_state.results['teleconsult']})")

        if any(word in user_input.lower() for word in ["anxiety", "stress", "depressed"]):
            st.warning("💡 **Mental Health Note**: You mentioned symptoms related to mental health. Consider reaching out to a professional.")

        st.subheader("Condition Probabilities")
        chart_data = pd.DataFrame(
            data=[condition["probability"] * 100 for condition in st.session_state.results["conditions"]],
            index=[condition["name"] for condition in st.session_state.results["conditions"]],
            columns=["Probability (%)"]
        )
        st.bar_chart(chart_data)

# Export results for hackathon presentation
if st.button("Export Results", key="export"):
    if st.session_state.results is None:
        st.error("No results to export. Please analyze symptoms first.")
    else:
        try:
            results_df = pd.DataFrame(st.session_state.results["conditions"])
            results_df["risk_score"] = st.session_state.results["risk_score"]
            results_df["red_flags"] = ", ".join(st.session_state.results["red_flags"])
            results_df["recommendations"] = ", ".join(st.session_state.results["recommendations"])
            results_df["outbreak_data"] = st.session_state.results["outbreak_data"]
            
            # Save the file
            results_df.to_csv("sympai_results.csv", index=False)
            
            # Debug: Confirm the file exists
            if os.path.exists("sympai_results.csv"):
                st.write("**Debug**: File successfully saved at sympai_results.csv")
            else:
                st.write("**Debug**: File save failed - path issue detected")
            
            # Provide a download button for Streamlit Cloud
            with open("sympai_results.csv", "rb") as file:
                st.download_button(
                    label="Download sympai_results.csv",
                    data=file,
                    file_name="sympai_results.csv",
                    mime="text/csv"
                )
            st.success("Results exported to sympai_results.csv")
        except Exception as e:
            st.error(f"Error exporting results: {e}")
        
# Footer
st.markdown("---")
st.markdown("**SympAI** - Built for Hackathon Success | Powered by AI and Streamlit | June 16, 2025")
