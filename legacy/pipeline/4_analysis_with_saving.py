import os
import spacy
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

# --- Configuration ---
INPUT_DIR = "text_output"
OUTPUT_DIR = "analysis_results"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "analysis_output.json")
NUM_TOPICS = 10

def load_documents(input_dir):
    """Loads all text documents from a directory."""
    documents = []
    print(f"📚 Loading documents from '{input_dir}'...")
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    print(f"✅ Loaded {len(documents)} document(s).")
    return documents

def analyze_topics_sklearn(documents):
    """Performs topic modeling and returns the results as a dictionary."""
    print("\n--- Starting Topic Modeling with Scikit-learn (LDA) ---")
    vectorizer = CountVectorizer(stop_words='english', max_df=1.0, min_df=1)
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Store topics in a dictionary instead of just printing
    topic_results = {}
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n✅ LDA Complete. Top words for each of the {NUM_TOPICS} topics:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_list = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topic_key = f"Topic #{topic_idx + 1}"
        topic_results[topic_key] = top_words_list
        print(f"{topic_key}: {' | '.join(top_words_list)}")
        
    return topic_results

def recognize_entities(documents):
    """Performs NER and returns the results as a dictionary."""
    print("\n--- Starting Named Entity Recognition with spaCy ---")
    nlp = spacy.load("en_core_web_sm")
    full_text = " ".join(documents)[:500000]
    
    print("Processing text with spaCy...")
    doc = nlp(full_text)
    
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    gpes = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]
    
    # Store entity counts in a dictionary
    entity_results = {
        "organizations": dict(Counter(orgs).most_common(10)),
        "locations": dict(Counter(gpes).most_common(10))
    }
    
    print("\n✅ NER Complete. Most common entities found and prepared for saving.")
    return entity_results

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created directory: {OUTPUT_DIR}")

    docs = load_documents(INPUT_DIR)
    if docs:
        topic_data = analyze_topics_sklearn(docs)
        entity_data = recognize_entities(docs)
        
        # Combine all results into a single dictionary
        final_output = {
            "topic_modeling_results": topic_data,
            "named_entity_results": entity_data
        }
        
        # Save the final dictionary to a JSON file
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4)
        
        print(f"\n💾 Successfully saved analysis results to '{OUTPUT_FILENAME}'")
    else:
        print("No documents found to analyze.")

    print("\n🎉 NLP analysis and saving complete.")