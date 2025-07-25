import os
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

# --- Configuration ---
INPUT_DIR = "text_output"
NUM_TOPICS = 10 # You can adjust the number of topics you want to find

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
    """Performs topic modeling using Scikit-learn's LDA."""
    print("\n--- Starting Topic Modeling with Scikit-learn (LDA) ---")
    
    # Step 1: Create a document-term matrix.
    # We use stop_words='english' to remove common English words.
    # max_df=0.9 filters out words that appear in more than 90% of documents.
    # min_df=2 filters out words that appear in only one document.
    vectorizer = CountVectorizer(stop_words='english', max_df=1.0, min_df=1)
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    # Step 2: Run the LDA model.
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Step 3: Display the topics.
    print(f"\n✅ LDA Complete. Top words for each of the {NUM_TOPICS} topics:")
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = " | ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        print(f"Topic #{topic_idx + 1}: {top_words}")

def recognize_entities(documents):
    """Performs Named Entity Recognition on the documents (unchanged)."""
    print("\n--- Starting Named Entity Recognition with spaCy ---")
    nlp = spacy.load("en_core_web_sm")
    full_text = " ".join(documents)[:500000]
    
    print("Processing text with spaCy (this may take a moment)...")
    doc = nlp(full_text)
    
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    gpes = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]
    
    print("\n✅ NER Complete. Most common entities found:")
    print("\n--- Top 10 Organizations (ORG) ---")
    for item, count in Counter(orgs).most_common(10):
        print(f"{item}: {count}")

    print("\n--- Top 10 Locations (GPE) ---")
    for item, count in Counter(gpes).most_common(10):
        print(f"{item}: {count}")

if __name__ == "__main__":
    docs = load_documents(INPUT_DIR)
    if docs:
        # Perform Topic Modeling with the new, stable method
        analyze_topics_sklearn(docs)
        
        # Perform Named Entity Recognition
        recognize_entities(docs)
    else:
        print("No documents found to analyze.")

    print("\n🎉 NLP analysis complete.")