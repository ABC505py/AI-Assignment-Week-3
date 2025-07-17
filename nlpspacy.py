# nlp_spacy.py
import spacy

nlp = spacy.load("en_core_web_sm")

review = "I love the sound quality of Bose headphones. The Sony ones are also decent."

# NER
doc = nlp(review)
print("Named Entities:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Rule-based Sentiment
if "love" in review.lower() or "great" in review.lower():
    print("Sentiment: Positive 😊")
elif "hate" in review.lower() or "bad" in review.lower():
    print("Sentiment: Negative 😠")
else:
    print("Sentiment: Neutral 😐")
