import os
import json
import spacy
import pandas as pd

# === 1. Load spaCy model ===
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# === 2. Load transparency keywords ===
with open("keywords.json", "r", encoding="utf-8") as f:
    keywords = json.load(f)
eta_keywords = keywords["eta_keywords"]

# === 3. Input / Output folders ===
input_folder = "translated_texts"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# === 4. Process files ===
results = []

for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        parts = file.split("_")
        country = parts[0].upper()
        file_type = parts[1].lower()

        if country in ["AU", "CA", "COMMON", "INDIA", "IRELAND", "UK", "US", "NZ", "HK", "NIGERIA", "SINGAPORE"]:
            law_type = "common"
        else:
            law_type = "civil"

        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
            text = f.read().lower()
            doc = nlp(text)
            words = [
                token.lemma_ for token in doc
                if token.is_alpha and token.lemma_ not in stopwords
            ]

        total_words = len(words)

        # Count each transparency keyword for η
        eta_details = {word: words.count(word) for word in eta_keywords}
        eta_count = sum(eta_details.values())
        eta_ratio = eta_count / total_words if total_words > 0 else 0

        result = {
            "file": file,
            "country": country,
            "law_type": law_type,
            "file_type": file_type,
            "total_words": total_words,
            "eta_count": eta_count,
            "eta_ratio": eta_ratio
        }

        # Save detailed η keyword counts
        for key, val in eta_details.items():
            result[f"eta_{key}"] = val

        results.append(result)

# === 5. Save results ===
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_folder, "raw_eta.csv"), index=False)

grouped = df.groupby(["law_type", "file_type"]).mean(numeric_only=True).reset_index()
grouped.to_csv(os.path.join(output_folder, "grouped_eta.csv"), index=False)

print("✅ η-only pipeline done! Results saved in outputs/")
