"""
Convert 1000 Q&A.csv to qa1.json in the same format as qa_example.json.
Usage: run from project root: python data/csv_to_qa_json.py
"""
import csv
import json
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "1000 Q&A.csv")
JSON_PATH = os.path.join(os.path.dirname(__file__), "qa1.json")

def main():
    pairs = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            _id, question, answer = row[0], row[1].strip(), row[2].strip()
            if not question and not answer:
                continue
            pairs.append({"question": question, "answer": answer})

    out = {"source": "qa1.json", "pairs": pairs}
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pairs)} Q&A pairs to {JSON_PATH}")

if __name__ == "__main__":
    main()
