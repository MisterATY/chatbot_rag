"""
Convert Q&A CSV file(s) in data/ to QA JSON format.
- Reads CSV with columns: id, question, answer (no header row).
- Converts all Cyrillic text to Latin (Uzbek mapping + Russian fallback).
- Writes { "pairs": [ {"question": "...", "answer": "..."}, ... ], "source": "..." }.
Usage: python -m scripts.csv_to_qa_json [data/1000 Q&A.csv]
"""

import csv
import json
import os
import sys

# Data folder and default input
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CSV = os.path.join(DATA_FOLDER, "1000 Q&A.csv")

# Uzbek Cyrillic → Latin (official 1995 mapping)
_UZBEK_MAP = {
    "А": "A", "а": "a", "Б": "B", "б": "b", "В": "V", "в": "v", "Г": "G", "г": "g",
    "Д": "D", "д": "d", "Е": "E", "е": "e", "Ё": "Yo", "ё": "yo", "Ж": "J", "ж": "j",
    "З": "Z", "з": "z", "И": "I", "и": "i", "Й": "Y", "й": "y", "К": "K", "к": "k",
    "Л": "L", "л": "l", "М": "M", "м": "m", "Н": "N", "н": "n", "О": "O", "о": "o",
    "П": "P", "п": "p", "Р": "R", "р": "r", "С": "S", "с": "s", "Т": "T", "т": "t",
    "У": "U", "у": "u", "Ф": "F", "ф": "f", "Х": "X", "х": "x", "Ц": "Ts", "ц": "ts",
    "Ч": "Ch", "ч": "ch", "Ш": "Sh", "ш": "sh", "Щ": "Sh", "щ": "sh", "Ъ": "'", "ъ": "'",
    "Ы": "I", "ы": "i", "Ь": "'", "ь": "'", "Э": "E", "э": "e", "Ю": "Yu", "ю": "yu",
    "Я": "Ya", "я": "ya", "Ў": "Oʻ", "ў": "oʻ", "Қ": "Q", "қ": "q", "Ғ": "Gʻ", "ғ": "gʻ",
    "Ҳ": "H", "ҳ": "h",
}

# Russian / other Cyrillic → Latin for any remaining chars
_RUSSIAN_MAP = {
    "А": "A", "а": "a", "Б": "B", "б": "b", "В": "V", "в": "v", "Г": "G", "г": "g",
    "Д": "D", "д": "d", "Е": "E", "е": "e", "Ё": "Yo", "ё": "yo", "Ж": "Zh", "ж": "zh",
    "З": "Z", "з": "z", "И": "I", "и": "i", "Й": "Y", "й": "y", "К": "K", "к": "k",
    "Л": "L", "л": "l", "М": "M", "м": "m", "Н": "N", "н": "n", "О": "O", "о": "o",
    "П": "P", "п": "p", "Р": "R", "р": "r", "С": "S", "с": "s", "Т": "T", "т": "t",
    "У": "U", "у": "u", "Ф": "F", "ф": "f", "Х": "Kh", "х": "kh", "Ц": "Ts", "ц": "ts",
    "Ч": "Ch", "ч": "ch", "Ш": "Sh", "ш": "sh", "Щ": "Shch", "щ": "shch", "Ъ": "'", "ъ": "'",
    "Ы": "Y", "ы": "y", "Ь": "'", "ь": "'", "Э": "E", "э": "e", "Ю": "Yu", "ю": "yu",
    "Я": "Ya", "я": "ya",
}

# Combined: Uzbek first, then Russian for any Cyrillic not in Uzbek
_CYRILLIC_TO_LATIN = {**_RUSSIAN_MAP, **_UZBEK_MAP}
_CYRILLIC_TO_LATIN = str.maketrans(_CYRILLIC_TO_LATIN)


def cyrillic_to_latin(text: str) -> str:
    """Convert Cyrillic text to Latin (Uzbek + Russian)."""
    if not text:
        return text
    return text.translate(_CYRILLIC_TO_LATIN)


def convert_csv_to_qa_json(csv_path: str, out_path: str | None = None, source_name: str | None = None) -> str:
    """
    Read CSV (columns: id, question, answer), convert Cyrillic to Latin, write QA JSON.
    Returns path to written JSON file.
    """
    source_name = source_name or os.path.splitext(os.path.basename(csv_path))[0]
    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), source_name + ".json")

    pairs = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            # CSV: id, question, answer (answer may be quoted and contain newlines/commas → one cell)
            if len(row) == 2:
                q, a = (row[0] or "").strip(), (row[1] or "").strip()
            else:
                q = (row[1] or "").strip()
                a = (row[2] or "").strip()
            if not q and not a:
                continue
            pairs.append({"question": cyrillic_to_latin(q), "answer": cyrillic_to_latin(a)})

    data = {"source": source_name, "pairs": pairs}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.isfile(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)
    out_path = convert_csv_to_qa_json(csv_path)
    print(f"Wrote {len(json.load(open(out_path, encoding='utf-8'))['pairs'])} pairs to {out_path}")


if __name__ == "__main__":
    main()
