#############################################################################
# File: mab_ttl_id_decoder.py
#
# Description:
#   Decodes MemoryAgentBench Test-Time Learning entity IDs into readable titles.
#
#   - Fetches the benchmark entity-to-ID mapping through the Hugging Face cache/download path.
#   - Converts DBpedia-style entity names into readable titles.
#   - Adds decoded answer fields while preserving the original TTL records.
#############################################################################

import json
from pathlib import Path
from urllib.parse import unquote

from huggingface_hub import hf_hub_download

INPUT_PATH = Path("separated_splits/Test_Time_Learning.json")
OUTPUT_PATH = Path("separated_splits/Test_Time_Learning_decoded.json")

REPO_ID = "ai-hyz/MemoryAgentBench"


def dbpedia_uri_to_title(uri: str) -> str:
    uri = uri.strip("<>")
    raw_title = uri.rsplit("/", 1)[-1]
    return unquote(raw_title).replace("_", " ")


def load_entity_mapping() -> dict[str, str]:
    entity_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="entity2id.json",
    )

    with open(entity_path, "r", encoding="utf-8") as f:
        entity2id = json.load(f)

    return {str(v): k for k, v in entity2id.items()}


def enrich_test_time_learning() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    id2entity = load_entity_mapping()

    enriched_rows = []

    for row in rows:
        row = dict(row)

        decoded_answers = []
        decoded_answer_titles = []
        qa_pairs = []

        questions = row.get("questions", [])
        answers = row.get("answers", [])

        metadata = row.get("metadata", {})
        qa_pair_ids = metadata.get("qa_pair_ids") or [None] * len(questions)
        question_ids = metadata.get("question_ids") or [None] * len(questions)
        question_types = metadata.get("question_types") or [None] * len(questions)
        question_dates = metadata.get("question_dates") or [None] * len(questions)

        for i, (question, answer_ids) in enumerate(zip(questions, answers)):
            answer_ids = [str(a) for a in answer_ids]

            uris = [id2entity.get(answer_id) for answer_id in answer_ids]

            titles = [
                dbpedia_uri_to_title(uri) if uri is not None else None for uri in uris
            ]

            decoded_answers.append(uris)
            decoded_answer_titles.append(titles)

            qa_pairs.append(
                {
                    "qa_pair_id": qa_pair_ids[i] if i < len(qa_pair_ids) else None,
                    "question_id": question_ids[i] if i < len(question_ids) else None,
                    "question_type": (
                        question_types[i] if i < len(question_types) else None
                    ),
                    "question_date": (
                        question_dates[i] if i < len(question_dates) else None
                    ),
                    "question": question,
                    "answer_ids": answer_ids,
                    "answer_uris": uris,
                    "answer_titles": titles,
                }
            )

        row["decoded_answers"] = decoded_answers
        row["decoded_answer_titles"] = decoded_answer_titles
        row["qa_pairs_decoded"] = qa_pairs

        enriched_rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(enriched_rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    enrich_test_time_learning()
