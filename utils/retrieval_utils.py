from pinecone import Pinecone


# -----------------------------
# Pinecone Connection
# -----------------------------
def connect_pinecone(api_key, detection_index_name, mitigation_index_name):
    pc = Pinecone(api_key=api_key)

    detection_index = pc.Index(detection_index_name)
    mitigation_index = pc.Index(mitigation_index_name)

    return detection_index, mitigation_index


# -----------------------------
# Detection Search
# -----------------------------
def search_detection_global(index, query_vector, top_k=10):
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]


def search_detection_filtered(index, query_vector, predicted_label, top_k=10):
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"attack_true": {"$eq": predicted_label}}
    )
    return results["matches"]


# -----------------------------
# Confidence-Aware Retrieval
# -----------------------------
def confidence_aware_retrieval(
    index,
    query_vector,
    predicted_label,
    confidence,
    top_k=10
):
    if confidence >= 0.8:
        global_matches = search_detection_global(index, query_vector, top_k)
        filtered_matches = search_detection_filtered(index, query_vector, predicted_label, top_k)

    elif confidence >= 0.5:
        global_matches = search_detection_global(index, query_vector, top_k)
        filtered_matches = search_detection_filtered(index, query_vector, predicted_label, int(top_k / 2))

    else:
        global_matches = search_detection_global(index, query_vector, top_k)
        filtered_matches = []

    return global_matches, filtered_matches


# -----------------------------
# Merge + Deduplicate
# -----------------------------
def merge_and_dedupe_matches(global_matches, filtered_matches):
    merged = []
    seen_ids = set()

    for match in filtered_matches + global_matches:
        if match["id"] not in seen_ids:
            merged.append(match)
            seen_ids.add(match["id"])

    return merged


# -----------------------------
# Dataset Diversity
# -----------------------------
def limit_per_dataset(matches, max_per_dataset=3):
    selected = []
    dataset_counts = {}

    for match in matches:
        dataset = match["metadata"].get("dataset", "UNKNOWN")

        if dataset_counts.get(dataset, 0) < max_per_dataset:
            selected.append(match)
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    return selected


# -----------------------------
# Reranker
# -----------------------------
def build_reranker_pairs(query_text, matches):
    pairs = []

    for match in matches:
        meta = match["metadata"]
        doc_text = meta.get("summary", "") or str(meta)

        pairs.append((query_text, doc_text))

    return pairs


def rerank_matches(query_text, matches, reranker_model, top_k=5):
    if not matches:
        return []

    pairs = build_reranker_pairs(query_text, matches)
    scores = reranker_model.predict(pairs)

    rescored = []

    for match, score in zip(matches, scores):
        rescored.append({
            "id": match["id"],
            "score": match["score"],
            "metadata": match["metadata"],
            "rerank_score": float(score)
        })

    rescored = sorted(rescored, key=lambda x: x["rerank_score"], reverse=True)

    return rescored[:top_k]


# -----------------------------
# Mitigation Retrieval
# -----------------------------
def search_mitigation_filtered(index, query_vector, predicted_label, top_k=8):
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"attack_name": {"$eq": predicted_label}}
    )
    return results["matches"]


def search_mitigation_global(index, query_vector, top_k=8):
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]


def retrieve_mitigations(index, query_vector, predicted_label):
    matches = search_mitigation_filtered(index, query_vector, predicted_label)

    if len(matches) == 0:
        matches = search_mitigation_global(index, query_vector)

    return matches


# -----------------------------
# Formatting (LLM input)
# -----------------------------
def format_historical_cases(cases, predicted_label):
    lines = ["SIMILAR HISTORICAL CASES"]

    for i, case in enumerate(cases, start=1):
        meta = case["metadata"]
        attack = meta.get("attack_true", "N/A")

        relation = "CONSISTENT" if attack == predicted_label else "CONFLICTING"

        lines.append(f"\nCase {i} ({relation}):")
        lines.append(f"Attack: {attack}")
        lines.append(f"Dataset: {meta.get('dataset')}")
        lines.append(f"Confidence: {meta.get('confidence')}")
        lines.append("Behavior:")
        lines.append(meta.get("summary", ""))

    return "\n".join(lines)


def format_mitigation_items(items):
    lines = ["MITIGATION KNOWLEDGE"]

    for i, item in enumerate(items, start=1):
        meta = item["metadata"]

        lines.append(f"\nMitigation Item {i}:")
        lines.append(f"Attack Name: {meta.get('attack_name', 'N/A')}")
        lines.append(f"Framework: {meta.get('framework', 'N/A')}")
        lines.append(f"Section: {meta.get('section', 'N/A')}")
        lines.append(f"ID: {meta.get('id', 'N/A')}")
        lines.append(f"Name: {meta.get('name', 'N/A')}")
        lines.append("Content:")
        lines.append(meta.get("text", ""))

    return "\n".join(lines)