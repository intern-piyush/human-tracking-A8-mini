import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_INDEX_PATH = os.path.join(MODELS_DIR, "model_index.json")
SUPPORTED_MODEL_EXTENSIONS = {".pt", ".onnx", ".engine", ".torchscript"}

OBJECT_SYNONYMS = {
    "human": "person",
    "people": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "pedestrian": "person",
    "car": "car",
    "auto": "car",
    "automobile": "car",
    "bike": "bicycle",
    "cycle": "bicycle",
    "motorbike": "motorcycle",
    "bike rider": "person",
    "truck": "truck",
    "lorry": "truck",
    "plane": "airplane",
    "aeroplane": "airplane",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "phone": "cell phone",
    "tv": "tv",
    "television": "tv",
}


def ensure_models_dir() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)


def normalize_label(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.endswith("s") and len(cleaned) > 3:
        cleaned = cleaned[:-1]
    return OBJECT_SYNONYMS.get(cleaned, cleaned)


def collect_model_files() -> List[str]:
    ensure_models_dir()
    model_files: List[str] = []
    seen_signatures = set()

    def add_model_file(path: str) -> None:
        signature = None
        try:
            stat = os.stat(path)
            signature = (os.path.basename(path).lower(), stat.st_size, stat.st_mtime)
        except OSError:
            signature = (os.path.basename(path).lower(), path)

        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        model_files.append(path)

    for entry in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, entry)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(entry)[1].lower() in SUPPORTED_MODEL_EXTENSIONS:
            add_model_file(path)

    for entry in sorted(os.listdir(PROJECT_DIR)):
        path = os.path.join(PROJECT_DIR, entry)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(entry)[1].lower() in SUPPORTED_MODEL_EXTENSIONS and path not in model_files:
            add_model_file(path)

    return model_files


def _build_model_inventory(model_files: List[str]) -> List[Dict[str, object]]:
    inventory: List[Dict[str, object]] = []
    for path in model_files:
        stat = os.stat(path)
        inventory.append(
            {
                "path": path,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return inventory


def _extract_names(model) -> List[str]:
    names = model.names
    if isinstance(names, dict):
        ordered = [names[key] for key in sorted(names)]
    elif isinstance(names, (list, tuple)):
        ordered = list(names)
    else:
        ordered = []
    return [str(name) for name in ordered]


def index_model(model_path: str) -> Dict[str, object]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    classes = _extract_names(model)
    return {
        "name": os.path.basename(model_path),
        "path": model_path,
        "classes": classes,
    }


def rebuild_model_index() -> List[Dict[str, object]]:
    ensure_models_dir()
    model_files = collect_model_files()
    entries: List[Dict[str, object]] = []

    for model_path in model_files:
        try:
            entries.append(index_model(model_path))
        except Exception as exc:
            entries.append(
                {
                    "name": os.path.basename(model_path),
                    "path": model_path,
                    "classes": [],
                    "error": str(exc),
                }
            )

    payload = {
        "models_dir": MODELS_DIR,
        "model_count": len(entries),
        "inventory": _build_model_inventory(model_files),
        "models": entries,
    }
    with open(MODEL_INDEX_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return entries


def model_index_needs_refresh() -> bool:
    ensure_models_dir()
    if not os.path.exists(MODEL_INDEX_PATH):
        return True

    with open(MODEL_INDEX_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    saved_inventory = payload.get("inventory", [])
    current_inventory = _build_model_inventory(collect_model_files())
    return saved_inventory != current_inventory


def load_model_index(refresh: bool = False) -> List[Dict[str, object]]:
    ensure_models_dir()
    if refresh or model_index_needs_refresh():
        return rebuild_model_index()

    with open(MODEL_INDEX_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("models", []))


def find_class_matches(query: str, classes: List[str]) -> List[Tuple[float, str]]:
    normalized_query = normalize_label(query)
    matches: List[Tuple[float, str]] = []

    for class_name in classes:
        normalized_class = normalize_label(class_name)
        if not normalized_class:
            continue

        score = SequenceMatcher(None, normalized_query, normalized_class).ratio()
        if normalized_query == normalized_class:
            score += 1.0
        elif normalized_query in normalized_class or normalized_class in normalized_query:
            score += 0.35

        if score >= 0.55:
            matches.append((score, class_name))

    matches.sort(key=lambda item: item[0], reverse=True)
    return matches


def _model_aliases(model_entry: Dict[str, object]) -> List[str]:
    aliases: List[str] = []
    name = str(model_entry.get("name", ""))
    path = str(model_entry.get("path", ""))

    for raw_value in (name, os.path.splitext(name)[0], os.path.splitext(os.path.basename(path))[0]):
        normalized = normalize_label(raw_value)
        if normalized and normalized not in aliases:
            aliases.append(normalized)

    return aliases


def find_model_matches(query: str, models: List[Dict[str, object]]) -> List[Tuple[float, Dict[str, object]]]:
    normalized_query = normalize_label(query)
    matches: List[Tuple[float, Dict[str, object]]] = []

    for model_entry in models:
        best_score = 0.0
        for alias in _model_aliases(model_entry):
            score = SequenceMatcher(None, normalized_query, alias).ratio()
            if normalized_query == alias:
                score += 1.0
            elif normalized_query in alias or alias in normalized_query:
                score += 0.35
            best_score = max(best_score, score)

        if best_score >= 0.55:
            matches.append((best_score, model_entry))

    matches.sort(key=lambda item: item[0], reverse=True)
    return matches


def choose_best_model(query: str, refresh: bool = False) -> Tuple[Optional[Dict[str, object]], Optional[str], List[Dict[str, object]]]:
    models = load_model_index(refresh=refresh)
    best_model: Optional[Dict[str, object]] = None
    best_class: Optional[str] = None
    best_score = 0.0
    ranked: List[Dict[str, object]] = []

    model_matches = find_model_matches(query, models)
    if model_matches:
        model_score, model_entry = model_matches[0]
        ranked.append(
            {
                "model_name": model_entry.get("name"),
                "model_path": model_entry.get("path"),
                "class_name": None,
                "score": round(model_score, 4),
            }
        )
        best_model = model_entry
        best_score = model_score

    for model_entry in models:
        matches = find_class_matches(query, list(model_entry.get("classes", [])))
        if not matches:
            continue

        score, class_name = matches[0]
        ranked.append(
            {
                "model_name": model_entry.get("name"),
                "model_path": model_entry.get("path"),
                "class_name": class_name,
                "score": round(score, 4),
            }
        )
        if score > best_score:
            best_score = score
            best_model = model_entry
            best_class = class_name

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return best_model, best_class, ranked
