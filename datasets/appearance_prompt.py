import json
import re
from collections import Counter
from pathlib import Path


COARSE_SUPERCATEGORY_BY_PREFIX = {
    "i": "indication",
    "p": "prohibition",
    "w": "warning",
}
COARSE_GROUP_NAMES = ("indication", "prohibition", "warning")
COARSE_GROUP_TO_ID = {name: idx for idx, name in enumerate(COARSE_GROUP_NAMES)}
INVALID_COARSE_GROUP_ID = -1
COARSE_GROUP_ALIASES = {
    "indication": "indication",
    "mandatory": "indication",
    "information": "indication",
    "guide": "indication",
    "guidance": "indication",
    "prohibition": "prohibition",
    "prohibitory": "prohibition",
    "regulatory": "prohibition",
    "restriction": "prohibition",
    "termination": "prohibition",
    "yield": "prohibition",
    "stop": "prohibition",
    "priority": "prohibition",
    "warning": "warning",
}
STRUCTURED_TEXT_SOURCES = {
    "attr_text",
    "disc_text",
    "visual_cue_text",
    "train_text",
    "infer_text",
    "full_text",
    "legacy_description",
}

FACB_COMPOSITIONAL_FACET_NAMES = (
    "semantic",
    "family",
    "appearance",
    "symbol",
    "value",
)

FACB_SHARED_FACET_NAMES = (
    "group",
    "shape",
    "symbol",
    "value",
)

_FACB_FAMILY_PATTERNS = (
    (r"\bminimum\s+speed\s+limit\b", "minimum-speed traffic sign"),
    (r"\bend\s+of\s+speed\s+limit\b", "end-of-speed-limit traffic sign"),
    (r"\bspeed\s+limit\b", "speed-limit traffic sign"),
    (r"\bheight\s+limit\b", "height-limit traffic sign"),
    (r"\bweight\s+limit\b", "weight-limit traffic sign"),
    (r"\blimit\s+axle\s+load\b", "axle-load-limit traffic sign"),
    (r"\bwidth\s+limit\b", "width-limit traffic sign"),
    (r"\b(?:no|prohibit(?:ed|ion)?(?:\s+of)?)\b.*\bturn", "turn-prohibition traffic sign"),
    (r"\bturn\b", "turn-direction traffic sign"),
    (r"\bno\s+overtaking\b", "overtaking-prohibition traffic sign"),
    (r"\bno\s+parking\b", "parking-prohibition traffic sign"),
    (r"\bpedestrian", "pedestrian traffic sign"),
    (r"\bcycl", "cycle traffic sign"),
)

_FACB_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    ".": "decimal point",
}


def _facb_family_text(semantic_text, meta):
    normalized = _normalize_text(semantic_text).lower()
    for pattern, family in _FACB_FAMILY_PATTERNS:
        if re.search(pattern, normalized):
            return family
    super_group = (
        _normalize_text(meta.get("super_group"))
        or _normalize_text(meta.get("coarse_group"))
        or "traffic-sign"
    )
    return f"{super_group} traffic sign"


def _facb_value_text(semantic_text):
    text = _normalize_text(semantic_text)
    match = re.search(r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        return ""
    value = match.group(1)
    lowered = text.lower()
    if re.search(r"km\s*/?\s*h", lowered):
        unit = "kilometers per hour"
    elif re.search(r"\b(?:ton|tonne|t)\b", lowered):
        unit = "tonnes"
    elif re.search(r"\b(?:meter|metre|m)\b", lowered):
        unit = "meters"
    else:
        unit = ""
    digit_sequence = " followed by ".join(
        f"digit {_FACB_DIGIT_WORDS[ch]}"
        if ch != "."
        else _FACB_DIGIT_WORDS[ch]
        for ch in value
    )
    unit_text = f"; unit {unit}" if unit else ""
    return f"numeric composition: {digit_sequence}; value {value}{unit_text}"


def _facb_numeric_value(semantic_text):
    match = re.search(
        r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)",
        _normalize_text(semantic_text),
    )
    return "" if match is None else match.group(1)


def _facb_shape_text(meta):
    """Return a compact, reusable color/shape label instead of a long prompt."""

    appearance_group = _normalize_text(meta.get("appearance_group")).lower()
    source = " ".join(appearance_group.replace("_", " ").split())
    source = re.sub(
        r"\b(?:regulatory|prohibitory|warning|mandatory|indication|information)\b",
        "",
        source,
    )
    source = " ".join(source.split())
    if source and source != "unknown":
        return f"{source} traffic sign"

    attr_text = _normalize_text(meta.get("attr_text")).lower()
    colors = re.findall(
        r"\b(?:red|blue|yellow|white|black|green|orange)\b", attr_text
    )
    shapes = re.findall(
        r"\b(?:circular|circle|triangular|triangle|rectangular|rectangle|"
        r"octagonal|octagon|diamond|square)\b",
        attr_text,
    )
    tokens = list(dict.fromkeys(colors + shapes))
    return f"{' '.join(tokens)} traffic sign" if tokens else "traffic sign"


_FACB_SHARED_SYMBOL_PATTERNS = (
    (r"\bu[ -]?turn\b", "u-turn arrow"),
    (r"\bleft\b", "left-directed symbol"),
    (r"\bright\b", "right-directed symbol"),
    (r"\bstraight\b", "straight-directed symbol"),
    (r"\broundabout\b|\bround the island\b", "roundabout arrows"),
    (r"\bpedestrian|\bwalk\b", "pedestrian symbol"),
    (r"\bbicy|\bcycl", "bicycle symbol"),
    (r"\bbus(?:es)?\b", "bus symbol"),
    (r"\btruck|\bfreight\b", "truck symbol"),
    (r"\banimal\b", "animal symbol"),
    (r"\bfalling rocks?\b", "falling-rocks symbol"),
    (r"\bconstruction\b", "construction-worker symbol"),
    (r"\bslippery\b", "skid-car symbol"),
    (r"\bparking\b", "parking symbol"),
    (r"\bno entry\b|\bprohibition of entry\b", "no-entry symbol"),
)


def _facb_composite_numeric_symbol_text(value):
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    if "." in normalized:
        return f"numeric symbol {normalized}"
    return f"digit {normalized}"


def _facb_symbol_texts(
    semantic_text,
    meta,
    include_composite_numeric=False,
    composite_numeric_only=False,
):
    """Build shared symbol labels; numeric signs supervise visible digits.

    When requested, numeric signs also supervise a composite symbol such as
    ``digit 90``.  This preserves single-digit sharing while giving the verifier
    an explicit target for confusable multi-digit signs such as 80/90.
    """

    value = _facb_numeric_value(semantic_text)
    if value:
        symbols = []
        if include_composite_numeric:
            composite = _facb_composite_numeric_symbol_text(value)
            if composite:
                symbols.append(composite)
            if composite_numeric_only:
                return symbols
        for char in value:
            if char == ".":
                label = "decimal point"
            else:
                label = f"digit {char}"
            if label not in symbols:
                symbols.append(label)
        return symbols

    searchable = " ".join(
        part
        for part in (
            _normalize_text(semantic_text),
            _normalize_text(meta.get("visual_cue_text")),
            _normalize_text(meta.get("disc_text")),
        )
        if part
    ).lower()
    symbols = [
        label
        for pattern, label in _FACB_SHARED_SYMBOL_PATTERNS
        if re.search(pattern, searchable)
    ]
    if symbols:
        return list(dict.fromkeys(symbols))
    fallback = (
        _normalize_text(meta.get("visual_cue_text"))
        or _normalize_text(meta.get("disc_text"))
        or _normalize_text(semantic_text)
    )
    return [fallback] if fallback else []


def build_facb_shared_facet_schema(
    raw_label_map,
    category_meta_path=None,
    facet_names=None,
    include_composite_symbol=False,
    composite_symbol_only=False,
):
    """Build shared facet vocabularies and class-to-facet supervision.

    Unlike class-aligned facet prompts, this schema deduplicates attributes
    across classes.  A rare class therefore reuses supervision accumulated by
    every class that contains the same family, shape, symbol, or digit.
    """

    if facet_names is None:
        facet_names = FACB_SHARED_FACET_NAMES
    facet_names = tuple(
        name
        for name in facet_names
        if str(name).strip().lower() in set(FACB_SHARED_FACET_NAMES)
    )
    if not facet_names:
        raise ValueError("FACB shared facet schema needs at least one facet.")

    raw_label_map = normalize_raw_label_map(raw_label_map)
    builder = AppearancePromptBuilder(
        enabled=False,
        text_source="semantic",
        category_meta_path=category_meta_path,
    )
    facet_rows = []
    for idx in range(len(raw_label_map)):
        label_entry = raw_label_map.get(str(idx))
        meta = builder._resolve_meta(idx, label_entry)
        semantic = (
            _normalize_text(meta.get("semantic"))
            or _normalize_text(meta.get("label"))
            or _normalize_text(meta.get("name"))
            or builder._default_text(label_entry)
        )
        value = _facb_numeric_value(semantic)
        facet_rows.append(
            {
                "group": [_facb_family_text(semantic, meta)],
                "shape": [_facb_shape_text(meta)],
                "symbol": _facb_symbol_texts(
                    semantic,
                    meta,
                    include_composite_numeric=include_composite_symbol,
                    composite_numeric_only=composite_symbol_only,
                ),
                "value": [f"value {value}"] if value else [],
            }
        )

    vocabularies = {name: [] for name in facet_names}
    vocab_to_id = {name: {} for name in facet_names}
    class_target_ids = []
    for row in facet_rows:
        target_row = []
        for facet_name in facet_names:
            target_ids = []
            for text in row.get(facet_name, []):
                text = " ".join(_normalize_text(text).split())
                if not text:
                    continue
                key = text.lower()
                if key not in vocab_to_id[facet_name]:
                    vocab_to_id[facet_name][key] = len(vocabularies[facet_name])
                    vocabularies[facet_name].append(text)
                target_id = vocab_to_id[facet_name][key]
                if target_id not in target_ids:
                    target_ids.append(target_id)
            target_row.append(target_ids)
        class_target_ids.append(target_row)

    return {
        "facet_names": list(facet_names),
        "vocab_texts": [
            vocabularies[name] for name in facet_names
        ],
        "class_target_ids": class_target_ids,
        "class_facet_texts": [
            [row[name] for name in facet_names]
            for row in facet_rows
        ],
    }


def build_facb_compositional_facet_texts(
    raw_label_map,
    category_meta_path=None,
):
    """Build reproducible class-anchor + four-facet texts for FACB.

    The output is aligned to global category id and has five slots per class:
    semantic anchor, functional family, appearance, symbol, and numeric value.
    Empty value slots are intentional and are masked by the prototype encoder.
    No image-level annotations are consumed.
    """

    raw_label_map = normalize_raw_label_map(raw_label_map)
    builder = AppearancePromptBuilder(
        enabled=False,
        text_source="semantic",
        category_meta_path=category_meta_path,
    )
    facets_by_class = []
    for idx in range(len(raw_label_map)):
        label_entry = raw_label_map.get(str(idx))
        meta = builder._resolve_meta(idx, label_entry)
        semantic = (
            _normalize_text(meta.get("semantic"))
            or _normalize_text(meta.get("label"))
            or _normalize_text(meta.get("name"))
            or builder._default_text(label_entry)
        )
        family = _facb_family_text(semantic, meta)
        appearance = (
            _normalize_text(meta.get("facet_shape"))
            or _normalize_text(meta.get("attr_text"))
            or builder._build_structured_attr_fallback(meta)
        )
        symbol = (
            _normalize_text(meta.get("facet_symbol"))
            or _normalize_text(meta.get("visual_cue_text"))
            or _normalize_text(meta.get("disc_text"))
        )
        value = _facb_value_text(semantic)
        facets_by_class.append(
            [
                semantic,
                f"traffic sign family: {family}" if family else "",
                f"sign appearance: {appearance}" if appearance else "",
                f"painted symbol: {symbol}" if symbol else "",
                value,
            ]
        )
    return facets_by_class

APPEARANCE_TEMPLATES = {
    "indication": "mandatory traffic sign blue circular sign with white number",
    "prohibition": "prohibitory traffic sign red-bordered circular sign with white center",
    "warning": "warning traffic sign triangular sign with white background and red border",
}


def _normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _strip_trailing_caption_punct(text):
    value = _normalize_text(text)
    while value and value[-1] in ".!?;:":
        value = value[:-1].rstrip()
    return value


def _join_nonempty_texts(*parts):
    normalized_parts = []
    for part in parts:
        value = _strip_trailing_caption_punct(part)
        if value:
            normalized_parts.append(value)
    return ", ".join(normalized_parts)


class AppearancePromptBuilder:
    """Builds optional appearance-aware prompts from category metadata."""

    def __init__(
        self,
        enabled=False,
        text_source="semantic",
        separator=", ",
        category_meta_path=None,
    ):
        self.enabled = enabled
        self.text_source = text_source
        self.separator = separator
        self.category_meta_path = category_meta_path
        self.category_meta = self._load_category_meta(category_meta_path)

    def _load_category_meta(self, category_meta_path):
        if not category_meta_path:
            return {}

        meta_path = Path(category_meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Category metadata file not found: {category_meta_path}")

        with meta_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "categories" in data:
            categories = data["categories"]
        elif isinstance(data, dict):
            return {str(key): value for key, value in data.items()}
        else:
            raise ValueError(f"Unsupported category metadata format: {category_meta_path}")

        meta = {}
        for item in categories:
            meta[str(item["id"])] = item
        return meta

    def _default_text(self, label_entry):
        if isinstance(label_entry, dict):
            for key in (
                "infer_text",
                "train_text",
                "disc_text",
                "semantic",
                "label",
                "name",
                "description",
                "text",
            ):
                text = _normalize_text(label_entry.get(key))
                if text:
                    return text
            return ""
        return _normalize_text(label_entry)

    def _resolve_meta(self, label_id, label_entry):
        meta = {}
        if isinstance(label_entry, dict):
            meta.update(label_entry)
        for key, value in self.category_meta.get(str(label_id), {}).items():
            meta.setdefault(key, value)
        return meta

    def _resolve_base_text(self, label_entry, meta):
        description_text = _normalize_text(meta.get("description"))
        semantic_text = _normalize_text(meta.get("semantic"))
        label_text = _normalize_text(meta.get("label"))
        name_text = _normalize_text(meta.get("name"))
        attr_text = _normalize_text(meta.get("attr_text"))
        disc_text = _normalize_text(meta.get("disc_text"))
        visual_cue_text = _normalize_text(meta.get("visual_cue_text"))
        train_text = _normalize_text(meta.get("train_text"))
        infer_text = _normalize_text(meta.get("infer_text"))
        full_text = _normalize_text(meta.get("full_text"))
        legacy_description = _normalize_text(meta.get("legacy_description"))
        natural_name_text = label_text or semantic_text or name_text
        semantic_or_label_text = semantic_text or label_text or name_text
        if self.text_source == "attr_text":
            return attr_text or self._build_structured_attr_fallback(meta)
        elif self.text_source == "disc_text":
            return disc_text or infer_text or semantic_or_label_text
        elif self.text_source == "visual_cue_text":
            return visual_cue_text
        elif self.text_source == "train_text":
            return train_text or _join_nonempty_texts(disc_text or infer_text or semantic_or_label_text, visual_cue_text)
        elif self.text_source == "infer_text":
            return infer_text or disc_text or semantic_or_label_text
        elif self.text_source == "full_text":
            return full_text or _join_nonempty_texts(
                train_text or disc_text or infer_text or semantic_or_label_text,
                attr_text,
            )
        elif self.text_source == "legacy_description":
            return legacy_description or description_text
        if self.text_source == "description":
            if description_text:
                return description_text
        elif self.text_source == "semantic_with_description":
            if semantic_or_label_text and description_text:
                return f"{_strip_trailing_caption_punct(semantic_or_label_text)}, {description_text}"
            if semantic_or_label_text or description_text:
                return semantic_or_label_text or description_text
        elif self.text_source == "semantic_with_visual":
            if semantic_or_label_text and visual_cue_text:
                return f"{_strip_trailing_caption_punct(semantic_or_label_text)}: {visual_cue_text}"
            if semantic_or_label_text or visual_cue_text:
                return semantic_or_label_text or visual_cue_text
        elif self.text_source == "description_with_semantic":
            if description_text and semantic_or_label_text:
                return f"{_strip_trailing_caption_punct(description_text)}, {semantic_or_label_text}"
            if description_text or semantic_or_label_text:
                return description_text or semantic_or_label_text
        elif self.text_source == "description_or_semantic":
            preferred = description_text or semantic_or_label_text
            if preferred:
                return preferred
        elif self.text_source == "name":
            if natural_name_text:
                return natural_name_text
        elif self.text_source == "semantic":
            if semantic_or_label_text:
                return semantic_or_label_text
        elif self.text_source == "semantic_or_name":
            preferred = semantic_or_label_text
            if preferred:
                return preferred
        elif self.text_source == "semantic_with_name":
            if semantic_or_label_text and name_text and name_text != semantic_or_label_text:
                return f"{semantic_or_label_text} ({name_text})"
            if semantic_or_label_text or name_text:
                return semantic_or_label_text or name_text
        elif self.text_source == "name_with_semantic":
            if natural_name_text and semantic_or_label_text and natural_name_text != semantic_or_label_text:
                return f"{natural_name_text}: {semantic_or_label_text}"
            if natural_name_text or semantic_or_label_text:
                return natural_name_text or semantic_or_label_text
        elif self.text_source == "name_with_description":
            if natural_name_text and description_text:
                return f"{_strip_trailing_caption_punct(natural_name_text)}: {description_text}"
            if description_text or natural_name_text:
                return description_text or natural_name_text
        elif self.text_source == "name_with_visual_cue":
            if natural_name_text and visual_cue_text:
                return f"{_strip_trailing_caption_punct(natural_name_text)}: {visual_cue_text}"
            if visual_cue_text or natural_name_text:
                return visual_cue_text or natural_name_text

        return self._default_text(label_entry)

    def _canonical_group_name(self, value):
        normalized = _normalize_text(value).lower()
        if not normalized:
            return ""
        return COARSE_GROUP_ALIASES.get(normalized, "")

    def _build_structured_attr_fallback(self, meta):
        coarse_group = self._infer_supercategory(meta)
        return APPEARANCE_TEMPLATES.get(coarse_group, "")

    def _infer_supercategory(self, meta):
        explicit_group = (
            _normalize_text(meta.get("coarse_group"))
            or _normalize_text(meta.get("coarse_supercategory"))
            or _normalize_text(meta.get("super_group"))
        )
        canonical_explicit = self._canonical_group_name(explicit_group)
        if canonical_explicit:
            return canonical_explicit

        appearance_group = _normalize_text(meta.get("appearance_group")).lower()
        if appearance_group:
            for alias, canonical_name in COARSE_GROUP_ALIASES.items():
                if alias in appearance_group:
                    return canonical_name

        name = _normalize_text(meta.get("name")).lower()
        if not name:
            return ""
        return COARSE_SUPERCATEGORY_BY_PREFIX.get(name[:1], "")

    def get_group_name(self, label_id, label_entry):
        meta = self._resolve_meta(label_id, label_entry)
        return self._infer_supercategory(meta)

    def get_group_id(self, label_id, label_entry, default=INVALID_COARSE_GROUP_ID):
        group_name = self.get_group_name(label_id, label_entry)
        return COARSE_GROUP_TO_ID.get(group_name, default)

    def get_appearance_group(self, label_id, label_entry):
        meta = self._resolve_meta(label_id, label_entry)
        appearance_group = _normalize_text(meta.get("appearance_group"))
        if appearance_group:
            return appearance_group
        return self.get_group_name(label_id, label_entry)

    def get_label_alias(self, label_id, label_entry):
        meta = self._resolve_meta(label_id, label_entry)
        alias = _normalize_text(meta.get("name"))
        if alias:
            return alias
        return str(label_id)

    def build(self, label_id, label_entry):
        meta = self._resolve_meta(label_id, label_entry)
        base_text = self._resolve_base_text(label_entry, meta)
        if not self.enabled:
            return base_text
        if not base_text:
            return base_text
        if self.text_source in STRUCTURED_TEXT_SOURCES:
            return base_text

        coarse_supercategory = self._infer_supercategory(meta)
        appearance = APPEARANCE_TEMPLATES.get(coarse_supercategory)
        if not appearance:
            return base_text

        return f"{base_text}{self.separator}{appearance}"


def _fine_group_key_from_name(name):
    """Derive a fine-grained group key from a TT100K class name.

    TT100K names encode a coarse symbol family in the leading letters and the
    fine instance in the trailing number, e.g.:
        pl40 / pl60 / pl100 -> "pl"   (speed limits, identical sign apart from the number)
        ph4   / ph4.5       -> "ph"   (height limits)
        pne                 -> "pne"  (no trailing number -> name unchanged)
        i5    / i10         -> "i"
    Stripping the trailing numeric suffix groups exactly the classes that are
    visually near-identical and differ only by the painted value — the confusers
    the appearance head exists to separate.  This is intentionally a *finer*
    partition than the i/p/w coarse supercategory in COARSE_SUPERCATEGORY_BY_PREFIX.
    """
    text = _normalize_text(name).lower()
    if not text:
        return ""
    # Strip a trailing number (optionally decimal), e.g. "pl40" -> "pl",
    # "ph4.5" -> "ph".  Keep the leading alphabetic stem.
    idx = len(text)
    while idx > 0 and (text[idx - 1].isdigit() or text[idx - 1] in "."):
        idx -= 1
    stem = text[:idx]
    return stem if stem else text


def build_fine_group_ids(
    raw_label_map,
    category_meta_path=None,
):
    """Build a global [num_classes] fine-group id vector aligned to class id order.

    Returns (fine_group_ids, group_key_to_id) where fine_group_ids[i] is the
    integer fine-group id of class i (class id == global label id == prompt index,
    matching main._refresh_tt100k_label_list which orders by range(len(label_map))).

    Classes sharing a fine-group key (e.g. all "pl*" speed limits) get the same id.
    Unresolvable names fall back to their own singleton group so they never pull
    spurious negatives.
    """
    raw_label_map = normalize_raw_label_map(raw_label_map)
    builder = AppearancePromptBuilder(
        enabled=False,
        text_source="name",
        category_meta_path=category_meta_path,
    )

    num_classes = len(raw_label_map)
    group_key_to_id = {}
    fine_group_ids = [INVALID_COARSE_GROUP_ID] * num_classes
    for idx in range(num_classes):
        label_entry = raw_label_map.get(str(idx))
        # Prefer the explicit name; the alias resolver already falls back to the id.
        name = builder.get_label_alias(idx, label_entry)
        key = _fine_group_key_from_name(name)
        if not key:
            # Unique singleton group so this class has no same-group siblings.
            key = f"__singleton_{idx}"
        if key not in group_key_to_id:
            group_key_to_id[key] = len(group_key_to_id)
        fine_group_ids[idx] = group_key_to_id[key]
    return fine_group_ids, group_key_to_id


def normalize_raw_label_map(raw_label_map):
    """Normalize supported label-map JSON formats to an id->entry dict."""

    if isinstance(raw_label_map, dict):
        return {str(label_id): label_entry for label_id, label_entry in raw_label_map.items()}

    if isinstance(raw_label_map, list):
        normalized = {}
        for idx, item in enumerate(raw_label_map):
            if isinstance(item, dict):
                label_id = item.get("index", idx)
                label_entry = dict(item)
                label_entry.pop("index", None)
                if "name" not in label_entry and "label" in label_entry:
                    label_entry["name"] = label_entry["label"]
            else:
                label_id = idx
                label_entry = item
            normalized[str(label_id)] = label_entry
        return normalized

    raise ValueError(
        f"Unsupported label_map format: {type(raw_label_map).__name__}. "
        "Expected dict or list."
    )


def build_label_text_map(
    raw_label_map,
    enabled=False,
    text_source="semantic",
    separator=", ",
    category_meta_path=None,
    deduplicate=True,
):
    raw_label_map = normalize_raw_label_map(raw_label_map)
    builder = AppearancePromptBuilder(
        enabled=enabled,
        text_source=text_source,
        separator=separator,
        category_meta_path=category_meta_path,
    )

    label_text_map = {}
    for label_id, label_entry in raw_label_map.items():
        label_text_map[str(label_id)] = builder.build(label_id, label_entry)

    if not deduplicate:
        return label_text_map

    duplicate_counter = Counter(label_text_map.values())
    for label_id, label_entry in raw_label_map.items():
        key = str(label_id)
        label_text = label_text_map[key]
        if duplicate_counter[label_text] <= 1:
            continue

        label_alias = builder.get_label_alias(label_id, label_entry)
        label_text_map[key] = f"{_strip_trailing_caption_punct(label_text)} ({label_alias})"

    return label_text_map


def build_multi_facet_texts(
    raw_label_map,
    category_meta_path=None,
    facet_sources=("facet_shape", "facet_symbol"),
    allow_semantic_fallback=False,
):
    """Build [num_classes, num_facets] text facets aligned to class id order."""

    raw_label_map = normalize_raw_label_map(raw_label_map)
    builder = AppearancePromptBuilder(
        enabled=False,
        text_source="semantic",
        category_meta_path=category_meta_path,
    )
    num_classes = len(raw_label_map)
    facets_by_class = []
    for idx in range(num_classes):
        label_entry = raw_label_map.get(str(idx))
        meta = builder._resolve_meta(idx, label_entry)
        description_text = _normalize_text(meta.get("description"))
        semantic_text = (
            _normalize_text(meta.get("semantic"))
            or _normalize_text(meta.get("label"))
            or _normalize_text(meta.get("name"))
        )
        visual_fallback = (
            _normalize_text(meta.get("facet_symbol"))
            or _normalize_text(meta.get("visual_cue_text"))
            or _normalize_text(meta.get("facet_shape"))
            or _normalize_text(meta.get("attr_text"))
        )
        if allow_semantic_fallback:
            fallback = (
                _normalize_text(meta.get("disc_text"))
                or _normalize_text(meta.get("infer_text"))
                or description_text
                or semantic_text
                or visual_fallback
                or str(idx)
            )
        else:
            fallback = visual_fallback
        class_facets = []
        for source in facet_sources:
            source = str(source).strip()
            if source == "attr_text":
                text = (
                    _normalize_text(meta.get("attr_text"))
                    or _normalize_text(meta.get("facet_shape"))
                    or builder._build_structured_attr_fallback(meta)
                )
            elif source == "visual_cue_text":
                text = _normalize_text(meta.get("visual_cue_text")) or _normalize_text(meta.get("facet_symbol"))
            elif source == "disc_text":
                text = _normalize_text(meta.get("disc_text")) or semantic_text
                if not allow_semantic_fallback and text == semantic_text:
                    text = ""
            elif source == "facet_shape":
                text = _normalize_text(meta.get("facet_shape")) or _normalize_text(meta.get("attr_text"))
            elif source == "facet_symbol":
                text = _normalize_text(meta.get("facet_symbol")) or _normalize_text(meta.get("visual_cue_text"))
            elif source == "semantic":
                text = semantic_text if allow_semantic_fallback else ""
            elif source == "description":
                text = description_text
            else:
                text = _normalize_text(meta.get(source))
            if not text and not fallback:
                raise ValueError(
                    "Missing Self-FACB facet text for class "
                    f"{idx}, source={source}. Provide visual facet fields or enable semantic fallback."
                )
            class_facets.append(text or fallback)
        facets_by_class.append(class_facets)
    return facets_by_class
