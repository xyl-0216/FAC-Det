TRAILING_CAPTION_PUNCT = ".!?;:。！？；："


def normalize_caption_label_text(text):
    value = str(text).strip()
    while value and value[-1] in TRAILING_CAPTION_PUNCT:
        value = value[:-1].rstrip()
    return value


def build_caption_from_labels(label_texts):
    normalized = [normalize_caption_label_text(text) for text in label_texts]
    normalized = [text for text in normalized if text]
    if not normalized:
        return "", []
    caption = " . ".join(normalized) + " ."
    return caption, normalized
