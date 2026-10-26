from transformers import AutoConfig, AutoTokenizer, AutoModel
import os


_MODERN_BERT_IDS = {
    "answer-ai/ModernBERT-base",
    "answer-ai/ModernBERT-large",
    "answerdotai/ModernBERT-base",
    "answerdotai/ModernBERT-large",
}


def _resolve_local_encoder_path(text_encoder_type):
    if os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
        return text_encoder_type

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    workspace_root = os.path.abspath(os.path.join(project_root, ".."))
    for base_dir in (os.getcwd(), project_root, workspace_root):
        candidate = os.path.join(base_dir, text_encoder_type)
        if os.path.isdir(candidate):
            return candidate

    return text_encoder_type


def _is_modern_bert(text_encoder_type):
    resolved = _resolve_local_encoder_path(text_encoder_type)
    if resolved in _MODERN_BERT_IDS:
        return True
    if os.path.isdir(resolved):
        try:
            config = AutoConfig.from_pretrained(resolved, local_files_only=True)
            return getattr(config, "model_type", None) == "modernbert"
        except Exception:
            return False
    return False


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    text_encoder_type = _resolve_local_encoder_path(text_encoder_type)
    print("final text_encoder_type: {}".format(text_encoder_type))
    if _is_modern_bert(text_encoder_type):
        tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_type, local_files_only=os.path.isdir(text_encoder_type)
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, local_files_only=True)
    print("load tokenizer done.")
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    text_encoder_type = _resolve_local_encoder_path(text_encoder_type)
    if _is_modern_bert(text_encoder_type):
        return AutoModel.from_pretrained(text_encoder_type, local_files_only=os.path.isdir(text_encoder_type))
    if text_encoder_type == "bert-base-uncased":
        from transformers import BertModel
        return BertModel.from_pretrained(text_encoder_type, local_files_only=True)
    if text_encoder_type == "roberta-base":
        from transformers import RobertaModel
        return RobertaModel.from_pretrained(text_encoder_type, local_files_only=True)
    if os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
        return AutoModel.from_pretrained(text_encoder_type, local_files_only=True)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
