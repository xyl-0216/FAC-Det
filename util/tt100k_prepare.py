"""Shared method utilities."""
import json


from pathlib import Path


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_text(entry, *keys):
    if not isinstance(entry, dict):
        return str(entry).strip()
    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def normalize_label_map(raw_label_map):
    if isinstance(raw_label_map, dict):
        return {str(key): value for key, value in raw_label_map.items()}
    if isinstance(raw_label_map, list):
        normalized = {}
        for fallback_id, entry in enumerate(raw_label_map):
            if isinstance(entry, dict):
                label_id = entry.get("index", fallback_id)
                label_entry = dict(entry)
                label_entry.pop("index", None)
                if "name" not in label_entry and "label" in label_entry:
                    label_entry["name"] = label_entry["label"]
            else:
                label_id = fallback_id
                label_entry = entry
            normalized[str(label_id)] = label_entry
        return normalized
    raise ValueError(f"Unsupported label map format: {type(raw_label_map).__name__}")


def build_category_tables(types, label_map):
    categories = []
    class_table = []
    name_to_id = {}
    id_to_semantic = {}
    for idx, name in enumerate(types):
        raw_entry = label_map.get(str(idx), {})
        semantic = extract_text(raw_entry, "label", "semantic", "name", "text", "description")
        if not semantic:
            semantic = str(name)
        categories.append(
            {
                "id": int(idx),
                "name": str(name),
                "supercategory": "traffic_sign",
                "semantic": str(semantic),
            }
        )
        class_table.append(
            {
                "orig_id": int(idx),
                "new_id": int(idx),
                "name": str(name),
                "semantic": str(semantic),
            }
        )
        name_to_id[str(name)] = int(idx)
        id_to_semantic[int(idx)] = str(semantic)
    return categories, class_table, name_to_id, id_to_semantic


def convert_objects_to_instances(objects, name_to_id, id_to_semantic, allowed_ids=None):
    instances = []
    for obj in objects:
        cat_name = str(obj["category"])
        if cat_name not in name_to_id:
            continue
        cat_id = int(name_to_id[cat_name])
        if allowed_ids is not None and cat_id not in allowed_ids:
            continue
        bbox = obj.get("bbox", {})
        x1 = float(bbox["xmin"])
        y1 = float(bbox["ymin"])
        x2 = float(bbox["xmax"])
        y2 = float(bbox["ymax"])
        instances.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": int(cat_id),
                "category": str(id_to_semantic[cat_id]),
            }
        )
    return instances


def build_full_train_records(imgs, name_to_id, id_to_semantic):
    records = []
    image_ids = []
    for image in imgs.values():
        split = str(image["path"]).split("/")[0]
        if split not in {"train", "other"}:
            continue
        objects = image.get("objects", [])
        if not objects:
            continue
        instances = convert_objects_to_instances(objects, name_to_id, id_to_semantic, allowed_ids=None)
        if not instances:
            continue
        records.append(
            {
                "filename": str(image["path"]),
                "height": int(image.get("height", 2048)),
                "width": int(image.get("width", 2048)),
                "detection": {"instances": instances},
            }
        )
        image_ids.append(int(image["id"]))
    return records, image_ids


def build_test_coco(imgs, categories, name_to_id):
    images = []
    annotations = []
    ann_id = 1
    for image in imgs.values():
        split = str(image["path"]).split("/")[0]
        if split != "test":
            continue
        image_id = int(image["id"])
        images.append(
            {
                "id": image_id,
                "file_name": str(image["path"]),
                "height": int(image.get("height", 2048)),
                "width": int(image.get("width", 2048)),
            }
        )
        for obj in image.get("objects", []):
            cat_name = str(obj["category"])
            if cat_name not in name_to_id:
                continue
            bbox = obj.get("bbox", {})
            x1 = float(bbox["xmin"])
            y1 = float(bbox["ymin"])
            x2 = float(bbox["xmax"])
            y2 = float(bbox["ymax"])
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(name_to_id[cat_name]),
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {"description": "TT100K official test COCO export"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


