from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image

BOX_SCALE = 1024
TRAIN_VAL_SPLIT = 0
TEST_SPLIT = 2
UNKNOWN_VALUES = {"", "unknown", "none", "null", "false", "no", "nan"}


@dataclass(frozen=True)
class ExportSummary:
    h5_file: Path
    dict_file: Path
    image_data_file: Path
    manifest_file: Path
    image_count: int
    object_count: int
    relation_count: int
    skipped_relation_count: int
    num_val_im: int


def build_dataset(
    instances_file: Path,
    relationships_dir: Path,
    output_dir: Path,
    images_dir: Path | None = None,
    copy_images: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 13,
    split_file: Path | None = None,
    max_attributes: int = 10,
    overwrite: bool = False,
) -> ExportSummary:
    instances_file = Path(instances_file)
    relationships_dir = Path(relationships_dir)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir) if images_dir else None

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = _load_json(instances_file)
    images = _read_images(instances, images_dir)
    categories = _read_categories(instances)
    category_name_by_id = {str(item["id"]): item["name"] for item in categories}
    label_to_idx = {
        item["name"]: index + 1
        for index, item in enumerate(sorted(categories, key=lambda category: _sort_key(category["id"])))
    }

    split_by_id = _assign_splits(instances, split_file, train_ratio, val_ratio, test_ratio, seed)
    images = sorted(
        [{**image, "split": split_by_id.get(str(image["id"]), "train")} for image in images],
        key=lambda image: ({"val": 0, "train": 1, "test": 2}.get(image["split"], 1), _sort_key(image["id"])),
    )
    annotations_by_image = _group_annotations(instances)
    objects_by_image, attribute_names = _build_objects(images, annotations_by_image, category_name_by_id, label_to_idx)
    relationships = _load_relationships(images, relationships_dir)
    predicate_to_idx = {
        predicate: index + 1
        for index, predicate in enumerate(sorted({rel["predicate"] for rel in relationships}))
    }
    attribute_to_idx = {name: index + 1 for index, name in enumerate(sorted(attribute_names))}

    arrays = _make_h5_arrays(
        images,
        objects_by_image,
        relationships,
        predicate_to_idx,
        attribute_to_idx,
        max_attributes=max_attributes,
    )

    h5_file = output_dir / "VG-SGG.h5"
    dict_file = output_dir / "VG-SGG-dicts.json"
    image_data_file = output_dir / "image_data.json"
    manifest_file = output_dir / "manifest.json"

    with h5py.File(h5_file, "w") as h5:
        for key, value in arrays.items():
            if key != "skipped_relation_count":
                h5.create_dataset(key, data=value)

    _write_json(
        dict_file,
        {
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(index): name for name, index in label_to_idx.items()},
            "predicate_to_idx": predicate_to_idx,
            "idx_to_predicate": {str(index): name for name, index in predicate_to_idx.items()},
            "attribute_to_idx": attribute_to_idx,
            "idx_to_attribute": {str(index): name for name, index in attribute_to_idx.items()},
        },
    )

    _write_json(
        image_data_file,
        [
            {
                "image_id": image["id"],
                "width": image["width"],
                "height": image["height"],
                "file_name": f"{image['id']}.jpg" if copy_images else image["file_name"],
                "original_file_name": image["file_name"],
                "split": image["split"],
            }
            for image in images
        ],
    )

    if copy_images:
        if images_dir is None:
            raise ValueError("--copy-images requires --images-dir")
        _copy_images(images, images_dir, output_dir / "VG_100K")

    num_val_im = sum(1 for image in images if image["split"] == "val")
    _write_json(
        manifest_file,
        {
            "format": "visual-genome-style-sgg-benchmark",
            "box_scale": BOX_SCALE,
            "files": {
                "roidb": "VG-SGG.h5",
                "dict": "VG-SGG-dicts.json",
                "image_data": "image_data.json",
                "image_dir": "VG_100K" if copy_images else None,
            },
            "counts": {
                "images": len(images),
                "objects": int(arrays["boxes_1024"].shape[0]),
                "relations": int(arrays["relationships"].shape[0]),
                "skipped_relations": int(arrays["skipped_relation_count"]),
                "object_classes": len(label_to_idx),
                "predicate_classes": len(predicate_to_idx),
                "attribute_classes": len(attribute_to_idx),
            },
            "sgg_benchmark_loader_notes": {
                "num_val_im": num_val_im,
                "train_val_split_value": TRAIN_VAL_SPLIT,
                "test_split_value": TEST_SPLIT,
            },
        },
    )

    return ExportSummary(
        h5_file=h5_file,
        dict_file=dict_file,
        image_data_file=image_data_file,
        manifest_file=manifest_file,
        image_count=len(images),
        object_count=int(arrays["boxes_1024"].shape[0]),
        relation_count=int(arrays["relationships"].shape[0]),
        skipped_relation_count=int(arrays["skipped_relation_count"]),
        num_val_im=num_val_im,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export scene graph annotations to SGG-Benchmark files.")
    parser.add_argument("--instances", required=True, type=Path, help="COCO instances JSON file.")
    parser.add_argument("--relationships-dir", required=True, type=Path, help="Directory with per-image annotation JSON files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--images-dir", type=Path, help="Source image directory.")
    parser.add_argument("--copy-images", action="store_true", help="Copy/convert images to VG_100K/<image_id>.jpg.")
    parser.add_argument("--split-file", type=Path, help="Optional JSON split map.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-attributes", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    summary = build_dataset(
        instances_file=args.instances,
        relationships_dir=args.relationships_dir,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        copy_images=args.copy_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_file=args.split_file,
        max_attributes=args.max_attributes,
        overwrite=args.overwrite,
    )
    print(f"Wrote {summary.h5_file}")
    print(f"Wrote {summary.dict_file}")
    print(f"Wrote {summary.image_data_file}")
    print(f"Wrote {summary.manifest_file}")
    print(
        "Counts: "
        f"{summary.image_count} images, "
        f"{summary.object_count} objects, "
        f"{summary.relation_count} relations, "
        f"{summary.skipped_relation_count} skipped relations"
    )
    print(f"Use num_val_im={summary.num_val_im} in SGG-Benchmark configs that need it.")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _read_categories(instances: dict[str, Any]) -> list[dict[str, Any]]:
    categories = []
    for item in instances.get("categories", []):
        if item.get("id") is None or not item.get("name"):
            continue
        categories.append({"id": item["id"], "name": str(item["name"])})
    if not categories:
        raise ValueError("The instances file has no categories.")
    return categories


def _read_images(instances: dict[str, Any], images_dir: Path | None) -> list[dict[str, Any]]:
    images = []
    for item in instances.get("images", []):
        image = {
            "id": item["id"],
            "file_name": item["file_name"],
            "width": item.get("width"),
            "height": item.get("height"),
        }
        if (not image["width"] or not image["height"]) and images_dir:
            with Image.open(images_dir / image["file_name"]) as source:
                image["width"] = source.width
                image["height"] = source.height
        if not image["width"] or not image["height"]:
            raise ValueError(f"Image size is missing for {image['file_name']}.")
        image["width"] = int(image["width"])
        image["height"] = int(image["height"])
        images.append(image)
    if not images:
        raise ValueError("The instances file has no images.")
    return images


def _assign_splits(
    instances: dict[str, Any],
    split_file: Path | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if split_file:
        payload = _load_json(split_file)
        image_name_to_id = {str(image["file_name"]): str(image["id"]) for image in instances.get("images", [])}
        return {image_name_to_id.get(str(key), str(key)): str(value) for key, value in payload.items()}

    image_ids = [str(image["id"]) for image in instances.get("images", [])]
    random.Random(seed).shuffle(image_ids)
    total = train_ratio + val_ratio + test_ratio
    val_count = int(round(len(image_ids) * (val_ratio / total))) if total else 0
    test_count = int(round(len(image_ids) * (test_ratio / total))) if total else 0
    split_by_id = {}
    for image_id in image_ids[:val_count]:
        split_by_id[image_id] = "val"
    for image_id in image_ids[val_count : val_count + test_count]:
        split_by_id[image_id] = "test"
    for image_id in image_ids[val_count + test_count :]:
        split_by_id[image_id] = "train"
    return split_by_id


def _group_annotations(instances: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for annotation in instances.get("annotations", []):
        grouped.setdefault(str(annotation.get("image_id")), []).append(annotation)
    for annotations in grouped.values():
        annotations.sort(key=lambda annotation: _sort_key(annotation.get("id")))
    return grouped


def _build_objects(
    images: list[dict[str, Any]],
    annotations_by_image: dict[str, list[dict[str, Any]]],
    category_name_by_id: dict[str, str],
    label_to_idx: dict[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    objects_by_image: dict[str, list[dict[str, Any]]] = {}
    attribute_names: set[str] = set()
    global_index = 0
    for image in images:
        image_key = str(image["id"])
        category_counts: dict[str, int] = {}
        objects = []
        for annotation in annotations_by_image.get(image_key, []):
            bbox = _bbox_from_annotation(annotation)
            if not bbox:
                continue
            category_name = category_name_by_id.get(str(annotation.get("category_id")), f"category_{annotation.get('category_id')}")
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
            attributes = _extract_attributes(annotation.get("attributes"))
            attribute_names.update(attributes)
            objects.append(
                {
                    "id": annotation.get("id"),
                    "image_id": image["id"],
                    "bbox": bbox,
                    "label": label_to_idx[category_name],
                    "name": f"{category_name}_{category_counts[category_name]}",
                    "attributes": attributes,
                    "global_index": global_index,
                }
            )
            global_index += 1
        objects_by_image[image_key] = objects
    return objects_by_image, attribute_names


def _load_relationships(images: list[dict[str, Any]], relationships_dir: Path) -> list[dict[str, Any]]:
    relationships = []
    for image in images:
        source = relationships_dir / f"{Path(image['file_name']).stem}.json"
        if not source.exists():
            continue
        payload = _load_json(source)
        for entry in payload.get("relationships", []):
            if isinstance(entry, list) and len(entry) >= 3:
                subject, obj, predicate = entry[:3]
            elif isinstance(entry, dict):
                subject = entry.get("subject_id", entry.get("subject"))
                obj = entry.get("object_id", entry.get("object"))
                predicate = entry.get("predicate", entry.get("relationship"))
            else:
                continue
            if subject is not None and obj is not None and predicate:
                relationships.append(
                    {
                        "image_id": image["id"],
                        "subject": subject,
                        "object": obj,
                        "predicate": str(predicate),
                    }
                )
    return relationships


def _make_h5_arrays(
    images: list[dict[str, Any]],
    objects_by_image: dict[str, list[dict[str, Any]]],
    relationships: list[dict[str, Any]],
    predicate_to_idx: dict[str, int],
    attribute_to_idx: dict[str, int],
    max_attributes: int,
) -> dict[str, Any]:
    boxes = []
    labels = []
    attrs = []
    split = []
    img_to_first_box = []
    img_to_last_box = []
    img_to_first_rel = []
    img_to_last_rel = []
    rel_pairs = []
    predicates = []
    skipped = 0

    rels_by_image: dict[str, list[dict[str, Any]]] = {}
    for rel in relationships:
        rels_by_image.setdefault(str(rel["image_id"]), []).append(rel)

    for image in images:
        objects = objects_by_image.get(str(image["id"]), [])
        first_box = len(boxes)
        lookup = {}
        for obj in objects:
            lookup[str(obj["id"])] = obj
            lookup[str(obj["name"])] = obj
            boxes.append(_scale_box(obj["bbox"], image["width"], image["height"]))
            labels.append([obj["label"]])
            row = [0] * max_attributes
            for index, name in enumerate(obj["attributes"][:max_attributes]):
                row[index] = attribute_to_idx[name]
            attrs.append(row)

        img_to_first_box.append(first_box if objects else -1)
        img_to_last_box.append(len(boxes) - 1 if objects else -1)

        first_rel = len(rel_pairs)
        for rel in rels_by_image.get(str(image["id"]), []):
            subject = lookup.get(str(rel["subject"]))
            obj = lookup.get(str(rel["object"]))
            if subject is None or obj is None:
                skipped += 1
                continue
            rel_pairs.append([subject["global_index"], obj["global_index"]])
            predicates.append([predicate_to_idx[rel["predicate"]]])

        img_to_first_rel.append(first_rel if len(rel_pairs) > first_rel else -1)
        img_to_last_rel.append(len(rel_pairs) - 1 if len(rel_pairs) > first_rel else -1)
        split.append(TEST_SPLIT if image["split"] == "test" else TRAIN_VAL_SPLIT)

    return {
        "boxes_1024": np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        "labels": np.asarray(labels, dtype=np.int64).reshape((-1, 1)),
        "attributes": np.asarray(attrs, dtype=np.int64).reshape((-1, max_attributes)),
        "relationships": np.asarray(rel_pairs, dtype=np.int64).reshape((-1, 2)),
        "predicates": np.asarray(predicates, dtype=np.int64).reshape((-1, 1)),
        "split": np.asarray(split, dtype=np.int64),
        "img_to_first_box": np.asarray(img_to_first_box, dtype=np.int64),
        "img_to_last_box": np.asarray(img_to_last_box, dtype=np.int64),
        "img_to_first_rel": np.asarray(img_to_first_rel, dtype=np.int64),
        "img_to_last_rel": np.asarray(img_to_last_rel, dtype=np.int64),
        "skipped_relation_count": skipped,
    }


def _bbox_from_annotation(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
        return tuple(float(value) for value in bbox[:4])
    points = list(_iter_points(annotation.get("segmentation")))
    if len(points) < 4:
        return None
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _iter_points(value: Any) -> Iterable[float]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                yield float(item)
            elif isinstance(item, list):
                yield from _iter_points(item)


def _extract_attributes(value: Any) -> list[str]:
    names = []

    def add(name: str) -> None:
        name = name.strip()
        if name and name.lower() not in UNKNOWN_VALUES and name not in names:
            names.append(name)

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, bool):
                if item:
                    add(str(key))
            elif isinstance(item, (int, float)):
                if item:
                    add(f"{key}:{item}")
            elif isinstance(item, str):
                add(item if item.lower() not in {"true", "yes"} else str(key))
    elif isinstance(value, list):
        for item in value:
            for attr in _extract_attributes(item):
                add(attr)
    elif isinstance(value, str):
        add(value)
    return names


def _scale_box(bbox: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = bbox
    scale = BOX_SCALE / max(float(width), float(height))
    return [(x + box_width / 2) * scale, (y + box_height / 2) * scale, box_width * scale, box_height * scale]


def _copy_images(images: list[dict[str, Any]], images_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        source = images_dir / image["file_name"]
        target = output_dir / f"{image['id']}.jpg"
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(source, target)
        else:
            with Image.open(source) as input_image:
                input_image.convert("RGB").save(target, format="JPEG", quality=95)


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)
