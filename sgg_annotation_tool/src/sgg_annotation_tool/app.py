from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .exporter import build_dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class AnnotationView(QGraphicsView):
    objectClicked = pyqtSignal(object, bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom_factor = 1.15

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self._object_item_at(event.pos())
            if item is not None:
                self.objectClicked.emit(item.data(0), bool(event.modifiers() & Qt.ControlModifier))
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self._zoom_factor, self._zoom_factor)
        else:
            self.scale(1 / self._zoom_factor, 1 / self._zoom_factor)

    def _object_item_at(self, view_pos):
        scene_pos = self.mapToScene(view_pos)
        for item in self.scene().items(scene_pos):
            current = item
            while current is not None:
                if current.data(0) is not None:
                    return current
                current = current.parentItem()
        return None


class SceneGraphAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images_dir: Path | None = None
        self.instances_file: Path | None = None
        self.output_dir: Path | None = None
        self.images: list[dict[str, Any]] = []
        self.categories: dict[str, str] = {}
        self.annotations_by_image: dict[str, list[dict[str, Any]]] = {}
        self.relationships_by_image: dict[str, list[list[Any]]] = {}
        self.current_index = 0
        self.object_name_by_id: dict[str, str] = {}
        self.selected_object_ids: list[Any] = []
        self.item_by_object_id: dict[str, QGraphicsItemGroup] = {}
        self.color_by_object_id: dict[str, QColor] = {}

        self.scene = QGraphicsScene(self)
        self.view = AnnotationView(self)
        self.view.setScene(self.scene)
        self.view.objectClicked.connect(self.select_object)

        self._build_ui()
        self.setWindowTitle("SGG Annotation Tool")
        self.resize(1400, 850)

    def _build_ui(self) -> None:
        root = QSplitter(self)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel())
        root.setStretchFactor(0, 5)
        root.setStretchFactor(1, 2)
        self.setCentralWidget(root)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        toolbar = QHBoxLayout()
        self.load_images_btn = QPushButton("Load Images")
        self.load_images_btn.clicked.connect(self.load_images_dir)
        self.load_instances_btn = QPushButton("Load COCO JSON")
        self.load_instances_btn.clicked.connect(self.load_instances_file)
        self.output_btn = QPushButton("Set Output")
        self.output_btn.clicked.connect(self.set_output_dir)
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_current_image)
        for button in [
            self.load_images_btn,
            self.load_instances_btn,
            self.output_btn,
            self.prev_btn,
            self.next_btn,
            self.save_btn,
        ]:
            toolbar.addWidget(button)

        self.image_label = QLabel("No image loaded")
        self.image_label.setMinimumHeight(24)
        layout.addLayout(toolbar)
        layout.addWidget(self.image_label)
        layout.addWidget(self.view)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("Objects"))
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.object_list_clicked)
        layout.addWidget(self.object_list, 2)

        layout.addWidget(QLabel("Selected"))
        self.selected_label = QLabel("-")
        layout.addWidget(self.selected_label)

        rel_layout = QGridLayout()
        rel_layout.addWidget(QLabel("Predicate"), 0, 0)
        self.predicate_input = QLineEdit()
        self.predicate_input.setPlaceholderText("near, attached to, occludes, supports")
        self.predicate_input.returnPressed.connect(self.add_relationship)
        rel_layout.addWidget(self.predicate_input, 0, 1)
        self.add_rel_btn = QPushButton("Add Relationship")
        self.add_rel_btn.clicked.connect(self.add_relationship)
        self.remove_rel_btn = QPushButton("Remove Selected")
        self.remove_rel_btn.clicked.connect(self.remove_selected_relationship)
        rel_layout.addWidget(self.add_rel_btn, 1, 0)
        rel_layout.addWidget(self.remove_rel_btn, 1, 1)
        layout.addLayout(rel_layout)

        layout.addWidget(QLabel("Relationships"))
        self.relationship_list = QListWidget()
        layout.addWidget(self.relationship_list, 2)

        attr_layout = QGridLayout()
        attr_layout.addWidget(QLabel("Attribute Key"), 0, 0)
        self.attribute_key_input = QLineEdit()
        self.attribute_key_input.setPlaceholderText("color")
        attr_layout.addWidget(self.attribute_key_input, 0, 1)
        attr_layout.addWidget(QLabel("Value"), 1, 0)
        self.attribute_value_input = QLineEdit()
        self.attribute_value_input.setPlaceholderText("green")
        attr_layout.addWidget(self.attribute_value_input, 1, 1)
        self.add_attr_btn = QPushButton("Add Attribute")
        self.add_attr_btn.clicked.connect(self.add_attribute)
        attr_layout.addWidget(self.add_attr_btn, 2, 0, 1, 2)
        layout.addLayout(attr_layout)

        layout.addWidget(QLabel("Object Details"))
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 1)

        self.export_btn = QPushButton("Export SGG-Benchmark Dataset")
        self.export_btn.clicked.connect(self.export_dataset)
        layout.addWidget(self.export_btn)
        return panel

    def load_images_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image directory")
        if not folder:
            return
        self.images_dir = Path(folder)
        if self.output_dir is None:
            self.output_dir = self.images_dir / "annotation_output"
        self._load_image_files_without_instances()
        self.show_current_image()

    def load_instances_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Select COCO instances JSON", "", "JSON Files (*.json)")
        if not file_name:
            return
        self.instances_file = Path(file_name)
        with self.instances_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        self.categories = {str(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
        self.images = [
            {
                "id": item["id"],
                "file_name": item["file_name"],
                "width": item.get("width"),
                "height": item.get("height"),
            }
            for item in payload.get("images", [])
        ]
        self.annotations_by_image = {}
        for annotation in payload.get("annotations", []):
            self.annotations_by_image.setdefault(str(annotation.get("image_id")), []).append(annotation)
        for annotations in self.annotations_by_image.values():
            annotations.sort(key=lambda annotation: _sort_key(annotation.get("id")))
        self.current_index = 0
        self.load_saved_relationships()
        self.show_current_image()

    def set_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select annotation output directory")
        if folder:
            self.output_dir = Path(folder)
            self.load_saved_relationships()

    def _load_image_files_without_instances(self) -> None:
        if self.instances_file is not None or self.images_dir is None:
            return
        files = sorted(path for path in self.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        self.images = [
            {"id": index + 1, "file_name": path.name, "width": None, "height": None}
            for index, path in enumerate(files)
        ]

    def load_saved_relationships(self) -> None:
        self.relationships_by_image = {}
        if not self.output_dir:
            return
        rel_dir = self.output_dir / "json"
        if not rel_dir.exists():
            return
        for image in self.images:
            path = rel_dir / f"{Path(image['file_name']).stem}.json"
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
                self.relationships_by_image[str(image["id"])] = payload.get("relationships", [])
                if payload.get("annotations"):
                    self.annotations_by_image[str(image["id"])] = payload["annotations"]
            except (OSError, json.JSONDecodeError):
                continue

    def show_current_image(self) -> None:
        self.scene.clear()
        self.object_list.clear()
        self.relationship_list.clear()
        self.details.clear()
        self.selected_object_ids = []
        self.item_by_object_id = {}
        self.object_name_by_id = {}

        if not self.images or self.images_dir is None:
            self.image_label.setText("Load an image directory and a COCO JSON file.")
            return

        image = self.images[self.current_index]
        image_path = self.images_dir / image["file_name"]
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Image Error", f"Could not load image: {image_path}")
            return

        if not image.get("width"):
            image["width"] = pixmap.width()
        if not image.get("height"):
            image["height"] = pixmap.height()

        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(pixmap.rect())
        self.image_label.setText(f"{self.current_index + 1}/{len(self.images)}  {image['file_name']}")
        self._draw_objects(image)
        self._refresh_relationship_list()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _draw_objects(self, image: dict[str, Any]) -> None:
        annotations = self.annotations_by_image.get(str(image["id"]), [])
        category_counts: dict[str, int] = {}
        for annotation in annotations:
            object_id = annotation.get("id")
            category_name = self.categories.get(str(annotation.get("category_id")), f"category_{annotation.get('category_id')}")
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
            object_name = f"{category_name}_{category_counts[category_name]}"
            self.object_name_by_id[str(object_id)] = object_name

            group = QGraphicsItemGroup()
            group.setData(0, object_id)
            color = self._color_for_object(object_id)
            item = self._make_shape_item(annotation, color)
            if item is None:
                continue
            item.setData(0, object_id)
            group.addToGroup(item)
            label = QGraphicsTextItem(object_name)
            label.setDefaultTextColor(color)
            x, y, _, _ = _bbox_from_annotation(annotation) or (0, 0, 0, 0)
            label.setPos(float(x), max(0.0, float(y) - 20.0))
            group.addToGroup(label)
            self.scene.addItem(group)
            self.item_by_object_id[str(object_id)] = group

            list_item = QListWidgetItem(object_name)
            list_item.setData(Qt.UserRole, object_id)
            self.object_list.addItem(list_item)

    def _make_shape_item(self, annotation: dict[str, Any], color: QColor):
        polygon = _first_polygon(annotation.get("segmentation"))
        pen = QPen(color, 2)
        brush = QColor(color.red(), color.green(), color.blue(), 70)
        if polygon:
            points = [QPointF(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            item = QGraphicsPolygonItem(QPolygonF(points))
        else:
            bbox = _bbox_from_annotation(annotation)
            if not bbox:
                return None
            x, y, width, height = bbox
            item = QGraphicsRectItem(float(x), float(y), float(width), float(height))
        item.setPen(pen)
        item.setBrush(brush)
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        return item

    def object_list_clicked(self, item: QListWidgetItem) -> None:
        self.select_object(item.data(Qt.UserRole), bool(QApplication.keyboardModifiers() & Qt.ControlModifier))

    def select_object(self, object_id: Any, append: bool) -> None:
        if object_id is None:
            return
        if append:
            if object_id in self.selected_object_ids:
                self.selected_object_ids.remove(object_id)
            elif len(self.selected_object_ids) < 2:
                self.selected_object_ids.append(object_id)
        else:
            self.selected_object_ids = [object_id]
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        for object_id, group in self.item_by_object_id.items():
            selected = any(str(value) == object_id for value in self.selected_object_ids)
            for child in group.childItems():
                if isinstance(child, QGraphicsTextItem):
                    child.setDefaultTextColor(QColor(255, 180, 0) if selected else self._color_for_object(object_id))
                elif hasattr(child, "setBrush"):
                    color = QColor(255, 180, 0) if selected else self._color_for_object(object_id)
                    child.setBrush(QColor(color.red(), color.green(), color.blue(), 110 if selected else 70))
        selected_names = [self.object_name_by_id.get(str(value), str(value)) for value in self.selected_object_ids]
        self.selected_label.setText(" -> ".join(selected_names) if selected_names else "-")
        self._refresh_details()

    def add_relationship(self) -> None:
        if len(self.selected_object_ids) != 2:
            QMessageBox.information(self, "Relationship", "Select exactly two objects. Use Ctrl+click for the second object.")
            return
        predicate = self.predicate_input.text().strip()
        if not predicate:
            return
        image_id = str(self.images[self.current_index]["id"])
        subject_id, object_id = self.selected_object_ids
        relationships = self.relationships_by_image.setdefault(image_id, [])
        relationships[:] = [
            rel for rel in relationships
            if not (str(rel[0]) == str(subject_id) and str(rel[1]) == str(object_id))
        ]
        relationships.append([subject_id, object_id, predicate])
        self.predicate_input.clear()
        self._refresh_relationship_list()
        self.save_current_image()

    def remove_selected_relationship(self) -> None:
        item = self.relationship_list.currentItem()
        if item is None:
            return
        image_id = str(self.images[self.current_index]["id"])
        index = item.data(Qt.UserRole)
        relationships = self.relationships_by_image.get(image_id, [])
        if isinstance(index, int) and 0 <= index < len(relationships):
            relationships.pop(index)
            self._refresh_relationship_list()
            self.save_current_image()

    def add_attribute(self) -> None:
        if not self.selected_object_ids:
            return
        key = self.attribute_key_input.text().strip()
        value = self.attribute_value_input.text().strip()
        if not key:
            return
        image = self.images[self.current_index]
        annotations = self.annotations_by_image.get(str(image["id"]), [])
        selected = {str(object_id) for object_id in self.selected_object_ids}
        for annotation in annotations:
            if str(annotation.get("id")) not in selected:
                continue
            attrs = annotation.setdefault("attributes", {})
            attrs[key] = value if value else True
        self.attribute_key_input.clear()
        self.attribute_value_input.clear()
        self._refresh_details()
        self.save_current_image()

    def _refresh_relationship_list(self) -> None:
        self.relationship_list.clear()
        if not self.images:
            return
        image_id = str(self.images[self.current_index]["id"])
        for index, rel in enumerate(self.relationships_by_image.get(image_id, [])):
            subject = self.object_name_by_id.get(str(rel[0]), str(rel[0]))
            obj = self.object_name_by_id.get(str(rel[1]), str(rel[1]))
            item = QListWidgetItem(f"{subject} -- {rel[2]} -> {obj}")
            item.setData(Qt.UserRole, index)
            self.relationship_list.addItem(item)

    def _refresh_details(self) -> None:
        if not self.images:
            return
        image = self.images[self.current_index]
        annotations = self.annotations_by_image.get(str(image["id"]), [])
        selected = {str(value) for value in self.selected_object_ids}
        lines = []
        for annotation in annotations:
            if str(annotation.get("id")) not in selected:
                continue
            name = self.object_name_by_id.get(str(annotation.get("id")), str(annotation.get("id")))
            lines.append(f"{name}  id={annotation.get('id')}")
            lines.append(f"category={self.categories.get(str(annotation.get('category_id')), annotation.get('category_id'))}")
            lines.append(f"bbox={annotation.get('bbox')}")
            lines.append(f"attributes={annotation.get('attributes', {})}")
            lines.append("")
        self.details.setPlainText("\n".join(lines).strip())

    def save_current_image(self) -> None:
        if not self.images or self.output_dir is None:
            return
        image = self.images[self.current_index]
        rel_dir = self.output_dir / "json"
        rel_dir.mkdir(parents=True, exist_ok=True)
        path = rel_dir / f"{Path(image['file_name']).stem}.json"
        payload = {
            "image": image,
            "categories": [{"id": key, "name": value} for key, value in self.categories.items()],
            "annotations": self.annotations_by_image.get(str(image["id"]), []),
            "relationships": self.relationships_by_image.get(str(image["id"]), []),
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def export_dataset(self) -> None:
        if self.instances_file is None or self.output_dir is None:
            QMessageBox.information(self, "Export", "Load a COCO JSON file and set an output directory first.")
            return
        self.save_current_image()
        export_dir = self.output_dir / "sgg_benchmark"
        try:
            summary = build_dataset(
                instances_file=self.instances_file,
                relationships_dir=self.output_dir / "json",
                output_dir=export_dir,
                images_dir=self.images_dir,
                copy_images=False,
                overwrite=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        QMessageBox.information(
            self,
            "Export Complete",
            f"Wrote {summary.h5_file}\n"
            f"Images: {summary.image_count}\n"
            f"Objects: {summary.object_count}\n"
            f"Relations: {summary.relation_count}\n"
            f"Skipped relations: {summary.skipped_relation_count}",
        )

    def previous_image(self) -> None:
        if self.current_index > 0:
            self.save_current_image()
            self.current_index -= 1
            self.show_current_image()

    def next_image(self) -> None:
        if self.current_index < len(self.images) - 1:
            self.save_current_image()
            self.current_index += 1
            self.show_current_image()

    def _color_for_object(self, object_id: Any) -> QColor:
        key = str(object_id)
        if key not in self.color_by_object_id:
            self.color_by_object_id[key] = QColor.fromHsv(abs(hash(key)) % 360, 210, 230)
        return self.color_by_object_id[key]


def _first_polygon(segmentation: Any) -> list[float] | None:
    if not isinstance(segmentation, list):
        return None
    for item in segmentation:
        if isinstance(item, list) and len(item) >= 6 and all(isinstance(value, (int, float)) for value in item):
            return item
        if isinstance(item, list):
            nested = _first_polygon(item)
            if nested:
                return nested
    return None


def _bbox_from_annotation(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return tuple(float(value) for value in bbox[:4])
    polygon = _first_polygon(annotation.get("segmentation"))
    if not polygon:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)


def main() -> None:
    app = QApplication(sys.argv)
    window = SceneGraphAnnotator()
    window.show()
    sys.exit(app.exec_())
