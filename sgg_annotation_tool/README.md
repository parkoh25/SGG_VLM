# SGG Annotation Tool

SGG Annotation Tool is a PyQt desktop application for creating scene graph
annotations for SGG-Benchmark style datasets.

It is designed for this workflow:

1. Load an image directory.
2. Load COCO-style object annotations.
3. Select two objects and add a relationship predicate.
4. Add object attributes when needed.
5. Save per-image scene graph JSON files.
6. Export `VG-SGG.h5`, `VG-SGG-dicts.json`, and `image_data.json`.

## Install

Windows PowerShell:

```powershell
cd sgg_annotation_tool
py -m pip install -e .
```

Linux or macOS:

```bash
cd sgg_annotation_tool
python -m pip install -e .
```

## Run the Annotation App

```powershell
sgg-annotator
```

Or:

```powershell
py -m sgg_annotation_tool
```

## GUI Usage

Use the buttons in the top toolbar:

- `Load Images`: select the folder that contains the image files.
- `Load COCO JSON`: select a COCO-style instances JSON file.
- `Set Output`: select where annotation JSON and exports should be written.
- `Previous` / `Next`: move between images. The current image is saved before navigation.
- `Save`: save the current image annotation JSON.

Object selection:

- Click an object in the object list or in the image.
- Use `Ctrl+click` to select the second object.
- A relationship is created from the first selected object to the second selected object.

Relationship annotation:

- Enter a predicate such as `near`, `attached to`, `occludes`, or `supports`.
- Click `Add Relationship`.
- Existing relationships for the same subject/object pair are replaced.
- Select a relationship row and click `Remove Selected` to delete it.

Attribute annotation:

- Select one or more objects.
- Enter an attribute key and value.
- Click `Add Attribute`.
- Empty values are saved as boolean `true`.

## Saved Annotation JSON

The app writes one JSON file per image:

```text
annotation_output/
  json/
    image_001.json
    image_002.json
```

Each file contains:

```json
{
  "image": {
    "id": 1,
    "file_name": "image_001.jpg",
    "width": 640,
    "height": 480
  },
  "categories": [
    {
      "id": "1",
      "name": "fruit"
    }
  ],
  "annotations": [],
  "relationships": [
    [11, 12, "attached to"]
  ]
}
```

Relationship object ids are the original COCO annotation ids.

## Export SGG-Benchmark Dataset

Inside the GUI, click `Export SGG-Benchmark Dataset`.

The app writes:

```text
annotation_output/
  sgg_benchmark/
    VG-SGG.h5
    VG-SGG-dicts.json
    image_data.json
    manifest.json
```

You can also export from the command line:

```powershell
sgg-export-dataset `
  --instances ..\instances_Train.json `
  --relationships-dir annotation_output\json `
  --output-dir annotation_output\sgg_benchmark `
  --images-dir ..\images `
  --overwrite
```

To copy images into the Visual Genome naming layout:

```powershell
sgg-export-dataset `
  --instances ..\instances_Train.json `
  --relationships-dir annotation_output\json `
  --output-dir annotation_output\sgg_benchmark `
  --images-dir ..\images `
  --copy-images `
  --overwrite
```

This creates `VG_100K/<image_id>.jpg`.

## SGG-Benchmark Config Values

Use the generated files as:

- `roidb_file`: `annotation_output/sgg_benchmark/VG-SGG.h5`
- `dict_file`: `annotation_output/sgg_benchmark/VG-SGG-dicts.json`
- `image_file`: `annotation_output/sgg_benchmark/image_data.json`
- `img_dir`: source image folder, or `annotation_output/sgg_benchmark/VG_100K` when `--copy-images` is used

The `manifest.json` file contains `num_val_im` for loaders that need the
validation image count.

## Notes

- The GUI edits relationships and attributes; object boxes and masks come from
  the COCO instances JSON.
- Polygon segmentations are drawn when available. Otherwise the app draws the
  COCO bounding box.
- RLE masks are not decoded by the GUI; their bounding boxes are still usable.
- Generated annotation output and HDF5 files are ignored by git.
