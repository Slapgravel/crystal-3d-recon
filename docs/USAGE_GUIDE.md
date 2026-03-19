# Crystal 3D Reconstruction — Usage Guide

This guide walks through the complete pipeline from first-time setup through to a finished 3D mesh. The pipeline is split across several Jupyter notebooks, each handling a distinct stage of the process.

---

## Overview of the Pipeline

The full workflow runs in this order:

| Step | Notebook / Script | What It Does |
| :--- | :--- | :--- |
| 1 | `install_dependencies.ipynb` | One-time install of all Python packages |
| 2 | `camera_capture.ipynb` | Calibrate the camera and capture 360 images of the crystal |
| 3 | `measure_scale.ipynb` | Compute the pixel-to-mm scale factor using ArUco markers |
| 4 | `process_real_crystal.ipynb` | Segment the crystal, reconstruct the 3D point cloud, and generate a mesh |

The `estimate_camera_angle.ipynb` notebook is an optional utility for verifying the camera's viewing angle relative to the stage.

---

## Step 1 — Install Dependencies

Open `notebooks/install_dependencies.ipynb` and run all cells. This installs every required package into your virtual environment, including:

- `opencv-python` and `opencv-contrib-python` — image processing and ArUco marker detection
- `open3d` — 3D point cloud processing and Poisson surface reconstruction
- `trimesh` and `pymeshfix` — mesh repair and watertight mesh generation
- `pyvista` — 3D visualisation
- `segment-anything` — Meta's SAM model for crystal segmentation (installed from GitHub)
- `torch` and `torchvision` — required by SAM

> **Note:** The `genicam` and `harvesters` packages are for the camera acquisition hardware. If you are working only with pre-captured images, these can be skipped.

After running, you can verify the install by running the test cell at the bottom of the notebook.

---

## Step 2 — Camera Calibration and Image Capture

Open `notebooks/camera_capture.ipynb`.

### 2a. Camera Calibration

The calibration corrects for lens distortion. It uses a physical checkerboard target (Edmund Optics part 15979 — 80×80mm glass checkerboard, 1.4mm squares).

At the top of the notebook, two flags control what runs:

```python
captureImages = False    # Set to True to capture new calibration images
runCalibration = False   # Set to True to compute calibration from images
```

To calibrate from scratch:

1. Set `captureImages = True` and run the cell. The Zaber stage will rotate and the camera will capture checkerboard images, saving them to the `cal_images2/` folder.
2. Set `captureImages = False` and `runCalibration = True`, then run again. The calibration result is saved as `cal_images2/Camera_Calibration_0.125.pickle`.

If you already have calibration images (the 11 `.png` files in `calibration/images/` are a reference set), you can skip capture and run calibration only.

### 2b. Crystal Image Capture

Further down in the same notebook, the crystal capture section is controlled by:

```python
captureCrystalImages = True
outputFolder = 'real_crystal_ring_white_light'   # Name of the output folder
stepSize = 1                                      # Degrees per step (1 = 360 images)
```

Set `captureCrystalImages = True`, choose a descriptive `outputFolder` name, and run. The stage will rotate 360° in 1° increments, capturing one image per step. Images are saved as `crystal_0000.jpg`, `crystal_0001.jpg`, etc.

> **Important:** The output folder is created inside `notebooks/` — this is where `process_real_crystal.ipynb` will look for it.

---

## Step 3 — Measure Scale (Optional but Recommended)

Open `notebooks/measure_scale.ipynb`.

This notebook uses the ArUco markers physically attached to the rotation stage to compute the precise pixel-to-millimetre scale factor. The markers have known real-world coordinates (defined in the notebook and illustrated in `docs/ArUcoCoordinates.png`).

The key setting is:

```python
imageFolderPath = 'real_crystal_ring_white_light'   # Must match your capture folder
```

Run all cells. The notebook will detect the ArUco markers in a sample image and compute the scale factor, which is then used in the reconstruction step to produce correctly-scaled output in millimetres.

---

## Step 4 — 3D Reconstruction

Open `notebooks/process_real_crystal.ipynb`. This is the main pipeline notebook.

### 4a. Configuration

At the top of the first code cell, set these variables to match your dataset:

```python
imageFolderPath = 'real_crystal_ring_white_light'   # Your image folder name
resolution = np.array([5328, 4608])                 # Camera resolution (pixels)
scaleFactor_default = (1/8)                         # Downsample factor (1/8 = 12.5% of full res)
zAxisStepSize = 5                                   # Z-axis resolution of the 3D block
calibrationFilePath = './cal_images2/Camera_Calibration_0.125.pickle'
```

Also set the SAM model:

```python
checkpointPath = 'sam_vit_b_01ec64.pth'   # ViT-B (faster, less accurate)
modelType = 'vit_b'
# OR:
checkpointPath = 'sam_vit_l_0b3195.pth'   # ViT-L (slower, more accurate)
modelType = 'vit_l'
```

Both `.pth` files must be placed in the `notebooks/` folder. See the main README for download links.

### 4b. What Each Section Does

The notebook is divided into clearly labelled sections. Here is what each one does:

**Load Images** — Defines the `load_image(degrees)` function, which reads `crystal_XXXX.jpg` from your image folder, resizes it by `scaleFactor_default`, and returns it as a numpy array.

**SAM Segmentation** — Loads the SAM model and defines `make_mask(img)`. For each image, SAM segments the crystal from the background using a centre-point prompt (it assumes the crystal is roughly centred in the frame) and a bounding box covering 20%–80% of the image width and 20%–85% of the height. The output is a binary black-and-white mask.

**Contour Detection** — The `get_boundary(img)` function traces the outer edge of the crystal mask using OpenCV's `findContours`. It returns the contour as a set of (x, y) pixel coordinates, centred about the image midpoint.

**3D Block Carving** — This is the core reconstruction step. It initialises a 3D voxel block and iterates through every 5° of rotation (configurable via `angleStep`). For each angle, it loads the image, generates the mask, extracts the contour, and "carves" away the voxels outside the crystal's silhouette. After all 360° are processed, the remaining voxels form the 3D shape of the crystal.

**Scale Calibration** — The voxel coordinates (in pixels) are converted to millimetres using the scale factor loaded from the calibration pickle file.

**Point Cloud Export** — The scaled voxel block is written to a `.asc` file (space-delimited XYZ point cloud). The filename is auto-generated from your settings, e.g. `real_crystal_ring_white_light-z=5,s=0.125.asc`.

**Open3D Meshing** — The point cloud is loaded into Open3D, normals are estimated, and a surface mesh is generated using **Poisson surface reconstruction**. The mesh is saved as `_mesh.ply`.

**Mesh Repair** — The mesh is loaded into trimesh, which fixes inverted normals, fills holes, and produces a watertight mesh. This is saved as `_trimesh_fixed.ply`. The notebook also reports the mesh volume, surface area, and whether it is watertight.

### 4c. Output Files

All output files are written to the `notebooks/` directory:

| File | Description |
| :--- | :--- |
| `real_crystal_ring_white_light-z=5,s=0.125.asc` | Raw point cloud (XYZ in mm) |
| `real_crystal_ring_white_light-z=5,s=0.125_mesh.ply` | Poisson surface mesh |
| `real_crystal_ring_white_light-z=5,s=0.125_trimesh_fixed.ply` | Repaired watertight mesh |

These `.ply` files can be opened in **MeshLab**, **CloudCompare**, or **Blender** for visualisation and further processing.

---

## Key Parameters to Tune

If results are poor, these are the first things to adjust:

| Parameter | Location | Effect |
| :--- | :--- | :--- |
| `scaleFactor_default` | `process_real_crystal.ipynb` cell 1 | Lower = faster but less detail. `1/8` is a good starting point; try `1/4` for higher quality |
| `zAxisStepSize` | `process_real_crystal.ipynb` cell 1 | Lower = finer Z resolution but much slower. `5` is a reasonable default |
| `angleStep` | `process_real_crystal.ipynb` cell 26 | Degrees between reconstruction samples. `5` uses every 5th image; `1` uses all 360 |
| SAM bounding box | `make_mask()` function | Adjust `x1, x2, y1, y2` if the crystal is not centred or is being clipped |
| `scaling *= 0.88` | `scale_coords()` function | A manual correction factor noted as "suspicious" in the code — may need recalibration |

---

## Notes on the Prototype Notebooks

The `notebooks/crystal_rotation/` folder contains earlier prototype work using a synthetic 3D bunny model instead of a real crystal. These notebooks (`Bunny_outline.ipynb`, `process_osc_3d.ipynb`, `rotate_osc_3d.ipynb`) are useful for understanding the underlying geometry and testing the reconstruction algorithm without needing hardware. They are not part of the main workflow but are worth reading if you want to understand how the carving algorithm was developed.

---

## Common Issues

**"ERROR: Image file not found"** — The `imageFolderPath` variable does not match the name of your image folder, or the folder is not inside `notebooks/`.

**"ERROR: could not find calibration file"** — The `cal_images2/` folder or the `.pickle` file inside it is missing. Run the calibration step first.

**SAM produces a poor mask** — The crystal may not be centred, or the lighting may be causing issues. Try adjusting the bounding box in `make_mask()` or adding background hint points to the `pointList`.

**Mesh has holes or is not watertight** — Try reducing `zAxisStepSize` to `1` or `2` for a denser point cloud, or increase the Poisson reconstruction depth parameter in the Open3D meshing cell.
