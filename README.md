# crystal-3d-recon

A photogrammetric 3D reconstruction pipeline for crystals. The system captures a series of photographs of a crystal rotating on a motorised stage (Zaber XRSW60AE03), then uses computer vision and the Meta Segment Anything Model (SAM) to reconstruct a 3D point cloud and surface mesh of the crystal.

---

## How It Works

1. **Image Capture** — `notebooks/camera_capture.ipynb` controls the Zaber rotary stage and a GenICam-compatible camera (e.g. Basler) to capture one image per degree of rotation (360 images total).
2. **Camera Calibration** — `scripts/camera_calibration.py` computes lens distortion parameters from a checkerboard calibration target. Results are saved as a `.pickle` file.
3. **3D Reconstruction** — `notebooks/process_real_crystal.ipynb` loads the image set, segments the crystal from the background using SAM, and reconstructs a 3D point cloud. The point cloud is then meshed using Open3D and trimesh.
4. **Visualisation** — Output meshes are saved as `.ply` and `.asc` files, viewable in any 3D viewer (e.g. MeshLab, CloudCompare, or the included notebooks).

---

## Project Structure

```
crystal-3d-recon/
├── notebooks/                  # Jupyter notebooks (main workflow)
│   ├── process_real_crystal.ipynb      # Main reconstruction pipeline
│   ├── process_pseudo_crystal.ipynb    # Reconstruction on synthetic data
│   ├── camera_capture.ipynb            # Image acquisition from camera + stage
│   ├── measure_scale.ipynb             # Scale measurement from ArUco markers
│   ├── estimate_camera_angle.ipynb     # Camera angle estimation
│   ├── install_dependencies.ipynb      # Dependency installation helper
│   └── crystal_rotation/               # Earlier prototype notebooks
├── scripts/                    # Standalone Python scripts
│   ├── camera_calibration.py           # Camera calibration
│   ├── camera_scale.py                 # Scale factor computation
│   ├── undistort.py                    # Image undistortion
│   ├── dense_reconstruction.py         # Dense point cloud reconstruction
│   ├── obj3d.py                        # 3D object utilities
│   └── show_bunny.py                   # 3D bunny visualisation (prototype)
├── calibration/                # Calibration data
│   ├── camera_calibration.csv          # Calibration results
│   ├── K.txt                           # Camera intrinsic matrix
│   └── images/                         # Checkerboard calibration images
├── markers/                    # ArUco marker images (IDs 0–15)
└── docs/                       # Reference images and notes
```

---

## Image Data

**Image datasets are not stored in this repository** — they are too large (each full 360° set is ~4 GB). Place your image folders alongside the notebooks before running the pipeline. The expected folder naming convention is:

```
notebooks/
├── real_crystal_ring_white_light/
│   ├── crystal_0000.jpg
│   ├── crystal_0001.jpg
│   └── ... (crystal_0000.jpg to crystal_0359.jpg)
├── real_crystal_iso_white_light/
└── ...
```

The active image folder is set at the top of each notebook via the `imageFolderPath` variable.

---

## SAM Model Weights

The Segment Anything Model checkpoint file is also not stored in this repository. Download it and place it in the `notebooks/` directory before running `process_real_crystal.ipynb`:

```bash
# ViT-B (smaller, faster — recommended for most use)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P notebooks/

# ViT-L (larger, more accurate)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P notebooks/
```

---

## Setup

### Requirements

- Python 3.9 or newer
- NI-DAQmx driver (if using NI hardware)
- Zaber Motion Library (for stage control)
- GenICam-compatible camera SDK (e.g. Basler Pylon, for image capture)

### Installation

```bash
git clone https://github.com/Slapgravel/crystal-3d-recon
cd crystal-3d-recon

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Install SAM from source
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Running the Notebooks

```bash
jupyter notebook
```

Open `notebooks/process_real_crystal.ipynb` to run the main reconstruction pipeline.

---

## Hardware

| Component | Model |
| :--- | :--- |
| Rotary stage | Zaber XRSW60AE03 |
| Camera | GenICam-compatible (e.g. Basler) |
| Illumination | Ring light (white) |

---

## Output Files

The pipeline generates the following output files in the `notebooks/` directory:

| Extension | Description |
| :--- | :--- |
| `.asc` | Raw point cloud (space-delimited XYZ) |
| `_mesh.ply` | Triangulated surface mesh |
| `_trimesh_fixed.ply` | Watertight, repaired mesh (via pymeshfix) |
