# MillingAround

Python utilities for studying tool/workpiece interactions in point milling. The codebase collects the collision-detection routines from the paper **Collision-free Tool Motion Planning for 5-Axis CNC Machining with Toroidal Cutters** (https://doi.org/10.1016/j.cad.2024.103725, https://bird.bcamath.org/handle/20.500.11824/1842) covering cylinders versus triangle meshes, and the tooling used to model toroidal/flat-end cutters moving over bicubic Bézier patches. Jupyter notebooks and precomputed media illustrate the workflows.

## Repository Layout
- `CylinderCollisionDetection.py` – collision detection between multiple cylinders and a triangle mesh using Open3D raycasting. Provides `MultipleCylinders` plus the `CollisionInformation` wrapper returned by `collision_detection_no_gaps` and `collision_detection_no_gaps_detect_all_cylinders`.
- `point_milling_backend.py` – low-level geometry for point milling.
- `point_milling_frontend.py` – user-facing helpers.
- `MeshGenerator.py` – trimesh-based primitives (cylinders, cones, pipes, disks, control nets, arrows) plus helpers to export meshes and build meshes from the backend surface.
- Notebooks: `example_CylinderCollisionDetection.ipynb`, `examples_point_milling_frontend.ipynb`, `generate_frames.ipynb` demonstrate the collision routines, milling envelope construction, and frame generation.
- `video_stuff/` – generated media (STLs, PNGs, MP4) used in the demos.

## Animation
Animation of the collision-detection algorithm. A translucent cylinder moves over the metal surface: starting at the top of the medial axis, the first footpoint is computed (cyan) and the first safe ball is constructed (green), then the process iterates from the boundary of the safe ball. It stops when a footpoint lies below the cutting plane (black circle), finalizing that cylinder.
![Collision detection animation](video_stuff/collision_detection.gif)

## Documentation
For a browsable version of the API docs, open `docs/build/api/index.html`.

## Installation
Everything has been tested on a MacBook Pro with an Apple M1 Pro chip running Python 3.11.7.
To replicate the envoronment, run the following:
   ```bash
   conda env create -f environment.yml
   conda activate millingaround
   ```

For examples of use, refer yourself to the notebooks.
