"""
mesh.py — Point cloud to surface mesh pipeline.

Takes a numpy point cloud and produces:
  1. A Poisson surface mesh (.ply) via Open3D
  2. A repaired, watertight mesh (_trimesh_fixed.ply) via trimesh + pymeshfix
"""

import os
import time

import numpy as np


def build_mesh(point_cloud: np.ndarray, output_base_path: str,
               poisson_depth: int = 12, visualize: bool = False) -> dict:
    """
    Build a surface mesh from a 3D point cloud.

    Pipeline:
      1. Load point cloud into Open3D
      2. Remove duplicate points
      3. Estimate surface normals (outward-facing)
      4. Poisson surface reconstruction
      5. Save raw mesh as .ply
      6. Repair mesh with trimesh (fix normals, fill holes)
      7. Save repaired watertight mesh as _trimesh_fixed.ply

    Args:
        point_cloud:      (N, 3) numpy array of (x, y, z) coordinates in mm.
        output_base_path: Base path for output files (without extension).
                          e.g. "output/real_crystal_ring_white_light-z=5,s=0.125"
        poisson_depth:    Poisson reconstruction depth. Higher = more detail
                          but slower and more memory. 8–12 is typical.
        visualize:        If True, open the Open3D viewer after meshing.

    Returns:
        Dictionary with keys:
          "point_cloud_file": path to .asc file
          "mesh_file":        path to raw .ply mesh
          "fixed_mesh_file":  path to repaired .ply mesh
          "is_watertight":    bool
          "volume_mm3":       float (volume in cubic mm, if watertight)
          "surface_area_mm2": float (surface area in square mm)
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is not installed. Run: pip install open3d")

    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is not installed. Run: pip install trimesh")

    # -----------------------------------------------------------------------
    # 1. Load into Open3D
    # -----------------------------------------------------------------------
    print("Building point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(point_cloud))

    pcd.remove_duplicated_points()
    print(f"  {len(pcd.points):,} points after deduplication")

    # -----------------------------------------------------------------------
    # 2. Estimate normals (outward-facing)
    # -----------------------------------------------------------------------
    print("Estimating normals...")
    pcd.estimate_normals(fast_normal_computation=False)

    # Ensure normals point outward from the centre of mass
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    center = pcd.get_center()
    v = points - center

    for i in range(len(points)):
        if np.dot(v[i], normals[i]) < 0:
            normals[i] *= -1

    pcd.normals = o3d.utility.Vector3dVector(normals)

    # -----------------------------------------------------------------------
    # 3. Poisson surface reconstruction
    # -----------------------------------------------------------------------
    print(f"Running Poisson reconstruction (depth={poisson_depth})...")
    t_start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=True
    )
    t_elapsed = time.time() - t_start
    print(f"  Poisson reconstruction complete in {t_elapsed:.1f}s")

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.remove_degenerate_triangles()

    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)

    # -----------------------------------------------------------------------
    # 4. Save raw mesh
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_base_path) or ".", exist_ok=True)
    mesh_file = output_base_path + "_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_file, mesh)
    print(f"  Raw mesh saved: {mesh_file}")

    # -----------------------------------------------------------------------
    # 5. Repair mesh with trimesh
    # -----------------------------------------------------------------------
    print("Repairing mesh...")
    tri_mesh = trimesh.load(mesh_file, force="mesh")
    tri_mesh.fix_normals()
    tri_mesh.fill_holes()

    fixed_mesh_file = output_base_path + "_trimesh_fixed.ply"
    trimesh.exchange.export.export_mesh(tri_mesh, fixed_mesh_file)
    print(f"  Repaired mesh saved: {fixed_mesh_file}")

    is_watertight = tri_mesh.is_watertight
    volume = tri_mesh.volume if is_watertight else None
    surface_area = tri_mesh.area

    print(f"  Watertight: {is_watertight}")
    if volume is not None:
        print(f"  Volume:       {volume:.2f} mm³")
    print(f"  Surface area: {surface_area:.2f} mm²")
    if is_watertight and volume is not None:
        print(f"  Convex hull volume: {tri_mesh.convex_hull.volume:.2f} mm³")

    return {
        "mesh_file": mesh_file,
        "fixed_mesh_file": fixed_mesh_file,
        "is_watertight": is_watertight,
        "volume_mm3": volume,
        "surface_area_mm2": surface_area,
    }


def save_point_cloud(block: np.ndarray, output_path: str) -> str:
    """
    Save a point cloud to an ASC file (space-delimited XYZ).

    ASC files can be opened in MeshLab, CloudCompare, or any standard
    3D point cloud viewer.

    Args:
        block:       (N, 3) numpy matrix of (x, y, z) coordinates.
        output_path: Full path to the output .asc file.

    Returns:
        The output_path that was written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for pt in block:
            pt_arr = np.asarray(pt).flatten()
            f.write(f"{pt_arr[0]:.6f} {pt_arr[1]:.6f} {pt_arr[2]:.6f}\n")
    print(f"Point cloud saved: {output_path} ({len(block):,} points)")
    return output_path
