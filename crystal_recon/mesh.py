"""
mesh.py — Point cloud to surface mesh pipeline.

Takes a numpy point cloud and produces:
  1. A Poisson surface mesh (.ply) via Open3D, with density-based trimming
  2. A repaired, watertight mesh (_trimesh_fixed.ply) via trimesh

The Open3D mesh is converted to trimesh in memory, avoiding a wasteful
disk round-trip between the two libraries.
"""

import os
import time

import numpy as np


def build_mesh(point_cloud: np.ndarray, output_base_path: str,
               poisson_depth: int = 9,
               density_quantile: float = 0.05,
               visualize: bool = False) -> dict:
    """
    Build a surface mesh from a 3D point cloud.

    Pipeline:
      1. Load point cloud into Open3D
      2. Remove duplicate points
      3. Estimate surface normals (outward-facing)
      4. Poisson surface reconstruction
      5. Trim low-density vertices (removes the "skirt" around the edges)
      6. Save raw mesh as .ply
      7. Convert to trimesh in memory (no disk round-trip)
      8. Repair mesh (fix normals, fill holes)
      9. Save repaired watertight mesh as _trimesh_fixed.ply

    Args:
        point_cloud:       (N, 3) numpy array of (x, y, z) coordinates in mm.
        output_base_path:  Base path for output files (without extension).
                           e.g. "output/real_crystal_ring_white_light-z=5,s=0.125"
        poisson_depth:     Poisson reconstruction depth. Higher = more detail
                           but slower and more memory.
                             8  = fast, lower detail (good for testing)
                             9  = default, good balance
                             10 = high detail (final quality runs)
                             12 = maximum detail (slow, high RAM)
        density_quantile:  Fraction of lowest-density vertices to trim after
                           Poisson reconstruction. These are the low-confidence
                           surface regions that form around the edges of the
                           object. 0.05 = trim the bottom 5% by density.
                           Set to 0.0 to disable trimming.
        visualize:         If True, open the Open3D viewer after meshing.

    Returns:
        Dictionary with keys:
          "mesh_file":        path to raw .ply mesh
          "fixed_mesh_file":  path to repaired .ply mesh
          "is_watertight":    bool
          "volume_mm3":       float or None (volume in mm³, if watertight)
          "surface_area_mm2": float (surface area in mm²)
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
    pcd.points = o3d.utility.Vector3dVector(np.asarray(point_cloud, dtype=float))
    pcd.remove_duplicated_points()
    print(f"  {len(pcd.points):,} points after deduplication")

    # -----------------------------------------------------------------------
    # 2. Estimate normals (outward-facing)
    # -----------------------------------------------------------------------
    print("Estimating normals...")
    pcd.estimate_normals(fast_normal_computation=False)

    # Orient normals to point outward from the centre of mass
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    center = pcd.get_center()
    outward = points - center
    flip = np.einsum("ij,ij->i", outward, normals) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # -----------------------------------------------------------------------
    # 3. Poisson surface reconstruction
    # -----------------------------------------------------------------------
    print(f"Running Poisson reconstruction (depth={poisson_depth})...")
    t_start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=True
    )
    print(f"  Poisson complete in {time.time() - t_start:.1f}s  "
          f"({len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles)")

    # -----------------------------------------------------------------------
    # 4. Density-based trimming
    # -----------------------------------------------------------------------
    if density_quantile > 0.0:
        densities_np = np.asarray(densities)
        threshold = np.quantile(densities_np, density_quantile)
        vertices_to_remove = densities_np < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        print(f"  After density trim (q={density_quantile}): "
              f"{len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    mesh.compute_vertex_normals()

    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)

    # -----------------------------------------------------------------------
    # 5. Save raw mesh
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_base_path) or ".", exist_ok=True)
    mesh_file = output_base_path + "_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_file, mesh)
    print(f"  Raw mesh saved: {mesh_file}")

    # -----------------------------------------------------------------------
    # 6. Convert to trimesh in memory (no disk round-trip)
    # -----------------------------------------------------------------------
    print("Repairing mesh...")
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # -----------------------------------------------------------------------
    # 7. Repair with trimesh
    # -----------------------------------------------------------------------
    tri_mesh.fix_normals()
    tri_mesh.fill_holes()

    fixed_mesh_file = output_base_path + "_trimesh_fixed.ply"
    tri_mesh.export(fixed_mesh_file)
    print(f"  Repaired mesh saved: {fixed_mesh_file}")

    is_watertight = tri_mesh.is_watertight
    volume = tri_mesh.volume if is_watertight else None
    surface_area = tri_mesh.area

    print(f"  Watertight:   {is_watertight}")
    if volume is not None:
        print(f"  Volume:       {volume:.2f} mm³")
    print(f"  Surface area: {surface_area:.2f} mm²")

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
        block:       (N, 3) numpy array of (x, y, z) coordinates.
        output_path: Full path to the output .asc file.

    Returns:
        The output_path that was written.
    """
    block = np.asarray(block)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savetxt(output_path, block, fmt="%.6f", delimiter=" ")
    print(f"Point cloud saved: {output_path} ({len(block):,} points)")
    return output_path
