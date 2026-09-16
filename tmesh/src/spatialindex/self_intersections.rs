use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::math::{Pose3, Vec3};
use parry3d_f64::partitioning::{Bvh, BvhBuildStrategy};
use parry3d_f64::query::intersection_test;
use parry3d_f64::shape::Triangle;

use crate::mesh::{BoundaryMesh3d, Mesh, Simplex};

/// Finds all pairs of intersecting triangles in a mesh, returning their indices.
#[must_use]
pub fn find_self_intersections(msh: &BoundaryMesh3d) -> Vec<(usize, usize)> {
    let mut intersecting_pairs = Vec::new();

    // 1. Compute an AABB for every triangle
    let mut aabbs = Vec::with_capacity(msh.n_elems());
    for ge in msh.gelems() {
        // Parry 0.26+ now accepts raw vectors directly (no Point3 conversion needed)
        let v0 = Vec3::from_slice(ge[0].as_slice());
        let v1 = Vec3::from_slice(ge[1].as_slice());
        let v2 = Vec3::from_slice(ge[2].as_slice());

        aabbs.push(Aabb::from_points([v0, v1, v2]));
    }

    // 2. Build the BVH (Bounding Volume Hierarchy) tree
    let bvh = Bvh::from_leaves(BvhBuildStrategy::default(), &aabbs);

    // 3. Query the BVH for intersections
    for (i, elem) in msh.elems().enumerate() {
        let ge = msh.gelem(&elem);
        let v0_a = Vec3::from_slice(ge[0].as_slice());
        let v1_a = Vec3::from_slice(ge[1].as_slice());
        let v2_a = Vec3::from_slice(ge[2].as_slice());

        let aabb_a = Aabb::from_points([v0_a, v1_a, v2_a]);
        let parry_t1 = Triangle::new(v0_a, v1_a, v2_a);

        // 4. Narrow Phase: Run Parry's intersection_test on candidates
        for leaf_id in bvh.intersect_aabb(&aabb_a) {
            let j = leaf_id as usize;

            // Only check i < j to avoid duplicate checks and self-checks
            if i < j {
                let elem_j = msh.elem(j);

                // Discard triangles that are simply neighbors in the mesh
                if share_vertices(elem.as_slice(), elem_j.as_slice()) {
                    continue;
                }
                let ge = msh.gelem(&elem_j);
                let v0_b = Vec3::from_slice(ge[0].as_slice());
                let v1_b = Vec3::from_slice(ge[1].as_slice());
                let v2_b = Vec3::from_slice(ge[2].as_slice());
                let parry_t2 = Triangle::new(v0_b, v1_b, v2_b);

                // Run the exact intersection test using nalgebra's Isometry3
                let intersects =
                    intersection_test(&Pose3::identity(), &parry_t1, &Pose3::identity(), &parry_t2)
                        .unwrap_or(false);

                if intersects {
                    intersecting_pairs.push((i, j));
                }
            }
        }
    }

    intersecting_pairs
}

/// Helper function to ignore triangles that share vertices
fn share_vertices(t1: &[usize], t2: &[usize]) -> bool {
    for &v1 in t1 {
        for &v2 in t2 {
            if v1 == v2 {
                return true;
            }
        }
    }
    false
}
