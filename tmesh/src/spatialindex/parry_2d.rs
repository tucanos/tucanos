use crate::{
    Vertex,
    mesh::{GSimplex, Mesh, Simplex},
};
use parry2d_f64::{
    bounding_volume::Aabb,
    math::{Pose, Vector},
    partitioning::{Bvh, BvhBuildStrategy},
    query::{
        PointProjection, PointQuery, PointQueryWithLocation, RayCast, details::NormalConstraints,
    },
    shape::{CompositeShape, CompositeShapeRef, Shape, TypedCompositeShape},
};

pub struct SimplexShape<const D: usize, C: GSimplex<D>>(C);

impl<const D: usize, C: GSimplex<D>> SimplexShape<D, C> {
    pub const fn new(c: C) -> Self {
        Self(c)
    }
}

impl<const D: usize, C: GSimplex<D>> PointQuery for SimplexShape<D, C> {
    fn project_local_point(&self, _pt: Vector, _solid: bool) -> PointProjection {
        todo!()
    }

    fn project_local_point_and_get_feature(
        &self,
        _pt: Vector,
    ) -> (PointProjection, parry2d_f64::shape::FeatureId) {
        todo!()
    }
}

impl<const D: usize, C: GSimplex<D>> PointQueryWithLocation for SimplexShape<D, C> {
    type Location = ();

    fn project_local_point_and_get_location(
        &self,
        pt: Vector,
        _solid: bool,
    ) -> (PointProjection, Self::Location) {
        let pt = Vertex::from_fn(|i, _| pt[i]);
        let (proj, is_inside) = self.0.project(&pt);
        (
            PointProjection {
                is_inside,
                point: Vector::from_slice(proj.as_slice()),
            },
            (),
        )
    }
}

impl<const D: usize, C: GSimplex<D>> RayCast for SimplexShape<D, C> {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &parry2d_f64::query::Ray,
        _max_time_of_impact: f64,
        _solid: bool,
    ) -> Option<parry2d_f64::query::RayIntersection> {
        todo!()
    }
}

impl<const D: usize, C: GSimplex<D>> Shape for SimplexShape<D, C> {
    fn compute_local_aabb(&self) -> Aabb {
        todo!()
    }

    fn compute_local_bounding_sphere(&self) -> parry2d_f64::bounding_volume::BoundingSphere {
        todo!()
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        todo!()
    }

    fn scale_dyn(&self, _scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        todo!()
    }

    fn mass_properties(&self, _density: f64) -> parry2d_f64::mass_properties::MassProperties {
        todo!()
    }

    fn shape_type(&self) -> parry2d_f64::shape::ShapeType {
        todo!()
    }

    fn as_typed_shape(&self) -> parry2d_f64::shape::TypedShape<'_> {
        todo!()
    }

    fn ccd_thickness(&self) -> f64 {
        todo!()
    }

    fn ccd_angular_thickness(&self) -> f64 {
        todo!()
    }
}

pub struct ObjectIndex2d<const D: usize, M: Mesh<D>> {
    mesh: M,
    tree: Bvh,
}

impl<const D: usize, M: Mesh<D>> ObjectIndex2d<D, M> {
    pub fn new(mesh: M) -> Self {
        assert_eq!(D, 2);

        let data = mesh.gelems().enumerate().map(|(i, ge)| {
            let (min, max) = ge.bounding_box();
            (
                i,
                Aabb::new(
                    Vector::from_slice(min.as_slice()),
                    Vector::from_slice(max.as_slice()),
                ),
            )
        });
        let tree = Bvh::from_iter(BvhBuildStrategy::Binned, data);

        Self { mesh, tree }
    }

    /// Get a reference to the mesh
    pub const fn mesh(&self) -> &M {
        &self.mesh
    }

    /// Get the index of the nearest element
    #[must_use]
    pub fn nearest_elem(&self, pt: &Vertex<D>) -> usize {
        let (_, (id, ())) =
            self.project_local_point_and_get_location(Vector::new(pt[0], pt[1]), true);
        id as usize
    }

    /// Project a point onto the nearest element
    #[must_use]
    pub fn project(&self, pt: &Vertex<D>) -> (f64, Vertex<D>) {
        let (p, _) = self.project_local_point_and_get_location(Vector::new(pt[0], pt[1]), true);
        let p = p.point;
        let res = Vertex::from_fn(|i, _j| p[i]);
        ((pt - res).norm(), res)
    }
}

impl<const D: usize, M: Mesh<D>> PointQueryWithLocation for ObjectIndex2d<D, M> {
    type Location = (u32, ());

    fn project_local_point_and_get_location(
        &self,
        point: Vector,
        solid: bool,
    ) -> (PointProjection, Self::Location) {
        let (seg_id, (proj, loc)) = CompositeShapeRef(self)
            .project_local_point_and_get_location(point, f64::MAX, solid)
            .unwrap();
        (proj, (seg_id, loc))
    }
}

impl<const D: usize, M: Mesh<D>> CompositeShape for ObjectIndex2d<D, M> {
    fn map_part_at(
        &self,
        shape_id: u32,
        f: &mut dyn FnMut(Option<&Pose>, &dyn Shape, Option<&dyn NormalConstraints>),
    ) {
        f(
            None,
            &SimplexShape::new(self.mesh.gelem(&self.mesh.elem(shape_id as usize))),
            None,
        );
    }

    fn bvh(&self) -> &Bvh {
        &self.tree
    }
}

impl<const D: usize, M: Mesh<D>> TypedCompositeShape for ObjectIndex2d<D, M> {
    type PartShape = SimplexShape<D, <M::C as Simplex>::GEOM<D>>;

    type PartNormalConstraints = dyn NormalConstraints;

    fn map_typed_part_at<T>(
        &self,
        shape_id: u32,
        mut f: impl FnMut(Option<&Pose>, &Self::PartShape, Option<&Self::PartNormalConstraints>) -> T,
    ) -> Option<T> {
        Some(f(
            None,
            &SimplexShape::new(self.mesh.gelem(&self.mesh.elem(shape_id as usize))),
            None,
        ))
    }

    fn map_untyped_part_at<T>(
        &self,
        shape_id: u32,
        mut f: impl FnMut(Option<&Pose>, &dyn Shape, Option<&dyn NormalConstraints>) -> T,
    ) -> Option<T> {
        Some(f(
            None,
            &SimplexShape::new(self.mesh.gelem(&self.mesh.elem(shape_id as usize))),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Vert2d, Vertex, assert_delta,
        mesh::{
            BoundaryMesh2d, Edge, GSimplex, Mesh, Mesh2d, QuadraticBoundaryMesh2d, circle_mesh,
            quadratic_circle_mesh, rectangle_mesh,
        },
        spatialindex::{ObjectIndex, parry_2d::ObjectIndex2d},
    };
    use parry2d_f64::math::Vector;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn nearest_elem_naive<const D: usize>(msh: &impl Mesh<D>, pt: &Vertex<D>) -> usize {
        let mut dst = f64::MAX;
        let mut res = 0;
        for (i, ge) in msh.gelems().enumerate() {
            let (p, _) = ge.project(pt);
            let d = (p - pt).norm_squared();
            if d < dst {
                res = i;
                dst = d;
            }
        }
        res
    }

    #[test]
    fn test_mesh2d() {
        let n = 5;
        let mesh = rectangle_mesh::<Mesh2d>(1.0, n, 1.0, n).random_shuffle();

        let tree = ObjectIndex2d::new(mesh.clone());

        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let pt = Vert2d::from_fn(|_, _| -0.5 + 2.0 * rng.random::<f64>());
            let i = tree.nearest_elem(&pt);
            let i2 = nearest_elem_naive(&mesh, &pt);
            if i != i2 {
                let (p, _) = mesh.gelem(&mesh.elem(i)).project(&pt);
                let d = (p - pt).norm();
                let (p2, _) = mesh.gelem(&mesh.elem(i2)).project(&pt);
                let d2 = (p2 - pt).norm();
                assert_delta!(d, d2, 1e-10);
            }
        }
    }

    #[test]
    fn test_boundarymesh2d() {
        let mesh = circle_mesh::<BoundaryMesh2d>(1.0, 20).random_shuffle();

        let tree = ObjectIndex2d::new(mesh.clone());

        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let pt = Vert2d::from_fn(|_, _| -2.0 + 4.0 * rng.random::<f64>());
            let i = tree.nearest_elem(&pt);
            let i2 = nearest_elem_naive(&mesh, &pt);
            if i != i2 {
                let (p, _) = mesh.gelem(&mesh.elem(i)).project(&pt);
                let d = (p - pt).norm();
                let (p2, _) = mesh.gelem(&mesh.elem(i2)).project(&pt);
                let d2 = (p2 - pt).norm();
                assert_delta!(d, d2, 1e-10);
            }
        }
    }

    #[test]
    fn test_quadraticboundarymesh2d() {
        let mesh = quadratic_circle_mesh::<QuadraticBoundaryMesh2d>(1.0, 10).random_shuffle();
        let tree = ObjectIndex2d::new(mesh.clone());

        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let pt = Vert2d::from_fn(|_, _| -2.0 + 4.0 * rng.random::<f64>());
            let i = tree.nearest_elem(&pt);
            let i2 = nearest_elem_naive(&mesh, &pt);
            if i != i2 {
                let (p, _) = mesh.gelem(&mesh.elem(i)).project(&pt);
                let d = (p - pt).norm();
                let (p2, _) = mesh.gelem(&mesh.elem(i2)).project(&pt);
                let d2 = (p2 - pt).norm();
                assert_delta!(d, d2, 1e-10);
            }
        }
    }

    #[test]
    fn test_edgs() {
        let coords = vec![
            Vert2d::new(0., 0.),
            Vert2d::new(1., 0.),
            Vert2d::new(1., 1.),
            Vert2d::new(0., 1.),
        ];
        let elems = vec![
            Edge::new(0, 1),
            Edge::new(1, 2),
            Edge::new(2, 3),
            Edge::new(3, 0),
        ];
        let etags = vec![1, 2, 3, 4];
        let faces = Vec::new();
        let ftags = Vec::new();

        let mesh = BoundaryMesh2d::from_vecs(coords, elems, etags, faces, ftags);

        let tree = ObjectIndex::new(mesh);

        let pt = Vert2d::new(1.5, 0.25);
        let (dist, p) = tree.project(&pt);
        assert!(f64::abs(dist - 0.5) < 1e-12);
        assert!((p - Vert2d::new(1., 0.25)).norm() < 1e-12);
    }

    #[test]
    fn test_gtriangle() {
        use parry2d_f64::query::PointQuery;
        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..100 {
            let to_project = Vert2d::from_fn(|_, _| 10.0 * (rng.random::<f64>() - 0.5));
            let t_v = std::array::from_fn(|_| Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5));
            let p_v = t_v.map(|x| Vector::from_slice(x.as_slice()));
            let shape = parry2d_f64::shape::Triangle::new(p_v[0], p_v[1], p_v[2]);
            let p_p = shape.project_local_point(Vector::from_slice(to_project.as_slice()), true);
            let (proj, is_inside) = crate::mesh::GTriangle::from(t_v).project(&to_project);
            let proj = Vector::from_slice(proj.as_slice());
            assert_eq!(is_inside, p_p.is_inside);
            let d = (proj - p_p.point).length();
            assert_delta!(d, 0.0, 1e-10);
        }
    }
}
