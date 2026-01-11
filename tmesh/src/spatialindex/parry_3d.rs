use crate::{
    Vertex,
    mesh::{GSimplex, Mesh, Simplex},
};
use parry3d_f64::{
    bounding_volume::Aabb,
    math::{Isometry, Point},
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
    fn project_local_point(&self, _pt: &Point<f64>, _solid: bool) -> PointProjection {
        todo!()
    }

    fn project_local_point_and_get_feature(
        &self,
        _pt: &Point<f64>,
    ) -> (PointProjection, parry3d_f64::shape::FeatureId) {
        todo!()
    }
}

impl<const D: usize, C: GSimplex<D>> PointQueryWithLocation for SimplexShape<D, C> {
    type Location = ();

    fn project_local_point_and_get_location(
        &self,
        pt: &Point<f64>,
        _solid: bool,
    ) -> (PointProjection, Self::Location) {
        let pt = Vertex::from_fn(|i, _| pt[i]);
        let (proj, is_inside) = self.0.project(&pt);
        (
            PointProjection {
                is_inside,
                point: Point::from_slice(proj.as_slice()),
            },
            (),
        )
    }
}

impl<const D: usize, C: GSimplex<D>> RayCast for SimplexShape<D, C> {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &parry3d_f64::query::Ray,
        _max_time_of_impact: f64,
        _solid: bool,
    ) -> Option<parry3d_f64::query::RayIntersection> {
        todo!()
    }
}

impl<const D: usize, C: GSimplex<D>> Shape for SimplexShape<D, C> {
    fn compute_local_aabb(&self) -> Aabb {
        todo!()
    }

    fn compute_local_bounding_sphere(&self) -> parry3d_f64::bounding_volume::BoundingSphere {
        todo!()
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        todo!()
    }

    fn scale_dyn(
        &self,
        _scale: &parry3d_f64::math::Vector<f64>,
        _num_subdivisions: u32,
    ) -> Option<Box<dyn Shape>> {
        todo!()
    }

    fn mass_properties(&self, _density: f64) -> parry3d_f64::mass_properties::MassProperties {
        todo!()
    }

    fn shape_type(&self) -> parry3d_f64::shape::ShapeType {
        todo!()
    }

    fn as_typed_shape(&self) -> parry3d_f64::shape::TypedShape<'_> {
        todo!()
    }

    fn ccd_thickness(&self) -> f64 {
        todo!()
    }

    fn ccd_angular_thickness(&self) -> f64 {
        todo!()
    }
}

pub struct ObjectIndex3d<const D: usize, M: Mesh<D>> {
    mesh: M,
    tree: Bvh,
}

impl<const D: usize, M: Mesh<D>> ObjectIndex3d<D, M> {
    pub fn new(mesh: M) -> Self {
        assert_eq!(D, 3);
        let data = mesh.gelems().enumerate().map(|(i, ge)| {
            let (min, max) = ge.bounding_box();
            (
                i,
                Aabb::new(
                    Point::from_slice(min.as_slice()),
                    Point::from_slice(max.as_slice()),
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
            self.project_local_point_and_get_location(&Point::new(pt[0], pt[1], pt[2]), true);
        id as usize
    }

    /// Project a point onto the nearest element
    #[must_use]
    pub fn project(&self, pt: &Vertex<D>) -> (f64, Vertex<D>) {
        let (p, _) =
            self.project_local_point_and_get_location(&Point::new(pt[0], pt[1], pt[2]), true);
        let p = p.point;
        let res = Vertex::from_fn(|i, _j| p[i]);
        ((pt - res).norm(), res)
    }
}

impl<const D: usize, M: Mesh<D>> PointQueryWithLocation for ObjectIndex3d<D, M> {
    type Location = (u32, ());

    fn project_local_point_and_get_location(
        &self,
        point: &Point<f64>,
        solid: bool,
    ) -> (PointProjection, Self::Location) {
        let (seg_id, (proj, loc)) = CompositeShapeRef(self)
            .project_local_point_and_get_location(point, f64::MAX, solid)
            .unwrap();
        (proj, (seg_id, loc))
    }
}

impl<const D: usize, M: Mesh<D>> CompositeShape for ObjectIndex3d<D, M> {
    fn map_part_at(
        &self,
        shape_id: u32,
        f: &mut dyn FnMut(Option<&Isometry<f64>>, &dyn Shape, Option<&dyn NormalConstraints>),
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

impl<const D: usize, M: Mesh<D>> TypedCompositeShape for ObjectIndex3d<D, M> {
    type PartShape = SimplexShape<D, <M::C as Simplex>::GEOM<D>>;

    type PartNormalConstraints = dyn NormalConstraints;

    fn map_typed_part_at<T>(
        &self,
        shape_id: u32,
        mut f: impl FnMut(
            Option<&Isometry<f64>>,
            &Self::PartShape,
            Option<&Self::PartNormalConstraints>,
        ) -> T,
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
        mut f: impl FnMut(Option<&Isometry<f64>>, &dyn Shape, Option<&dyn NormalConstraints>) -> T,
    ) -> Option<T> {
        Some(f(
            None,
            &SimplexShape::new(self.mesh.gelem(&self.mesh.elem(shape_id as usize))),
            None,
        ))
    }
}

#[cfg(test)]
pub mod tests {
    use std::f64::consts::PI;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        Vert3d, Vertex, assert_delta,
        mesh::{
            BoundaryMesh3d, GSimplex, Mesh, Mesh3d, QuadraticBoundaryMesh3d, Triangle, box_mesh,
            quadratic_sphere_mesh, sphere_mesh,
        },
        spatialindex::parry_3d::ObjectIndex3d,
    };
    use nalgebra::Point3;

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
    fn test_mesh3d() {
        let n = 5;
        let mesh = box_mesh::<Mesh3d>(1.0, n, 1.0, n, 1.0, n).random_shuffle();

        let tree = ObjectIndex3d::new(mesh.clone());

        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let pt = Vert3d::from_fn(|_, _| -0.5 + 2.0 * rng.random::<f64>());
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
    fn test_boundarymesh3d() {
        let mesh = sphere_mesh::<BoundaryMesh3d>(1.0, 4).random_shuffle();

        let tree = ObjectIndex3d::new(mesh.clone());

        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let pt = Vert3d::from_fn(|_, _| -2.0 + 4.0 * rng.random::<f64>());
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
    fn test_quadraticboundarymesh3d() {
        let mesh = quadratic_sphere_mesh::<QuadraticBoundaryMesh3d>(1.0, 1).random_shuffle();
        let tree = ObjectIndex3d::new(mesh.clone());

        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let pt = Vert3d::from_fn(|_, _| -2.0 + 4.0 * rng.random::<f64>());
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
    fn test_proj() {
        let r_in = 1.;
        let r_out = 1000.;
        let n = 100;
        let dim = 3;

        let mut coords = Vec::with_capacity(2 * n * dim);
        for i in 0..n {
            let theta = 2.0 * PI * i as f64 / n as f64;
            coords.push(Vert3d::new(
                r_in * f64::cos(theta),
                r_in * f64::sin(theta),
                0.0,
            ));
            coords.push(Vert3d::new(
                r_out * f64::cos(theta),
                r_out * f64::sin(theta),
                0.0,
            ));
        }

        let mut tris = Vec::with_capacity(2 * n);
        for i in 0..n - 1 {
            tris.push(Triangle::new(2 * i, 2 * i + 1, 2 * i + 2));
            tris.push(Triangle::new(2 * i + 2, 2 * i + 1, 2 * i + 3));
        }
        tris.push(Triangle::new(2 * n - 2, 2 * n - 1, 0));
        tris.push(Triangle::new(0, 2 * n - 1, 1));

        let tri_tags = vec![1; 2 * n];

        let msh = BoundaryMesh3d::from_vecs(coords, tris, tri_tags, Vec::new(), Vec::new());
        // msh.write_vtk("dbg.vtu", None, None);

        let tree = ObjectIndex3d::new(msh);

        let pt = Vert3d::new(-11.134_680_738_954_373, 10.371_256_484_784_858, 0.);
        assert_eq!(tree.nearest_elem(&pt), 76);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 2e-11, "{d:.2e} vs 0");

        let pt = Vert3d::new(-360., -105., 0.);
        assert_eq!(tree.nearest_elem(&pt), 109);

        let pt = Vert3d::new(41.905, -7.933, 0.);
        assert_eq!(tree.nearest_elem(&pt), 194);

        let pt = Vert3d::new(977.405_622_304_933_2, -193.219_725_123_763_82, 0.);
        assert_eq!(tree.nearest_elem(&pt), 193);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d:.2e} vs 0");

        let pt = Vert3d::new(732.254_535_699_460_3, 628.314_474_637_604_1, 0.);
        assert_eq!(tree.nearest_elem(&pt), 23);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 2e-12, "{d:.2e} vs 0");

        let pt = Vert3d::new(41.905_036_870_164_33, -7.932_967_693_525_678, 0.);
        assert_eq!(tree.nearest_elem(&pt), 194);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 4e-12, "{d:.2e} vs 0");

        let mut rng = StdRng::seed_from_u64(0);
        let mut worst = (0., None);
        // let start = Instant::now();
        let num_proj = 10000;
        for _ in 0..num_proj {
            let tmp = Vert3d::from_fn(|_, _| rng.random());
            let theta = 2.0 * PI * tmp[0];
            let r = r_in + tmp[1] * (r_out * 0.999 - r_in);
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            let z = r_out * (tmp[2] - 0.5);
            let pt = Vert3d::new(x, y, z);
            let (d, pt_proj) = tree.project(&pt);
            assert!(f64::abs(d - z.abs()) < 1e-12);
            for (i, v) in [x, y, 0.].iter().enumerate() {
                let d = f64::abs(pt_proj[i] - v);
                if d > worst.0 {
                    worst.0 = d;
                    worst.1 = Some(pt_proj);
                    // println!("{pt:?} -> {pt_proj:?}, {d:e}");
                }
                assert!(f64::abs(pt_proj[i] - v) < 3e-10, "{} != {}", pt_proj[i], v);
            }
        }
        // println!("Time by projection: {:?}", start.elapsed() / num_proj);
    }

    pub fn test_gsimplex<const N: usize, G, P, F>(build_parry_shape: F)
    where
        G: GSimplex<3> + From<[Vert3d; N]>,
        P: parry3d_f64::query::PointQuery,
        F: Fn([Point3<f64>; N]) -> P,
    {
        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..100 {
            let to_project = Vert3d::from_fn(|_, _| 10.0 * (rng.random::<f64>() - 0.5));
            let t_v = std::array::from_fn(|_| Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5));
            let p_v = t_v.map(|x| Point3::from_slice(x.as_slice()));
            let shape = build_parry_shape(p_v);
            let p_p = shape.project_local_point(&Point3::from_slice(to_project.as_slice()), true);
            let (proj, is_inside) = G::from(t_v).project(&to_project);
            let proj = Point3::from_slice(proj.as_slice());
            assert_eq!(is_inside, p_p.is_inside);
            let d = (proj - p_p.point).norm();
            assert_delta!(d, 0.0, 1e-10);
        }
    }

    #[test]
    fn test_gtetrahedron() {
        test_gsimplex::<_, crate::mesh::GTetrahedron<3>, _, _>(|p| {
            parry3d_f64::shape::Tetrahedron::new(p[0], p[1], p[2], p[3])
        });
    }

    #[test]
    fn test_gtriangle() {
        test_gsimplex::<_, crate::mesh::GTriangle<3>, _, _>(|p| {
            parry3d_f64::shape::Triangle::new(p[0], p[1], p[2])
        });
    }
}
