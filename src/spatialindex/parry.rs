use parry2d_f64::query::{PointQuery as _, PointQueryWithLocation as _};
use parry3d_f64::query::{PointQuery as _, PointQueryWithLocation as _};

use crate::{
    mesh::{Elem, Point, QuadraticElem, QuadraticMesh, SimplexMesh},
    Idx,
};

mod parry2d {
    use crate::{
        mesh::{Elem, SimplexMesh},
        Idx,
    };
    use nalgebra::Point2;
    use parry2d_f64::{
        bounding_volume::Aabb,
        math::{Isometry, Point, Real},
        partitioning::Qbvh,
        query::{
            details::NormalConstraints, point::PointCompositeShapeProjWithLocationBestFirstVisitor,
            PointProjection, PointQueryWithLocation,
        },
        shape::{Segment, SegmentPointLocation, Shape, TypedSimdCompositeShape},
    };
    use std::marker::PhantomData;

    /// Create a parry Shape from a tucanos Elem
    pub trait MeshToShape {
        type Shape: Shape + PointQueryWithLocation<Location = Self::Location> + Copy + Clone;
        type Location: Clone + Copy;
        fn shape(id: u32, verts: &[f64], elems: &[Idx]) -> Self::Shape;
    }

    fn points<const COUNT: usize>(id: u32, verts: &[f64], elems: &[Idx]) -> [Point2<Real>; COUNT] {
        std::array::from_fn(|i| {
            let vid = elems[COUNT * id as usize + i] as usize;
            Point2::new(verts[vid * 2], verts[vid * 2 + 1])
        })
    }

    pub struct EdgeToSegment {}
    impl MeshToShape for EdgeToSegment {
        type Shape = Segment;
        type Location = SegmentPointLocation;
        fn shape(id: u32, verts: &[f64], elems: &[Idx]) -> Self::Shape {
            let [v1, v2] = points(id, verts, elems);
            Self::Shape::new(v1, v2)
        }
    }

    pub struct MeshShape<const D: usize, MS: MeshToShape> {
        // TODO: this is copied from SimplexMesh. It would nice to just reference it. Sadly Rust forbid
        // self referencing objects and spatialindex::ObjectIndex is a member of SimplexMesh. An
        // alternative would be to pass verts and elem as arguments of ObjectIndex::nearest but
        // TypedSimdCompositeShape does not accept additionnals arguments.
        verts: Vec<f64>,
        elems: Vec<Idx>,
        tree: Qbvh<u32>,
        phantom: PhantomData<MS>,
    }

    impl<const D: usize, MS: MeshToShape> MeshShape<D, MS> {
        fn local_aabb<E: Elem>(mesh: &SimplexMesh<D, E>, elem: &E) -> Aabb {
            let mut min = Point::origin();
            let mut max = min;

            for d in 0..D {
                let (mn, mx) = elem
                    .iter()
                    .map(|&idx| mesh.vert(idx)[d])
                    .fold((f64::MAX, -f64::MAX), |(mn, mx), x| (mn.min(x), mx.max(x)));
                min.coords[d] = mn;
                max.coords[d] = mx;
            }

            parry2d_f64::bounding_volume::Aabb::new(min, max)
        }

        pub fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
            let mut tree = Qbvh::new();
            let data = mesh
                .elems()
                .enumerate()
                .map(|(i, e)| (i as u32, Self::local_aabb(mesh, &e)));
            tree.clear_and_rebuild(data, 0.);
            let mut elems = Vec::with_capacity((mesh.n_elems() as usize) * (E::N_VERTS as usize));
            for e in mesh.elems() {
                elems.extend(e.iter().copied());
            }
            let mut verts = Vec::with_capacity(2 * mesh.n_verts() as usize);
            for e in mesh.verts() {
                verts.extend(e.iter().copied());
            }
            Self {
                verts,
                elems,
                tree,
                phantom: PhantomData,
            }
        }
    }

    impl<const D: usize, MS: MeshToShape> PointQueryWithLocation for MeshShape<D, MS> {
        type Location = (u32, MS::Location);

        fn project_local_point_and_get_location(
            &self,
            point: &Point<Real>,
            solid: bool,
        ) -> (PointProjection, Self::Location) {
            let mut visitor =
                PointCompositeShapeProjWithLocationBestFirstVisitor::new(self, point, solid);
            self.tree.traverse_best_first(&mut visitor).unwrap().1
        }
    }

    impl<const D: usize, MS: MeshToShape> TypedSimdCompositeShape for MeshShape<D, MS> {
        type PartShape = MS::Shape;
        type PartId = u32;
        type PartNormalConstraints = dyn NormalConstraints;

        fn map_typed_part_at(
            &self,
            shape_id: Self::PartId,
            mut f: impl FnMut(
                Option<&Isometry<Real>>,
                &Self::PartShape,
                Option<&Self::PartNormalConstraints>,
            ),
        ) {
            f(None, &MS::shape(shape_id, &self.verts, &self.elems), None);
        }

        fn map_untyped_part_at(
            &self,
            shape_id: Self::PartId,
            mut f: impl FnMut(Option<&Isometry<Real>>, &dyn Shape, Option<&dyn NormalConstraints>),
        ) {
            f(None, &MS::shape(shape_id, &self.verts, &self.elems), None);
        }

        fn typed_qbvh(&self) -> &Qbvh<Self::PartId> {
            &self.tree
        }
    }
}

pub mod parry3d {
    use crate::{
        mesh::{
            Elem, GQuadraticElem, GQuadraticTriangle, QuadraticElem, QuadraticMesh, SimplexMesh,
        },
        metric::{AnisoMetric, AnisoMetric3d},
        Idx,
    };
    use nalgebra::{Matrix3, Point3};

    use parry3d_f64::{
        bounding_volume::{Aabb, BoundingSphere},
        mass_properties::MassProperties,
        math::{Isometry, Point, Real},
        partitioning::Qbvh,
        query::{
            details::NormalConstraints, point::PointCompositeShapeProjWithLocationBestFirstVisitor,
            PointProjection, PointQuery, PointQueryWithLocation, Ray, RayCast, RayIntersection,
        },
        shape::{
            FeatureId, Segment, SegmentPointLocation, Shape, ShapeType, Tetrahedron,
            TetrahedronPointLocation, TypedShape, TypedSimdCompositeShape,
        },
    };
    use std::marker::PhantomData;
    /// Create a parry Shape from a tucanos Elem
    pub trait MeshToShape {
        type Shape: Shape + PointQueryWithLocation<Location = Self::Location> + Copy + Clone;
        type Location: Clone + Copy;
        fn shape(id: u32, verts: &[f64], elems: &[Idx]) -> Self::Shape;
    }

    /// Wrap the parry3d Tetrahedron to make it parry3d Shape
    #[derive(Copy, Clone)]
    pub struct TetraShape(Tetrahedron);

    impl TetraShape {
        const fn new(a: &[Point<Real>; 4]) -> Self {
            Self(Tetrahedron {
                a: a[0],
                b: a[1],
                c: a[2],
                d: a[3],
            })
        }
    }
    impl parry3d_f64::shape::Shape for TetraShape {
        fn compute_local_aabb(&self) -> Aabb {
            todo!()
        }

        fn compute_local_bounding_sphere(&self) -> BoundingSphere {
            todo!()
        }

        fn clone_box(&self) -> Box<dyn parry3d_f64::shape::Shape> {
            todo!()
        }

        fn mass_properties(&self, _density: Real) -> MassProperties {
            todo!()
        }

        fn shape_type(&self) -> ShapeType {
            todo!()
        }

        fn as_typed_shape(&self) -> TypedShape {
            todo!()
        }

        fn ccd_thickness(&self) -> Real {
            todo!()
        }

        fn ccd_angular_thickness(&self) -> Real {
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
    }

    impl PointQueryWithLocation for TetraShape {
        type Location = TetrahedronPointLocation;

        fn project_local_point_and_get_location(
            &self,
            pt: &Point<Real>,
            solid: bool,
        ) -> (PointProjection, Self::Location) {
            self.0.project_local_point_and_get_location(pt, solid)
        }
    }

    impl PointQuery for TetraShape {
        fn project_local_point(&self, pt: &Point<Real>, solid: bool) -> PointProjection {
            self.0.project_local_point(pt, solid)
        }

        fn project_local_point_and_get_feature(
            &self,
            pt: &Point<Real>,
        ) -> (PointProjection, FeatureId) {
            self.0.project_local_point_and_get_feature(pt)
        }
    }

    impl RayCast for TetraShape {
        fn cast_local_ray_and_get_normal(
            &self,
            _ray: &Ray,
            _max_toi: Real,
            _solid: bool,
        ) -> Option<RayIntersection> {
            todo!()
        }
    }

    pub struct TetraToTetra {}
    impl MeshToShape for TetraToTetra {
        type Shape = TetraShape;
        type Location = TetrahedronPointLocation;
        fn shape(id: u32, verts: &[f64], elems: &[Idx]) -> Self::Shape {
            Self::Shape::new(&points(id, verts, elems))
        }
    }
    fn points<const COUNT: usize>(id: u32, verts: &[f64], elems: &[Idx]) -> [Point3<Real>; COUNT] {
        std::array::from_fn(|i| {
            let vid = elems[COUNT * id as usize + i] as usize;
            Point3::new(verts[vid * 3], verts[vid * 3 + 1], verts[vid * 3 + 2])
        })
    }
    pub struct EdgeToSegment {}
    impl MeshToShape for EdgeToSegment {
        type Shape = Segment;
        type Location = SegmentPointLocation;
        fn shape(id: u32, verts: &[f64], elems: &[Idx]) -> Self::Shape {
            let [v1, v2] = points(id, verts, elems);
            Self::Shape::new(v1, v2)
        }
    }

    pub struct MeshShape<const D: usize, MS: MeshToShape> {
        // TODO: this is copied from SimplexMesh. It would nice to just reference it. Sadly Rust forbid
        // self referencing objects and spatialindex::ObjectIndex is a member of SimplexMesh. An
        // alternative would be to pass verts and elem as arguments of ObjectIndex::nearest but
        // TypedSimdCompositeShape does not accept additionnals arguments.
        verts: Vec<f64>,
        elems: Vec<Idx>,
        tree: Qbvh<u32>,
        phantom: PhantomData<MS>,
    }
    impl<const D: usize, MS: MeshToShape> MeshShape<D, MS> {
        fn local_aabb<E: Elem>(mesh: &SimplexMesh<D, E>, elem: &E) -> Aabb {
            let mut min = Point::origin();
            let mut max = min;

            for d in 0..D {
                let (mn, mx) = elem
                    .iter()
                    .map(|&idx| mesh.vert(idx)[d])
                    .fold((f64::MAX, -f64::MAX), |(mn, mx), x| (mn.min(x), mx.max(x)));
                min.coords[d] = mn;
                max.coords[d] = mx;
            }

            Aabb::new(min, max)
        }
        pub fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
            let mut tree = Qbvh::new();
            let data = mesh
                .elems()
                .enumerate()
                .map(|(i, e)| (i as u32, Self::local_aabb(mesh, &e)));
            tree.clear_and_rebuild(data, 0.);
            let mut elems = Vec::with_capacity((mesh.n_elems() as usize) * (E::N_VERTS as usize));
            for e in mesh.elems() {
                elems.extend(e.iter().copied());
            }
            let mut verts = Vec::with_capacity(2 * mesh.n_verts() as usize);
            for e in mesh.verts() {
                verts.extend(e.iter().copied());
            }
            Self {
                verts,
                elems,
                tree,
                phantom: PhantomData,
            }
        }
    }

    impl<const D: usize, MS: MeshToShape> PointQueryWithLocation for MeshShape<D, MS> {
        type Location = (u32, MS::Location);

        fn project_local_point_and_get_location(
            &self,
            point: &Point<Real>,
            solid: bool,
        ) -> (PointProjection, Self::Location) {
            let mut visitor =
                PointCompositeShapeProjWithLocationBestFirstVisitor::new(self, point, solid);
            self.tree.traverse_best_first(&mut visitor).unwrap().1
        }
    }

    impl<const D: usize, MS: MeshToShape> TypedSimdCompositeShape for MeshShape<D, MS> {
        type PartShape = MS::Shape;
        type PartId = u32;
        type PartNormalConstraints = dyn NormalConstraints;

        fn map_typed_part_at(
            &self,
            shape_id: Self::PartId,
            mut f: impl FnMut(
                Option<&Isometry<Real>>,
                &Self::PartShape,
                Option<&Self::PartNormalConstraints>,
            ),
        ) {
            f(None, &MS::shape(shape_id, &self.verts, &self.elems), None);
        }

        fn map_untyped_part_at(
            &self,
            shape_id: Self::PartId,
            mut f: impl FnMut(Option<&Isometry<Real>>, &dyn Shape, Option<&dyn NormalConstraints>),
        ) {
            f(None, &MS::shape(shape_id, &self.verts, &self.elems), None);
        }

        fn typed_qbvh(&self) -> &Qbvh<Self::PartId> {
            &self.tree
        }
    }

    pub fn project<const D: usize, S: PointQueryWithLocation>(
        shape: &S,
        pt: &crate::mesh::Point<D>,
    ) -> (f64, crate::mesh::Point<D>) {
        let (p, _) =
            shape.project_local_point_and_get_location(&Point3::new(pt[0], pt[1], pt[2]), false);
        let p = p.point;
        let mut res = crate::mesh::Point::<D>::zeros();
        for i in 0..D {
            res[i] = p[i];
        }
        ((pt - res).norm(), res)
    }

    #[derive(Debug, Copy, Clone)]
    pub struct QuadraticTriangle {
        pub a: Point<Real>,
        pub b: Point<Real>,
        pub c: Point<Real>,
        pub d: Point<Real>,
        pub e: Point<Real>,
        pub f: Point<Real>,
    }

    impl QuadraticTriangle {
        #[inline]
        pub fn bounding_sphere(&self) -> parry3d_f64::bounding_volume::BoundingSphere {
            let a = self.a.coords;
            let b = self.b.coords;
            let c = self.c.coords;
            let d = self.d.coords;
            let e = self.e.coords;
            let f = self.f.coords;

            let p0 = Point3::from(a);
            let p1 = Point3::from(b);
            let p2 = Point3::from(c);

            // Bezier control points
            let p3 = Point3::from(2.0 * d - 0.5 * a - 0.5 * b);
            let p4 = Point3::from(2.0 * e - 0.5 * b - 0.5 * c);
            let p5 = Point3::from(2.0 * f - 0.5 * c - 0.5 * a);

            let pts = [p0, p1, p2, p3, p4, p5];
            parry3d_f64::bounding_volume::details::point_cloud_bounding_sphere(&pts[..])
        }
    }

    #[allow(clippy::enum_variant_names)]
    #[derive(Debug, Copy, Clone)]
    pub enum QuadraticTrianglePointLocation {
        OnVertex(u32),
        OnEdge(u32, [Real; 2]),
        OnFace(u32, [Real; 3]),
    }

    /// Wrap the Quadratic Triangle to make it parry 3d shape
    #[derive(Debug, Copy, Clone)]
    pub struct QuadraticTriangleShape(pub QuadraticTriangle);

    impl QuadraticTriangleShape {
        #[allow(dead_code)]
        pub const fn new(a: &[Point<Real>; 6]) -> Self {
            Self(QuadraticTriangle {
                a: a[0],
                b: a[1],
                c: a[2],
                d: a[3],
                e: a[4],
                f: a[5],
            })
        }
    }

    impl parry3d_f64::shape::Shape for QuadraticTriangleShape {
        fn compute_local_aabb(&self) -> Aabb {
            let mut min = Point::origin();
            let mut max = Point::origin();
            let a = self.0.a;
            let b = self.0.b;
            let c = self.0.c;
            let d = self.0.d;
            let e = self.0.e;
            let f = self.0.f;

            for i in 0..3 {
                min.coords[i] = a[i].min(b[i]).min(c[i]).min(d[i]).min(e[i]).min(f[i]);
                max.coords[i] = a[i].max(b[i]).max(c[i]).max(d[i]).max(e[i]).max(f[i]);
            }

            Aabb::new(min, max)
        }

        fn compute_local_bounding_sphere(&self) -> BoundingSphere {
            self.0.bounding_sphere()
        }

        fn clone_box(&self) -> Box<dyn parry3d_f64::shape::Shape> {
            todo!()
        }

        fn mass_properties(&self, _density: Real) -> MassProperties {
            todo!()
        }

        fn shape_type(&self) -> ShapeType {
            todo!()
        }

        fn as_typed_shape(&self) -> TypedShape {
            todo!()
        }

        fn ccd_thickness(&self) -> Real {
            todo!()
        }

        fn ccd_angular_thickness(&self) -> Real {
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
    }

    impl PointQueryWithLocation for QuadraticTriangleShape {
        type Location = QuadraticTrianglePointLocation;
        // Project a point on self (QuadraticTriangleShape)
        fn project_local_point_and_get_location(
            &self,
            pt: &Point<Real>,
            _solid: bool,
        ) -> (PointProjection, Self::Location) {
            let a = self.0.a.coords;
            let b = self.0.b.coords;
            let c = self.0.c.coords;
            let d = self.0.d.coords;
            let e = self.0.e.coords;
            let f = self.0.f.coords;

            let points = [a, b, c, d, e, f];
            let identity = Matrix3::identity();
            let metric = AnisoMetric3d::from_mat(identity);
            let points_and_metrics = points.iter().map(|&p| (p, metric));

            let pt = pt.coords;
            let triangle = GQuadraticTriangle::from_verts(points_and_metrics);
            let (proj_point_coords, location) = triangle.project(pt, &[0.3, 0.3], 100, 0.000001);
            // Rebuild a Point<Real> from the projected coords
            let proj_point = Point::from(proj_point_coords);

            // Determine whether the projected point is “inside” according to the location
            let is_inside = matches!(location, QuadraticTrianglePointLocation::OnFace(_, _));

            // Build and return the projection
            (
                PointProjection {
                    is_inside,
                    point: proj_point,
                },
                location,
            )
        }
    }

    impl PointQuery for QuadraticTriangleShape {
        fn project_local_point(&self, pt: &Point<Real>, solid: bool) -> PointProjection {
            let (projection, _) = self.project_local_point_and_get_location(pt, solid);
            projection
        }

        fn project_local_point_and_get_feature(
            &self,
            _pt: &Point<Real>,
        ) -> (PointProjection, FeatureId) {
            todo!()
            // self.0.project_local_point_and_get_feature(pt)
        }
    }
    impl RayCast for QuadraticTriangleShape {
        fn cast_local_ray_and_get_normal(
            &self,
            _ray: &Ray,
            _max_toi: Real,
            _solid: bool,
        ) -> Option<RayIntersection> {
            todo!()
        }
    }

    pub struct QuadraticTriangleToQuadraticTriangle {}
    impl MeshToShape for QuadraticTriangleToQuadraticTriangle {
        type Shape = QuadraticTriangleShape;
        type Location = QuadraticTrianglePointLocation;
        fn shape(id: u32, verts: &[f64], elems: &[Idx]) -> Self::Shape {
            Self::Shape::new(&points(id, verts, elems))
        }
    }

    pub struct QuadraticMeshShape<MS: MeshToShape> {
        // TODO: this is copied from SimplexMesh. It would nice to just reference it. Sadly Rust forbid
        // self referencing objects and spatialindex::ObjectIndex is a member of SimplexMesh. An
        // alternative would be to pass verts and elem as arguments of ObjectIndex::nearest but
        // TypedSimdCompositeShape does not accept additionnals arguments.
        verts: Vec<f64>,
        elems: Vec<Idx>,
        tree: Qbvh<u32>,
        phantom: PhantomData<MS>,
    }
    impl<MS: MeshToShape> QuadraticMeshShape<MS> {
        fn local_aabb<QE: QuadraticElem>(mesh: &QuadraticMesh<QE>, quadratic_elem: &QE) -> Aabb {
            let mut min = Point::origin();
            let mut max = min;

            for d in 0..3 {
                let (mn, mx) = quadratic_elem
                    .iter()
                    .map(|&idx| mesh.vert(idx)[d])
                    .fold((f64::MAX, -f64::MAX), |(mn, mx), x| (mn.min(x), mx.max(x)));
                min.coords[d] = mn;
                max.coords[d] = mx;
            }

            Aabb::new(min, max)
        }
        pub fn new<QE: QuadraticElem>(mesh: &QuadraticMesh<QE>) -> Self {
            let mut tree = Qbvh::new();
            let data = mesh
                .tris()
                .enumerate()
                .map(|(i, qe)| (i as u32, Self::local_aabb(mesh, &qe)));
            tree.clear_and_rebuild(data, 0.);
            let mut elems = Vec::with_capacity((mesh.n_tris() as usize) * (QE::N_VERTS as usize));
            for qe in mesh.tris() {
                elems.extend(qe.iter().copied());
            }
            let mut verts = Vec::with_capacity(2 * mesh.n_verts() as usize);
            for qe in mesh.verts() {
                verts.extend(qe.iter().copied());
            }
            Self {
                verts,
                elems,
                tree,
                phantom: PhantomData,
            }
        }
    }

    impl<MS: MeshToShape> PointQueryWithLocation for QuadraticMeshShape<MS> {
        type Location = (u32, MS::Location);

        fn project_local_point_and_get_location(
            &self,
            point: &Point<Real>,
            solid: bool,
        ) -> (PointProjection, Self::Location) {
            let mut visitor =
                PointCompositeShapeProjWithLocationBestFirstVisitor::new(self, point, solid);
            self.tree.traverse_best_first(&mut visitor).unwrap().1
        }
    }

    impl<MS: MeshToShape> TypedSimdCompositeShape for QuadraticMeshShape<MS> {
        type PartShape = MS::Shape;
        type PartId = u32;
        type PartNormalConstraints = dyn NormalConstraints;

        fn map_typed_part_at(
            &self,
            shape_id: Self::PartId,
            mut f: impl FnMut(
                Option<&Isometry<Real>>,
                &Self::PartShape,
                Option<&Self::PartNormalConstraints>,
            ),
        ) {
            f(None, &MS::shape(shape_id, &self.verts, &self.elems), None);
        }

        fn map_untyped_part_at(
            &self,
            shape_id: Self::PartId,
            mut f: impl FnMut(Option<&Isometry<Real>>, &dyn Shape, Option<&dyn NormalConstraints>),
        ) {
            f(None, &MS::shape(shape_id, &self.verts, &self.elems), None);
        }

        fn typed_qbvh(&self) -> &Qbvh<Self::PartId> {
            &self.tree
        }
    }
}

pub struct ObjectIndex<const D: usize> {
    inner: ParryImpl<D>,
}
impl<const D: usize> std::fmt::Debug for ObjectIndex<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // We have not that much to print but SimplexMesh requires Debug
        f.debug_struct("ObjectIndex").finish()
    }
}

#[allow(clippy::large_enum_variant)]
enum ParryImpl<const D: usize> {
    Tria3D(parry3d_f64::shape::TriMesh),
    Tria2D(parry2d_f64::shape::TriMesh),
    Edge2D(parry2d::MeshShape<D, parry2d::EdgeToSegment>),
    Edge3D(parry3d::MeshShape<D, parry3d::EdgeToSegment>),
    Tetra(parry3d::MeshShape<D, parry3d::TetraToTetra>),
    QuadraticTria3D(parry3d::QuadraticMeshShape<parry3d::QuadraticTriangleToQuadraticTriangle>),
}

impl<const D: usize> super::ObjectIndex<D> for ObjectIndex<D> {
    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
        if D == 3 && E::N_VERTS == 3 {
            let mut coords = Vec::with_capacity(3 * mesh.n_verts() as usize);
            for p in mesh.verts() {
                coords.push(nalgebra::Point3::new(p[0], p[1], p[2]));
            }
            let elems: Vec<_> = mesh
                .elems()
                .map(|e| {
                    let mut i = e.iter();
                    std::array::from_fn(|_| *i.next().unwrap())
                })
                .collect();
            Self {
                inner: ParryImpl::Tria3D(parry3d_f64::shape::TriMesh::new(coords, elems).unwrap()),
            }
        } else if D == 2 && E::N_VERTS == 2 {
            Self {
                inner: ParryImpl::Edge2D(parry2d::MeshShape::new(mesh)),
            }
        } else if D == 3 && E::N_VERTS == 4 {
            Self {
                inner: ParryImpl::Tetra(parry3d::MeshShape::new(mesh)),
            }
        } else if D == 3 && E::N_VERTS == 2 {
            Self {
                inner: ParryImpl::Edge3D(parry3d::MeshShape::new(mesh)),
            }
        } else if D == 2 && E::N_VERTS == 3 {
            let mut coords = Vec::with_capacity(2 * mesh.n_verts() as usize);
            for p in mesh.verts() {
                coords.push(nalgebra::Point2::new(p[0], p[1]));
            }
            let elems: Vec<_> = mesh
                .elems()
                .map(|e| {
                    let mut i = e.iter();
                    std::array::from_fn(|_| *i.next().unwrap())
                })
                .collect();
            Self {
                inner: ParryImpl::Tria2D(parry2d_f64::shape::TriMesh::new(coords, elems).unwrap()),
            }
        } else {
            unimplemented!("D={D} E::N_VERTS={}", E::N_VERTS);
        }
    }

    fn newquadratic<QE: QuadraticElem>(mesh: &QuadraticMesh<QE>) -> Self {
        if QE::N_VERTS == 6 {
            Self {
                inner: ParryImpl::QuadraticTria3D(parry3d::QuadraticMeshShape::new(mesh)),
            }
        } else {
            unimplemented!("QE::N_VERTS={}", QE::N_VERTS);
        }
    }

    fn nearest_elem(&self, pt: &Point<D>) -> Idx {
        match &self.inner {
            ParryImpl::Tria3D(shape) => {
                let (_, (id, _)) = shape.project_local_point_and_get_location(
                    &nalgebra::Point3::new(pt[0], pt[1], pt[2]),
                    true,
                );
                id
            }
            ParryImpl::Edge2D(shape) => {
                let (_, (id, _)) = shape.project_local_point_and_get_location(
                    &nalgebra::Point2::new(pt[0], pt[1]),
                    true,
                );
                id
            }
            ParryImpl::Tetra(shape) => {
                let (_, (id, _)) = shape.project_local_point_and_get_location(
                    &nalgebra::Point3::new(pt[0], pt[1], pt[2]),
                    true,
                );
                id
            }
            ParryImpl::Edge3D(_) => todo!(),
            ParryImpl::Tria2D(shape) => {
                let (_, (id, _)) = shape.project_local_point_and_get_location(
                    &nalgebra::Point2::new(pt[0], pt[1]),
                    true,
                );
                id
            }
            ParryImpl::QuadraticTria3D(shape) => {
                let (_, (id, _)) = shape.project_local_point_and_get_location(
                    &nalgebra::Point3::new(pt[0], pt[1], pt[2]),
                    true,
                );
                id
            }
        }
    }

    fn project(&self, pt: &Point<D>) -> (f64, Point<D>) {
        match &self.inner {
            ParryImpl::Tria3D(shape) => {
                // https://docs.rs/parry3d/latest/parry3d/query/point/trait.PointQuery.html#method.project_point
                let p =
                    shape.project_local_point(&nalgebra::Point3::new(pt[0], pt[1], pt[2]), true);
                let p = p.point;
                let mut res = Point::<D>::zeros();
                for i in 0..D {
                    res[i] = p[i];
                }
                ((pt - res).norm(), res)
            }
            ParryImpl::Edge2D(shape) => {
                let (p, _) = shape.project_local_point_and_get_location(
                    &nalgebra::Point2::new(pt[0], pt[1]),
                    false,
                );
                let p = p.point;
                let mut res = Point::<D>::zeros();
                for i in 0..D {
                    res[i] = p[i];
                }
                ((pt - res).norm(), res)
            }
            ParryImpl::Tetra(_) => todo!(),
            ParryImpl::Edge3D(shape) => parry3d::project(shape, pt),
            ParryImpl::Tria2D(shape) => {
                let p = shape.project_local_point(&nalgebra::Point2::new(pt[0], pt[1]), true);
                let p = p.point;
                let mut res = Point::<D>::zeros();
                for i in 0..D {
                    res[i] = p[i];
                }
                ((pt - res).norm(), res)
            }
            ParryImpl::QuadraticTria3D(shape) => {
                let (p, _) = shape.project_local_point_and_get_location(
                    &nalgebra::Point3::new(pt[0], pt[1], pt[2]),
                    true,
                );
                let p = p.point;
                let mut res = Point::<D>::zeros();
                for i in 0..D {
                    res[i] = p[i];
                }
                ((pt - res).norm(), res)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mesh::test_meshes::test_mesh_2d_quadratic,
        mesh::Point,
        spatialindex::{DefaultObjectIndex, ObjectIndex},
    };

    #[test]
    fn test_quadratic_triangle_bounding_sphere_and_projection() {
        let mesh = test_mesh_2d_quadratic();

        let tree: DefaultObjectIndex<3> = DefaultObjectIndex::newquadratic(&mesh);

        let query_points = [
            Point::<3>::new(-1.7, 0.5, 1.1),
            Point::<3>::new(20.0, 10., -1.),
        ];

        for (i, &pt) in query_points.iter().enumerate() {
            let id = tree.nearest_elem(&pt);
            println!("nearest elem for point {} = {}", i + 1, id);

            let (dist, proj) = tree.project(&pt);
            println!("distance = {dist}");
            println!("projected point = {proj}");
        }
    }
}
