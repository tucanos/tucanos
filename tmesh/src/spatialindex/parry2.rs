use std::marker::PhantomData;

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

impl<const D: usize, C: GSimplex<D> + 'static> SimplexShape<D, C> {
    pub const fn new(c: C) -> Self {
        Self(c)
    }
}

impl<const D: usize, C: GSimplex<D> + 'static> PointQuery for SimplexShape<D, C> {
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

impl<const D: usize, C: GSimplex<D> + 'static> PointQueryWithLocation for SimplexShape<D, C> {
    type Location = C::BCOORDS;

    fn project_local_point_and_get_location(
        &self,
        pt: &Point<f64>,
        _solid: bool,
    ) -> (PointProjection, Self::Location) {
        let pt = Vertex::from_fn(|i, _| pt[i]);
        let bcoords = self.0.bcoords(&pt);
        let proj = self.0.vert(&bcoords);
        (
            PointProjection {
                is_inside: bcoords.into_iter().all(|x| x > -1e-12),
                point: Point::from_slice(proj.as_slice()),
            },
            bcoords,
        )
    }
}

impl<const D: usize, C: GSimplex<D> + 'static> RayCast for SimplexShape<D, C> {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &parry3d_f64::query::Ray,
        _max_time_of_impact: f64,
        _solid: bool,
    ) -> Option<parry3d_f64::query::RayIntersection> {
        todo!()
    }
}

impl<const D: usize, C: GSimplex<D> + 'static> Shape for SimplexShape<D, C> {
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

pub struct ObjectIndex2<'a, const D: usize, C: Simplex, M: Mesh<D, C>> {
    mesh: &'a M,
    tree: Bvh,
    _c: PhantomData<C>,
}

impl<'a, const D: usize, C: Simplex + 'static, M: Mesh<D, C>> ObjectIndex2<'a, D, C, M> {
    pub fn new(mesh: &'a M) -> Self {
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

        Self {
            mesh,
            tree,
            _c: PhantomData,
        }
    }

    /// Get the index of the nearest element
    #[must_use]
    pub fn nearest_elem(&self, pt: &Vertex<D>) -> usize {
        let (_, (id, _)) =
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

impl<const D: usize, C: Simplex + 'static, M: Mesh<D, C>> PointQueryWithLocation
    for ObjectIndex2<'_, D, C, M>
{
    type Location = (u32, <C::GEOM<D> as GSimplex<D>>::BCOORDS);

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

impl<const D: usize, C: Simplex + 'static, M: Mesh<D, C>> CompositeShape
    for ObjectIndex2<'_, D, C, M>
{
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

impl<const D: usize, C: Simplex + 'static, M: Mesh<D, C>> TypedCompositeShape
    for ObjectIndex2<'_, D, C, M>
{
    type PartShape = SimplexShape<D, C::GEOM<D>>;

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
mod tests {
    use crate::{
        Vert3d, Vertex,
        mesh::{GSimplex, Mesh, Mesh3d, Simplex, box_mesh},
        spatialindex::parry2::ObjectIndex2,
    };

    fn nearest_elem_naive<const D: usize, C: Simplex>(
        msh: &impl Mesh<D, C>,
        pt: &Vertex<D>,
    ) -> usize {
        let mut dst = f64::MAX;
        let mut res = 0;
        for (i, ge) in msh.gelems().enumerate() {
            let p = ge.project(pt);
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
        let n = 10;
        let mesh = box_mesh::<Mesh3d>(1.0, n, 1.0, n, 1.0, n).random_shuffle();

        let tree = ObjectIndex2::new(&mesh);

        let pt = Vert3d::new(0.123, 0.234, 0.345);
        let i = tree.nearest_elem(&pt);

        println!("i = {i}");
        println!("{:?}", mesh.gelem(&mesh.elem(i)).bcoords(&pt));

        let i2 = nearest_elem_naive(&mesh, &pt);
        println!("i = {i2}");
        println!("{:?}", mesh.gelem(&mesh.elem(i2)).bcoords(&pt));
    }
}
