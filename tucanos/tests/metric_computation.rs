use std::{collections::HashMap, fs};
use tmesh::{
    Result,
    mesh::{BoundaryMesh3d, Mesh, Mesh3d},
};
use tucanos::{
    geometry::{VertexNormalWeight, compute_vertex_normals},
    metric::{AnisoMetric, AnisoMetric3d, Metric, MetricField},
};

fn parse_names(json_path: &str) -> Result<HashMap<u64, String>> {
    let json_text = fs::read_to_string(json_path)?;
    let json_value: serde_json::Value = serde_json::from_str(&json_text)?;

    let names_obj = json_value
        .get("names")
        .and_then(serde_json::Value::as_object)
        .ok_or("Missing or invalid 'names' object in JSON")?;

    let mut sections = HashMap::with_capacity(names_obj.len());
    for (name, id_value) in names_obj {
        let id = id_value
            .as_u64()
            .ok_or("Section id in 'names' must be an unsigned integer")?;
        sections.insert(id, name.clone());
    }

    Ok(sections)
}

fn main() -> Result<()> {
    let aniso_max = 10.0;

    let mesh_path = "../metric/adapt_in.meshb";
    let metric_path = "../metric/adapt_in_m.solb";
    let json_path = "../metric/adapt_in.json";

    let names = parse_names(json_path)?;
    println!("Parsed {} names from {json_path}", names.len());

    let mesh = Mesh3d::from_meshb(mesh_path)?;
    let (metric, n_comp) = Mesh3d::read_solb(metric_path)?;
    assert_eq!(n_comp, 6, "Expected 6 components in the metric");
    let mut metric = metric
        .chunks(6)
        .map(AnisoMetric3d::from_slice)
        .collect::<Vec<_>>();

    println!("Loaded mesh: {mesh_path}");
    println!("Loaded metric: {metric_path}");
    println!(
        "Mesh vertices: {}, elements: {}",
        mesh.n_verts(),
        mesh.n_elems()
    );
    println!("Metric values: {}, components: {}", metric.len(), n_comp);

    let (bdy, bdy_vert_ids) = mesh.boundary::<BoundaryMesh3d>();

    let mut flg = vec![false; bdy.n_verts()];
    for (tri, tag) in bdy.elems().zip(bdy.etags()) {
        let name = names.get(&(tag as u64)).unwrap();
        if name != "Farfield" && name != "Symmetry" {
            for v in tri {
                flg[v] = true;
            }
        }
    }

    let n = flg.iter().filter(|&&f| f).count();
    println!("Number of vertices on the boundary (excluding Farfield and Symmetry): {n}");

    let normals = compute_vertex_normals(&bdy, VertexNormalWeight::Volume);
    for (i, flg) in flg.iter().enumerate() {
        if *flg {
            let normal = normals[i];
            let m = &mut metric[bdy_vert_ids[i]];
            let mat = m.as_mat();
            let mut eig = mat.symmetric_eigen();
            // println!("Normal = {normal:?}");
            // println!("eigvals = {:?}", eig.eigenvalues);
            // println!("eigvecs = \n{}", eig.eigenvectors);

            // let e2 = eig.eigenvectors.column(2);
            // let dot = normal.dot(&e2);
            // println!("n.e2 = {dot}");

            let mut e0 = eig.eigenvectors.column(0).clone_owned();
            e0 -= normal * normal.dot(&e0);
            let e1 = normal.cross(&e0);

            eig.eigenvectors.set_column(0, &e0);
            eig.eigenvectors.set_column(1, &e1);
            eig.eigenvectors.set_column(2, &normal);

            eig.eigenvalues[0] =
                eig.eigenvalues[0].max(eig.eigenvalues[1] / (aniso_max * aniso_max));
            let mat = eig.recompose();
            *m = AnisoMetric3d::from_mat(mat);
        }
    }

    let mut field = MetricField::new(&mesh, metric);
    let v2v = mesh.vertex_to_vertices();

    let beta = 1.5;
    let t = 1.0;
    let max_iter = 25;
    field.apply_metric_gradation(&v2v, beta, t, max_iter)?;

    let m = field
        .metric()
        .iter()
        .flat_map(|x| x.into_iter())
        .collect::<Vec<_>>();

    mesh.write_solb(
        &m,
        "../metric/test.solb",
        tmesh::mesh::SolutionLocation::Vertices,
    )?;
    Ok(())
}
