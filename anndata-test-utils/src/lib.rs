mod common;
pub use common::*;

use anndata::concat::{JoinType, concat};
use anndata::{data::CsrNonCanonical, *};
use data::{ArrayConvert, SelectInfoElem};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::Array2;
use proptest::prelude::*;

pub fn test_basic<B: Backend>() {
    with_tmp_dir(|dir| {
        let ann1 = AnnData::<B>::new(dir.join("test1")).unwrap();
        let csc = rand_csc::<i32>(10, 5, 3, 1, 100);
        ann1.obsm().add("csc", &csc).unwrap();
        assert!(ann1.obsm().get_item::<CsrMatrix<i32>>("csc").is_err());

        let ann2 = AnnData::<B>::new(dir.join("test2")).unwrap();
        AnnDataSet::<B>::new(
            [("ann1", ann1), ("ann2", ann2)],
            dir.join("dataset"),
            "sample",
            false,
        )
        .unwrap();
    })
}

pub fn test_save<B: Backend>() {
    with_tmp_dir(|dir| {
        let input = dir.join("input");
        let output = dir.join("output");
        let anndatas = ((0_usize..100), (0_usize..100)).prop_flat_map(|(n_obs, n_vars)| {
            (
                anndata_strat::<B, _>(&input, n_obs, n_vars),
                select_strat(n_obs),
                select_strat(n_vars),
            )
        });
        proptest!(ProptestConfig::with_cases(100), |((adata, slice_obs, slice_var) in anndatas)| {
            adata.write::<B, _>(&output, None, None).unwrap();
            let adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();

            let index = adata.obs_names().select(&slice_obs);
            assert_eq!(index.len(), index.into_vec().len());

            let select = [slice_obs, slice_var];
            adata.write_select::<B, _, _>(&select, &output).unwrap();
            adata.subset(&select).unwrap();
            let adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();
        });
    });
}

pub fn test_speacial_cases<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let adata = adata_gen();

    let arr = Array2::<i32>::zeros((0, 0));
    adata.set_x(&arr).unwrap();

    // Adding matrices with wrong shapes should fail
    let arr2 = Array2::<i32>::zeros((10, 20));
    assert!(adata.obsm().add("test", &arr2).is_err());

    // Data type casting
    let _: Array2<f64> = adata
        .x()
        .get::<ArrayData>()
        .unwrap()
        .unwrap()
        .try_convert()
        .expect("data type casting failed");
}

pub fn test_noncanonical<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let adata = adata_gen();
    let coo: CooMatrix<i32> = CooMatrix::try_from_triplets(
        5,
        4,
        vec![0, 1, 1, 1, 2, 3, 4],
        vec![0, 0, 0, 2, 3, 1, 3],
        vec![1, 2, 3, 4, 5, 6, 7],
    )
    .unwrap();
    adata.set_x(CsrNonCanonical::from(&coo)).unwrap();
    assert!(adata.x().get::<CsrMatrix<i32>>().is_err());
    adata.x().get::<CsrNonCanonical<i32>>().unwrap().unwrap();
    adata.x().get::<ArrayData>().unwrap().unwrap();
}

pub fn test_io<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays =
        proptest::collection::vec(0_usize..50, 2..4).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(256), |(x in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(adata.x().get::<ArrayData>().unwrap().unwrap(), x);
    });
}

pub fn test_index<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(0_usize..50, 2..4)
        .prop_flat_map(|shape| array_slice_strat(&shape));
    proptest!(ProptestConfig::with_cases(256), |((x, select) in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(
            adata.x().slice::<ArrayData, _>(&select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );

        adata.obsm().add("test", &x).unwrap();
        prop_assert_eq!(
            adata.obsm().get_item_slice::<ArrayData, _>("test", &select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );
    });
}

pub fn test_iterator<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays =
        proptest::collection::vec(20_usize..50, 2..3).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(10), |(x in arrays)| {
        if let ArrayData::CscMatrix(_) = x {
        } else {
            let adata = adata_gen();
            adata.obsm().add_iter("test", array_chunks(&x, 7)).unwrap();
            prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test").unwrap().unwrap(), x.clone());

            adata.obsm().add_iter("test2", adata.obsm().get_item_iter::<ArrayData>("test", 7).unwrap().map(|x| x.0)).unwrap();
            prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test2").unwrap().unwrap(), x);
        }
    });
}

pub fn test_concat<B: Backend>() {
    with_tmp_dir(|dir| {
        let input1 = dir.join("input1");
        let input2 = dir.join("input2");
        let output = dir.join("output");
        let anndatas = (
            (0_usize..100),
            (0_usize..100),
            (0_usize..100),
            (0_usize..100),
        )
            .prop_flat_map(|(n_obs1, n_vars1, n_obs2, n_vars2)| {
                (
                    anndata_strat::<B, _>(&input1, n_obs1, n_vars1),
                    anndata_strat::<B, _>(&input2, n_obs2, n_vars2),
                )
            });

        proptest!(ProptestConfig::with_cases(100), |((adata1, adata2) in anndatas)| {
            let adatas = [adata1, adata2];

            let out = AnnData::<B>::new(&output).unwrap();
            concat::<_, _, String>(&adatas, JoinType::Inner, None, None, &out).unwrap();

            let out = AnnData::<B>::new(&output).unwrap();
            concat::<_, _, String>(&adatas, JoinType::Outer, None, None, &out).unwrap();
        })
    });
}

//-----------------------------------------------------------------------------
// View tests
//-----------------------------------------------------------------------------

pub fn test_view_shape<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = create_test_adata::<B>(
            &dir,
            "test_shape",
            Array2::from_shape_vec((3, 3), vec![1, 2, 0, 4, 5, 0, 7, 8, 0]).unwrap(),
        );

        let view = adata.view((..2).into(), (..2).into());
        assert_eq!(view.shape(), (2, 2));
        assert_eq!(view.n_obs(), 2);
        assert_eq!(view.n_vars(), 2);
        assert_eq!(view.parent_n_obs(), 3);
        assert_eq!(view.parent_n_vars(), 3);
    })
}

pub fn test_view_obs_names<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = create_test_adata::<B>(
            &dir,
            "test_names",
            Array2::from_shape_vec((3, 3), vec![1, 2, 0, 4, 5, 0, 7, 8, 0]).unwrap(),
        );

        let view = adata.view((1..3).into(), SelectInfoElem::full());
        let names: Vec<String> = view.obs_names().into_iter().collect();
        assert_eq!(names, vec!["test_names_cell_1", "test_names_cell_2"]);
    })
}

pub fn test_view_var_names<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = create_test_adata::<B>(
            &dir,
            "test_vnames",
            Array2::from_shape_vec((3, 4), vec![1, 2, 0, 0, 4, 5, 0, 0, 7, 8, 0, 0]).unwrap(),
        );

        let view = adata.view(SelectInfoElem::full(), (1..3).into());
        let names: Vec<String> = view.var_names().into_iter().collect();
        assert_eq!(names, vec!["gene_1", "gene_2"]);
    })
}

pub fn test_view_empty_selection<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = create_test_adata::<B>(
            &dir,
            "test_empty",
            Array2::from_shape_vec((3, 3), vec![1, 2, 0, 4, 5, 0, 7, 8, 0]).unwrap(),
        );

        let view = adata.view(
            SelectInfoElem::from(Vec::<usize>::new()),
            SelectInfoElem::full(),
        );
        assert_eq!(view.n_obs(), 0);
        assert_eq!(view.n_vars(), 3);
        assert_eq!(view.parent_n_obs(), 3);
    })
}

pub fn test_view_full_selection_shape<B: Backend>() {
    with_tmp_dir(|dir| {
        let adata = create_test_adata::<B>(
            &dir,
            "test_full",
            Array2::from_shape_vec((3, 3), vec![1, 2, 0, 4, 5, 0, 7, 8, 0]).unwrap(),
        );

        let view = adata.view(SelectInfoElem::full(), SelectInfoElem::full());
        assert_eq!(view.shape(), (3, 3));
        assert_eq!(view.parent_n_obs(), 3);
        assert_eq!(view.parent_n_vars(), 3);
    })
}
