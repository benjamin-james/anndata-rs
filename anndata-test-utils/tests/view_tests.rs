//! Integration tests for the read-only view types.

use anndata_hdf5::H5;
use anndata_test_utils as utils;
use anndata_zarr::Zarr;

#[test]
fn test_view_shape() {
    utils::test_view_shape::<H5>();
    utils::test_view_shape::<Zarr>();
}

#[test]
fn test_view_obs_names() {
    utils::test_view_obs_names::<H5>();
    utils::test_view_obs_names::<Zarr>();
}

#[test]
fn test_view_var_names() {
    utils::test_view_var_names::<H5>();
    utils::test_view_var_names::<Zarr>();
}

#[test]
fn test_view_empty_selection() {
    utils::test_view_empty_selection::<H5>();
    utils::test_view_empty_selection::<Zarr>();
}

#[test]
fn test_view_full_selection_shape() {
    utils::test_view_full_selection_shape::<H5>();
    utils::test_view_full_selection_shape::<Zarr>();
}
