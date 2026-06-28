//! Integration tests for the read-only view types.
//!
//! Each test runs against both the HDF5 and Zarr backends by delegating to
//! the generic functions in `anndata_test_utils`.

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

#[test]
fn test_view_read_x() {
    utils::test_view_read_x::<H5>();
    utils::test_view_read_x::<Zarr>();
}

#[test]
fn test_view_read_obs() {
    utils::test_view_read_obs::<H5>();
    utils::test_view_read_obs::<Zarr>();
}

#[test]
fn test_view_read_var() {
    utils::test_view_read_var::<H5>();
    utils::test_view_read_var::<Zarr>();
}

#[test]
fn test_view_index_selection() {
    utils::test_view_index_selection::<H5>();
    utils::test_view_index_selection::<Zarr>();
}

#[test]
fn test_view_duplicate_index() {
    utils::test_view_duplicate_index::<H5>();
    utils::test_view_duplicate_index::<Zarr>();
}
