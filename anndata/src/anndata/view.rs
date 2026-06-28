//! Read-only views over [`AnnData`](crate::AnnData) and
//! [`AnnDataSet`](crate::AnnDataSet) objects.
//!
//! A *view* is a lightweight, lazy selection of an existing annotated-data
//! object. It does not copy the underlying data; instead it stores
//! [`SelectInfoElem`] selections for the observation (`obs`) and variable
//! (`var`) axes and applies them on read.
//!

use crate::data::{ArrayData, DataFrameIndex, SelectInfoElem, SelectInfoElemBounds};
use crate::traits::{AnnDataOp, ArrayElemOp};
use crate::{ArrayElem, Backend, DataFrameElem};
use anyhow::Result;
use polars::prelude::DataFrame;

/// Return the number of elements selected by `sel` within an axis of length
/// `parent_len`.
pub(crate) fn sel_len(sel: &SelectInfoElem, parent_len: usize) -> usize {
    SelectInfoElemBounds::new(sel, parent_len).len()
}

/// A read-only view of an [`AnnData`] object.
///
/// The view holds cheaply-cloned references to the parent's data elements
/// together with `obs` and `var` selections. Reads apply the selections
/// lazily; no data is copied until a `read_*` method is called.
///
/// [`AnnData`]: crate::AnnData
pub struct AnnDataView<B: Backend> {
    /// Original parent `n_obs` (the bound the selections are relative to).
    parent_n_obs: usize,
    /// Original parent `n_vars`.
    parent_n_vars: usize,
    /// Selection on the observation axis, relative to the parent.
    obs_sel: SelectInfoElem,
    /// Selection on the variable axis, relative to the parent.
    var_sel: SelectInfoElem,
    obs_names_idx: DataFrameIndex,
    var_names_idx: DataFrameIndex,
    x: ArrayElem<B>,
    obs: DataFrameElem<B>,
    var: DataFrameElem<B>,
    // Further data elements (obsm, obsp, varm, ...) are added in later
    // commits as the corresponding `read_*` methods are introduced.
}

impl<B: Backend> std::fmt::Display for AnnDataView<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AnnDataView object with n_obs x n_vars = {} x {}",
            self.n_obs(),
            self.n_vars()
        )
    }
}

impl<B: Backend> AnnDataView<B> {
    /// Construct a view of `adata` with the given `obs`/`var` selections.
    ///
    /// The selections are interpreted relative to the parent's dimensions.
    /// This is called by [`AnnData::view`](crate::AnnData::view).
    pub(crate) fn new(
        adata: &crate::AnnData<B>,
        obs_sel: SelectInfoElem,
        var_sel: SelectInfoElem,
    ) -> Self {
        Self {
            parent_n_obs: adata.n_obs(),
            parent_n_vars: adata.n_vars(),
            obs_names_idx: adata.obs_names(),
            var_names_idx: adata.var_names(),
            x: adata.x.clone(),
            obs: adata.obs.clone(),
            var: adata.var.clone(),
            obs_sel,
            var_sel,
        }
    }

    /// The original parent `n_obs`.
    pub fn parent_n_obs(&self) -> usize {
        self.parent_n_obs
    }

    /// The original parent `n_vars`.
    pub fn parent_n_vars(&self) -> usize {
        self.parent_n_vars
    }

    /// The number of observations in the view (length of `obs_sel`).
    pub fn n_obs(&self) -> usize {
        sel_len(&self.obs_sel, self.parent_n_obs)
    }

    /// The number of variables in the view (length of `var_sel`).
    pub fn n_vars(&self) -> usize {
        sel_len(&self.var_sel, self.parent_n_vars)
    }

    /// The view's shape as `(n_obs, n_vars)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Borrow the observation selection, relative to the parent.
    pub fn obs_sel(&self) -> &SelectInfoElem {
        &self.obs_sel
    }

    /// Borrow the variable selection, relative to the parent.
    pub fn var_sel(&self) -> &SelectInfoElem {
        &self.var_sel
    }

    /// The selected observation names.
    pub fn obs_names(&self) -> DataFrameIndex {
        self.obs_names_idx.select(&self.obs_sel)
    }

    /// The selected variable names.
    pub fn var_names(&self) -> DataFrameIndex {
        self.var_names_idx.select(&self.var_sel)
    }

    /// Read the selected `X` matrix.
    ///
    /// Returns `None` if the parent has no `X`. The selection is applied with
    /// full two-dimensional dimensionality (`[obs_sel, var_sel]`).
    pub fn read_x(&self) -> Result<Option<ArrayData>> {
        self.x
            .slice::<ArrayData, _>(&[self.obs_sel.clone(), self.var_sel.clone()])
    }

    /// Read the selected `obs` DataFrame (rows sliced by `obs_sel`).
    ///
    /// Returns an empty DataFrame if the parent has no `obs`.
    pub fn read_obs(&self) -> Result<DataFrame> {
        if self.obs.is_none() {
            return Ok(DataFrame::empty());
        }
        self.obs.inner().select_axis(0, &self.obs_sel)
    }

    /// Read the selected `var` DataFrame (rows sliced by `var_sel`).
    ///
    /// Returns an empty DataFrame if the parent has no `var`.
    pub fn read_var(&self) -> Result<DataFrame> {
        if self.var.is_none() {
            return Ok(DataFrame::empty());
        }
        self.var.inner().select_axis(0, &self.var_sel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn sel_len_basic() {
        assert_eq!(sel_len(&SelectInfoElem::full(), 100), 100);
        let slice: SelectInfoElem = (2..8).into();
        assert_eq!(sel_len(&slice, 100), 6);
        let index = SelectInfoElem::from(vec![0usize, 3, 7]);
        assert_eq!(sel_len(&index, 100), 3);
    }

    proptest! {
        /// `sel_len` of a range slice `a..b` is `b - a`.
        #[test]
        fn sel_len_range_slice(start in 0u64..500, end in 0u64..500) {
            let (s, e) = (start.min(end) as usize, start.max(end) as usize);
            let sel: SelectInfoElem = (s..e).into();
            prop_assert_eq!(sel_len(&sel, 1000), e - s);
        }

        /// `sel_len` of an index selection is the number of indices.
        #[test]
        fn sel_len_index_is_count(indices in prop::collection::vec(0usize..1000, 0..100)) {
            let sel = SelectInfoElem::from(indices.clone());
            prop_assert_eq!(sel_len(&sel, 1000), indices.len());
        }
    }
}
