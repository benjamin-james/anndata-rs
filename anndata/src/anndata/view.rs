//! Read-only views over [`AnnData`](crate::AnnData) and
//! [`AnnDataSet`](crate::AnnDataSet) objects.
//!
//! A *view* is a lightweight, lazy selection of an existing annotated-data
//! object. It does not copy the underlying data; instead it stores
//! [`SelectInfoElem`] selections for the observation (`obs`) and variable
//! (`var`) axes and applies them on read.
//!

use crate::Backend;
use crate::data::{DataFrameIndex, SelectInfoElem, SelectInfoElemBounds};
use crate::traits::AnnDataOp;

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
    // Data elements (x, obs, var, obsm, ...) are added in later commits as
    // the corresponding `read_*` methods are introduced.
    // The `B` parameter is kept so the public type `AnnDataView<B>` is stable.
    _phantom: std::marker::PhantomData<B>,
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
            obs_sel,
            var_sel,
            _phantom: std::marker::PhantomData,
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
