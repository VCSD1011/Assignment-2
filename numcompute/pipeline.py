import numpy as np
if __name__ == "__main__":
    from utils import BaseTransformer
else:
    from numcompute.utils import BaseTransformer


# Pipeline

class Pipeline(BaseTransformer):
    """Chains a fixed sequence of transformers, with an optional final estimator.

    Each intermediate step must implement the BaseTransformer API (.fit and
    .transform).  The last step may be a plain estimator that only implements
    .fit and .predict (no .transform required).

    Parameters
    ----------
    steps : list of (str, transformer/estimator) tuples
        Ordered list of ``(name, object)`` pairs.  Names must be unique strings;
        they are used for error messages and future parameter access.

    Attributes
    ----------
    steps : list of (str, object)
        The validated pipeline steps.
    _is_fitted : bool
        Inherited safety flag; True after .fit() has been called.

    Raises
    ------
    TypeError
        If ``steps`` is not a list of 2-tuples.
    ValueError
        If step names are not unique strings, or if an intermediate step does
        not implement the required transformer interface.
    RuntimeError
        If .transform() or .predict() is called before .fit().

    Time Complexity
    ---------------
    fit          : O(sum of each step's fit cost)
    transform    : O(sum of each step's transform cost)
    fit_transform: O(fit + transform) — uses a single pass, not two.

    Examples
    --------
    >>> pipe = Pipeline([('scale', StandardScaler()), ('encode', OneHotEncoder())])
    >>> X_tr = pipe.fit_transform(X_train)
    >>> X_te = pipe.transform(X_test)
    """

    def __init__(self, steps):
        # Inherit _is_fitted safety flag from BaseEstimator via BaseTransformer
        super().__init__()
        self._validate_steps(steps)
        self.steps = list(steps)

    # Validation helpers

    @staticmethod
    def _validate_steps(steps):
        """Raises informative errors if the steps list is malformed.

        Parameters
        ----------
        steps : any
            The value passed to __init__; must be a list of (str, object) pairs.

        Raises
        ------
        TypeError
            If ``steps`` is not a list of 2-tuples.
        ValueError
            If names are not unique strings, or if an intermediate step is
            missing .fit / .transform.
        """
        if not isinstance(steps, list) or len(steps) == 0:
            raise TypeError(
                "Pipeline 'steps' must be a non-empty list of (name, estimator) tuples. "
                f"Got: {type(steps)}"
            )

        names = []
        for i, step in enumerate(steps):
            # Each element must be a 2-tuple
            if not (isinstance(step, tuple) and len(step) == 2):
                raise TypeError(
                    f"Pipeline step {i} must be a (name, estimator) tuple. "
                    f"Got: {type(step)}"
                )

            name, estimator = step

            # Names must be non-empty strings
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Pipeline step {i}: name must be a non-empty string. Got: {repr(name)}"
                )

            # Names must be unique
            if name in names:
                raise ValueError(
                    f"Pipeline step names must be unique. Duplicate found: '{name}'"
                )
            names.append(name)

            # Every step must have a .fit method
            if not hasattr(estimator, "fit"):
                raise ValueError(
                    f"Step '{name}' (index {i}) does not implement .fit(). "
                    "All pipeline steps must implement .fit(X, y=None)."
                )

        # All steps except the last must also implement .transform
        for i, (name, estimator) in enumerate(steps[:-1]):
            if not hasattr(estimator, "transform"):
                raise ValueError(
                    f"Intermediate step '{name}' (index {i}) does not implement "
                    ".transform(). Only the final step may be a non-transformer."
                )

    # Public properties

    @property
    def named_steps(self):
        """dict: Maps step names to estimator objects for convenient access.

        Returns
        -------
        dict
            ``{name: estimator}`` for every step in the pipeline.

        Examples
        --------
        >>> pipe.named_steps['scale']
        StandardScaler()
        """
        return dict(self.steps)

    @property
    def _final_step(self):
        """(str, estimator): The last (name, object) pair in the pipeline."""
        return self.steps[-1]

    @property
    def _intermediate_steps(self):
        """list: All steps except the last."""
        return self.steps[:-1]

    # Core API

    def fit(self, X, y=None):
        """Fit every step in the pipeline sequentially on X (and y).

        Intermediate steps are fitted and their transformation is applied
        before passing data to the next step.  The final step is fitted on
        the output of all previous transformations.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input training data.
        y : np.ndarray, shape (n_samples,), optional
            Target values.  Passed through to each step unchanged.

        Returns
        -------
        self : Pipeline
            Fitted pipeline (allows method chaining).

        Raises
        ------
        RuntimeError
            If any step raises during fitting.
        """
        X_current = np.array(X, dtype=float)

        # Fit-transform all intermediate steps so each step learns on the
        # already-processed representation of the data.
        for name, transformer in self._intermediate_steps:
            try:
                X_current = transformer.fit_transform(X_current, y)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed during fit at step '{name}': {exc}"
                ) from exc

        # Fit (but do NOT transform) the final step
        final_name, final_estimator = self._final_step
        try:
            final_estimator.fit(X_current, y)
        except Exception as exc:
            raise RuntimeError(
                f"Pipeline failed during fit at final step '{final_name}': {exc}"
            ) from exc

        self._is_fitted = True
        return self

    def transform(self, X):
        """Apply the learned transformations to X.

        The final step must implement .transform().  If the final step is a
        pure estimator (e.g., a classifier), use .predict() instead.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_out : np.ndarray
            Transformed data after passing through every step.

        Raises
        ------
        RuntimeError
            If called before .fit(), or if the final step has no .transform().
        """
        # Inherited guard from BaseTransformer
        super().transform(X)

        final_name, final_estimator = self._final_step
        if not hasattr(final_estimator, "transform"):
            raise RuntimeError(
                f"The final step '{final_name}' does not implement .transform(). "
                "Use .predict() for estimator-ended pipelines."
            )

        X_current = np.array(X, dtype=float)
        for name, transformer in self.steps:
            try:
                X_current = transformer.transform(X_current)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed during transform at step '{name}': {exc}"
                ) from exc

        return X_current

    def fit_transform(self, X, y=None):
        """Fit all steps and return the transformed training data.

        More efficient than calling .fit(X).transform(X) because intermediate
        steps each perform a single fit+transform pass instead of two passes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,), optional
            Target values.

        Returns
        -------
        X_out : np.ndarray
            Fully transformed training data.
        """
        X_current = np.array(X, dtype=float)

        for name, transformer in self._intermediate_steps:
            try:
                X_current = transformer.fit_transform(X_current, y)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed during fit_transform at step '{name}': {exc}"
                ) from exc

        final_name, final_estimator = self._final_step
        if hasattr(final_estimator, "fit_transform"):
            try:
                X_current = final_estimator.fit_transform(X_current, y)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed during fit_transform at final step "
                    f"'{final_name}': {exc}"
                ) from exc
        else:
            # Final step is a plain estimator — just fit, no transform output
            try:
                final_estimator.fit(X_current, y)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed during fit at final step '{final_name}': {exc}"
                ) from exc

        self._is_fitted = True
        return X_current

    def predict(self, X):
        """Transform X through all steps, then call .predict() on the final step.

        Use this when the last pipeline step is a model (e.g., a classifier or
        regressor) rather than a transformer.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predictions from the final estimator.

        Raises
        ------
        RuntimeError
            If called before .fit(), or if the final step has no .predict().
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict.")

        final_name, final_estimator = self._final_step
        if not hasattr(final_estimator, "predict"):
            raise RuntimeError(
                f"Final step '{final_name}' does not implement .predict()."
            )

        # Pass X through all intermediate transformers
        X_current = np.array(X, dtype=float)
        for name, transformer in self._intermediate_steps:
            try:
                X_current = transformer.transform(X_current)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed during predict at step '{name}': {exc}"
                ) from exc

        return final_estimator.predict(X_current)


    def partial_fit(self, X, y=None):
        """Incrementally fit transformers and the final estimator on one chunk.

        Each intermediate step is updated with partial_fit when available,
        otherwise fit is used as a fallback. The transformed chunk is then
        passed forward to the next step. The final estimator must implement
        partial_fit or fit.
        """
        X_current = np.array(X, dtype=float)

        for name, transformer in self._intermediate_steps:
            try:
                if hasattr(transformer, "partial_fit"):
                    transformer.partial_fit(X_current, y)
                else:
                    transformer.fit(X_current, y)
                X_current = transformer.transform(X_current)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed during partial_fit at step '{name}': {exc}"
                ) from exc

        final_name, final_estimator = self._final_step
        try:
            if hasattr(final_estimator, "partial_fit"):
                final_estimator.partial_fit(X_current, y)
            else:
                final_estimator.fit(X_current, y)
        except Exception as exc:
            raise RuntimeError(
                f"Pipeline failed during partial_fit at final step '{final_name}': {exc}"
            ) from exc

        self._is_fitted = True
        return self

    def get_params(self):
        """Return a dict of all step names mapped to their estimator objects.

        Returns
        -------
        dict
            ``{name: estimator}`` for every step.
        """
        return self.named_steps

    def __repr__(self):
        step_repr = "\n  ".join(f"('{n}', {e!r})" for n, e in self.steps)
        return f"Pipeline(steps=[\n  {step_repr}\n])"


# FeatureUnion

class FeatureUnion(BaseTransformer):
    """Runs multiple transformers in parallel and concatenates their outputs.

    Each transformer in the union is fitted independently on the same input X.
    During transform, their outputs are concatenated column-wise (axis=1).

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        Transformers to run in parallel.  Each must implement .fit and .transform.

    Attributes
    ----------
    transformer_list : list of (str, transformer)
        The validated transformers.
    _is_fitted : bool
        True after .fit() has been called.

    Raises
    ------
    TypeError / ValueError
        If the transformer list is malformed (same rules as Pipeline).
    RuntimeError
        If .transform() is called before .fit().

    Time Complexity
    ---------------
    fit       : O(sum of individual fit costs)  — transformers run sequentially
    transform : O(sum of transform costs + concatenation)

    Examples
    --------
    >>> union = FeatureUnion([
    ...     ('numeric', StandardScaler()),
    ...     ('encoded', OneHotEncoder()),
    ... ])
    >>> X_combined = union.fit_transform(X)
    """

    def __init__(self, transformer_list):
        super().__init__()
        self._validate_transformer_list(transformer_list)
        self.transformer_list = list(transformer_list)

    # Validation

    @staticmethod
    def _validate_transformer_list(transformer_list):
        """Check that every entry is a (str, transformer) pair with .fit/.transform.

        Parameters
        ----------
        transformer_list : any

        Raises
        ------
        TypeError
            If not a list of 2-tuples.
        ValueError
            If names are not unique, or a transformer lacks .fit/.transform.
        """
        if not isinstance(transformer_list, list) or len(transformer_list) == 0:
            raise TypeError(
                "FeatureUnion 'transformer_list' must be a non-empty list of "
                f"(name, transformer) tuples. Got: {type(transformer_list)}"
            )

        names = []
        for i, item in enumerate(transformer_list):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError(
                    f"FeatureUnion entry {i} must be a (name, transformer) tuple. "
                    f"Got: {type(item)}"
                )

            name, transformer = item

            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"FeatureUnion entry {i}: name must be a non-empty string. "
                    f"Got: {repr(name)}"
                )

            if name in names:
                raise ValueError(
                    f"FeatureUnion transformer names must be unique. "
                    f"Duplicate: '{name}'"
                )
            names.append(name)

            for method in ("fit", "transform"):
                if not hasattr(transformer, method):
                    raise ValueError(
                        f"FeatureUnion transformer '{name}' (index {i}) must "
                        f"implement .{method}()."
                    )

    # Core API

    def fit(self, X, y=None):
        """Fit each transformer independently on X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, optional

        Returns
        -------
        self : FeatureUnion
        """
        X_arr = np.array(X, dtype=float)
        for name, transformer in self.transformer_list:
            try:
                transformer.fit(X_arr, y)
            except Exception as exc:
                raise RuntimeError(
                    f"FeatureUnion failed during fit at transformer '{name}': {exc}"
                ) from exc
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform X with every transformer and concatenate column-wise.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_out : np.ndarray, shape (n_samples, sum_of_output_features)
            Column-wise concatenation of each transformer's output.

        Raises
        ------
        RuntimeError
            If called before .fit().
        """
        # Inherited fitted guard
        super().transform(X)

        X_arr = np.array(X, dtype=float)
        parts = []
        for name, transformer in self.transformer_list:
            try:
                out = transformer.transform(X_arr)
            except Exception as exc:
                raise RuntimeError(
                    f"FeatureUnion failed during transform at '{name}': {exc}"
                ) from exc

            # Ensure 2-D so hstack works uniformly
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            parts.append(out)

        return np.hstack(parts)

    def get_params(self):
        """Return a dict of transformer names mapped to their objects.

        Returns
        -------
        dict
        """
        return dict(self.transformer_list)

    def __repr__(self):
        tr_repr = "\n  ".join(f"('{n}', {t!r})" for n, t in self.transformer_list)
        return f"FeatureUnion(transformer_list=[\n  {tr_repr}\n])"


# Quick smoke-test

if __name__ == "__main__":
    from utils import BaseTransformer as BT

    # Minimal stub transformers for testing without importing preprocessing
    class _AddOne(BT):
        def fit(self, X, y=None):
            self._is_fitted = True
            return self
        def transform(self, X):
            super().transform(X)
            return X + 1.0

    class _Scale(BT):
        def __init__(self, factor=2.0):
            super().__init__()
            self.factor = factor
        def fit(self, X, y=None):
            self._is_fitted = True
            return self
        def transform(self, X):
            super().transform(X)
            return X * self.factor

    print("=== Pipeline smoke-test ===")
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    pipe = Pipeline([("add", _AddOne()), ("scale", _Scale(factor=3.0))])
    X_out = pipe.fit_transform(X)
    print("Input:\n", X)
    print("Output (add 1 then x3):\n", X_out)
    # Expected: (X + 1) * 3
    assert np.allclose(X_out, (X + 1) * 3), "Pipeline result mismatch!"
    print("Pipeline test PASSED.\n")

    print("=== FeatureUnion smoke-test ===")
    union = FeatureUnion([("add", _AddOne()), ("scale", _Scale(factor=10.0))])
    X_union = union.fit_transform(X)
    print("FeatureUnion output (cols = [X+1, X*10]):\n", X_union)
    assert X_union.shape == (3, 4), "FeatureUnion shape mismatch!"
    print("FeatureUnion test PASSED.")