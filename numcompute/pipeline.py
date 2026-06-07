import numpy as np
if __name__ == "__main__":
    from utils import BaseTransformer
else:
    from numcompute.utils import BaseTransformer


# Pipeline

class Pipeline(BaseTransformer):

    def __init__(self, steps):
        # Inherit _is_fitted safety flag from BaseEstimator via BaseTransformer
        super().__init__()
        self._validate_steps(steps)
        self.steps = list(steps)

    # Validation helpers

    @staticmethod
    def _validate_steps(steps):

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

        return self.named_steps

    def __repr__(self):
        step_repr = "\n  ".join(f"('{n}', {e!r})" for n, e in self.steps)
        return f"Pipeline(steps=[\n  {step_repr}\n])"


# FeatureUnion

class FeatureUnion(BaseTransformer):


    def __init__(self, transformer_list):
        super().__init__()
        self._validate_transformer_list(transformer_list)
        self.transformer_list = list(transformer_list)

    # Validation

    @staticmethod
    def _validate_transformer_list(transformer_list):

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
