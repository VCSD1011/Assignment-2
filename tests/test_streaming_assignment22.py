from env import *

from numcompute.preprocessing import StandardScaler, Imputer, OneHotEncoder
from numcompute.pipeline import Pipeline
from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import EnsembleClassifier
from numcompute.metrics import StreamingClassificationMetrics
from numcompute.stats import StreamingStats
from numcompute.stream import StreamTrainer


class TestAssignment22Streaming(unittest.TestCase):

    def test_standard_scaler_partial_fit_matches_batch(self):
        X1 = np.array([[1.0, 2.0], [3.0, np.nan]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        batch = StandardScaler().fit(np.vstack([X1, X2]))
        stream = StandardScaler().partial_fit(X1).partial_fit(X2)
        assert_allclose(stream.mean_, batch.mean_)
        assert_allclose(stream.var_, batch.var_)

    def test_imputer_and_encoder_partial_fit(self):
        imp = Imputer(strategy="mean").partial_fit(np.array([[1.0], [np.nan]])).partial_fit(np.array([[3.0]]))
        self.assertAlmostEqual(imp.statistics[0], 2.0)
        enc = OneHotEncoder(handle_unknown="ignore").partial_fit(np.array([["a"], ["b"]], dtype=object))
        enc.partial_fit(np.array([["c"]], dtype=object))
        self.assertEqual(enc.transform(np.array([["c"]], dtype=object)).shape[1], 3)

    def test_streaming_metrics_update_result_reset(self):
        m = StreamingClassificationMetrics(labels=[0, 1])
        m.update([1, 0], [1, 1]).update([0, 1], [0, 1])
        res = m.result()
        self.assertEqual(res["total"], 4)
        self.assertEqual(res["accuracy"], 0.75)
        m.reset()
        self.assertEqual(m.result()["total"], 0)

    def test_streaming_stats(self):
        s = StreamingStats().update_stats([[1, 2], [3, 4]]).update_stats([[5, 6]])
        res = s.result()
        assert_allclose(res["mean"], [3, 4])
        assert_allclose(res["variance"], [8/3, 8/3])

    def test_tree_ensemble_pipeline_and_stream_trainer(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3]], dtype=float)
        y = np.array([0, 0, 1, 1, 1, 1])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1)),
        ])
        trainer = StreamTrainer(pipe, labels=[0, 1])
        trainer.fit_chunk(X[:3], y[:3])
        trainer.score_chunk(X[:3], y[:3])
        trainer.fit_chunk(X[3:], y[3:])
        log = trainer.score_chunk(X[3:], y[3:])
        self.assertIn("cumulative_accuracy", log)
        self.assertEqual(len(pipe.predict(X)), len(y))

    def test_decision_tree_partial_fit(self):
        tree = DecisionTreeClassifier(max_depth=2)
        tree.partial_fit([[0], [1]], [0, 1])
        tree.partial_fit([[2], [3]], [1, 1])
        preds = tree.predict([[0], [3]])
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], 1)


if __name__ == "__main__":
    unittest.main()
