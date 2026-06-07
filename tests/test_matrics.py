from env import *
from numcompute.metrics import *
import unittest
class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        # Basic binary classification data
        self.y_true = [1, 0, 1, 1, 0, 1, 0, 0]
        self.y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
        # Probability scores for ROC
        self.y_scores = [0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3]
        # Regression data
        self.reg_true = [3.0, -0.5, 2.0, 7.0]
        self.reg_pred = [2.5, 0.0, 2.0, 8.0]

    ## --- Classification Tests ---

    def test_accuracy(self):
        # 6 matches out of 8 = 0.75
        self.assertEqual(accuracy(self.y_true, self.y_pred), 0.75)

    def test_confusion_matrix(self):
        # TP: 3, TN: 3, FP: 1, FN: 1
        cm = confusion_matrix(self.y_true, self.y_pred)
        expected = np.array([[3, 1], [1, 3]])
        np.testing.assert_array_equal(cm, expected)

    def test_precision_recall_f1(self):
        # Precision: 3 / (3 + 1) = 0.75
        # Recall: 3 / (3 + 1) = 0.75
        # F1: 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
        self.assertEqual(precision(self.y_true, self.y_pred), 0.75)
        self.assertEqual(recall(self.y_true, self.y_pred), 0.75)
        self.assertEqual(f1(self.y_true, self.y_pred), 0.75)

    def test_metrics_division_by_zero(self):
        # Case where there are no positive predictions
        yt = [1, 0]
        yp = [0, 0]
        self.assertEqual(precision(yt, yp), 0.0)
        self.assertEqual(f1(yt, yp), 0.0)

    ## --- Regression Tests ---

    def test_mse(self):
        # Errors: [0.5, -0.5, 0.0, -1.0]
        # Squared: [0.25, 0.25, 0.0, 1.0]
        # Mean: 1.5 / 4 = 0.375
        self.assertEqual(mse(self.reg_true, self.reg_pred), 0.375)

    ## --- ROC / AUC Tests ---

    def test_roc_auc(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        score = auc(fpr, tpr)
        # AUC should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # Verify first element is (0,0) as per your implementation
        self.assertEqual(fpr[0], 0.0)
        self.assertEqual(tpr[0], 0.0)

if __name__ == '__main__':
    unittest.main()