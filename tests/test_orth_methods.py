import unittest
import torch
import sys
import os

# Ensure we import from the local source, not installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fca import (
    orthogonalize_vector,
    orthogonalize_vector_mgs,
    orthogonalize_batch_qr,
    gram_schmidt,
    FunctionalComponentAnalysis,
)


class TestModifiedGramSchmidt(unittest.TestCase):
    """Tests for Modified Gram-Schmidt (MGS) orthogonalization."""

    def setUp(self):
        self.size = 10
        self.vec = torch.randn(self.size)
        self.prev_vecs = [torch.randn(self.size) for _ in range(3)]
        # Pre-orthogonalize prev_vecs for testing
        self.ortho_prev = gram_schmidt(self.prev_vecs)

    def test_mgs_produces_unit_vectors(self):
        """MGS output should be normalized."""
        result = orthogonalize_vector_mgs(self.vec, self.ortho_prev)
        self.assertAlmostEqual(torch.norm(result, 2).item(), 1.0, places=5)

    def test_mgs_produces_orthogonal_vectors(self):
        """MGS output should be orthogonal to previous vectors."""
        result = orthogonalize_vector_mgs(self.vec, self.ortho_prev)
        for prev in self.ortho_prev:
            self.assertAlmostEqual(
                torch.dot(result, prev).item(), 0.0, places=5
            )

    def test_mgs_with_empty_prev_vectors(self):
        """MGS should just normalize when no previous vectors."""
        result = orthogonalize_vector_mgs(self.vec, [])
        expected = self.vec / torch.norm(self.vec, 2)
        self.assertAlmostEqual(
            ((result - expected) ** 2).mean().item(), 0.0, places=5
        )

    def test_mgs_with_none_prev_vectors(self):
        """MGS should handle None prev_vectors."""
        result = orthogonalize_vector_mgs(self.vec, None)
        expected = self.vec / torch.norm(self.vec, 2)
        self.assertAlmostEqual(
            ((result - expected) ** 2).mean().item(), 0.0, places=5
        )

    def test_mgs_with_matrix_input(self):
        """MGS should accept matrix form of previous vectors."""
        matrix = torch.stack(self.ortho_prev)
        result = orthogonalize_vector_mgs(self.vec, matrix)
        self.assertAlmostEqual(torch.norm(result, 2).item(), 1.0, places=5)
        for prev in self.ortho_prev:
            self.assertAlmostEqual(
                torch.dot(result, prev).item(), 0.0, places=5
            )

    def test_mgs_double_precision(self):
        """MGS should work with double precision."""
        result = orthogonalize_vector_mgs(
            self.vec, self.ortho_prev, double_precision=True
        )
        self.assertEqual(result.dtype, self.vec.dtype)
        self.assertAlmostEqual(torch.norm(result, 2).item(), 1.0, places=5)

    def test_mgs_no_norm(self):
        """MGS should optionally skip normalization."""
        result = orthogonalize_vector_mgs(self.vec, self.ortho_prev, norm=False)
        # Result should be orthogonal but not normalized
        for prev in self.ortho_prev:
            self.assertAlmostEqual(
                torch.dot(result, prev).item(), 0.0, places=5
            )

    def test_mgs_via_orthogonalize_vector(self):
        """Test MGS via the method parameter of orthogonalize_vector."""
        result1 = orthogonalize_vector_mgs(self.vec, self.ortho_prev)
        result2 = orthogonalize_vector(
            self.vec, self.ortho_prev, method="modified"
        )
        self.assertAlmostEqual(
            ((result1 - result2) ** 2).mean().item(), 0.0, places=5
        )


class TestHouseholderQR(unittest.TestCase):
    """Tests for Householder QR orthogonalization."""

    def setUp(self):
        self.size = 10
        self.vectors = [torch.randn(self.size) for _ in range(5)]

    def test_qr_produces_orthonormal_vectors(self):
        """QR output should be orthonormal."""
        result = orthogonalize_batch_qr(self.vectors)
        self.assertEqual(len(result), len(self.vectors))

        for i, v in enumerate(result):
            # Check unit norm
            self.assertAlmostEqual(torch.norm(v, 2).item(), 1.0, places=5)
            # Check orthogonality
            for j, u in enumerate(result):
                if i != j:
                    self.assertAlmostEqual(
                        torch.dot(v, u).item(), 0.0, places=5
                    )

    def test_qr_with_empty_list(self):
        """QR should handle empty input."""
        result = orthogonalize_batch_qr([])
        self.assertEqual(result, [])

    def test_qr_with_matrix_input(self):
        """QR should accept tensor matrix input."""
        matrix = torch.stack(self.vectors)
        result = orthogonalize_batch_qr(matrix)
        self.assertEqual(len(result), len(self.vectors))

        for i, v in enumerate(result):
            self.assertAlmostEqual(torch.norm(v, 2).item(), 1.0, places=5)

    def test_qr_double_precision(self):
        """QR should work with double precision."""
        result = orthogonalize_batch_qr(self.vectors, double_precision=True)
        self.assertEqual(result[0].dtype, self.vectors[0].dtype)
        self.assertEqual(len(result), len(self.vectors))

    def test_qr_preserves_span(self):
        """QR output should span same subspace as input."""
        result = orthogonalize_batch_qr(self.vectors)
        # The span of result should contain projections of original vectors
        Q = torch.stack(result)
        for v in self.vectors:
            v_norm = v / torch.norm(v, 2)
            # Project v onto span(Q)
            proj = torch.matmul(Q.T, torch.matmul(Q, v_norm))
            # proj should equal v_norm (up to numerical precision)
            self.assertAlmostEqual(
                torch.norm(proj - v_norm).item(), 0.0, places=4
            )


class TestFCAOrthMethods(unittest.TestCase):
    """Tests for FCA with different orth_method settings."""

    def setUp(self):
        self.size = 50
        self.init_rank = 10

    def test_classical_method(self):
        """Test FCA with classical orthogonalization."""
        fca = FunctionalComponentAnalysis(
            size=self.size,
            init_rank=self.init_rank,
            orth_method="classical",
        )
        weight = fca.weight
        self._check_orthogonality(weight)

    def test_modified_method(self):
        """Test FCA with modified Gram-Schmidt."""
        fca = FunctionalComponentAnalysis(
            size=self.size,
            init_rank=self.init_rank,
            orth_method="modified",
        )
        weight = fca.weight
        self._check_orthogonality(weight)

    def test_householder_method(self):
        """Test FCA with Householder QR."""
        fca = FunctionalComponentAnalysis(
            size=self.size,
            init_rank=self.init_rank,
            orth_method="householder",
        )
        weight = fca.weight
        self._check_orthogonality(weight)

    def test_hybrid_method(self):
        """Test FCA with hybrid orthogonalization."""
        fca = FunctionalComponentAnalysis(
            size=self.size,
            init_rank=self.init_rank,
            orth_method="hybrid",
        )
        weight = fca.weight
        self._check_orthogonality(weight)

    def test_backward_compatibility(self):
        """Ensure orth_method='classical' matches default behavior."""
        fca_default = FunctionalComponentAnalysis(
            size=self.size,
            init_rank=self.init_rank,
        )
        fca_classical = FunctionalComponentAnalysis(
            size=self.size,
            init_rank=self.init_rank,
            orth_method="classical",
        )
        # Both should have same orth_method
        self.assertEqual(fca_default.orth_method, "classical")
        self.assertEqual(fca_classical.orth_method, "classical")

    def test_all_methods_gradient_flow(self):
        """Verify gradients flow through all methods."""
        for method in ["classical", "modified", "householder", "hybrid"]:
            fca = FunctionalComponentAnalysis(
                size=self.size,
                init_rank=self.init_rank,
                orth_method=method,
            )
            # Forward pass
            x = torch.randn(4, self.size)
            out = fca(x)
            loss = out.sum()

            # Backward pass
            loss.backward()

            # Check that gradients exist
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in fca.parameters()
            )
            self.assertTrue(
                has_grad,
                f"Method '{method}' should have gradients flowing"
            )

    def test_reorthogonalize(self):
        """Test the reorthogonalize() utility method."""
        fca = FunctionalComponentAnalysis(
            size=self.size,
            init_rank=self.init_rank,
            orth_method="classical",
        )

        # Manually corrupt orthogonality slightly
        with torch.no_grad():
            fca.parameters_list[0].data += 0.1 * torch.randn(self.size)

        # Re-orthogonalize
        fca.reorthogonalize()

        # Check orthogonality is restored
        weight = fca.weight
        self._check_orthogonality(weight)

    def test_method_with_frozen_params(self):
        """Test orthogonalization methods with frozen parameters."""
        for method in ["classical", "modified", "hybrid"]:
            fca = FunctionalComponentAnalysis(
                size=self.size,
                init_rank=5,
                orth_method=method,
            )
            # Freeze first 3 components
            fca.freeze_parameters()
            # Add new trainable components
            fca.add_component()
            fca.add_component()

            weight = fca.weight
            self._check_orthogonality(weight, tolerance=1e-4)

    def _check_orthogonality(self, weight, tolerance=1e-4):
        """Helper to check orthogonality of weight matrix."""
        for i in range(weight.shape[0]):
            # Check unit norm
            self.assertAlmostEqual(
                torch.norm(weight[i], 2).item(), 1.0, places=4
            )
            # Check orthogonality
            for j in range(i):
                dot = torch.dot(weight[i], weight[j]).item()
                self.assertTrue(
                    abs(dot) < tolerance,
                    f"Vectors {i} and {j} not orthogonal: dot={dot}"
                )


class TestNumericalStability(unittest.TestCase):
    """Tests comparing numerical stability across methods."""

    def test_high_rank_orthogonality(self):
        """Compare orthogonality error at high ranks."""
        size = 100
        rank = 50

        methods = ["classical", "modified", "householder", "hybrid"]
        errors = {}

        for method in methods:
            fca = FunctionalComponentAnalysis(
                size=size,
                init_rank=rank,
                orth_method=method,
                orth_with_doubles=False,  # Use float32 to expose stability issues
            )

            weight = fca.weight
            max_error = 0.0

            for i in range(weight.shape[0]):
                for j in range(i):
                    dot = abs(torch.dot(weight[i], weight[j]).item())
                    if dot > max_error:
                        max_error = dot

            errors[method] = max_error

        # All methods should produce orthogonal results
        for method, error in errors.items():
            self.assertTrue(
                error < 1e-3,
                f"Method '{method}' has error {error} at rank {rank}"
            )

    def test_double_precision_improves_stability(self):
        """Double precision should improve stability for all methods."""
        size = 100
        rank = 80

        for method in ["classical", "modified"]:
            fca_float = FunctionalComponentAnalysis(
                size=size,
                init_rank=rank,
                orth_method=method,
                orth_with_doubles=False,
            )
            fca_double = FunctionalComponentAnalysis(
                size=size,
                init_rank=rank,
                orth_method=method,
                orth_with_doubles=True,
            )

            error_float = self._max_orthogonality_error(fca_float.weight)
            error_double = self._max_orthogonality_error(fca_double.weight)

            # Double precision should be at least as good
            self.assertTrue(
                error_double <= error_float + 1e-6,
                f"Method '{method}': double ({error_double}) should be <= float ({error_float})"
            )

    def _max_orthogonality_error(self, weight):
        """Calculate maximum orthogonality error in weight matrix."""
        max_error = 0.0
        for i in range(weight.shape[0]):
            for j in range(i):
                dot = abs(torch.dot(weight[i], weight[j]).item())
                if dot > max_error:
                    max_error = dot
        return max_error


if __name__ == '__main__':
    unittest.main()
