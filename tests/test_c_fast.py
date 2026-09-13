import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.generate_c_fast import c_model_id, prune_network, quantize_network_static
from model_zoo.core import sha256
from model_zoo.loaders import unified


class CFastGenerationTests(unittest.TestCase):
    def test_fixed_id_mapping_preserves_four_transform_slots(self):
        self.assertEqual(c_model_id("A001", "ft5"), "C001")
        self.assertEqual(c_model_id("A001", "prune50"), "C002")
        self.assertEqual(c_model_id("A001", "ptq_int8"), "C003")
        self.assertEqual(c_model_id("A001", "prune20"), "C004")
        self.assertEqual(c_model_id("A002", "ft5"), "C005")
        self.assertEqual(c_model_id("A029", "prune50"), "C038")
        self.assertEqual(c_model_id("A029", "ptq_int8"), "C039")
        self.assertEqual(c_model_id("A029", "prune20"), "C040")

    def test_global_pruning_masks_half_and_removes_reparameterization(self):
        network = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, bias=True),
            nn.Flatten(),
            nn.Linear(18, 2, bias=True),
        )
        with torch.no_grad():
            for index, parameter in enumerate(network.parameters()):
                parameter.copy_(
                    torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter)
                    + index * 100
                )

        _, details = prune_network(network, amount=0.5)

        self.assertEqual(details["eligible_weight_elements"], 44)
        self.assertEqual(details["masked_weight_elements"], 22)
        self.assertEqual(details["actual_mask_sparsity"], 0.5)
        self.assertFalse(details["recovery_fine_tuning"])
        self.assertFalse(any(name.endswith("weight_orig") for name, _ in network.named_parameters()))
        self.assertFalse(any(name.endswith("weight_mask") for name, _ in network.named_buffers()))
        # Biases are outside the pruning target and therefore remain nonzero.
        self.assertTrue(torch.count_nonzero(network[0].bias) == network[0].bias.numel())
        self.assertTrue(torch.count_nonzero(network[2].bias) == network[2].bias.numel())

    def test_prune20_uses_a_distinct_exact_mask_ratio(self):
        network = nn.Sequential(nn.Linear(10, 10, bias=False))
        with torch.no_grad():
            network[0].weight.copy_(torch.arange(1, 101, dtype=torch.float32).reshape(10, 10))
        _, details = prune_network(network, amount=0.2)
        self.assertEqual(details["eligible_weight_elements"], 100)
        self.assertEqual(details["masked_weight_elements"], 20)
        self.assertEqual(details["actual_mask_sparsity"], 0.2)

    def test_static_ptq_contains_quantized_weights_and_activation_boundary(self):
        network = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 10),
        ).eval()
        images = torch.rand(8, 3, 32, 32)
        labels = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        mean = torch.zeros(1, 3, 1, 1)
        std = torch.ones(1, 3, 1, 1)

        scripted, details = quantize_network_static(network, loader, mean, std)
        with torch.inference_mode():
            output = scripted(torch.rand(3, 3, 32, 32))

        self.assertEqual(tuple(output.shape), (3, 10))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(details["quantized_weight_module_count"], 0)
        self.assertGreater(details["activation_quantizer_count"], 0)
        self.assertEqual(details["calibration_samples"], 8)

        with TemporaryDirectory() as directory:
            path = f"{directory}/quantized.pt"
            torch.jit.save(scripted, path)
            row = {
                "model_id": "C_TEST",
                "status": "verified",
                "checkpoint_format": "torchscript_int8",
                "local_checkpoint_path": path,
                "sha256": sha256(path),
                "normalization_mean": [0.0, 0.0, 0.0],
                "normalization_std": [1.0, 1.0, 1.0],
            }
            with patch.object(unified, "models", return_value=[row]):
                loaded = unified.load_model("C_TEST", device="cpu")
                with torch.inference_mode():
                    loaded_output = loaded(torch.rand(2, 3, 32, 32))
                self.assertEqual(tuple(loaded_output.shape), (2, 10))
                with self.assertRaisesRegex(ValueError, "CPU quantized operators"):
                    unified.load_model("C_TEST", device="cuda")


if __name__ == "__main__":
    unittest.main()
