import unittest

import torch
from torch import nn

from scripts.generate_c_fast import c_model_id, prune_network


class CFastGenerationTests(unittest.TestCase):
    def test_fixed_id_mapping_preserves_four_transform_slots(self):
        self.assertEqual(c_model_id("A001", "ft5"), "C001")
        self.assertEqual(c_model_id("A001", "prune50"), "C002")
        self.assertEqual(c_model_id("A002", "ft5"), "C005")
        self.assertEqual(c_model_id("A029", "prune50"), "C038")

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


if __name__ == "__main__":
    unittest.main()
