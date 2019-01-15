import unittest

import torch

from mobile_block import MobileBlock


class MyTestCase(unittest.TestCase):
    def test_mobile_block(self):
        mock_block_id = 'w112_i16_o32_s1_e10_k3_g1'
        block = MobileBlock.factory(mock_block_id)
        self.assertEqual(block.block_id, mock_block_id)

        self.assertEqual(block.input_size, 112)
        self.assertEqual(block.in_channels, 16)
        self.assertEqual(block.out_channels, 32)
        self.assertEqual(block.stride, 1)
        self.assertEqual(block.expansion, 10)
        self.assertEqual(block.kernel, 3)
        self.assertEqual(block.groups, 1)

        # noinspection PyUnresolvedReferences
        mock_input = torch.randn(1, 16, 112, 112)
        mock_output = block(mock_input)
        self.assertEqual(mock_output.shape, (1, 32, 112, 112))


if __name__ == '__main__':
    unittest.main()
