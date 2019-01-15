# mobile-block [![Build Status](https://travis-ci.com/FebruaryBreeze/mobile-block.svg?branch=master)](https://travis-ci.com/FebruaryBreeze/mobile-block)

MobileNet-style block for PyTorch

### Usage

```python
from mobile_block import MobileBlock

# normal usage
block = MobileBlock(
  input_size=112,
  in_channels=16,
  out_channels=32,
  stride=1,
  expansion=6,
  kernel=3,
  groups=1
)

# factory usage
mock_block_id = 'w112_i16_o32_s1_e6_k3_g1'
block = MobileBlock.factory(mock_block_id)
```
