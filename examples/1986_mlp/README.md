# Example 01: The MLP Revival (1986)



##  Structure

### 01_rumelhart_tinydigits.py
**Purpose:** Prove MLPs work on real images (fast iteration)

- **Dataset:** TinyDigits (1000 train + 200 test, 8×8 images)
- **Architecture:** Input(64) → Linear(64→32) → ReLU → Linear(32→10)
- **Expected:** 85%+ accuracy in a few minutes
- **Key Learning:** "MLPs can learn hierarchical features from images!"

**Why TinyDigits First?**
- Fast training = quick feedback loop
- Small size = easy to understand what's happening
- Decent accuracy = proves concept works
- Ships with TinyTorch = no downloads needed

## Expected Results

<table width="100%">
  <thead>
<tr>
<th width="18%"><b>Script</b></th>
<th width="12%">Dataset</th>
<th width="12%">Image Size</th>
<th width="15%">Parameters</th>
<th width="12%">Loss</th>
<th width="15%">Accuracy</th>
<th width="16%">Training Time</th>
</tr>
</thead>
<tbody>
<tr><td><b>(TinyDigits)</b></td><td>1K train</td><td>8×8</td><td>~2.4K</td><td>&lt; 0.5</td><td>85%+</td><td>3-5 min</td></tr>
</tbody>
</table>

## Key outcome: Hierarchical Feature Learning

MLPs don't just memorize - they learn useful internal representations:

**Hidden Layer Discovers:**
- Edge detectors (low-level features)
- Curve patterns (mid-level features)
- Digit-specific combinations (high-level features)



## Running the example

```bash
cd examples/1986_mlp

pip install rich
python 01_rumelhart_tinydigits.py
```

## Achievement Unlocked

- How MLPs learn hierarchical features from raw pixels
- Why hidden layers discover useful representations
- The power of backpropagation for multi-layer training
- How to scale from toy datasets to real benchmarks


---
