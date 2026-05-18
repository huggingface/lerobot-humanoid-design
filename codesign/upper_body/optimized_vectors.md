# Upper-Body Co-Design Legacy Vector Notes

Historical vectors recovered from older `codesign/upper_body/arm_evaluation.py` `__main__` blocks.

## Legacy CMA Start / Tuned Seed (6D)

Source: `codesign/upper_body/arm_evaluation.py` in commit `cf07b56` (also present in `HEAD~1` before split).

```python
np.array([45.9095, 137.2909, -81.4492, -30.54, -9.0298, 37.7497])
```

Notes:
- Used as `CMA(mean=np.float64(dx), sigma=15)`.
- Represents shoulder-only optimization (6D).

## Additional Historical Candidate Vectors (8D)

Unlabeled exploratory candidates from the same historical `__main__` block.

`dx1`
```python
np.array([-1.2386, 0.7196, 2.974, -1.0484, -4.4469, -5.9738, -2.6952, -4.374])
```

`dx2`
```python
np.array([-2.3684, -0.8973, 3.9837, -3.3336, -5.7216, -3.4934, -3.1598, -2.4684])
```

`dx3`
```python
np.array([-0.1227, 0.4567, 5.4507, -2.4195, -6.3413, -6.6535, -5.2577, -5.1175])
```

`dx4`
```python
np.array([13.6416, 38.854, 64.0544, 58.2893, 18.4691, 15.987, 13.7576, -21.0037])
```

## Historical Vectors With Attached Values

These had explicit value annotations in code comments.

1. Value `2116`
```python
np.array([45.9095, 137.2909, -81.4492, -30.54, -9.0298, 37.7497])
```

2. Value `1875`
```python
np.array([54.262, 153.5354, -97.7809, -53.5955, -12.3195, 29.3566])
```

3. Value `2163`
```python
np.array([54.0, 153.5, -97.0, -53.5, -12.0, 29.0])
```
