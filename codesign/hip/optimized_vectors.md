# Hip Co-Design Legacy Vector Notes

Historical vectors recovered from older `codesign/hip/hip_optim.py` `__main__` blocks.

## Legacy CMA Start / Tuned Seed (7D)

Source: `codesign/hip/hip_optim.py` in commit `c934f20` (also in `HEAD~1`).

```python
np.array([
    -8.41673617,
    -0.67836718,
    -56.14311573,
    17.5914297,
    -24.48712805,
    -80.15278452,
    0.0,
])
```

Notes:
- Built from `dx6` plus a 7th entry for ankle offset sweep.
- Used as `CMA(mean=np.float64(dx), sigma=1.3)`.
- No explicit scalar cost was stored next to this vector.

## Additional Historical Candidate Vectors (6D)

Source: same file in commit `c934f20` (`dx1` ... `dx10`).

`dx1`
```python
np.array([0.0, 0.0, 0.0, 90.0, 0.0, 0.0])
```

`dx2`
```python
np.array([0.0, 0.0, 0.0, 45.0, 0.0, 0.0])
```

`dx3`
```python
np.array([-6.30157784, -2.9773718, -67.92956906, 37.73483886, -23.44075896, -79.37300617])
```

`dx4`
```python
np.array([-6.65684339, -1.73665785, -68.81551501, 37.43446151, -23.24196683, -78.50777904])
```

`dx5`
```python
np.array([-5.76495799, -2.60621304, -62.80618123, 36.23841476, -22.81023506, -78.14087952])
```

`dx6`
```python
np.array([-8.41673617, -0.67836718, -56.14311573, 17.5914297, -24.48712805, -80.15278452])
```

`dx7`
```python
np.array([-7.54528979, -0.98467531, -57.23192973, 15.80347566, -25.11051615, -76.66889449])
```

`dx8`
```python
np.array([-8.10875616, -0.8085541, -56.51608165, 14.00093572, -11.02247696, -84.22985613])
```

`dx9`
```python
np.array([-7.07608636, -1.1400171, -57.48542757, 13.71672827, -11.36045942, -83.37281477])
```

`dx10`
```python
np.array([-8.08829439, -0.20494419, -56.72015355, 18.32296692, -23.56834157, -80.45340603])
```

Notes:
- These vectors were used as manual evaluation candidates/checkpoints.
- No inline numeric objective values were stored next to these hip vectors in code history.
