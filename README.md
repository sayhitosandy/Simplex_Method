# Revised Simplex Method
Implementation of Revised Simplex Method in Python (My Assignment in Linear Optimization course [MTH305]  [IIIT-Delhi]). [Algo from `Introduction to Linear Optimization by Dimitris Bertsimas and John N. Tsitsiklis`]

Further, we solved the following Linear Problem:
```
min   1'u + 1'v
st. 
      a'xi + b >=  1 - ui      i=1..N
      a'yj + b <= -1 + vj      j=1..M
      u >= 0
      v >= 0

```
We were provided with:
1. `DB_Vecs.npy` (Sequence vectors for training)
2. `DB_Labels.npy` (corresponding labels)
3. `Q_Vecs.npy` (Test sequences)

We were to submit the results (Labels for `Q_vecs`) as list/numpy array.
