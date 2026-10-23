import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

THR = 0.87
FACTORS = 32
REGULARIZATION = 0.25
ITERATIONS = 120
CONFIDENCE = 400.0

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

pos = train[train["time"] > THR].copy()

users = np.sort(train["user"].unique())
items = np.sort(train["track"].unique())

u_map = {u: i for i, u in enumerate(users)}
i_map = {t: i for i, t in enumerate(items)}

rows = pos["user"].map(u_map).to_numpy()
cols = pos["track"].map(i_map).to_numpy()

X = sp.csr_matrix(
    (np.ones(len(pos), dtype=np.float32), (rows, cols)),
    shape=(len(users), len(items)),
    dtype=np.float32,
)
X.data[:] = 1.0
X.eliminate_zeros()

item_users = (X.T * CONFIDENCE).tocsr()

model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    random_state=42,
)
model.fit(item_users)

tu = test["user"].map(u_map).to_numpy()
ti = test["track"].map(i_map).to_numpy()

scores = np.sum(
    model.item_factors[tu] * model.user_factors[ti],
    axis=1,
)

submission = test.copy()
submission["score"] = scores
submission[["user", "track", "score"]].to_csv(
    "submission_als.csv", index=False
)

print("Saved submission_als.csv")
