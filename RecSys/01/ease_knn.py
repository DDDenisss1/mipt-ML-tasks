import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Parameters
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 2400.0
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.87
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
BETA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

K = 200
ALS_FILE = "submission_als.csv"

if not 0.0 <= ALPHA <= 1.0:
    raise ValueError("ALPHA must be between 0 and 1")

if not 0.0 <= BETA <= 1.0:
    raise ValueError("BETA must be between 0 and 1")

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

required_train = {"user", "track", "time"}
required_test = {"user", "track"}

if not required_train.issubset(train.columns):
    raise ValueError("train.csv must contain user, track, time")

if not required_test.issubset(test.columns):
    raise ValueError("test.csv must contain user, track")

pos = train[train["time"] > THR].copy()

# Build mappings from all training users and tracks
uniq_u = np.sort(train["user"].unique())
uniq_i = np.sort(train["track"].unique())

u_pos = {u: k for k, u in enumerate(uniq_u)}
i_pos = {t: k for k, t in enumerate(uniq_i)}

# Build binary user-item matrix
def build_matrix(df):
    r = df["user"].map(u_pos).to_numpy()
    c = df["track"].map(i_pos).to_numpy()

    if pd.isna(r).any() or pd.isna(c).any():
        raise ValueError("Unknown user or track while building matrix")

    X = sp.csr_matrix(
        (
            np.ones(len(df), dtype=np.float32),
            (r, c),
        ),
        shape=(len(uniq_u), len(uniq_i)),
        dtype=np.float32,
    )

    # Binary interactions; repeated interactions count once
    X.data[:] = 1.0
    X.eliminate_zeros()
    return X

def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std > 0:
        return (x - x.mean()) / std
    return x

# EASE
def ease(X, lam):
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0.0)
    return B

# Train EASE
print(f"Training EASE: LAM={LAM}, THR={THR}")

X = build_matrix(pos)
B_ease = ease(X, LAM)
S_ease = (X @ B_ease).astype(np.float32)

# Candidate pairs from test.csv, preserving original order
tu = test["user"].map(u_pos).to_numpy()
ti = test["track"].map(i_pos).to_numpy()

if pd.isna(tu).any() or pd.isna(ti).any():
    raise ValueError("Test contains users or tracks absent from training")

ease_scores = np.asarray(S_ease[tu, ti]).ravel()
ease_scores = standardize(ease_scores)

# Item-Item KNN
print(f"Building Item-Item KNN: K={K}")

C = (X.T @ X).tocsr().astype(np.float32)
C.setdiag(0)
C.eliminate_zeros()

item_freq = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
inv_freq = 1.0 / np.sqrt(np.maximum(item_freq, 1.0))

D = sp.diags(inv_freq)
C = (D @ C @ D).tocsr()

rows = []
cols = []
vals = []

for i in range(C.shape[0]):
    start, end = C.indptr[i], C.indptr[i + 1]
    row_cols = C.indices[start:end]
    row_vals = C.data[start:end]

    if len(row_vals) > K:
        idx = np.argpartition(row_vals, -K)[-K:]
        row_cols = row_cols[idx]
        row_vals = row_vals[idx]

    rows.extend([i] * len(row_cols))
    cols.extend(row_cols)
    vals.extend(row_vals)

W = sp.csr_matrix(
    (
        np.asarray(vals, dtype=np.float32),
        (
            np.asarray(rows, dtype=np.int32),
            np.asarray(cols, dtype=np.int32),
        ),
    ),
    shape=C.shape,
)

S_knn = X @ W
knn_scores = np.asarray(S_knn[tu, ti]).ravel()
knn_scores = standardize(knn_scores)

# EASE + KNN ensemble
final_scores = (
    (1.0 - ALPHA) * ease_scores
    + ALPHA * knn_scores
)


# Optional ALS blending
if BETA > 0:
    print(f"Loading ALS predictions: {ALS_FILE}")

    als_sub = pd.read_csv(ALS_FILE)

    required_als = {"user", "track", "score"}

    if not required_als.issubset(als_sub.columns):
        raise ValueError(
            f"{ALS_FILE} must contain columns: user, track, score"
        )

    if als_sub[["user", "track", "score"]].isna().any().any():
        raise ValueError(f"{ALS_FILE} contains missing values")

    # Average repeated predictions for identical user-track pairs
    als_sub = (
        als_sub.groupby(
            ["user", "track"],
            as_index=False,
            sort=False,
        )["score"]
        .mean()
    )

    # Use a MultiIndex to align ALS predictions to test pairs
    als_index = pd.MultiIndex.from_frame(
        als_sub[["user", "track"]]
    )

    test_index = pd.MultiIndex.from_frame(
        test[["user", "track"]]
    )

    als_series = pd.Series(
        als_sub["score"].to_numpy(dtype=np.float64),
        index=als_index,
    )

    als_scores = als_series.reindex(test_index).to_numpy()

    if not np.isfinite(als_scores).all():
        missing = int((~np.isfinite(als_scores)).sum())
        raise ValueError(
            f"{missing} test rows have missing or invalid ALS scores"
        )

    als_scores = standardize(als_scores)
    final_scores = (
        (1.0 - BETA) * standardize(final_scores)
        + BETA * als_scores
    )

# Save submission
if not np.isfinite(final_scores).all():
    raise ValueError("Final scores contain NaN or infinite values")

submission = test.copy()
submission["score"] = final_scores

out = (
    f"CONFIDENCE_480.0_factors_32_ITERATIONS_120.csv"
)

submission[["user", "track", "score"]].to_csv(
    out,
    index=False,
)

print(f"Saved {out}")
print(
    f"Parameters: LAM={LAM}, THR={THR}, "
    f"K={K}, ALPHA={ALPHA}, BETA={BETA}"
)
print(f"Rows written: {len(submission)}")
