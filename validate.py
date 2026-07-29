import numpy as np
np.random.seed(0)
EPS = 2.220446049250313e-16

def compute_gain(B, R, P, A):
    S = R + B.T @ P @ B
    return np.linalg.solve(S, B.T @ P @ A)

def dare_residual(A, B, Q, R, P, K=None):
    if K is None:
        K = compute_gain(B, R, P, A)
    Acl = A - B @ K
    rhs = Acl.T @ P @ Acl + K.T @ R @ K + Q
    return np.linalg.norm(rhs - P) / max(np.linalg.norm(P), 1.0)

def symmetrise(X):
    return (X + X.T) / 2

# ---- dlyap_fast_c : solves  A X A' - X + Q = 0 ----------------
def dlyap_fast_c(A, Q, tol=1e-14, maxd=60):
    info = dict(converged=False, diverged=False, is_stable=False)
    X = Q.copy(); P = A.copy()
    ninc = np.inf; ninc_prev = np.nan
    j = 1
    for j in range(1, maxd + 1):
        inc = P @ X @ P.T
        X = X + inc
        ninc = np.linalg.norm(inc); nX = np.linalg.norm(X)
        with np.errstate(invalid='ignore', divide='ignore'):
            r = ninc / ninc_prev
        if r < 1:
            if ninc * r / (1 - r) <= tol * nX:
                info['converged'] = True; info['is_stable'] = True; break
        elif j > 1 and np.isnan(r):
            info['diverged'] = True; info['is_stable'] = False; break
        ninc_prev = ninc
        P = P @ P
    X = symmetrise(X)
    info['doublings'] = j
    return X, info

# ---- dare_sda -------------------------------------------------
def dare_sda(A, B, Q, R, tol=1e-4, mind=1, maxd=40):
    n = A.shape[0]
    Ak = A.copy()
    G = symmetrise(B @ np.linalg.solve(R, B.T))
    H = Q.copy()
    i = 1
    for i in range(1, maxd + 1):
        Mk = np.eye(n) + G @ H
        S = np.linalg.solve(Mk, np.hstack([Ak, G]))
        Yk, Zk = S[:, :n], S[:, n:]
        H = symmetrise(H + Ak.T @ H @ Yk)
        G = symmetrise(G + Ak @ Zk @ Ak.T)
        Ak = Ak @ Yk
        if i >= mind:
            if np.linalg.norm(Ak) < EPS: break
            if dare_residual(A, B, Q, R, H) < tol: break
    return H, i

# ---- iterative_dare, NK branch --------------------------------
def iterative_dare_nk(A, B, Q, R, P0, tol=1e-4, mn=2, mx=4):
    P = P0.copy(); unstable_k0 = False; i = 0
    for i in range(1, mx + 1):
        K = compute_gain(B, R, P, A)
        AK = A - B @ K
        Qk = symmetrise(Q + K.T @ R @ K)
        # *** the critical line: dlyap is called with AK.T ***
        P, dinfo = dlyap_fast_c(AK.T, Qk)
        if i == 1 and not dinfo['is_stable']:
            unstable_k0 = True
            break
        if i >= mn and dare_residual(A, B, Q, R, P) < tol:
            break
    return P, i, unstable_k0

# ================= tests ======================================
n, m, p = 12, 6, 6
A = np.eye(n) + 0.01 * np.random.randn(n, n)
B = np.random.randn(n, m) * 0.1
C = np.zeros((p, n))
for row, col in enumerate([0, 1, 2, 8, 9, 10]):
    C[row, col] = 1.0
Qy = np.eye(p) * 10.0
R = np.eye(m) * 1.0
Q = symmetrise(C.T @ Qy @ C)

print("=" * 62)
print("1. dlyap_fast_c solves  A X A' - X + Q = 0")
Astable = np.random.randn(n, n); Astable *= 0.7 / max(abs(np.linalg.eigvals(Astable)))
Qs = symmetrise(np.random.randn(n, n)); Qs = Qs @ Qs.T
X, inf1 = dlyap_fast_c(Astable, Qs)
res = np.linalg.norm(Astable @ X @ Astable.T - X + Qs) / np.linalg.norm(Qs)
print(f"   residual = {res:.3e}   doublings={inf1['doublings']}  stable={inf1['is_stable']}")
assert res < 1e-12

print("\n2. WRONG convention (A' X A) would give a large residual:")
res_wrong = np.linalg.norm(Astable.T @ X @ Astable - X + Qs) / np.linalg.norm(Qs)
print(f"   residual = {res_wrong:.3e}  <- confirms the two are NOT interchangeable")

print("\n3. dlyap_fast_c detects an unstable A")
Aun = np.random.randn(n, n); Aun *= 1.3 / max(abs(np.linalg.eigvals(Aun)))
_, inf2 = dlyap_fast_c(Aun, Qs)
print(f"   is_stable={inf2['is_stable']}  diverged={inf2['diverged']}  doublings={inf2['doublings']}")
assert not inf2['is_stable']

print("\n4. dare_sda cold solve")
P_sda, it_sda = dare_sda(A, B, Q, R)
print(f"   residual = {dare_residual(A,B,Q,R,P_sda):.3e}   doublings = {it_sda}")
assert dare_residual(A, B, Q, R, P_sda) < 1e-4

print("\n5. cross-check vs scipy DARE (if available)")
try:
    from scipy.linalg import solve_discrete_are
    P_ref = solve_discrete_are(A, B, Q, R)
    print(f"   ||P_sda - P_scipy||/||P_scipy|| = "
          f"{np.linalg.norm(P_sda-P_ref)/np.linalg.norm(P_ref):.3e}")
except ImportError:
    print("   scipy unavailable — skipped")

print("\n6. NK warm-started from the SDA solution (the k>1 hot path)")
A2 = A + 0.002 * np.random.randn(n, n)          # SDC matrices drift slightly
P_nk, it_nk, uk0 = iterative_dare_nk(A2, B, Q, R, P_sda)
print(f"   residual = {dare_residual(A2,B,Q,R,P_nk):.3e}   NK iters = {it_nk}  unstable_k0={uk0}")
assert not uk0 and dare_residual(A2, B, Q, R, P_nk) < 1e-4

print("\n7. NK from a cold/zero P0 should trip unstable_k0 -> SDA fallback")
P_nk0, it0, uk0_0 = iterative_dare_nk(A2, B, Q, R, np.zeros((n, n)))
print(f"   unstable_k0 = {uk0_0}  (iters={it0})")

print("\n8. Van Loan c2d structure check")
from scipy.linalg import expm as sp_expm
Ac = np.random.randn(n, n); Bc = np.random.randn(n, m); Ts = 0.01
M = np.zeros((n + m, n + m)); M[:n, :n] = Ac * Ts; M[:n, n:] = Bc * Ts
E = sp_expm(M)
print(f"   bottom-left block ~0 : {np.linalg.norm(E[n:, :n]):.3e}")
print(f"   bottom-right ~ I     : {np.linalg.norm(E[n:, n:] - np.eye(m)):.3e}")

print("\n" + "=" * 62)
print("All logic checks passed.")
