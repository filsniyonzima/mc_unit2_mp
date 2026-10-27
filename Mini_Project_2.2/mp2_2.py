"""
Mobile Communications - Unit 2 - Mini Project 2.2
OFDM waveform generator with a cyclic-prefix (CP) visualizer
Chain: bits -> Gray 16-QAM -> N=64 subcarriers -> IFFT -> add CP -> multipath FIR -> (AWGN) -> remove CP -> FFT -> 1-tap equaliser -> detect
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------- 16-QAM mapper (Gray, unit energy)
def gray_decode(g):
    b = g.copy(); s = g >> 1
    while s.any():
        b ^= s; s >>= 1
    return b
gray_encode = lambda b: b ^ (b >> 1)
NORM = np.sqrt(10.0)          # 16-QAM: levels -3,-1,1,3 -> average energy 10

def qam16_mod(bits):
    b = bits.reshape(-1, 4)
    gi = b[:, 0] * 2 + b[:, 1]; gq = b[:, 2] * 2 + b[:, 3]
    lvl = lambda g: 2 * gray_decode(g) - 3
    return (lvl(gi) + 1j * lvl(gq)) / NORM

def qam16_demod(sym):
    def ax(x):
        g = gray_encode(np.clip(np.round((x * NORM + 3) / 2), 0, 3).astype(int))
        return np.stack([(g >> 1) & 1, g & 1], axis=1)
    return np.hstack([ax(sym.real), ax(sym.imag)]).ravel()

# ----------------------------------------------------------- OFDM blocks
N = 64
def ofdm_mod(X):                      # X: (nsym, N) frequency domain -> time domain, unit power
    return np.fft.ifft(X, axis=1) * np.sqrt(N)
def ofdm_demod(x):
    return np.fft.fft(x, axis=1) / np.sqrt(N)
def add_cp(x, ncp):
    return x if ncp == 0 else np.hstack([x[:, -ncp:], x])
def remove_cp(y, ncp):
    return y[:, ncp:]

# ----------------------------------------------------------- Multipath channel (FIR, 3 taps)
delays = np.array([0, 3, 7])                                   # in samples
gains = np.array([0.85, 0.45 * np.exp(1j * 0.9), 0.25 * np.exp(-1j * 1.6)])
gains = gains / np.sqrt(np.sum(np.abs(gains) ** 2))
h = np.zeros(delays.max() + 1, complex); h[delays] = gains
L_SPREAD = delays.max()                                        # max excess delay = 7 samples
H = np.fft.fft(h, N)

def run_link(X, tx_bits_sym, ncp, snr_dB=None):
    """Transmit all OFDM symbols back-to-back through the channel; return equalised X_hat."""
    nsym = X.shape[0]
    tx = add_cp(ofdm_mod(X), ncp).ravel()                      # serial stream
    rx = np.convolve(tx, h)[:tx.size]                          # symbols smear into each other here
    if snr_dB is not None:
        pw = np.mean(np.abs(rx) ** 2)
        nv = pw / 10 ** (snr_dB / 10)
        rx = rx + np.sqrt(nv / 2) * (np.random.randn(rx.size) + 1j * np.random.randn(rx.size))
    y = remove_cp(rx.reshape(nsym, N + ncp), ncp)
    return ofdm_demod(y) / H                                   # one-tap zero-forcing equaliser

def metrics(Xhat, X, bits):
    sl = slice(1, None)                                        # skip 1st symbol (no predecessor)
    bhat = qam16_demod(Xhat[sl].ravel()).reshape(-1, 4)
    btx = bits.reshape(X.shape[0], N, 4)[sl].reshape(-1, 4)
    ser = np.mean(np.any(bhat != btx, axis=1))
    mse = np.mean(np.abs(Xhat[sl] - X[sl]) ** 2)
    return ser, mse

# ----------------------------------------------------------- data
NSYM = 400
bits = np.random.randint(0, 2, size=NSYM * N * 4)
X = qam16_mod(bits).reshape(NSYM, N)

# ----------------------------------------------------------- self-checks
print("=== Self-checks ===")
assert np.array_equal(qam16_demod(qam16_mod(bits[:4000])), bits[:4000]); print("1. Mapper round-trip (noiseless): PASS")
assert abs(np.mean(np.abs(X) ** 2) - 1) < 0.02; print(f"   Mean symbol energy = {np.mean(np.abs(X)**2):.3f}")
xt = ofdm_mod(X[:5])
err = np.max(np.abs(ofdm_demod(xt) - X[:5])); assert err < 1e-12
print(f"2. IFFT->FFT recovery of subcarrier values: max error = {err:.2e}  PASS")
xc = add_cp(xt, 16)
assert np.allclose(xc[:, :16], xt[:, -16:]) and np.allclose(xc[:, 16:], xt)
print("3. CP (16 samples) equals the last 16 samples of the OFDM symbol: PASS")
print(f"   Channel taps (delay, gain): " + ", ".join(f"({d}, {abs(g):.3f}∠{np.degrees(np.angle(g)):.0f}°)" for d, g in zip(delays, gains)))
print(f"   Max excess delay = {L_SPREAD} samples (so CP >= {L_SPREAD} should remove ISI); min|H| = {np.abs(H).min():.3f}, max|H| = {np.abs(H).max():.3f}")

# ----------------------------------------------------------- core comparison (noiseless)
print("\n=== Noiseless, same channel, 16-QAM ===")
cases = {"CP = 16 (sufficient)": 16, "CP = 4 (too short)": 4, "CP = 0 (none)": 0}
res = {}
for name, ncp in cases.items():
    Xh = run_link(X, bits, ncp)
    ser, mse = metrics(Xh, X, bits)
    res[name] = (Xh, ser, mse)
    print(f"{name:22s}: SER = {ser:.4f}   MSE = {mse:.3e}")

# ----------------------------------------------------------- Fig 1: constellations
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
ideal = np.unique(qam16_mod(np.array([int(c) for i in range(16) for c in np.binary_repr(i, 4)])))
for a, (name, (Xh, ser, mse)) in zip(ax, res.items()):
    pts = Xh[1:6].ravel() if False else Xh[1:].ravel()
    a.scatter(pts.real, pts.imag, s=2, alpha=.35, color="#1f77b4")
    a.scatter(ideal.real, ideal.imag, s=18, color="red", marker="+", lw=1)
    a.set_title(f"{name}\nSER={ser:.3f}, MSE={mse:.1e}", fontsize=9)
    a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.set_aspect("equal"); a.grid(alpha=.3)
    a.set_xlabel("In-phase"); 
ax[0].set_ylabel("Quadrature")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_constellations.png"), dpi=200); plt.close()

# ----------------------------------------------------------- Sweep CP length
cps = np.arange(0, 21)
sweep = {"noiseless": [], "snr30": []}
np.random.seed(1)
for c in cps:
    sweep["noiseless"].append(metrics(run_link(X, bits, c), X, bits))
    sweep["snr30"].append(metrics(run_link(X, bits, c, snr_dB=30), X, bits))
print("\n=== CP sweep (16-QAM, channel max delay = 7 samples) ===")
print("CP | SER(noiseless) | MSE(noiseless) | SER(30 dB) | MSE(30 dB) | overhead N/(N+CP)")
for c, a, b in zip(cps, sweep["noiseless"], sweep["snr30"]):
    print(f"{c:2d} | {a[0]:14.4f} | {a[1]:14.3e} | {b[0]:10.4f} | {b[1]:10.3e} | {N/(N+c):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for key, lab, st in (("noiseless", "noiseless", "o-"), ("snr30", "SNR = 30 dB", "s--")):
    ax[0].semilogy(cps, [max(v[0], 1e-5) for v in sweep[key]], st, ms=4, label=lab)
    ax[1].semilogy(cps, [max(v[1], 1e-12) for v in sweep[key]], st, ms=4, label=lab)
for a, yl in zip(ax, ("Symbol error rate (floored at 1e-5)", "Constellation MSE (floored at 1e-12)")):
    a.axvline(L_SPREAD, color="k", ls=":", label=f"delay spread = {L_SPREAD}")
    a.set_xticks(range(0, 21, 2)); a.set_xlabel("Cyclic prefix length (samples)"); a.set_ylabel(yl); a.grid(True, which="both", alpha=.3); a.legend(fontsize=8)
ax[0].set_title("SER vs CP length", fontsize=10); ax[1].set_title("MSE vs CP length", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_cp_sweep.png"), dpi=200); plt.close()

# ----------------------------------------------------------- Fig 3: time-domain CP visualiser
fig, ax = plt.subplots(2, 1, figsize=(10, 4.6))
for a, ncp, ttl in zip(ax, (16, 0), ("CP = 16: ISI lands inside the discarded prefix", "CP = 0: ISI lands inside the FFT window")):
    nsym = 3
    tx = add_cp(ofdm_mod(X[:nsym]), ncp).ravel()
    rx = np.convolve(tx, h)[:tx.size]
    blk = N + ncp
    a.plot(np.abs(rx), color="#1f77b4", lw=.9)
    for s in range(nsym):
        b0 = s * blk
        if ncp: a.axvspan(b0, b0 + ncp, color="#9ecae1", alpha=.5, label="cyclic prefix (discarded)" if s == 0 else None)
        if s > 0: a.axvspan(b0, b0 + L_SPREAD, color="red", alpha=.35, label="ISI from previous symbol" if s == 1 else None)
        a.axvline(b0, color="k", lw=.7, ls="--")
    a.set_title(ttl, fontsize=10); a.set_ylabel("|received|"); a.set_xlim(0, nsym * blk)
    a.legend(fontsize=8, loc="upper right")
ax[1].set_xlabel("Sample index (dashed lines = symbol boundaries)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_cp_visualiser.png"), dpi=200); plt.close()
print("\nFigures saved in", OUT)