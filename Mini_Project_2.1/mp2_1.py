"""
Mobile Communications - Unit 2 - Mini Project 2.1
Free-Space Path Loss (Friis) and Digital Modulation BER over AWGN
Part (a): Friis path loss vs distance, 6 dB-per-doubling check
Part (b): Monte Carlo BER for BPSK, QPSK, 16-QAM (+64-QAM extension) and Shannon-Hartley link
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

np.random.seed(0)
OUT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ Part (a)
def friis_rx_power(Ptx_dBm, Gtx_dBi, Grx_dBi, d_km, f_MHz):
    """Received power in dBm (d in km, f in MHz)."""
    return (Ptx_dBm + Gtx_dBi + Grx_dBi
            - 20 * np.log10(d_km) - 20 * np.log10(f_MHz) - 32.44)

Ptx, Gtx, Grx = 43, 15, 0
d = np.linspace(0.05, 10, 500)
freqs = [900, 1900, 3500]

plt.figure(figsize=(7, 4.4))
for f in freqs:
    plt.semilogx(d, friis_rx_power(Ptx, Gtx, Grx, d, f), label=f"{f} MHz")
plt.xlabel("Distance (km, log scale)"); plt.ylabel("Received power (dBm)")
plt.title("Friis free-space received power (PTX=43 dBm, GTX=15 dBi, GRX=0 dBi)", fontsize=9)
plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_friis.png"), dpi=200); plt.close()

# 6 dB-per-doubling verification
print("=== Friis 6 dB-per-doubling check ===")
f0, d0 = 900, 1.0
p1 = friis_rx_power(Ptx, Gtx, Grx, d0, f0)
p2 = friis_rx_power(Ptx, Gtx, Grx, 2 * d0, f0)
print(f"Distance: P_RX(1 km)={p1:.2f} dBm, P_RX(2 km)={p2:.2f} dBm, drop = {p1-p2:.3f} dB")
p3 = friis_rx_power(Ptx, Gtx, Grx, d0, 2 * f0)
print(f"Frequency: P_RX(900 MHz)={p1:.2f} dBm, P_RX(1800 MHz)={p3:.2f} dBm, drop = {p1-p3:.3f} dB")
print(f"Theory: 20*log10(2) = {20*np.log10(2):.3f} dB")
for f in freqs:
    print(f"  f={f} MHz: P_RX(0.5 km)={friis_rx_power(Ptx,Gtx,Grx,0.5,f):.2f} dBm, "
          f"P_RX(5 km)={friis_rx_power(Ptx,Gtx,Grx,5,f):.2f} dBm")

# ------------------------------------------------------------------ Part (b)
def gray_decode(g):
    """Gray code -> binary index (vectorised)."""
    b = g.copy()
    shift = g >> 1
    while shift.any():
        b ^= shift
        shift >>= 1
    return b

def gray_encode(b):
    return b ^ (b >> 1)

def qam_mod(bits, M):
    """Gray-coded square M-QAM (M=4,16,64) normalised to unit average symbol energy."""
    k = int(np.log2(M)); kh = k // 2; L = 2 ** kh
    b = bits.reshape(-1, k)
    w = 2 ** np.arange(kh - 1, -1, -1)
    gi, gq = b[:, :kh] @ w, b[:, kh:] @ w
    lvl = lambda g: 2 * gray_decode(g) - (L - 1)       # e.g. -3,-1,+1,+3 for 16-QAM
    norm = np.sqrt(2 * (M - 1) / 3)                    # average energy of un-normalised grid
    return (lvl(gi) + 1j * lvl(gq)) / norm

def qam_demod(sym, M):
    k = int(np.log2(M)); kh = k // 2; L = 2 ** kh
    norm = np.sqrt(2 * (M - 1) / 3)
    def axis(x):
        idx = np.clip(np.round((x * norm + (L - 1)) / 2), 0, L - 1).astype(int)  # min-distance
        g = gray_encode(idx)
        return ((g[:, None] >> np.arange(kh - 1, -1, -1)) & 1)
    return np.hstack([axis(sym.real), axis(sym.imag)]).ravel()

def bpsk_mod(bits):   return (2 * bits - 1).astype(complex)      # 0 -> -1, 1 -> +1
def bpsk_demod(sym):  return (sym.real > 0).astype(int)

def awgn(sym, ebn0_dB, bits_per_sym):
    """Unit-energy symbols: Es=1, N0 = Es / (k * Eb/N0). Complex noise, variance N0/2 per dimension."""
    N0 = 1.0 / (bits_per_sym * 10 ** (ebn0_dB / 10))
    n = np.sqrt(N0 / 2) * (np.random.randn(sym.size) + 1j * np.random.randn(sym.size))
    return sym + n

N = 1_800_000                       # divisible by 1,2,4,6 -> works for all schemes
bits = np.random.randint(0, 2, size=N)
schemes = {
    "BPSK":   (1, bpsk_mod, bpsk_demod),
    "QPSK":   (2, lambda b: qam_mod(b, 4),  lambda s: qam_demod(s, 4)),
    "16-QAM": (4, lambda b: qam_mod(b, 16), lambda s: qam_demod(s, 16)),
    "64-QAM": (6, lambda b: qam_mod(b, 64), lambda s: qam_demod(s, 64)),
}
ebn0_dB = np.arange(0, 17, 1.0)

# mapper sanity checks
c16 = qam_mod(np.array([int(x) for i in range(16) for x in np.binary_repr(i, 4)]), 16)
assert abs(np.mean(np.abs(c16) ** 2) - 1) < 1e-12, "16-QAM not unit energy"
assert np.array_equal(qam_demod(qam_mod(bits[:600], 16), 16), bits[:600]), "16-QAM noiseless loop failed"
assert np.array_equal(qam_demod(qam_mod(bits[:600], 64), 64), bits[:600]), "64-QAM noiseless loop failed"

ber = {}
for name, (k, mod, dem) in schemes.items():
    tx = mod(bits)
    ber[name] = np.array([np.mean(dem(awgn(tx, e, k)) != bits) for e in ebn0_dB])

print("\n=== Empirical BER (N = %d bits) ===" % N)
print("Eb/N0 | " + " | ".join(f"{n:>9}" for n in schemes))
for i, e in enumerate(ebn0_dB):
    print(f"{e:5.0f} | " + " | ".join(f"{ber[n][i]:9.2e}" for n in schemes))

# theory
e_lin = 10 ** (ebn0_dB / 10)
th_bpsk = 0.5 * erfc(np.sqrt(e_lin))
th_16 = (3 / 8) * erfc(np.sqrt(0.4 * e_lin))          # standard Gray 16-QAM approximation
print("\nMax |log10(sim/theory)| for BPSK where BER>1e-5:",
      f"{np.max(np.abs(np.log10(ber['BPSK'][th_bpsk>1e-5]/th_bpsk[th_bpsk>1e-5]))):.3f}")

# Figure 2 - BER curves
plt.figure(figsize=(7, 4.8))
sty = {"BPSK": "o-", "QPSK": "x--", "16-QAM": "s-", "64-QAM": "^-"}
for n in schemes:
    m = ber[n] > 0
    plt.semilogy(ebn0_dB[m], ber[n][m], sty[n], ms=4, label=f"{n} (simulated)")
plt.semilogy(ebn0_dB, th_bpsk, "k:", lw=1.3, label="BPSK theory 0.5·erfc(√(Eb/N0))")
plt.semilogy(ebn0_dB, th_16, color="gray", ls=":", lw=1.3, label="16-QAM theory (approx.)")
plt.ylim(1e-6, 1); plt.xlim(0, 16)
plt.xlabel("Eb/N0 (dB)"); plt.ylabel("Bit error rate")
plt.title("BER vs Eb/N0 over AWGN"); plt.grid(True, which="both", alpha=.3)
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_ber.png"), dpi=200); plt.close()

# Gap at target BER (log-domain interpolation)
def ebn0_at(ber_curve, target):
    m = ber_curve > 0
    return float(np.interp(np.log10(target), np.log10(ber_curve[m])[::-1], ebn0_dB[m][::-1]))

target = 1e-3
need = {n: ebn0_at(ber[n], target) for n in schemes}
print(f"\n=== Eb/N0 required for BER = {target:g} ===")
for n, (k, _, _) in schemes.items():
    es = need[n] + 10 * np.log10(k)
    shannon = 10 * np.log10((2 ** k - 1) / k)
    print(f"{n:7s}: Eb/N0 = {need[n]:5.2f} dB | Es/N0 = {es:5.2f} dB | "
          f"Shannon min Eb/N0 @ {k} b/s/Hz = {shannon:5.2f} dB | gap to Shannon = {need[n]-shannon:5.2f} dB")
g_eb = need["16-QAM"] - need["BPSK"]
g_es = (need["16-QAM"] + 10*np.log10(4)) - need["BPSK"]
print(f"\n16-QAM minus BPSK: Eb/N0 gap = {g_eb:.2f} dB ; Es/N0 (SNR per symbol) gap = {g_es:.2f} dB")
print(f"Shannon-limit gap (Eb/N0) between 4 and 1 b/s/Hz = "
      f"{10*np.log10(15/4):.2f} dB")

# Figure 3 - Shannon bound vs operating points
eta = np.logspace(-1, 1.1, 300)
plt.figure(figsize=(7, 4.4))
plt.semilogy(10 * np.log10((2 ** eta - 1) / eta), eta, "k-", label="Shannon limit  η = log2(1+η·Eb/N0)")
for n, (k, _, _) in schemes.items():
    plt.semilogy(need[n], k, "o", ms=7, label=f"{n} @ BER=1e-3")
plt.xlabel("Eb/N0 (dB)"); plt.ylabel("Spectral efficiency η (bit/s/Hz)")
plt.xlim(-2, 16); plt.ylim(0.5, 12); plt.grid(True, which="both", alpha=.3)
plt.title("Uncoded modulation vs the Shannon limit"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_shannon.png"), dpi=200); plt.close()

# Task 6 proxy: throughput = bits/symbol * (1 - BER) vs Shannon capacity per Hz
print("\n=== Throughput proxy k*(1-BER) [bit/symbol] at selected Eb/N0 ===")
for i in (4, 8, 12, 16):
    print(f"Eb/N0={ebn0_dB[i]:4.0f} dB: " + ", ".join(
        f"{n}={k*(1-ber[n][i]):.3f}" for n, (k, _, _) in schemes.items()))
print("\nFigures saved in", OUT)