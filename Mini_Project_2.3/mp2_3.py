"""
Mobile Communications - Unit 2 - Mini Project 2.3
WPAN vs WLAN vs WMAN: range / throughput / power trade-off tool
Dataset -> derived metrics -> charts -> scenario evaluation
Source keys [n] refer to the numbered list in SOURCES (also reproduced in the report).
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
 1: "Electronic Design, '5 Things You Must Know About the New Bluetooth 5' - electronicdesign.com/technologies/communications/wireless/article/21802861/5-things-you-must-know-about-the-new-bluetooth-5",
 2: "BZZ course wiki, LU07b Bluetooth (BT5: ~40 m indoors / ~200 m outdoors at 1 Mbit/s) - wiki.bzz.ch/en/modul/m286/learningunits/lu07/bluetooth",
 3: "Android Authority, 'Bluetooth 5: speed, range and more' (BLE 5 2 Mbps PHY, ~1.4 Mbps throughput) - androidauthority.com/bluetooth-5-speed-range-762369/",
 4: "Intel Support, 'Different Wi-Fi Protocols and Data Rates' (802.11ax 2.4 Gbps, 802.11ac wave 2 1.73 Gbps, 866 Mbps 2x2 80 MHz) - intel.com/content/www/us/en/support/articles/000005725.html",
 5: "Aruba 500H series datasheet (HE80 2SS 1.2 Gbps; +21 dBm 5 GHz) - open-vision.ru/catalog/network-equipment/besprovodnyie-seti/aruba-500h-series/",
 6: "Intel Wi-Fi 6E AX210 technical specifications (5 GHz client 23 dBm EIRP max) - onlinehelp.ncrvoyix.com/Retail/SafetyCompliance/B005-0000-1589/Topics/RFIStatements/AX210/AX210TechSpec.htm",
 7: "Electronics Notes, Wi-Fi standards infographic (Wi-Fi 6: ~30 m indoor, ~120 m outdoor) - electronics-notes.com/articles/summary-infographics/wifi-80211/wifi-standards-infographic-v3.pdf",
 8: "L-com, '802.11 Wireless Standards' (11ac ~90 ft indoor, ~1000 ft outdoor; 11ax 50 ft / 900 ft at 5 GHz) - l-com.com/images/downloadables/Technical-Briefs-And-Articles/802-11-Wireless-Standards.pdf",
 9: "EDN, WiMAX power profile (BS ~+43 dBm / 20 W, MS ~+23 dBm) - edn.com/?p=4005852",
 10: "EE Times India, 'The File', 1 Dec 2008 (Wi-Fi AP/client typically +18 dBm; WiMAX BS +43 dBm) - nxtbook.com/nxtbooks/emedia/eetindia_20081201/index.php?startid=6",
 11: "RF Wireless World, 'Understanding IEEE 802.16 WiMAX Standards' (16e: up to 70 Mbps fixed, 3-5 km mobile, up to 10 km fixed; 16m: 1 Gbps stationary / 100 Mbps mobile, up to 10 km) - rfwireless-world.com/articles/understanding-ieee-802-16-wimax-standards",
 12: "arXiv:1605.00889, 'The Role of communication and network technologies in vehicular applications' (802.16e up to 70 Mbps, ~10 km; measured mean 2 and 5.3 Mbps) - arxiv.org/pdf/1605.00889",
 13: "itwissen.info, 'IEEE 802.16m' (1 Gbit/s nomadic, 100 Mbit/s at high mobility) - itwissen.info/en/IEEE-802-DOT-16m-119759.html",
 14: "FierceWireless, 'Sprint to shutter WiMAX network around Nov. 6, 2015' - fierce-network.com/wireless/sprint-to-shutter-wimax-network-around-nov-6-2015",
 15: "TP-Link Omada 5 GHz 867 Mbps wireless bridge (up to 5 km point-to-point) - omadanetworks.com/ca/business-networking/omada-wifi-wireless-bridge/",
}

# ---------------------------------------------------------------- dataset
data = [
 dict(tech="BLE 5 (802.15.1)", cls="WPAN", band_GHz=2.4, access="FHSS, GFSK",
      range_low_m=40,   range_high_m=200,   range_note="40 m indoor / 200 m outdoor LOS (1M PHY)",
      rate_typ_Mbps=1.4, rate_peak_Mbps=2,     rate_note="1.4 Mbps app throughput / 2 Mbps PHY",
      tx_infra_dBm=0,  tx_client_dBm=0,  power_class="Low",
      src="[1][2][3]"),
 dict(tech="Wi-Fi 5 (802.11ac)", cls="WLAN", band_GHz=5.5, access="OFDM, CSMA/CA",
      range_low_m=27,   range_high_m=300,   range_note="90 ft indoor / 1000 ft outdoor",
      rate_typ_Mbps=867, rate_peak_Mbps=1730,  rate_note="2x2 80 MHz / 2x2 160 MHz",
      tx_infra_dBm=18, tx_client_dBm=18, power_class="Medium",
      src="[4][8][10]"),
 dict(tech="Wi-Fi 6 (802.11ax)", cls="WLAN", band_GHz=5.5, access="OFDMA, CSMA/CA",
      range_low_m=30,   range_high_m=120,   range_note="30 m indoor / 120 m outdoor",
      rate_typ_Mbps=1201, rate_peak_Mbps=2402, rate_note="2SS HE80 / 2x2 160 MHz",
      tx_infra_dBm=21, tx_client_dBm=23, power_class="Medium",
      src="[4][5][6][7]"),
 dict(tech="WiMAX 802.16e", cls="WMAN", band_GHz=2.5, access="Scalable OFDMA",
      range_low_m=3000, range_high_m=10000, range_note="3-5 km mobile / up to 10 km fixed",
      rate_typ_Mbps=5.3, rate_peak_Mbps=70,    rate_note="5.3 Mbps measured / 70 Mbps fixed peak",
      tx_infra_dBm=43, tx_client_dBm=23, power_class="High",
      src="[9][11][12]"),
 dict(tech="WiMAX 802.16m", cls="WMAN", band_GHz=2.5, access="Scalable OFDMA",
      range_low_m=3000, range_high_m=10000, range_note="up to 10 km (low end assumed = 16e)",
      rate_typ_Mbps=100, rate_peak_Mbps=1000,  rate_note="100 Mbps high-mobility / 1 Gbps stationary",
      tx_infra_dBm=43, tx_client_dBm=23, power_class="High",
      src="[11][13] (power assumed = 16e)"),
]
df = pd.DataFrame(data)

# ---------------------------------------------------------------- derived metrics
def fspl_dB(d_m, f_GHz):                      # Friis, as in Mini Project 2.1
    return 20*np.log10(d_m/1000) + 20*np.log10(f_GHz*1000) + 32.44
df["tx_infra_mW"]   = 10 ** (df.tx_infra_dBm / 10)
df["rate_per_m"]    = df.rate_peak_Mbps / df.range_low_m              # Mbps per metre of range
df["rate_x_range"]  = df.rate_peak_Mbps * df.range_low_m / 1000       # capacity-distance product, Mbps*km
df["fspl_at_range"] = fspl_dB(df.range_low_m, df.band_GHz)            # dB at typical range
df["fspl_at_max"]   = fspl_dB(df.range_high_m, df.band_GHz)
df.to_csv(os.path.join(OUT, "dataset.csv"), index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("=== Dataset ==="); print(df[["tech","cls","band_GHz","range_low_m","range_high_m","rate_typ_Mbps","rate_peak_Mbps","tx_infra_dBm","tx_client_dBm","power_class"]].to_string(index=False))
print("\n=== Derived metrics ==="); print(df[["tech","tx_infra_mW","rate_per_m","rate_x_range","fspl_at_range","fspl_at_max"]].round(2).to_string(index=False))

colors = {"WPAN": "#2ca02c", "WLAN": "#1f77b4", "WMAN": "#d62728"}
off = {"BLE 5 (802.15.1)": (10, -14), "Wi-Fi 5 (802.11ac)": (-10, -26), "Wi-Fi 6 (802.11ax)": (10, 6),
       "WiMAX 802.16e": (24, -16), "WiMAX 802.16m": (24, 6)}

# ---------------------------------------------------------------- Fig 1: trade-off space
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for _, r in df.iterrows():
    ax.hlines(r.rate_peak_Mbps, r.range_low_m, r.range_high_m, color=colors[r.cls], alpha=.35, lw=2)
    ax.scatter(r.range_low_m, r.rate_peak_Mbps, s=90 + 20*r.tx_infra_dBm, color=colors[r.cls],
               edgecolor="k", zorder=3, alpha=.9)
    ax.annotate(f"{r.tech}\n{r.tx_infra_dBm:.0f} dBm", (r.range_low_m, r.rate_peak_Mbps), xytext=off[r.tech],
                textcoords="offset points", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(8, 4e4); ax.set_ylim(0.8, 6000)
ax.set_xlabel("Range (m, log): dot = typical, line extends to max/outdoor")
ax.set_ylabel("Peak data rate (Mbps, log)")
ax.set_title("Range - throughput - power trade-off (marker size = infrastructure Tx power)", fontsize=9.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, mec="k", ms=8, label=k) for k, c in colors.items()],
          loc="lower right", fontsize=8)
ax.grid(True, which="both", alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_tradeoff.png"), dpi=200); plt.close()

# ---------------------------------------------------------------- scenarios
scen = {
 "S1 Smart home": dict(range_m=15,   rate_Mbps=0.5,  basis="typ",  rbasis="low",   max_tx_dBm=10, note="50 coin-cell sensors, ~10 kbps each (0.5 Mbps aggregate), 15 m radius, >=2 yr battery"),
 "S2 Campus":     dict(range_m=25,   rate_Mbps=300,  basis="typ",  rbasis="low", max_tx_dBm=99, note="~30 users/AP x 10 Mbps = 300 Mbps per cell, 25 m cells, mains powered"),
 "S3 Metro FWA":  dict(range_m=6000, rate_Mbps=150,  basis="peak", rbasis="high", max_tx_dBm=99, note="300 fixed homes, 5 Mbps each, 10:1 contention = 150 Mbps per site, 6 km radius"),
}
print("\n=== Scenario feasibility (requirement vs dataset) ===")
rows = []
for sname, s in scen.items():
    for _, r in df.iterrows():
        rate = r.rate_peak_Mbps if s["basis"] == "peak" else r.rate_typ_Mbps
        ok_range = (r.range_high_m if s["rbasis"] == "high" else r.range_low_m) >= s["range_m"]
        ok_rate  = rate >= s["rate_Mbps"]
        ok_pow   = r.tx_client_dBm <= s["max_tx_dBm"]
        rows.append(dict(scenario=sname, tech=r.tech, range_ok=ok_range, rate_ok=ok_rate, power_ok=ok_pow,
                         all_ok=ok_range and ok_rate and ok_pow, rate_used=rate,
                         sectors_for_rate=math.ceil(s["rate_Mbps"]/rate) if not ok_rate else 1))
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(os.path.join(OUT, "scenario_results.csv"), index=False)

# coverage maths for S2 (campus) using the dataset's Wi-Fi 6 numbers
area = 200_000   # m^2 campus (0.2 km^2)
cell = math.pi * 25**2
print(f"\nS2: campus {area} m2 / one indoor cell ({cell:.0f} m2 at 25 m radius) = {area/cell:.0f} APs (ideal, no overlap)")
print(f"S2: aggregate demand 1500 users x 10 Mbps = 15000 Mbps -> {15000/1201:.1f} APs (Wi-Fi 6, 1201 Mbps typ.) vs {15000/867:.1f} APs (Wi-Fi 5, 867 Mbps typ.) as capacity floor")
print(f"S2: WiMAX 16e at 70 Mbps peak would need {15000/70:.0f} base stations for capacity alone")
print(f"S3: 150 Mbps / 70 Mbps (16e fixed peak) -> {math.ceil(150/70)} sectors; 16m 1000 Mbps -> {math.ceil(150/1000)} sector")
print(f"S1: BLE 0 dBm vs Wi-Fi client 18-23 dBm = 18-23 dB = {10**1.8:.0f}-{10**2.3:.0f}x power difference")

# ---------------------------------------------------------------- Fig 2: scenarios on the trade-off space
lab2 = {"BLE 5 (802.15.1)": (8, 4, "left"), "Wi-Fi 5 (802.11ac)": (-8, 2, "right"), "Wi-Fi 6 (802.11ax)": (8, 4, "left"),
        "WiMAX 802.16e": (-10, -4, "right"), "WiMAX 802.16m": (-10, 2, "right")}
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for _, r in df.iterrows():
    ax.hlines(r.rate_peak_Mbps, r.range_low_m, r.range_high_m, color=colors[r.cls], lw=1.5, alpha=.5)
    ax.vlines(r.range_low_m, r.rate_typ_Mbps, r.rate_peak_Mbps, color=colors[r.cls], lw=5, alpha=.5)
    ax.scatter(r.range_low_m, r.rate_typ_Mbps, s=40, color=colors[r.cls], edgecolor="k", zorder=3)
    dx, dy, ha = lab2[r.tech]
    ax.annotate(r.tech, (r.range_low_m, r.rate_peak_Mbps), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, ha=ha)
slab = {"S1 Smart home": (10, -4, "left"), "S2 Campus": (-10, -4, "right"), "S3 Metro FWA": (10, 8, "left")}
for sname, s_ in scen.items():
    ax.scatter(s_["range_m"], s_["rate_Mbps"], marker="*", s=260, color="gold", edgecolor="k", zorder=5)
    dx, dy, ha = slab[sname]
    ax.annotate(sname, (s_["range_m"], s_["rate_Mbps"]), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, weight="bold", ha=ha)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5, 4e4); ax.set_ylim(0.08, 6000)
ax.set_xlabel("Range (m, log): dot = typical, thin line extends to max range")
ax.set_ylabel("Data rate (Mbps, log): bar = typical to peak")
ax.set_title("Scenario requirements (stars) against technology capability", fontsize=10)
ax.grid(True, which="both", alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_scenarios.png"), dpi=200); plt.close()

# ---------------------------------------------------------------- Fig 3: power and path loss
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
c = [colors[k] for k in df.cls]; lab = ["BLE 5", "Wi-Fi 5", "Wi-Fi 6", "WiMAX\n16e", "WiMAX\n16m"]
ax[0].bar(lab, df.tx_infra_dBm, color=c, edgecolor="k"); ax[0].set_ylabel("Infrastructure Tx power (dBm)")
for i, v in enumerate(df.tx_infra_dBm): ax[0].text(i, v + .8, f"{v:.0f}", ha="center", fontsize=8)
ax[1].bar(lab, df.fspl_at_range, color=c, edgecolor="k"); ax[1].set_ylabel("Free-space path loss at typical range (dB)")
for i, v in enumerate(df.fspl_at_range): ax[1].text(i, v + 1, f"{v:.0f}", ha="center", fontsize=8)
for a in ax: a.tick_params(axis="x", labelsize=8); a.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_power_pathloss.png"), dpi=200); plt.close()

with open(os.path.join(OUT, "sources.txt"), "w") as f:
    for k, v in SOURCES.items(): f.write(f"[{k}] {v}\n")
print("\nFiles saved in", OUT)