Documentation about devices involved

# PILAR V — Microphone AFE (Analog Front-End) Notes

## 1. Confirmed Mic Hardware (from multimeter reverse-engineering)

**Connector:** Souriau 851-series, 4-pin MIL-DTL-26482

**Topology confirmed:** JFET source-follower

|Pin| Function |
|---|----------|
| A | Chassis GND |
| B | Capsule backplate / JFET gate |
| C | Audio output (via 33Ω source resistor) |
| D | Bias supply (via 3kΩ drain resistor) |

*Note: the raw multimeter voltage/resistance readings that led to this conclusion were not carried forward in memory — only the final resistor values and topology. If you have the original readings, they can be reconciled against the JFET bias point below.*

---

## 2. Generic Assumed 4-Pin Structure (initial working hypothesis, before confirmation above)

Single-ended version (most likely for source-follower):

| Pin | Function |
|---|---|
| 1 | V+ / Bias (drain supply) |
| 2 | Output (source follower, via coupling cap) |
| 3 | GND (source resistor return / shield) |
| 4 | Shield / Case ground |

(Superseded by the confirmed A/B/C/D mapping above — kept here for traceability.)

---

## 3. Generic JFET Source-Follower Background

- Electret capsules have an internal JFET
- Gate ← diaphragm/backplate variable capacitor (high-Z)
- Drain ← V+ bias supply
- Source ← source resistor to GND; output taken here (voltage gain ≈ 1, impedance conversion only)
- Multimeter confirmation methods: resistance/continuity to ground, diode-mode gate-channel junction test, DC voltage check under bias

---

## 4. Signal Conditioning Chain: Mic → ADC (generic values)

Target: DE10-Standard onboard ADC (ADC128S022 assumed, 12-bit, SPI, 0–3.3V), 16kHz sample rate, ~7kHz LPF cutoff (scream/gunshot detection bandwidth).

**Stage 1 — Mic Bias**
- Rbias = 2.2kΩ, 3.3V → Pin1 (V+/Drain)
- ~0.5mA typical electret bias current

**Stage 2 — DC Blocking / Coupling**
- C1 = 1µF
- Corner freq ≈ 16Hz (with ~10kΩ downstream impedance) — well below signal band

**Stage 3 — Mid-Rail Bias (single-supply op-amp)**
- R1 = R2 = 10kΩ → bias point = 1.65V
- Cbypass = 0.1µF at midpoint

**Stage 4 — Gain Stage (non-inverting)**
- Op-amp: MCP6002 (rail-to-rail, single-supply)
- Rf = 47kΩ, Rg = 4.7kΩ → Av ≈ 11x
- Rg = 2.2kΩ option → Av ≈ 22x (watch clipping on close-range gunshot transients)

**Stage 5 — Anti-Aliasing LPF (Sallen-Key, 2nd order, Butterworth Q=0.707)**
- R = 2.2kΩ (both), C = 10nF → fc ≈ 7kHz

**Stage 6 — Protection Before ADC**
- Series R = 100Ω
- Clamp diodes: BAT54S dual Schottky → 3.3V rail and GND
- Optional 100pF–1nF cap at ADC pin

**Full chain:**
```
Mic Pin2 → C1(1µF) → [R1=R2=10k bias] → MCP6002 gain (Av≈11x)
  → Sallen-Key LPF (R=2.2k, C=10nF, fc≈7kHz) → 100Ω + BAT54S clamp → ADC128S022
```

---

## 5. Open Items / Next Steps Flagged

- [ ] Reconcile confirmed pinout (A/B/C/D, 33Ω source / 3kΩ drain) with the generic component values above — bias resistor and gain stage should be recalculated using the **actual** 33Ω source / 3kΩ drain values rather than the generic 2.2kΩ assumption
- [ ] Verify actual ADC128S022 revision on your specific DE10-Standard board
- [ ] SPI timing for Nios-V integration with the ADC
- [ ] Gain headroom analysis for high-SPL gunshot transients (130–160dB) to avoid clipping
- [ ] Recover/re-derive original multimeter readings if available, to cross-check JFET operating point
