"""
CIM ADC energy-accuracy behavioral simulator, v5 (pure NumPy).

What is new compared with v1:
  1. ADC full-scale (FS) is optimized per layer AND per ADC resolution
     (minimum quantization MSE on calibration data) instead of one fixed percentile.
  2. Adaptive per-layer ADC resolution is run for several tolerances -> a Pareto front,
     and compared with the cheapest uniform design of equal-or-better accuracy.
  3. Results are averaged over several independently trained networks (mean +- std).
  4. Larger analog-noise levels (noise is relative to the nominal full-scale).
  5. Fixed figures (2-bit energy bar visible, array-size figure with energy).
  8. NEW in v5: explicit device / input mapping
       --in-bits d     bits of the activation applied per cycle (8 = one analog step, 1 = bit-serial)
       --cell-bits c   bits of the weight magnitude stored per device (8 = one analog device,
                       2 = e.g. 2-bit RRAM cells, 1 = binary SRAM-like cells)
       One ADC conversion is needed per (tile, column, input cycle, weight slice), the digital
       side shift-and-adds them.  The ADC full scale is calibrated per layer, cycle and slice.
       --quick         skips the noise levels and the tile-size study (much faster)
       Example (bit-serial inputs, 2-bit cells):  --in-bits 1 --cell-bits 2 --quick
       Defaults (8, 8) reproduce the idealized mapping of v1-v4.
  7. (v4, answers the reviewer comments):
       - exhaustive enumeration of ALL per-layer resolution combinations (7^L) -> true Pareto
         front and the optimality gap of the greedy search (exp6, fig6)
       - per-layer sensitivity curves (exp7, fig7)
       - sample standard deviation (ddof=1), 5 networks by default, and paired McNemar tests
         of adaptive vs. uniform designs on the test set (exp8)
       - Fashion-MNIST as a second workload:  --dataset fmnist
  6. (v3): sensitivity analysis to FOM_ADC and E_MAC (exp5_sensitivity.csv, fig5),
     legend-based labels in fig4, and raw_results.pkl so that plots / sensitivity can be
     regenerated without re-simulating (set LOAD_RAW = True, or pass --load).

Usage (Spyder: just press Run; or from a terminal):
  python cim_adc_sim_v2.py --dataset mnist --seeds 3
  python cim_adc_sim_v2.py --dataset digits --hidden 64 32 --seeds 2   # tiny offline test
"""
import argparse, csv, os, time, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Assumed hardware parameters (justify them in the paper) ----
FOM_ADC = 30e-15      # J per conversion-step (Walden FOM); take from Murmann ADC survey
E_MAC = 5e-15         # J per analog MAC inside the array (assumption)
T_ADC = 10e-9         # s per ADC conversion (assumption, used for sharing latency)
W_BITS, A_BITS = 8, 8
PCT = 99.9            # percentile giving the nominal ADC full-scale
# ------------------------------------------------------------------------------
BITS = list(range(2, 9))
NOISES = [0.0, 0.02, 0.05, 0.1, 0.2]       # sigma as a fraction of nominal full-scale
TOLS = [0.0025, 0.005, 0.01, 0.02]         # adaptive-search accuracy tolerances
SIZES_N = [32, 64, 128, 256]
ALPHAS = np.linspace(0.05, 1.0, 20)        # candidate FS scale factors
FOM_SCALES = [0.25, 0.5, 1, 2, 4]          # sensitivity grid: multiples of FOM_ADC
EMAC_SCALES = [0.2, 0.5, 1, 2, 5]          # sensitivity grid: multiples of E_MAC
RATIOS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 20]   # FOM/E_MAC relative to the baseline
STAR_COLORS = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c"]
REF_BITS = {0.0025: 5, 0.005: 4, 0.01: 4, 0.02: 3}   # nearest uniform design used for the paired tests
LOAD_RAW = False      # True: skip simulation and reuse <out>/raw_results.pkl


# ---------------- Data -------------------------------------------------------
def load_data(name, seed=0):
    rng = np.random.RandomState(seed)
    if name in ("mnist", "fmnist"):
        from sklearn.datasets import fetch_openml
        oid = "mnist_784" if name == "mnist" else "Fashion-MNIST"
        X, y = fetch_openml(oid, version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0
        y = y.astype(int)
        return X[:60000], y[:60000], X[60000:], y[60000:]
    from sklearn.datasets import load_digits
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n = int(0.8 * len(X))
    return X[:n], y[:n], X[n:], y[n:]


# ---------------- Float MLP training (NumPy) ---------------------------------
def train_mlp(Xtr, ytr, sizes, epochs=15, lr=0.1, bs=128, seed=0):
    rng = np.random.RandomState(seed)
    Ws = [(rng.randn(i, o) * np.sqrt(2.0 / i)).astype(np.float32)
          for i, o in zip(sizes[:-1], sizes[1:])]
    Y = np.eye(sizes[-1], dtype=np.float32)[ytr]
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), bs):
            b = perm[s:s + bs]
            acts = [Xtr[b]]
            for k, W in enumerate(Ws):
                z = acts[-1] @ W
                acts.append(np.maximum(z, 0) if k < len(Ws) - 1 else z)
            logits = acts[-1] - acts[-1].max(1, keepdims=True)
            p = np.exp(logits); p /= p.sum(1, keepdims=True)
            g = (p - Y[b]) / len(b)
            for k in reversed(range(len(Ws))):
                gW = acts[k].T @ g
                if k > 0:
                    g = (g @ Ws[k].T) * (acts[k] > 0)
                Ws[k] -= lr * gW
    return Ws


# ---------------- Quantization helpers ---------------------------------------
def quant_sym(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = np.float32(np.abs(w).max() / qmax)
    return (np.round(w / s).clip(-qmax, qmax) * s).astype(np.float32)


def quant_unsigned(x, bits, xmax):
    qmax = 2 ** bits - 1
    s = np.float32(xmax / qmax)
    return (np.round(x / s).clip(0, qmax) * s).astype(np.float32)


def adc(psum, bits, fs, noise_abs=0.0, rng=None):
    """Signed uniform ADC over [-fs, fs]. bits=None -> ideal (no quantization)."""
    if noise_abs > 0:
        psum = psum + rng.standard_normal(psum.shape, dtype=np.float32) * np.float32(noise_abs)
    if bits is None:
        return psum
    step = np.float32(fs / 2 ** (bits - 1))
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    return (np.clip(np.round(psum / step), lo, hi) * step).astype(np.float32)


def relu_or_id(z, k, L):
    return np.maximum(z, 0) if k < L - 1 else z


# ---------------- CIM network -------------------------------------------------
class CIMNet:
    """Behavioral CIM model with an explicit device/input mapping.

    * Weights (signed 8-bit integers) are split in sign-magnitude into K = ceil(7 / cell_bits)
      slices of `cell_bits` bits (signed slices, e.g. differential column pairs).
    * Activations (unsigned 8-bit integers) are applied in J = ceil(8 / in_bits) cycles of
      `in_bits` bits each (bit-serial when in_bits < 8).
    * One ADC conversion per (tile, column, input cycle, weight slice); the digital side
      shift-and-adds the J*K partial results.  cell_bits >= 7 and in_bits = 8 reproduce the
      idealized "one 8-bit analog device, one analog input step" mapping of v1-v4.
    * ADC full-scale ranges are calibrated per layer, per input cycle and per weight slice.
    * Array energy per complete 8b x 8b MAC (E_MAC) is assumed independent of the mapping.
    """

    def __init__(self, Ws, Xcal, N=128, in_bits=8, cell_bits=8):
        self.N, self.d, self.c = N, in_bits, cell_bits
        self.J = int(np.ceil(A_BITS / in_bits))
        self.K = int(np.ceil((W_BITS - 1) / cell_bits))
        self.mult = self.J * self.K                 # ADC conversions per (tile, column)
        self.opt_fs = True                          # use optimized full-scale?
        qmax = 2 ** (W_BITS - 1) - 1
        self.Wint, self.Wscale, self.Wq, self.Wcat = [], [], [], []
        for W in Ws:
            sc = np.float32(np.abs(W).max() / qmax)
            wi = np.clip(np.rint(W / sc), -qmax, qmax).astype(np.int32)
            self.Wint.append(wi); self.Wscale.append(sc)
            self.Wq.append((wi * sc).astype(np.float32))
            self.Wcat.append(self._slice_weights(wi))
        L = len(Ws)
        self.amax, self.s_in, self.fs0, self.fs_opt = [], [], [], []
        h = Xcal
        for k in range(L):
            amax = np.float32(np.percentile(h, PCT))
            amax = amax if amax > 0 else np.float32(1.0)
            self.amax.append(amax)
            s = np.float32(amax / (2 ** A_BITS - 1)); self.s_in.append(s)
            a = self._quant_in(h, k)
            O = self.Wint[k].shape[1]
            fs0 = np.ones((self.J, self.K), np.float32)
            tab = {b: np.ones((self.J, self.K), np.float32) for b in BITS}
            pre = np.zeros((len(h), O), np.float32)
            for j, P in self._psums(a, k):
                for kk in range(self.K):
                    v = P[:, :, kk * O:(kk + 1) * O]
                    f0 = np.float32(max(float(np.percentile(np.abs(v), PCT)), 1.0))
                    fs0[j, kk] = f0
                    t = self._optimize_fs(v, f0)
                    for b in BITS:
                        tab[b][j, kk] = t[b]
                    pre += v.sum(0) * np.float32(2 ** (self.d * j + self.c * kk))
            self.fs0.append(fs0); self.fs_opt.append(tab)
            h = relu_or_id(pre * (s * self.Wscale[k]), k, L)

    # ---- mapping helpers ----
    def _slice_weights(self, wi):
        mag, sgn = np.abs(wi), np.sign(wi)
        sl = [((mag >> (self.c * k)) & (2 ** self.c - 1)) * sgn for k in range(self.K)]
        pad = (-wi.shape[0]) % self.N
        sl = [np.pad(x, ((0, pad), (0, 0))).astype(np.float32) for x in sl]
        T = sl[0].shape[0] // self.N
        return np.concatenate([x.reshape(T, self.N, -1) for x in sl], axis=2)   # (T, N, K*O)

    def _quant_in(self, h, k):
        return np.clip(np.rint(h / self.s_in[k]), 0, 2 ** A_BITS - 1).astype(np.int32)

    def _psums(self, a, k):
        """Yield (cycle j, psum array (T, B, K*O)) for integer activations `a`."""
        N = self.N
        pad = (-self.Wint[k].shape[0]) % N
        ap = np.pad(a, ((0, 0), (0, pad)))
        T = ap.shape[1] // N
        ap = ap.reshape(len(a), T, N)
        for j in range(self.J):
            plane = ((ap >> (self.d * j)) & (2 ** self.d - 1)).astype(np.float32).transpose(1, 0, 2)
            yield j, np.matmul(np.ascontiguousarray(plane), self.Wcat[k])

    def _optimize_fs(self, ps, fs0):
        """For every ADC resolution pick the FS (alpha * fs0) with minimum quantization MSE."""
        n_sub = max(60, 500 // self.mult)
        sub = ps[:, ::max(1, ps.shape[1] // n_sub)]
        table = {}
        for b in BITS:
            errs = [np.mean((adc(sub, b, al * fs0) - sub) ** 2) for al in ALPHAS]
            table[b] = np.float32(ALPHAS[int(np.argmin(errs))] * fs0)
        return table

    def fs(self, k, b):
        return self.fs_opt[k][b] if (self.opt_fs and b is not None) else self.fs0[k]

    def n_tiles(self):
        return [int(np.ceil(W.shape[0] / self.N)) for W in self.Wq]

    # ---- inference ----
    def _layer(self, h, k, b, noise=0.0, rng=None):
        L = len(self.Wq)
        O = self.Wint[k].shape[1]
        acc = np.zeros((len(h), O), np.float32)
        fs = self.fs(k, b)
        for j, P in self._psums(self._quant_in(h, k), k):
            for kk in range(self.K):
                v = P[:, :, kk * O:(kk + 1) * O]
                if noise > 0:
                    v += rng.standard_normal(v.shape, dtype=np.float32) * np.float32(noise * self.fs0[k][j, kk])
                w = np.float32(2 ** (self.d * j + self.c * kk))
                if b is None:
                    acc += v.sum(0) * w
                else:
                    step = np.float32(fs[j, kk] / 2 ** (b - 1))
                    v /= step
                    np.rint(v, out=v)
                    np.clip(v, -2 ** (b - 1), 2 ** (b - 1) - 1, out=v)
                    acc += v.sum(0) * (step * w)
        return relu_or_id(acc * (self.s_in[k] * self.Wscale[k]), k, L)

    def predict(self, X, bits, noise=0.0, seed=0, bs=1000):
        rng = np.random.default_rng(seed)
        out = []
        for s in range(0, len(X), bs):
            h = X[s:s + bs]
            for k in range(len(self.Wq)):
                h = self._layer(h, k, bits[k], noise, rng)
            out.append(h.argmax(1))
        return np.concatenate(out)

    def layer_forward(self, h, k, b, bs=2000):
        """Noiseless forward pass of ONE layer with ADC resolution b (activation applied)."""
        return np.concatenate([self._layer(h[s:s + bs], k, b) for s in range(0, len(h), bs)])

    def accuracy(self, X, y, bits, noise=0.0, seed=0):
        return float((self.predict(X, bits, noise, seed) == y).mean())

    # ---- energy / latency model (per inference) ----
    def energy(self, bits_per_layer):
        e_adc, e_arr = 0.0, 0.0
        for W, T, b in zip(self.Wq, self.n_tiles(), bits_per_layer):
            conv = T * W.shape[1] * self.mult            # ADC conversions per inference
            e_adc += conv * FOM_ADC * 2 ** (b if b else A_BITS + 4)
            e_arr += W.shape[0] * W.shape[1] * E_MAC     # per complete 8b x 8b MAC
        return e_adc, e_arr

    def latency(self, share):
        # `share` columns served by one ADC -> `share` sequential conversions per tile
        return sum(self.n_tiles()) * share * self.mult * T_ADC


# ---------------- Adaptive (proposed) per-layer ADC resolution ----------------
def adaptive_allocation(net, Xv, yv, tol, bmin=2, bmax=8, acc_fn=None):
    """Greedy: lower the ADC bits of the layer with the best energy saved per accuracy lost,
    as long as validation accuracy stays within `tol` of the all-bmax design."""
    L = len(net.Wq)
    if acc_fn is None:
        acc_fn = lambda bb: net.accuracy(Xv, yv, bb)
    bits = [bmax] * L
    base = acc_fn(bits)
    cur_e = sum(net.energy(bits))
    while True:
        best, cand = -1.0, None
        for k in range(L):
            if bits[k] <= bmin:
                continue
            t = bits.copy(); t[k] -= 1
            acc = acc_fn(t)
            if base - acc > tol:
                continue
            gain = (cur_e - sum(net.energy(t))) / (max(base - acc, 0.0) + 1e-4)
            if gain > best:
                best, cand = gain, t
        if cand is None:
            return bits
        bits = cand
        cur_e = sum(net.energy(bits))


def enumerate_all(net, X, y):
    """Accuracy of EVERY per-layer resolution combination (len(BITS)**L), with prefix caching."""
    L = len(net.Wq)
    out = {}

    def rec(h, k, prefix):
        for b in BITS:
            z = net.layer_forward(h, k, b)
            if k == L - 1:
                out[tuple(prefix + [b])] = float((z.argmax(1) == y).mean())
            else:
                rec(z, k + 1, prefix + [b])
    rec(X, 0, [])
    return out


def pareto_front(cfgs, acc, en):
    """Non-dominated configurations (min energy, max accuracy)."""
    order = sorted(cfgs, key=lambda c: (en[c], -acc[c]))
    front, best = [], -1.0
    for c in order:
        if acc[c] > best + 1e-12:
            front.append(c); best = acc[c]
    return front


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts b, c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def sd(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def matched_saving(uni_acc, uni_e, ad_acc, ad_e):
    """Energy saving (%) of the adaptive design vs the cheapest uniform one with >= accuracy."""
    ok = [(e, b) for b, a, e in zip(BITS, uni_acc, uni_e) if a >= ad_acc]
    if not ok:
        return np.nan, 0
    e, b = min(ok)
    return 100.0 * (1 - ad_e / e), b


# ---------------- One full experiment for one trained network -----------------
def run_seed(seed, a, data):
    Xtr, ytr, Xte, yte = data
    t0 = time.time()
    nv = len(Xtr) // 10                       # held-out validation split (adaptive search only)
    Xv, yv, Xt, yt = Xtr[-nv:], ytr[-nv:], Xtr[:-nv], ytr[:-nv]
    sizes = [Xt.shape[1]] + a.hidden + [10]
    Ws = train_mlp(Xt, yt, sizes, epochs=a.epochs, seed=seed)
    h = Xte
    for k, W in enumerate(Ws):
        h = relu_or_id(h @ W, k, len(Ws))
    r = {"float_acc": float((h.argmax(1) == yte).mean())}

    rng = np.random.RandomState(100 + seed)
    cal = Xt[rng.choice(len(Xt), min(1000, len(Xt)), replace=False)]
    L = len(Ws)
    net = CIMNet(Ws, cal, N=128, in_bits=a.in_bits, cell_bits=a.cell_bits)
    r["ideal"] = net.accuracy(Xte, yte, [None] * L)

    # Exp 1: uniform ADC bits; optimized FS + noise levels; naive FS baseline
    r["uni_opt"] = {n: [net.accuracy(Xte, yte, [b] * L, noise=n, seed=seed) for b in BITS]
                    for n in NOISES}
    net.opt_fs = False
    r["uni_naive"] = [net.accuracy(Xte, yte, [b] * L) for b in BITS]
    net.opt_fs = True
    r["e_adc"] = [net.energy([b] * L)[0] for b in BITS]
    r["e_arr"] = net.energy([8] * L)[1]
    r["e_tot"] = [e + r["e_arr"] for e in r["e_adc"]]

    # Exp 2: array size
    r["size"] = {}
    for N in SIZES_N:
        nn = net if N == 128 else CIMNet(Ws, cal, N=N, in_bits=a.in_bits, cell_bits=a.cell_bits)
        r["size"][N] = {"acc": [nn.accuracy(Xte, yte, [b] * L) for b in BITS],
                        "e": [sum(nn.energy([b] * L)) for b in BITS]}

    # Exp 3: ADC sharing (latency only depends on the architecture)
    r["latency"] = {s: net.latency(s) for s in [1, 2, 4, 8, 16]}
    r["tiles"] = net.n_tiles()
    r["shapes"] = [tuple(W.shape) for W in net.Wq]
    r["mult"] = net.mult

    # Exhaustive enumeration first: it gives the validation accuracy of every combination, which the
    # greedy search below simply looks up (identical result, no repeated evaluations).
    val = enumerate_all(net, Xv, yv)
    cfgs = list(val)
    en = {c: sum(net.energy(list(c))) for c in cfgs}
    acc_fn = lambda bb: val[tuple(bb)]

    # Exp 4: adaptive per-layer ADC resolution
    r["adaptive"] = {}
    for tol in TOLS:
        bits = adaptive_allocation(net, Xv, yv, tol, acc_fn=acc_fn)
        acc = net.accuracy(Xte, yte, bits)
        e = sum(net.energy(bits))
        sav, ub = matched_saving(r["uni_opt"][0.0], r["e_tot"], acc, e)
        r["adaptive"][tol] = {"bits": bits, "acc": acc, "e": e, "saving": sav, "uni_bits": ub}
        # paired McNemar test against the nearest uniform design
        rb = REF_BITS[tol]
        ok_a = net.predict(Xte, bits) == yte
        ok_r = net.predict(Xte, [rb] * L) == yte
        nb, nc = int((ok_a & ~ok_r).sum()), int((~ok_a & ok_r).sum())
        r["adaptive"][tol]["sig"] = {"ref_bits": rb, "d_acc": float(ok_a.mean() - ok_r.mean()),
                                     "b": nb, "c": nc, "p": mcnemar_exact(nb, nc)}

    # Exp 6: exhaustive search over all 7^L resolution combinations (validation set)
    front = pareto_front(cfgs, val, en)
    base = val[tuple([8] * L)]
    opt = {}
    for tol in TOLS:
        feas = [c for c in cfgs if val[c] >= base - tol - 1e-12]
        c = min(feas, key=lambda c: (en[c], -val[c]))
        opt[tol] = {"bits": list(c), "e": en[c], "val_acc": val[c], "test_acc": net.accuracy(Xte, yte, list(c))}
        r["adaptive"][tol]["val_acc"] = val[tuple(r["adaptive"][tol]["bits"])]
    r["exh"] = {"val": val, "e": en, "n": len(cfgs), "opt": opt,
                "front": [(list(c), en[c], val[c], None) for c in front]}

    # Exp 7: per-layer sensitivity (one layer varies, the others stay at 8 bits)
    r["layer_sens"] = [[net.accuracy(Xte, yte, [b if j == k else 8 for j in range(L)]) for b in BITS]
                       for k in range(L)]
    print(f"  seed {seed}: float={r['float_acc']:.4f} ideal={r['ideal']:.4f} "
          f"adaptive={[(t, v['bits']) for t, v in r['adaptive'].items()]}  ({time.time()-t0:.0f}s)")
    return r


# ---------------- Sensitivity to FOM_ADC and E_MAC -----------------------------
# NOTE: the greedy adaptive allocation does not depend on FOM_ADC or E_MAC (E_MAC cancels in
# energy differences and FOM_ADC scales all gains equally), so the sensitivity analysis only
# needs to re-evaluate energies of the already-found designs. Also, relative results depend
# only on the ratio FOM_ADC / E_MAC.
def energy_with(shapes, tiles, bits, fom, emac, mult=1):
    e_adc = sum(T * sh[1] * mult * fom * 2 ** b for sh, T, b in zip(shapes, tiles, bits))
    e_arr = sum(sh[0] * sh[1] * emac for sh in shapes)
    return e_adc, e_arr


def nanmean_safe(x):
    arr = np.array(x, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def sens_point(R, fom, emac):
    acc = {"share8": [], "share4": [], "e8": [], "e5": [], "sav85": []}
    sav = {t: [] for t in TOLS}
    for r in R:
        L = len(r["shapes"])
        ue, ea_l, er_l = [], [], []
        for b in BITS:
            ea, er = energy_with(r["shapes"], r["tiles"], [b] * L, fom, emac, r.get("mult", 1))
            ue.append(ea + er); ea_l.append(ea); er_l.append(er)
        i8, i5, i4 = BITS.index(8), BITS.index(5), BITS.index(4)
        acc["share8"].append(100 * ea_l[i8] / ue[i8]); acc["share4"].append(100 * ea_l[i4] / ue[i4])
        acc["e8"].append(ue[i8] * 1e9); acc["e5"].append(ue[i5] * 1e9)
        acc["sav85"].append(100 * (1 - ue[i5] / ue[i8]))
        for t in TOLS:
            v = r["adaptive"][t]
            ea, er = energy_with(r["shapes"], r["tiles"], v["bits"], fom, emac, r.get("mult", 1))
            sav[t].append(matched_saving(r["uni_opt"][0.0], ue, v["acc"], ea + er)[0])
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["sav"] = {t: nanmean_safe(v) for t, v in sav.items()}
    return out


def sensitivity(R, a):
    r0 = R[0]
    with open(f"{a.out}/exp5_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FOM_scale", "E_MAC_scale", "FOM_fJ", "E_MAC_fJ", "FOM/E_MAC", "ADC_share_8b_%",
                    "ADC_share_4b_%", "E_8b_nJ", "E_5b_nJ", "saving_8b_to_5b_%"] +
                   [f"adaptive_saving_tol{t*100:g}%_vs_matched_uniform" for t in TOLS])
        for fs in FOM_SCALES:
            for es in EMAC_SCALES:
                p = sens_point(R, FOM_ADC * fs, E_MAC * es)
                w.writerow([fs, es, f"{FOM_ADC*fs*1e15:.1f}", f"{E_MAC*es*1e15:.2f}",
                            f"{FOM_ADC*fs/(E_MAC*es):.2f}", f"{p['share8']:.1f}", f"{p['share4']:.1f}",
                            f"{p['e8']:.3f}", f"{p['e5']:.3f}", f"{p['sav85']:.1f}"] +
                           [f"{p['sav'][t]:.1f}" for t in TOLS])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    base_adc, base_arr = np.array(r0["e_adc"]), r0["e_arr"]
    for rho in [0.1, 0.5, 1, 2, 10]:
        share = 100 * rho * base_adc / (rho * base_adc + base_arr)
        ax[0].plot(BITS, share, marker="o", ms=3, lw=2 if rho == 1 else 1,
                   label=f"FOM/E_MAC = {rho:g}x" + (" (baseline)" if rho == 1 else ""))
    ax[0].set_xlabel("ADC resolution (bits)"); ax[0].set_ylabel("ADC share of total energy (%)")
    ax[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[0].grid(alpha=.3)
    pts = [sens_point(R, FOM_ADC * rho, E_MAC) for rho in RATIOS]
    for t in TOLS:
        ax[1].plot(RATIOS, [p["sav"][t] for p in pts], marker="o", ms=3, label=f"tol={t*100:g}%")
    ax[1].axvline(1, ls="--", c="gray", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("FOM_ADC / E_MAC (relative to baseline)")
    ax[1].set_ylabel("Adaptive energy saving (%)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig5_sensitivity.png", dpi=300, bbox_inches="tight"); plt.close()
    lo, hi = pts[0], pts[-1]
    print("\n=== Sensitivity (FOM/E_MAC from %gx to %gx of baseline) ===" % (RATIOS[0], RATIOS[-1]))
    print(f"ADC share at 8 bits: {lo['share8']:.1f}% -> {hi['share8']:.1f}% | "
          f"at 4 bits: {lo['share4']:.1f}% -> {hi['share4']:.1f}%")
    print(f"Energy saving 8b->5b uniform: {lo['sav85']:.1f}% -> {hi['sav85']:.1f}%")
    for t in TOLS:
        print(f"Adaptive saving (tol {t*100:g}%): " +
              ", ".join(f"{rho:g}x: {p['sav'][t]:.1f}%" for rho, p in zip(RATIOS, pts)))


# ---------------- Exhaustive search, layer sensitivity, significance ----------
def extra_analyses(R, a, ideal_m):
    L = len(R[0]["shapes"])
    ylo, yhi = ideal_m * 100 - 8, ideal_m * 100 + 1.2

    # ---- exp6: greedy vs exhaustive optimum ----
    gaps, same = {t: [] for t in TOLS}, {t: 0 for t in TOLS}
    with open(f"{a.out}/exp6_greedy_vs_exhaustive.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "tol", "greedy_bits", "greedy_E_nJ", "greedy_val_acc", "optimum_bits",
                    "optimum_E_nJ", "optimum_val_acc", "energy_gap_%", "greedy_test_acc",
                    "optimum_test_acc", "n_configs"])
        for s, r in enumerate(R):
            for t in TOLS:
                g, o = r["adaptive"][t], r["exh"]["opt"][t]
                gap = 100 * (g["e"] / o["e"] - 1)
                gaps[t].append(gap); same[t] += int(list(g["bits"]) == list(o["bits"]))
                w.writerow([s, t, g["bits"], f"{g['e']*1e9:.3f}", f"{g['val_acc']:.4f}", o["bits"],
                            f"{o['e']*1e9:.3f}", f"{o['val_acc']:.4f}", f"{gap:.1f}", f"{g['acc']:.4f}",
                            f"{o['test_acc']:.4f}", r["exh"]["n"]])
    # figure: all configurations of network 0 (validation accuracy)
    ex = R[0]["exh"]
    plt.figure(figsize=(5.2, 3.8))
    E = np.array([ex["e"][c] for c in ex["val"]]) * 1e9
    A = np.array([ex["val"][c] for c in ex["val"]]) * 100
    plt.scatter(E, A, s=6, c="lightgray", label=f"all {ex['n']} per-layer combinations")
    fr = ex["front"]
    plt.step([x[1] * 1e9 for x in fr], [x[2] * 100 for x in fr], where="post", c="k", lw=1,
             label="exhaustive Pareto front")
    uni = [tuple([b] * L) for b in BITS]
    plt.plot([ex["e"][c] * 1e9 for c in uni], [ex["val"][c] * 100 for c in uni], "o-", c="#1f77b4",
             ms=3, label="uniform ADC")
    for t, c in zip(TOLS, STAR_COLORS):
        g = R[0]["adaptive"][t]
        plt.plot(g["e"] * 1e9, g["val_acc"] * 100, "*", ms=11, c=c, label=f"greedy, tol={t*100:g}%")
    plt.xscale("log"); plt.ylim(ylo, yhi)
    plt.xlabel("Energy / inference (nJ)"); plt.ylabel("Validation accuracy (%)")
    plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{a.out}/fig6_exhaustive.png", dpi=300); plt.close()

    # ---- exp7: per-layer sensitivity ----
    conv = [T * sh[1] for T, sh in zip(R[0]["tiles"], R[0]["shapes"])]
    plt.figure(figsize=(5.2, 3.6))
    with open(f"{a.out}/exp7_layer_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "conversions", "bits", "acc_mean", "acc_std"])
        for k in range(L):
            m, sdv = ms([r["layer_sens"][k] for r in R])
            for b, mm, ss in zip(BITS, m, sdv):
                w.writerow([k + 1, conv[k], b, f"{mm:.4f}", f"{ss:.4f}"])
            plt.errorbar(BITS, m * 100, yerr=sdv * 100, marker="o", ms=3, capsize=2,
                         label=f"layer {k+1} ({100*conv[k]/sum(conv):.1f}% of conversions)")
    plt.axhline(ideal_m * 100, ls=":", c="gray")
    plt.xlabel("ADC resolution of this layer (bits); other layers at 8 bits"); plt.ylabel("Test accuracy (%)")
    plt.ylim(ylo, yhi); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{a.out}/fig7_layer_sensitivity.png", dpi=300); plt.close()

    # ---- exp8: paired significance ----
    with open(f"{a.out}/exp8_significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "tol", "reference_uniform_bits", "delta_acc_pp", "adaptive_only_correct",
                    "reference_only_correct", "mcnemar_p"])
        for s, r in enumerate(R):
            for t in TOLS:
                g = r["adaptive"][t]["sig"]
                w.writerow([s, t, g["ref_bits"], f"{g['d_acc']*100:+.2f}", g["b"], g["c"], f"{g['p']:.4f}"])

    # ---- console summary ----
    print(f"\n=== Greedy vs exhaustive optimum ({R[0]['exh']['n']} combinations per network) ===")
    for t in TOLS:
        print(f"tol {t*100:>5g}%: greedy = optimum in {same[t]}/{len(R)} networks; "
              f"energy gap mean {np.mean(gaps[t]):.1f}% (max {np.max(gaps[t]):.1f}%)")
    print("\n=== Per-layer sensitivity (test acc, other layers at 8 bits) ===")
    for k in range(L):
        m, _ = ms([r["layer_sens"][k] for r in R])
        print(f"layer {k+1} ({100*conv[k]/sum(conv):.1f}% of conversions): " +
              ", ".join(f"{b}b {x*100:.2f}" for b, x in zip(BITS, m)))
    print("\n=== Adaptive vs nearest uniform design: paired McNemar test on the test set ===")
    for t in TOLS:
        d = [r["adaptive"][t]["sig"] for r in R]
        pos = sum(1 for x in d if x["p"] < 0.05 and x["d_acc"] > 0)
        neg = sum(1 for x in d if x["p"] < 0.05 and x["d_acc"] < 0)
        dm = np.array([x["d_acc"] * 100 for x in d])
        print(f"tol {t*100:>5g}% vs uniform {REF_BITS[t]}b: delta acc {dm.mean():+.2f} +- {sd(dm):.2f} pp "
              f"(sample std); p<0.05 in {pos+neg}/{len(R)} networks (better: {pos}, worse: {neg})")


# ---------------- Aggregation / output ----------------------------------------
def ms(vals):
    A = np.array(vals, dtype=float)
    return A.mean(0), (A.std(0, ddof=1) if len(A) > 1 else np.zeros_like(A.mean(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["digits", "mnist", "fmnist"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--in-bits", type=int, default=1, dest="in_bits", help="activation bits applied per cycle")
    ap.add_argument("--cell-bits", type=int, default=2, dest="cell_bits", help="weight magnitude bits per device")
    ap.add_argument("--quick", action="store_true", default=True
                    , help="skip noise levels and tile-size study")
    ap.add_argument("--out", default=None)
    ap.add_argument("--load", action="store_true", help="reuse raw_results.pkl, skip simulation")
    a, _ = ap.parse_known_args()
    if a.out is None:
        tag = "" if (a.in_bits, a.cell_bits) == (8, 8) else f"_in{a.in_bits}_cell{a.cell_bits}"
        a.out = f"results_v5_{a.dataset}{tag}"
    if a.quick:
        NOISES[:] = [0.0]; SIZES_N[:] = [128]
    os.makedirs(a.out, exist_ok=True)

    if a.load or LOAD_RAW:
        with open(f"{a.out}/raw_results.pkl", "rb") as f:
            R = pickle.load(f)
        print(f"Loaded {len(R)} seeds from {a.out}/raw_results.pkl")
    else:
        data = load_data(a.dataset)
        print("Layer sizes:", [data[0].shape[1]] + a.hidden + [10])
        print(f"Mapping: {a.in_bits}-bit input cycles x {a.cell_bits}-bit cells -> "
              f"{int(np.ceil(A_BITS / a.in_bits))} cycles x {int(np.ceil((W_BITS - 1) / a.cell_bits))} slices per conversion group")
        R = [run_seed(s, a, data) for s in range(a.seeds)]
        with open(f"{a.out}/raw_results.pkl", "wb") as f:
            pickle.dump(R, f)
    r0 = R[0]

    # ---- Exp 1 CSV ----
    nv_m, nv_s = ms([r["uni_naive"] for r in R])
    op = {n: ms([r["uni_opt"][n] for r in R]) for n in NOISES}
    e_adc = np.mean([r["e_adc"] for r in R], 0)
    e_arr = r0["e_arr"]
    with open(f"{a.out}/exp1_uniform.csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["bits", "acc_naiveFS_mean", "acc_naiveFS_std"]
        for n in NOISES:
            hdr += [f"acc_optFS_noise{n}_mean", f"acc_optFS_noise{n}_std"]
        w.writerow(hdr + ["E_adc_nJ", "E_array_nJ", "E_total_nJ", "ADC_share_%"])
        for i, b in enumerate(BITS):
            row = [b, f"{nv_m[i]:.4f}", f"{nv_s[i]:.4f}"]
            for n in NOISES:
                row += [f"{op[n][0][i]:.4f}", f"{op[n][1][i]:.4f}"]
            w.writerow(row + [f"{e_adc[i]*1e9:.3f}", f"{e_arr*1e9:.3f}",
                              f"{(e_adc[i]+e_arr)*1e9:.3f}", f"{100*e_adc[i]/(e_adc[i]+e_arr):.1f}"])
    ideal_m = np.mean([r["ideal"] for r in R])

    # ---- Fig 1: accuracy vs bits ----
    plt.figure(figsize=(5.2, 3.6))
    for n in NOISES:
        plt.errorbar(BITS, op[n][0] * 100, yerr=op[n][1] * 100, marker="o", ms=3, capsize=2,
                     label=f"σ={n}·FS")
    plt.plot(BITS, nv_m * 100, "k--", marker="s", ms=3, label="σ=0, naive FS")
    plt.axhline(ideal_m * 100, ls=":", c="gray", label="ideal ADC")
    plt.xlabel("ADC resolution (bits)"); plt.ylabel("Accuracy (%)")
    plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{a.out}/fig1_acc_vs_bits.png", dpi=300); plt.close()

    # ---- Fig 2: energy breakdown ----
    plt.figure(figsize=(5.2, 3.6))
    plt.bar(BITS, e_adc * 1e9, label="ADC")
    plt.bar(BITS, [e_arr * 1e9] * len(BITS), bottom=e_adc * 1e9, label="Array")
    plt.yscale("log"); plt.ylim(bottom=0.1)
    plt.xlabel("ADC resolution (bits)"); plt.ylabel("Energy / inference (nJ)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{a.out}/fig2_energy_vs_bits.png", dpi=300); plt.close()

    # ---- Exp 2: array size (CSV + Fig 3) ----
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    with open(f"{a.out}/exp2_array_size.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "bits", "acc_mean", "acc_std", "E_total_nJ"])
        for N in SIZES_N:
            am, asd = ms([r["size"][N]["acc"] for r in R])
            em = np.mean([r["size"][N]["e"] for r in R], 0)
            for i, b in enumerate(BITS):
                w.writerow([N, b, f"{am[i]:.4f}", f"{asd[i]:.4f}", f"{em[i]*1e9:.3f}"])
            ax[0].errorbar(BITS, am * 100, yerr=asd * 100, marker="o", ms=3, capsize=2, label=f"{N} rows")
            ax[1].plot(em * 1e9, am * 100, marker="o", ms=3, label=f"{N} rows")
    ax[0].set_xlabel("ADC resolution (bits)"); ax[0].set_ylabel("Accuracy (%)")
    ax[1].set_xlabel("Energy / inference (nJ)"); ax[1].set_xscale("log"); ax[1].set_ylim(ideal_m * 100 - 8, ideal_m * 100 + 1.2)
    for x in ax: x.legend(fontsize=7); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig3_array_size.png", dpi=300); plt.close()

    # ---- Exp 3: ADC sharing ----
    with open(f"{a.out}/exp3_sharing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["share_factor", "latency_us", "ADCs_per_128_cols"])
        for s, lat in r0["latency"].items():
            w.writerow([s, f"{lat*1e6:.3f}", int(np.ceil(128 / s))])

    # ---- Exp 4: adaptive Pareto ----
    with open(f"{a.out}/exp4_adaptive_perseed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "tol", "bits_per_layer", "acc", "E_total_nJ",
                    "saving_vs_matched_uniform_%", "matched_uniform_bits"])
        for s, r in enumerate(R):
            for tol, v in r["adaptive"].items():
                w.writerow([s, tol, v["bits"], f"{v['acc']:.4f}", f"{v['e']*1e9:.3f}",
                            f"{v['saving']:.1f}", v["uni_bits"]])
    plt.figure(figsize=(5.2, 3.8))
    uni_acc, _ = ms([r["uni_opt"][0.0] for r in R])
    uni_e = np.mean([r["e_tot"] for r in R], 0) * 1e9
    plt.plot(uni_e, uni_acc * 100, "o-", label="Uniform ADC (optimized FS)")
    plt.plot(uni_e, nv_m * 100, "s--", c="gray", ms=3, label="Uniform ADC (naive FS)")
    for b, e, ac in zip(BITS, uni_e, uni_acc * 100):
        plt.annotate(f"{b}b", (e, ac), fontsize=6, xytext=(3, -8), textcoords="offset points")
    rows = []
    with open(f"{a.out}/exp4_adaptive.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tol", "acc_mean", "acc_std", "E_total_nJ_mean", "saving_vs_matched_uniform_%_mean",
                    "bits_per_layer_per_seed"])
        for tol in TOLS:
            accs = [r["adaptive"][tol]["acc"] for r in R]
            es = [r["adaptive"][tol]["e"] * 1e9 for r in R]
            sav = np.nanmean([r["adaptive"][tol]["saving"] for r in R]) if any(
                not np.isnan(r["adaptive"][tol]["saving"]) for r in R) else np.nan
            w.writerow([tol, f"{np.mean(accs):.4f}", f"{sd(accs):.4f}", f"{np.mean(es):.3f}",
                        f"{sav:.1f}", [r["adaptive"][tol]["bits"] for r in R]])
            plt.errorbar(np.mean(es), np.mean(accs) * 100, yerr=sd(accs) * 100, fmt="*", ms=10,
                         capsize=2, color=STAR_COLORS[TOLS.index(tol)],
                         label=f"Adaptive, tol={tol*100:g}%")
            rows.append((tol, np.mean(accs), sd(accs), np.mean(es), sav))
    plt.xscale("log"); plt.ylim(ideal_m * 100 - 8, ideal_m * 100 + 1.2)
    plt.xlabel("Energy / inference (nJ)"); plt.ylabel("Accuracy (%)")
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{a.out}/fig4_pareto.png", dpi=300); plt.close()

    # ---- console summary ----
    print(f"\n=== Summary over {len(R)} seeds ===")
    print(f"float acc: {np.mean([r['float_acc'] for r in R]):.4f} | ideal-ADC acc: {ideal_m:.4f}")
    print("bits | naive-FS acc | opt-FS acc | E_total(nJ) | ADC share")
    for i, b in enumerate(BITS):
        print(f"{b:>4} | {nv_m[i]:.4f}±{nv_s[i]:.4f} | {op[0.0][0][i]:.4f}±{op[0.0][1][i]:.4f} | "
              f"{uni_e[i]:8.3f} | {100*e_adc[i]/(e_adc[i]+e_arr):.1f}%")
    print("Adaptive: tol | acc | E(nJ) | saving vs cheapest uniform with >= accuracy")
    for tol, am, asd, em, sav in rows:
        print(f"  {tol*100:>5g}% | {am:.4f}±{asd:.4f} | {em:.3f} | {sav:.1f}%")
    sensitivity(R, a)
    extra_analyses(R, a, ideal_m)
    print(f"Done. Results and figures in ./{a.out}/")


if __name__ == "__main__":
    main()
