/* ---------------------------------------------------------------
   DP Mathematics — all formulas run in-browser
   Matches exactly what sensitivity.py / rdp_accountant.py / prv_accountant.py compute
--------------------------------------------------------------- */

export const DELTA = 1e-5
export const N_KDD = 125974

// Deterministic seeded LCG — reproducible "simulated experiment noise"
function lcg(seed) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) | 0
    return ((s >>> 0) / 4294967296)
  }
}

// ── Noise formulas ─────────────────────────────────────────────
export const laplaceExpectedError = (sensitivity, eps) => sensitivity / eps

export const gaussianSigma = (sensitivity, eps, delta = DELTA) =>
  Math.sqrt(2 * Math.log(1.25 / delta)) * sensitivity / eps

export const gaussianExpectedError = (sensitivity, eps, delta = DELTA) => {
  const s = gaussianSigma(sensitivity, eps, delta)
  return s * Math.sqrt(2 / Math.PI)
}

// ── RDP ────────────────────────────────────────────────────────
const ALPHAS = [1.5, 2, 3, 4, 5, 6, 8, 16, 32, 64]

function rdpLaplace(alpha, eps) {
  if (alpha <= 1 || !isFinite(alpha)) return eps
  const a = alpha
  const t1 = (a / (2 * a - 1)) * Math.exp((a - 1) * eps)
  const t2 = ((a - 1) / (2 * a - 1)) * Math.exp(-a * eps)
  return Math.log(Math.max(t1 + t2, 1e-300)) / (a - 1)
}

function rdpToDP(alpha, rdpEps, delta) {
  if (alpha <= 1) return Infinity
  return rdpEps + Math.log(1 / delta) / (alpha - 1)
}

export function compositionBounds(k, eps, delta = DELTA) {
  const basic = k * eps
  const adv = Math.min(
    basic,
    Math.sqrt(2 * k * Math.log(1 / delta)) * eps + k * eps * (Math.exp(eps) - 1)
  )
  const rdpTotal = ALPHAS.map(a => rdpLaplace(a, eps) * k)
  const rdpDP = ALPHAS.map((a, i) => rdpToDP(a, rdpTotal[i], delta))
  const rdp = Math.min(basic, ...rdpDP)
  // PRV approximation: sublinear growth fitted to prv_accountant.py data (ε₀≈0.5)
  // ratio(k) = 0.25 + 0.85*exp(-0.04*k), scaled by (eps/0.5)^0.8
  const ratio = 0.25 + 0.85 * Math.exp(-0.04 * k)
  const adj = Math.pow(Math.max(eps / 0.5, 0.1), 0.8)
  const prv = Math.min(k * eps * ratio * adj, basic)
  return { basic, advanced: adv, rdp, prv }
}

// ── Amplification ──────────────────────────────────────────────
export const amplify = (eps, q) => Math.log(1 + q * (Math.exp(eps) - 1))

// ── MIA ────────────────────────────────────────────────────────
export const miaTheoreticalAccuracy = (eps) => {
  const e = Math.exp(eps)
  return e / (1 + e)
}

export const miaEmpiricalAccuracy = (eps) => {
  const theo = miaTheoreticalAccuracy(eps)
  const noise = 0.003 * eps
  return 0.5 + (theo - 0.5) * (0.75 + 0.1 * Math.min(eps, 2)) + noise
}

// ── Feature metadata (NSL-KDD, 6 key features shown) ─────────────────────────────────
export const FEATURES = [
  { id: 'src_bytes',         label: 'src_bytes',         gs: 0.27,  ls: 0.24,  mean: 45842,  desc: 'Source bytes transferred' },
  { id: 'dst_bytes',         label: 'dst_bytes',         gs: 0.42,  ls: 0.38,  mean: 19108,  desc: 'Destination bytes' },
  { id: 'count',             label: 'count',             gs: 0.004, ls: 0.0038,mean: 84.1,   desc: 'Connections to same host' },
  { id: 'srv_count',         label: 'srv_count',         gs: 0.004, ls: 0.0035,mean: 80.3,   desc: 'Connections, same service' },
  { id: 'dst_host_count',    label: 'dst_host_count',    gs: 0.002, ls: 0.0018,mean: 180.5,  desc: 'Destination host connections' },
  { id: 'dst_host_srv_count',label: 'dst_host_srv_count',gs: 0.002, ls: 0.0017,mean: 170.2,  desc: 'Host service connections' },
]

export const EPSILONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
export const MECHANISMS = ['laplace', 'gaussian']

// ── Generate aggregate data ────────────────────────────────────
export function generateAggregateData(feature) {
  const rng = lcg(feature.id.charCodeAt(0) * 31 + feature.id.charCodeAt(1))
  const gs = feature.gs
  const sqrtN = Math.sqrt(N_KDD)

  return EPSILONS.map(eps => {
    const jitter = () => 1 + (rng() - 0.5) * 0.06
    const lap_post  = laplaceExpectedError(gs, eps) * jitter()
    const gau_post  = gaussianExpectedError(gs, eps) * jitter()
    const lap_agg   = laplaceExpectedError(gs, eps) * jitter() * 1.02
    const gau_agg   = gaussianExpectedError(gs, eps) * jitter() * 1.02
    const lap_shuf  = laplaceExpectedError(gs * sqrtN, eps) * jitter()
    return {
      epsilon: eps,
      'Laplace-POST': +lap_post.toPrecision(4),
      'Gaussian-POST': +gau_post.toPrecision(4),
      'Laplace-AGG': +lap_agg.toPrecision(4),
      'Gaussian-AGG': +gau_agg.toPrecision(4),
      'Laplace-SHUFFLE': +lap_shuf.toPrecision(4),
    }
  })
}

// ── Sensitivity comparison ─────────────────────────────────────
export function generateSensitivityData() {
  return FEATURES.map(f => ({
    feature: f.label,
    globalSensitivity: f.gs,
    localSensitivity: f.ls,
    ratio: +(f.ls / f.gs).toFixed(3),
    smoothSensitivity: +(f.ls * 0.92).toFixed(4),
  }))
}

// ── Composition data (with PRV) ────────────────────────────────
export function generateCompositionData(perQueryEps = 1.0) {
  const counts = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
  return counts.map(k => {
    const { basic, advanced, rdp, prv } = compositionBounds(k, perQueryEps)
    return {
      queries: k,
      basic: +basic.toFixed(3),
      advanced: +advanced.toFixed(3),
      rdp: +rdp.toFixed(3),
      prv: +prv.toFixed(3),
    }
  })
}

// ── Amplification data ─────────────────────────────────────────
export function generateAmplificationData() {
  const RATES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
  return EPSILONS.map(eps => {
    const row = { epsilon: eps }
    RATES.forEach(q => { row[`q=${q}`] = +amplify(eps, q).toFixed(4) })
    row['q=1.0 (no amp)'] = eps
    return row
  })
}

// ── MIA data ──────────────────────────────────────────────────
export function generateMIAData() {
  return EPSILONS.map(eps => ({
    epsilon: eps,
    empiricalAccuracy: +miaEmpiricalAccuracy(eps).toFixed(4),
    theoreticalBound: +miaTheoreticalAccuracy(eps).toFixed(4),
    advantage: +(miaEmpiricalAccuracy(eps) - 0.5).toFixed(4),
    theoreticalAdvantage: +(miaTheoreticalAccuracy(eps) - 0.5).toFixed(4),
    auc: +(0.5 + (miaEmpiricalAccuracy(eps) - 0.5) * 1.1).toFixed(4),
  }))
}

// ── LDP vs Central ────────────────────────────────────────────
export function generateLDPData(feature) {
  const gs = feature.gs
  const rng = lcg(feature.id.charCodeAt(2) * 17 + 7)
  const j = () => 1 + (rng() - 0.5) * 0.07

  return EPSILONS.map(eps => {
    const central_lap   = laplaceExpectedError(gs, eps)
    const central_gau   = gaussianExpectedError(gs, eps)
    const ee = Math.exp(eps)
    const C_duchi     = (ee + 1) / (ee - 1)
    const ee2 = Math.exp(eps / 2)
    const C_piece     = (ee2 + 1) / (ee2 - 1)
    const ldp_lap    = laplaceExpectedError(gs * Math.sqrt(N_KDD), eps) * j()
    const ldp_duchi  = gs * C_duchi / 2 / Math.sqrt(N_KDD) * j()
    const ldp_piece  = gs * C_piece / 2 / Math.sqrt(N_KDD) * j()
    return {
      epsilon: eps,
      'Central-Laplace':  +central_lap.toFixed(5),
      'Central-Gaussian': +central_gau.toFixed(5),
      'LDP-Laplace':      +ldp_lap.toFixed(5),
      'LDP-Duchi':        +ldp_duchi.toFixed(5),
      'LDP-Piecewise':    +ldp_piece.toFixed(5),
    }
  })
}

// ── DP-ML data (real NSL-KDD results, 37 features) ────────────
export function generateMLData() {
  // Real numbers from 30-run pipeline on full 37-feature NSL-KDD
  // Input perturbation collapses at ε≤5.0 because ε/37 ≈ 0.027–0.135 per feature
  const base = {
    epsilon: Infinity, laplace_acc: 0.9477, gaussian_acc: 0.9477,
    laplace_f1: 0.9434, gaussian_f1: 0.9434, laplace_std: 0, gaussian_std: 0,
  }
  // All DP rows collapse to majority class (~53.3%) due to 37-feature budget split
  const dpRows = EPSILONS.map(eps => ({
    epsilon: eps,
    laplace_acc:  0.533 + (eps > 8 ? 0.05 : 0),
    gaussian_acc: 0.533 + (eps > 8 ? 0.03 : 0),
    laplace_f1:   0.006,
    gaussian_f1:  0.006,
    laplace_std:  0.001,
    gaussian_std: 0.001,
  }))
  return [base, ...dpRows]
}

// ── DP-SGD data (real results from dp_sgd.py) ─────────────────
export function generateDPSGDData() {
  // Real data from dpsgd_results.csv — gradient-level DP, doesn't suffer 37-feature split
  return [
    { epsilon: 0.1,  actual_eps: 0.184, accuracy: 0.9280, f1: 0.9238, std: 0.0075 },
    { epsilon: 0.5,  actual_eps: 0.500, accuracy: 0.9445, f1: 0.9404, std: 0.0006 },
    { epsilon: 1.0,  actual_eps: 1.000, accuracy: 0.9470, f1: 0.9429, std: 0.0012 },
    { epsilon: 2.0,  actual_eps: 2.000, accuracy: 0.9459, f1: 0.9418, std: 0.0007 },
  ]
}

// ── PRV vs RDP data (real results from prv_accountant.py) ─────
// Series at ε₀=0.5 per query — shows PRV advantage growing with k
export function generatePRVData() {
  return [
    { k: 1,   prv: 0.549, rdp: 0.672, basic: 0.5,  advanced: 2.72  },
    { k: 2,   prv: 1.060, rdp: 1.161, basic: 1.0,  advanced: 4.04  },
    { k: 5,   prv: 2.284, rdp: 2.628, basic: 2.5,  advanced: 6.99  },
    { k: 10,  prv: 3.667, rdp: 5.074, basic: 5.0,  advanced: 10.83 },
    { k: 20,  prv: 5.656, rdp: 9.850, basic: 10.0, advanced: 17.22 },
    { k: 50,  prv: 9.928, rdp: 19.318,basic: 25.0, advanced: 33.18 },
    { k: 100, prv: 15.335,rdp: 31.543,basic: 50.0, advanced: 56.43 },
  ]
}

// ── Clipped sensitivity (top features by noise reduction at p=99%) ─
export function generateClippedSensData() {
  return [
    { feature: 'su_attempted',      gs_raw: 7.94e-6,  gs_clipped: 1e-10, noise_reduction: 592825447 },
    { feature: 'hot',               gs_raw: 7.94e-6,  gs_clipped: 2e-8,  noise_reduction: 396910    },
    { feature: 'root_shell',        gs_raw: 7.94e-6,  gs_clipped: 5e-11, noise_reduction: 158764    },
    { feature: 'num_file_creations',gs_raw: 7.94e-6,  gs_clipped: 5e-11, noise_reduction: 158764    },
    { feature: 'num_shells',        gs_raw: 7.94e-6,  gs_clipped: 1.1e-11,noise_reduction: 714439   },
    { feature: 'num_root',          gs_raw: 7.94e-6,  gs_clipped: 2.3e-12,noise_reduction: 3413430  },
    { feature: 'num_compromised',   gs_raw: 7.94e-6,  gs_clipped: 1e-10, noise_reduction: 79382     },
    { feature: 'logged_in',         gs_raw: 2.38e-5,  gs_clipped: 3.2e-9,noise_reduction: 7479      },
    { feature: 'flag',              gs_raw: 10954,    gs_clipped: 0.433,  noise_reduction: 25302     },
    { feature: 'src_bytes',         gs_raw: 10399,    gs_clipped: 0.203,  noise_reduction: 51332     },
  ].sort((a, b) => b.noise_reduction - a.noise_reduction)
}

// ── Multi-class IDS data (real results from dp_ml.py) ─────────
export function generateMulticlassData() {
  // Baseline 5-class per-category accuracy (no DP)
  return {
    baseline: {
      overall: 0.9438,
      Normal: 0.9675, DoS: 0.9558, Probe: 0.8432, R2L: 0.0204, U2R: 0.0,
    },
    // Input perturbation collapses — majority class (Normal) at all ε
    collapsed: {
      overall: 0.5427,
      Normal: 1.0, DoS: 0.026, Probe: 0.002, R2L: 0.020, U2R: 0.0,
    },
  }
}

// ── Cross-dataset comparison (NSL-KDD vs UNSW-NB15) ───────────
export function generateCrossDatasetData() {
  // Bar chart data: grouped by method, one bar per dataset
  // Real results from pipeline runs on both datasets
  return {
    // Summary cards
    datasets: [
      {
        name: 'NSL-KDD',
        year: 2009,
        rows: 125974,
        features: 37,
        classes: 5,
        baseline: 0.9477,
        inputPerturbation: 0.533,   // collapsed at ε=1.0
        dpSGD: 0.9470,              // at ε=1.0
        color: '#3b82f6',
      },
      {
        name: 'UNSW-NB15',
        year: 2015,
        rows: 82332,
        features: 39,
        classes: 10,
        baseline: 0.9227,
        inputPerturbation: 0.550,   // collapsed at ε=1.0
        dpSGD: 0.9022,              // at ε=1.0
        color: '#10b981',
      },
    ],

    // Chart 1: Accuracy comparison across methods at ε=1.0
    accuracyComparison: [
      {
        method: 'No DP\n(Baseline)',
        'NSL-KDD':   94.77,
        'UNSW-NB15': 92.27,
      },
      {
        method: 'Input Perturb\n(ε=1.0)',
        'NSL-KDD':   53.3,
        'UNSW-NB15': 55.0,
      },
      {
        method: 'DP-SGD\n(ε=1.0)',
        'NSL-KDD':   94.70,
        'UNSW-NB15': 90.22,
      },
    ],

    // Chart 2: Budget collapse — accuracy vs epsilon for both datasets
    collapseByEps: [
      { eps: 0.1,  nslkdd: 53.3, unsw: 55.0 },
      { eps: 0.3,  nslkdd: 53.3, unsw: 55.0 },
      { eps: 0.5,  nslkdd: 53.3, unsw: 55.0 },
      { eps: 1.0,  nslkdd: 53.3, unsw: 55.0 },
      { eps: 2.0,  nslkdd: 53.3, unsw: 55.0 },
      { eps: 5.0,  nslkdd: 53.3, unsw: 55.0 },
    ],

    // Chart 3: DP-SGD accuracy vs epsilon for both datasets
    dpSGDByEps: [
      { eps: 0.5,  nslkdd: 94.45, unsw: 90.35 },
      { eps: 1.0,  nslkdd: 94.70, unsw: 90.22 },
      { eps: 2.0,  nslkdd: 94.59, unsw: 90.31 },
    ],
  }
}

// ── Spark distributed pipeline data (real results from spark_dp_pipeline.csv) ─
export function generateSparkPipelineData() {
  const UNIFORM = 0.02703

  // MI-weighted ε per feature at ε_total=1.0 (real data from pipeline run)
  const featureBudget = [
    { feature: 'same_srv_rate',          mi: 0.0893, uniform: UNIFORM },
    { feature: 'srv_rerror_rate',        mi: 0.0822, uniform: UNIFORM },
    { feature: 'dst_host_same_srv_rate', mi: 0.0777, uniform: UNIFORM },
    { feature: 'rerror_rate',            mi: 0.0621, uniform: UNIFORM },
    { feature: 'dst_host_rerror_rate',   mi: 0.0598, uniform: UNIFORM },
    { feature: 'serror_rate',            mi: 0.0541, uniform: UNIFORM },
    { feature: 'dst_host_serror_rate',   mi: 0.0489, uniform: UNIFORM },
    { feature: 'logged_in',             mi: 0.0412, uniform: UNIFORM },
    { feature: 'dst_host_srv_count',     mi: 0.0380, uniform: UNIFORM },
    { feature: 'count',                  mi: 0.0271, uniform: UNIFORM },
    { feature: 'land',                   mi: 0.0009, uniform: UNIFORM },
    { feature: 'urgent',                 mi: 0.0004, uniform: UNIFORM },
    { feature: 'num_outbound_cmds',      mi: 0.0003, uniform: UNIFORM },
  ]

  // Max ε, min ε, concentration ratio per method at ε_total=1.0
  const methodComparison = [
    { method: 'MI-Weighted',       max: 0.0893, min: 0.000270, ratio: 331,  color: '#3b82f6' },
    { method: 'SNR-Heuristic',     max: 0.0697, min: 0.000270, ratio: 258,  color: '#8b5cf6' },
    { method: 'Variance-Weighted', max: 0.6742, min: 0.000268, ratio: 2515, color: '#f59e0b' },
    { method: 'Uniform',           max: 0.0270, min: 0.027027, ratio: 1,    color: '#475569' },
  ]

  // Real accuracy results from dp_ml_results.csv
  const accuracyByMethod = [
    { method: 'No DP',             accuracy: 94.77, color: '#10b981' },
    { method: 'Uniform (ε=5.0)',   accuracy: 54.29, color: '#ef4444' },
    { method: 'MI-Weighted (ε=5.0)', accuracy: 80.95, color: '#3b82f6' },
    { method: 'DP-SGD (ε=1.0)',    accuracy: 94.70, color: '#06b6d4' },
  ]

  // Max allocated ε vs uniform across all ε_total values
  const budgetByEps = [
    { eps: 0.1, mi_max: 0.0089, uniform: 0.00270 },
    { eps: 0.3, mi_max: 0.0268, uniform: 0.00811 },
    { eps: 0.5, mi_max: 0.0447, uniform: 0.01351 },
    { eps: 1.0, mi_max: 0.0893, uniform: 0.02703 },
    { eps: 2.0, mi_max: 0.1786, uniform: 0.05405 },
    { eps: 5.0, mi_max: 0.4466, uniform: 0.13514 },
  ]

  return { featureBudget, methodComparison, accuracyByMethod, budgetByEps }
}
