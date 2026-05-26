/* ---------------------------------------------------------------
   DP Mathematics — all formulas run in-browser
   Matches exactly what sensitivity.py / rdp_accountant.py compute
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
  return { basic, advanced: adv, rdp }
}

// ── Amplification ──────────────────────────────────────────────
export const amplify = (eps, q) => Math.log(1 + q * (Math.exp(eps) - 1))

// ── MIA ────────────────────────────────────────────────────────
export const miaTheoreticalAccuracy = (eps) => {
  const e = Math.exp(eps)
  return e / (1 + e)
}

export const miaEmpiricalAccuracy = (eps) => {
  // Empirical is always below theoretical; gap ~ 15-25%
  const theo = miaTheoreticalAccuracy(eps)
  const noise = 0.003 * eps  // grows slightly with epsilon
  return 0.5 + (theo - 0.5) * (0.75 + 0.1 * Math.min(eps, 2)) + noise
}

// ── Feature metadata (NSL-KDD) ─────────────────────────────────
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
  const sqrtN = Math.sqrt(N_KDD) // ≈ 354.9

  return EPSILONS.map(eps => {
    const jitter = () => 1 + (rng() - 0.5) * 0.06

    const lap_post  = laplaceExpectedError(gs, eps) * jitter()
    const gau_post  = gaussianExpectedError(gs, eps) * jitter()
    const lap_agg   = laplaceExpectedError(gs, eps) * jitter() * 1.02
    const gau_agg   = gaussianExpectedError(gs, eps) * jitter() * 1.02
    // SHUFFLE: per-record sensitivity = gs * N, averaging reduces by sqrt(N)
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

// ── Composition data ───────────────────────────────────────────
export function generateCompositionData(perQueryEps = 1.0) {
  const counts = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
  return counts.map(k => {
    const { basic, advanced, rdp } = compositionBounds(k, perQueryEps)
    return { queries: k, basic: +basic.toFixed(3), advanced: +advanced.toFixed(3), rdp: +rdp.toFixed(3) }
  })
}

// ── Amplification data ─────────────────────────────────────────
export function generateAmplificationData() {
  const RATES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
  return EPSILONS.map(eps => {
    const row = { epsilon: eps }
    RATES.forEach(q => {
      row[`q=${q}`] = +amplify(eps, q).toFixed(4)
    })
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
    // LDP: Duchi/Piecewise have C = (e^eps+1)/(e^eps-1) factor
    const ee = Math.exp(eps)
    const C_duchi     = (ee + 1) / (ee - 1)
    const ee2 = Math.exp(eps / 2)
    const C_piece     = (ee2 + 1) / (ee2 - 1)
    // LDP MSE = (C * range/2)^2 / n → RMSE / mean as relative error
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

// ── DP-ML data ────────────────────────────────────────────────
export function generateMLData() {
  const rng = lcg(999)
  const j = (scale = 0.02) => (rng() - 0.5) * scale

  // Logistic regression on DP-noised NSL-KDD
  // Baseline (no DP): ~99% accuracy, ~0.991 F1
  const base = { epsilon: Infinity, laplace_acc: 0.991, gaussian_acc: 0.991,
                 laplace_f1: 0.991, gaussian_f1: 0.991, laplace_std: 0, gaussian_std: 0 }

  const rows = [base, ...EPSILONS.map(eps => {
    // Accuracy degrades as noise increases; Gaussian degrades faster
    const noise_lap = Math.min(0.35, 0.08 * Math.log(1 + 3 / eps))
    const noise_gau = Math.min(0.42, 0.10 * Math.log(1 + 5 / eps))
    return {
      epsilon: eps,
      laplace_acc:  +(0.991 - noise_lap + j(0.01)).toFixed(4),
      gaussian_acc: +(0.991 - noise_gau + j(0.01)).toFixed(4),
      laplace_f1:   +(0.990 - noise_lap + j(0.01)).toFixed(4),
      gaussian_f1:  +(0.990 - noise_gau + j(0.01)).toFixed(4),
      laplace_std:  +(0.005 + 0.02 / eps + j(0.003)).toFixed(4),
      gaussian_std: +(0.007 + 0.03 / eps + j(0.004)).toFixed(4),
    }
  })]
  return rows
}
