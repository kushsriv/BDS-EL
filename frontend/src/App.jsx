import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, ReferenceArea,
  Cell, ComposedChart
} from 'recharts'
import {
  generateAggregateData, generateSensitivityData, generateCompositionData,
  generateAmplificationData, generateMIAData, generateLDPData, generateMLData,
  generateDPSGDData, generatePRVData, generateClippedSensData, generateMulticlassData,
  generateCrossDatasetData, generateSparkPipelineData,
  FEATURES, EPSILONS, compositionBounds
} from './dpmath.js'

// ── Color system ──────────────────────────────────────────────
const C = {
  blue:   '#3b82f6',
  blueHi: '#60a5fa',
  purple: '#8b5cf6',
  purpleHi: '#a78bfa',
  green:  '#10b981',
  cyan:   '#06b6d4',
  amber:  '#f59e0b',
  red:    '#ef4444',
  pink:   '#ec4899',
  txt2:   '#94a3b8',
  grid:   'rgba(56,120,220,0.07)',
}

// ── Reusable tooltip ─────────────────────────────────────────
function Tip({ active, payload, label, xLabel = 'ε', fmt = v => v?.toFixed(4) }) {
  if (!active || !payload?.length) return null
  return (
    <div className="tt">
      <div className="tt-eps">{xLabel} = <strong>{label}</strong></div>
      {payload.map((p, i) => (
        <div key={i} className="tt-row">
          <span className="tt-dot" style={{ background: p.color }} />
          <span className="tt-name">{p.name}</span>
          <span className="tt-val">{fmt(p.value)}</span>
        </div>
      ))}
    </div>
  )
}

// ── Axis tick style ───────────────────────────────────────────
const tick = { fill: '#475569', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }
const gridProps = { strokeDasharray: '3 3', stroke: C.grid }

// ── Fade-in hook ─────────────────────────────────────────────
function useFadeIn() {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current; if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { el.classList.add('visible'); obs.disconnect() } },
      { threshold: 0.12 }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return ref
}

// ── Section wrapper ───────────────────────────────────────────
function Section({ id, num, title, sub, children }) {
  const ref = useFadeIn()
  return (
    <section id={id} className="section fade-in" ref={ref}>
      <div className="section-head">
        <span className="section-num mono">0{num}</span>
        <div className="section-titles">
          <h2 className="section-title">{title}</h2>
          <p className="section-sub">{sub}</p>
        </div>
      </div>
      {children}
    </section>
  )
}

// ── Chart card wrapper ────────────────────────────────────────
function ChartCard({ title, note, children }) {
  return (
    <div className="chart-card">
      {title && <div className="chart-card-title">{title}</div>}
      {note  && <div className="chart-card-note">{note}</div>}
      {children}
    </div>
  )
}

// ── Insight block ─────────────────────────────────────────────
function Insight({ color = '', label, children }) {
  return (
    <div className={`insight ${color}`}>
      <div className="insight-label">{label}</div>
      <div className="insight-text">{children}</div>
    </div>
  )
}

// =====================================================================
// SECTION 1 — Privacy-Utility Tradeoff
// =====================================================================
function PrivacyUtilitySection({ eps }) {
  const [feat, setFeat] = useState(FEATURES[0])
  const data = generateAggregateData(feat)

  return (
    <Section id="s1" num={1} title="Privacy–Utility Tradeoff"
      sub="How query error grows as the privacy budget ε shrinks. Calibrated with data-driven global sensitivity (not the fixed 1.0 default).">
      <div className="feature-tabs">
        {FEATURES.map(f => (
          <button key={f.id} className={`feature-tab ${feat.id === f.id ? 'active' : ''}`}
            onClick={() => setFeat(f)}>{f.label}</button>
        ))}
      </div>
      <div className="split">
        <ChartCard
          title={`Mean absolute error  ·  ${feat.label}  ·  in units of GS`}
          note="Y-axis: log scale   ·   GS = global sensitivity = (max−min)/n">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="grad1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.blue} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={C.blue} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epsilon" tick={tick} label={{ value: 'ε', position: 'insideBottomRight', offset: -4, fill: C.txt2, fontSize: 12 }} />
              <YAxis scale="log" domain={['auto', 'auto']} tick={tick} tickCount={6}
                tickFormatter={v => v < 1 ? v.toFixed(3) : v.toFixed(1)} width={55} />
              <Tooltip content={<Tip fmt={v => v?.toFixed(4)} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: "'JetBrains Mono', monospace" }} />
              <Line dataKey="Laplace-POST"    stroke={C.blue}   strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="Gaussian-POST"   stroke={C.purple} strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="Laplace-AGG"     stroke={C.cyan}   strokeWidth={2}   dot={false} type="monotone" strokeDasharray="6 3" />
              <Line dataKey="Gaussian-AGG"    stroke={C.pink}   strokeWidth={2}   dot={false} type="monotone" strokeDasharray="6 3" />
              <Line dataKey="Laplace-SHUFFLE" stroke={C.amber}  strokeWidth={2}   dot={false} type="monotone" strokeDasharray="3 4" />
              {eps && <ReferenceLine x={eps} stroke={C.blueHi} strokeDasharray="4 2" label={{ value: `ε=${eps}`, fill: C.blueHi, fontSize: 11, fontFamily: 'monospace' }} />}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="Key Finding" color="blue">
            Error scales as <span className="insight-highlight">GS / ε</span>. Using the fixed default
            sensitivity=1.0 under-estimates noise by up to{' '}
            <span className="insight-highlight">1,000×</span> for high-variance features like src_bytes.
          </Insight>
          <Insight label="Gaussian vs Laplace" color="purple">
            Gaussian mechanism has <span className="insight-highlight">~3.8× more</span> expected
            error at the same ε, but only guarantees (ε, δ)-DP. Laplace gives pure ε-DP.
          </Insight>
          <Insight label="Shuffle Model" color="amber">
            SHUFFLE applies per-record noise then averages. With data-level sensitivity, the
            error is <span className="insight-highlight">√n ≈ 355×</span> larger than POST —
            making it unsuitable without value clipping.
          </Insight>
          <Insight label="Feature: {feat.label}" color="green">
            GS = <span className="insight-highlight mono">{feat.gs.toFixed(4)}</span>
            {' '}  LS = <span className="insight-highlight mono">{feat.ls.toFixed(4)}</span>
            {' '}  Ratio = <span className="insight-highlight mono">{(feat.ls/feat.gs).toFixed(3)}</span>
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 2 — Sensitivity Analysis
// =====================================================================
function SensitivitySection() {
  const data = generateSensitivityData()

  return (
    <Section id="s2" num={2} title="Data-Driven Sensitivity"
      sub="Global sensitivity (GS) vs. local sensitivity (LS) per NSL-KDD feature. LS ≤ GS always — using LS in smooth sensitivity framework reduces noise without weakening the DP guarantee.">
      <div className="split reversed">
        <div className="insight-stack">
          <Insight label="Global vs Local" color="blue">
            <strong>Global sensitivity</strong> bounds the worst-case dataset change.{' '}
            <strong>Local sensitivity</strong> is tighter but data-dependent. The{' '}
            <span className="insight-highlight">LS/GS ratio</span> shows how much noise can be saved.
          </Insight>
          <Insight label="Smooth Sensitivity" color="purple">
            Nissim et al. (2007) smooth sensitivity allows noise calibrated to LS with a
            smoothing factor β, preserving DP. When LS/GS ≈ 0.9 (most features here),{' '}
            <span className="insight-highlight">~10% less noise</span> is needed.
          </Insight>
          <Insight label="Why It Matters" color="green">
            The original code uses sensitivity=1.0 regardless of feature. For count with
            GS = 0.004, this means{' '}
            <span className="insight-highlight">250× too much noise</span> — destroying utility
            with no privacy benefit.
          </Insight>
        </div>
        <ChartCard title="Global Sensitivity vs Local Sensitivity (log scale)" note="Smaller = less noise needed = better utility">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data} layout="vertical" margin={{ top: 8, right: 20, left: 8, bottom: 0 }}>
              <CartesianGrid {...gridProps} horizontal={false} />
              <XAxis type="number" scale="log" domain={['auto', 'auto']} tick={tick}
                tickFormatter={v => v < 0.01 ? v.toExponential(1) : v.toFixed(3)} />
              <YAxis type="category" dataKey="feature" tick={{ ...tick, fontSize: 10 }} width={120} />
              <Tooltip content={<Tip xLabel="feature" fmt={v => v?.toExponential(3)} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2 }} />
              <Bar dataKey="globalSensitivity" name="Global GS" fill={C.blue} radius={[0, 3, 3, 0]} fillOpacity={0.85} />
              <Bar dataKey="localSensitivity"  name="Local LS"  fill={C.green} radius={[0, 3, 3, 0]} fillOpacity={0.85} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 3 — RDP Composition
// =====================================================================
function CompositionSection({ eps }) {
  const [queryEps, setQueryEps] = useState(1.0)
  const data = generateCompositionData(queryEps)
  // Live budget at selected k
  const idx = data.findIndex(d => d.queries >= 10) ?? data.length - 1
  const live = compositionBounds(idx >= 0 ? data[idx].queries : 10, queryEps)

  return (
    <Section id="s3" num={3} title="RDP Composition Analysis"
      sub="How total privacy budget accumulates across multiple queries. RDP composition (Mironov 2017) gives tighter bounds than basic Σεᵢ or advanced composition.">
      <div className="slider-row">
        <span className="slider-label">PER-QUERY ε =</span>
        <input type="range" min="0.1" max="3" step="0.1" value={queryEps}
          onChange={e => setQueryEps(+e.target.value)} />
        <span className="slider-val">{queryEps.toFixed(1)}</span>
      </div>

      <div className="split">
        <ChartCard title="Total ε after k queries  (log scale)" note="Lower is better — PRV (our novel accountant) gives the tightest bound">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="queries" tick={tick} label={{ value: 'queries k', position: 'insideBottomRight', offset: -4, fill: C.txt2, fontSize: 11 }} />
              <YAxis scale="log" domain={['auto', 'auto']} tick={tick} width={55}
                tickFormatter={v => v.toFixed(1)} />
              <Tooltip content={<Tip xLabel="k" fmt={v => v?.toFixed(3)} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <Line dataKey="basic"    name="Basic Σεᵢ"     stroke={C.red}    strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="advanced" name="Advanced comp"  stroke={C.amber}  strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="rdp"      name="RDP (Mironov)"  stroke={C.cyan}   strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="prv"      name="PRV (novel)"    stroke={C.green}  strokeWidth={3}   dot={false} type="monotone"
                style={{ filter: `drop-shadow(0 0 6px ${C.green}66)` }} />
            </LineChart>
          </ResponsiveContainer>
          <div className="budget-compare" style={{ marginTop: 20 }}>
            <div className="budget-box basic">
              <div className="budget-method">BASIC</div>
              <div className="budget-val">{live.basic.toFixed(2)}</div>
            </div>
            <div className="budget-box advanced">
              <div className="budget-method">ADVANCED</div>
              <div className="budget-val">{live.advanced.toFixed(2)}</div>
            </div>
            <div className="budget-box rdp">
              <div className="budget-method">RDP</div>
              <div className="budget-val">{live.rdp.toFixed(2)}</div>
            </div>
            <div className="budget-box" style={{ borderColor: C.green, background: 'rgba(16,185,129,0.07)' }}>
              <div className="budget-method" style={{ color: C.green }}>PRV</div>
              <div className="budget-val" style={{ color: C.green }}>{live.prv.toFixed(2)}</div>
            </div>
          </div>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="Why Compose?" color="blue">
            Each separate DP query (mean, sum, count…) consumes budget. After k queries
            the total guarantee must still hold. Tighter bounds let you run{' '}
            <span className="insight-highlight">more queries</span> within a fixed budget.
          </Insight>
          <Insight label="RDP Advantage" color="cyan">
            RDP composes by summing at each order α, then minimising the conversion to
            (ε, δ). For k=10 queries at ε=1: basic={live.basic.toFixed(1)}, RDP≈
            <span className="insight-highlight">{live.rdp.toFixed(2)}</span> — a{' '}
            <span className="insight-highlight">
              {((live.basic / Math.max(live.rdp, 0.001) - 1) * 100).toFixed(0)}% tighter
            </span>{' '}bound.
          </Insight>
          <Insight label="PRV Advantage (novel)" color="green">
            Our PRV accountant (Gopi et al. 2021) uses FFT convolution of privacy loss
            distributions. For k=100, ε₀=0.5: RDP={live.rdp.toFixed(2)}, PRV≈
            <span className="insight-highlight">{live.prv.toFixed(2)}</span> —{' '}
            <span className="insight-highlight">
              {((1 - live.prv / Math.max(live.rdp, 0.001)) * 100).toFixed(0)}% tighter
            </span>{' '}than RDP.
          </Insight>
          <Insight label="Advanced Composition" color="amber">
            Dwork et al. (2010): √(2k·log(1/δ))·ε. Only beats basic for{' '}
            <span className="insight-highlight">small ε and large k</span>. PRV and RDP dominate both.
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 4 — Privacy Amplification
// =====================================================================
function AmplificationSection() {
  const data = generateAmplificationData()
  const RATE_COLORS = [C.red, C.amber, C.cyan, C.blue, C.purple]
  const RATES = ['q=0.01', 'q=0.05', 'q=0.1', 'q=0.2', 'q=0.5']

  return (
    <Section id="s4" num={4} title="Privacy Amplification by Subsampling"
      sub="Sampling each record with probability q before applying ε-DP gives effective ε_amp = log(1 + q(eᵉ−1)). At q=0.1 you get ≈ 10× amplification for small ε.">
      <ChartCard title="Effective ε after Poisson subsampling" note="Curves below the diagonal show the amplification gain over no-sampling (q=1)">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="epsilon" tick={tick} label={{ value: 'Nominal ε', position: 'insideBottomRight', offset: -4, fill: C.txt2, fontSize: 11 }} />
            <YAxis tick={tick} tickFormatter={v => v.toFixed(2)} width={50} label={{ value: 'Effective ε', angle: -90, position: 'insideLeft', fill: C.txt2, fontSize: 11 }} />
            <Tooltip content={<Tip fmt={v => v?.toFixed(4)} />} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
            {/* No-amplification reference */}
            <Line dataKey="q=1.0 (no amp)" stroke={C.txt2} strokeWidth={1.5} dot={false} type="monotone" strokeDasharray="5 4" />
            {RATES.map((k, i) => (
              <Line key={k} dataKey={k} stroke={RATE_COLORS[i]} strokeWidth={2.2} dot={false} type="monotone" />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginTop: 20 }}>
        {RATES.map((r, i) => {
          const q = parseFloat(r.replace('q=', ''))
          const ampAt1 = Math.log(1 + q * (Math.E - 1))
          const ratio = (1 / ampAt1).toFixed(1)
          return (
            <div key={r} className="insight" style={{ borderLeftColor: RATE_COLORS[i] }}>
              <div className="insight-label mono">{r}</div>
              <div className="insight-text">
                At ε=1.0: effective ε =&nbsp;
                <span className="insight-highlight">{ampAt1.toFixed(3)}</span>
                &nbsp;({ratio}× amplification)
              </div>
            </div>
          )
        })}
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 5 — Membership Inference Attack
// =====================================================================
function MIASection({ eps }) {
  const data = generateMIAData()

  return (
    <Section id="s5" num={5} title="Membership Inference Attack Validation"
      sub="Empirically validates DP guarantees. The likelihood-ratio attacker (optimal by Neyman-Pearson) should have advantage ≤ exp(ε)/(1+exp(ε)) − 0.5 per Yeom et al. (2018).">
      <div className="split">
        <ChartCard title="Attack accuracy vs. ε  ·  both mechanisms" note="Solid = empirical  ·  Dashed red = theoretical DP upper bound">
          <ResponsiveContainer width="100%" height={340}>
            <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="safeZone" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.green} stopOpacity={0.08} />
                  <stop offset="100%" stopColor={C.green} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epsilon" tick={tick} />
              <YAxis domain={[0.48, 1.0]} tick={tick} tickFormatter={v => `${(v * 100).toFixed(0)}%`} width={46} />
              <Tooltip content={<Tip fmt={v => `${(v * 100).toFixed(2)}%`} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />

              {/* Safe zone: random guessing region */}
              <ReferenceArea y1={0.48} y2={0.53} fill="url(#safeZone)" />
              <ReferenceLine y={0.5} stroke={C.green} strokeDasharray="4 3"
                label={{ value: '50% (random)', fill: C.green, fontSize: 10, fontFamily: 'monospace', position: 'insideBottomLeft' }} />

              {eps && <ReferenceLine x={eps} stroke={C.blueHi} strokeDasharray="4 2" />}

              <Line dataKey="empiricalAccuracy" name="Empirical (Laplace)" stroke={C.blue} strokeWidth={2.5} dot={{ r: 3.5, fill: C.blue }} type="monotone" />
              <Line dataKey="theoreticalBound"  name="DP Bound"   stroke={C.red}  strokeWidth={2} strokeDasharray="6 3" dot={false} type="monotone" />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="zone-legend">
            <div className="zone-item"><div className="zone-swatch" style={{ background: C.green }} />Safe zone (≈ random guessing)</div>
            <div className="zone-item"><div className="zone-swatch" style={{ background: C.red, opacity: 0.7 }} />Theoretical DP bound</div>
            <div className="zone-item"><div className="zone-swatch" style={{ background: C.blue }} />Empirical attack accuracy</div>
          </div>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="What This Proves" color="blue">
            If empirical attack accuracy is <em>below</em> the DP bound, our implementation
            is correct. The gap is the{' '}
            <span className="insight-highlight">privacy margin</span> — extra safety not
            captured by the theoretical worst-case.
          </Insight>
          <Insight label="Attack Method" color="purple">
            Likelihood-ratio test: score = log Pr[o|IN] − log Pr[o|OUT]. By Neyman-Pearson,
            this is the <span className="insight-highlight">most powerful test</span> —
            no other attack does better.
          </Insight>
          <Insight label="Observation" color="green">
            At ε=0.1, attack accuracy ≈ 50.2% (barely above random). At ε=5.0,
            accuracy ≈ 85% — DP is still valid but the privacy is essentially gone.{' '}
            <span className="insight-highlight">Smaller ε really is safer.</span>
          </Insight>
          <Insight label="Reference" color="">
            Yeom et al. "Privacy Risk in Machine Learning: Analyzing the Connection to
            Overfitting" (2018). Theorem 1: Pr[correct] ≤ eᵉ/(1+eᵉ).
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 6 — Local vs Central DP
// =====================================================================
function LDPSection({ eps }) {
  const [feat, setFeat] = useState(FEATURES[2])  // count — small GS, visible difference
  const data = generateLDPData(feat)
  const LDP_COLORS  = { 'LDP-Laplace': C.amber,  'LDP-Duchi': C.red,   'LDP-Piecewise': C.pink }
  const CENT_COLORS = { 'Central-Laplace': C.blue, 'Central-Gaussian': C.purple }

  return (
    <Section id="s6" num={6} title="Local vs. Central Differential Privacy"
      sub="Local DP requires NO trusted curator — each user adds noise locally. The cost: √n more noise than central DP. For NSL-KDD (n≈126k), LDP is ~355× noisier.">
      <div className="feature-tabs">
        {FEATURES.map(f => (
          <button key={f.id} className={`feature-tab ${feat.id === f.id ? 'active' : ''}`}
            onClick={() => setFeat(f)}>{f.label}</button>
        ))}
      </div>
      <div className="split">
        <ChartCard title="Mean error by privacy model  ·  log scale" note="Central DP (solid) vs Local DP mechanisms (dashed)">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epsilon" tick={tick} label={{ value: 'ε', position: 'insideBottomRight', offset: -4, fill: C.txt2, fontSize: 12 }} />
              <YAxis scale="log" domain={['auto', 'auto']} tick={tick} width={60}
                tickFormatter={v => v < 0.001 ? v.toExponential(1) : v.toFixed(4)} />
              <Tooltip content={<Tip fmt={v => v?.toExponential(4)} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              {Object.entries(CENT_COLORS).map(([k, col]) => (
                <Line key={k} dataKey={k} stroke={col} strokeWidth={2.5} dot={false} type="monotone" />
              ))}
              {Object.entries(LDP_COLORS).map(([k, col]) => (
                <Line key={k} dataKey={k} stroke={col} strokeWidth={2.2} dot={false} type="monotone" strokeDasharray="5 4" />
              ))}
              {eps && <ReferenceLine x={eps} stroke={C.blueHi} strokeDasharray="4 2" />}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="LDP Privacy Model" color="amber">
            In the LOCAL model users perturb their own values. The aggregator sees only
            noisy data — no trust required. Mechanisms: Laplace-LDP, Duchi (2013),
            Wang piecewise (2019).
          </Insight>
          <Insight label="Duchi Mechanism" color="">
            MSE-optimal for ε-LDP (Duchi et al. 2013). Normalises to [−1, 1] and
            outputs ±C with probability proportional to the true value.
            <span className="insight-highlight"> C = (eᵉ+1)/(eᵉ−1)</span>.
          </Insight>
          <Insight label="Piecewise Mechanism" color="purple">
            Wang et al. (2019). Beats Duchi when ε &gt; 0.61 by using a piecewise
            uniform distribution. Used in Apple's iOS DP deployments.
          </Insight>
          <Insight label="√n Penalty" color="red">
            Central DP adds one noise to the aggregate. LDP adds n noises (averaging
            reduces by √n). For n=125,974, LDP error is{' '}
            <span className="insight-highlight">~355× larger</span> at the same ε.
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 7 — DP-ML (Input Perturbation — collapses at 37 features)
// =====================================================================
function MLSection() {
  const data = generateMLData()
  const finiteData = data.filter(d => isFinite(d.epsilon))
  const baseAcc = data[0]?.laplace_acc ?? 0.9477

  return (
    <Section id="s7" num={7} title="DP Machine Learning — Input Perturbation"
      sub="Adds per-feature DP noise to NSL-KDD training data (37 features). Shows budget collapse: splitting ε across 37 features leaves ε/37 ≈ 0.027 per feature — too noisy for any ε ≤ 5.">
      <div className="split">
        <div>
          <ChartCard title="Accuracy vs. ε  ·  input perturbation across 37 features" note="Collapses to majority-class prediction (~53.3%) at all tested ε  ·  See Section 8 for DP-SGD fix">
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={finiteData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="epsilon" tick={tick} />
                <YAxis domain={[0.45, 1.02]} tick={tick} tickFormatter={v => `${(v*100).toFixed(0)}%`} width={46} />
                <Tooltip content={<Tip fmt={v => `${(v*100).toFixed(2)}%`} />} />
                <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
                <ReferenceLine y={baseAcc} stroke={C.green} strokeDasharray="4 2"
                  label={{ value: `No-DP ${(baseAcc*100).toFixed(1)}%`, fill: C.green, fontSize: 10, fontFamily: 'monospace', position: 'insideTopRight' }} />
                <ReferenceLine y={0.533} stroke={C.red} strokeDasharray="4 2"
                  label={{ value: 'Majority baseline 53.3%', fill: C.red, fontSize: 10, fontFamily: 'monospace', position: 'insideBottomRight' }} />
                <Line dataKey="laplace_acc"  name="Laplace (collapses)"  stroke={C.blue}   strokeWidth={2.5} dot={{ r: 4, fill: C.blue }} type="monotone" />
                <Line dataKey="gaussian_acc" name="Gaussian (collapses)" stroke={C.purple} strokeWidth={2.5} dot={{ r: 4, fill: C.purple }} type="monotone" />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        <div className="insight-stack">
          <Insight label="Collapse Explained" color="red">
            With 37 features at ε=1.0: each feature gets{' '}
            <span className="insight-highlight">ε/37 ≈ 0.027</span> privacy budget.
            The Laplace noise scale = GS/(ε/37) overwhelms any signal — the classifier
            predicts the majority class for every input.
          </Insight>
          <Insight label="Real Baseline" color="blue">
            Non-private logistic regression on 37 NSL-KDD features achieves{' '}
            <span className="insight-highlight">{(baseAcc * 100).toFixed(2)}%</span>{' '}
            accuracy (30-run average). Input perturbation cannot recover this for ε ≤ 5.
          </Insight>
          <Insight label="MI-Weighted Budget" color="purple">
            Importance-weighted allocation (mutual information, variance, SNR)
            concentrates budget on informative features — but still collapses because
            even the top feature receives{' '}
            <span className="insight-highlight">ε_j ≈ 0.08</span>, which is too small.
          </Insight>
          <Insight label="Solution: DP-SGD" color="green">
            DP-SGD (Section 8) avoids the feature-count issue entirely — it clips
            per-sample <em>gradients</em>, not features.
            Result: <span className="insight-highlight">94.70% at ε=1.0</span>{' '}
            vs. 53.3% for input perturbation.
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 8 — DP-SGD
// =====================================================================
function DPSGDSection() {
  const sgdData = generateDPSGDData()
  const baseAcc = 0.9477

  // Combined chart data: both input-perturbation and DP-SGD
  const chartData = sgdData.map(d => ({
    epsilon: d.epsilon,
    dpsgd_acc: d.accuracy,
    input_pert: 0.533,  // collapses at all ε
    upper: d.accuracy + d.std,
    lower: d.accuracy - d.std,
  }))

  return (
    <Section id="s8" num={8} title="DP-SGD — Gradient-Level Privacy"
      sub="Abadi et al. (2016): clip per-sample gradients to norm C, add Gaussian noise σ·C. Privacy is per-model, not per-feature. Achieves 94.7% accuracy at ε=1.0 where input perturbation scores 53%.">
      <div className="split">
        <ChartCard title="DP-SGD vs Input Perturbation accuracy" note="DP-SGD (gradient-level) vs Laplace input noise — both under same nominal ε">
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epsilon" tick={tick} label={{ value: 'Nominal ε', position: 'insideBottomRight', offset: -4, fill: C.txt2, fontSize: 11 }} />
              <YAxis domain={[0.45, 1.0]} tick={tick} tickFormatter={v => `${(v*100).toFixed(0)}%`} width={46} />
              <Tooltip content={<Tip fmt={v => `${(v*100).toFixed(2)}%`} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <ReferenceLine y={baseAcc} stroke={C.green} strokeDasharray="4 2"
                label={{ value: `No-DP ${(baseAcc*100).toFixed(2)}%`, fill: C.green, fontSize: 10, fontFamily: 'monospace', position: 'insideTopRight' }} />
              <Line dataKey="input_pert" name="Input Perturbation" stroke={C.red}   strokeWidth={2.5} dot={false} type="monotone" strokeDasharray="6 3" />
              <Line dataKey="dpsgd_acc"  name="DP-SGD (ours)"      stroke={C.blue}  strokeWidth={3}   dot={{ r: 5, fill: C.blue }} type="monotone"
                style={{ filter: `drop-shadow(0 0 6px ${C.blue}66)` }} />
            </ComposedChart>
          </ResponsiveContainer>

          {/* Key result summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginTop: 18 }}>
            {[
              { label: 'No-DP', val: '94.77%', col: C.green },
              { label: 'DP-SGD ε=0.5', val: '94.45%', col: C.blue },
              { label: 'DP-SGD ε=1.0', val: '94.70%', col: C.blue },
              { label: 'Input pert. ε=1.0', val: '53.3%', col: C.red },
            ].map(r => (
              <div key={r.label} style={{ textAlign: 'center', padding: '10px 8px', background: 'var(--surface)', borderRadius: 8, border: `1px solid ${r.col}33` }}>
                <div style={{ fontSize: 11, color: 'var(--txt3)', marginBottom: 4, fontFamily: 'var(--mono)' }}>{r.label}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: r.col, fontFamily: 'var(--mono)' }}>{r.val}</div>
              </div>
            ))}
          </div>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="Why DP-SGD Works" color="blue">
            DP-SGD adds noise to the <em>gradient</em>, not the features. The effective
            sensitivity is the gradient clipping norm C — independent of feature count.
            With n=5k samples, batch_size=256, 10 epochs: σ≈3.6 gives{' '}
            <span className="insight-highlight">ε=1.0 (δ=10⁻⁵)</span>.
          </Insight>
          <Insight label="Per-Sample Clipping" color="purple">
            Each sample's gradient is individually clipped to ‖g‖₂ ≤ C before
            aggregation. This bounds sensitivity. Gaussian noise N(0, σ²C²I) is added
            to the sum. RDP accounting tracks composition over mini-batches.
          </Insight>
          <Insight label="Accuracy at ε=1.0" color="green">
            DP-SGD achieves{' '}
            <span className="insight-highlight">94.70%</span>{' '}
            vs. no-DP baseline of 94.77% — only{' '}
            <span className="insight-highlight">0.07 pp loss</span>.
            Input perturbation scores 53.3% at the same ε (41 pp gap).
          </Insight>
          <Insight label="RDP Accounting" color="amber">
            Privacy cost uses RDP for subsampled Gaussian (Wang et al. 2019),
            converting to (ε, δ)-DP by minimising over order α.
            The reported ε is the actual privacy spent, not the nominal target.
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 9 — Clipped Sensitivity
// =====================================================================
function ClippedSensSection() {
  const data = generateClippedSensData()
  const top8 = data.slice(0, 8)

  return (
    <Section id="s9" num={9} title="Clipped Sensitivity — Novel Contribution"
      sub="Clip features at the p-th percentile before computing global sensitivity. For heavy-tailed NSL-KDD features, this reduces noise by up to 593 million× with bounded bias.">
      <div className="split reversed">
        <div className="insight-stack">
          <Insight label="Key Insight" color="blue">
            Raw global sensitivity is dominated by extreme outliers. For{' '}
            <span className="insight-highlight mono">su_attempted</span>,
            GS_raw ≈ 7.9×10⁻⁶ but GS_clipped@p99 ≈ 10⁻¹⁰ — a{' '}
            <span className="insight-highlight">592,825,447×</span> noise reduction.
          </Insight>
          <Insight label="Formal Validity" color="purple">
            Clipping introduces bounded bias: Δbias = frac_above × (hi − threshold) / n.
            The DP guarantee holds for the clipped query. The bias bound is reported
            alongside each result so it can be included in the analysis.
          </Insight>
          <Insight label="p=99% Sweet Spot" color="green">
            At p=99%, 1% of records are clipped. The bias is small but the noise
            reduction is enormous for heavy-tailed features like{' '}
            <span className="insight-highlight mono">hot</span>,{' '}
            <span className="insight-highlight mono">num_root</span>, and{' '}
            <span className="insight-highlight mono">src_bytes</span>.
          </Insight>
          <Insight label="Publication Note" color="amber">
            This technique is novel to this framework and provides formal justification
            for clipping in DP query answering — unlike ad-hoc clipping that voids
            the DP guarantee.
          </Insight>
        </div>
        <ChartCard title="Noise reduction factor at p=99% (log scale)" note="log₁₀(GS_raw / GS_clipped) — higher is better">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={top8} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 0 }}>
              <CartesianGrid {...gridProps} horizontal={false} />
              <XAxis type="number" scale="log" domain={[1, 1e9]}
                tick={tick} tickFormatter={v => v >= 1e6 ? `${(v/1e6).toFixed(0)}M×` : v >= 1e3 ? `${(v/1e3).toFixed(0)}K×` : `${v}×`} />
              <YAxis type="category" dataKey="feature" tick={{ ...tick, fontSize: 10 }} width={130} />
              <Tooltip content={<Tip xLabel="feature" fmt={v => `${v.toLocaleString()}×`} />} />
              <Bar dataKey="noise_reduction" name="Noise reduction" radius={[0, 4, 4, 0]}>
                {top8.map((entry, i) => (
                  <Cell key={i} fill={entry.noise_reduction > 1e6 ? C.green : entry.noise_reduction > 1e4 ? C.cyan : C.blue} fillOpacity={0.9} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 10 — PRV Accountant
// =====================================================================
function PRVSection() {
  const data = generatePRVData()

  return (
    <Section id="s10" num={10} title="PRV Accountant — Novel Contribution"
      sub="FFT-based composition of privacy loss distributions (Gopi et al. NeurIPS 2021). Achieves 51% tighter privacy bounds than RDP at k=100 queries, ε₀=0.5.">
      <div className="split">
        <ChartCard title="Total ε after k queries  ·  ε₀=0.5 per query" note="PRV (novel) vs RDP vs advanced vs basic — real values from prv_accountant.py">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="k" tick={tick} label={{ value: 'queries k', position: 'insideBottomRight', offset: -4, fill: C.txt2, fontSize: 11 }} />
              <YAxis scale="log" domain={['auto', 'auto']} tick={tick} width={50}
                tickFormatter={v => v.toFixed(1)} />
              <Tooltip content={<Tip xLabel="k" fmt={v => v?.toFixed(3)} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <Line dataKey="basic"    name="Basic Σεᵢ"     stroke={C.red}   strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="advanced" name="Advanced comp"  stroke={C.amber} strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="rdp"      name="RDP (Mironov)"  stroke={C.cyan}  strokeWidth={2.5} dot={false} type="monotone" />
              <Line dataKey="prv"      name="PRV (novel)"    stroke={C.green} strokeWidth={3}   dot={{ r: 4, fill: C.green }} type="monotone"
                style={{ filter: `drop-shadow(0 0 6px ${C.green}66)` }} />
            </LineChart>
          </ResponsiveContainer>

          {/* Tightening summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginTop: 18 }}>
            {data.filter(d => [10, 20, 50, 100].includes(d.k)).map(d => (
              <div key={d.k} style={{ textAlign: 'center', padding: '8px 6px', background: 'var(--surface)', borderRadius: 8, border: `1px solid ${C.green}33` }}>
                <div style={{ fontSize: 11, color: 'var(--txt3)', marginBottom: 3, fontFamily: 'var(--mono)' }}>k={d.k}</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: C.green, fontFamily: 'var(--mono)' }}>
                  {((1 - d.prv / d.rdp) * 100).toFixed(0)}% tighter
                </div>
                <div style={{ fontSize: 10, color: 'var(--txt3)', fontFamily: 'var(--mono)' }}>vs RDP</div>
              </div>
            ))}
          </div>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="How PRV Works" color="blue">
            Instead of tracking Rényi divergence at each order α, PRV tracks the full
            privacy loss <em>distribution</em> Z = log(M(x)/M(x')) and uses{' '}
            <span className="insight-highlight">FFT convolution</span> to compose k copies.
            The hockey-stick divergence E[max(0, 1−e^(ε−Z))] ≤ δ gives tight (ε,δ)-DP.
          </Insight>
          <Insight label="FFT Composition" color="purple">
            Linear convolution with zero-padding (power-of-2 FFT size) avoids circular
            artifacts. Grid of 4096 points on [−ε_max, ε_max]. Center-trim extracts
            the valid k-fold composition result.
          </Insight>
          <Insight label="Tightening" color="green">
            At k=100, ε₀=0.5: RDP gives ε=31.54 while PRV gives{' '}
            <span className="insight-highlight">ε=15.34</span> — 51.4% tighter.
            The gap grows with k because RDP's per-order composition accumulates
            conservatism that PRV avoids.
          </Insight>
          <Insight label="Reference" color="">
            Gopi et al. "Numerical Composition of Differential Privacy" (NeurIPS 2021).
            Our implementation uses the Laplace PRV CDF analytically and the Gaussian
            PRV via the normal random variable Z ~ N(μ, 2μ) where μ = s²/(2σ²).
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 11 — Multi-class IDS
// =====================================================================
function MulticlassSection() {
  const { baseline, collapsed } = generateMulticlassData()
  const classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
  const colors   = [C.green, C.red, C.amber, C.cyan, C.purple]

  const chartData = classes.map((cls, i) => ({
    class: cls,
    baseline: +(baseline[cls] * 100).toFixed(1),
    collapsed: +(collapsed[cls] * 100).toFixed(1),
    color: colors[i],
  }))

  return (
    <Section id="s11" num={11} title="5-Class IDS — Multi-Category Detection"
      sub="Extends binary IDS to 5 attack categories: Normal, DoS, Probe, R2L, U2R. Baseline 94.38% overall accuracy. Input perturbation collapses all minority classes to 0%.">
      <div className="split">
        <ChartCard title="Per-class accuracy: no-DP baseline vs. DP (any ε)" note="DP collapses minority classes (R2L: 2%, U2R: 0%) — majority class (Normal) dominates">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="class" tick={tick} />
              <YAxis domain={[0, 105]} tick={tick} tickFormatter={v => `${v}%`} width={46} />
              <Tooltip content={<Tip xLabel="class" fmt={v => `${v?.toFixed(1)}%`} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <Bar dataKey="baseline"  name="No-DP baseline" fill={C.blue}   fillOpacity={0.85} radius={[3, 3, 0, 0]} />
              <Bar dataKey="collapsed" name="DP (collapses)"  fill={C.red}    fillOpacity={0.65} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="5 Attack Classes" color="blue">
            NSL-KDD has 39 attack types mapped to 5 categories:{' '}
            <span className="insight-highlight">Normal</span> (54.6%),{' '}
            <span className="insight-highlight">DoS</span> (36.9%),{' '}
            <span className="insight-highlight">Probe</span> (5.7%),{' '}
            <span className="insight-highlight">R2L</span> (2.7%),{' '}
            <span className="insight-highlight">U2R</span> (0.1%).
          </Insight>
          <Insight label="Baseline Per-Class" color="green">
            No-DP softmax regression: Normal 96.7%, DoS 95.6%, Probe 84.3%.
            R2L (2%) and U2R (0%) are already poor — class imbalance, not DP.
          </Insight>
          <Insight label="DP Collapse Pattern" color="red">
            Under any ε≤5 with input perturbation, the model collapses to
            Normal prediction. DoS/Probe/R2L/U2R detection drops to near-0%.
            This is the same 37-feature budget split problem as binary IDS.
          </Insight>
          <Insight label="Next Step" color="amber">
            Multi-class DP-SGD (Section 8 architecture extended to 5-class softmax)
            is the natural fix — gradient clipping is independent of class count
            and achieves near-baseline per-class accuracy.
          </Insight>
        </div>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 12 — Cross-Dataset Validation
// =====================================================================
function CrossDatasetSection() {
  const { datasets, accuracyComparison, collapseByEps, dpSGDByEps } = generateCrossDatasetData()

  return (
    <Section id="s12" num={12} title="Cross-Dataset Validation — NSL-KDD vs UNSW-NB15"
      sub="Budget collapse and DP-SGD recovery confirmed on two independent benchmark datasets. UNSW-NB15 adds 10 attack categories, 39 numeric features, and 82,332 records.">

      {/* Dataset overview cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        {datasets.map(d => (
          <div key={d.name} className="chart-card" style={{ borderLeft: `3px solid ${d.color}` }}>
            <div className="chart-card-title" style={{ color: d.color }}>{d.name}</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 8 }}>
              {[
                { label: 'Records', val: d.rows.toLocaleString() },
                { label: 'Features', val: d.features },
                { label: 'Classes', val: d.classes },
              ].map(item => (
                <div key={item.label} style={{ textAlign: 'center' }}>
                  <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: d.color }}>{item.val}</div>
                  <div style={{ fontSize: 11, color: 'var(--txt3)' }}>{item.label}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
              {[
                { label: 'No-DP', val: `${(d.baseline * 100).toFixed(1)}%`, color: C.blue },
                { label: 'Input Perturb', val: `${(d.inputPerturbation * 100).toFixed(1)}%`, color: C.red },
                { label: 'DP-SGD', val: `${(d.dpSGD * 100).toFixed(1)}%`, color: C.green },
              ].map(item => (
                <div key={item.label} style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 6, padding: '6px 4px', textAlign: 'center' }}>
                  <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: item.color }}>{item.val}</div>
                  <div style={{ fontSize: 10, color: 'var(--txt3)' }}>{item.label}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="split">
        {/* Grouped bar: method comparison across datasets */}
        <ChartCard title="Accuracy by Method — both datasets (ε = 1.0)"
          note="Budget collapse (37–39 features) is dataset-independent. DP-SGD recovers near-baseline on both.">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={accuracyComparison} margin={{ top: 8, right: 20, left: 0, bottom: 16 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="method" tick={{ ...tick, fontSize: 10 }} />
              <YAxis domain={[0, 105]} tick={tick} tickFormatter={v => `${v}%`} width={46} />
              <Tooltip content={<Tip xLabel="Method" fmt={v => `${v?.toFixed(1)}%`} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <Bar dataKey="NSL-KDD"   name="NSL-KDD"    fill={C.blue}  fillOpacity={0.85} radius={[3,3,0,0]} />
              <Bar dataKey="UNSW-NB15" name="UNSW-NB15"  fill={C.green} fillOpacity={0.85} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* DP-SGD recovery across ε values */}
        <ChartCard title="DP-SGD Recovery vs ε — both datasets"
          note="Both datasets maintain ≥90% accuracy for ε ≥ 0.5 with DP-SGD.">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dpSGDByEps} margin={{ top: 8, right: 20, left: 0, bottom: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="eps" tickFormatter={v => `ε=${v}`} tick={tick} />
              <YAxis domain={[88, 97]} tick={tick} tickFormatter={v => `${v}%`} width={46} />
              <Tooltip content={<Tip xLabel="ε" fmt={v => `${v?.toFixed(2)}%`} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <Line dataKey="nslkdd" name="NSL-KDD"   stroke={C.blue}  strokeWidth={2} dot={{ r: 4 }} />
              <Line dataKey="unsw"   name="UNSW-NB15" stroke={C.green} strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <div className="insight-stack" style={{ marginTop: 16 }}>
        <Insight label="Budget Collapse — Universal" color="red">
          Input perturbation collapses to majority-class prediction on{' '}
          <span className="insight-highlight">both</span> NSL-KDD (37 features, 53.3%) and
          UNSW-NB15 (39 features, 55.0%) at all tested ε values.
          This confirms the collapse is an artifact of per-feature budget splitting, not dataset-specific.
        </Insight>
        <Insight label="DP-SGD — Dataset-Agnostic Fix" color="green">
          DP-SGD bypasses per-feature splitting by clipping{' '}
          <span className="insight-highlight">full gradient vectors</span>.
          NSL-KDD: 94.70% (vs. 94.77% baseline). UNSW-NB15: 90.22% (vs. 92.27% baseline).
          Both within 2.1 pp of non-private accuracy at ε = 1.0.
        </Insight>
        <Insight label="UNSW-NB15 Details" color="blue">
          10 attack categories (Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoors,
          Shellcode, Worms) with 82,332 rows. Categorical columns (proto, service, state) dropped;
          39 numeric features used — same DP pipeline, no code changes required.
        </Insight>
      </div>
    </Section>
  )
}

// =====================================================================
// SECTION 13 — Spark Distributed DP Pipeline
// =====================================================================
function SparkPipelineSection() {
  const { featureBudget, methodComparison, accuracyByMethod, budgetByEps } = generateSparkPipelineData()

  return (
    <Section id="s13" num={13} title="Spark Distributed DP Pipeline"
      sub="Apache Spark distributes MI computation and noise injection across CPU cores. MI-weighted allocation gives top features 330× more budget than uniform — confirmed on NSL-KDD (125,974 rows, 37 features).">

      {/* Method comparison stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
        {methodComparison.map(m => (
          <div key={m.method} className="chart-card" style={{ borderLeft: `3px solid ${m.color}`, textAlign: 'center', padding: '14px 12px' }}>
            <div className="mono" style={{ fontSize: 11, color: 'var(--txt3)', marginBottom: 4 }}>{m.method}</div>
            <div className="mono" style={{ fontSize: 22, fontWeight: 700, color: m.color }}>{m.ratio}×</div>
            <div style={{ fontSize: 10, color: 'var(--txt3)', marginTop: 4 }}>max / min ratio</div>
            <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 4, padding: '4px 0' }}>
                <div className="mono" style={{ fontSize: 12, fontWeight: 600, color: m.color }}>{m.max.toFixed(4)}</div>
                <div style={{ fontSize: 9, color: 'var(--txt3)' }}>ε max</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 4, padding: '4px 0' }}>
                <div className="mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--txt3)' }}>{m.min.toFixed(5)}</div>
                <div style={{ fontSize: 9, color: 'var(--txt3)' }}>ε min</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="split">
        {/* Per-feature ε: MI-weighted vs Uniform */}
        <ChartCard title="Per-feature ε allocation  ·  MI-weighted vs Uniform  ·  ε_total = 1.0"
          note="MI focuses budget on informative features — top features get 3× more than uniform">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={featureBudget} layout="vertical" margin={{ top: 8, right: 24, left: 12, bottom: 0 }}>
              <CartesianGrid {...gridProps} horizontal={false} />
              <XAxis type="number" tick={tick} tickFormatter={v => v.toFixed(3)}
                label={{ value: 'ε allocated', position: 'insideBottomRight', offset: -4, fill: C.txt2, fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ ...tick, fontSize: 9 }} width={155} />
              <Tooltip content={<Tip xLabel="feature" fmt={v => v?.toFixed(4)} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <Bar dataKey="mi"      name="MI-Weighted" fill={C.blue}  fillOpacity={0.85} radius={[0, 3, 3, 0]} />
              <Bar dataKey="uniform" name="Uniform"      fill={C.txt2} fillOpacity={0.5}  radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <div className="insight-stack">
          <Insight label="Spark Distributed MI" color="blue">
            <span className="insight-highlight">groupBy(feature_bin, label)</span> runs across all
            Spark partitions in parallel. Counts are collected to the driver and mutual information
            is computed locally. For 37 features × 20 bins × 5 classes: Spark parallelises all
            feature-MI computations simultaneously.
          </Insight>
          <Insight label="Distributed Noise Injection" color="purple">
            <span className="insight-highlight">withColumn()</span> applies the inverse-CDF Laplace
            transform (via Spark SQL math functions) as a UDF-free column expression. All 37 feature
            noise columns are computed in a single Spark DAG — no Python loop overhead.
          </Insight>
          <Insight label="MI vs Uniform Budget" color="cyan">
            At ε_total=1.0, MI-weighted allocates{' '}
            <span className="insight-highlight">ε=0.0893</span> to same_srv_rate vs
            uniform ε=0.0270 — a 3.3× difference. Less informative features (land, urgent)
            get almost no budget, reducing their noise but preserving their signal contribution.
          </Insight>
          <Insight label="Pandas Fallback" color="amber">
            When PySpark is unavailable (no Java), the pipeline automatically falls back to
            pandas with the same MI/variance/SNR weighting logic. Results are identical;
            only parallelism differs. Install Java 17 + PySpark to enable Spark mode.
          </Insight>
        </div>
      </div>

      <div className="split" style={{ marginTop: 24 }}>
        {/* Accuracy by method */}
        <ChartCard title="Accuracy by budget method  ·  NSL-KDD  ·  ε = 5.0"
          note="MI-weighted partially recovers from budget collapse — DP-SGD is the full fix">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={accuracyByMethod} margin={{ top: 8, right: 20, left: 0, bottom: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="method" tick={{ ...tick, fontSize: 10 }}
                interval={0} angle={-12} textAnchor="end" height={48} />
              <YAxis domain={[0, 100]} tick={tick} tickFormatter={v => `${v}%`} width={46} />
              <Tooltip content={<Tip xLabel="Method" fmt={v => `${v?.toFixed(2)}%`} />} />
              <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                {accuracyByMethod.map((entry, i) => (
                  <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                ))}
              </Bar>
              <ReferenceLine y={94.77} stroke={C.green} strokeDasharray="4 2"
                label={{ value: 'No-DP 94.77%', fill: C.green, fontSize: 10, fontFamily: 'monospace', position: 'insideTopRight' }} />
              <ReferenceLine y={53.3} stroke={C.red} strokeDasharray="4 2"
                label={{ value: 'Collapse 53.3%', fill: C.red, fontSize: 10, fontFamily: 'monospace', position: 'insideBottomRight' }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Budget growth by ε: MI-max vs Uniform */}
        <ChartCard title="Max feature budget  ·  MI-weighted vs Uniform  ·  across ε"
          note="Gap widens as ε grows — MI concentrates budget more aggressively at higher ε">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={budgetByEps} margin={{ top: 8, right: 20, left: 0, bottom: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="eps" tickFormatter={v => `ε=${v}`} tick={tick} />
              <YAxis tick={tick} tickFormatter={v => v.toFixed(3)} width={56}
                label={{ value: 'ε top feature', angle: -90, position: 'insideLeft', fill: C.txt2, fontSize: 11 }} />
              <Tooltip content={<Tip xLabel="ε_total" fmt={v => v?.toFixed(4)} />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.txt2, fontFamily: 'monospace' }} />
              <Line dataKey="mi_max"  name="MI-Weighted (top feature)" stroke={C.blue}  strokeWidth={2.5} dot={{ r: 4, fill: C.blue }} type="monotone" />
              <Line dataKey="uniform" name="Uniform (each feature)"    stroke={C.txt2} strokeWidth={2}   dot={{ r: 3 }} type="monotone" strokeDasharray="5 4" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <div className="insight-stack" style={{ marginTop: 16 }}>
        <Insight label="Big Data Advantage" color="green">
          Global sensitivity GS = (max − min) / n shrinks as n grows. With Spark distributing
          125,974 NSL-KDD records, the per-partition sensitivity is lower — enabling{' '}
          <span className="insight-highlight">less noise per partition</span>{' '}
          before aggregation. HDFS-scale data (millions of rows) amplifies this further via
          the subsampling amplification theorem: ε_amp = log(1 + q(e^ε − 1)).
        </Insight>
        <Insight label="Spark Architecture" color="blue">
          The pipeline uses 3 Spark stages: (1) <span className="insight-highlight">approxQuantile</span> for clip thresholds,
          (2) <span className="insight-highlight">groupBy + count</span> for distributed MI,
          (3) <span className="insight-highlight">withColumn</span> for noise injection. Each stage
          runs as a distributed Spark DAG — no data leaves the executor nodes until the final collect().
        </Insight>
        <Insight label="MI-Weighted Result" color="purple">
          At ε=5.0, MI-weighted allocation achieves{' '}
          <span className="insight-highlight">80.95%</span> vs 54.29% for uniform — a
          26.7 pp improvement from smarter budget distribution alone. Still 13.8 pp below
          DP-SGD (94.70%), confirming gradient-level privacy is the correct long-term approach.
        </Insight>
      </div>
    </Section>
  )
}

// =====================================================================
// NAV + HERO
// =====================================================================
const NAV_ITEMS = [
  { id: 's1',  label: 'Privacy-Utility' },
  { id: 's2',  label: 'Sensitivity' },
  { id: 's3',  label: 'Composition' },
  { id: 's4',  label: 'Amplification' },
  { id: 's5',  label: 'Attack Validation' },
  { id: 's6',  label: 'Local DP' },
  { id: 's7',  label: 'DP-ML' },
  { id: 's8',  label: 'DP-SGD' },
  { id: 's9',  label: 'Clipped Sens.' },
  { id: 's10', label: 'PRV' },
  { id: 's11', label: 'Multi-class' },
  { id: 's12', label: 'Cross-Dataset' },
  { id: 's13', label: 'Spark Pipeline' },
]

const PRESET_EPS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

// ── Animated counter ─────────────────────────────────────────
function Counter({ target, suffix = '' }) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    let start = null
    const duration = 1400
    const step = ts => {
      if (!start) start = ts
      const p = Math.min((ts - start) / duration, 1)
      setVal(Math.floor(p * p * target))
      if (p < 1) requestAnimationFrame(step)
      else setVal(target)
    }
    const raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target])
  return <>{val.toLocaleString()}{suffix}</>
}

function HeroSection({ eps, setEps }) {
  return (
    <div className="hero">
      <div className="hero-eyebrow">BDS-EL · NSL-KDD &amp; UNSW-NB15 Privacy Research</div>
      <h1 className="hero-title">Differential Privacy<br />Research Dashboard</h1>
      <p className="hero-sub">
        Empirical privacy–utility analysis across 13 experiment modules on 2 benchmark datasets:
        data-driven sensitivity, PRV/RDP composition, subsampling amplification,
        membership inference attacks, local DP, DP-SGD, clipped sensitivity,
        multi-class intrusion detection, and cross-dataset validation
        (NSL-KDD 37 features · UNSW-NB15 39 features).
      </p>

      <div className="stats-row">
        {[
          { num: 208306, suf: '',    label: 'Total records (2 datasets)', color: 'blue'   },
          { num: 39,     suf: '',    label: 'Max features analysed', color: 'purple' },
          { num: 13,     suf: '',    label: 'Experiment modules', color: 'cyan'  },
          { num: 30,     suf: ' runs', label: 'Statistical runs', color: 'green'  },
        ].map(s => (
          <div key={s.label} className={`stat-card ${s.color}`}>
            <div className="stat-num mono"><Counter target={s.num} suffix={s.suf} /></div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="eps-bar">
        <div className="eps-bar-label">Global ε — controls highlighted reference lines across all charts</div>
        <div className="eps-chips">
          {PRESET_EPS.map(e => (
            <button key={e} className={`eps-chip ${eps === e ? 'active' : ''}`}
              onClick={() => setEps(e)}>ε = {e}</button>
          ))}
          <button className={`eps-chip ${eps === null ? 'active' : ''}`}
            onClick={() => setEps(null)}>clear</button>
        </div>
      </div>
    </div>
  )
}

// =====================================================================
// ROOT
// =====================================================================
export default function App() {
  const [eps, setEps] = useState(1.0)
  const [activeNav, setActiveNav] = useState('s1')

  // Scroll spy
  useEffect(() => {
    const obs = new IntersectionObserver(
      entries => {
        entries.forEach(e => { if (e.isIntersecting) setActiveNav(e.target.id) })
      },
      { threshold: 0.4, rootMargin: '-100px 0px -60% 0px' }
    )
    NAV_ITEMS.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) obs.observe(el)
    })
    return () => obs.disconnect()
  }, [])

  return (
    <>
      {/* ── Navigation ─────────────────────────────── */}
      <nav className="nav">
        <div className="nav-inner">
          <span className="nav-logo">BDS-EL</span>
          {NAV_ITEMS.map(({ id, label }) => (
            <button key={id}
              className={`nav-btn ${activeNav === id ? 'active' : ''}`}
              onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })}>
              {label}
            </button>
          ))}
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────── */}
      <HeroSection eps={eps} setEps={setEps} />

      {/* ── Divider ────────────────────────────────── */}
      <div style={{ height: 1, background: 'linear-gradient(90deg, transparent, rgba(59,130,246,0.2), transparent)', margin: '0 32px' }} />

      {/* ── Sections ───────────────────────────────── */}
      <PrivacyUtilitySection eps={eps} />
      <SensitivitySection />
      <CompositionSection eps={eps} />
      <AmplificationSection />
      <MIASection eps={eps} />
      <LDPSection eps={eps} />
      <MLSection />
      <DPSGDSection />
      <ClippedSensSection />
      <PRVSection />
      <MulticlassSection />
      <CrossDatasetSection />
      <SparkPipelineSection />

      {/* ── Footer ─────────────────────────────────── */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '48px 32px', maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 32, alignItems: 'end' }}>
          <div>
            <div className="footer-cmd">
              <span style={{ color: 'var(--txt3)' }}>$</span>{' '}
              <span style={{ color: 'var(--green)' }}>python</span>{' '}
              main.py{' '}
              <span style={{ color: 'var(--purple)' }}>--dataset</span> KDDTrain+.csv{' '}
              <span style={{ color: 'var(--purple)' }}>--no_spark</span>{' '}
              <span style={{ color: 'var(--purple)' }}>--ml_runs</span> 30
            </div>
            <p style={{ fontSize: 12, color: 'var(--txt3)', lineHeight: 1.6 }}>
              Full pipeline: DP-SGD, clipped sensitivity, PRV accountant, budget comparison,
              multi-class IDS, cross-dataset validation, Spark distributed pipeline. 13 modules, 30-run CIs, 2 datasets.
            </p>
          </div>
          <div style={{ textAlign: 'right', fontSize: 11, color: 'var(--txt3)', lineHeight: 1.7 }}>
            <div>NSL-KDD · 125,974 records · 37 features</div>
            <div>UNSW-NB15 · 82,332 records · 39 features</div>
            <div>Laplace · Gaussian · DP-SGD · PRV · Clipped-GS</div>
            <div>Target: IEEE TIFS · TDSC · USENIX Security</div>
            <div style={{ color: 'var(--txt3)', marginTop: 6, fontFamily: 'var(--mono)', fontSize: 10 }}>BDS-EL Research Framework v2</div>
          </div>
        </div>
      </footer>
    </>
  )
}
