import React, { useState, useCallback, useMemo } from 'react';

/* ── All 5 Exposure conditions from IS 456:2000 Table 3 & Clause 35.3.2 ── */
const EXPOSURE_CONDITIONS = [
  {
    key: 'mild',
    label: 'Mild',
    limit: 0.3,
    badgeColor: 'emerald',
    desc: 'Concrete surfaces protected against weather or aggressive conditions, except those in coastal areas.',
    clause: 'IS 456:2000 Table 3 (i) & Cl 35.3.2',
  },
  {
    key: 'moderate',
    label: 'Moderate',
    limit: 0.2,
    badgeColor: 'teal',
    desc: 'Exposed to condensation & rain, continuously under water, or in contact with non-aggressive soil/groundwater.',
    clause: 'IS 456:2000 Table 3 (ii) & Cl 35.3.2',
  },
  {
    key: 'severe',
    label: 'Severe',
    limit: 0.1,
    badgeColor: 'amber',
    desc: 'Exposed to severe rain, alternate wetting & drying, occasional freezing, seawater immersion, or coastal air.',
    clause: 'IS 456:2000 Table 3 (iii) & Cl 35.3.2',
  },
  {
    key: 'very_severe',
    label: 'Very Severe',
    limit: 0.1,
    badgeColor: 'orange',
    desc: 'Exposed to seawater spray, corrosive fumes, severe freezing whilst wet, or aggressive sub-soil/groundwater.',
    clause: 'IS 456:2000 Table 3 (iv) & Cl 35.3.2',
  },
  {
    key: 'extreme',
    label: 'Extreme',
    limit: 0.1,
    badgeColor: 'rose',
    desc: 'Members in tidal zone or direct contact with liquid/solid aggressive chemicals. Special protection required.',
    clause: 'IS 456:2000 Table 3 (v) & Cl 35.3.2',
  },
];

/* ── Standard Section Presets ── */
const PRESETS = [
  {
    name: 'Standard Beam (300×500)',
    values: { b: 300, h: 500, d: 450, x: 150, fs: 230, Es: 200000, As: 1256, cmin: 40, s: 150, phi: 16 },
  },
  {
    name: 'Heavy Girder (400×750)',
    values: { b: 400, h: 750, d: 680, x: 220, fs: 250, Es: 200000, As: 2450, cmin: 45, s: 120, phi: 25 },
  },
  {
    name: 'Bridge Deck / Slab (1000×200)',
    values: { b: 1000, h: 200, d: 165, x: 55, fs: 200, Es: 200000, As: 1130, cmin: 25, s: 100, phi: 12 },
  },
  {
    name: 'Retaining Wall (1000×350)',
    values: { b: 1000, h: 350, d: 300, x: 95, fs: 220, Es: 200000, As: 1570, cmin: 40, s: 150, phi: 16 },
  },
];

/* ── Input field definitions for Annex F Calculation ── */
const FIELDS = [
  { key: 'b',    label: 'Width of Section (b)',             unit: 'mm',  min: 50,   max: 2000,  step: 10,   def: 300,   info: 'Width of the section at centroid of tension steel' },
  { key: 'h',    label: 'Overall Depth (h)',                unit: 'mm',  min: 100,  max: 3000,  step: 10,   def: 500,   info: 'Total overall depth of the concrete member' },
  { key: 'd',    label: 'Effective Depth (d)',               unit: 'mm',  min: 50,   max: 2900,  step: 10,   def: 450,   info: 'Distance from compression face to centroid of tension steel' },
  { key: 'x',    label: 'Neutral Axis Depth (x)',           unit: 'mm',  min: 10,   max: 1500,  step: 5,    def: 150,   info: 'Depth of neutral axis for the cracked section' },
  { key: 'fs',   label: 'Steel Stress (fₛ)',                unit: 'MPa', min: 10,   max: 500,   step: 5,    def: 230,   info: 'Service tensile stress in reinforcement (0.58·fy or calculated)' },
  { key: 'Es',   label: 'Elastic Modulus of Steel (Eₛ)',    unit: 'MPa', min: 100000, max: 250000, step: 5000, def: 200000, info: 'Modulus of elasticity of steel (usually 200,000 N/mm²)' },
  { key: 'As',   label: 'Area of Tension Steel (Aₛ)',       unit: 'mm²', min: 50,   max: 20000, step: 50,   def: 1256,  info: 'Total cross-sectional area of tension bars (e.g. 4×Ø20 = 1256 mm²)' },
  { key: 'cmin', label: 'Min. Clear Cover (c_min)',         unit: 'mm',  min: 10,   max: 100,   step: 5,    def: 40,    info: 'Minimum concrete cover to longitudinal bars (Table 16)' },
  { key: 's',    label: 'Bar Spacing c/c (s)',              unit: 'mm',  min: 30,   max: 600,   step: 5,    def: 150,   info: 'Centre-to-centre spacing between tension bars' },
  { key: 'phi',  label: 'Bar Diameter (φ)',                 unit: 'mm',  min: 6,    max: 40,    step: 2,    def: 16,    info: 'Nominal diameter of tension reinforcement bars' },
];

/* ── Severity Classification Rules based on IS 456 Standards ── */
function classifyCrackSeverity(wcr) {
  if (wcr <= 0.10) {
    return {
      level: 'Mild',
      color: 'emerald',
      bgClass: 'sev-mild',
      icon: '🟢',
      summary: 'Hairline / Mild Crack',
      description: 'Fully compliant with ALL IS 456:2000 exposure classes (Mild, Moderate, Severe, Very Severe, and Extreme). Cosmetic in nature with negligible ingress risk.',
      action: 'Routine periodic monitoring. No structural intervention required.',
      permissibleIn: ['Mild (≤ 0.3mm)', 'Moderate (≤ 0.2mm)', 'Severe (≤ 0.1mm)', 'Very Severe (≤ 0.1mm)', 'Extreme (≤ 0.1mm)'],
      exceedsIn: [],
    };
  } else if (wcr <= 0.20) {
    return {
      level: 'Moderate',
      color: 'teal',
      bgClass: 'sev-moderate',
      icon: '🟡',
      summary: 'Moderate Crack Width',
      description: 'Compliant for Mild and Moderate exposure conditions (≤ 0.2 mm). Exceeds limits for Severe, Very Severe, and Extreme environments.',
      action: 'Suitable for dry interior or sheltered areas. If exposed to marine, coastal, or chemical spray, apply protective coating.',
      permissibleIn: ['Mild (≤ 0.3mm)', 'Moderate (≤ 0.2mm)'],
      exceedsIn: ['Severe (limit 0.1mm)', 'Very Severe (limit 0.1mm)', 'Extreme (limit 0.1mm)'],
    };
  } else if (wcr <= 0.30) {
    return {
      level: 'Severe',
      color: 'amber',
      bgClass: 'sev-severe',
      icon: '🟠',
      summary: 'Severe Crack Width',
      description: 'Meets permissible limit ONLY for protected Mild exposure (≤ 0.3 mm). Non-compliant in Moderate, Severe, Very Severe, and Extreme environments.',
      action: 'Higher risk of moisture ingress and reinforcement corrosion. Reduce bar spacing, increase steel area, or apply surface sealant.',
      permissibleIn: ['Mild (≤ 0.3mm)'],
      exceedsIn: ['Moderate (limit 0.2mm)', 'Severe (limit 0.1mm)', 'Very Severe (limit 0.1mm)', 'Extreme (limit 0.1mm)'],
    };
  } else if (wcr <= 0.50) {
    return {
      level: 'Very Severe',
      color: 'orange',
      bgClass: 'sev-very-severe',
      icon: '🔴',
      summary: 'Very Severe Crack Width',
      description: 'Exceeds permissible limits for ALL IS 456:2000 exposure categories. Rapid ingress of moisture, oxygen, and carbonation causing accelerated rebar depassivation.',
      action: 'Non-compliant. Requires prompt repair: low-viscosity epoxy or polyurethane injection pressure grouting and structural review.',
      permissibleIn: [],
      exceedsIn: ['Mild (limit 0.3mm)', 'Moderate (limit 0.2mm)', 'Severe (limit 0.1mm)', 'Very Severe (limit 0.1mm)', 'Extreme (limit 0.1mm)'],
    };
  } else {
    return {
      level: 'Extreme',
      color: 'rose',
      bgClass: 'sev-extreme',
      icon: '⚠️',
      summary: 'Extreme / Critical Structural Hazard',
      description: 'Major wide fissure (> 0.50 mm). Significant loss of aggregate interlock and potential shear/moment capacity compromise. Risk of progressive spalling.',
      action: 'CRITICAL HAZARD. Immediate structural audit, load restriction, propping, and full structural retrofitting / carbon fiber wrapping.',
      permissibleIn: [],
      exceedsIn: ['All IS 456:2000 Exposure Conditions (Gross violation of serviceability limit state)'],
    };
  }
}

/* ── Local Pure Calculation (Annex F Formula) ── */
function calculateCrackWidthLocal(inputs, crackPoint, exposureKey) {
  const { b, h, d, x, fs, Es, As, cmin, s, phi } = inputs;
  const a = h; // crack measured at tension surface

  let acr;
  if (crackPoint === 'midway') {
    acr = Math.sqrt(Math.pow(s / 2, 2) + Math.pow(cmin + phi / 2, 2)) - phi / 2;
  } else {
    acr = cmin;
  }

  // ε₁: Strain at level considered
  const epsilon1 = (d - x) !== 0 ? (fs / Es) * ((a - x) / (d - x)) : 0;

  // Tension stiffening
  const stiffeningDenom = 3 * Es * As * (d - x);
  const stiffening = stiffeningDenom !== 0 ? (b * (h - x) * (a - x)) / stiffeningDenom : 0;

  let epsilonM = epsilon1 - stiffening;
  const epsilonMRaw = epsilonM;
  if (epsilonM < 0) epsilonM = 0;

  const denom = (h - x) !== 0 ? 1 + (2 * (acr - cmin)) / (h - x) : 1;
  const Wcr = denom !== 0 ? (3 * acr * epsilonM) / denom : 0;

  const expObj = EXPOSURE_CONDITIONS.find((e) => e.key === exposureKey) || EXPOSURE_CONDITIONS[1];
  const severity = classifyCrackSeverity(Wcr);
  const pass = Wcr <= expObj.limit;

  return {
    a,
    acr: round(acr, 2),
    epsilon1: round(epsilon1, 6),
    stiffening: round(stiffening, 6),
    epsilonMRaw: round(epsilonMRaw, 6),
    epsilonM: round(epsilonM, 6),
    denominator: round(denom, 4),
    Wcr: round(Wcr, 4),
    severity,
    pass,
    expObj,
  };
}

function round(val, places) {
  const f = Math.pow(10, places);
  return Math.round(val * f) / f;
}

/* ── Main Component ── */
function CrackWidthCalculator({ onNavigate, detectionData }) {
  const [calcMode, setCalcMode] = useState('parameters'); // 'parameters' or 'direct'
  const [form, setForm] = useState(
    Object.fromEntries(FIELDS.map((f) => [f.key, f.def]))
  );
  const [directWidth, setDirectWidth] = useState(0.25);
  const [exposure, setExposure] = useState('moderate');
  const [crackPoint, setCrackPoint] = useState('midway');
  const [results, setResults] = useState(() => calculateCrackWidthLocal(
    Object.fromEntries(FIELDS.map((f) => [f.key, f.def])),
    'midway',
    'moderate'
  ));
  const [showSteps, setShowSteps] = useState(true);
  const [showFormulaInfo, setShowFormulaInfo] = useState(true);
  const [selectedPreset, setSelectedPreset] = useState(null);

  const expObj = useMemo(
    () => EXPOSURE_CONDITIONS.find((e) => e.key === exposure) || EXPOSURE_CONDITIONS[0],
    [exposure]
  );

  const handleChange = useCallback((key, val) => {
    const num = parseFloat(val);
    if (!isNaN(num)) {
      setForm((prev) => {
        const next = { ...prev, [key]: num };
        return next;
      });
      setSelectedPreset(null);
    }
  }, []);

  const handleApplyPreset = (preset) => {
    setForm(preset.values);
    setSelectedPreset(preset.name);
    const res = calculateCrackWidthLocal(preset.values, crackPoint, exposure);
    setResults(res);
  };

  const handleCalculate = (e) => {
    if (e) e.preventDefault();
    if (calcMode === 'parameters') {
      const res = calculateCrackWidthLocal(form, crackPoint, exposure);
      setResults(res);
    } else {
      const sev = classifyCrackSeverity(directWidth);
      setResults({
        Wcr: directWidth,
        severity: sev,
        pass: directWidth <= expObj.limit,
        expObj,
        isDirect: true,
      });
    }
  };

  const handleExposureChange = (newExp) => {
    setExposure(newExp);
    if (calcMode === 'parameters') {
      const res = calculateCrackWidthLocal(form, crackPoint, newExp);
      setResults(res);
    } else if (results) {
      const exp = EXPOSURE_CONDITIONS.find((x) => x.key === newExp);
      setResults({
        ...results,
        expObj: exp,
        pass: results.Wcr <= exp.limit,
      });
    }
  };

  const handleCrackPointChange = (newPoint) => {
    setCrackPoint(newPoint);
    if (calcMode === 'parameters') {
      const res = calculateCrackWidthLocal(form, newPoint, exposure);
      setResults(res);
    }
  };

  const handleDirectWidthChange = (val) => {
    const w = parseFloat(val) || 0;
    setDirectWidth(w);
    const sev = classifyCrackSeverity(w);
    setResults({
      Wcr: w,
      severity: sev,
      pass: w <= expObj.limit,
      expObj,
      isDirect: true,
    });
  };

  const importFromDetection = () => {
    if (detectionData?.estimated_width) {
      setCalcMode('direct');
      handleDirectWidthChange(detectionData.estimated_width);
    }
  };

  const currentWcr = results ? results.Wcr : directWidth;
  const currentSeverity = results ? results.severity : classifyCrackSeverity(currentWcr);
  const currentPass = results ? results.pass : (currentWcr <= expObj.limit);

  return (
    <div className="anim-fade-up cw-container">
      {/* ── Page Header ── */}
      <div className="pg-header">
        <div className="cw-header-badge">IS 456 : 2000 CODE OF PRACTICE</div>
        <h1>Crack Width Analysis & Severity Classification</h1>
        <p>
          Evaluate reinforced concrete crack width per <strong>IS 456:2000 Annex F (Clause 35.3.2 & Table 3)</strong>,
          classify into <strong>Mild, Moderate, Severe, Very Severe, or Extreme</strong>, and inspect mathematical step-by-step formulations.
        </p>
      </div>

      {/* ── Mode Selector & Presets ── */}
      <div className="cw-top-bar g-card">
        <div className="cw-mode-select-wrap">
          <label className="cw-field-label">Input Mode:</label>
          <div className="toggle-group" style={{ marginBottom: 0 }}>
            <button
              type="button"
              className={`toggle-opt ${calcMode === 'parameters' ? 'active' : ''}`}
              onClick={() => { setCalcMode('parameters'); handleCalculate(); }}
            >
              📐 Structural Parameters (Annex F Engine)
            </button>
            <button
              type="button"
              className={`toggle-opt ${calcMode === 'direct' ? 'active' : ''}`}
              onClick={() => { setCalcMode('direct'); handleDirectWidthChange(directWidth); }}
            >
              📏 Direct Crack Width Input (mm)
            </button>
          </div>
        </div>

        {calcMode === 'parameters' && (
          <div className="cw-presets-wrap">
            <span className="cw-preset-label">Quick Section Presets:</span>
            <div className="cw-presets-btns">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  type="button"
                  className={`btn btn-sm ${selectedPreset === p.name ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => handleApplyPreset(p)}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {detectionData && (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={importFromDetection}
            title="Import crack width from AI image detection"
          >
            📸 Import AI Detection Width ({detectionData.estimated_width} mm)
          </button>
        )}
      </div>

      {/* ── Main Two-Column Grid ── */}
      <div className="grid-2" style={{ marginTop: 24 }}>
        {/* ════════════════ LEFT COLUMN: INPUT CONTROLS ════════════════ */}
        <div className="g-card">
          <div className="section-title">
            <span>⚙️ {calcMode === 'parameters' ? 'Design Parameters & Environment' : 'Direct Measurement & Environment'}</span>
          </div>

          {/* 1. Exposure Condition Selection (Table 3) */}
          <div className="cw-exposure-section">
            <div className="cw-field-header-row">
              <label className="cw-field-label">Environmental Exposure Condition (IS 456 Table 3)</label>
              <span className="cw-field-tag">Clause 35.3.2</span>
            </div>
            <div className="cw-exposure-grid-5">
              {EXPOSURE_CONDITIONS.map((ec) => (
                <button
                  key={ec.key}
                  type="button"
                  className={`cw-expo-card-btn ${exposure === ec.key ? 'active' : ''} ${ec.key}`}
                  onClick={() => handleExposureChange(ec.key)}
                >
                  <div className="cw-expo-card-top">
                    <span className="cw-expo-card-name">{ec.label}</span>
                    <span className={`badge badge-${ec.badgeColor}`}>≤ {ec.limit} mm</span>
                  </div>
                  <div className="cw-expo-card-desc">{ec.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* 2. Direct Mode Inputs */}
          {calcMode === 'direct' ? (
            <div className="cw-direct-input-box" style={{ marginTop: 24 }}>
              <label className="cw-field-label">Observed / Measured Surface Crack Width (W_cr in mm)</label>
              <div className="cw-direct-slider-row">
                <input
                  type="number"
                  className="cw-number-input"
                  min="0.01"
                  max="5.00"
                  step="0.01"
                  value={directWidth}
                  onChange={(e) => handleDirectWidthChange(e.target.value)}
                />
                <input
                  type="range"
                  className="slider-input"
                  min="0.01"
                  max="1.50"
                  step="0.01"
                  value={directWidth}
                  onChange={(e) => handleDirectWidthChange(e.target.value)}
                />
                <span className="cw-unit-pill">mm</span>
              </div>
              <p className="cw-input-info" style={{ marginTop: 8 }}>
                Enter the maximum crack width measured using a crack width microscope, optical gauge, or ultrasonic probe.
              </p>
            </div>
          ) : (
            /* 3. Parameters Mode Inputs */
            <form onSubmit={handleCalculate} style={{ marginTop: 20 }}>
              {/* Crack Check Location */}
              <div className="cw-location-selector">
                <label className="cw-field-label">Crack Calculation Location (Annex F)</label>
                <div className="toggle-group" style={{ width: '100%', marginBottom: 12 }}>
                  <button
                    type="button"
                    style={{ flex: 1 }}
                    className={`toggle-opt ${crackPoint === 'midway' ? 'active' : ''}`}
                    onClick={() => handleCrackPointChange('midway')}
                  >
                    ⭐ Midway Between Tension Bars (Worst Case)
                  </button>
                  <button
                    type="button"
                    style={{ flex: 1 }}
                    className={`toggle-opt ${crackPoint === 'below' ? 'active' : ''}`}
                    onClick={() => handleCrackPointChange('below')}
                  >
                    📍 Directly Below Tension Bar
                  </button>
                </div>
              </div>

              {/* Parameter Inputs Grid */}
              <div className="cw-params-grid">
                {FIELDS.map((f) => (
                  <div key={f.key} className="cw-param-card">
                    <div className="cw-param-header">
                      <label className="cw-param-title" htmlFor={`input-${f.key}`}>
                        {f.label}
                      </label>
                      <div className="cw-param-val-box">
                        <input
                          id={`input-${f.key}`}
                          type="number"
                          className="cw-param-num-input"
                          min={f.min}
                          max={f.max}
                          step={f.step}
                          value={form[f.key]}
                          onChange={(e) => handleChange(f.key, e.target.value)}
                        />
                        <span className="cw-param-unit">{f.unit}</span>
                      </div>
                    </div>
                    <input
                      type="range"
                      className="slider-input"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={form[f.key]}
                      onChange={(e) => handleChange(f.key, e.target.value)}
                    />
                    <div className="cw-param-desc">{f.info}</div>
                  </div>
                ))}
              </div>

              <button
                type="submit"
                className="btn btn-primary btn-lg btn-full"
                style={{ marginTop: 20 }}
              >
                ⚡ Recalculate IS 456 Annex F Crack Width
              </button>
            </form>
          )}
        </div>

        {/* ════════════════ RIGHT COLUMN: RESULTS & CLASSIFICATION ════════════════ */}
        <div className="cw-results-column">
          {/* 1. SEVERITY CLASSIFICATION CARD */}
          <div className={`g-card cw-severity-card ${currentSeverity.bgClass}`}>
            <div className="cw-sev-header">
              <span className="cw-sev-icon">{currentSeverity.icon}</span>
              <div>
                <div className="cw-sev-subtitle">IS 456:2000 CRACK CLASSIFICATION</div>
                <h2 className="cw-sev-title">{currentSeverity.level.toUpperCase()} SEVERITY</h2>
              </div>
              <span className={`badge badge-${currentSeverity.color} cw-sev-badge`}>
                {currentSeverity.summary}
              </span>
            </div>

            {/* Severity Multi-segment Gauge */}
            <div className="cw-gauge-container">
              <div className="cw-gauge-track">
                <div className={`cw-gauge-seg mild ${currentSeverity.level === 'Mild' ? 'active' : ''}`}>Mild (≤0.1)</div>
                <div className={`cw-gauge-seg moderate ${currentSeverity.level === 'Moderate' ? 'active' : ''}`}>Moderate (0.1-0.2)</div>
                <div className={`cw-gauge-seg severe ${currentSeverity.level === 'Severe' ? 'active' : ''}`}>Severe (0.2-0.3)</div>
                <div className={`cw-gauge-seg very-severe ${currentSeverity.level === 'Very Severe' ? 'active' : ''}`}>Very Severe (0.3-0.5)</div>
                <div className={`cw-gauge-seg extreme ${currentSeverity.level === 'Extreme' ? 'active' : ''}`}>Extreme (&gt;0.5)</div>
              </div>
              <div
                className="cw-gauge-pointer"
                style={{
                  left: `${Math.min(98, Math.max(2, (currentWcr / 0.60) * 100))}%`,
                }}
              >
                ▲ W<sub>cr</sub>: {currentWcr.toFixed(3)} mm
              </div>
            </div>

            {/* Exposure Compliance Banner */}
            <div className={`cw-compliance-banner ${currentPass ? 'pass' : 'fail'}`}>
              <span className="cw-comp-icon">{currentPass ? '✅' : '❌'}</span>
              <div className="cw-comp-body">
                <div className="cw-comp-title">
                  {currentPass ? 'COMPLIANT WITH EXPOSURE CRITERIA' : 'EXCEEDS PERMISSIBLE CRACK LIMIT'}
                </div>
                <div className="cw-comp-desc">
                  Calculated W<sub>cr</sub> = <strong>{currentWcr.toFixed(4)} mm</strong> vs Permissible Limit = <strong>{expObj.limit} mm</strong> for <em>{expObj.label}</em> exposure ({expObj.clause}).
                </div>
              </div>
            </div>

            <p className="cw-sev-description">{currentSeverity.description}</p>

            {/* Engineering Action Callout */}
            <div className="cw-action-box">
              <strong>🛠️ Recommended Action:</strong> {currentSeverity.action}
            </div>

            {/* Permissibility matrix breakdown */}
            <div className="cw-matrix-box">
              <div className="cw-matrix-item">
                <span className="cw-matrix-label">✅ Permissible in:</span>
                <span className="cw-matrix-val">
                  {currentSeverity.permissibleIn.length > 0 ? currentSeverity.permissibleIn.join(', ') : 'None (Exceeds all codes)'}
                </span>
              </div>
              {currentSeverity.exceedsIn.length > 0 && (
                <div className="cw-matrix-item exceeds">
                  <span className="cw-matrix-label">⚠️ Exceeds Limit in:</span>
                  <span className="cw-matrix-val">{currentSeverity.exceedsIn.join(', ')}</span>
                </div>
              )}
            </div>
          </div>

          {/* 2. NUMERICAL METRICS SUMMARY */}
          <div className="g-card" style={{ marginTop: 20 }}>
            <div className="section-title">Key Crack Width Quantities</div>
            <div className="metrics-row">
              <div className="m-card purple">
                <div className="m-label">Crack Width (W_cr)</div>
                <div className="m-value">{currentWcr.toFixed(4)}</div>
                <div className="m-sub">mm (Annex F)</div>
              </div>
              <div className={`m-card ${currentPass ? 'emerald' : 'rose'}`}>
                <div className="m-label">Permissible Limit</div>
                <div className="m-value">{expObj.limit.toFixed(2)}</div>
                <div className="m-sub">mm ({expObj.label})</div>
              </div>
              {results && !results.isDirect && (
                <>
                  <div className="m-card amber">
                    <div className="m-label">Bar Dist. (a_cr)</div>
                    <div className="m-value">{results.acr.toFixed(1)}</div>
                    <div className="m-sub">mm to rebar</div>
                  </div>
                  <div className="m-card blue">
                    <div className="m-label">Avg Strain (ε_m)</div>
                    <div className="m-value">{(results.epsilonM * 1e3).toFixed(3)}</div>
                    <div className="m-sub">×10⁻³ strain</div>
                  </div>
                </>
              )}
            </div>

            {/* Intermediate Values Table */}
            {results && !results.isDirect && (
              <div className="cw-intermediate-section" style={{ marginTop: 16 }}>
                <div className="cw-values-grid">
                  <div className="cw-val-item">
                    <span className="cw-val-label">Raw Steel Strain (ε₁)</span>
                    <span className="cw-val-num">{results.epsilon1.toFixed(6)}</span>
                  </div>
                  <div className="cw-val-item">
                    <span className="cw-val-label">Tension Stiffening Subtracted</span>
                    <span className="cw-val-num">−{results.stiffening.toFixed(6)}</span>
                  </div>
                  <div className="cw-val-item">
                    <span className="cw-val-label">Effective Strain (ε_m)</span>
                    <span className="cw-val-num">{results.epsilonM.toFixed(6)}</span>
                  </div>
                  <div className="cw-val-item">
                    <span className="cw-val-label">Cover Dampening Denominator</span>
                    <span className="cw-val-num">{results.denominator.toFixed(4)}</span>
                  </div>
                  <div className="cw-val-item">
                    <span className="cw-val-label">Calculated acr</span>
                    <span className="cw-val-num">{results.acr.toFixed(2)} mm</span>
                  </div>
                  <div className="cw-val-item">
                    <span className="cw-val-label">Depth to Surface (a)</span>
                    <span className="cw-val-num">{results.a} mm (= h)</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ════════════════ FULL-WIDTH SECTION: FORMULA EXPLANATION & HOW IT WORKS ════════════════ */}
      <div className="g-card cw-formula-master-card" style={{ marginTop: 24 }}>
        <div className="cw-formula-header-bar" onClick={() => setShowFormulaInfo(!showFormulaInfo)}>
          <div className="cw-formula-title-wrap">
            <span className="cw-formula-icon">📐</span>
            <div>
              <h3>IS 456 : 2000 Annex F — Exact Formula & How It Is Calculated</h3>
              <p>Theoretical formulation, tension stiffening physics, and mathematical derivation</p>
            </div>
          </div>
          <button type="button" className="btn btn-ghost btn-sm">
            {showFormulaInfo ? 'Hide Formula Details ▲' : 'Show Formula Details ▼'}
          </button>
        </div>

        {showFormulaInfo && (
          <div className="cw-formula-content anim-fade-up">
            {/* Equation Cards Grid */}
            <div className="cw-equations-grid">
              {/* Primary Equation */}
              <div className="cw-eq-card primary-eq">
                <div className="cw-eq-tag">Primary Equation (Annex F)</div>
                <div className="cw-math-box">
                  <div className="cw-math-formula">
                    W<sub>cr</sub> = <span className="cw-frac"><span className="cw-num">3 · a<sub>cr</sub> · ε<sub>m</sub></span><span className="cw-den">1 + 2 · (a<sub>cr</sub> − C<sub>min</sub>) / (h − x)</span></span>
                  </div>
                </div>
                <div className="cw-eq-desc">
                  Calculates design surface crack width <strong>W<sub>cr</sub></strong> at tension face of flexural members.
                </div>
              </div>

              {/* Strain & Tension Stiffening */}
              <div className="cw-eq-card">
                <div className="cw-eq-tag">Average Strain (ε_m)</div>
                <div className="cw-math-box">
                  <div className="cw-math-formula">
                    ε<sub>m</sub> = ε₁ − <span className="cw-frac"><span className="cw-num">b · (h − x) · (a − x)</span><span className="cw-den">3 · E<sub>s</sub> · A<sub>s</sub> · (d − x)</span></span>
                  </div>
                </div>
                <div className="cw-eq-desc">
                  Applies <strong>tension stiffening correction</strong> (concrete between cracks carrying tensile load).
                </div>
              </div>

              {/* Raw Strain ε1 */}
              <div className="cw-eq-card">
                <div className="cw-eq-tag">Strain at Crack Level (ε₁)</div>
                <div className="cw-math-box">
                  <div className="cw-math-formula">
                    ε₁ = <span className="cw-frac"><span className="cw-num">f<sub>s</sub></span><span className="cw-den">E<sub>s</sub></span></span> · <span className="cw-frac"><span className="cw-num">a − x</span><span className="cw-den">d − x</span></span>
                  </div>
                </div>
                <div className="cw-eq-desc">
                  Elastic strain computed from service steel stress <em>f<sub>s</sub></em> projected to the outer surface <em>a = h</em>.
                </div>
              </div>

              {/* Bar Distance acr */}
              <div className="cw-eq-card">
                <div className="cw-eq-tag">Nearest Rebar Distance (a_cr)</div>
                <div className="cw-math-box">
                  <div className="cw-math-formula">
                    a<sub>cr</sub> = √[ (s/2)² + (C<sub>min</sub> + φ/2)² ] − φ/2
                  </div>
                </div>
                <div className="cw-eq-desc">
                  Euclidean distance from surface check point to the nearest rebar surface (midway between bars).
                </div>
              </div>
            </div>

            {/* "HOW THE FORMULA WORKS" Physical Concept Explanations */}
            <div className="cw-explanation-box" style={{ marginTop: 24 }}>
              <h4>🔬 How Each Term Controls Crack Width in Practice</h4>
              <div className="cw-concepts-grid">
                <div className="cw-concept-item">
                  <div className="cw-concept-title">1. Why does distance from rebar (a<sub>cr</sub>) matter?</div>
                  <p>
                    Concrete directly adjacent to reinforcing bars is bonded tightly and restrained from opening.
                    As you move further away (a<sub>cr</sub> increases, e.g. wider rebar spacing <em>s</em>), the restraining bond diminishes,
                    causing cracks to open significantly wider.
                  </p>
                </div>
                <div className="cw-concept-item">
                  <div className="cw-concept-title">2. What is Tension Stiffening?</div>
                  <p>
                    Even when concrete cracks in tension, the concrete segments <em>between</em> adjacent cracks still grip the rebar and carry tension.
                    This reduces the average strain (ε<sub>m</sub> &lt; ε₁), preventing overly conservative crack width overestimation.
                  </p>
                </div>
                <div className="cw-concept-item">
                  <div className="cw-concept-title">3. How does cover (C<sub>min</sub>) and (h−x) act in the denominator?</div>
                  <p>
                    The denominator factor 1 + 2(a<sub>cr</sub> − C<sub>min</sub>)/(h − x) represents the triangular strain profile rotating about the neutral axis <em>x</em>.
                    A deeper section (h − x) provides a shallower strain gradient, moderating surface crack opening.
                  </p>
                </div>
                <div className="cw-concept-item">
                  <div className="cw-concept-title">4. Limiting Criteria per Clause 35.3.2</div>
                  <p>
                    IS 456:2000 mandates W<sub>cr</sub> ≤ 0.3 mm for Mild exposure, ≤ 0.2 mm for Moderate exposure,
                    and ≤ 0.1 mm for Severe / Aggressive environments to avoid premature corrosion of reinforcement bars.
                  </p>
                </div>
              </div>
            </div>

            {/* Step-by-Step Interactive Evaluation */}
            {results && !results.isDirect && (
              <div className="cw-live-steps-wrap" style={{ marginTop: 24 }}>
                <div className="cw-field-header-row">
                  <h4>📝 Live Step-by-Step Substitution for Your Values</h4>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => setShowSteps(!showSteps)}
                  >
                    {showSteps ? 'Collapse Steps ▲' : 'Expand Steps ▼'}
                  </button>
                </div>

                {showSteps && (
                  <div className="cw-steps anim-fade-up">
                    {/* Step 1 */}
                    <div className="cw-step">
                      <div className="cw-step-num">Step 1</div>
                      <div className="cw-step-body">
                        <div className="cw-step-title">Calculate distance to nearest rebar surface (a<sub>cr</sub>)</div>
                        {crackPoint === 'midway' ? (
                          <div className="cw-step-formula">
                            a<sub>cr</sub> = √[(s/2)² + (C<sub>min</sub> + φ/2)²] − φ/2<br />
                            = √[({form.s}/2)² + ({form.cmin} + {form.phi}/2)²] − {form.phi}/2<br />
                            = √[{(form.s / 2).toFixed(1)}² + {(form.cmin + form.phi / 2).toFixed(1)}²] − {(form.phi / 2).toFixed(1)}<br />
                            <strong>= {results.acr.toFixed(2)} mm</strong>
                          </div>
                        ) : (
                          <div className="cw-step-formula">
                            a<sub>cr</sub> = C<sub>min</sub> = {form.cmin} mm<br />
                            <strong>= {results.acr.toFixed(2)} mm</strong>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Step 2 */}
                    <div className="cw-step">
                      <div className="cw-step-num">Step 2</div>
                      <div className="cw-step-body">
                        <div className="cw-step-title">Calculate raw steel strain at tension surface (ε₁)</div>
                        <div className="cw-step-formula">
                          ε₁ = (f<sub>s</sub> / E<sub>s</sub>) × (a − x) / (d − x)<br />
                          = ({form.fs} / {form.Es}) × ({results.a} − {form.x}) / ({form.d} − {form.x})<br />
                          = {(form.fs / form.Es).toFixed(6)} × {((results.a - form.x) / (form.d - form.x)).toFixed(4)}<br />
                          <strong>= {results.epsilon1.toFixed(6)}</strong>
                        </div>
                      </div>
                    </div>

                    {/* Step 3 */}
                    <div className="cw-step">
                      <div className="cw-step-num">Step 3</div>
                      <div className="cw-step-body">
                        <div className="cw-step-title">Subtract concrete tension stiffening to determine average strain (ε<sub>m</sub>)</div>
                        <div className="cw-step-formula">
                          Tension Stiffening = b · (h − x) · (a − x) / [ 3 · E<sub>s</sub> · A<sub>s</sub> · (d − x) ]<br />
                          = {form.b} × ({form.h} − {form.x}) × ({results.a} − {form.x}) / [ 3 × {form.Es} × {form.As} × ({form.d} − {form.x}) ]<br />
                          = {results.stiffening.toFixed(6)}<br />
                          ε<sub>m</sub> = {results.epsilon1.toFixed(6)} − {results.stiffening.toFixed(6)} = <strong>{results.epsilonM.toFixed(6)}</strong>
                          {results.epsilonMRaw < 0 && (
                            <span className="cw-val-note"> (clamped to 0 as concrete uncracked under minimal stress)</span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Step 4 */}
                    <div className="cw-step">
                      <div className="cw-step-num">Step 4</div>
                      <div className="cw-step-body">
                        <div className="cw-step-title">Compute Design Crack Width (W<sub>cr</sub>) per Annex F</div>
                        <div className="cw-step-formula">
                          W<sub>cr</sub> = 3 · a<sub>cr</sub> · ε<sub>m</sub> / [ 1 + 2 · (a<sub>cr</sub> − C<sub>min</sub>) / (h − x) ]<br />
                          = 3 × {results.acr.toFixed(2)} × {results.epsilonM.toFixed(6)} / [ 1 + 2 × ({results.acr.toFixed(2)} − {form.cmin}) / ({form.h} − {form.x}) ]<br />
                          = {(3 * results.acr * results.epsilonM).toFixed(6)} / {results.denominator.toFixed(4)}<br />
                          <strong>= {results.Wcr.toFixed(4)} mm</strong>
                        </div>
                      </div>
                    </div>

                    {/* Step 5 */}
                    <div className="cw-step">
                      <div className="cw-step-num">Step 5</div>
                      <div className="cw-step-body">
                        <div className="cw-step-title">Classify Severity & Verify Compliance for {expObj.label} Exposure</div>
                        <div className="cw-step-formula">
                          W<sub>cr</sub> = <strong>{results.Wcr.toFixed(4)} mm</strong> → Severity Level: <strong className={`badge badge-${currentSeverity.color}`}>{currentSeverity.level}</strong><br />
                          Exposure Limit: <strong>≤ {expObj.limit} mm</strong> ({expObj.label})<br />
                          <strong className={currentPass ? 'cw-pass-text' : 'cw-fail-text'}>
                            Result: {currentPass ? '✅ SATISFACTORY COMPLIANCE' : '❌ EXCEEDS LIMIT — REMEDIAL ACTION REQUIRED'}
                          </strong>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── IS 456 Code Permissible Table ── */}
      <div className="g-card" style={{ marginTop: 24 }}>
        <div className="section-title">IS 456:2000 Permissible Crack Width Reference Table (Clause 35.3.2 & Table 3)</div>
        <div className="cw-table-wrap">
          <table className="cw-table">
            <thead>
              <tr>
                <th>Exposure Condition</th>
                <th>Environmental Description</th>
                <th>Permissible Crack Width Limit</th>
                <th>Your Result Status ({currentWcr.toFixed(3)} mm)</th>
              </tr>
            </thead>
            <tbody>
              {EXPOSURE_CONDITIONS.map((ec) => {
                const isCurrent = exposure === ec.key;
                const satisfies = currentWcr <= ec.limit;
                return (
                  <tr key={ec.key} className={isCurrent ? 'cw-table-active' : ''}>
                    <td>
                      <strong>{ec.label}</strong>
                      {isCurrent && <span className="cw-current-tag"> (Selected)</span>}
                    </td>
                    <td style={{ fontSize: 13, color: 'var(--text-2)' }}>{ec.desc}</td>
                    <td><span className="cw-limit-pill">≤ {ec.limit} mm</span></td>
                    <td>
                      <span className={`badge ${satisfies ? 'badge-emerald' : 'badge-rose'}`}>
                        {satisfies ? '✓ Compliant' : '✗ Exceeds Limit'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default CrackWidthCalculator;
