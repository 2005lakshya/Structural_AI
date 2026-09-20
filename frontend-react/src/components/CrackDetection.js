import React, { useState, useRef } from 'react';
import { analyzeImage, measureCrackWidthFromImage } from '../services/api';

// Severity colors for direct vision classification
const SEVERITY_COLORS = {
  'Mild':       'var(--emerald)',
  'Moderate':   '#0d9488',
  'Severe':     'var(--amber)',
  'Very Severe':'#ea580c',
  'Extreme':    'var(--rose)',
};

function CrackDetection({ onNavigate, onDetectionResult }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [dragging, setDragging] = useState(false);
  const ref = useRef(null);

  // Image-based width measurement state
  const [widthResult, setWidthResult] = useState(null);
  const [widthLoading, setWidthLoading] = useState(false);
  const [widthErr, setWidthErr] = useState(null);
  const [scaleMode, setScaleMode] = useState('camera');   // 'camera' | 'reference' | 'dpi'
  const [cameraDistance, setCameraDistance] = useState(50);
  const [refLenPx, setRefLenPx] = useState(0);
  const [refLenMm, setRefLenMm] = useState(25);
  const [dpiVal, setDpiVal] = useState(300);
  const [coverMm, setCoverMm] = useState(40);
  const [designLifeYears, setDesignLifeYears] = useState(50);

  const pick = (f) => {
    if (f && f.type.startsWith('image/')) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResults(null);
      setErr(null);
      setWidthResult(null);
      setWidthErr(null);
    }
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setErr(null);
    try {
      const res = await analyzeImage(file);
      setResults(res);
      if (onDetectionResult) {
        const avgArea = res.count > 0 ? res.crack_area / res.count : 0;
        const estWidth = res.count > 0 ? Math.round(Math.min(10.0, Math.max(0.1, avgArea / 500.0)) * 10) / 10 : 0;
        onDetectionResult({
          file: file,
          count: res.count,
          density: res.density,
          crack_area: res.crack_area,
          total_px: res.total_px,
          max_crack_depth: res.max_crack_depth,
          estimated_width: estWidth,
        });
      }
    } catch {
      setErr('Analysis failed. Check backend.');
    }
    setLoading(false);
  };

  const measureWidth = async () => {
    if (!file) return;
    setWidthLoading(true);
    setWidthErr(null);
    try {
      const params = {};
      if (scaleMode === 'camera')    params.camera_distance_cm = cameraDistance;
      if (scaleMode === 'reference') { params.known_length_px = refLenPx; params.known_length_mm = refLenMm; }
      if (scaleMode === 'dpi')       params.dpi = dpiVal;
      params.cover_mm = coverMm;
      params.design_life_years = designLifeYears;
      const res = await measureCrackWidthFromImage(file, params);
      setWidthResult(res);
      // Push the p95 width to App state
      if (onDetectionResult) {
        onDetectionResult(prev => ({
          ...(prev || {}),
          estimated_width: res.p95_width_mm,
        }));
      }
    } catch (e) {
      setWidthErr('Width measurement failed. Check backend.');
    }
    setWidthLoading(false);
  };

  const severityColor = (level) => SEVERITY_COLORS[level] || 'var(--text-1)';

  return (
    <div className="anim-fade-up">
      <div className="pg-header">
        <h1>Crack Detection & Width Measurement</h1>
        <p>AI-powered Computer Vision pipeline for crack detection, segmentation & physical width measurement</p>
      </div>

      <div className="grid-2">
        {/* ── Upload Card ── */}
        <div className="g-card">
          <div className="section-title">Upload Image</div>
          <input ref={ref} type="file" accept="image/*"
            onChange={e => pick(e.target.files[0])} style={{ display: 'none' }} />

          {!preview ? (
            <div className={`drop-zone ${dragging ? 'dragging' : ''}`}
              onClick={() => ref.current?.click()}
              onDrop={e => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]); }}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              style={{ minHeight: 220, display: 'flex', flexDirection: 'column',
                       alignItems: 'center', justifyContent: 'center', padding: '32px 20px' }}>
              <div style={{ fontSize: 44, marginBottom: 12 }}>📷</div>
              <h4>{dragging ? 'Drop image here' : 'Click or drag an image'}</h4>
              <span>JPG, PNG up to 10MB</span>
            </div>
          ) : (
            <div className="preview-wrap" style={{ width: '100%' }}>
              <div style={{ width: '100%', borderRadius: 'var(--radius-l)', overflow: 'hidden',
                            border: '1px solid var(--border)', background: 'rgba(0,0,0,0.03)',
                            display: 'flex', justifyContent: 'center', alignItems: 'center',
                            cursor: 'pointer', padding: 8 }}
                onClick={() => ref.current?.click()}>
                <img src={preview} alt="Upload preview"
                  style={{ width: '100%', height: 'auto', objectFit: 'contain',
                           display: 'block', borderRadius: 'var(--radius-m)' }} />
              </div>
              <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
                <button className="btn btn-ghost" style={{ flex: 1 }}
                  onClick={() => ref.current?.click()}>Change Image</button>
                <button className="btn btn-ghost" style={{ flex: 1, color: 'var(--rose)' }}
                  onClick={() => { setFile(null); setPreview(null); setResults(null); setWidthResult(null); }}>
                  Remove
                </button>
              </div>
            </div>
          )}

          <button className="btn btn-primary btn-lg btn-full" style={{ marginTop: 16 }}
            onClick={analyze} disabled={!file || loading}>
            {loading ? 'Analyzing...' : 'Run Detection + Segmentation'}
          </button>
          {err && <div className="banner error" style={{ marginTop: 14 }}>{err}</div>}
        </div>

        {/* ── Detection Results Card ── */}
        <div className="g-card">
          <div className="section-title">Detection Results</div>
          {loading ? (
            <div className="loader-wrap"><div className="loader" />
              <div className="loader-text">Running detection pipeline...</div></div>
          ) : !results ? (
            <div className="empty"><h4>No results</h4><p>Upload an image and run detection</p></div>
          ) : (
            <div className="anim-scale">
              <div className="metrics-row" style={{ marginBottom: 16 }}>
                <div className="m-card purple"><div className="m-label">Detections</div>
                  <div className="m-value">{results.count}</div></div>
                <div className="m-card rose"><div className="m-label">Coverage</div>
                  <div className="m-value">{results.density}%</div></div>
                <div className="m-card amber"><div className="m-label">Est. Depth</div>
                  <div className="m-value">{results.max_crack_depth}mm</div></div>
              </div>

              <div className="section-title">
                {results.detector === 'physics_informed_cracknet' ? 'PI-CrackNet Detection' :
                 results.detector === 'swin_transformer' ? 'SwinCrackNet Detection' : 'CrackNet Detection'}
              </div>
              <div style={{ width: '100%', borderRadius: 'var(--radius-m)', overflow: 'hidden',
                            border: '1px solid var(--border)' }}>
                <img src={`data:image/png;base64,${results.annotated_image_b64}`} alt="Detection"
                  style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }} />
              </div>
              {results.detector === 'physics_informed_cracknet' && results.detections?.length > 0 && (
                <div className="img-caption">
                  Avg. fracture ratio (K<sub>I</sub>/K<sub>IC</sub>): {
                    (results.detections.reduce((s, d) => s + (d.fracture_ratio || 0), 0) / results.detections.length).toFixed(2)
                  } — physics-consistency score from the dual-branch network, not just a vision confidence score
                </div>
              )}
              <div className="img-caption">{results.count} crack region(s) · conf ≥ 0.55</div>
              <div className="hr" />
              <div className="section-title">U-Net Segmentation</div>
              <div style={{ width: '100%', borderRadius: 'var(--radius-m)', overflow: 'hidden',
                            border: '1px solid var(--border)' }}>
                <img src={`data:image/png;base64,${results.mask_image_b64}`} alt="Segmentation"
                  style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }} />
              </div>
              <div className="img-caption">{results.density}% coverage · {results.crack_area?.toLocaleString()} px</div>
            </div>
          )}
        </div>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          IMAGE-BASED CRACK WIDTH MEASUREMENT (AI Computer Vision Method)
          ════════════════════════════════════════════════════════════════════ */}
      <div className="g-card" style={{ marginTop: 28 }}>
        <div className="section-title">
          📏 AI Computer Vision Crack Width Measurement
          <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--text-3)',
                         marginLeft: 10, fontStyle: 'italic' }}>
            Direct measurement from image pixels via distance transform & skeletonization
          </span>
        </div>

        <div className="banner info" style={{ marginBottom: 20 }}>
          This method utilizes our <strong>U-Net segmentation → morphological skeletonization → Euclidean distance transform ($L_2$)</strong> pipeline to directly measure physical crack width (mm) and classify crack severity from the photo.
        </div>

        <div style={{ maxWidth: 640, marginBottom: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--text-2)' }}>
            Scale Calibration (how to convert image pixels → physical mm)
          </div>
          <div className="toggle-group" style={{ marginBottom: 14 }}>
            {[['camera','📷 Camera Distance'],['reference','📐 Reference Object'],['dpi','🖨 Scanner DPI']].map(([k,l]) => (
              <button key={k} type="button"
                className={`toggle-opt ${scaleMode === k ? 'active' : ''}`}
                onClick={() => setScaleMode(k)} style={{ flex: 1 }}>{l}</button>
            ))}
          </div>

          {scaleMode === 'camera' && (
            <div style={{ background: 'var(--card-bg, rgba(255,255,255,0.03))', padding: '14px 18px', borderRadius: 10, border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>
                  Camera distance from wall/surface:
                </label>
                <span style={{ fontWeight: 700, color: 'var(--primary)', fontSize: 15 }}>{cameraDistance} cm</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 10 }}>
                <input type="range" className="slider-input" min={10} max={300} step={5}
                  value={cameraDistance} onChange={e => setCameraDistance(+e.target.value)} style={{ flex: 1 }} />
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 8 }}>
                Typical phone camera photo at arm's length ≈ 50 cm. Closer distance provides higher optical resolution.
              </p>
            </div>
          )}

          {scaleMode === 'reference' && (
            <div style={{ background: 'var(--card-bg, rgba(255,255,255,0.03))', padding: '14px 18px', borderRadius: 10, border: '1px solid var(--border)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 13, color: 'var(--text-2)' }}>Reference size in image (px)</label>
                  <input type="number" className="cw-number-input" value={refLenPx}
                    onChange={e => setRefLenPx(+e.target.value)} min={1} style={{ marginTop: 4, width: '100%' }} />
                </div>
                <div>
                  <label style={{ fontSize: 13, color: 'var(--text-2)' }}>Known physical size (mm)</label>
                  <input type="number" className="cw-number-input" value={refLenMm}
                    onChange={e => setRefLenMm(+e.target.value)} min={1} style={{ marginTop: 4, width: '100%' }} />
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 8 }}>
                Example: A standard ₹1 / ₹5 coin placed next to the crack is 25 mm in diameter.
              </p>
            </div>
          )}

          {scaleMode === 'dpi' && (
            <div style={{ background: 'var(--card-bg, rgba(255,255,255,0.03))', padding: '14px 18px', borderRadius: 10, border: '1px solid var(--border)' }}>
              <label style={{ fontSize: 13, color: 'var(--text-2)', display: 'block', marginBottom: 8 }}>Scanner Resolution (DPI)</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                {[72, 150, 300, 600].map(d => (
                  <button key={d} type="button"
                    className={`btn btn-sm ${dpiVal === d ? 'btn-primary' : 'btn-ghost'}`}
                    onClick={() => setDpiVal(d)}>{d} DPI</button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div style={{ maxWidth: 640, marginBottom: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 10, color: 'var(--text-2)' }}>
            Durability Impact Inputs (chloride-ingress service-life model)
          </div>
          <div style={{ background: 'var(--card-bg, rgba(255,255,255,0.03))', padding: '14px 18px', borderRadius: 10, border: '1px solid var(--border)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <label style={{ fontSize: 13, color: 'var(--text-2)' }}>Rebar cover depth (mm)</label>
              <input type="number" className="cw-number-input" value={coverMm}
                onChange={e => setCoverMm(+e.target.value)} min={10} style={{ marginTop: 4, width: '100%' }} />
            </div>
            <div>
              <label style={{ fontSize: 13, color: 'var(--text-2)' }}>Design life (years)</label>
              <input type="number" className="cw-number-input" value={designLifeYears}
                onChange={e => setDesignLifeYears(+e.target.value)} min={1} style={{ marginTop: 4, width: '100%' }} />
            </div>
          </div>
        </div>

        <button className="btn btn-primary btn-lg" onClick={measureWidth}
          disabled={!file || widthLoading} style={{ minWidth: 260 }}>
          {widthLoading ? 'Measuring Crack Width...' : '📐 Calculate Crack Width from Image'}
        </button>
        {widthErr && <div className="banner error" style={{ marginTop: 14 }}>{widthErr}</div>}

        {/* ── Width Measurement Results ── */}
        {widthResult && (
          <div className="anim-scale" style={{ marginTop: 24 }}>
            <div className="hr" />

            {/* Severity Verdict Banner */}
            <div style={{
              padding: '16px 20px', borderRadius: 12, marginBottom: 20,
              background: widthResult.severity_level === 'Mild'
                ? 'rgba(5,150,105,0.12)'
                : widthResult.severity_level === 'Moderate'
                ? 'rgba(13,148,136,0.12)'
                : widthResult.severity_level === 'Severe'
                ? 'rgba(217,119,6,0.12)'
                : 'rgba(220,38,38,0.12)',
              border: `2px solid ${severityColor(widthResult.severity_level)}`,
            }}>
              <div style={{ fontSize: 18, fontWeight: 800,
                            color: severityColor(widthResult.severity_level) }}>
                {widthResult.severity_level === 'Mild' ? '🟢' : widthResult.severity_level === 'Moderate' ? '🟡' : widthResult.severity_level === 'Severe' ? '🟠' : '🔴'} SEVERITY LEVEL: {widthResult.severity_level?.toUpperCase()} CRACK
              </div>
              <div style={{ marginTop: 6, fontSize: 13, color: 'var(--text-1)' }}>
                Design Crack Width (p95): <strong>{widthResult.p95_width_mm} mm</strong>
                &nbsp;·&nbsp;
                Scale: {widthResult.scale_method.replace(/_/g, ' ')} ({widthResult.pixels_per_mm.toFixed(1)} px/mm)
              </div>
              <div style={{ marginTop: 8, fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>
                {widthResult.severity_description}
              </div>
              <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-3)' }}>
                🛠 <strong>Recommended Action:</strong> {widthResult.severity_action}
              </div>
            </div>

            {/* Metrics grid */}
            <div className="metrics-row" style={{ marginBottom: 20 }}>
              <div className="m-card purple">
                <div className="m-label">Max Width</div>
                <div className="m-value">{widthResult.max_width_mm}</div>
                <div className="m-sub">mm</div>
              </div>
              <div className="m-card" style={{ textAlign:'center',
                background: 'rgba(139, 92, 246, 0.12)',
                border: '1px solid var(--primary)' }}>
                <div className="m-label">Design Width (p95)</div>
                <div className="m-value" style={{ color: 'var(--primary)' }}>
                  {widthResult.p95_width_mm}
                </div>
                <div className="m-sub">mm</div>
              </div>
              <div className="m-card emerald">
                <div className="m-label">Mean Width</div>
                <div className="m-value">{widthResult.mean_width_mm}</div>
                <div className="m-sub">mm</div>
              </div>
              <div className="m-card amber">
                <div className="m-label">Median Width</div>
                <div className="m-value">{widthResult.median_width_mm}</div>
                <div className="m-sub">mm</div>
              </div>
            </div>

            {/* Skeleton stats */}
            <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
              <div className="m-card" style={{ flex: 1, minWidth: 120 }}>
                <div className="m-label">Skeleton Length</div>
                <div className="m-value" style={{ fontSize: 18 }}>{widthResult.skeleton_length_px}</div>
                <div className="m-sub">px (crack path)</div>
              </div>
              <div className="m-card" style={{ flex: 1, minWidth: 120 }}>
                <div className="m-label">Crack Pixels</div>
                <div className="m-value" style={{ fontSize: 18 }}>{widthResult.crack_pixels?.toLocaleString()}</div>
                <div className="m-sub">segmentation area</div>
              </div>
              <div className="m-card" style={{ flex: 1, minWidth: 120 }}>
                <div className="m-label">Scale Factor</div>
                <div className="m-value" style={{ fontSize: 18 }}>{widthResult.pixels_per_mm}</div>
                <div className="m-sub">px / mm</div>
              </div>
            </div>

            {/* Overlay image */}
            {widthResult.overlay_image_b64 && (
              <>
                <div className="section-title">Width Measurement Overlay</div>
                <div style={{ width: '100%', borderRadius: 'var(--radius-m)', overflow: 'hidden',
                              border: '1px solid var(--border)', marginBottom: 8 }}>
                  <img src={`data:image/png;base64,${widthResult.overlay_image_b64}`}
                    alt="Width overlay"
                    style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }} />
                </div>
                <div className="img-caption">
                  Green = thin crack · Yellow/Red = wide crack profile
                </div>
              </>
            )}

            {/* Durability / Service-Life Impact */}
            {widthResult.durability_impact && (
              <>
                <div className="hr" />
                <div className="section-title">
                  ⏳ Durability Impact — Chloride-Ingress Service-Life Model
                  <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--text-3)',
                                 marginLeft: 10, fontStyle: 'italic' }}>
                    Converts this crack's measured width into a corrosion-initiation & remaining-life estimate
                  </span>
                </div>

                <div style={{
                  padding: '14px 18px', borderRadius: 12, marginBottom: 16,
                  background: 'rgba(220,38,38,0.08)', border: '1px solid var(--border)',
                }}>
                  <p style={{ fontSize: 13, color: 'var(--text-1)', margin: 0 }}>
                    {widthResult.durability_impact.narrative}
                  </p>
                </div>

                <div className="metrics-row" style={{ marginBottom: 12 }}>
                  <div className="m-card emerald">
                    <div className="m-label">Initiation (Uncracked)</div>
                    <div className="m-value">{widthResult.durability_impact.initiation_years_uncracked}</div>
                    <div className="m-sub">years</div>
                  </div>
                  <div className="m-card rose">
                    <div className="m-label">Initiation (This Crack)</div>
                    <div className="m-value">{widthResult.durability_impact.initiation_years_cracked}</div>
                    <div className="m-sub">years</div>
                  </div>
                  <div className="m-card amber">
                    <div className="m-label">Years Lost</div>
                    <div className="m-value">{widthResult.durability_impact.years_lost_to_crack}</div>
                    <div className="m-sub">years</div>
                  </div>
                  <div className="m-card purple">
                    <div className="m-label">Design Life Consumed</div>
                    <div className="m-value">{widthResult.durability_impact.life_fraction_consumed_pct}%</div>
                    <div className="m-sub">of {widthResult.durability_impact.design_life_years}yr</div>
                  </div>
                </div>

                <div className="img-caption">
                  Diffusion enhancement factor: {widthResult.durability_impact.diffusion_enhancement}× baseline ·
                  &nbsp;Model: Fick's law chloride diffusion (Crank's error-function solution), crack-width-dependent
                  diffusion coefficient — engineering approximation, calibrate against site chloride-profiling data
                  before relying on it for safety-critical decisions.
                </div>
              </>
            )}

            {/* Cross-reference navigation */}
            <div style={{ marginTop: 16 }}>
              <button className="btn btn-secondary" onClick={() => onNavigate('crackwidth')}>
                → Cross-reference with Theoretical IS 456 Annex F Formula Calculator
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CrackDetection;
