import React from 'react';

// The five certified stages returned by the backend as `certified` (scripts/certified_width.py).

const VERDICT = {
  conforms: { text: 'CONFORMS', color: '#059669' },
  does_not_conform: { text: 'DOES NOT CONFORM', color: '#dc2626' },
  inconclusive: { text: 'INCONCLUSIVE', color: '#d97706' },
};

const box = {
  background: 'var(--card-bg, rgba(255,255,255,0.03))', padding: '14px 18px', borderRadius: 10,
  border: '1px solid var(--border)', marginBottom: 12,
};
const stepTitle = { fontWeight: 700, fontSize: 14, marginBottom: 6 };
const small = { fontSize: 13, color: 'var(--text-2)' };

function Verdict({ d, big }) {
  const v = VERDICT[d] || { text: d, color: 'var(--text-2)' };
  return (
    <span style={{
      color: v.color, border: `1.5px solid ${v.color}`, borderRadius: 6, fontWeight: 800,
      padding: big ? '4px 12px' : '1px 7px', fontSize: big ? 16 : 12, whiteSpace: 'nowrap',
    }}>{v.text}</span>
  );
}

const range = r => `${r[0]} – ${r[1]} mm`;

export default function CertifiedAssessment({ c }) {
  if (!c) return null;
  if (!c.available) {
    return <div className="banner info" style={{ marginBottom: 16 }}>Certified assessment not available: {c.reason}</div>;
  }
  const m = c.measurability;
  const p = c.estimators[c.primary];
  const order = ['photo', 'fusion', 'cnn'].filter(k => c.estimators[k]);
  const a = c.anchoring;

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontWeight: 800, fontSize: 17, marginBottom: 10 }}>
        Certified measurement (limit {c.limit_mm} mm, {c.pixels_per_mm} px/mm, {c.n_profiles} profiles)
      </div>

      <div style={{ ...box, borderLeft: '4px solid var(--accent, #6366f1)' }}>
        <div style={stepTitle}>Step 1 · Can this photo measure the crack?</div>
        <div style={small}>
          Smallest width this photo can determine: <strong>w<sub>c</sub> = {m.wc_mm} mm</strong>{' '}
          (background {m.background_grey} grey, noise {m.noise_grey} grey, SNR {m.snr}).{' '}
          {m.measurable
            ? <span style={{ color: '#059669', fontWeight: 700 }}>The crack is above this limit: measurable.</span>
            : <span style={{ color: '#d97706', fontWeight: 700 }}>
                The crack is below this limit: its width cannot be determined from this photo, only bounded.
              </span>}
        </div>
      </div>

      <div style={box}>
        <div style={stepTitle}>Steps 2–3 · Resolution-robust width with a 90 % certified interval</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: 'var(--text-3)' }}>
                <th style={{ padding: '4px 6px' }}>Estimator</th>
                <th style={{ padding: '4px 6px' }}>Width (95th pct.)</th>
                <th style={{ padding: '4px 6px' }}>90 % interval</th>
                <th style={{ padding: '4px 6px' }}>Decision</th>
              </tr>
            </thead>
            <tbody>
              {order.map(k => {
                const e = c.estimators[k];
                return (
                  <tr key={k} style={{ borderTop: '1px solid var(--border)', fontWeight: k === c.primary ? 700 : 400 }}>
                    <td style={{ padding: '6px' }}>{e.label}{k === c.primary ? ' (primary)' : ''}</td>
                    <td style={{ padding: '6px' }}>{e.measurable ? `${e.p95_mm} mm` : `< ${m.wc_mm} mm`}</td>
                    <td style={{ padding: '6px' }}>{range(e.interval90_mm)}</td>
                    <td style={{ padding: '6px' }}><Verdict d={e.decision} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div style={box}>
        <div style={stepTitle}>Step 4 · Three-way compliance decision</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, alignItems: 'center' }}>
          <div>
            <div style={small}>At the widest 5 % of the crack</div>
            <div style={{ marginTop: 4 }}><Verdict d={c.decision} big /></div>
          </div>
          <div>
            <div style={small}>Element maximum ({p.element_max_mm} mm, interval {range(p.element_interval90_mm)})</div>
            <div style={{ marginTop: 4 }}><Verdict d={c.element_decision} big /></div>
          </div>
        </div>
        <div style={{ ...small, marginTop: 8 }}>
          CONFORMS only if the whole interval is below the limit, DOES NOT CONFORM only if it is all above;
          otherwise the photo cannot decide.
        </div>
      </div>

      {c.capture_advice && (
        <div style={{ ...box, borderLeft: '4px solid #d97706' }}>
          <div style={stepTitle}>Step 5 · How to get a conclusive answer</div>
          <div style={small}><strong>{c.capture_advice.advice}</strong></div>
          {c.capture_advice.closer_factor > 1 && !c.capture_advice.camera_given && (
            <div style={{ ...small, marginTop: 4 }}>
              Enter the lens focal length and sensor pixel size to get the maximum camera distance.
            </div>
          )}
          {c.capture_advice.p_conclusive_now !== undefined && (
            <div style={{ ...small, marginTop: 4 }}>
              Chance that a photo at the current resolution decides such a crack: {Math.round(100 * c.capture_advice.p_conclusive_now)} %.
            </div>
          )}
        </div>
      )}

      {a && (
        <div style={box}>
          <div style={stepTitle}>Gauge anchoring · location “{a.location_id}”</div>
          {a.status === 'installed' && <div style={small}>{a.note} (gauge {a.gauge_mm} mm)</div>}
          {a.status === 'no_anchor' && <div style={small}>{a.note} Enter a gauge reading once to install it.</div>}
          {a.status === 'anchored' && (
            <>
              <div style={small}>
                Corrected with the installation gauge reading {a.gauge_mm} mm ({a.installed}); visit {a.visits}.
                Healing ratio = 1 − (width now / width at installation); positive means the crack is closing.
              </div>
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse', marginTop: 6 }}>
                <tbody>
                  {order.filter(k => a.estimators[k]).map(k => (
                    <tr key={k} style={{ borderTop: '1px solid var(--border)' }}>
                      <td style={{ padding: '6px' }}>{c.estimators[k].label}</td>
                      <td style={{ padding: '6px' }}>{a.estimators[k].width_mm} mm</td>
                      <td style={{ padding: '6px' }}>{range(a.estimators[k].interval90_mm)}</td>
                      <td style={{ padding: '6px' }}><Verdict d={a.estimators[k].decision} /></td>
                      <td style={{ padding: '6px' }}>
                        {a.estimators[k].healing_ratio !== null ? `healing ratio ${a.estimators[k].healing_ratio}` : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {a.warning && <div style={{ ...small, color: '#d97706', marginTop: 6 }}>{a.warning}</div>}
            </>
          )}
        </div>
      )}
    </div>
  );
}
