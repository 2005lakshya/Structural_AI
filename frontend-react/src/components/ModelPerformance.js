import React, { useState, useEffect } from 'react';
import { getModelPerformance } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function ModelPerformance() {
  const [perf, setPerf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => { load(); }, []);

  const load = async () => {
    setLoading(true); setErr(null);
    try { setPerf(await getModelPerformance()); }
    catch { setErr('Cannot fetch metrics.'); }
    setLoading(false);
  };

  if (loading) return <div className="loader-wrap"><div className="loader" /><div className="loader-text">Computing metrics...</div></div>;
  if (err) return <div className="anim-fade-up"><div className="pg-header"><h1>Model Performance</h1></div><div className="banner error">{err}</div><button className="btn btn-primary" onClick={load}>Retry</button></div>;

  const overview = [
    { name: 'SHI (R²)', val: perf.shi.r2 * 100, fill: '#7C3AED' },
    { name: 'Risk (Accuracy)', val: perf.risk.accuracy * 100, fill: '#E11D48' },
    { name: 'RUL (R²)', val: perf.rul.r2 * 100, fill: '#059669' },
    { name: 'U-Net (Dice)', val: perf.unet.best_dice * 100, fill: '#D97706' },
    ...(perf.cracknet ? [{ name: 'CrackNet (Acc)', val: perf.cracknet.val_acc * 100, fill: '#0891B2' }] : []),
  ];

  const cm = perf.risk.confusion_matrix || [];
  const labels = ['Safe', 'Moderate', 'Critical'];
  const cmMax = Math.max(...cm.flat());
  const cellStyle = (v) => {
    const t = cmMax > 0 ? v / cmMax : 0;
    if (t > 0.7) return { background: 'var(--primary)', color: '#fff' };
    if (t > 0.3) return { background: 'var(--primary-wash-strong)', color: 'var(--primary)' };
    return { background: 'var(--bg-warm)', color: 'var(--text-3)' };
  };

  return (
    <div className="anim-fade-up">
      <div className="pg-header pg-header-row">
        <div><h1>Model Performance</h1><p>Evaluation metrics on 20% test split</p></div>
        <button className="btn btn-secondary" onClick={load}>Refresh</button>
      </div>

      <div className="chart-panel" style={{ marginBottom: 24 }}>
        <div className="section-title">Overview</div>
        <div style={{ height: 240 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={overview} layout="vertical" margin={{ top: 0, right: 24, left: 10, bottom: 0 }}>
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: '#8a839a' }} axisLine={false} tickLine={false} />
              <YAxis dataKey="name" type="category" width={130} tick={{ fontSize: 12, fill: '#544d63', fontWeight: 600 }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: 'rgba(124,58,237,0.04)' }} contentStyle={{ background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.5)', borderRadius: 10, fontSize: 13 }} formatter={v => [`${v.toFixed(1)}%`, 'Score']} />
              <Bar dataKey="val" radius={[0, 6, 6, 0]} barSize={20}>
                {overview.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 24 }}>
        <div className="g-card">
          <div className="section-title">SHI Regressor (XGBoost)</div>
          <div className="grid-3 stagger">
            <div className="m-card purple"><div className="m-label">R²</div><div className="m-value">{perf.shi.r2.toFixed(3)}</div><div className="m-sub">1.0 = perfect</div></div>
            <div className="m-card emerald"><div className="m-label">MAE</div><div className="m-value">{perf.shi.mae.toFixed(2)}</div><div className="m-sub">lower = better</div></div>
            <div className="m-card amber"><div className="m-label">RMSE</div><div className="m-value">{perf.shi.rmse.toFixed(2)}</div></div>
          </div>
        </div>
        <div className="g-card">
          <div className="section-title">RUL Regressor (Random Forest)</div>
          <div className="grid-3 stagger">
            <div className="m-card emerald"><div className="m-label">R²</div><div className="m-value">{perf.rul.r2.toFixed(3)}</div><div className="m-sub">1.0 = perfect</div></div>
            <div className="m-card purple"><div className="m-label">MAE</div><div className="m-value">{perf.rul.mae.toFixed(2)}</div><div className="m-sub">years</div></div>
            <div className="m-card amber"><div className="m-label">RMSE</div><div className="m-value">{perf.rul.rmse.toFixed(2)}</div></div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="g-card">
          <div className="section-title">Risk Classifier (XGBoost)</div>
          <div className="grid-4 stagger" style={{ marginBottom: 20 }}>
            <div className="m-card purple"><div className="m-label">Accuracy</div><div className="m-value">{(perf.risk.accuracy * 100).toFixed(1)}%</div></div>
            <div className="m-card rose"><div className="m-label">Precision</div><div className="m-value">{perf.risk.precision.toFixed(3)}</div></div>
            <div className="m-card amber"><div className="m-label">Recall</div><div className="m-value">{perf.risk.recall.toFixed(3)}</div></div>
            <div className="m-card emerald"><div className="m-label">F1</div><div className="m-value">{perf.risk.f1.toFixed(3)}</div></div>
          </div>

          <div className="section-title">Confusion Matrix</div>
          <div className="cm-grid">
            <div className="cm-head" />
            {labels.map(l => <div key={l} className="cm-head">{l}</div>)}
            {cm.map((row, i) => (
              <React.Fragment key={i}>
                <div className="cm-row-label">{labels[i]}</div>
                {row.map((v, j) => (
                  <div key={j} className="cm-cell" style={cellStyle(v)}>{v}</div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="g-card">
          <div className="section-title">U-Net Segmentation</div>
          <div className="grid-2 stagger" style={{ marginBottom: 18 }}>
            <div className="m-card rose"><div className="m-label">Dice Score</div><div className="m-value">{perf.unet.best_dice.toFixed(4)}</div><div className="m-sub">higher = better</div></div>
            <div className="m-card emerald"><div className="m-label">Parameters</div><div className="m-value">{(perf.unet.params / 1000).toFixed(1)}k</div><div className="m-sub">TinyU-Net</div></div>
          </div>
          <div className="banner info">Lightweight architecture optimized for crack segmentation on resource-constrained devices.</div>
        </div>

        {perf.cracknet ? (
          <div className="g-card">
            <div className="section-title">CrackNet Detector (Custom CNN)</div>
            <div className="grid-3 stagger" style={{ marginBottom: 18 }}>
              <div className="m-card" style={{textAlign:'center',background:'rgba(8,145,178,0.08)',border:'1px solid rgba(8,145,178,0.25)'}}>
                <div className="m-label">Val Accuracy</div>
                <div className="m-value" style={{color:'#0891B2'}}>{(perf.cracknet.val_acc * 100).toFixed(1)}%</div>
                <div className="m-sub">crack / no-crack</div>
              </div>
              <div className="m-card emerald">
                <div className="m-label">Parameters</div>
                <div className="m-value">{(perf.cracknet.params / 1e6).toFixed(2)}M</div>
                <div className="m-sub">MobileNet-style</div>
              </div>
              <div className="m-card purple">
                <div className="m-label">Architecture</div>
                <div className="m-value" style={{fontSize:13}}>CrackNet</div>
                <div className="m-sub">DW-Sep Conv</div>
              </div>
            </div>
            <div className="banner info">Custom MobileNet-inspired classifier trained on 40K crack images. Replaces YOLOv8 for bounding-box detection via sliding window.</div>
          </div>
        ) : (
          <div className="g-card">
            <div className="section-title">CrackNet Detector (Custom CNN)</div>
            <div className="banner warn">Model not trained yet. Run: <code>python scripts/train_cracknet.py</code></div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelPerformance;
