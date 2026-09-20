import React, { useState, useEffect } from 'react';
import { getModelPerformance } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function Dashboard({ onNavigate }) {
  const [perf, setPerf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => { load(); }, []);

  const load = async () => {
    setLoading(true); setErr(null);
    try { setPerf(await getModelPerformance()); }
    catch { setErr('Cannot reach backend.'); }
    setLoading(false);
  };

  if (loading) return (
    <div className="loader-wrap">
      <div className="loader" />
      <div className="loader-text">Loading models...</div>
    </div>
  );

  if (err) return (
    <div className="anim-fade-up">
      <div className="pg-header"><h1>Dashboard</h1></div>
      <div className="banner error">{err}</div>
      <button className="btn btn-primary" onClick={load}>Retry</button>
    </div>
  );

  const chartData = [
    { name: 'SHI Regressor (R²)', val: perf.shi.r2 * 100, fill: '#7C3AED' },
    { name: 'Risk Classifier (Acc)', val: perf.risk.accuracy * 100, fill: '#E11D48' },
    { name: 'RUL Regressor (R²)', val: perf.rul.r2 * 100, fill: '#059669' },
    { name: 'U-Net (Dice)', val: perf.unet.best_dice * 100, fill: '#D97706' },
    ...(perf.cracknet ? [{ name: 'CrackNet (Acc)', val: perf.cracknet.val_acc * 100, fill: '#0891B2' }] : []),
  ];

  return (
    <div className="anim-fade-up">
      <div className="pg-header pg-header-row">
        <div>
          <h1>Dashboard</h1>
          <p>Structural health monitoring overview</p>
        </div>
        <button className="btn btn-secondary" onClick={load}>Refresh</button>
      </div>

      {/* Feature navigation */}
      <div className="feat-grid stagger">
        {[
          { key: 'detection', title: 'Crack Detection', desc: 'Custom CrackNet CNN detection and U-Net segmentation on uploaded images', color: 'var(--primary)' },
          { key: 'risk', title: 'Risk Predictor', desc: 'Predict structural health index, risk level, and remaining useful life', color: 'var(--rose)' },
          { key: 'crackwidth', title: 'Crack Width Calculator', desc: 'IS 456:2000 Annex F crack width calculation with exposure-based limits', color: '#9333ea' },
          { key: 'performance', title: 'Model Metrics', desc: 'Evaluation metrics, confusion matrices, and model performance', color: 'var(--emerald)' },
          { key: 'explain', title: 'Explainability', desc: 'SHAP waterfall plots and feature importance analysis', color: 'var(--amber)' },
          { key: 'reports', title: 'AI Reports', desc: 'Generate comprehensive inspection reports with AI', color: 'var(--primary)' },
          { key: 'chat', title: 'Chat Inspector', desc: 'Ask questions about your structural analysis results', color: 'var(--rose)' },
        ].map(c => (
          <div key={c.key} className="feat-card" onClick={() => onNavigate(c.key)}>
            <div className="feat-accent" style={{ background: c.color }} />
            <h3>{c.title}</h3>
            <p>{c.desc}</p>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="chart-panel" style={{ marginBottom: 24 }}>
        <div className="section-title">Performance Overview</div>
        <div style={{ height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 24, left: 10, bottom: 0 }}>
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: '#8a839a' }} axisLine={false} tickLine={false} />
              <YAxis dataKey="name" type="category" width={150} tick={{ fontSize: 12, fill: '#544d63', fontWeight: 600 }} axisLine={false} tickLine={false} />
              <Tooltip
                cursor={{ fill: 'rgba(124,58,237,0.04)' }}
                contentStyle={{ background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.5)', borderRadius: 10, fontSize: 13, fontWeight: 600, boxShadow: '0 4px 16px rgba(0,0,0,0.06)' }}
                formatter={(v) => [`${v.toFixed(1)}%`, 'Score']}
              />
              <Bar dataKey="val" radius={[0, 6, 6, 0]} barSize={22}>
                {chartData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Quick metrics */}
      <div className="grid-2">
        <div className="g-card">
          <div className="section-title">SHI Regressor</div>
          <div className="grid-3">
            <div className="m-card purple"><div className="m-label">R²</div><div className="m-value">{perf.shi.r2.toFixed(3)}</div></div>
            <div className="m-card emerald"><div className="m-label">MAE</div><div className="m-value">{perf.shi.mae.toFixed(3)}</div></div>
            <div className="m-card amber"><div className="m-label">RMSE</div><div className="m-value">{perf.shi.rmse.toFixed(3)}</div></div>
          </div>
        </div>
        <div className="g-card">
          <div className="section-title">Risk Classifier</div>
          <div className="grid-3">
            <div className="m-card rose"><div className="m-label">Accuracy</div><div className="m-value">{perf.risk.accuracy.toFixed(3)}</div></div>
            <div className="m-card amber"><div className="m-label">F1</div><div className="m-value">{perf.risk.f1.toFixed(3)}</div></div>
            <div className="m-card purple"><div className="m-label">Precision</div><div className="m-value">{perf.risk.precision.toFixed(3)}</div></div>
          </div>
        </div>
        <div className="g-card">
          <div className="section-title">RUL Regressor</div>
          <div className="grid-3">
            <div className="m-card emerald"><div className="m-label">R²</div><div className="m-value">{perf.rul.r2.toFixed(3)}</div></div>
            <div className="m-card purple"><div className="m-label">MAE</div><div className="m-value">{perf.rul.mae.toFixed(3)}</div></div>
            <div className="m-card amber"><div className="m-label">RMSE</div><div className="m-value">{perf.rul.rmse.toFixed(3)}</div></div>
          </div>
        </div>
        <div className="g-card">
          <div className="section-title">U-Net Segmentation</div>
          <div className="grid-2">
            <div className="m-card rose"><div className="m-label">Dice Score</div><div className="m-value">{perf.unet.best_dice.toFixed(4)}</div></div>
            <div className="m-card emerald"><div className="m-label">Params</div><div className="m-value">{(perf.unet.params / 1000).toFixed(1)}k</div></div>
          </div>
        </div>
        {perf.cracknet && (
          <div className="g-card">
            <div className="section-title">CrackNet Detector</div>
            <div className="grid-2">
              <div className="m-card" style={{textAlign:'center',background:'rgba(8,145,178,0.08)',border:'1px solid rgba(8,145,178,0.2)'}}>
                <div className="m-label">Val Accuracy</div>
                <div className="m-value" style={{color:'#0891B2'}}>{(perf.cracknet.val_acc * 100).toFixed(1)}%</div>
              </div>
              <div className="m-card emerald">
                <div className="m-label">Params</div>
                <div className="m-value">{(perf.cracknet.params / 1e6).toFixed(1)}M</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
