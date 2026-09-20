import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import RiskPredictor from './components/RiskPredictor';
import CrackDetection from './components/CrackDetection';
import ModelPerformance from './components/ModelPerformance';
import Explainability from './components/Explainability';
import Reports from './components/Reports';
import ChatInspector from './components/ChatInspector';
import CrackWidthCalculator from './components/CrackWidthCalculator';
import { healthCheck } from './services/api';

const PAGES = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'detection', label: 'Detection' },
  { key: 'risk', label: 'Risk' },
  { key: 'crackwidth', label: 'Crack Width' },
  { key: 'performance', label: 'Models' },
  { key: 'explain', label: 'XAI' },
  { key: 'reports', label: 'Reports' },
  { key: 'chat', label: 'Chat' },
];

function App() {
  const [page, setPage] = useState('dashboard');
  const [online, setOnline] = useState(false);
  const [inputs, setInputs] = useState(null);
  const [preds, setPreds] = useState(null);
  const [rec, setRec] = useState(null);
  const [detectionData, setDetectionData] = useState(null);

  useEffect(() => {
    ping();
    const t = setInterval(ping, 15000);
    return () => clearInterval(t);
  }, []);

  const ping = async () => {
    try { await healthCheck(); setOnline(true); }
    catch { setOnline(false); }
  };

  const go = (p) => setPage(p);

  const onPredict = (i, p, r) => {
    setInputs(i);
    setPreds(p);
    setRec(r);
  };

  const renderPage = () => {
    switch (page) {
      case 'detection': return <CrackDetection onNavigate={go} onDetectionResult={setDetectionData} />;
      case 'risk': return <RiskPredictor onNavigate={go} onPrediction={onPredict} />;
      case 'crackwidth': return <CrackWidthCalculator onNavigate={go} detectionData={detectionData} />;
      case 'performance': return <ModelPerformance />;
      case 'explain': return <Explainability onNavigate={go} lastInputs={inputs} />;
      case 'reports': return <Reports onNavigate={go} lastInputs={inputs} lastPredictions={preds} lastRecommendation={rec} />;
      case 'chat': return <ChatInspector onNavigate={go} lastRecommendation={rec} />;
      default: return <Dashboard onNavigate={go} />;
    }
  };

  return (
    <div className="app">
      <nav className="floating-nav">
        <div className="nav-brand">StructuralAI</div>
        <div className="nav-divider" />
        {PAGES.map(p => (
          <button
            key={p.key}
            className={`nav-link ${page === p.key ? 'active' : ''}`}
            onClick={() => go(p.key)}
          >
            {p.label}
          </button>
        ))}
        <div className="nav-divider" />
        <div className={`nav-status ${online ? 'online' : 'offline'}`}
             title={online ? 'Backend connected' : 'Backend offline'} />
      </nav>

      <div className="page-wrap">
        {renderPage()}
      </div>
    </div>
  );
}

export default App;
