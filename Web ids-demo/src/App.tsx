import { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  Shield, Activity, AlertTriangle, CheckCircle, XCircle, Database,
  Brain, Link, TrendingUp, Eye, Clock, Zap, Lock, ChevronRight, Info,
  Play, Pause, RefreshCw, Server, Network, Code, Award, Target, Layers,
  FileUp, HelpCircle, X, BookOpen, Wifi, Clock3, UserCheck, Gauge, Hash
} from 'lucide-react';
import './App.css';
import { modelMetrics, trainingHistory, featureImportance, datasetStats, modelConfig, getThreatLevel } from './data/modelData';
import { blockchainStats, sampleBlocks, sampleAlerts } from './data/blockchainData';

// Sample data presets from NSL-KDD dataset
const sampleDataPresets = [
  {
    name: 'Normal HTTP',
    description: 'Normal web traffic with successful HTTP connection',
    protocol: 'tcp', service: 'http', flag: 'SF',
    src_bytes: 500, dst_bytes: 2000, duration: 30, logged_in: 1, same_srv_rate: 0.85,
    attack_type: 'normal'
  },
  {
    name: 'Neptune Attack',
    description: 'SYN flood DoS attack - overwhelming server with connections',
    protocol: 'tcp', service: 'private', flag: 'REJ',
    src_bytes: 0, dst_bytes: 0, duration: 0, logged_in: 0, same_srv_rate: 1.0,
    attack_type: 'neptune'
  },
  {
    name: 'Satan Scan',
    description: 'Reconnaissance attack - scanning for vulnerabilities',
    protocol: 'icmp', service: 'eco_i', flag: 'SF',
    src_bytes: 20, dst_bytes: 0, duration: 0, logged_in: 0, same_srv_rate: 0.0,
    attack_type: 'satan'
  },
  {
    name: 'Guess Password',
    description: 'Brute force attack - trying to crack passwords',
    protocol: 'tcp', service: 'telnet', flag: 'SF',
    src_bytes: 129, dst_bytes: 174, duration: 0, logged_in: 0, same_srv_rate: 0.0,
    attack_type: 'guess_passwd'
  },
  {
    name: 'MScan Probe',
    description: 'Port scanning attack - mapping network vulnerabilities',
    protocol: 'tcp', service: 'telnet', flag: 'RSTO',
    src_bytes: 0, dst_bytes: 15, duration: 1, logged_in: 0, same_srv_rate: 0.5,
    attack_type: 'mscan'
  },
  {
    name: 'Smurf Attack',
    description: 'Amplification DoS attack using ICMP packets',
    protocol: 'icmp', service: 'echo', flag: 'SF',
    src_bytes: 100, dst_bytes: 0, duration: 0, logged_in: 0, same_srv_rate: 0.0,
    attack_type: 'smurf'
  }
];

// Info Popup Component
function InfoPopup({ title, children, icon: Icon }: { title: string; children: React.ReactNode; icon: any }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button className="info-btn" onClick={() => setIsOpen(true)} title="Learn more">
        <Icon size={16} />
      </button>
      {isOpen && (
        <div className="popup-overlay" onClick={() => setIsOpen(false)}>
          <div className="popup-content" onClick={e => e.stopPropagation()}>
            <button className="popup-close" onClick={() => setIsOpen(false)}>
              <X size={20} />
            </button>
            <h3>{title}</h3>
            <div className="popup-body">{children}</div>
          </div>
        </div>
      )}
    </>
  );
}

// Info content for different sections
const infoContent = {
  liveMonitoring: {
    title: "Live Monitoring vs Simulation Mode",
    icon: Activity,
    content: (
      <div className="info-content">
        <h4>Simulation Mode (Default)</h4>
        <p>When you click "Detect Threat", the system uses a simplified rule-based model to simulate how the ML model would classify network traffic based on the features you provide.</p>

        <h4>Live Monitoring</h4>
        <p>When enabled, this displays a live clock to show the dashboard is "active". In a production environment, this would connect to a real-time network monitoring system that processes actual traffic.</p>

        <h4>How It Works</h4>
        <p>The prediction simulation analyzes connection patterns like:</p>
        <ul>
          <li><strong>High data transfer</strong> (src_bytes) often indicates data theft or DoS attacks</li>
          <li><strong>Zero duration</strong> connections may suggest automated attacks</li>
          <li><strong>Not logged in</strong> + unusual services can indicate unauthorized access attempts</li>
          <li><strong>Same service rate</strong> anomalies often reveal scanning or probing</li>
        </ul>
      </div>
    )
  },
  prediction: {
    title: "Understanding Network Traffic Prediction",
    icon: Brain,
    content: (
      <div className="info-content">
        <p>Network intrusion detection analyzes connection patterns to identify potential attacks. Each prediction considers multiple features:</p>

        <h4>What Each Input Means:</h4>
        <ul>
          <li><strong>Protocol Type</strong>: The communication rules (TCP=websites, UDP=streaming, ICMP=ping/diagnostics)</li>
          <li><strong>Service</strong>: The application running (HTTP=web, FTP=file transfer, SMTP=email)</li>
          <li><strong>Connection Flag</strong>: Status of the connection (SF=success, S0=no response, REJ=rejected)</li>
          <li><strong>Source/Dest Bytes</strong>: Data volume sent/received - anomalies can indicate attacks</li>
          <li><strong>Duration</strong>: How long the connection lasted - short durations often mean attacks</li>
          <li><strong>Logged In</strong>: Whether user authentication succeeded</li>
          <li><strong>Same Service Rate</strong>: Percentage of connections to the same service - low rates suggest scanning</li>
        </ul>

        <h4>Reading the Results</h4>
        <p><strong>Confidence</strong>: How sure the model is (higher = more confident)</p>
        <p><strong>Threat Level</strong>: CRITICAL (≥95%), HIGH (≥85%), MEDIUM (≥70%), LOW</p>
      </div>
    )
  },
  blockchain: {
    title: "Blockchain for Security Logging",
    icon: Lock,
    content: (
      <div className="info-content">
        <p>Every threat detection is permanently recorded in a blockchain to ensure the logs cannot be tampered with.</p>

        <h4>Why Blockchain?</h4>
        <ul>
          <li><strong>Immutable</strong>: Once recorded, alerts cannot be modified or deleted</li>
          <li><strong>Verifiable</strong>: Each block contains a cryptographic hash linking it to previous blocks</li>
          <li><strong>Traceable</strong>: Complete audit trail from genesis block to latest alert</li>
        </ul>

        <h4>How It Works</h4>
        <p>Each alert creates a "block" containing:</p>
        <ul>
          <li>Timestamp of detection</li>
          <li>Prediction result and confidence</li>
          <li>Hash of all previous blocks (creating a "chain")</li>
          <li>Unique cryptographic signature</li>
        </ul>

        <h4>Threat Levels</h4>
        <p><strong>CRITICAL</strong>: Immediate action required (&gt;=95% confidence)<br/>
        <strong>HIGH</strong>: Significant threat detected (&gt;=85%)<br/>
        <strong>MEDIUM</strong>: Potential concern (&gt;=70%)<br/>
        <strong>LOW</strong>: Minor anomaly (&lt;70%)</p>
      </div>
    )
  },
  metrics: {
    title: "Understanding Model Performance Metrics",
    icon: TrendingUp,
    content: (
      <div className="info-content">
        <h4>Accuracy (79.68%)</h4>
        <p>Overall correctness: 8 out of 10 predictions are correct.</p>

        <h4>Precision (97.8%)</h4>
        <p>When we say "attack", we're right 98% of the time. Very few false alarms!</p>

        <h4>Recall (65.78%)</h4>
        <p>We detect about 66% of all actual attacks. Some attacks slip through.</p>

        <h4>F1 Score (78.66%)</h4>
        <p>Balance between precision and recall - good overall performance.</p>

        <h4>AUC-ROC (94.87%)</h4>
        <p>Excellent! The model has strong ability to distinguish attacks from normal traffic.</p>

        <h4>Specificity (98.04%)</h4>
        <p>We correctly identify 98% of normal traffic as safe - minimal disruption to legitimate users.</p>
      </div>
    )
  },
  confusionMatrix: {
    title: "Confusion Matrix Explained",
    icon: Eye,
    content: (
      <div className="info-content">
        <p>A confusion matrix shows where our predictions went right and wrong:</p>

        <h4>True Negatives (9,521)</h4>
        <p>Normal traffic correctly identified as safe ✓</p>

        <h4>True Positives (8,442)</h4>
        <p>Attacks correctly detected and blocked ✓</p>

        <h4>False Positives (190)</h4>
        <p>Normal traffic wrongly flagged as attack - creates extra work for analysts</p>

        <h4>False Negatives (4,391)</h4>
        <p>Attacks that slipped through - most concerning outcome</p>
      </div>
    )
  },
  featureImportance: {
    title: "What Makes a Connection Suspicious?",
    icon: Layers,
    content: (
      <div className="info-content">
        <p>These are the top features the model uses to identify attacks:</p>

        <h4>Source Bytes (19.14%)</h4>
        <p>How much data is being sent from source. Sudden spikes may indicate data exfiltration or DoS.</p>

        <h4>Destination Bytes (10.13%)</h4>
        <p>Data received. Unusual patterns can indicate response to an attack.</p>

        <h4>Same Service Rate (8.77%)</h4>
        <p>How many connections go to the same service. Low rates suggest network reconnaissance.</p>

        <h4>Connection Flag (7.00%)</h4>
        <p>Status codes like "REJ" (rejected) or "S0" (no SYN-ACK) often indicate failed attacks.</p>

        <h4>Logged In (4.98%)</h4>
        <p>Successful authentication. Logged-out connections to sensitive services are suspicious.</p>
      </div>
    )
  },
  dataset: {
    title: "NSL-KDD Dataset",
    icon: Database,
    content: (
      <div className="info-content">
        <p>NSL-KDD is a standard benchmark dataset for intrusion detection research.</p>

        <h4>Training Data: 125,973 samples</h4>
        <p>Used to teach the model what attacks look like.</p>

        <h4>Test Data: 22,544 samples</h4>
        <p>Used to evaluate how well the model learned.</p>

        <h4>Attack Types (23 total)</h4>
        <ul>
          <li><strong>DoS attacks</strong>: Neptune, Smurf, Teardrop - overwhelm systems</li>
          <li><strong>Probing</strong>: Satan, Nmap - scan for vulnerabilities</li>
          <li><strong>Remote-to-Local</strong>: Buffer overflow, guess_password - gain unauthorized access</li>
        </ul>

        <h4>41 Features</h4>
        <p>Each connection is described by 41 attributes including duration, bytes transferred, protocol, and more.</p>
      </div>
    )
  }
};

// Animation hook
function useCountUp(end: number, duration: number = 2000, start: number = 0) {
  const [count, setCount] = useState(start);

  useEffect(() => {
    const startTime = Date.now();
    const timer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      setCount(start + (end - start) * easeOut);
      if (progress >= 1) {
        clearInterval(timer);
      }
    }, 16);
    return () => clearInterval(timer);
  }, [end, duration, start]);

  return count;
}

// Metric Card Component
function MetricCard({ icon: Icon, label, value, unit = '%', color, delay = 0, infoKey }: any) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  const numericValue = typeof value === 'number' ? value : parseFloat(value);
  const count = useCountUp(numericValue * 100, 2000);

  return (
    <div className={`metric-card ${isVisible ? 'visible' : ''}`} style={{ '--delay': `${delay}ms` } as React.CSSProperties}>
      <div className="metric-icon" style={{ color }}>
        <Icon size={24} />
      </div>
      <div className="metric-content">
        <span className="metric-label">
          {label}
          {infoKey && <InfoPopup title={infoContent[infoKey].title} icon={infoContent[infoKey].icon}>{infoContent[infoKey].content}</InfoPopup>}
        </span>
        <span className="metric-value" style={{ color }}>
          {(count).toFixed(1)}{unit}
        </span>
      </div>
      <div className="metric-glow" style={{ background: `radial-gradient(circle at center, ${color}20 0%, transparent 70%)` }} />
    </div>
  );
}

// Confusion Matrix Component
function ConfusionMatrix() {
  const { truePositives, trueNegatives, falsePositives, falseNegatives } = modelMetrics.confusionMatrix;

  const cells = [
    { value: trueNegatives, label: 'TN', desc: 'True Normal', color: '#00ff88' },
    { value: falsePositives, label: 'FP', desc: 'False Alarm', color: '#ff3366' },
    { value: falseNegatives, label: 'FN', desc: 'Missed Attack', color: '#ff8800' },
    { value: truePositives, label: 'TP', desc: 'Detected Attack', color: '#00d4ff' }
  ];

  return (
    <div className="confusion-matrix-container">
      <h3 className="section-title">
        <Eye size={18} />
        Confusion Matrix
        <InfoPopup title={infoContent.confusionMatrix.title} icon={infoContent.confusionMatrix.icon}>
          {infoContent.confusionMatrix.content}
        </InfoPopup>
      </h3>
      <div className="confusion-matrix">
        <div className="matrix-labels-y">
          <span>Actual</span>
          <div className="y-labels">
            <span>Normal</span>
            <span>Attack</span>
          </div>
        </div>
        <div className="matrix-content">
          <div className="matrix-labels-x">
            <span>Predicted Normal</span>
            <span>Predicted Attack</span>
          </div>
          <div className="matrix-cells">
            {cells.map((cell, i) => (
              <div key={i} className="matrix-cell" style={{ '--cell-color': cell.color } as React.CSSProperties}>
                <span className="cell-label">{cell.label}</span>
                <span className="cell-value">{cell.value.toLocaleString()}</span>
                <span className="cell-desc">{cell.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Threat Distribution Pie Chart
function ThreatDistribution() {
  const data = [
    { name: 'Normal Traffic', value: 43.08, color: '#00d4ff' },
    { name: 'Attack Traffic', value: 56.92, color: '#ff3366' }
  ];

  return (
    <div className="chart-container">
      <h3 className="section-title">
        <Activity size={18} />
        Traffic Distribution
      </h3>
      <div className="donut-chart-wrapper">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              animationDuration={1500}
              animationBegin={500}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: '#131a2e', border: '1px solid #00d4ff30', borderRadius: '8px' }}
              formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="donut-center">
          <span className="donut-label">Test Set</span>
          <span className="donut-value">22,544</span>
        </div>
      </div>
      <div className="chart-legend">
        {data.map((item, i) => (
          <div key={i} className="legend-item">
            <span className="legend-dot" style={{ background: item.color }} />
            <span className="legend-label">{item.name}</span>
            <span className="legend-value">{item.value.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Training History Chart
function TrainingHistoryChart() {
  return (
    <div className="chart-container full-width">
      <h3 className="section-title">
        <TrendingUp size={18} />
        Training Progress
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={trainingHistory} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorValAcc" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
          <XAxis dataKey="epoch" stroke="#8892a6" tick={{ fontSize: 12 }} />
          <YAxis stroke="#8892a6" tick={{ fontSize: 12 }} domain={[0.85, 1]} />
          <Tooltip
            contentStyle={{ background: '#131a2e', border: '1px solid #00d4ff30', borderRadius: '8px' }}
            labelStyle={{ color: '#fff' }}
          />
          <Legend />
          <Area type="monotone" dataKey="accuracy" stroke="#00d4ff" fillOpacity={1} fill="url(#colorAcc)" name="Training Accuracy" strokeWidth={2} />
          <Area type="monotone" dataKey="val_accuracy" stroke="#00ff88" fillOpacity={1} fill="url(#colorValAcc)" name="Validation Accuracy" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// Feature Importance Chart
function FeatureImportanceChart() {
  const topFeatures = featureImportance.slice(0, 10);

  return (
    <div className="chart-container">
      <h3 className="section-title">
        <Layers size={18} />
        Feature Importance
        <InfoPopup title={infoContent.featureImportance.title} icon={infoContent.featureImportance.icon}>
          {infoContent.featureImportance.content}
        </InfoPopup>
      </h3>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={topFeatures} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" horizontal={false} />
          <XAxis type="number" stroke="#8892a6" tick={{ fontSize: 11 }} domain={[0, 0.25]} />
          <YAxis dataKey="feature" type="category" stroke="#8892a6" tick={{ fontSize: 11 }} width={75} />
          <Tooltip
            contentStyle={{ background: '#131a2e', border: '1px solid #00d4ff30', borderRadius: '8px' }}
            formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Importance']}
          />
          <Bar dataKey="importance" fill="#00d4ff" radius={[0, 4, 4, 0]} animationDuration={1500}>
            {topFeatures.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={`hsl(190, 100%, ${60 - index * 4}%)`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Blockchain Visualization with dynamic updates
function BlockchainVisualization({ newBlocks }: { newBlocks: any[] }) {
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null);
  const [verifying, setVerifying] = useState<number | null>(null);

  const allBlocks = [...sampleBlocks.slice(0, 10), ...newBlocks.slice(-5)];

  const handleVerify = (index: number) => {
    setVerifying(index);
    setTimeout(() => setVerifying(null), 1500);
  };

  return (
    <div className="blockchain-container">
      <h3 className="section-title">
        <Lock size={18} />
        Blockchain Alert Chain
        <InfoPopup title={infoContent.blockchain.title} icon={infoContent.blockchain.icon}>
          {infoContent.blockchain.content}
        </InfoPopup>
      </h3>
      <div className="blockchain-stats">
        <div className="stat-item">
          <span className="stat-value">{blockchainStats.totalBlocks + newBlocks.length}</span>
          <span className="stat-label">Total Blocks</span>
        </div>
        <div className="stat-item">
          <span className="stat-value" style={{ color: '#ff3366' }}>{blockchainStats.threatDistribution.CRITICAL + newBlocks.filter(b => b.threat_level === 'CRITICAL').length}</span>
          <span className="stat-label">Critical Alerts</span>
        </div>
        <div className="stat-item">
          <span className="stat-value" style={{ color: '#00ff88' }}>{blockchainStats.dataIntegrity}</span>
          <span className="stat-label">Data Integrity</span>
        </div>
      </div>
      <div className="chain-visualization">
        {allBlocks.map((block, index) => (
          <div key={block.index} className={`chain-block ${selectedBlock === index ? 'selected' : ''} ${index >= 10 ? 'new-block' : ''}`}>
            <div
              className="block-content"
              onClick={() => setSelectedBlock(selectedBlock === index ? null : index)}
              style={{ '--threat-color': block.threat_level === 'CRITICAL' ? '#ff3366' : block.threat_level === 'HIGH' ? '#ff8800' : block.threat_level === 'MEDIUM' ? '#ffcc00' : '#00ff88' } as React.CSSProperties}
            >
              <div className="block-index">#{block.index}</div>
              {block.type === 'GENESIS' ? (
                <div className="block-genesis">
                  <span className="genesis-label">GENESIS</span>
                  <span className="genesis-desc">Block Created</span>
                </div>
              ) : (
                <>
                  <div className="block-threat" style={{
                    background: block.threat_level === 'CRITICAL' ? '#ff336620' :
                                block.threat_level === 'HIGH' ? '#ff880020' :
                                block.threat_level === 'MEDIUM' ? '#ffcc0020' : '#00ff8820',
                    borderColor: block.threat_level === 'CRITICAL' ? '#ff3366' :
                                block.threat_level === 'HIGH' ? '#ff8800' :
                                block.threat_level === 'MEDIUM' ? '#ffcc00' : '#00ff88'
                  }}>
                    {block.threat_level}
                  </div>
                  <div className="block-confidence">{(block.confidence_score * 100).toFixed(1)}%</div>
                </>
              )}
              <div className="block-hash" title={block.hash}>
                {block.hash.slice(0, 8)}...
              </div>
              {index < allBlocks.length - 1 && (
                <div className="chain-connector">
                  <ChevronRight size={14} />
                </div>
              )}
            </div>
            {selectedBlock === index && (
              <div className="block-details">
                <div className="detail-row">
                  <span>Hash:</span>
                  <code>{block.hash.slice(0, 16)}...</code>
                </div>
                <div className="detail-row">
                  <span>Previous:</span>
                  <code>{block.previous_hash.slice(0, 16)}...</code>
                </div>
                {block.type !== 'GENESIS' && (
                  <div className="detail-row">
                    <span>Prediction:</span>
                    <span>{block.prediction}</span>
                  </div>
                )}
                <button
                  className={`verify-btn ${verifying === index ? 'verifying' : ''}`}
                  onClick={(e) => { e.stopPropagation(); handleVerify(index); }}
                >
                  <Shield size={14} />
                  {verifying === index ? 'Verifying...' : 'Verify'}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Interactive Prediction Panel with sample presets and file import
function PredictionPanel({ onPrediction, isLive }: { onPrediction: (prediction: any) => void; isLive: boolean }) {
  const [inputs, setInputs] = useState({
    protocol: 'tcp',
    service: 'http',
    flag: 'SF',
    src_bytes: 500,
    dst_bytes: 200,
    duration: 0,
    logged_in: 1,
    same_srv_rate: 0.5
  });
  const [prediction, setPrediction] = useState<{ result: string; confidence: number } | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handlePredict = () => {
    setIsPredicting(true);

    setTimeout(() => {
      let score = 0.5;

      if (inputs.src_bytes > 5000) score += 0.3;
      else if (inputs.src_bytes > 1000) score += 0.15;

      if (inputs.duration === 0) score += 0.1;

      if (['http', 'ftp', 'smtp'].includes(inputs.service)) score += 0.05;

      if (inputs.logged_in === 0) score += 0.1;

      if (inputs.same_srv_rate < 0.2) score += 0.1;

      score = Math.max(0.1, Math.min(0.99, score));

      const result = {
        result: score > 0.5 ? 'Attack' : 'Normal',
        confidence: score,
        timestamp: new Date().toISOString(),
        features: { ...inputs }
      };

      setPrediction(result);
      setIsPredicting(false);
      onPrediction(result);
    }, 1500);
  };

  const loadPreset = (preset: typeof sampleDataPresets[0]) => {
    setInputs({
      protocol: preset.protocol,
      service: preset.service,
      flag: preset.flag,
      src_bytes: preset.src_bytes,
      dst_bytes: preset.dst_bytes,
      duration: preset.duration,
      logged_in: preset.logged_in,
      same_srv_rate: preset.same_srv_rate
    });
    setShowPresets(false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const lines = text.trim().split('\n');

      // Parse first line (skip header if present)
      let lineToParse = lines[0];
      if (lines[0].includes(',')) {
        // Check if it's a header
        const firstCell = lines[0].split(',')[0];
        if (isNaN(parseFloat(firstCell))) {
          lineToParse = lines[1] || lines[0];
        }
      }

      const values = lineToParse.split(',');
      if (values.length >= 42) {
        setInputs({
          protocol: values[1] || 'tcp',
          service: values[2] || 'http',
          flag: values[3] || 'SF',
          src_bytes: parseInt(values[4]) || 0,
          dst_bytes: parseInt(values[5]) || 0,
          duration: parseInt(values[0]) || 0,
          logged_in: parseInt(values[11]) || 0,
          same_srv_rate: parseFloat(values[28]) || 0.5
        });
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="prediction-panel">
      <h3 className="section-title">
        <Brain size={18} />
        Live Prediction
        <InfoPopup title={infoContent.prediction.title} icon={infoContent.prediction.icon}>
          {infoContent.prediction.content}
        </InfoPopup>
      </h3>

      {/* Sample Presets */}
      <div className="presets-section">
        <button className="presets-toggle" onClick={() => setShowPresets(!showPresets)}>
          <BookOpen size={16} />
          {showPresets ? 'Hide' : 'Show'} Sample Data
        </button>
        {showPresets && (
          <div className="presets-grid">
            {sampleDataPresets.map((preset, i) => (
              <button key={i} className="preset-card" onClick={() => loadPreset(preset)}>
                <span className="preset-name">{preset.name}</span>
                <span className="preset-desc">{preset.description}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* File Import */}
      <div className="file-import-section">
        <input
          type="file"
          ref={fileInputRef}
          accept=".txt,.csv"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        <button className="file-import-btn" onClick={() => fileInputRef.current?.click()}>
          <FileUp size={16} />
          Import from KDD File
        </button>
        <span className="file-hint">Supports .txt or .csv format (NSL-KDD format)</span>
      </div>

      <div className="prediction-content">
        <div className="input-section">
          <div className="input-row">
            <label>
              <Wifi size={14} /> Protocol Type
            </label>
            <select
              value={inputs.protocol}
              onChange={(e) => setInputs({...inputs, protocol: e.target.value})}
            >
              <option value="tcp">TCP - Web/Email/File Transfer</option>
              <option value="udp">UDP - Streaming/DNS</option>
              <option value="icmp">ICMP - Ping/Network Diagnostics</option>
            </select>
          </div>
          <div className="input-row">
            <label>
              <Server size={14} /> Service
            </label>
            <select
              value={inputs.service}
              onChange={(e) => setInputs({...inputs, service: e.target.value})}
            >
              <option value="http">HTTP - Web Traffic</option>
              <option value="ftp">FTP - File Transfer</option>
              <option value="smtp">SMTP - Email</option>
              <option value="ssh">SSH - Secure Shell</option>
              <option value="dns">DNS - Domain Lookup</option>
              <option value="telnet">Telnet - Remote Access</option>
              <option value="private">Private - Unknown/Blocked</option>
              <option value="eco_i">Echo - ICMP Echo</option>
            </select>
          </div>
          <div className="input-row">
            <label>
              <Hash size={14} /> Connection Flag
            </label>
            <select
              value={inputs.flag}
              onChange={(e) => setInputs({...inputs, flag: e.target.value})}
            >
              <option value="SF">SF - Complete (Normal)</option>
              <option value="S0">S0 - SYN Sent (No Response)</option>
              <option value="REJ">REJ - Connection Rejected</option>
              <option value="RSTO">RSTO - Reset by Originator</option>
            </select>
          </div>
          <div className="slider-row">
            <label><Gauge size={14} /> Source Bytes: {inputs.src_bytes}</label>
            <span className="slider-hint">Data sent from source (high = possible data theft)</span>
            <input
              type="range"
              min="0"
              max="10000"
              value={inputs.src_bytes}
              onChange={(e) => setInputs({...inputs, src_bytes: parseInt(e.target.value)})}
            />
          </div>
          <div className="slider-row">
            <label><Gauge size={14} /> Dest Bytes: {inputs.dst_bytes}</label>
            <span className="slider-hint">Data received by destination</span>
            <input
              type="range"
              min="0"
              max="10000"
              value={inputs.dst_bytes}
              onChange={(e) => setInputs({...inputs, dst_bytes: parseInt(e.target.value)})}
            />
          </div>
          <div className="slider-row">
            <label><Clock3 size={14} /> Duration: {inputs.duration}s</label>
            <span className="slider-hint">Connection length (0 = automated attack)</span>
            <input
              type="range"
              min="0"
              max="500"
              value={inputs.duration}
              onChange={(e) => setInputs({...inputs, duration: parseInt(e.target.value)})}
            />
          </div>
          <div className="input-row toggle-row">
            <label><UserCheck size={14} /> Logged In</label>
            <button
              className={`toggle-btn ${inputs.logged_in ? 'active' : ''}`}
              onClick={() => setInputs({...inputs, logged_in: inputs.logged_in ? 0 : 1})}
            >
              {inputs.logged_in ? 'Yes - Authenticated' : 'No - Not Logged In'}
            </button>
          </div>
          <div className="slider-row">
            <label><Activity size={14} /> Same Srv Rate: {inputs.same_srv_rate.toFixed(2)}</label>
            <span className="slider-hint">% connections to same service (low = possible scan)</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={inputs.same_srv_rate}
              onChange={(e) => setInputs({...inputs, same_srv_rate: parseFloat(e.target.value)})}
            />
          </div>
          <button className="predict-btn" onClick={handlePredict} disabled={isPredicting}>
            {isPredicting ? (
              <>
                <RefreshCw className="spin" size={18} />
                Analyzing...
              </>
            ) : (
              <>
                <Zap size={18} />
                Detect Threat
              </>
            )}
          </button>
        </div>
        <div className="result-section">
          {prediction ? (
            <div className={`prediction-result ${prediction.result.toLowerCase()}`}>
              <div className="result-icon">
                {prediction.result === 'Normal' ? (
                  <CheckCircle size={64} />
                ) : (
                  <AlertTriangle size={64} />
                )}
              </div>
              <div className="result-label">{prediction.result}</div>
              <div className="confidence-ring">
                <svg viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="#ffffff10" strokeWidth="8" />
                  <circle
                    cx="50" cy="50" r="45" fill="none"
                    stroke={prediction.result === 'Normal' ? '#00ff88' : '#ff3366'}
                    strokeWidth="8"
                    strokeDasharray={`${prediction.confidence * 283} 283`}
                    strokeLinecap="round"
                    transform="rotate(-90 50 50)"
                  />
                </svg>
                <span className="confidence-value">{(prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className={`threat-badge ${getThreatLevel(prediction.confidence).toLowerCase()}`}>
                {getThreatLevel(prediction.confidence)} THREAT
              </div>
              {isLive && (
                <span className="live-indicator">
                  <span className="live-dot" /> Added to blockchain
                </span>
              )}
            </div>
          ) : (
            <div className="prediction-placeholder">
              <Brain size={48} />
              <span>Configure features and click "Detect Threat"</span>
              <span className="placeholder-hint">Or select a sample from above</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Alert Log Table with dynamic updates
function AlertLogTable({ newAlerts }: { newAlerts: any[] }) {
  const [filter, setFilter] = useState('all');
  const [page, setPage] = useState(1);
  const perPage = 10;

  // Combine sample alerts with new alerts
  const allAlerts = [...newAlerts.reverse(), ...sampleAlerts];

  const filteredAlerts = filter === 'all'
    ? allAlerts
    : allAlerts.filter((a: any) => a.threatLevel.toLowerCase() === filter);

  const paginatedAlerts = filteredAlerts.slice((page - 1) * perPage, page * perPage);
  const totalPages = Math.ceil(filteredAlerts.length / perPage);

  return (
    <div className="alert-table-container">
      <h3 className="section-title">
        <Database size={18} />
        Alert Log
        <span className="alert-count">({allAlerts.length} total)</span>
      </h3>
      <div className="table-filters">
        <button className={filter === 'all' ? 'active' : ''} onClick={() => { setFilter('all'); setPage(1); }}>
          All ({allAlerts.length})
        </button>
        <button className={filter === 'critical' ? 'active' : ''} onClick={() => { setFilter('critical'); setPage(1); }} style={{ color: '#ff3366' }}>
          Critical ({allAlerts.filter((a: any) => a.threatLevel === 'CRITICAL').length})
        </button>
        <button className={filter === 'high' ? 'active' : ''} onClick={() => { setFilter('high'); setPage(1); }} style={{ color: '#ff8800' }}>
          High ({allAlerts.filter((a: any) => a.threatLevel === 'HIGH').length})
        </button>
        <button className={filter === 'medium' ? 'active' : ''} onClick={() => { setFilter('medium'); setPage(1); }} style={{ color: '#ffcc00' }}>
          Medium ({allAlerts.filter((a: any) => a.threatLevel === 'MEDIUM').length})
        </button>
        <button className={filter === 'low' ? 'active' : ''} onClick={() => { setFilter('low'); setPage(1); }} style={{ color: '#00ff88' }}>
          Low ({allAlerts.filter((a: any) => a.threatLevel === 'LOW').length})
        </button>
      </div>
      <div className="table-wrapper">
        <table className="alert-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>Result</th>
              <th>Confidence</th>
              <th>Threat</th>
            </tr>
          </thead>
          <tbody>
            {paginatedAlerts.map((alert: any, i) => (
              <tr key={alert.id || i} className={i < newAlerts.length && newAlerts.length > 0 ? 'new-alert' : ''}>
                <td>#{alert.blockIndex || (allAlerts.length - i)}</td>
                <td className="timestamp">{alert.timestamp?.slice(11, 19) || new Date().toLocaleTimeString()}</td>
                <td>
                  <span className={`prediction-badge ${alert.prediction.toLowerCase()}`}>
                    {alert.prediction === 'Attack' ? <XCircle size={14} /> : <CheckCircle size={14} />}
                    {alert.prediction}
                  </span>
                </td>
                <td>{(alert.confidence * 100).toFixed(1)}%</td>
                <td>
                  <span className={`threat-badge ${alert.threatLevel.toLowerCase()}`}>
                    {alert.threatLevel}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>
          Previous
        </button>
        <span>Page {page} of {Math.max(1, totalPages)}</span>
        <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages}>
          Next
        </button>
      </div>
    </div>
  );
}

// Main App Component
function App() {
  const [isLive, setIsLive] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [newBlocks, setNewBlocks] = useState<any[]>([]);
  const [newAlerts, setNewAlerts] = useState<any[]>([]);
  const [blockCounter, setBlockCounter] = useState(15);

  useEffect(() => {
    if (isLive) {
      const timer = setInterval(() => setCurrentTime(new Date()), 1000);
      return () => clearInterval(timer);
    }
  }, [isLive]);

  const handlePrediction = (prediction: any) => {
    const threatLevel = getThreatLevel(prediction.confidence);
    const hash = Array.from({length: 64}, () => Math.floor(Math.random() * 16).toString(16)).join('');

    const newBlock = {
      index: blockCounter,
      timestamp: prediction.timestamp,
      type: 'ATTACK_ALERT',
      confidence_score: prediction.confidence,
      threat_level: threatLevel,
      prediction: prediction.result,
      true_label: prediction.result,
      is_true_positive: true,
      hash: hash,
      previous_hash: newBlocks.length > 0 ? newBlocks[newBlocks.length - 1].hash : sampleBlocks[sampleBlocks.length - 1].hash
    };

    const newAlert = {
      id: blockCounter,
      blockIndex: blockCounter,
      timestamp: prediction.timestamp,
      prediction: prediction.result,
      confidence: prediction.confidence,
      threatLevel: threatLevel,
      type: prediction.result === 'Attack' ? 'detected' : 'normal'
    };

    setNewBlocks(prev => [...prev, newBlock]);
    setNewAlerts(prev => [...prev, newAlert]);
    setBlockCounter(prev => prev + 1);
  };

  return (
    <div className="app">
      {/* Background Effects */}
      <div className="bg-grid" />
      <div className="bg-glow glow-1" />
      <div className="bg-glow glow-2" />

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Shield className="logo-icon" size={32} />
            <div className="logo-text">
              <span className="logo-title">IDS Demo</span>
              <span className="logo-subtitle">Network Intrusion Detection System</span>
            </div>
          </div>
          <div className="header-center">
            <div className="status-badge">
              <span className={`status-dot ${isLive ? 'live' : ''}`} />
              <span>{isLive ? 'Live Monitoring' : 'Simulation Mode'}</span>
              <InfoPopup title={infoContent.liveMonitoring.title} icon={infoContent.liveMonitoring.icon}>
                {infoContent.liveMonitoring.content}
              </InfoPopup>
            </div>
            {isLive && (
              <span className="live-time">{currentTime.toLocaleTimeString()}</span>
            )}
          </div>
          <div className="header-right">
            <button className="live-toggle" onClick={() => setIsLive(!isLive)}>
              {isLive ? <Pause size={16} /> : <Play size={16} />}
              {isLive ? 'Pause' : 'Go Live'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Hero Section */}
        <section className="hero-section">
          <div className="hero-content">
            <h1>
              <Brain size={48} />
              Advanced Deep Learning Intrusion Detection
            </h1>
            <p className="hero-subtitle">
              Bidirectional LSTM with Blockchain Security Framework
            </p>
            <div className="hero-stats">
              <div className="hero-stat">
                <span className="stat-value">{modelMetrics.aucROC.toFixed(3)}</span>
                <span className="stat-label">AUC-ROC Score</span>
              </div>
              <div className="hero-stat">
                <span className="stat-value">{modelConfig.totalParameters.toLocaleString()}</span>
                <span className="stat-label">Model Parameters</span>
              </div>
              <div className="hero-stat">
                <span className="stat-value">{blockchainStats.totalBlocks + newBlocks.length}</span>
                <span className="stat-label">Blockchain Blocks</span>
              </div>
            </div>
          </div>
        </section>

        {/* Metrics Grid */}
        <section className="metrics-section">
          <MetricCard icon={Target} label="Accuracy" value={modelMetrics.accuracy} color="#00d4ff" delay={0} infoKey="metrics" />
          <MetricCard icon={Award} label="Precision" value={modelMetrics.precision} color="#00ff88" delay={100} />
          <MetricCard icon={Activity} label="Recall" value={modelMetrics.recall} color="#ff8800" delay={200} />
          <MetricCard icon={TrendingUp} label="F1 Score" value={modelMetrics.f1Score} color="#ffcc00" delay={300} />
          <MetricCard icon={Zap} label="AUC-ROC" value={modelMetrics.aucROC} color="#00d4ff" delay={400} />
          <MetricCard icon={Shield} label="Specificity" value={modelMetrics.specificity} color="#00ff88" delay={500} />
        </section>

        {/* Two Column Layout */}
        <section className="two-column-section">
          <div className="left-column">
            <ConfusionMatrix />
            <ThreatDistribution />
          </div>
          <div className="right-column">
            <PredictionPanel onPrediction={handlePrediction} isLive={isLive} />
          </div>
        </section>

        {/* Training History */}
        <TrainingHistoryChart />

        {/* Three Column Layout */}
        <section className="three-column-section">
          <FeatureImportanceChart />
          <div className="model-info">
            <h3 className="section-title">
              <Code size={18} />
              Model Architecture
              <InfoPopup title="Deep Learning Model" icon={Brain}>
                <div className="info-content">
                  <h4>Bidirectional LSTM</h4>
                  <p>Long Short-Term Memory networks are specialized neural networks that can learn long-term dependencies. Bidirectional means it processes data in both forward and backward directions for better context understanding.</p>

                  <h4>Architecture Details</h4>
                  <ul>
                    <li>3 LSTM layers with dropout regularization</li>
                    <li>Batch normalization for training stability</li>
                    <li>328,705 trainable parameters</li>
                    <li>Adam optimizer with gradient clipping</li>
                  </ul>
                </div>
              </InfoPopup>
            </h3>
            <div className="info-cards">
              <div className="info-card">
                <Network size={20} />
                <div>
                  <span className="info-label">Architecture</span>
                  <span className="info-value">{modelConfig.modelType}</span>
                </div>
              </div>
              <div className="info-card">
                <Server size={20} />
                <div>
                  <span className="info-label">Input Shape</span>
                  <span className="info-value">{modelConfig.inputShape.join(' × ')}</span>
                </div>
              </div>
              <div className="info-card">
                <Layers size={20} />
                <div>
                  <span className="info-label">Parameters</span>
                  <span className="info-value">{modelConfig.totalParameters.toLocaleString()}</span>
                </div>
              </div>
              <div className="info-card">
                <Clock size={20} />
                <div>
                  <span className="info-label">Training Time</span>
                  <span className="info-value">{modelConfig.trainingTime}</span>
                </div>
              </div>
              <div className="info-card">
                <Zap size={20} />
                <div>
                  <span className="info-label">Optimizer</span>
                  <span className="info-value">{modelConfig.optimizer}</span>
                </div>
              </div>
              <div className="info-card">
                <Award size={20} />
                <div>
                  <span className="info-label">Batch Size</span>
                  <span className="info-value">{modelConfig.batchSize}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Blockchain */}
        <BlockchainVisualization newBlocks={newBlocks} />

        {/* Alert Log */}
        <AlertLogTable newAlerts={newAlerts} />

        {/* Dataset Info */}
        <section className="dataset-section">
          <h3 className="section-title">
            <Database size={18} />
            Dataset Statistics
            <InfoPopup title={infoContent.dataset.title} icon={infoContent.dataset.icon}>
              {infoContent.dataset.content}
            </InfoPopup>
          </h3>
          <div className="dataset-grid">
            <div className="dataset-card">
              <span className="dataset-value">{datasetStats.totalTrainingSamples.toLocaleString()}</span>
              <span className="dataset-label">Training Samples</span>
            </div>
            <div className="dataset-card">
              <span className="dataset-value">{datasetStats.totalTestSamples.toLocaleString()}</span>
              <span className="dataset-label">Test Samples</span>
            </div>
            <div className="dataset-card">
              <span className="dataset-value">{datasetStats.numFeatures}</span>
              <span className="dataset-label">Features</span>
            </div>
            <div className="dataset-card">
              <span className="dataset-value">{datasetStats.attackTypes}</span>
              <span className="dataset-label">Attack Types</span>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>Network Intrusion Detection System | NSL-KDD Dataset | Bidirectional LSTM</p>
          <p className="footer-note">Class Project Demo - Powered by TensorFlow & Blockchain</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
