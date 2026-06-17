// Model Performance Metrics from real training results
export const modelMetrics = {
  accuracy: 0.7968,
  precision: 0.978,
  recall: 0.6578,
  specificity: 0.9804,
  f1Score: 0.7866,
  mcc: 0.6502,
  cohenKappa: 0.6064,
  aucROC: 0.9487,
  aucPR: 0.9616,
  falsePositiveRate: 0.0196,
  negativePredictiveValue: 0.6844,
  confusionMatrix: {
    truePositives: 8442,
    trueNegatives: 9521,
    falsePositives: 190,
    falseNegatives: 4391
  }
};

// Training History from CSV logs
export const trainingHistory = [
  { epoch: 0, accuracy: 0.9521, loss: 0.1868, val_accuracy: 0.9109, val_loss: 0.4180, val_auc: 0.9621 },
  { epoch: 1, accuracy: 0.9789, loss: 0.1096, val_accuracy: 0.9876, val_loss: 0.0881, val_auc: 0.9977 },
  { epoch: 2, accuracy: 0.9841, loss: 0.0886, val_accuracy: 0.9856, val_loss: 0.0802, val_auc: 0.9982 },
  { epoch: 3, accuracy: 0.9859, loss: 0.0761, val_accuracy: 0.9897, val_loss: 0.0652, val_auc: 0.9988 },
  { epoch: 4, accuracy: 0.9871, loss: 0.0686, val_accuracy: 0.9912, val_loss: 0.0604, val_auc: 0.9989 },
  { epoch: 5, accuracy: 0.9882, loss: 0.0608, val_accuracy: 0.9877, val_loss: 0.0644, val_auc: 0.9983 },
  { epoch: 6, accuracy: 0.9885, loss: 0.0557, val_accuracy: 0.9899, val_loss: 0.0502, val_auc: 0.9989 },
  { epoch: 7, accuracy: 0.9888, loss: 0.0528, val_accuracy: 0.9871, val_loss: 0.0562, val_auc: 0.9983 },
  { epoch: 8, accuracy: 0.9898, loss: 0.0483, val_accuracy: 0.9899, val_loss: 0.0462, val_auc: 0.9992 },
  { epoch: 9, accuracy: 0.9899, loss: 0.0457, val_accuracy: 0.9920, val_loss: 0.0403, val_auc: 0.9993 },
  { epoch: 10, accuracy: 0.9904, loss: 0.0434, val_accuracy: 0.9910, val_loss: 0.0401, val_auc: 0.9993 },
  { epoch: 11, accuracy: 0.9908, loss: 0.0408, val_accuracy: 0.9926, val_loss: 0.0367, val_auc: 0.9994 },
  { epoch: 12, accuracy: 0.9904, loss: 0.0411, val_accuracy: 0.9897, val_loss: 0.0411, val_auc: 0.9992 },
  { epoch: 13, accuracy: 0.9911, loss: 0.0390, val_accuracy: 0.9928, val_loss: 0.0354, val_auc: 0.9992 },
  { epoch: 14, accuracy: 0.9915, loss: 0.0365, val_accuracy: 0.9925, val_loss: 0.0340, val_auc: 0.9993 },
  { epoch: 15, accuracy: 0.9914, loss: 0.0372, val_accuracy: 0.9922, val_loss: 0.0358, val_auc: 0.9989 },
  { epoch: 16, accuracy: 0.9918, loss: 0.0360, val_accuracy: 0.9923, val_loss: 0.0353, val_auc: 0.9992 },
  { epoch: 17, accuracy: 0.9918, loss: 0.0352, val_accuracy: 0.9919, val_loss: 0.0334, val_auc: 0.9992 },
  { epoch: 18, accuracy: 0.9921, loss: 0.0340, val_accuracy: 0.9915, val_loss: 0.0353, val_auc: 0.9993 }
];

// Feature Importance from Random Forest analysis
export const featureImportance = [
  { feature: 'src_bytes', importance: 0.1914, description: 'Bytes sent from source to destination' },
  { feature: 'dst_bytes', importance: 0.1013, description: 'Bytes sent from destination to source' },
  { feature: 'same_srv_rate', importance: 0.0877, description: 'Percentage of connections to same service' },
  { feature: 'dst_host_same_srv_rate', importance: 0.0702, description: 'Same service rate to destination host' },
  { feature: 'flag', importance: 0.0700, description: 'Connection status flag (SF, S0, REJ, etc.)' },
  { feature: 'dst_host_srv_count', importance: 0.0650, description: 'Count of connections to same service on host' },
  { feature: 'logged_in', importance: 0.0498, description: 'Whether user is logged in (1=yes, 0=no)' },
  { feature: 'srv_serror_rate', importance: 0.0393, description: 'SYN error rate to same service' },
  { feature: 'protocol_type', importance: 0.0383, description: 'Network protocol (TCP, UDP, ICMP)' },
  { feature: 'diff_srv_rate', importance: 0.0373, description: 'Percentage of connections to different services' },
  { feature: 'serror_rate', importance: 0.0282, description: 'SYN error rate' },
  { feature: 'dst_host_diff_srv_rate', importance: 0.0264, description: 'Different service rate to destination host' },
  { feature: 'service', importance: 0.0258, description: 'Network service (http, ftp, smtp, etc.)' },
  { feature: 'count', importance: 0.0253, description: 'Connections to same host in 2 seconds' },
  { feature: 'dst_host_same_src_port_rate', importance: 0.0244, description: 'Same source port rate to host' }
];

// Dataset statistics
export const datasetStats = {
  totalTrainingSamples: 125973,
  totalTestSamples: 22544,
  numFeatures: 41,
  attackTypes: 23,
  trainingNormalPercent: 53.46,
  trainingAttackPercent: 46.54,
  testNormalPercent: 43.08,
  testAttackPercent: 56.92
};

// Model configuration
export const modelConfig = {
  modelType: 'Bidirectional LSTM',
  totalParameters: 328705,
  inputShape: [41, 1],
  dropoutRates: [0.35, 0.30, 0.25, 0.20, 0.15],
  l2Regularization: 0.0001,
  batchSize: 256,
  maxEpochs: 50,
  actualEpochs: 19,
  trainingTime: '200.13 seconds',
  optimizer: 'Adam',
  learningRate: 0.001,
  classWeights: {
    normal: 0.9353,
    attack: 1.0743
  }
};

// Protocol types
export const protocolTypes = ['tcp', 'udp', 'icmp'];

// Service types
export const serviceTypes = [
  'http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'finger', 'whois',
  'pop_3', 'nntp', 'ftp_data', 'netbios_ns', 'ldap', 'imap', 'auth',
  'iso_tsap', 'uucp', 'echo', 'discard', 'daytime', 'sunrpc', 'name'
];

// Flag types
export const flagTypes = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'SH', 'S1', 'S2', 'S3', 'OTH'];

// Threat level thresholds
export const threatLevels = {
  CRITICAL: { threshold: 0.95, color: '#ff3366' },
  HIGH: { threshold: 0.85, color: '#ff8800' },
  MEDIUM: { threshold: 0.70, color: '#ffcc00' },
  LOW: { threshold: 0, color: '#00ff88' }
};

// Get threat level from confidence score
export function getThreatLevel(confidence: number): string {
  if (confidence >= 0.95) return 'CRITICAL';
  if (confidence >= 0.85) return 'HIGH';
  if (confidence >= 0.70) return 'MEDIUM';
  return 'LOW';
}
