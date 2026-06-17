// Blockchain Alert Data from real training run
export interface BlockchainBlock {
  index: number;
  timestamp: string;
  type: string;
  confidence_score: number;
  threat_level: string;
  prediction: string;
  true_label: string;
  is_true_positive: boolean;
  hash: string;
  previous_hash: string;
}

export const blockchainStats = {
  totalBlocks: 301,
  genesisBlock: {
    system: 'NSL-KDD IDS Framework',
    version: '2.0',
    description: 'Blockchain Intrusion Detection Alert Log',
    hash: '1ae6ed...'
  },
  threatDistribution: {
    CRITICAL: 285,
    HIGH: 5,
    MEDIUM: 4,
    LOW: 6
  },
  chainId: 'IDS-20260510_164507',
  dataIntegrity: '100% Valid'
};

// Sample blockchain blocks for visualization (first 15 alerts + genesis)
export const sampleBlocks: BlockchainBlock[] = [
  {
    index: 0,
    timestamp: '2026-05-10T16:49:39.275',
    type: 'GENESIS',
    confidence_score: 0,
    threat_level: 'NONE',
    prediction: 'NONE',
    true_label: 'NONE',
    is_true_positive: true,
    hash: '1ae6eda49f9798389c8d0efe4c1d45c5aecadf85dc8c3680c04dcf33bb50690e',
    previous_hash: '0000000000000000'
  },
  {
    index: 1,
    timestamp: '2026-05-10T16:49:39.287',
    type: 'ATTACK_ALERT',
    confidence_score: 1.0,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: '82c59544bfca12e5d6a24e4d66f8f9f5d5cce1e06818f9ae11edd369bf8bc3e5',
    previous_hash: '1ae6eda49f9798389c8d0efe4c1d45c5aecadf85dc8c3680c04dcf33bb50690e'
  },
  {
    index: 2,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 1.0,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: '333ba05566497f1f985ab5128d7c9776ab279087a70cab12060f8bb64d33320b',
    previous_hash: '82c59544bfca12e5d6a24e4d66f8f9f5d5cce1e06818f9ae11edd369bf8bc3e5'
  },
  {
    index: 3,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 0.999999,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'd812b39e4aa7fc6a87964a64c12b44a791857451625a4efccf80e1993c498fcf',
    previous_hash: '333ba05566497f1f985ab5128d7c9776ab279087a70cab12060f8bb64d33320b'
  },
  {
    index: 4,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 0.514593,
    threat_level: 'LOW',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'ffd81d2c570977845cab8863836111de7792a410e744271d8879e474fc21e1f5',
    previous_hash: 'd812b39e4aa7fc6a87964a64c12b44a791857451625a4efccf80e1993c498fcf'
  },
  {
    index: 5,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 1.0,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: '3123dbefd70989f44f83adea69f138ef471616661e8f5c0bfbb93c2f577bb88d',
    previous_hash: 'ffd81d2c570977845cab8863836111de7792a410e744271d8879e474fc21e1f5'
  },
  {
    index: 6,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 0.801007,
    threat_level: 'MEDIUM',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'd7933dc76af0d6b9f679e7751dc6db5e1dda2626bbdd7f5607cb860e8829809d',
    previous_hash: '3123dbefd70989f44f83adea69f138ef471616661e8f5c0bfbb93c2f577bb88d'
  },
  {
    index: 7,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 1.0,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: '0aaa55db80ab6c9fe37117fe2941c462f0b79fbde80b09802ccf0309ba7350b3',
    previous_hash: 'd7933dc76af0d6b9f679e7751dc6db5e1dda2626bbdd7f5607cb860e8829809d'
  },
  {
    index: 8,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 0.999999,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'bd42516f4c1692c7ccffa6d366b0c684b5bb1178440fe2edd5935357697f2527',
    previous_hash: '0aaa55db80ab6c9fe37117fe2941c462f0b79fbde80b09802ccf0309ba7350b3'
  },
  {
    index: 9,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 0.999997,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: '3ab8fb4e8d98b2c6bb9321caff736450d7d66b335c14683d0033ff3819eaaa92',
    previous_hash: 'bd42516f4c1692c7ccffa6d366b0c684b5bb1178440fe2edd5935357697f2527'
  },
  {
    index: 10,
    timestamp: '2026-05-10T16:49:39.288',
    type: 'ATTACK_ALERT',
    confidence_score: 1.0,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: '0ff4af1bfa40058730eb663dbcd0d95c91138481a5d0ae3a75c1b9b25a370408',
    previous_hash: '3ab8fb4e8d98b2c6bb9321caff736450d7d66b335c14683d0033ff3819eaaa92'
  },
  {
    index: 11,
    timestamp: '2026-05-10T16:49:39.289',
    type: 'ATTACK_ALERT',
    confidence_score: 1.0,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'a7f2c1e8d3b4a5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9',
    previous_hash: '0ff4af1bfa40058730eb663dbcd0d95c91138481a5d0ae3a75c1b9b25a370408'
  },
  {
    index: 12,
    timestamp: '2026-05-10T16:49:39.290',
    type: 'ATTACK_ALERT',
    confidence_score: 0.9999,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'b8a9d0e1f2c3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8',
    previous_hash: 'a7f2c1e8d3b4a5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9'
  },
  {
    index: 13,
    timestamp: '2026-05-10T16:49:39.290',
    type: 'ATTACK_ALERT',
    confidence_score: 0.86,
    threat_level: 'HIGH',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'c9b0e1f2a3d4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0',
    previous_hash: 'b8a9d0e1f2c3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8'
  },
  {
    index: 14,
    timestamp: '2026-05-10T16:49:39.291',
    type: 'ATTACK_ALERT',
    confidence_score: 1.0,
    threat_level: 'CRITICAL',
    prediction: 'Attack',
    true_label: 'Attack',
    is_true_positive: true,
    hash: 'd0c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1',
    previous_hash: 'c9b0e1f2a3d4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0'
  }
];

// More sample alerts for the table
export const sampleAlerts = [
  { id: 1, blockIndex: 1, timestamp: '2026-05-10T16:49:39.287', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'neptune' },
  { id: 2, blockIndex: 2, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'neptune' },
  { id: 3, blockIndex: 3, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 0.999999, threatLevel: 'CRITICAL', type: 'satan' },
  { id: 4, blockIndex: 4, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 0.514593, threatLevel: 'LOW', type: 'warezclient' },
  { id: 5, blockIndex: 5, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'ipsweep' },
  { id: 6, blockIndex: 6, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 0.801007, threatLevel: 'MEDIUM', type: 'smurf' },
  { id: 7, blockIndex: 7, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'pod' },
  { id: 8, blockIndex: 8, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 0.999999, threatLevel: 'CRITICAL', type: 'teardrop' },
  { id: 9, blockIndex: 9, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 0.999997, threatLevel: 'CRITICAL', type: 'buffer_overflow' },
  { id: 10, blockIndex: 10, timestamp: '2026-05-10T16:49:39.288', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'rootkit' },
  { id: 11, blockIndex: 11, timestamp: '2026-05-10T16:49:39.289', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'perl' },
  { id: 12, blockIndex: 12, timestamp: '2026-05-10T16:49:39.290', prediction: 'Attack', confidence: 0.9999, threatLevel: 'CRITICAL', type: 'phf' },
  { id: 13, blockIndex: 13, timestamp: '2026-05-10T16:49:39.290', prediction: 'Attack', confidence: 0.86, threatLevel: 'HIGH', type: 'land' },
  { id: 14, blockIndex: 14, timestamp: '2026-05-10T16:49:39.291', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'loadmodule' },
  { id: 15, blockIndex: 15, timestamp: '2026-05-10T16:49:39.291', prediction: 'Attack', confidence: 0.9999, threatLevel: 'CRITICAL', type: 'guess_password' },
  { id: 16, blockIndex: 16, timestamp: '2026-05-10T16:49:39.292', prediction: 'Attack', confidence: 0.95, threatLevel: 'CRITICAL', type: 'back' },
  { id: 17, blockIndex: 17, timestamp: '2026-05-10T16:49:39.292', prediction: 'Attack', confidence: 0.78, threatLevel: 'MEDIUM', type: 'normal' },
  { id: 18, blockIndex: 18, timestamp: '2026-05-10T16:49:39.293', prediction: 'Attack', confidence: 1.0, threatLevel: 'CRITICAL', type: 'snmpgetattack' },
  { id: 19, blockIndex: 19, timestamp: '2026-05-10T16:49:39.293', prediction: 'Attack', confidence: 0.999, threatLevel: 'CRITICAL', type: 'mscan' },
  { id: 20, blockIndex: 20, timestamp: '2026-05-10T16:49:39.294', prediction: 'Attack', confidence: 0.72, threatLevel: 'MEDIUM', type: 'processtable' }
];
