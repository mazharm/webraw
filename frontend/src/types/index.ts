export type HslChannel = 'red' | 'orange' | 'yellow' | 'green' | 'aqua' | 'blue' | 'purple' | 'magenta';

export interface CurvePoint {
  x: number;
  y: number;
}

export interface ParametricCurve {
  highlights: number;
  lights: number;
  darks: number;
  shadows: number;
}

export interface ToneCurve {
  mode: 'POINT' | 'PARAMETRIC';
  points: CurvePoint[];
  parametric?: ParametricCurve;
}

export interface HslValues {
  h: number;
  s: number;
  l: number;
}

export interface ColorGrading {
  shadows: HslValues;
  mids: HslValues;
  highlights: HslValues;
  balance: number;
}

export interface Sharpening {
  amount: number;
  radius: number;
  detail: number;
  masking: number;
}

export interface Denoise {
  luma: number;
  chroma: number;
  enhanced?: boolean;
}

export interface Optics {
  enable: boolean;
  distortion: number;
  vignette: number;
  ca: number;
}

export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Crop {
  angle: number;
  rect: CropRect;
}

export interface Transform {
  vertical: number;
  horizontal: number;
  rotate: number;
  aspect: number;
}

export interface Effects {
  grainAmount: number;
  grainSize: number;
  vignetteAmount: number;
}

export interface GlobalAdjustments {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  temperature: number;
  tint: number;
  vibrance: number;
  saturation: number;
  texture: number;
  clarity: number;
  dehaze: number;
  hdr: number;
  toneCurve: ToneCurve;
  hsl: Record<HslChannel, HslValues>;
  grading?: ColorGrading;
  sharpening: Sharpening;
  denoise: Denoise;
  optics: Optics;
  crop?: Crop;
  transform?: Transform;
  effects?: Effects;
}

export interface FilmSimState {
  id: string;
  strength: number;
  grainAmount: number;
  grainSize: number;
  bwFilter?: 'R' | 'Y' | 'G';
}

export interface BrushStroke {
  points: Array<{ x: number; y: number; pressure: number }>;
  size: number;
  feather: number;
  flow: number;
  erase: boolean;
}

export interface BrushMask {
  type: 'BRUSH';
  strokes: BrushStroke[];
}

export interface LinearGradientMask {
  type: 'LINEAR';
  start: { x: number; y: number };
  end: { x: number; y: number };
  feather: number;
}

export interface RadialGradientMask {
  type: 'RADIAL';
  center: { x: number; y: number };
  radiusX: number;
  radiusY: number;
  rotation: number;
  feather: number;
  invert: boolean;
}

export interface HealCloneMask {
  type: 'HEAL' | 'CLONE';
  sourcePoint: { x: number; y: number };
  targetPoint: { x: number; y: number };
  radius: number;
  feather: number;
}

export type MaskDefinition = BrushMask | LinearGradientMask | RadialGradientMask | HealCloneMask;

export interface LocalAdjustmentParams {
  exposure?: number;
  contrast?: number;
  highlights?: number;
  shadows?: number;
  whites?: number;
  blacks?: number;
  temperature?: number;
  tint?: number;
  vibrance?: number;
  saturation?: number;
  texture?: number;
  clarity?: number;
  dehaze?: number;
  sharpeningAmount?: number;
  denoiseLuma?: number;
}

export interface LocalAdjustment {
  id: string;
  type: 'BRUSH' | 'LINEAR' | 'RADIAL' | 'HEAL' | 'CLONE';
  mask: MaskDefinition;
  params: LocalAdjustmentParams;
  enabled: boolean;
}

export interface AiLayerMeta {
  provider: 'gemini';
  model: string;
  prompt?: string;
  negativePrompt?: string;
  promptHash?: string;
  negativePromptHash?: string;
  createdAt: string;
  options?: Record<string, unknown>;
}

export interface AiLayer {
  id: string;
  assetId: string;
  opacity: number;
  blendMode: 'NORMAL' | 'MULTIPLY' | 'SCREEN' | 'OVERLAY';
  maskId?: string;
  meta: AiLayerMeta;
  enabled: boolean;
}

export interface HistoryNode {
  id: string;
  parentId?: string;
  label: string;
  snapshot: Omit<EditState, 'history' | 'snapshots'>;
  ts: string;
}

export interface History {
  headId: string;
  nodes: Record<string, HistoryNode>;
}

export interface Snapshot {
  id: string;
  name: string;
  stateHash: string;
  createdAt: string;
}

export interface EditState {
  schemaVersion: number;
  assetId: string;
  global: GlobalAdjustments;
  filmSim?: FilmSimState;
  localAdjustments: LocalAdjustment[];
  aiLayers: AiLayer[];
  history: History;
  snapshots: Snapshot[];
}

export interface Asset {
  id: string;
  filename: string;
  mime: string;
  sourceType: 'RAW' | 'RASTER';
  originalUri: string;
  createdAt: string;
  exif?: Record<string, unknown>;
  hash?: string;
  fileId?: string; // backend ephemeral cache ID
  thumbnailUrl?: string;
  flag?: 'pick' | 'reject' | null;
  rating?: number;
  colorLabel?: string;
}

export interface HistogramData {
  r: number[];
  g: number[];
  b: number[];
  lum: number[];
}

export type JobStatus = 'PENDING' | 'PROCESSING' | 'COMPLETE' | 'FAILED';
export type JobKind = 'PREVIEW' | 'EXPORT' | 'AI_EDIT' | 'AUTO_ENHANCE';

export interface JobInfo {
  jobId: string;
  kind: JobKind;
  status: JobStatus;
  progress?: number;
  result?: unknown;
  error?: ProblemDetail;
  createdAt: string;
}

export interface ProblemDetail {
  type: string;
  title: string;
  status: number;
  detail: string;
  code: string;
  requestId: string;
  retryAfter?: number;
}

export type EnhanceModelKind = 'PARAMETER_PREDICTOR' | 'IMAGE_TO_IMAGE';

export interface EnhanceModelDescriptor {
  id: string;
  name: string;
  description: string;
  kind: EnhanceModelKind;
  version: string;
  builtin: boolean;
  requiresApiKey: boolean;
}

export interface EnhanceResultParameters {
  type: 'Parameters';
  values: Record<string, number>;
}

export interface EnhanceResultImage {
  type: 'Image';
  resultFileId: string;
  mimeType: string;
}

export type EnhanceResult = EnhanceResultParameters | EnhanceResultImage;

export type ViewMode = 'library' | 'develop';

export interface FilmSimLook {
  id: string;
  name: string;
  baseCurve: string;
  lutFile: string;
  matrix: number[];
  defaults: {
    strength: number;
    grainAmount: number;
    grainSize: number;
    saturation: number;
  };
}
