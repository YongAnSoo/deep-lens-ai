export type PredictResult = {
  task_id: string;
  filename: string;
  label: string;
  fake_probability: number;
  real_probability: number;
  risk_score?: number | null;
  severity?: string | null;
  confidence?: number | null;
  consistency?: number | null;
  frames_analyzed?: number;
  num_faces?: number;
  method?: string;
  suspicious_frames?: number[];
  gradcam_paths?: Record<string, Record<string, string>>;
  module_b?: Record<string, unknown>;
  module_b_details?: Record<string, any>;
  llm_analysis?: {
    provider?: string;
    model?: string;
    text?: string;
  };
  explanation?: {
    summary?: string;
    main_reasons?: string[];
  };
  raw_result?: Record<string, unknown>;
};
