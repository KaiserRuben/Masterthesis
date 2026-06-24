/**
 * Hand-written TypeScript interfaces mirroring the three HS-01 JSON Schemas.
 * These are the typed contract for the whole app — every later task imports from here.
 *
 * Schemas: experiments/HS-01/schemas/hs01.{itempool,study-config,session}.schema.json
 */

// ─── Item Pool ───────────────────────────────────────────────────────────────

export interface PromptAsset {
  text: string;
  sha256: string;
  char_count: number | null;
  contains_homoglyphs: boolean | null;
  original_text: string | null;
}

export interface ImageAsset {
  uri: string;
  sha256: string;
  width: number;
  height: number;
  format: "png";
}

export interface SourceAssets {
  prompt: PromptAsset | null;
  image: ImageAsset | null;
}

export interface ExperimentRef {
  experiment_id: string;
  run_id: string;
  seed_index: number | null;
  generation: number | null;
  individual_id: string | null;
}

export interface Sut {
  model_id: string;
  backend: "openvino" | "torch-mps" | "torch-cuda" | "other";
  scoring?: string;
}

export interface SourceCell {
  anchor_class: string;
  target_class: string;
  anchor_word: string;
  target_word: string;
  level_anchor: number | null;
  level_target: number | null;
  direction: "forward" | "reverse" | null;
  bucket_relation: "within" | "cross" | null;
}

export interface SourceSearch {
  modality: "joint" | "image_only" | "text_only" | null;
  tgtbal: number | null;
  crossed: boolean | null;
  gen_first_cross: number | null;
}

export interface SourceDrift {
  d_text: number | null;
  d_img: number | null;
  active_text_genes: number | null;
  hamming_to_anchor_norm: number | null;
}

export interface SourceStrata {
  text: "clean" | "low_drift" | "medium_drift" | "high_drift" | null;
  image: "raw" | "roundtrip" | "boundary_joint" | "image_heavy" | null;
  pair: "baseline" | "image_heavy" | "text_heavy" | "balanced" | null;
}

export interface Source {
  source_id: string;
  origin:
    | "raw_original"
    | "vqgan_roundtrip"
    | "boundary_individual"
    | "attention_synthetic";
  experiment_ref: ExperimentRef | null;
  sut: Sut | null;
  cell: SourceCell | null;
  search: SourceSearch | null;
  drift: SourceDrift | null;
  strata: SourceStrata;
  assets: SourceAssets;
  [key: string]: unknown; // for x_ extension fields
}

export interface CheckRule {
  metric: "scale_leq" | "scale_geq" | "choice_equals";
  value: number | string;
}

export interface Item {
  item_id: string;
  kind: "text" | "image" | "pair";
  source_id: string;
  is_attention_check?: boolean;
  check_rule?: CheckRule | null;
  [key: string]: unknown; // for x_ extension fields
}

export interface ItemPool {
  schema_version: "1.0.0";
  pool_id: string;
  created: string;
  frozen: boolean;
  generator?: {
    pipeline_commit: string | null;
    selection_script: string | null;
    notes: string | null;
  };
  composition_targets?: Record<string, Record<string, number>>;
  sources: Source[];
  items: Item[];
  [key: string]: unknown;
}

// ─── Study Config ─────────────────────────────────────────────────────────────

export interface Phase {
  phase_id: "consent" | "text" | "image" | "pair" | "demographics";
  target_duration_s: number;
  trials_per_rater: number | null;
}

export interface Scale {
  scale_id: string;
  applies_to: "text" | "image";
  statement: string;
  points: 5;
  point_labels: [string, string, string, string, string];
}

export interface PairResponse {
  semantic_options: [
    "ANCHOR_WORD",
    "TARGET_WORD",
    "OTHER_CLASS",
    "NOTHING_RECOGNIZABLE",
    "CANT_TELL",
  ];
  display_labels: {
    OTHER_CLASS: string;
    NOTHING_RECOGNIZABLE: string;
    CANT_TELL: string;
  };
  ab_order: "randomized_per_trial";
  fixed_tail?: boolean;
  other_class_free_text: boolean;
}

export interface Form {
  form_id: string;
  text_items: string[];
  image_items: string[];
  pair_items: string[];
}

export interface DemographicsField {
  field_id: string;
  label: string;
  type: "select" | "free_text";
  options: string[] | null;
  required: boolean;
}

export interface StudyConfig {
  schema_version: "1.0.0";
  study_id: string;
  config_version: string;
  preregistration?: {
    hypotheses_frozen: boolean;
    doc_ref: string | null;
  };
  pool_ref: {
    pool_id: string;
    pool_file_sha256: string;
  };
  locale: "en";
  phases: Phase[];
  scales: Scale[];
  pair_response: PairResponse;
  forms: Form[];
  randomization: {
    within_phase_shuffle: true;
    session_seed_source: "server_generated" | "client_generated";
    log_seed: true;
  };
  attention_policy: {
    item_ids: string[];
    placement: "interleaved";
    exclusion_fail_threshold: number;
  };
  demographics_fields: DemographicsField[];
  consent: {
    consent_version: string;
    text_sha256: string | null;
    required: true;
  };
  quality?: {
    log_integrity_events: boolean;
    render_check: boolean;
    min_rendered_image_css_px: number | null;
  };
  [key: string]: unknown;
}

// ─── Session Record ───────────────────────────────────────────────────────────

export type SemanticChoice =
  | "ANCHOR_WORD"
  | "TARGET_WORD"
  | "OTHER_CLASS"
  | "NOTHING_RECOGNIZABLE"
  | "CANT_TELL";

export interface TrialPresented {
  scale_id?: string | null;
  option_display_order?: SemanticChoice[] | null;
  option_labels?: {
    ANCHOR_WORD: string;
    TARGET_WORD: string;
  } | null;
  rendered_image?: {
    css_w: number;
    css_h: number;
    natural_w: number;
    natural_h: number;
  } | null;
}

export interface TrialResponse {
  scale_value?: number;
  choice?: SemanticChoice;
  other_class_text?: string | null;
  n_changes: number;
}

export interface TrialTiming {
  image_loaded_ms?: number | null;
  onset_ms: number;
  first_interaction_ms?: number | null;
  response_selected_ms?: number | null;
  submitted_ms: number;
}

export interface Trial {
  trial_index: number;
  phase_id: "text" | "image" | "pair";
  position_in_phase: number;
  item_id: string;
  source_id: string;
  item_kind: "text" | "image" | "pair";
  is_attention_check: boolean;
  presented: TrialPresented;
  response: TrialResponse;
  timing: TrialTiming;
  [key: string]: unknown; // for x_ extension fields
}

export interface PhaseTiming {
  phase_id:
    | "consent"
    | "instructions"
    | "text"
    | "image"
    | "pair"
    | "demographics";
  entered_ms: number;
  exited_ms: number;
}

export interface IntegrityEvent {
  at_ms: number;
  type:
    | "blur"
    | "focus"
    | "visibility_hidden"
    | "visibility_visible"
    | "resize"
    | "fullscreen_exit";
  detail: string | null;
}

export interface SessionRecord {
  schema_version: "1.0.0";
  study_id: string;
  config_version: string;
  config_sha256: string;
  app_version?: string | null;
  session_id: string; // uuid
  form_id: string;
  rng_seed: string | number;
  status: "completed" | "abandoned";
  participant: {
    participant_code: string;
    recruitment_channel: "tum" | "fortiss" | "personal" | "other" | null;
    consent: {
      given: true;
      consent_version: string;
      at_utc: string;
    };
  };
  environment: {
    user_agent: string;
    platform?: string | null;
    viewport: { w: number; h: number };
    screen?: { w: number; h: number } | null;
    device_pixel_ratio: number;
    is_touch?: boolean | null;
    render_check?: {
      passed: boolean;
      method: string | null;
    } | null;
  };
  timing: {
    started_at_utc: string;
    completed_at_utc?: string | null;
    server_received_at_utc?: string | null;
    total_duration_ms?: number | null;
  };
  phase_timings: PhaseTiming[];
  trials: Trial[];
  demographics?: {
    age_band:
      | "18_24"
      | "25_34"
      | "35_44"
      | "45_54"
      | "55_plus"
      | "prefer_not_to_say";
    ml_familiarity:
      | "no_experience"
      | "some_exposure"
      | "regular_practice"
      | "prefer_not_to_say";
    english_proficiency:
      | "A1"
      | "A2"
      | "B1"
      | "B2"
      | "C1"
      | "C2"
      | "native"
      | "prefer_not_to_say";
    comment?: string | null;
  } | null;
  integrity_events?: IntegrityEvent[];
  quality_summary?: {
    attention_total: number;
    attention_failed: number;
    focus_loss_count?: number | null;
  };
  [key: string]: unknown;
}
