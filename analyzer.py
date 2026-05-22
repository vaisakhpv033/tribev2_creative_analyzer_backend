"""
Brain Feature Analyzer
=======================
Extracts neural features from TRIBEv2 .npz prediction files.

Two categories of features are computed:

1. **Legacy Dimension Scores** (5 scores):
   - visual_mean, auditory_mean, emotional_mean, language_mean, attention_ratio
   - These are the original scores used for the radar-chart display.
   - Computed as mean activation of grouped brain regions per dimension.
   - Scaled to 0-100 in the worker using a simple linear transform.

2. **Model Features** (6 features):
   - longest_sustained_above_mean, emotional_mean, orbital_mean,
     visual_std, insula_short_mean, attention_onset_second
   - These are the features selected by the trained XGBoost CTR predictor.
   - They were chosen via Spearman correlation analysis against real CTR data
     and are the strongest neural predictors of ad performance.
   - Fed directly into the XGBoost model (no manual scaling).

See: model_training/results/best_model/selected_features.json
"""

import numpy as np
from nilearn import datasets


class BrainAnalyzer:
    """Extracts brain features from TRIBEv2 .npz predictions.

    Thread-safe: atlas is loaded lazily on first call and reused.
    """

    def __init__(self):
        # ── Region groups for LEGACY dimension scores ──────────────────────
        # These group brain regions into functional categories for the
        # radar-chart breakdown (visual, auditory, emotional, language).
        self.region_groups = {
            'visual': [
                'G_occipital_middle', 'G_occipital_sup', 'Pole_occipital',
                'G_and_S_occipital_inf', 'G_cuneus', 'S_calcarine',
                'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal',
                'G_oc-temp_lat-fusifor',
            ],
            'auditory': [
                'S_temporal_transverse', 'G_temp_sup-G_T_transv',
                'G_temp_sup-Lateral', 'G_temp_sup-Plan_tempo',
                'S_temporal_sup', 'G_temp_sup-Plan_polar',
            ],
            'emotional': [
                'G_orbital', 'S_orbital-H_Shaped', 'S_orbital_med-olfact',
                'S_orbital_lateral', 'G_rectus', 'S_circular_insula_inf',
                'S_circular_insula_sup', 'S_circular_insula_ant',
                'G_insular_short', 'G_Ins_lg_and_S_cent_ins',
                'G_and_S_cingul-Ant', 'G_subcallosal', 'G_front_inf-Orbital',
            ],
            'language': [
                'G_front_inf-Opercular', 'G_front_inf-Triangul',
                'G_temporal_middle', 'G_temporal_inf', 'S_front_inf',
                'G_and_S_cingul-Mid-Ant',
            ],
        }

        # ── Individual regions for MODEL features ──────────────────────────
        # These are specific brain regions whose individual activation is
        # used directly by the XGBoost CTR predictor.
        self.individual_regions = {
            'orbital':      ['G_orbital'],          # Reward center ("I want that")
            'insula_short': ['G_insular_short'],    # Gut feelings / emotional awareness
        }

        self._initialized = False

    def _ensure_initialized(self):
        """Lazily load the Destrieux atlas and precompute vertex indices."""
        if self._initialized:
            return

        print("Loading Destrieux surface atlas...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
                break
            except Exception as e:
                print(f"Error fetching atlas (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(2)

        self.roi_map = np.concatenate([
            self.destrieux_atlas['map_left'],
            self.destrieux_atlas['map_right'],
        ])
        self.labels = [
            lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl
            for lbl in self.destrieux_atlas['labels']
        ]

        # Precompute vertex indices for dimension groups
        self.region_indices = {
            group: self._get_indices_for_labels(labels)
            for group, labels in self.region_groups.items()
        }

        # Precompute vertex indices for individual model regions
        self.individual_indices = {
            name: self._get_indices_for_labels(labels)
            for name, labels in self.individual_regions.items()
        }

        self._initialized = True

    def _get_indices_for_labels(self, target_labels):
        """Map atlas label names to vertex indices on the fsaverage5 mesh."""
        indices = []
        for label in target_labels:
            try:
                roi_idx = self.labels.index(label)
                idx = np.where(self.roi_map == roi_idx)[0]
                indices.extend(idx)
            except ValueError:
                pass  # Label not found in atlas — skip silently
        return np.array(indices)

    # ──────────────────────────────────────────────────────────────────────
    # Main analysis entry point
    # ──────────────────────────────────────────────────────────────────────

    def analyze(self, npz_path: str) -> dict:
        """Extract all brain features from a TRIBEv2 .npz prediction file.

        Args:
            npz_path: Path to .npz file containing 'preds' array of
                      shape (n_seconds, 20484).

        Returns:
            dict with keys:
                - raw_scores: Legacy dimension means (for radar chart)
                - model_features: The 6 features used by the XGBoost CTR model
                - timeseries: Per-second time-series for each dimension
                - global_mean: Per-second whole-brain mean activation
        """
        self._ensure_initialized()
        print(f"Loading predictions from {npz_path}...")

        loaded_data = np.load(npz_path, allow_pickle=True)
        preds = loaded_data['preds']  # shape (n_timesteps, 20484)
        n_timesteps = preds.shape[0]

        # ── 1. LEGACY DIMENSION SCORES ────────────────────────────────────
        # Mean activation time-series for each functional dimension.
        # Used for the radar-chart breakdown display.

        timeseries = {}
        for group, indices in self.region_indices.items():
            if len(indices) > 0:
                timeseries[group] = np.mean(preds[:, indices], axis=1)
            else:
                timeseries[group] = np.zeros(n_timesteps)

        # Legacy attention score: ratio of first-3-seconds to whole-video
        global_ts = np.mean(preds, axis=1)  # (n_timesteps,)
        overall_mean = float(np.mean(global_ts))

        if n_timesteps >= 3:
            first_3s_mean = float(np.mean(global_ts[:3]))
            attention_ratio = first_3s_mean / overall_mean if overall_mean != 0 else 0.0
        else:
            first_3s_mean = float(np.mean(global_ts))
            attention_ratio = 0.0

        raw_scores = {
            'visual':    float(np.mean(timeseries['visual'])),
            'auditory':  float(np.mean(timeseries['auditory'])),
            'emotional': float(np.mean(timeseries['emotional'])),
            'language':  float(np.mean(timeseries['language'])),
            'attention': float(attention_ratio),
        }

        # ── 2. MODEL FEATURES (XGBoost CTR Predictor) ─────────────────────
        # These 6 features were selected by Spearman correlation analysis
        # against real Axon campaign CTR data. They are the inputs to the
        # trained XGBoost regressor + classifier.

        global_std = float(np.std(global_ts))

        # Feature 1: longest_sustained_above_mean
        # The longest consecutive streak of seconds where whole-brain
        # activation exceeds the video mean. Measures sustained engagement.
        # Spearman ρ = 0.587 with CTR.
        above_mean = global_ts > overall_mean
        longest_sustained = self._longest_consecutive_true(above_mean)

        # Feature 2: emotional_mean (already computed above)
        # Average activation of emotional/reward brain regions.
        # Spearman ρ = 0.584 with CTR.
        emotional_mean = raw_scores['emotional']

        # Feature 3: orbital_mean
        # Mean activation of the orbitofrontal cortex (G_orbital) —
        # the brain's reward valuation center ("I want that").
        # Spearman ρ = 0.588 with CTR.
        orbital_indices = self.individual_indices['orbital']
        if len(orbital_indices) > 0:
            orbital_ts = np.mean(preds[:, orbital_indices], axis=1)
            orbital_mean = float(np.mean(orbital_ts))
        else:
            orbital_mean = 0.0

        # Feature 4: visual_std
        # Standard deviation of visual cortex activation over time.
        # Measures visual "rhythm" — contrast between intense and calm moments.
        # Spearman ρ = -0.508 with CTR (NEGATIVE: lower variability = higher CTR).
        visual_std = float(np.std(timeseries['visual']))

        # Feature 5: insula_short_mean
        # Mean activation of the short insular gyrus — processes gut feelings
        # and visceral emotional responses.
        # Spearman ρ = 0.490 with CTR.
        insula_indices = self.individual_indices['insula_short']
        if len(insula_indices) > 0:
            insula_ts = np.mean(preds[:, insula_indices], axis=1)
            insula_short_mean = float(np.mean(insula_ts))
        else:
            insula_short_mean = 0.0

        # Feature 6: attention_onset_second
        # The first second where whole-brain activation exceeds
        # (mean + 0.5 × std). Measures how quickly engagement builds.
        # Spearman ρ = 0.520 with CTR.
        threshold = overall_mean + 0.5 * global_std
        onset_second = n_timesteps  # default: never reached
        for i, val in enumerate(global_ts):
            if val > threshold:
                onset_second = i
                break
        attention_onset_second = float(onset_second)

        model_features = {
            'longest_sustained_above_mean': float(longest_sustained),
            'emotional_mean':              emotional_mean,
            'orbital_mean':                orbital_mean,
            'visual_std':                  visual_std,
            'insula_short_mean':           insula_short_mean,
            'attention_onset_second':      attention_onset_second,
        }

        return {
            'raw_scores':     raw_scores,
            'model_features': model_features,
            'timeseries':     {k: v.tolist() for k, v in timeseries.items()},
            'global_mean':    global_ts.tolist(),
        }

    @staticmethod
    def _longest_consecutive_true(arr: np.ndarray) -> int:
        """Return the length of the longest consecutive True streak."""
        max_run = 0
        current_run = 0
        for val in arr:
            if val:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        return max_run


# Singleton instance (loaded once, reused across requests)
analyzer = BrainAnalyzer()
