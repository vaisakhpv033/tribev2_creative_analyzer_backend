import numpy as np
import pandas as pd
from nilearn import datasets

class BrainAnalyzer:
    def __init__(self):
        print("Loading Destrieux surface atlas...")
        self.destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
        self.roi_map = np.concatenate([self.destrieux_atlas['map_left'], self.destrieux_atlas['map_right']])
        self.labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in self.destrieux_atlas['labels']]
        
        # Define region groups based on the System Design specs
        self.region_groups = {
            'visual': [
                'G_occipital_middle', 'G_occipital_sup', 'Pole_occipital', 
                'G_and_S_occipital_inf', 'G_cuneus', 'S_calcarine', 
                'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'G_oc-temp_lat-fusifor'
            ],
            'auditory': [
                'S_temporal_transverse', 'G_temp_sup-G_T_transv', 'G_temp_sup-Lateral', 
                'G_temp_sup-Plan_tempo', 'S_temporal_sup', 'G_temp_sup-Plan_polar'
            ],
            'emotional': [
                'G_orbital', 'S_orbital-H_Shaped', 'S_orbital_med-olfact', 'S_orbital_lateral', 
                'G_rectus', 'S_circular_insula_inf', 'S_circular_insula_sup', 'S_circular_insula_ant', 
                'G_insular_short', 'G_Ins_lg_and_S_cent_ins', 'G_and_S_cingul-Ant', 'G_subcallosal', 
                'G_front_inf-Orbital'
            ],
            'language': [
                'G_front_inf-Opercular', 'G_front_inf-Triangul', 'G_temporal_middle', 
                'G_temporal_inf', 'S_front_inf', 'G_and_S_cingul-Mid-Ant'
            ]
        }

        self.region_indices = {
            group: self._get_indices_for_labels(labels)
            for group, labels in self.region_groups.items()
        }

    def _get_indices_for_labels(self, target_labels):
        indices = []
        for label in target_labels:
            try:
                roi_idx = self.labels.index(label)
                idx = np.where(self.roi_map == roi_idx)[0]
                indices.extend(idx)
            except ValueError:
                pass
        return np.array(indices)

    def analyze(self, npz_path: str):
        print(f"Loading predictions from {npz_path}...")
        loaded_data = np.load(npz_path, allow_pickle=True)
        preds = loaded_data['preds'] # shape (n_timesteps, 20484)
        n_timesteps = preds.shape[0]

        # Calculate mean timeseries for each functional group
        timeseries = {}
        for group, indices in self.region_indices.items():
            if len(indices) > 0:
                timeseries[group] = np.mean(preds[:, indices], axis=1)
            else:
                timeseries[group] = np.zeros(n_timesteps)
                
        # Calculate Attention Score (temporal dynamics of first 3 seconds)
        global_mean = np.mean(preds, axis=1) # Mean of all 20,484 regions per second
        overall_mean = np.mean(global_mean)  # Single scalar average of the whole video
        
        if n_timesteps >= 3:
            first_3_seconds_mean = np.mean(global_mean[:3])
            attention_raw = first_3_seconds_mean / overall_mean if overall_mean != 0 else 0
        else:
            attention_raw = 0

        raw_scores = {
            'visual': float(np.mean(timeseries['visual'])),
            'auditory': float(np.mean(timeseries['auditory'])),
            'emotional': float(np.mean(timeseries['emotional'])),
            'language': float(np.mean(timeseries['language'])),
            'attention': float(attention_raw)
        }

        return {
            'raw_scores': raw_scores,
            'timeseries': {k: v.tolist() for k, v in timeseries.items()},
            'global_mean': global_mean.tolist()
        }

# Singleton instance
analyzer = BrainAnalyzer()
