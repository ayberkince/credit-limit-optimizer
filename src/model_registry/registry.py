"""
Model Registry with versioning and loading.
Saves/loads models with timestamps for reproducibility.
Supports: save, load latest, load specific version, list versions, delete version.
"""

import os
import json
import joblib
import shutil
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
from sklearn.linear_model import LogisticRegression
    

class ModelRegistry:
    def __init__(self, base_dir: str = "models"):
        """
        Initialize the model registry.

        Parameters:
        base_dir: Directory where models and metadata are stored.
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._metadata_file = os.path.join(base_dir, "metadata.json")
        self._load_metadata()

    def _load_metadata(self):
        """Load or initialise metadata registry."""
        if os.path.exists(self._metadata_file):
            with open(self._metadata_file, 'r') as f:
                self.metadata = json.load(f)
            # Ensure 'latest' key exists
            if 'latest' not in self.metadata:
                self.metadata['latest'] = {}
            if 'versions' not in self.metadata:
                self.metadata['versions'] = {}
        else:
            self.metadata = {'versions': {}, 'latest': {}}

    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_version(self) -> str:
        """Generate a version string from timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to Python serializable types."""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        else:
            return obj

    def save_model(self, model, model_name: str, **kwargs) -> str:
        """
        Save a model with a versioned timestamp.

        Parameters:
        model: The model object (must be pickleable with joblib).
        model_name: Name of the model (e.g., 'propensity_model', 'survival_model').
        **kwargs: Additional metadata to store (e.g., accuracy, n_features).

        Returns:
        version: The version string assigned.
        """
        version = self._generate_version()
        version_dir = os.path.join(self.base_dir, f"{model_name}_{version}")
        os.makedirs(version_dir, exist_ok=True)

        # Save the model
        model_path = os.path.join(version_dir, "model.joblib")
        joblib.dump(model, model_path)

        # Build metadata and convert numpy types
        meta = {
            'model_name': model_name,
            'version': version,
            'saved_at': datetime.now().isoformat(),
            'path': model_path,
            **kwargs
        }
        # Convert any numpy types to Python native types for JSON
        meta_serializable = self._convert_to_serializable(meta)

        # Save metadata
        with open(os.path.join(version_dir, "metadata.json"), 'w') as f:
            json.dump(meta_serializable, f, indent=2)

        # Update registry (store the serialized version to be consistent)
        self.metadata['versions'][version] = meta_serializable
        self.metadata['latest'][model_name] = version
        self._save_metadata()

        print(f"✅ Model '{model_name}' saved as version {version}")
        return version

    def load_latest(self, model_name: str) -> Optional[object]:
        """
        Load the latest version of a model.

        Parameters:
        model_name: Name of the model to load.

        Returns:
        The loaded model object, or None if not found.
        """
        if model_name not in self.metadata.get('latest', {}):
            print(f"❌ No model found for '{model_name}'")
            return None

        version = self.metadata['latest'][model_name]
        return self.load_version(model_name, version)

    def load_version(self, model_name: str, version: str) -> Optional[object]:
        """
        Load a specific version of a model.

        Parameters:
        model_name: Name of the model.
        version: Version string to load.

        Returns:
        The loaded model object, or None if not found.
        """
        version_dir = os.path.join(self.base_dir, f"{model_name}_{version}")
        model_path = os.path.join(version_dir, "model.joblib")

        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None

        model = joblib.load(model_path)
        print(f"📂 Loaded model '{model_name}' version {version}")
        return model

    def list_versions(self, model_name: str) -> List[str]:
        """
        List all saved versions of a model.

        Parameters:
        model_name: Name of the model.

        Returns:
        List of version strings sorted chronologically.
        """
        versions = []
        for version, meta in self.metadata['versions'].items():
            if meta.get('model_name') == model_name:
                versions.append(version)
        return sorted(versions)

    def list_models(self) -> List[str]:
        """
        List all model names that have at least one saved version.

        Returns:
        List of model names.
        """
        models = set()
        for meta in self.metadata['versions'].values():
            models.add(meta.get('model_name'))
        return sorted([m for m in models if m is not None])

    def get_metadata(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific version or the latest version.

        Parameters:
        model_name: Name of the model.
        version: Optional version string. If None, uses the latest.

        Returns:
        Metadata dictionary, or None if not found.
        """
        if version is None:
            if model_name not in self.metadata.get('latest', {}):
                print(f"❌ No latest version found for '{model_name}'")
                return None
            version = self.metadata['latest'][model_name]

        if version not in self.metadata['versions']:
            print(f"❌ Version '{version}' not found")
            return None

        return self.metadata['versions'][version]

    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific version (removes both model files and metadata).

        Parameters:
        model_name: Name of the model.
        version: Version to delete.

        Returns:
        True if successful, False otherwise.
        """
        version_dir = os.path.join(self.base_dir, f"{model_name}_{version}")
        if not os.path.exists(version_dir):
            print(f"❌ Version directory not found: {version_dir}")
            return False

        # Remove the directory
        shutil.rmtree(version_dir)

        # Remove from metadata
        if version in self.metadata['versions']:
            del self.metadata['versions'][version]

        # If this was the latest, update latest to the next newest version
        if self.metadata.get('latest', {}).get(model_name) == version:
            remaining = self.list_versions(model_name)
            if remaining:
                self.metadata['latest'][model_name] = remaining[-1]  # latest = newest
            else:
                del self.metadata['latest'][model_name]

        self._save_metadata()
        print(f"🗑️ Deleted model '{model_name}' version {version}")
        return True

    def clear_all(self):
        """Delete all models and metadata (use with caution)."""
        shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.metadata = {'versions': {}, 'latest': {}}
        self._save_metadata()
        print("🗑️ All models cleared.")


# Example usage (if run as standalone)
if __name__ == "__main__":

    # Create a dummy model
    model = LogisticRegression()
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)

    # Initialise registry
    registry = ModelRegistry()

    # Save model with metadata (including numpy types)
    registry.save_model(model, "propensity_model", accuracy=0.75, n_features=5, n_samples=np.int64(100))

    # Save another version
    model2 = LogisticRegression(C=0.1)
    model2.fit(X, y)
    registry.save_model(model2, "propensity_model", accuracy=0.78, n_features=5, n_samples=np.int64(100))

    # List versions
    print("Versions:", registry.list_versions("propensity_model"))

    # Load latest
    loaded = registry.load_latest("propensity_model")
    print(f"Loaded model: {loaded}")

    # Get metadata
    meta = registry.get_metadata("propensity_model")
    print("Metadata:", meta)