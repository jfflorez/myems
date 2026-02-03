"""
Experiment Management Module
=============================

Provides base functionality for running reproducible experiments with automatic:
- Folder structure creation
- Metadata tracking (timestamps, configs, execution info)
- Parameter grid generation from baseline YAML
- Result organization

Design Pattern: Context Manager (RAII - Resource Acquisition Is Initialization)
Inspired by: MLflow, Weights & Biases, DVC

Usage:
------
    from utils.experiments import Experiment
    
    with Experiment(
        name="my_experiment",
        params_to_study={"learning_rate": [0.001, 0.01], "batch_size": [32, 64]}
    ) as exp:
        for params in exp.parameter_grid:
            results = my_pipeline(dataset, params)
            exp.save_output(results, name=f"run_{exp.current_run}")
"""

import json
import yaml
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Iterator
import shutil
import inspect


class ExperimentError(Exception):
    """Base exception for experiment-related errors"""
    pass


class Experiment:
    """
    Context manager for running experiments with automatic folder structure and metadata tracking.
    
    Attributes:
        name (str): Name of the experiment (used for folder creation)
        base_dir (Path): Base directory for all experiments
        experiment_dir (Path): Directory for this specific experiment
        params_baseline (Dict): Baseline parameter configuration
        params_to_study (Dict): Parameters to vary in grid search
        metadata (Dict): Experiment metadata (auto-populated)
    """
    
    def __init__(
        self,
        name: str,
        base_dir: Union[str, Path] = "experiments",
        params_baseline: Optional[Union[str, Path, Dict]] = None,
        params_to_study: Optional[Dict[str, List[Any]]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize experiment configuration.
        
        Args:
            name: Experiment name (alphanumeric, underscores, hyphens)
            base_dir: Base directory for all experiments
            params_baseline: Path to baseline YAML or dict with baseline params
            params_to_study: Dict mapping param names to lists of values to test
                           e.g., {"lr": [0.001, 0.01], "batch_size": [32, 64]}
            description: Human-readable experiment description
            tags: List of tags for categorizing experiments
        """
        self.name = self._validate_name(name)
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / self.name
        self.description = description
        self.tags = tags or []
        
        # Directories
        self.params_dir = self.experiment_dir / "parameters"
        self.outputs_dir = self.experiment_dir / "outputs"
        self.figures_dir = self.experiment_dir / "figures"
        
        # Load or set baseline parameters
        if params_baseline is None:
            self.params_baseline = {}
        elif isinstance(params_baseline, (str, Path)):
            self.params_baseline = self._load_yaml(Path(params_baseline))
        else:
            self.params_baseline = params_baseline
            
        # Parameters to study
        self.params_to_study = params_to_study or {}
        
        # Metadata (populated during execution)
        self.metadata = {
            "experiment_name": self.name,
            "description": self.description,
            "tags": self.tags,
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "script_name": None,
            "function_name": None,
            "config_files_used": [],
            "num_runs": 0,
            "status": "initialized",
            "error": None,
        }
        
        # Runtime state
        self._parameter_grid: List[Dict] = []
        self.current_run = 0
        self._start_timestamp: Optional[datetime] = None
        
    def _validate_name(self, name: str) -> str:
        """Validate experiment name (alphanumeric, underscores, hyphens only)"""
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ExperimentError(
                f"Invalid experiment name '{name}'. "
                "Use only alphanumeric characters, underscores, and hyphens."
            )
        return name
    
    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML file"""
        if not path.exists():
            raise ExperimentError(f"YAML file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_yaml(self, data: Dict, path: Path) -> None:
        """Save dict to YAML file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def _save_json(self, data: Dict, path: Path) -> None:
        """Save dict to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _create_folder_structure(self) -> None:
        """Create experiment folder structure"""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        
    def _generate_parameter_grid(self) -> List[Dict]:
        """
        Generate parameter grid from baseline and params_to_study.
        
        Creates Cartesian product of all parameter values.
        
        Returns:
            List of parameter dicts, each representing one configuration
        """
        if not self.params_to_study:
            # No grid search, just use baseline
            return [self.params_baseline.copy()]
        
        # Get keys and value lists
        keys = list(self.params_to_study.keys())
        value_lists = [self.params_to_study[k] for k in keys]
        
        # Generate all combinations
        grid = []
        for values in itertools.product(*value_lists):
            # Start with baseline
            params = self.params_baseline.copy()
            # Override with current grid values
            for key, value in zip(keys, values):
                # Support nested keys via dot notation (e.g., "model.lr")
                self._set_nested_param(params, key, value)
            grid.append(params)
        
        return grid
    
    def _set_nested_param(self, params: Dict, key: str, value: Any) -> None:
        """
        Set parameter value, supporting nested keys with dot notation.
        
        Example: key="model.learning_rate" sets params["model"]["learning_rate"]
        """
        keys = key.split(".")
        current = params
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _save_parameter_variants(self) -> None:
        """Save baseline and all parameter variants as YAML files"""
        # Save baseline
        baseline_path = self.params_dir / "params_baseline.yaml"
        self._save_yaml(self.params_baseline, baseline_path)
        self.metadata["config_files_used"].append(str(baseline_path.relative_to(self.base_dir)))
        
        # Save each variant
        for i, params in enumerate(self._parameter_grid):
            variant_path = self.params_dir / f"params_variant_{i:03d}.yaml"
            self._save_yaml(params, variant_path)
            self.metadata["config_files_used"].append(str(variant_path.relative_to(self.base_dir)))
    
    def _get_caller_info(self) -> tuple:
        """Get information about the calling script/function"""
        frame = inspect.currentframe()
        try:
            # Walk up the stack to find the first frame outside this module
            caller_frame = frame.f_back.f_back.f_back  # Skip __enter__ and __init__
            if caller_frame:
                filename = caller_frame.f_code.co_filename
                function_name = caller_frame.f_code.co_name
                return Path(filename).name, function_name
        finally:
            del frame
        return None, None
    
    def __enter__(self):
        """
        Context manager entry: setup experiment infrastructure.
        
        - Creates folder structure
        - Generates parameter grid
        - Saves parameter variants
        - Records start time and caller info
        """
        self._start_timestamp = datetime.now()
        self.metadata["start_time"] = self._start_timestamp.isoformat()
        
        # Get caller information
        script_name, function_name = self._get_caller_info()
        self.metadata["script_name"] = script_name
        self.metadata["function_name"] = function_name
        
        # Setup infrastructure
        self._create_folder_structure()
        self._parameter_grid = self._generate_parameter_grid()
        self.metadata["num_runs"] = len(self._parameter_grid)
        self._save_parameter_variants()
        
        self.metadata["status"] = "running"
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit: finalize metadata.
        
        - Records end time and duration
        - Saves metadata.json
        - Handles exceptions gracefully
        """
        end_timestamp = datetime.now()
        self.metadata["end_time"] = end_timestamp.isoformat()
        
        if self._start_timestamp:
            duration = (end_timestamp - self._start_timestamp).total_seconds()
            self.metadata["duration_seconds"] = duration
        
        # Handle exceptions
        if exc_type is not None:
            self.metadata["status"] = "failed"
            self.metadata["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
                "traceback": str(exc_tb),
            }
        else:
            self.metadata["status"] = "completed"
        
        # Save metadata
        metadata_path = self.experiment_dir / "metadata.json"
        self._save_json(self.metadata, metadata_path)
        
        # Don't suppress exceptions
        return False
    
    @property
    def parameter_grid(self) -> Iterator[Dict]:
        """
        Iterator over parameter configurations.
        
        Yields:
            Dict: Parameter configuration for each run
        """
        for i, params in enumerate(self._parameter_grid):
            self.current_run = i
            yield params
    
    def save_output(
        self,
        data: Any,
        name: Optional[str] = None,
        format: str = "json",
        subdir: Optional[str] = None,
    ) -> Path:
        """
        Save experiment output data.
        
        Args:
            data: Data to save (dict, list, or other)
            name: Output filename (without extension)
            format: Output format ("json", "yaml", "pkl", "npy")
            subdir: Optional subdirectory within outputs/
            
        Returns:
            Path to saved file
        """
        if name is None:
            name = f"output_run_{self.current_run:03d}"
        
        # Determine save directory
        save_dir = self.outputs_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        if format == "json":
            filepath = save_dir / f"{name}.json"
            self._save_json(data, filepath)
        elif format == "yaml":
            filepath = save_dir / f"{name}.yaml"
            self._save_yaml(data, filepath)
        elif format == "pkl":
            import pickle
            filepath = save_dir / f"{name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == "npy":
            import numpy as np
            filepath = save_dir / f"{name}.npy"
            np.save(filepath, data)
        else:
            raise ExperimentError(f"Unsupported format: {format}")
        
        return filepath
    
    def save_figure(
        self,
        fig,
        name: Optional[str] = None,
        format: str = "png",
        dpi: int = 300,
        **kwargs
    ) -> Path:
        """
        Save matplotlib/plotly figure.
        
        Args:
            fig: Figure object (matplotlib or plotly)
            name: Figure filename (without extension)
            format: Image format ("png", "pdf", "svg", "html" for plotly)
            dpi: Resolution for raster formats
            **kwargs: Additional arguments passed to savefig
            
        Returns:
            Path to saved figure
        """
        if name is None:
            name = f"figure_run_{self.current_run:03d}"
        
        filepath = self.figures_dir / f"{name}.{format}"
        
        # Detect figure type and save appropriately
        if hasattr(fig, 'savefig'):  # matplotlib
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', **kwargs)
        elif hasattr(fig, 'write_html'):  # plotly
            if format == 'html':
                fig.write_html(filepath, **kwargs)
            else:
                fig.write_image(filepath, **kwargs)
        else:
            raise ExperimentError(f"Unsupported figure type: {type(fig)}")
        
        return filepath
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value (useful for tracking training progress).
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        # Initialize metrics storage if needed
        if "metrics" not in self.metadata:
            self.metadata["metrics"] = {}
        
        if name not in self.metadata["metrics"]:
            self.metadata["metrics"][name] = []
        
        metric_entry = {"value": value, "run": self.current_run}
        if step is not None:
            metric_entry["step"] = step
        
        self.metadata["metrics"][name].append(metric_entry)
    
    def get_params(self, run_index: int) -> Dict:
        """
        Get parameters for a specific run.
        
        Args:
            run_index: Index of the run in the parameter grid
            
        Returns:
            Parameter dict for that run
        """
        if run_index >= len(self._parameter_grid):
            raise ExperimentError(f"Run index {run_index} out of range")
        return self._parameter_grid[run_index]
    
    def summary(self) -> str:
        """
        Get experiment summary.
        
        Returns:
            Formatted string with experiment details
        """
        lines = [
            f"Experiment: {self.name}",
            f"Status: {self.metadata['status']}",
            f"Start: {self.metadata['start_time']}",
            f"Runs: {self.metadata['num_runs']}",
            f"Parameters studied: {list(self.params_to_study.keys())}",
            f"Output directory: {self.experiment_dir}",
        ]
        return "\n".join(lines)


# Convenience function for simple experiments
def run_experiment(
    name: str,
    pipeline_func,
    dataset,
    params_baseline: Optional[Dict] = None,
    params_to_study: Optional[Dict[str, List]] = None,
    **exp_kwargs
) -> Dict:
    """
    Run a simple experiment with automatic tracking.
    
    Args:
        name: Experiment name
        pipeline_func: Function that takes (dataset, params) and returns results
        dataset: Input dataset
        params_baseline: Baseline parameters
        params_to_study: Parameters to vary
        **exp_kwargs: Additional arguments for Experiment()
        
    Returns:
        Dict with all results
        
    Example:
        results = run_experiment(
            name="test_model",
            pipeline_func=train_model,
            dataset=my_data,
            params_to_study={"lr": [0.001, 0.01]}
        )
    """
    all_results = {}
    
    with Experiment(
        name=name,
        params_baseline=params_baseline,
        params_to_study=params_to_study,
        **exp_kwargs
    ) as exp:
        for params in exp.parameter_grid:
            results = pipeline_func(dataset, params)
            
            # Save outputs
            exp.save_output(results, name=f"run_{exp.current_run:03d}")
            
            # Store in memory too
            all_results[exp.current_run] = {
                "params": params,
                "results": results,
            }
    
    return all_results
