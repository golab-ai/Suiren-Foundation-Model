import torch
import os
import yaml
from torch_geometric.data import Data
from .atom_ref_table import atomref_list, atomref, reverse_atomref

class ModelLoader:
    def __init__(self, config_path):
        """
        Initialize ModelLoader with configuration.
        
        Args:
            config (dict): Configuration dictionary containing model and normalization parameters
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model = None
        self.normalizer = None  # dict with 'mean' and 'std' for energy and forces
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """
        Load and instantiate the model from configuration.
        
        Returns:
            model: Instantiated model
        """
        from .model.EST_eqv2 import EST_Eqv2
        model_config = self.config.get('model', {})
        self.model = EST_Eqv2(**model_config).to(device=self.device)
        return self.model

    def load_weights(self, weights_path):
        """
        Load pre-trained model weights.
        
        Args:
            weights_path (str): Path to the model weights file
            
        Returns:
            model: Model with loaded weights
        """
        if self.model is None:
            raise ValueError("Model must be loaded before loading weights.")
        checkpoint = torch.load(weights_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        return self.model

    def load_normalizer(self, normalizer_path=None):
        """
        Load normalization parameters for energy and forces.
        
        Normalization parameters can be loaded from:
        1. A .pt file (highest priority)
        2. Configuration dictionary (lower priority)
        
        Args:
            normalizer_path (str, optional): Path to normalizer checkpoint file.
                If provided, will attempt to load from this file first.
                
        Returns:
            dict: Dictionary with keys:
                - 'energy_mean': mean value for energy normalization
                - 'energy_std': std value for energy normalization
                - 'forces_mean': mean value for forces normalization (optional)
                - 'forces_std': std value for forces normalization (optional)
                
        Raises:
            FileNotFoundError: If normalizer_path is provided but file doesn't exist
            ValueError: If no normalizer data found in either file or config
        """
        normalizer = {}
        
        # Try to load from .pt file first (higher priority)
        if normalizer_path is not None:
            if not os.path.exists(normalizer_path):
                raise FileNotFoundError(f"Normalizer file not found: {normalizer_path}")
            
            checkpoint = torch.load(normalizer_path, map_location='cpu')
            
            # Extract normalizer from checkpoint
            if isinstance(checkpoint, dict):
                if 'energy_mean' in checkpoint:
                    normalizer['energy_mean'] = checkpoint['energy_mean']
                if 'energy_std' in checkpoint:
                    normalizer['energy_std'] = checkpoint['energy_std']
                if 'forces_mean' in checkpoint:
                    normalizer['forces_mean'] = checkpoint['forces_mean']
                if 'forces_std' in checkpoint:
                    normalizer['forces_std'] = checkpoint['forces_std']
                if 'atom_ref' in checkpoint:
                    normalizer['atomref_list'] = checkpoint['atomref_list']
        
            if normalizer:
                self.normalizer = normalizer
                return normalizer
        
        # Try to load from config (lower priority)
        if 'normalizer' in self.config:
            norm_config = self.config['normalizer']
            if isinstance(norm_config, dict):
                if 'energy' in norm_config:
                    normalizer['energy_mean'] = norm_config['energy'].get('mean')
                    normalizer['energy_std'] = norm_config['energy'].get('std')
                if 'forces' in norm_config:
                    normalizer['forces_mean'] = norm_config['forces'].get('mean')
                    normalizer['forces_std'] = norm_config['forces'].get('std')
                normalizer['atomref_list'] = atomref_list
        
        # Validate that we found normalization parameters
        if not normalizer or 'energy_mean' not in normalizer or 'energy_std' not in normalizer:
            raise ValueError(
                "No normalization parameters found. Provide either:\n"
                "1. A normalizer checkpoint file (.pt) with 'norm_factor' or 'energy_mean'/'energy_std'\n"
                "2. Config dictionary with 'normalization' section containing energy mean/std"
            )
        
        self.normalizer = normalizer
        return normalizer

    def normalize(self, values, target='energy', batch=None, elements=None, atom_ref=False):
        """
        Normalize raw values using loaded normalization parameters.
        
        This is the inverse of denormalize and is useful for preprocessing
        raw data before feeding to the model or for normalizing target labels.
        
        Args:
            values (torch.Tensor): Raw values to normalize
            target (str): Type of value - 'energy' or 'forces'
            
        Returns:
            torch.Tensor: Normalized values
            
        Raises:
            ValueError: If normalizer not loaded or invalid target type
        """
        if self.normalizer is None:
            raise ValueError(
                "Normalizer not loaded. Call load_normalizer() first."
            )
        
        if target == 'energy':                
            mean = self.normalizer.get('energy_mean')
            std = self.normalizer.get('energy_std')
            if mean is None or std is None:
                raise ValueError("Energy normalization parameters not found in normalizer")
        elif target == 'forces':
            mean = self.normalizer.get('forces_mean')
            std = self.normalizer.get('forces_std')
            if mean is None or std is None:
                # Fall back to energy parameters if forces not specified
                mean = self.normalizer.get('energy_mean')
                std = self.normalizer.get('energy_std')
                if mean is None or std is None:
                    raise ValueError("Forces normalization parameters not found in normalizer")
        else:
            raise ValueError(f"Unknown target type: {target}. Use 'energy' or 'forces'")
        
        # Convert to tensor if necessary
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=values.dtype, device=values.device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=values.dtype, device=values.device)
            
        # atom reference
        if atom_ref:
            if self.normalizer.get('atomref_list') is None:
                raise ValueError("Using atomic reference, but not found atomref_list in normalizer.")
            if elements is None or batch is None:
                raise ValueError("If using atomic reference, provide elements and batch first.")
            values = atomref(values, batch=batch, atomic_numbers=elements)
            
        # Normalize: normalized = (x - mean) / std
        return (values - mean) / std
    
    def denormalize(self, predictions, target='energy', batch=None, elements=None, atom_ref=False):
        """
        Denormalize model predictions using loaded normalization parameters.
        
        Args:
            predictions (torch.Tensor): Model predictions to denormalize
            target (str): Type of prediction - 'energy' or 'forces'
            
        Returns:
            torch.Tensor: Denormalized predictions
            
        Raises:
            ValueError: If normalizer not loaded or invalid target type
        """
        if self.normalizer is None:
            raise ValueError(
                "Normalizer not loaded. Call load_normalizer() first."
            )
        
        if target == 'energy':
            mean = self.normalizer.get('energy_mean')
            std = self.normalizer.get('energy_std')
            if mean is None or std is None:
                raise ValueError("Energy normalization parameters not found in normalizer")
        elif target == 'forces':
            mean = self.normalizer.get('forces_mean')
            std = self.normalizer.get('forces_std')
            if mean is None or std is None:
                # Fall back to energy parameters if forces not specified
                mean = self.normalizer.get('energy_mean')
                std = self.normalizer.get('energy_std')
                if mean is None or std is None:
                    raise ValueError("Forces normalization parameters not found in normalizer")
        else:
            raise ValueError(f"Unknown target type: {target}. Use 'energy' or 'forces'")
        
        # Convert to tensor if necessary
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=predictions.dtype, device=predictions.device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=predictions.dtype, device=predictions.device)
        
        # Denormalize: x = (normalized_x * std) + mean
        values = predictions * std + mean
        if atom_ref:
            if self.normalizer.get('atomref_list') is None:
                raise ValueError("Using atomic reference, but not found atomref_list in normalizer.")
            if elements is None or batch is None:
                raise ValueError("If using atomic reference, provide elements and batch first.")
            values = reverse_atomref(values, batch=batch, atomic_numbers=elements)
            
        return values

    def data_process(self, data_list):
        if len(data_list) == 0:
            raise ValueError("data list is empty!")
        device = data_list[0]['x'].device
        x = torch.tensor([], device=device, dtype=data_list[0]['x'].dtype)
        pos = torch.tensor([], device=device, dtype=data_list[0]['pos'].dtype)
        ptr = [0]
        batch = torch.tensor([], device=device, dtype=torch.long)
        
        idx = 0
        idx_ptr = 0
        for data_mol in data_list:
            natoms = data_mol['x'].view(-1).shape[0]
            x = torch.cat([x, data_mol['x'].view(-1)], dim=0)
            pos = torch.cat([pos, data_mol['pos'].view(-1, 3)], dim=0)
            ptr.append(idx_ptr + natoms)
            batch_mol = torch.ones((natoms), device=device, dtype=torch.long) * idx
            batch = torch.cat([batch, batch_mol], dim=0)
            idx += 1
            idx_ptr = idx_ptr + natoms
        ptr = torch.tensor(ptr)
        return Data(x=x, pos=pos, ptr=ptr, batch=batch)
            

    def inference(self, input_data, denormalize_output=False, atom_ref=False):
        """
        Perform inference with the loaded model.
        
        Args:
            input_data: Input data for the model
            denormalize_output (bool): Whether to denormalize energy and forces
            
        Returns:
            dict: Model output dictionary with keys like 'energy', 'forces', 'embedding'
                If denormalize_output=True, energy and forces will be denormalized
                
        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            raise ValueError("Model must be loaded before inference.")
        
        self.model.eval()
        with torch.no_grad():
            input_data = input_data.to(device=self.device)
            output = self.model(input_data)
        
        # Denormalize if requested
        if denormalize_output:
            if isinstance(output, dict):
                if 'energy' in output and self.normalizer is not None:
                    output['energy'] = self.denormalize(output['energy'], target='energy', batch=input_data.batch, elements=input_data.x.long(), atom_ref=atom_ref)
                if 'forces' in output and self.normalizer is not None:
                    output['forces'] = self.denormalize(output['forces'], target='forces', atom_ref=False)
        
        return output
