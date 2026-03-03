import pytest
from suiren.loader import ModelLoader

def test_load_model_valid_config():
    config_path = "examples/config.yml"
    model_loader = ModelLoader(config_path)
    model = model_loader.load_model()
    assert model is not None

def test_load_weights_valid_model():
    config_path = "examples/config.yml"
    model_loader = ModelLoader(config_path)
    model = model_loader.load_model()
    weights_path = "path/to/weights.pt"  # Update with actual weights path
    model_loader.load_weights(model, weights_path)
    assert model.state_dict() is not None

def test_load_model_invalid_config():
    invalid_config_path = "invalid/path/to/config.yml"
    model_loader = ModelLoader(invalid_config_path)
    with pytest.raises(FileNotFoundError):
        model_loader.load_model()

def test_load_weights_invalid_model():
    config_path = "examples/config.yml"
    model_loader = ModelLoader(config_path)
    model = model_loader.load_model()
    invalid_weights_path = "invalid/path/to/weights.pt"
    with pytest.raises(RuntimeError):
        model_loader.load_weights(model, invalid_weights_path)