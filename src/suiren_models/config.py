class Config:
    def __init__(self, config_path):
        import yaml
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_model_params(self):
        return self.config.get('model', {})

    def get_training_params(self):
        return self.config.get('training', {})

    def get_inference_params(self):
        return self.config.get('inference', {})