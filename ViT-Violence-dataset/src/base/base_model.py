class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        # Save as .h5 instead of .tf
        if not checkpoint_path.endswith('.h5'):
            checkpoint_path = checkpoint_path.rsplit('.', 1)[0] + '.h5'
        self.model.save(checkpoint_path, save_format='h5')
        print("Model saved as .h5")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError
