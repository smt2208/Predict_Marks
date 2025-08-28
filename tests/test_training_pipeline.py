import os
import unittest
from src.pipeline.train_pipeline import TrainPipeline
from src.config import config


class TestTrainingPipeline(unittest.TestCase):
    def test_training_creates_artifacts(self):
        pipeline = TrainPipeline()
        try:
            score = pipeline.start_training()
        except Exception as e:
            self.fail(f"Training pipeline raised an exception: {e}")

        self.assertTrue(os.path.exists(config.MODEL_FILE_PATH), "Model file not created")
        self.assertTrue(os.path.exists(config.PREPROCESSOR_FILE_PATH), "Preprocessor file not created")
        metadata_path = os.path.join(config.ARTIFACTS_DIR, "model_metadata.json")
        self.assertTrue(os.path.exists(metadata_path), "Metadata file not created")
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()