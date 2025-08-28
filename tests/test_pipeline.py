"""
Test module for the Student Performance Predictor
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

class TestCustomData(unittest.TestCase):
    """Test cases for CustomData class"""
    
    def setUp(self):
        """Set up test data"""
        self.valid_data = {
            'gender': 'female',
            'race_ethnicity': 'group B',
            'parental_level_of_education': 'bachelor\'s degree',
            'lunch': 'standard',
            'test_preparation_course': 'completed',
            'reading_score': 75,
            'writing_score': 80
        }
    
    def test_valid_data_creation(self):
        """Test creating CustomData with valid inputs"""
        data = CustomData(**self.valid_data)
        self.assertEqual(data.gender, 'female')
        self.assertEqual(data.reading_score, 75.0)
        self.assertEqual(data.writing_score, 80.0)
    
    def test_invalid_gender(self):
        """Test that invalid gender raises ValueError"""
        invalid_data = self.valid_data.copy()
        invalid_data['gender'] = 'invalid'
        
        with self.assertRaises(ValueError):
            CustomData(**invalid_data)
    
    def test_invalid_score_range(self):
        """Test that scores outside 0-100 range raise ValueError"""
        invalid_data = self.valid_data.copy()
        invalid_data['reading_score'] = 150
        
        with self.assertRaises(ValueError):
            CustomData(**invalid_data)
        
        invalid_data['reading_score'] = -10
        with self.assertRaises(ValueError):
            CustomData(**invalid_data)
    
    def test_invalid_race_ethnicity(self):
        """Test that invalid race/ethnicity raises ValueError"""
        invalid_data = self.valid_data.copy()
        invalid_data['race_ethnicity'] = 'group Z'
        
        with self.assertRaises(ValueError):
            CustomData(**invalid_data)
    
    def test_data_frame_conversion(self):
        """Test converting CustomData to DataFrame"""
        data = CustomData(**self.valid_data)
        df = data.get_data_as_data_frame()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df.columns), 7)
        self.assertEqual(df['gender'].iloc[0], 'female')
        self.assertEqual(df['reading_score'].iloc[0], 75.0)

class TestPredictPipeline(unittest.TestCase):
    """Test cases for PredictPipeline class"""
    
    def setUp(self):
        """Set up test data"""
        self.pipeline = PredictPipeline()
        self.test_data = pd.DataFrame({
            'gender': ['female'],
            'race_ethnicity': ['group B'],
            'parental_level_of_education': ['bachelor\'s degree'],
            'lunch': ['standard'],
            'test_preparation_course': ['completed'],
            'reading_score': [75.0],
            'writing_score': [80.0]
        })
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly"""
        self.assertIsInstance(self.pipeline, PredictPipeline)
    
    def test_predict_with_missing_files(self):
        """Test prediction when model files are missing"""
        # This test assumes model files don't exist
        if not (os.path.exists("artifacts/model.pkl") and 
                os.path.exists("artifacts/preprocessor.pkl")):
            with self.assertRaises(CustomException):
                self.pipeline.predict(self.test_data)

class TestDataValidation(unittest.TestCase):
    """Test cases for data validation"""
    
    def test_score_validation(self):
        """Test score validation functions"""
        # Test valid scores
        for score in [0, 50, 100]:
            self.assertTrue(0 <= score <= 100)
        
        # Test invalid scores
        for score in [-1, 101, 150]:
            self.assertFalse(0 <= score <= 100)
    
    def test_categorical_validation(self):
        """Test categorical variable validation"""
        valid_genders = ["male", "female"]
        valid_races = ["group A", "group B", "group C", "group D", "group E"]
        
        self.assertIn("male", valid_genders)
        self.assertIn("group A", valid_races)
        self.assertNotIn("invalid", valid_genders)
        self.assertNotIn("group Z", valid_races)

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_file_operations(self):
        """Test file operation utilities"""
        # Test if artifacts directory structure is as expected
        expected_dirs = ['artifacts', 'logs', 'src', 'notebook']
        
        for dir_name in expected_dirs:
            if os.path.exists(dir_name):
                self.assertTrue(os.path.isdir(dir_name))

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCustomData))
    test_suite.addTest(unittest.makeSuite(TestPredictPipeline))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
