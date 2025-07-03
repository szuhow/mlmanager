import os
import sys
import django
import logging
from django.test import TestCase, Client
from django.urls import reverse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup Django
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
django.setup()

# Import models after Django setup
from apps.ml_manager.models import Model

class TemplateErrorTests(TestCase):
    """Test class to verify template error fixes."""

    def setUp(self):
        self.client = Client()
        self.logger = logging.getLogger(__name__)
        
        # Find the first model in the database
        self.model = Model.objects.first()
        if not self.model:
            self.logger.error("No models found in the database. Please create at least one model.")
            raise ValueError("No models found in the database")

        self.logger.info(f"Testing with model: {self.model.name} (ID: {self.model.pk})")

    def test_model_detail_page(self):
        """Test that the model detail page loads without template errors."""
        url = reverse('ml_manager:model-detail', kwargs={'pk': self.model.pk})
        self.logger.info(f"Testing URL: {url}")
        
        response = self.client.get(url)
        
        # Check status code
        self.assertEqual(response.status_code, 200)
        self.logger.info("✅ Model detail page loaded successfully")
        
        # The presence of certain elements indicates no template errors
        self.assertIn(b'<title>', response.content)
        self.assertIn(b'</html>', response.content)
        self.logger.info("✅ Page content looks valid")

    def test_model_list_page(self):
        """Test that the model list page loads without template errors."""
        url = reverse('ml_manager:model-list')
        self.logger.info(f"Testing URL: {url}")
        
        response = self.client.get(url)
        
        # Check status code
        self.assertEqual(response.status_code, 200)
        self.logger.info("✅ Model list page loaded successfully")
        
        # The presence of certain elements indicates no template errors
        self.assertIn(b'<title>', response.content)
        self.assertIn(b'</html>', response.content)
        self.logger.info("✅ Page content looks valid")

if __name__ == "__main__":
    # Create a test suite with our test classes
    from unittest import TestLoader, TextTestRunner
    suite = TestLoader().loadTestsFromTestCase(TemplateErrorTests)
    
    # Run the tests
    result = TextTestRunner(verbosity=2).run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())
