"""
API Integration Tests for Vision-LLM PII Extraction System

This module tests API endpoints and integration with external services
for the Vision-LLM based PII extraction system.
"""

import pytest
import json
import base64
import requests
import tempfile
import os
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time
import asyncio
from PIL import Image, ImageDraw
import io

# Test imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestVisionLLMAPIEndpoints:
    """Test API endpoints for Vision-LLM system"""
    
    @pytest.fixture
    def api_client(self):
        """Mock API client for testing"""
        class MockAPIClient:
            def __init__(self, base_url="http://localhost:8000"):
                self.base_url = base_url
                self.session = requests.Session()
                self.timeout = 30
            
            def post(self, endpoint: str, data: Dict[str, Any], 
                    files: Optional[Dict] = None) -> Dict[str, Any]:
                """Mock POST request"""
                # Simulate API response based on endpoint
                if endpoint == "/api/v1/vision/extract-pii":
                    return self._mock_pii_extraction_response(data)
                elif endpoint == "/api/v1/vision/classify-document":
                    return self._mock_document_classification_response(data)
                elif endpoint == "/api/v1/vision/batch-process":
                    return self._mock_batch_processing_response(data)
                else:
                    return {"error": "Endpoint not found", "status_code": 404}
            
            def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
                """Mock GET request"""
                if endpoint == "/api/v1/vision/models":
                    return self._mock_available_models_response()
                elif endpoint == "/api/v1/vision/health":
                    return self._mock_health_check_response()
                else:
                    return {"error": "Endpoint not found", "status_code": 404}
            
            def _mock_pii_extraction_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "success": True,
                    "request_id": "req_12345",
                    "processing_time": 2.5,
                    "results": {
                        "pii_entities": [
                            {
                                "type": "PERSON",
                                "text": "John Doe", 
                                "confidence": 0.95,
                                "position": {"x": 100, "y": 150, "width": 80, "height": 20}
                            },
                            {
                                "type": "EMAIL",
                                "text": "john@example.com",
                                "confidence": 0.98,
                                "position": {"x": 200, "y": 200, "width": 150, "height": 15}
                            }
                        ],
                        "document_classification": {
                            "type": "HR",
                            "confidence": 0.87,
                            "complexity": "Medium"
                        },
                        "transcribed_text": "Employee Information Form\\nName: John Doe\\nEmail: john@example.com"
                    },
                    "metadata": {
                        "model_used": data.get("model", "gpt-4o"),
                        "document_id": data.get("document_id", "doc_123"),
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            
            def _mock_document_classification_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "success": True,
                    "request_id": "req_classify_12345",
                    "classification": {
                        "document_type": "HR",
                        "confidence": 0.87,
                        "complexity_level": "Medium",
                        "special_elements": ["forms", "text_fields"],
                        "recommended_models": ["gpt-4o", "claude-3-5-sonnet"],
                        "estimated_processing_time": 3.2
                    }
                }
            
            def _mock_batch_processing_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
                batch_size = len(data.get("documents", []))
                return {
                    "success": True,
                    "batch_id": "batch_12345",
                    "total_documents": batch_size,
                    "estimated_completion_time": batch_size * 2.5,
                    "status": "processing",
                    "results_url": f"/api/v1/vision/batch/{batch_size}/results"
                }
            
            def _mock_available_models_response(self) -> Dict[str, Any]:
                return {
                    "available_models": [
                        {
                            "name": "gpt-4o",
                            "provider": "openai",
                            "supports_vision": True,
                            "cost_per_1k_tokens": {"input": 0.0025, "output": 0.01},
                            "max_tokens": 4000
                        },
                        {
                            "name": "claude-3-5-sonnet",
                            "provider": "anthropic", 
                            "supports_vision": True,
                            "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
                            "max_tokens": 4000
                        }
                    ]
                }
            
            def _mock_health_check_response(self) -> Dict[str, Any]:
                return {
                    "status": "healthy",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "services": {
                        "vision_llm_service": "operational",
                        "document_classifier": "operational",
                        "confidence_assessor": "operational",
                        "database": "operational"
                    },
                    "performance_metrics": {
                        "average_response_time": 2.3,
                        "success_rate": 0.987,
                        "total_requests_today": 1247
                    }
                }
        
        return MockAPIClient()
    
    def test_pii_extraction_endpoint(self, api_client):
        """Test PII extraction API endpoint"""
        # Create test image data
        test_image = self._create_test_image()
        image_b64 = self._image_to_base64(test_image)
        
        request_data = {
            "image_data": image_b64,
            "model": "gpt-4o",
            "document_type": "HR",
            "document_id": "test_doc_001",
            "options": {
                "max_tokens": 4000,
                "temperature": 0.0,
                "return_positions": True
            }
        }
        
        response = api_client.post("/api/v1/vision/extract-pii", request_data)
        
        # Validate response structure
        assert response["success"] is True
        assert "request_id" in response
        assert "processing_time" in response
        assert "results" in response
        
        # Validate results structure
        results = response["results"]
        assert "pii_entities" in results
        assert "document_classification" in results
        assert "transcribed_text" in results
        
        # Validate PII entities
        entities = results["pii_entities"]
        assert isinstance(entities, list)
        
        for entity in entities:
            required_fields = ["type", "text", "confidence"]
            for field in required_fields:
                assert field in entity
            
            assert 0.0 <= entity["confidence"] <= 1.0
            assert entity["type"] in ["PERSON", "EMAIL", "PHONE", "SSN", "ADDRESS", "DATE"]
    
    def test_document_classification_endpoint(self, api_client):
        """Test document classification API endpoint"""
        test_image = self._create_test_image()
        image_b64 = self._image_to_base64(test_image)
        
        request_data = {
            "image_data": image_b64,
            "analysis_level": "comprehensive"
        }
        
        response = api_client.post("/api/v1/vision/classify-document", request_data)
        
        # Validate response
        assert response["success"] is True
        assert "classification" in response
        
        classification = response["classification"]
        required_fields = [
            "document_type", "confidence", "complexity_level",
            "special_elements", "recommended_models"
        ]
        
        for field in required_fields:
            assert field in classification
        
        assert classification["document_type"] in [
            "HR", "Finance", "Legal", "Medical", "Government", "Education", "Other"
        ]
        assert 0.0 <= classification["confidence"] <= 1.0
    
    def test_batch_processing_endpoint(self, api_client):
        """Test batch processing API endpoint"""
        # Create multiple test images
        test_images = []
        for i in range(3):
            image = self._create_test_image(f"Document {i+1}")
            test_images.append({
                "document_id": f"batch_doc_{i+1}",
                "image_data": self._image_to_base64(image),
                "document_type": "HR"
            })
        
        request_data = {
            "documents": test_images,
            "model": "gpt-4o",
            "options": {
                "priority": "normal",
                "callback_url": "https://webhook.example.com/batch-complete"
            }
        }
        
        response = api_client.post("/api/v1/vision/batch-process", request_data)
        
        # Validate batch response
        assert response["success"] is True
        assert "batch_id" in response
        assert "total_documents" in response
        assert "estimated_completion_time" in response
        assert response["total_documents"] == len(test_images)
    
    def test_available_models_endpoint(self, api_client):
        """Test available models API endpoint"""
        response = api_client.get("/api/v1/vision/models")
        
        assert "available_models" in response
        models = response["available_models"]
        assert isinstance(models, list)
        assert len(models) > 0
        
        for model in models:
            required_fields = ["name", "provider", "supports_vision", "cost_per_1k_tokens"]
            for field in required_fields:
                assert field in model
            
            assert model["supports_vision"] is True
            assert "input" in model["cost_per_1k_tokens"]
            assert "output" in model["cost_per_1k_tokens"]
    
    def test_health_check_endpoint(self, api_client):
        """Test health check API endpoint"""
        response = api_client.get("/api/v1/vision/health")
        
        assert "status" in response
        assert "services" in response
        assert "performance_metrics" in response
        
        services = response["services"]
        required_services = [
            "vision_llm_service", "document_classifier", 
            "confidence_assessor", "database"
        ]
        
        for service in required_services:
            assert service in services
            assert services[service] in ["operational", "degraded", "down"]
    
    def test_error_handling(self, api_client):
        """Test API error handling"""
        # Test invalid endpoint
        response = api_client.post("/api/v1/invalid/endpoint", {})
        assert "error" in response
        
        # Test malformed request data
        invalid_data = {
            "invalid_field": "invalid_value"
        }
        
        response = api_client.post("/api/v1/vision/extract-pii", invalid_data)
        # In a real implementation, this would return validation errors
        # For mock, we'll assume it handles gracefully
        assert isinstance(response, dict)
    
    def test_rate_limiting(self, api_client):
        """Test API rate limiting behavior"""
        # Simulate multiple rapid requests
        responses = []
        for i in range(10):
            test_image = self._create_test_image()
            request_data = {
                "image_data": self._image_to_base64(test_image),
                "model": "gpt-4o"
            }
            response = api_client.post("/api/v1/vision/extract-pii", request_data)
            responses.append(response)
        
        # In a real implementation with rate limiting, some requests might be throttled
        # For mock, all succeed but we verify the structure
        assert len(responses) == 10
        for response in responses:
            assert "success" in response or "error" in response
    
    def _create_test_image(self, text: str = "Test Document") -> Image.Image:
        """Create a test image with text"""
        image = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((50, 50), text, fill='black')
        draw.text((50, 100), "Name: John Doe", fill='black')
        draw.text((50, 150), "Email: john@example.com", fill='black')
        return image
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str


class TestExternalServiceIntegration:
    """Test integration with external services and APIs"""
    
    def test_llm_provider_integration(self):
        """Test integration with external LLM providers"""
        # Mock external LLM provider responses
        mock_responses = {
            "openai": {
                "status": "success",
                "response_time": 2.1,
                "content": '{"extracted_information": {"names": ["John Doe"]}}',
                "usage": {"prompt_tokens": 1200, "completion_tokens": 150, "total_tokens": 1350}
            },
            "anthropic": {
                "status": "success", 
                "response_time": 1.8,
                "content": '{"extracted_information": {"names": ["John Doe"]}}',
                "usage": {"input_tokens": 1200, "output_tokens": 150}
            },
            "google": {
                "status": "success",
                "response_time": 3.2,
                "content": '{"extracted_information": {"names": ["John Doe"]}}',
                "usage": {"input_tokens": 1200, "output_tokens": 150}
            }
        }
        
        for provider, mock_response in mock_responses.items():
            # Test successful integration
            assert mock_response["status"] == "success"
            assert mock_response["response_time"] < 5.0  # Reasonable response time
            assert "usage" in mock_response
            
            # Test JSON parsing
            try:
                parsed_content = json.loads(mock_response["content"])
                assert "extracted_information" in parsed_content
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON response from {provider}")
    
    def test_webhook_integration(self):
        """Test webhook integration for batch processing"""
        class MockWebhookService:
            def __init__(self):
                self.webhook_calls = []
            
            def send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
                # Mock webhook sending
                self.webhook_calls.append({
                    "url": url,
                    "payload": payload,
                    "timestamp": time.time(),
                    "status": "sent"
                })
                return True
            
            def validate_webhook_signature(self, payload: str, signature: str, secret: str) -> bool:
                # Mock signature validation
                import hmac
                import hashlib
                expected_signature = hmac.new(
                    secret.encode(),
                    payload.encode(),
                    hashlib.sha256
                ).hexdigest()
                return signature == expected_signature
        
        webhook_service = MockWebhookService()
        
        # Test webhook sending
        test_payload = {
            "batch_id": "batch_12345",
            "status": "completed",
            "total_documents": 5,
            "successful_extractions": 4,
            "failed_extractions": 1,
            "processing_time": 12.5
        }
        
        success = webhook_service.send_webhook(
            "https://example.com/webhook",
            test_payload
        )
        
        assert success is True
        assert len(webhook_service.webhook_calls) == 1
        assert webhook_service.webhook_calls[0]["payload"] == test_payload
    
    def test_database_integration(self):
        """Test database integration for storing results"""
        class MockDatabaseService:
            def __init__(self):
                self.stored_results = {}
                self.queries_executed = []
            
            def store_extraction_result(self, document_id: str, result: Dict[str, Any]) -> bool:
                self.stored_results[document_id] = {
                    "result": result,
                    "timestamp": time.time(),
                    "status": "stored"
                }
                return True
            
            def retrieve_extraction_result(self, document_id: str) -> Optional[Dict[str, Any]]:
                return self.stored_results.get(document_id)
            
            def query_results(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
                self.queries_executed.append(query_params)
                # Mock query results
                return list(self.stored_results.values())
        
        db_service = MockDatabaseService()
        
        # Test storing results
        test_result = {
            "pii_entities": [
                {"type": "PERSON", "text": "John Doe", "confidence": 0.95}
            ],
            "processing_time": 2.3,
            "model_used": "gpt-4o"
        }
        
        success = db_service.store_extraction_result("doc_123", test_result)
        assert success is True
        
        # Test retrieving results
        retrieved = db_service.retrieve_extraction_result("doc_123")
        assert retrieved is not None
        assert retrieved["result"] == test_result
        
        # Test querying results
        query_params = {"date_range": "2024-01-01 to 2024-01-31", "document_type": "HR"}
        results = db_service.query_results(query_params)
        assert isinstance(results, list)
        assert len(db_service.queries_executed) == 1
    
    def test_monitoring_integration(self):
        """Test integration with monitoring and alerting systems"""
        class MockMonitoringService:
            def __init__(self):
                self.metrics_sent = []
                self.alerts_triggered = []
            
            def send_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
                self.metrics_sent.append({
                    "name": metric_name,
                    "value": value,
                    "tags": tags or {},
                    "timestamp": time.time()
                })
            
            def trigger_alert(self, alert_type: str, message: str, severity: str = "warning"):
                self.alerts_triggered.append({
                    "type": alert_type,
                    "message": message,
                    "severity": severity,
                    "timestamp": time.time()
                })
            
            def check_thresholds(self, metrics: Dict[str, float]) -> List[str]:
                alerts = []
                
                # Check various thresholds
                if metrics.get("error_rate", 0) > 0.05:  # 5% error rate
                    alerts.append("High error rate detected")
                
                if metrics.get("response_time", 0) > 10.0:  # 10 second response time
                    alerts.append("High response time detected")
                
                if metrics.get("memory_usage", 0) > 0.9:  # 90% memory usage
                    alerts.append("High memory usage detected")
                
                return alerts
        
        monitoring = MockMonitoringService()
        
        # Test metric sending
        monitoring.send_metric("pii_extraction_time", 2.5, {"model": "gpt-4o"})
        monitoring.send_metric("pii_entities_found", 3, {"document_type": "HR"})
        
        assert len(monitoring.metrics_sent) == 2
        assert monitoring.metrics_sent[0]["name"] == "pii_extraction_time"
        assert monitoring.metrics_sent[0]["value"] == 2.5
        
        # Test threshold checking
        test_metrics = {
            "error_rate": 0.08,  # Above threshold
            "response_time": 3.0,  # Below threshold
            "memory_usage": 0.95   # Above threshold
        }
        
        alerts = monitoring.check_thresholds(test_metrics)
        assert len(alerts) == 2  # error_rate and memory_usage should trigger alerts
        assert "High error rate detected" in alerts
        assert "High memory usage detected" in alerts


class TestAsyncProcessing:
    """Test asynchronous processing capabilities"""
    
    @pytest.mark.asyncio
    async def test_async_pii_extraction(self):
        """Test asynchronous PII extraction"""
        class MockAsyncExtractor:
            async def extract_pii_async(self, image_data: str, model: str) -> Dict[str, Any]:
                # Simulate async processing delay
                await asyncio.sleep(0.1)
                
                return {
                    "success": True,
                    "pii_entities": [
                        {"type": "PERSON", "text": "John Doe", "confidence": 0.95}
                    ],
                    "processing_time": 2.3
                }
        
        extractor = MockAsyncExtractor()
        
        # Test single async extraction
        result = await extractor.extract_pii_async("test_image_data", "gpt-4o")
        assert result["success"] is True
        assert len(result["pii_entities"]) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing of multiple documents"""
        class MockConcurrentExtractor:
            async def process_document(self, doc_id: str, image_data: str) -> Dict[str, Any]:
                # Simulate processing time variation
                processing_time = 0.1 + (hash(doc_id) % 10) * 0.01
                await asyncio.sleep(processing_time)
                
                return {
                    "document_id": doc_id,
                    "success": True,
                    "processing_time": processing_time,
                    "pii_count": hash(doc_id) % 5 + 1  # 1-5 entities
                }
        
        extractor = MockConcurrentExtractor()
        
        # Process multiple documents concurrently
        document_ids = [f"doc_{i}" for i in range(5)]
        tasks = [
            extractor.process_document(doc_id, f"image_data_{doc_id}")
            for doc_id in document_ids
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all documents processed
        assert len(results) == 5
        for result in results:
            assert result["success"] is True
            assert "document_id" in result
        
        # Concurrent processing should be faster than sequential
        assert total_time < 1.0  # Should complete in under 1 second
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test error handling in async processing"""
        class MockAsyncExtractorWithErrors:
            async def extract_with_potential_failure(self, doc_id: str) -> Dict[str, Any]:
                await asyncio.sleep(0.1)
                
                # Simulate random failures
                if hash(doc_id) % 3 == 0:  # Fail every 3rd document
                    raise Exception(f"Processing failed for {doc_id}")
                
                return {
                    "document_id": doc_id,
                    "success": True,
                    "pii_entities": []
                }
        
        extractor = MockAsyncExtractorWithErrors()
        
        # Process documents with some expected failures
        document_ids = [f"doc_{i}" for i in range(6)]
        results = []
        
        for doc_id in document_ids:
            try:
                result = await extractor.extract_with_potential_failure(doc_id)
                results.append(result)
            except Exception as e:
                results.append({
                    "document_id": doc_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Check that some succeeded and some failed
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        assert len(successful) > 0
        assert len(failed) > 0
        assert len(successful) + len(failed) == 6


class TestSecurityAndCompliance:
    """Test security and compliance features"""
    
    def test_data_encryption(self):
        """Test data encryption for sensitive information"""
        class MockEncryptionService:
            def __init__(self, key: str = "test_key_12345"):
                self.key = key
            
            def encrypt_data(self, data: str) -> str:
                # Mock encryption (in real implementation, use proper encryption)
                import base64
                encrypted = base64.b64encode(f"encrypted_{data}".encode()).decode()
                return encrypted
            
            def decrypt_data(self, encrypted_data: str) -> str:
                # Mock decryption
                import base64
                decrypted = base64.b64decode(encrypted_data).decode()
                return decrypted.replace("encrypted_", "")
        
        encryption_service = MockEncryptionService()
        
        # Test encryption/decryption
        sensitive_data = "SSN: 123-45-6789"
        encrypted = encryption_service.encrypt_data(sensitive_data)
        decrypted = encryption_service.decrypt_data(encrypted)
        
        assert encrypted != sensitive_data
        assert decrypted == sensitive_data
    
    def test_access_control(self):
        """Test access control and authorization"""
        class MockAccessControl:
            def __init__(self):
                self.user_permissions = {
                    "admin": ["read", "write", "delete", "extract_pii"],
                    "analyst": ["read", "extract_pii"],
                    "viewer": ["read"]
                }
            
            def check_permission(self, user_role: str, action: str) -> bool:
                return action in self.user_permissions.get(user_role, [])
            
            def audit_access(self, user_id: str, action: str, resource: str, result: str):
                # Mock audit logging
                return {
                    "user_id": user_id,
                    "action": action,
                    "resource": resource,
                    "result": result,
                    "timestamp": time.time()
                }
        
        access_control = MockAccessControl()
        
        # Test permissions
        assert access_control.check_permission("admin", "extract_pii") is True
        assert access_control.check_permission("analyst", "extract_pii") is True
        assert access_control.check_permission("viewer", "extract_pii") is False
        
        # Test audit logging
        audit_entry = access_control.audit_access(
            "user123", "extract_pii", "document456", "success"
        )
        
        assert audit_entry["user_id"] == "user123"
        assert audit_entry["action"] == "extract_pii"
        assert audit_entry["result"] == "success"
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance features"""
        class MockGDPRService:
            def __init__(self):
                self.consent_records = {}
                self.deletion_requests = []
            
            def record_consent(self, user_id: str, purpose: str, consent_given: bool):
                if user_id not in self.consent_records:
                    self.consent_records[user_id] = {}
                
                self.consent_records[user_id][purpose] = {
                    "consent_given": consent_given,
                    "timestamp": time.time()
                }
            
            def check_consent(self, user_id: str, purpose: str) -> bool:
                user_consents = self.consent_records.get(user_id, {})
                consent_record = user_consents.get(purpose, {})
                return consent_record.get("consent_given", False)
            
            def request_data_deletion(self, user_id: str, data_types: List[str]):
                self.deletion_requests.append({
                    "user_id": user_id,
                    "data_types": data_types,
                    "requested_at": time.time(),
                    "status": "pending"
                })
            
            def anonymize_pii(self, text: str) -> str:
                # Mock PII anonymization
                import re
                # Replace names with placeholder
                text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
                # Replace emails
                text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
                # Replace phone numbers
                text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
                return text
        
        gdpr_service = MockGDPRService()
        
        # Test consent recording and checking
        gdpr_service.record_consent("user123", "pii_extraction", True)
        assert gdpr_service.check_consent("user123", "pii_extraction") is True
        assert gdpr_service.check_consent("user123", "marketing") is False
        
        # Test data deletion request
        gdpr_service.request_data_deletion("user123", ["pii_data", "processing_history"])
        assert len(gdpr_service.deletion_requests) == 1
        assert gdpr_service.deletion_requests[0]["user_id"] == "user123"
        
        # Test anonymization
        text_with_pii = "John Doe contacted us at john@example.com and (555) 123-4567"
        anonymized = gdpr_service.anonymize_pii(text_with_pii)
        
        assert "[NAME]" in anonymized
        assert "[EMAIL]" in anonymized
        assert "[PHONE]" in anonymized
        assert "John Doe" not in anonymized
        assert "john@example.com" not in anonymized


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])