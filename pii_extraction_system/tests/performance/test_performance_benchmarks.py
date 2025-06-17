"""Performance benchmarking tests for PII extraction system."""

import pytest
import time
import psutil
import os
import threading
import statistics
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil

from src.core.pipeline import PIIExtractionPipeline
from src.core.config import PIIConfig
from src.extractors.rule_based import RuleBasedExtractor


class TestPerformanceBenchmarks:
    """Performance benchmarking test suite for PII extraction system."""
    
    @classmethod
    def setup_class(cls):
        """Set up performance testing environment."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.perf_test_dir = cls.temp_dir / "performance_tests"
        cls.perf_test_dir.mkdir(exist_ok=True)
        
        # Initialize system for performance testing
        cls.config = PIIConfig()
        cls.config.output_dir = str(cls.perf_test_dir)
        cls.config.use_local_storage = True
        
        cls.pipeline = PIIExtractionPipeline(cls.config)
        cls.rule_extractor = RuleBasedExtractor()
        
        # Performance metrics storage
        cls.performance_metrics = {
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'throughput': [],
            'accuracy_metrics': []
        }
        
        # Create performance test datasets
        cls._create_performance_test_datasets()
    
    @classmethod
    def _create_performance_test_datasets(cls):
        """Create datasets of various sizes for performance testing."""
        
        # Small document (< 1KB)
        small_doc = """
        Name: John Smith
        Email: john.smith@example.com
        Phone: (555) 123-4567
        SSN: 123-45-6789
        Address: 123 Main Street
        """
        (cls.perf_test_dir / "small_doc.txt").write_text(small_doc)
        
        # Medium document (~ 10KB)
        medium_doc = "EMPLOYEE DATABASE EXPORT\n\n"
        for i in range(50):
            employee = f"""
            Employee {i+1:03d}:
            Name: Employee{i:03d} Test{i:03d}
            Email: emp{i:03d}@company.com
            Phone: (555) {i:03d}-{(i*7)%10000:04d}
            Employee ID: EMP-{i:06d}
            Department: Engineering
            Address: {i+1} Business Drive, Suite {i%100+1}
            Date of Birth: {(i%12)+1:02d}/{(i%28)+1:02d}/{1970+(i%50)}
            SSN: {(i%900)+100:03d}-{(i%90)+10:02d}-{(i%9000)+1000:04d}
            Salary: ${45000 + (i * 1000):,}
            
            """
            medium_doc += employee
        (cls.perf_test_dir / "medium_doc.txt").write_text(medium_doc)
        
        # Large document (~ 100KB)
        large_doc = "COMPREHENSIVE EMPLOYEE AND CUSTOMER DATABASE\n\n"
        for i in range(500):
            record = f"""
            Record {i+1:04d}:
            Type: {'Employee' if i % 3 == 0 else 'Customer'}
            Name: Person{i:04d} Lastname{i:04d}
            Primary Email: person{i:04d}@{'company' if i % 3 == 0 else 'client'}.com
            Secondary Email: {['backup', 'personal', 'work'][i%3]}{i:04d}@email.org
            Phone: (555) {i:04d}-{(i*13)%10000:04d}
            Mobile: +1-800-{i:04d}-{(i*17)%10000:04d}
            ID: {'EMP' if i % 3 == 0 else 'CUS'}-{i:08d}
            Address: {i+1} Street Name {['Drive', 'Avenue', 'Boulevard'][i%3]}
            City: {'Toronto' if i % 4 == 0 else 'Vancouver' if i % 4 == 1 else 'Montreal' if i % 4 == 2 else 'Calgary'}
            Postal: {'M' if i % 4 == 0 else 'V' if i % 4 == 1 else 'H' if i % 4 == 2 else 'T'}{(i%9)+1}{'ABCDEFGHIJ'[i%10]} {(i%9)+1}{'KLMNOPQRST'[i%10]}{(i%9)+1}
            DOB: {(i%12)+1:02d}/{(i%28)+1:02d}/{1950+(i%70)}
            SIN: {(i%900)+100:03d}-{(i%900)+100:03d}-{(i%900)+100:03d}
            Credit Card: {4000 + (i%1000):04d}-{(i%9000)+1000:04d}-{(i%9000)+1000:04d}-{(i%9000)+1000:04d}
            Notes: Additional information for record {i+1}
            
            """
            large_doc += record
        (cls.perf_test_dir / "large_doc.txt").write_text(large_doc)
        
        # Very large document (~ 1MB)
        very_large_doc = "MASSIVE DATABASE EXPORT FOR PERFORMANCE TESTING\n\n"
        for i in range(2000):
            detailed_record = f"""
            ================== RECORD {i+1:05d} ==================
            Personal Information:
            Full Legal Name: {['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa'][i%6]} {['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Garcia'][i%6]}
            Preferred Name: {['Johnny', 'Janie', 'Mike', 'Sally', 'Dave', 'Liz'][i%6]}
            Date of Birth: {(i%12)+1:02d}/{(i%28)+1:02d}/{1940+(i%80)}
            Place of Birth: {'Toronto, ON' if i % 5 == 0 else 'Vancouver, BC' if i % 5 == 1 else 'Montreal, QC' if i % 5 == 2 else 'Calgary, AB' if i % 5 == 3 else 'Ottawa, ON'}
            Gender: {'Male' if i % 2 == 0 else 'Female'}
            Marital Status: {'Single' if i % 4 == 0 else 'Married' if i % 4 == 1 else 'Divorced' if i % 4 == 2 else 'Widowed'}
            
            Identification:
            Social Insurance Number: {(i%900)+100:03d}-{(i%900)+100:03d}-{(i%900)+100:03d}
            Driver's License: {['A', 'B', 'C', 'D', 'E'][i%5]}{i:08d}
            Passport: {'CA' if i % 3 == 0 else 'US' if i % 3 == 1 else 'UK'}{i:08d}
            Health Card: {i:04d}-{i:03d}-{i:03d}-{'ON' if i % 4 == 0 else 'BC' if i % 4 == 1 else 'QC' if i % 4 == 2 else 'AB'}
            
            Contact Information:
            Primary Email: {['john', 'jane', 'michael', 'sarah', 'david', 'lisa'][i%6]}.{['smith', 'johnson', 'brown', 'davis', 'wilson', 'garcia'][i%6]}@{'gmail.com' if i % 3 == 0 else 'yahoo.com' if i % 3 == 1 else 'hotmail.com'}
            Work Email: {['j', 'ja', 'mi', 'sa', 'd', 'li'][i%6]}.{['smith', 'johnson', 'brown', 'davis', 'wilson', 'garcia'][i%6]}@company{(i%100)+1:02d}.com
            Personal Phone: (416) {i:03d}-{(i*7)%10000:04d}
            Work Phone: (647) {i:03d}-{(i*11)%10000:04d}
            Mobile: +1-{800+(i%200):03d}-{i:03d}-{(i*13)%10000:04d}
            Fax: (905) {i:03d}-{(i*17)%10000:04d}
            
            Address Information:
            Home Address: {i+1} {['Main', 'Oak', 'Pine', 'Elm', 'First', 'Second'][i%6]} {['Street', 'Avenue', 'Drive', 'Boulevard', 'Lane', 'Road'][i%6]}
            Unit/Apt: {'Unit ' + str((i%500)+1) if i % 3 == 0 else 'Apt ' + str((i%200)+1) if i % 3 == 1 else ''}
            City: {'Toronto' if i % 6 == 0 else 'Mississauga' if i % 6 == 1 else 'Brampton' if i % 6 == 2 else 'Markham' if i % 6 == 3 else 'Vaughan' if i % 6 == 4 else 'Richmond Hill'}
            Province: {'Ontario' if i % 4 == 0 else 'British Columbia' if i % 4 == 1 else 'Quebec' if i % 4 == 2 else 'Alberta'}
            Postal Code: {'M' if i % 4 == 0 else 'V' if i % 4 == 1 else 'H' if i % 4 == 2 else 'T'}{(i%9)+1}{'ABCDEFGHIJ'[i%10]} {(i%9)+1}{'KLMNOPQRST'[i%10]}{(i%9)+1}
            Country: Canada
            
            Financial Information:
            Bank Institution: {'RBC' if i % 5 == 0 else 'TD' if i % 5 == 1 else 'BMO' if i % 5 == 2 else 'Scotia' if i % 5 == 3 else 'CIBC'}
            Transit Number: {(i%99000)+1000:05d}
            Account Number: {i:012d}
            Credit Card (Visa): 4{(i%1000):03d}-{(i%9000)+1000:04d}-{(i%9000)+1000:04d}-{(i%9000)+1000:04d}
            Credit Card (MC): 5{(i%1000):03d}-{(i%9000)+1000:04d}-{(i%9000)+1000:04d}-{(i%9000)+1000:04d}
            Credit Score: {600 + (i % 250)}
            Annual Income: ${25000 + (i * 500):,}
            
            Employment Information:
            Company: {'TechCorp Inc.' if i % 4 == 0 else 'DataSystems Ltd.' if i % 4 == 1 else 'Innovation Co.' if i % 4 == 2 else 'Solutions Inc.'}
            Employee ID: EMP-{i:08d}
            Department: {'Engineering' if i % 5 == 0 else 'Marketing' if i % 5 == 1 else 'Sales' if i % 5 == 2 else 'HR' if i % 5 == 3 else 'Finance'}
            Position: {'Senior' if i % 3 == 0 else 'Junior' if i % 3 == 1 else ''} {['Developer', 'Analyst', 'Manager', 'Coordinator', 'Specialist'][i%5]}
            Start Date: {(i%12)+1:02d}/{(i%28)+1:02d}/{2000+(i%24)}
            Salary: ${40000 + (i * 1000):,}
            Manager: Manager{(i//10)%100:02d}@company.com
            
            Medical Information:
            Medical Record: MRN-{i:010d}
            Blood Type: {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'][i%8]}
            Allergies: {'None' if i % 4 == 0 else 'Peanuts' if i % 4 == 1 else 'Shellfish' if i % 4 == 2 else 'Dairy'}
            Insurance Provider: {'Blue Cross' if i % 3 == 0 else 'Sun Life' if i % 3 == 1 else 'Manulife'}
            Policy Number: POL-{i:010d}
            
            Technology Information:
            Username: user{i:05d}
            Email Domain: @{'company' if i % 2 == 0 else 'organization'}{(i%50)+1:02d}.com
            Last Login: 2024-01-{(i%28)+1:02d} {(i%24):02d}:{(i%60):02d}:{(i%60):02d}
            IP Address: 192.168.{(i%255)+1}.{(i%255)+1}
            MAC Address: 00:1B:44:{i%256:02X}:{(i*7)%256:02X}:{(i*13)%256:02X}
            
            Emergency Contacts:
            Contact 1: Emergency{i:04d}Contact1 at (416) {i:03d}-{(i*19)%10000:04d}
            Contact 2: Emergency{i:04d}Contact2 at emergency{i:04d}@family.com
            
            Additional Notes:
            Record created: 2024-01-{(i%28)+1:02d}
            Last updated: 2024-01-{((i+7)%28)+1:02d}
            Verification status: {'Verified' if i % 3 == 0 else 'Pending' if i % 3 == 1 else 'Incomplete'}
            Risk level: {'Low' if i % 4 == 0 else 'Medium' if i % 4 <= 2 else 'High'}
            
            """
            very_large_doc += detailed_record
        (cls.perf_test_dir / "very_large_doc.txt").write_text(very_large_doc)
    
    def test_document_size_scaling(self):
        """Test performance scaling with different document sizes."""
        test_files = [
            ("small_doc.txt", "Small", 1000),      # < 1KB, expect < 1s
            ("medium_doc.txt", "Medium", 5000),    # ~10KB, expect < 5s
            ("large_doc.txt", "Large", 15000),     # ~100KB, expect < 15s
            ("very_large_doc.txt", "Very Large", 30000)  # ~1MB, expect < 30s
        ]
        
        size_performance = {}
        
        for filename, size_label, max_time_ms in test_files:
            file_path = str(self.perf_test_dir / filename)
            
            # Measure processing time
            start_time = time.time()
            result = self.pipeline.process_document(file_path)
            end_time = time.time()
            
            processing_time_ms = (end_time - start_time) * 1000
            
            # Validate result
            assert result is not None, f"Failed to process {size_label} document"
            assert len(result.pii_entities) > 0, f"No entities found in {size_label} document"
            
            # Performance assertion
            assert processing_time_ms < max_time_ms, f"{size_label} document took {processing_time_ms:.0f}ms, expected < {max_time_ms}ms"
            
            # Store metrics
            size_performance[size_label] = {
                'processing_time_ms': processing_time_ms,
                'entities_found': len(result.pii_entities),
                'file_size_bytes': Path(file_path).stat().st_size,
                'entities_per_second': len(result.pii_entities) / (processing_time_ms / 1000)
            }
        
        # Store performance metrics
        self.performance_metrics['document_scaling'] = size_performance
        
        # Validate scaling efficiency
        small_time = size_performance['Small']['processing_time_ms']
        large_time = size_performance['Large']['processing_time_ms']
        
        # Large document shouldn't take more than 50x the time of small document
        time_ratio = large_time / small_time if small_time > 0 else 0
        assert time_ratio < 50, f"Poor scaling: large doc takes {time_ratio:.1f}x longer than small doc"
    
    def test_throughput_measurement(self):
        """Test throughput with multiple documents."""
        test_file = str(self.perf_test_dir / "medium_doc.txt")
        
        # Process multiple times to measure throughput
        num_iterations = 10
        start_time = time.time()
        
        results = []
        for i in range(num_iterations):
            result = self.pipeline.process_document(test_file)
            results.append(result)
            assert result is not None
            assert len(result.pii_entities) > 0
        
        total_time = time.time() - start_time
        
        # Calculate throughput metrics
        throughput_docs_per_second = num_iterations / total_time
        avg_processing_time = total_time / num_iterations
        
        # Performance assertions
        assert throughput_docs_per_second > 0.5, f"Throughput too low: {throughput_docs_per_second:.2f} docs/sec"
        assert avg_processing_time < 10.0, f"Average processing time too high: {avg_processing_time:.2f}s"
        
        # Store metrics
        self.performance_metrics['throughput'].append({
            'docs_per_second': throughput_docs_per_second,
            'avg_processing_time': avg_processing_time,
            'total_time': total_time,
            'iterations': num_iterations
        })
    
    def test_memory_usage_profiling(self):
        """Test memory usage profiling during processing."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        test_file = str(self.perf_test_dir / "large_doc.txt")
        
        # Process document multiple times while monitoring memory
        memory_readings = []
        
        for i in range(5):
            before_memory = process.memory_info().rss / 1024 / 1024
            
            result = self.pipeline.process_document(test_file)
            assert result is not None
            
            after_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append({
                'iteration': i,
                'before_mb': before_memory,
                'after_mb': after_memory,
                'increase_mb': after_memory - before_memory
            })
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        assert total_memory_increase < 200, f"Memory usage increased too much: {total_memory_increase:.1f}MB"
        
        # Check for memory leaks (memory should stabilize)
        later_increases = [r['increase_mb'] for r in memory_readings[-3:]]
        avg_later_increase = sum(later_increases) / len(later_increases)
        
        assert avg_later_increase < 50, f"Possible memory leak: {avg_later_increase:.1f}MB average increase"
        
        # Store metrics
        self.performance_metrics['memory_usage'].extend(memory_readings)
    
    def test_cpu_usage_profiling(self):
        """Test CPU usage during processing."""
        import threading
        import time
        
        cpu_readings = []
        stop_monitoring = threading.Event()
        
        def monitor_cpu():
            while not stop_monitoring.is_set():
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_readings.append(cpu_percent)
                time.sleep(0.1)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Process document
        test_file = str(self.perf_test_dir / "large_doc.txt")
        start_time = time.time()
        
        result = self.pipeline.process_document(test_file)
        
        processing_time = time.time() - start_time
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_readings:
            avg_cpu = statistics.mean(cpu_readings)
            max_cpu = max(cpu_readings)
            
            # CPU usage assertions
            assert max_cpu < 90, f"CPU usage too high: {max_cpu}%"
            assert avg_cpu < 70, f"Average CPU usage too high: {avg_cpu}%"
            
            # Store metrics
            self.performance_metrics['cpu_usage'].append({
                'avg_cpu_percent': avg_cpu,
                'max_cpu_percent': max_cpu,
                'processing_time': processing_time,
                'readings_count': len(cpu_readings)
            })
        
        assert result is not None
        assert processing_time < 20.0, f"Processing took too long under CPU monitoring: {processing_time:.2f}s"
    
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent load."""
        test_file = str(self.perf_test_dir / "medium_doc.txt")
        num_threads = 4
        num_iterations_per_thread = 3
        
        def process_documents(thread_id):
            thread_results = []
            thread_times = []
            
            for i in range(num_iterations_per_thread):
                start_time = time.time()
                result = self.pipeline.process_document(test_file)
                end_time = time.time()
                
                thread_results.append(result)
                thread_times.append(end_time - start_time)
            
            return thread_id, thread_results, thread_times
        
        # Execute concurrent processing
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_documents, i) for i in range(num_threads)]
            
            all_results = []
            all_times = []
            
            for future in as_completed(futures):
                thread_id, results, times = future.result()
                all_results.extend(results)
                all_times.extend(times)
        
        total_concurrent_time = time.time() - start_time
        
        # Validate all results
        for result in all_results:
            assert result is not None
            assert len(result.pii_entities) > 0
        
        # Performance analysis
        total_operations = num_threads * num_iterations_per_thread
        avg_time_per_operation = statistics.mean(all_times)
        concurrent_throughput = total_operations / total_concurrent_time
        
        # Performance assertions
        assert avg_time_per_operation < 15.0, f"Average operation time too high: {avg_time_per_operation:.2f}s"
        assert concurrent_throughput > 0.2, f"Concurrent throughput too low: {concurrent_throughput:.2f} ops/sec"
        
        # Store metrics
        self.performance_metrics['concurrent_performance'] = {
            'num_threads': num_threads,
            'operations_per_thread': num_iterations_per_thread,
            'total_operations': total_operations,
            'total_time': total_concurrent_time,
            'avg_operation_time': avg_time_per_operation,
            'throughput_ops_per_sec': concurrent_throughput,
            'all_operation_times': all_times
        }
    
    def test_accuracy_vs_performance_tradeoff(self):
        """Test the tradeoff between accuracy and performance."""
        test_file = str(self.perf_test_dir / "medium_doc.txt")
        
        # Test with different confidence thresholds
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        tradeoff_metrics = []
        
        for confidence in confidence_levels:
            # Create config with specific confidence threshold
            config = PIIConfig()
            config.confidence_threshold = confidence
            config.use_local_storage = True
            
            pipeline = PIIExtractionPipeline(config)
            
            # Measure performance and accuracy
            start_time = time.time()
            result = pipeline.process_document(test_file)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            high_confidence_entities = [e for e in result.pii_entities if e.confidence >= 0.8]
            avg_confidence = sum(e.confidence for e in result.pii_entities) / len(result.pii_entities) if result.pii_entities else 0
            
            tradeoff_metrics.append({
                'confidence_threshold': confidence,
                'processing_time': processing_time,
                'total_entities': len(result.pii_entities),
                'high_confidence_entities': len(high_confidence_entities),
                'avg_confidence': avg_confidence,
                'entities_per_second': len(result.pii_entities) / processing_time if processing_time > 0 else 0
            })
        
        # Validate tradeoff expectations
        # Higher thresholds should generally mean fewer entities but faster processing
        low_threshold_metrics = tradeoff_metrics[0]  # 0.1 threshold
        high_threshold_metrics = tradeoff_metrics[-1]  # 0.9 threshold
        
        # Store comprehensive tradeoff metrics
        self.performance_metrics['accuracy_performance_tradeoff'] = tradeoff_metrics
        
        # Basic validation - system should work across all thresholds
        for metrics in tradeoff_metrics:
            assert metrics['processing_time'] < 30.0, f"Processing too slow at threshold {metrics['confidence_threshold']}"
            assert metrics['avg_confidence'] >= metrics['confidence_threshold'] * 0.8, "Average confidence below threshold"
    
    def test_large_batch_performance(self):
        """Test performance with large batch processing."""
        # Create multiple test files for batch processing
        batch_files = []
        for i in range(10):
            batch_file = self.perf_test_dir / f"batch_doc_{i}.txt"
            content = f"""
            Batch Document {i}
            Name: Employee{i:03d} Test{i:03d}
            Email: employee{i:03d}@batchtest.com
            Phone: (555) {i:03d}-{(i*7)%10000:04d}
            ID: BATCH-{i:06d}
            Address: {i+1} Batch Street
            """
            batch_file.write_text(content)
            batch_files.append(str(batch_file))
        
        # Process batch
        start_time = time.time()
        batch_results = self.pipeline.process_batch(batch_files)
        batch_time = time.time() - start_time
        
        # Validate batch results
        assert len(batch_results) == len(batch_files)
        
        total_entities = sum(len(result.pii_entities) for result in batch_results)
        batch_throughput = len(batch_files) / batch_time
        
        # Performance assertions
        assert batch_time < 60.0, f"Batch processing took too long: {batch_time:.2f}s"
        assert batch_throughput > 0.1, f"Batch throughput too low: {batch_throughput:.2f} docs/sec"
        assert total_entities > 20, f"Too few entities found in batch: {total_entities}"
        
        # Store batch performance metrics
        self.performance_metrics['batch_performance'] = {
            'num_documents': len(batch_files),
            'total_time': batch_time,
            'throughput_docs_per_sec': batch_throughput,
            'total_entities': total_entities,
            'avg_entities_per_doc': total_entities / len(batch_files),
            'avg_time_per_doc': batch_time / len(batch_files)
        }
    
    @pytest.mark.slow
    def test_stress_performance(self):
        """Stress test performance under extreme load."""
        test_file = str(self.perf_test_dir / "large_doc.txt")
        
        # Stress test parameters
        num_iterations = 50
        max_acceptable_time = 30.0  # seconds per iteration
        
        processing_times = []
        entity_counts = []
        
        start_total = time.time()
        
        for i in range(num_iterations):
            iteration_start = time.time()
            
            result = self.pipeline.process_document(test_file)
            
            iteration_time = time.time() - iteration_start
            processing_times.append(iteration_time)
            entity_counts.append(len(result.pii_entities))
            
            # Validate each iteration
            assert result is not None, f"Iteration {i} failed"
            assert len(result.pii_entities) > 0, f"No entities found in iteration {i}"
            assert iteration_time < max_acceptable_time, f"Iteration {i} too slow: {iteration_time:.2f}s"
        
        total_stress_time = time.time() - start_total
        
        # Performance analysis
        avg_time = statistics.mean(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        time_std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        
        avg_entities = statistics.mean(entity_counts)
        entity_std_dev = statistics.stdev(entity_counts) if len(entity_counts) > 1 else 0
        
        # Stress test assertions
        assert avg_time < 20.0, f"Average processing time too high under stress: {avg_time:.2f}s"
        assert time_std_dev < avg_time * 0.5, f"Processing time too inconsistent: {time_std_dev:.2f}s std dev"
        assert entity_std_dev < avg_entities * 0.2, f"Entity count too inconsistent: {entity_std_dev:.1f} std dev"
        
        # Store stress test metrics
        self.performance_metrics['stress_test'] = {
            'num_iterations': num_iterations,
            'total_time': total_stress_time,
            'avg_processing_time': avg_time,
            'max_processing_time': max_time,
            'min_processing_time': min_time,
            'time_std_dev': time_std_dev,
            'avg_entities': avg_entities,
            'entity_std_dev': entity_std_dev,
            'throughput_docs_per_sec': num_iterations / total_stress_time
        }
    
    def test_performance_regression_detection(self):
        """Test for performance regression detection."""
        test_file = str(self.perf_test_dir / "medium_doc.txt")
        
        # Baseline performance measurement
        baseline_times = []
        for _ in range(5):
            start_time = time.time()
            result = self.pipeline.process_document(test_file)
            baseline_times.append(time.time() - start_time)
            assert result is not None
        
        baseline_avg = statistics.mean(baseline_times)
        baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
        
        # Performance validation
        max_acceptable_time = baseline_avg + (3 * baseline_std)  # 3 sigma
        
        # Test additional iterations to detect regression
        regression_times = []
        for _ in range(10):
            start_time = time.time()
            result = self.pipeline.process_document(test_file)
            processing_time = time.time() - start_time
            regression_times.append(processing_time)
            
            # Each iteration should be within acceptable range
            assert processing_time < max_acceptable_time, f"Potential regression: {processing_time:.2f}s > {max_acceptable_time:.2f}s"
        
        regression_avg = statistics.mean(regression_times)
        
        # Regression detection
        performance_change = (regression_avg - baseline_avg) / baseline_avg * 100
        assert abs(performance_change) < 20, f"Performance changed by {performance_change:.1f}%"
        
        # Store regression metrics
        self.performance_metrics['regression_test'] = {
            'baseline_avg': baseline_avg,
            'baseline_std': baseline_std,
            'regression_avg': regression_avg,
            'performance_change_percent': performance_change,
            'max_acceptable_time': max_acceptable_time
        }
    
    @classmethod
    def teardown_class(cls):
        """Generate performance report and cleanup."""
        # Generate performance report
        report_file = cls.perf_test_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(cls.performance_metrics, f, indent=2, default=str)
        
        print(f"\n=== PERFORMANCE REPORT ===")
        print(f"Report saved to: {report_file}")
        
        # Print summary metrics
        if 'throughput' in cls.performance_metrics and cls.performance_metrics['throughput']:
            avg_throughput = statistics.mean([m['docs_per_second'] for m in cls.performance_metrics['throughput']])
            print(f"Average Throughput: {avg_throughput:.2f} docs/second")
        
        if 'document_scaling' in cls.performance_metrics:
            scaling = cls.performance_metrics['document_scaling']
            for size, metrics in scaling.items():
                print(f"{size} Document: {metrics['processing_time_ms']:.0f}ms, {metrics['entities_found']} entities")
        
        if 'stress_test' in cls.performance_metrics:
            stress = cls.performance_metrics['stress_test']
            print(f"Stress Test: {stress['avg_processing_time']:.2f}s avg, {stress['throughput_docs_per_sec']:.2f} docs/sec")
        
        # Cleanup
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])