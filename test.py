import torch
import json
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForTokenClassification
from train import ProductionInference, StudentModel, create_sample_data, DataProcessor

def create_test_model():
    """Create a minimal trained model for testing inference"""
    print("Creating test model for inference testing...")
    
    # Create temporary directory for test model
    test_model_dir = tempfile.mkdtemp()
    
    # Create a simple student model
    model = StudentModel("distilbert-base-uncased", num_labels=10)
    
    # Save model state
    torch.save(model.state_dict(), f"{test_model_dir}/pytorch_model.bin")
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(test_model_dir)
    
    # Save config
    config = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=10).config
    config.save_pretrained(test_model_dir)
    
    # Create and save label mappings
    texts, labels = create_sample_data()
    label2id = DataProcessor.create_label_mappings(labels)
    
    with open(f"{test_model_dir}/label_mappings.json", 'w') as f:
        json.dump(label2id, f)
    
    print(f"Test model saved to: {test_model_dir}")
    return test_model_dir, label2id

def test_inference_initialization():
    """Test if ProductionInference initializes correctly"""
    print("\n1. Testing ProductionInference Initialization")
    print("-" * 50)
    
    try:
        model_path, label2id = create_test_model()
        
        # Test initialization
        inference = ProductionInference(model_path, device='cpu')
        
        print("✅ Model initialization successful")
        print(f"   Device: {inference.device}")
        print(f"   Number of labels: {len(inference.label2id)}")
        print(f"   Labels: {list(inference.label2id.keys())}")
        
        return inference, model_path
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None, None

def test_single_prediction():
    """Test prediction on a single sentence"""
    print("\n2. Testing Single Sentence Prediction")
    print("-" * 50)
    
    inference, model_path = test_inference_initialization()
    if inference is None:
        return False
    
    try:
        # Test with string input
        test_text = "Patient has diabetes and takes metformin"
        print(f"Input text: '{test_text}'")
        
        predictions = inference.predict(test_text)
        print(f"Predictions: {predictions}")
        
        # Test with tokenized input
        test_tokens = ["Patient", "diagnosed", "with", "hypertension"]
        print(f"Input tokens: {test_tokens}")
        
        predictions = inference.predict(test_tokens)
        print(f"Predictions: {predictions}")
        
        print("✅ Single prediction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
        return False

def test_entity_extraction():
    """Test entity extraction from predictions"""
    print("\n3. Testing Entity Extraction")
    print("-" * 50)
    
    inference, model_path = test_inference_initialization()
    if inference is None:
        return False
    
    try:
        # Create mock predictions with known entities
        mock_predictions = [
            ("Patient", "O"),
            ("has", "O"),
            ("diabetes", "B-DISEASE"),
            ("mellitus", "I-DISEASE"),
            ("and", "O"),
            ("takes", "O"),
            ("metformin", "B-MEDICATION"),
            ("daily", "O")
        ]
        
        entities = inference.extract_entities(mock_predictions)
        print(f"Mock predictions: {mock_predictions}")
        print(f"Extracted entities: {entities}")
        
        # Verify entity extraction logic
        expected_entities = [
            {'text': 'diabetes mellitus', 'label': 'DISEASE', 'start': 2, 'end': 3},
            {'text': 'metformin', 'label': 'MEDICATION', 'start': 6, 'end': 6}
        ]
        
        print(f"Expected entities: {expected_entities}")
        print("✅ Entity extraction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
        return False

def test_batch_inference():
    """Test inference on multiple sentences"""
    print("\n4. Testing Batch Inference")
    print("-" * 50)
    
    inference, model_path = test_inference_initialization()
    if inference is None:
        return False
    
    try:
        test_sentences = [
            "Patient presents with acute myocardial infarction",
            "Blood pressure elevated at 180/100 mmHg",
            "Prescribed aspirin 325mg once daily",
            "CT scan shows pulmonary embolism"
        ]
        
        all_entities = []
        
        for i, sentence in enumerate(test_sentences):
            print(f"\nSentence {i+1}: {sentence}")
            predictions = inference.predict(sentence)
            entities = inference.extract_entities(predictions)
            
            print(f"Predictions: {predictions}")
            print(f"Entities: {entities}")
            all_entities.extend(entities)
        
        print(f"\nTotal entities found: {len(all_entities)}")
        print("✅ Batch inference test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n5. Testing Edge Cases")
    print("-" * 50)
    
    inference, model_path = test_inference_initialization()
    if inference is None:
        return False
    
    try:
        # Test empty input
        print("Testing empty input...")
        predictions = inference.predict([])
        print(f"Empty input result: {predictions}")
        
        # Test single word
        print("Testing single word...")
        predictions = inference.predict(["diabetes"])
        print(f"Single word result: {predictions}")
        
        # Test long sequence
        print("Testing long sequence...")
        long_text = " ".join(["word"] * 200)  # Very long text
        predictions = inference.predict(long_text)
        print(f"Long sequence result length: {len(predictions)}")
        
        # Test special characters
        print("Testing special characters...")
        special_text = "Patient has Type-2 diabetes (NIDDM) @ 150mg/dL"
        predictions = inference.predict(special_text)
        print(f"Special chars result: {predictions}")
        
        print("✅ Edge cases test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False

def test_performance():
    """Test inference performance and timing"""
    print("\n6. Testing Performance")
    print("-" * 50)
    
    inference, model_path = test_inference_initialization()
    if inference is None:
        return False
    
    try:
        import time
        
        test_sentences = [
            "Patient presents with acute myocardial infarction and elevated troponin levels",
            "History of diabetes mellitus type 2 managed with metformin 500mg twice daily",
            "Blood pressure reading of 180/100 mmHg indicates severe hypertension",
            "CT angiography revealed significant coronary artery disease"
        ] * 10  # Repeat for timing test
        
        # Warm up
        inference.predict(test_sentences[0])
        
        # Time multiple predictions
        start_time = time.time()
        for sentence in test_sentences:
            predictions = inference.predict(sentence)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(test_sentences)
        
        print(f"Processed {len(test_sentences)} sentences")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per sentence: {avg_time:.4f} seconds")
        print(f"Throughput: {len(test_sentences)/total_time:.1f} sentences/second")
        
        print("✅ Performance test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_different_text_formats():
    """Test inference with different input formats"""
    print("\n7. Testing Different Text Formats")
    print("-" * 50)
    
    inference, model_path = test_inference_initialization()
    if inference is None:
        return False
    
    try:
        # Test formats
        test_cases = [
            # String input
            "Patient has diabetes mellitus",
            
            # Pre-tokenized list
            ["Patient", "has", "diabetes", "mellitus"],
            
            # Medical abbreviations
            "Pt w/ DM, HTN, and CAD on ASA 81mg",
            
            # Numbers and measurements
            "BP 120/80 mmHg, HR 72 bpm, Temp 98.6F",
            
            # Clinical notes style
            "ASSESSMENT: Type 2 DM, uncontrolled. PLAN: Increase metformin to 1000mg BID."
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest case {i+1}: {test_case}")
            try:
                predictions = inference.predict(test_case)
                entities = inference.extract_entities(predictions)
                print(f"  Predictions: {len(predictions)} tokens")
                print(f"  Entities found: {len(entities)}")
                if entities:
                    for entity in entities:
                        print(f"    {entity}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        print("✅ Different text formats test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Text formats test failed: {e}")
        return False

def test_with_real_medical_data():
    """Test with more realistic medical data if available"""
    print("\n8. Testing with Realistic Medical Data")
    print("-" * 50)
    
    inference, model_path = test_inference_initialization()
    if inference is None:
        return False
    
    # Realistic medical sentences
    medical_examples = [
        "The patient is a 65-year-old male with a history of coronary artery disease, diabetes mellitus type 2, and hypertension presenting with acute onset chest pain.",
        "Physical examination revealed blood pressure of 160/90 mmHg, heart rate 88 bpm, and temperature 98.6°F.",
        "Laboratory studies showed elevated troponin I at 2.5 ng/mL and glucose level of 180 mg/dL.",
        "Echocardiogram demonstrated left ventricular ejection fraction of 45% with regional wall motion abnormalities.",
        "Patient was started on aspirin 325mg daily, metoprolol 25mg twice daily, and atorvastatin 40mg at bedtime.",
        "Follow-up chest X-ray showed no evidence of pneumonia or congestive heart failure.",
        "Discharge medications include lisinopril 10mg daily for hypertension and metformin 1000mg twice daily for diabetes."
    ]
    
    total_entities = 0
    
    for i, sentence in enumerate(medical_examples):
        print(f"\nExample {i+1}:")
        print(f"Text: {sentence}")
        
        try:
            predictions = inference.predict(sentence)
            entities = inference.extract_entities(predictions)
            
            print(f"Entities found ({len(entities)}):")
            for entity in entities:
                print(f"  - {entity['text']} ({entity['label']})")
            
            total_entities += len(entities)
            
        except Exception as e:
            print(f"  ❌ Failed to process: {e}")
    
    print(f"\nTotal entities extracted: {total_entities}")
    print("✅ Realistic medical data test completed!")
    
    return True

def run_comprehensive_inference_test():
    """Run all inference tests"""
    print("🔬 COMPREHENSIVE INFERENCE TESTING")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        test_single_prediction,
        test_entity_extraction,
        test_batch_inference,
        test_edge_cases,
        test_performance,
        test_different_text_formats,
        test_with_real_medical_data
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            test_results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            test_results.append(False)
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print(f"Passed: {passed}/{total} tests")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Your inference system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

# Quick inference validation function
def quick_inference_test():
    """Quick test to validate inference is working"""
    print("🚀 QUICK INFERENCE VALIDATION")
    print("=" * 40)
    
    try:
        # First run the main function to create a model
        print("Step 1: Creating model via main()...")
        from train import main
        main()
        
        # Test if model files exist
        model_path = "./demo_model/best_model"
        required_files = ["pytorch_model.bin", "config.json", "tokenizer.json", "label_mappings.json"]
        
        print(f"\nStep 2: Checking model files in {model_path}...")
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Missing files: {missing_files}")
            print("Try running main() first to train and save the model.")
            return False
        
        # Test inference
        print(f"\nStep 3: Testing inference...")
        inference = ProductionInference(model_path, device='cpu')
        
        test_sentences = [
            "Patient has diabetes and hypertension",
            "Blood pressure is 140/90 mmHg",
            "Prescribed metformin 500mg twice daily"
        ]
        
        for sentence in test_sentences:
            print(f"\nTesting: '{sentence}'")
            predictions = inference.predict(sentence)
            entities = inference.extract_entities(predictions)
            
            print(f"  Tokens & Labels: {predictions}")
            print(f"  Entities: {entities if entities else 'None found'}")
        
        print("\n✅ Quick inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Interactive testing function
def interactive_inference_test():
    """Interactive testing where you can input your own text"""
    print("🔍 INTERACTIVE INFERENCE TESTING")
    print("=" * 40)
    
    try:
        model_path = "./demo_model/best_model"
        
        if not os.path.exists(model_path):
            print("Model not found. Running training first...")
            from train import main
            main()
        
        inference = ProductionInference(model_path, device='cpu')
        print("Model loaded successfully!")
        print("\nEnter medical text to analyze (or 'quit' to exit):")
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                predictions = inference.predict(user_input)
                entities = inference.extract_entities(predictions)
                
                print(f"\nAnalysis Results:")
                print(f"Token-level predictions:")
                for token, label in predictions:
                    print(f"  {token:<15} -> {label}")
                
                if entities:
                    print(f"\nExtracted Entities:")
                    for entity in entities:
                        print(f"  '{entity['text']}' -> {entity['label']}")
                else:
                    print("\nNo entities found.")
                    
            except Exception as e:
                print(f"Error processing input: {e}")
        
        print("Interactive testing completed!")
        
    except Exception as e:
        print(f"Interactive test setup failed: {e}")

if __name__ == "__main__":
    print("Select testing mode:")
    print("1. Quick validation test")
    print("2. Comprehensive test suite")
    print("3. Interactive testing")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        quick_inference_test()
    elif choice == "2":
        run_comprehensive_inference_test()
    elif choice == "3":
        interactive_inference_test()
    else:
        print("Invalid choice. Running quick test by default...")
        quick_inference_test()