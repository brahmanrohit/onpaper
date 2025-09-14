"""
Simple Script to Process Your Collected Papers
Run this after collecting 100-500 papers
"""

import os
import json
from paper_processing_guide import PaperProcessor

def process_collected_papers():
    """Process your collected papers step by step"""
    
    print("📚 PAPER PROCESSING WORKFLOW")
    print("=" * 50)
    
    # Initialize processor
    processor = PaperProcessor()
    
    # Step 1: Create directory structure
    print("\n1️⃣ Creating directory structure...")
    processor.create_directory_structure()
    
    # Step 2: Check for raw papers
    raw_papers_dir = "processed_papers/raw_papers"
    if not os.path.exists(raw_papers_dir):
        print(f"\n❌ Directory '{raw_papers_dir}' not found!")
        print("Please create this directory and place your collected papers there.")
        return
    
    # Count papers in raw directory
    paper_files = [f for f in os.listdir(raw_papers_dir) if f.endswith(('.txt', '.md', '.pdf'))]
    
    if not paper_files:
        print(f"\n❌ No paper files found in '{raw_papers_dir}'!")
        print("Please add your collected papers to this directory.")
        print("Supported formats: .txt, .md, .pdf")
        return
    
    print(f"\n✅ Found {len(paper_files)} paper files")
    
    # Step 3: Process papers
    print("\n2️⃣ Processing papers...")
    processed_papers = []
    
    for i, filename in enumerate(paper_files, 1):
        print(f"Processing paper {i}/{len(paper_files)}: {filename}")
        
        try:
            # Read paper content
            file_path = os.path.join(raw_papers_dir, filename)
            
            if filename.endswith('.pdf'):
                # For PDF files, you might need additional processing
                print(f"⚠️ PDF file detected: {filename}")
                print("Please convert PDF to text format first, or use a PDF reader library.")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            
            # Process the paper
            processed_paper = processor.process_single_paper(paper_content, filename)
            processed_papers.append(processed_paper)
            
            print(f"   ✅ Classified as: {processed_paper['paper_type']} ({processed_paper['field']})")
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            continue
    
    if not processed_papers:
        print("\n❌ No papers were successfully processed!")
        return
    
    # Step 4: Create training dataset
    print(f"\n3️⃣ Creating training dataset from {len(processed_papers)} papers...")
    dataset = processor.create_training_dataset(processed_papers)
    
    # Step 5: Save processed data
    print("\n4️⃣ Saving processed data...")
    processor.save_processed_data(dataset, "training_dataset.json")
    
    # Step 6: Generate plagiarism test cases
    print("\n5️⃣ Generating plagiarism test cases...")
    plagiarism_cases = processor.generate_plagiarism_test_cases(processed_papers)
    
    # Save plagiarism test cases
    plagiarism_data = {
        "test_cases": plagiarism_cases,
        "statistics": {
            "total_cases": len(plagiarism_cases),
            "case_types": {
                "exact_copy": len([c for c in plagiarism_cases if c["variation_type"] == "exact_copy"]),
                "paraphrased": len([c for c in plagiarism_cases if c["variation_type"] == "paraphrased"]),
                "similar_topic": len([c for c in plagiarism_cases if c["variation_type"] == "similar_topic"]),
                "different_topic": len([c for c in plagiarism_cases if c["variation_type"] == "different_topic"])
            }
        }
    }
    
    plagiarism_path = os.path.join(processor.output_dir, "training_data", "plagiarism_test_cases.json")
    with open(plagiarism_path, 'w', encoding='utf-8') as f:
        json.dump(plagiarism_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved plagiarism test cases to: {plagiarism_path}")
    
    # Step 7: Display summary
    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
    
    print(f"\n📄 Paper Statistics:")
    print(f"   Total papers processed: {dataset['statistics']['total_papers']}")
    print(f"   Average word count: {dataset['statistics']['avg_word_count']:.0f}")
    
    print(f"\n📋 Paper Types:")
    for paper_type, count in dataset['statistics']['paper_types'].items():
        percentage = (count / dataset['statistics']['total_papers']) * 100
        print(f"   {paper_type}: {count} papers ({percentage:.1f}%)")
    
    print(f"\n🎓 Academic Fields:")
    for field, count in dataset['statistics']['fields'].items():
        percentage = (count / dataset['statistics']['total_papers']) * 100
        print(f"   {field}: {count} papers ({percentage:.1f}%)")
    
    print(f"\n🔍 Plagiarism Test Cases:")
    print(f"   Total test cases: {plagiarism_data['statistics']['total_cases']}")
    for case_type, count in plagiarism_data['statistics']['case_types'].items():
        print(f"   {case_type}: {count} cases")
    
    # Step 8: Next steps
    print("\n" + "=" * 50)
    print("🚀 NEXT STEPS")
    print("=" * 50)
    
    print("\n1. Review the classifications:")
    print("   - Check if paper types are correctly identified")
    print("   - Verify academic field classifications")
    print("   - Look for any obvious errors")
    
    print("\n2. Validate the data:")
    print("   - Open 'processed_papers/training_data/training_dataset.json'")
    print("   - Review a few sample papers")
    print("   - Make corrections if needed")
    
    print("\n3. Prepare for ML training:")
    print("   - The processed data is ready for model training")
    print("   - Use the enhanced ML models we created")
    print("   - Run the training scripts")
    
    print("\n4. Monitor performance:")
    print("   - Test the models on validation data")
    print("   - Track accuracy improvements")
    print("   - Adjust parameters as needed")
    
    print(f"\n✅ Processing complete! Your papers are ready for ML training.")

def create_sample_paper():
    """Create a sample paper for testing"""
    sample_paper = """
Title: Machine Learning Applications in Healthcare: A Comprehensive Review

Abstract: This paper presents a systematic review of machine learning applications in healthcare, focusing on recent developments in diagnostic systems, treatment planning, and patient outcome prediction. We analyze 150 studies published between 2018-2023, examining the effectiveness of various ML algorithms in clinical settings. Our findings demonstrate significant improvements in diagnostic accuracy, with deep learning models achieving 85% accuracy in medical image analysis. The review also identifies key challenges in implementation, including data privacy concerns and regulatory compliance issues.

Keywords: machine learning, healthcare, medical diagnosis, deep learning, clinical applications

Introduction: The integration of artificial intelligence and machine learning in healthcare has revolutionized medical practice, offering new opportunities for improved patient care and clinical decision-making. This systematic review examines the current state of ML applications in healthcare, providing insights into both successes and challenges.

Methodology: We conducted a comprehensive literature search using PubMed, IEEE, and ACM databases, focusing on peer-reviewed studies published in the last five years. Inclusion criteria required studies to present quantitative results and involve real clinical data. Data extraction focused on algorithm performance, clinical outcomes, and implementation challenges.

Results: Our analysis of 150 studies revealed that machine learning algorithms have been successfully applied across multiple healthcare domains. Deep learning models showed particular promise in medical imaging, achieving average accuracy rates of 85% across various diagnostic tasks. However, implementation challenges included data quality issues, regulatory barriers, and resistance from healthcare professionals.

Discussion: The findings suggest that while ML holds great promise for healthcare, successful implementation requires addressing technical, regulatory, and human factors. Future research should focus on developing more robust algorithms, improving data quality, and creating frameworks for ethical AI deployment in clinical settings.

Conclusion: Machine learning applications in healthcare show significant potential for improving patient outcomes and clinical efficiency. However, realizing this potential requires addressing current challenges in data quality, regulatory compliance, and clinical adoption.
"""
    
    # Create sample paper file
    os.makedirs("processed_papers/raw_papers", exist_ok=True)
    
    with open("processed_papers/raw_papers/sample_healthcare_paper.txt", "w", encoding="utf-8") as f:
        f.write(sample_paper)
    
    print("✅ Created sample paper: processed_papers/raw_papers/sample_healthcare_paper.txt")
    print("You can use this as a template for formatting your collected papers.")

if __name__ == "__main__":
    print("📚 Academic Paper Processing Tool")
    print("=" * 50)
    
    # Check if user wants to create a sample paper
    response = input("\nDo you want to create a sample paper for testing? (y/n): ").lower()
    
    if response == 'y':
        create_sample_paper()
        print("\nNow you can:")
        print("1. Add your collected papers to 'processed_papers/raw_papers/'")
        print("2. Run this script again to process them")
    
    # Check if there are papers to process
    if os.path.exists("processed_papers/raw_papers"):
        paper_files = [f for f in os.listdir("processed_papers/raw_papers") if f.endswith(('.txt', '.md'))]
        
        if paper_files:
            print(f"\nFound {len(paper_files)} papers to process.")
            response = input("Do you want to process them now? (y/n): ").lower()
            
            if response == 'y':
                process_collected_papers()
        else:
            print("\nNo papers found in 'processed_papers/raw_papers/'")
            print("Please add your collected papers there first.")
    else:
        print("\nDirectory 'processed_papers/raw_papers/' not found.")
        print("The script will create it when you run the processing.") 