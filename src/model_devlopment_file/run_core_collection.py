#!/usr/bin/env python3
"""
Automated CORE API Paper Collection
Uses the config file to automatically collect papers
"""

import json
import sys
import os
from core_api_collector import CoreAPICollector

def main():
    print("🚀 Starting CORE API Paper Collection...")
    print("="*50)
    
    # Load config
    try:
        with open('core_api_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Config file not found. Please run setup_core_api_training.py first.")
        return
    
    api_key = config['core_api_key']
    topics = config['topics']
    papers_per_topic = config['papers_per_topic']
    
    print(f"📚 API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"🎯 Topics: {len(topics)}")
    print(f"📄 Papers per topic: {papers_per_topic}")
    print(f"📊 Expected total: {len(topics) * papers_per_topic} papers")
    print("="*50)
    
    # Initialize collector
    collector = CoreAPICollector(api_key)
    
    # Start collection
    print("\n📚 Collecting papers...")
    papers = collector.collect_papers(topics, papers_per_topic)
    
    if papers:
        collector.save_training_data(papers)
        print(f"\n✅ SUCCESS! Collected {len(papers)} papers!")
        print("📁 Data saved to: processed_papers/training_data/core_training/")
        
        # Show statistics
        paper_types = {}
        fields = {}
        for paper in papers:
            paper_type = paper.get('paper_type', 'unknown')
            field = paper.get('field', 'unknown')
            paper_types[paper_type] = paper_types.get(paper_type, 0) + 1
            fields[field] = fields.get(field, 0) + 1
        
        print(f"\n📊 COLLECTION STATISTICS:")
        print(f"Paper Types: {paper_types}")
        print(f"Academic Fields: {fields}")
        
        return True
    else:
        print("❌ No papers collected!")
        return False

if __name__ == "__main__":
    main() 