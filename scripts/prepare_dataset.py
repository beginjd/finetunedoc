#!/usr/bin/env python3
"""
Convert extracted PDF text to instruction-following format for fine-tuning.
Creates JSONL dataset with instruction-input-output format.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random


class DatasetPreparer:
    """Prepare instruction-following dataset from extracted PDFs."""
    
    def __init__(self, extracted_dir: str = "data/extracted", output_file: str = "data/cobol_dataset.jsonl"):
        self.extracted_dir = Path(extracted_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Patterns for identifying different types of content
        self.syntax_patterns = [
            r'(\w+)\s+DIVISION',
            r'PERFORM\s+[A-Z\s]+',
            r'DATA\s+TYPE[:\s]+(\w+)',
            r'COMP-\d+',
            r'PIC\s+[X9S]+',
        ]
        
        self.code_patterns = [
            r'\d{6}\s+\w+.*',  # Line-numbered code
            r'IDENTIFICATION\s+DIVISION',
            r'PROCEDURE\s+DIVISION',
        ]
    
    def load_extracted_data(self) -> Dict:
        """Load all extracted PDF data."""
        extracted_data = {}
        
        for json_file in self.extracted_dir.glob("*_extracted.json"):
            if json_file.name == "extraction_summary.json":
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                extracted_data[json_file.stem.replace("_extracted", "")] = data
        
        return extracted_data
    
    def extract_syntax_examples(self, text: str, source: str) -> List[Dict]:
        """Extract syntax-related examples."""
        examples = []
        
        # Find syntax definitions
        syntax_section_pattern = r'(?:SYNTAX|Syntax|syntax)[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)'
        matches = re.finditer(syntax_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            syntax_text = match.group(1).strip()
            if len(syntax_text) > 50:  # Filter out too short matches
                examples.append({
                    "instruction": "What is the syntax for this COBOL construct?",
                    "input": syntax_text[:200],  # First 200 chars as context
                    "output": syntax_text,
                    "type": "syntax",
                    "source": source
                })
        
        return examples
    
    def extract_code_examples(self, text: str, source: str) -> List[Dict]:
        """Extract code examples."""
        examples = []
        
        # Find code blocks (usually indented or with line numbers)
        code_block_pattern = r'(?:EXAMPLE|Example|example|CODE|Code)[:\s]*\n((?:\s+\d{6}\s+.*\n?)+)'
        matches = re.finditer(code_block_pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            code = match.group(1).strip()
            if len(code) > 100:  # Filter out too short code
                # Extract what the code does from surrounding context
                context_start = max(0, match.start() - 200)
                context = text[context_start:match.start()].strip()
                
                examples.append({
                    "instruction": "Explain this COBOL code example",
                    "input": code,
                    "output": f"{context}\n\n{code}",
                    "type": "code_example",
                    "source": source
                })
        
        return examples
    
    def extract_qa_pairs(self, text: str, source: str) -> List[Dict]:
        """Extract question-answer pairs from documentation."""
        examples = []
        
        # Look for definition patterns
        definition_patterns = [
            r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an|the)\s+(.+?)(?=\.\s+[A-Z]|\.\n\n|$)',
            r'(\w+(?:\s+\w+)*)\s+means\s+(.+?)(?=\.\s+[A-Z]|\.\n\n|$)',
            r'(\w+(?:\s+\w+)*)\s+[:]\s+(.+?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                if len(definition) > 30 and len(definition) < 1000:
                    examples.append({
                        "instruction": f"What is {term} in COBOL?",
                        "input": "",
                        "output": f"{term} is {definition}",
                        "type": "definition",
                        "source": source
                    })
        
        return examples
    
    def extract_compiler_rules(self, text: str, source: str) -> List[Dict]:
        """Extract compiler rules and directives."""
        examples = []
        
        # Look for compiler directive patterns
        directive_patterns = [
            r'PROCESS\s+(.+?)(?=\n\n|\n[A-Z]|$)',
            r'COMPILER\s+OPTION[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)',
            r'DIRECTIVE[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in directive_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                directive_text = match.group(1).strip()
                if len(directive_text) > 50:
                    examples.append({
                        "instruction": "What are the compiler rules or directives for this?",
                        "input": directive_text[:200],
                        "output": directive_text,
                        "type": "compiler_rule",
                        "source": source
                    })
        
        return examples
    
    def create_instruction_examples(self, text: str, source: str) -> List[Dict]:
        """Create instruction-following examples from text chunks."""
        examples = []
        
        # Split text into meaningful chunks (by paragraphs or sections)
        chunks = re.split(r'\n\n+', text)
        
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 100 or len(chunk) > 2000:
                continue
            
            # Create different types of instructions for the same chunk
            instruction_templates = [
                ("Explain this COBOL concept", chunk),
                ("Summarize this COBOL documentation section", chunk),
                ("What does this COBOL documentation say about this topic?", chunk[:100]),
            ]
            
            for instruction, input_text in instruction_templates:
                examples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": chunk,
                    "type": "general",
                    "source": source
                })
        
        return examples
    
    def process_extracted_data(self, extracted_data: Dict) -> List[Dict]:
        """Process all extracted data into training examples."""
        all_examples = []
        
        for source_name, data in extracted_data.items():
            print(f"Processing {source_name}...")
            text = data.get("full_text", "")
            
            if not text:
                continue
            
            # Extract different types of examples
            examples = []
            examples.extend(self.extract_syntax_examples(text, source_name))
            examples.extend(self.extract_code_examples(text, source_name))
            examples.extend(self.extract_qa_pairs(text, source_name))
            examples.extend(self.extract_compiler_rules(text, source_name))
            examples.extend(self.create_instruction_examples(text, source_name))
            
            print(f"  Generated {len(examples)} examples from {source_name}")
            all_examples.extend(examples)
        
        return all_examples
    
    def format_for_training(self, examples: List[Dict]) -> List[Dict]:
        """Format examples for Mistral instruction format."""
        formatted = []
        
        for ex in examples:
            # Mistral Instruct format
            formatted_ex = {
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex["output"]
            }
            formatted.append(formatted_ex)
        
        return formatted
    
    def split_dataset(self, examples: List[Dict], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and validation sets."""
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        train = examples[:split_idx]
        val = examples[split_idx:]
        return train, val
    
    def save_jsonl(self, examples: List[Dict], filepath: Path):
        """Save examples to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    def prepare(self):
        """Main preparation method."""
        print("Loading extracted PDF data...")
        extracted_data = self.load_extracted_data()
        
        if not extracted_data:
            raise ValueError(f"No extracted data found in {self.extracted_dir}")
        
        print(f"Found {len(extracted_data)} extracted files")
        
        print("\nProcessing data into training examples...")
        examples = self.process_extracted_data(extracted_data)
        
        print(f"\nTotal examples generated: {len(examples)}")
        
        # Format for training
        formatted = self.format_for_training(examples)
        
        # Split into train/val
        train, val = self.split_dataset(formatted)
        
        print(f"Train examples: {len(train)}")
        print(f"Validation examples: {len(val)}")
        
        # Save datasets
        train_file = self.output_file.parent / f"{self.output_file.stem}_train.jsonl"
        val_file = self.output_file.parent / f"{self.output_file.stem}_val.jsonl"
        
        self.save_jsonl(train, train_file)
        self.save_jsonl(val, val_file)
        
        print(f"\nSaved training dataset to: {train_file}")
        print(f"Saved validation dataset to: {val_file}")
        
        # Save statistics
        stats = {
            "total_examples": len(examples),
            "train_examples": len(train),
            "val_examples": len(val),
            "sources": list(extracted_data.keys())
        }
        
        stats_file = self.output_file.parent / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics saved to: {stats_file}")
        
        return train, val


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent
    
    extracted_dir = docs_dir / "data" / "extracted"
    output_file = docs_dir / "data" / "cobol_dataset.jsonl"
    
    preparer = DatasetPreparer(extracted_dir, output_file)
    preparer.prepare()
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
