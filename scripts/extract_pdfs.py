#!/usr/bin/env python3
"""
Extract text from COBOL documentation PDFs.
Preserves structure (chapters, sections, code blocks) and handles tables.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import pdfplumber


class PDFExtractor:
    """Extract structured text from COBOL documentation PDFs."""
    
    def __init__(self, pdf_path: str, output_dir: str = "data/extracted"):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_with_pymupdf(self) -> List[Dict]:
        """Extract text using PyMuPDF (better for general text)."""
        doc = fitz.open(self.pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            
            # Try to identify structure
            blocks = page.get_text("blocks")
            
            page_data = {
                "page": page_num,
                "text": text,
                "blocks": blocks,
                "metadata": {
                    "width": page.rect.width,
                    "height": page.rect.height
                }
            }
            pages.append(page_data)
        
        doc.close()
        return pages
    
    def extract_with_pdfplumber(self) -> List[Dict]:
        """Extract text using pdfplumber (better for tables)."""
        pages = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                tables = page.extract_tables()
                
                page_data = {
                    "page": page_num,
                    "text": text or "",
                    "tables": tables or [],
                    "metadata": {
                        "width": page.width,
                        "height": page.height
                    }
                }
                pages.append(page_data)
        
        return pages
    
    def extract_combined(self) -> List[Dict]:
        """Combine both extraction methods for best results."""
        pymupdf_pages = self.extract_with_pymupdf()
        pdfplumber_pages = self.extract_with_pdfplumber()
        
        combined = []
        for pymu, plumb in zip(pymupdf_pages, pdfplumber_pages):
            combined_page = {
                "page": pymu["page"],
                "text": pymu["text"] or plumb["text"] or "",
                "tables": plumb.get("tables", []),
                "blocks": pymu.get("blocks", []),
                "metadata": pymu["metadata"]
            }
            combined.append(combined_page)
        
        return combined
    
    def identify_structure(self, text: str) -> Dict:
        """Identify document structure (chapters, sections, code blocks)."""
        structure = {
            "chapters": [],
            "sections": [],
            "code_blocks": [],
            "tables": []
        }
        
        # Identify chapter headings (common patterns)
        chapter_patterns = [
            r'^Chapter\s+(\d+)[:\.]\s*(.+)$',
            r'^CHAPTER\s+(\d+)[:\.]\s*(.+)$',
            r'^(\d+)\s+([A-Z][^\n]+)$',  # Numbered chapters
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for chapter headings
            for pattern in chapter_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    structure["chapters"].append({
                        "number": match.group(1),
                        "title": match.group(2).strip(),
                        "line": i
                    })
                    break
            
            # Check for section headings (usually all caps or numbered)
            if re.match(r'^[A-Z][A-Z\s]{10,}$', line) or re.match(r'^\d+\.\d+\s+[A-Z]', line):
                structure["sections"].append({
                    "title": line,
                    "line": i
                })
            
            # Identify code blocks (common COBOL patterns)
            if re.match(r'^\s+\d{6}\s+', line):  # Line numbers
                structure["code_blocks"].append({
                    "line": i,
                    "content": line
                })
        
        return structure
    
    def extract(self) -> Dict:
        """Main extraction method."""
        print(f"Extracting from: {self.pdf_path}")
        
        # Extract pages
        pages = self.extract_combined()
        
        # Combine all text
        full_text = "\n\n".join([p["text"] for p in pages])
        
        # Identify structure
        structure = self.identify_structure(full_text)
        
        # Extract tables
        all_tables = []
        for page in pages:
            for table in page.get("tables", []):
                if table:
                    all_tables.append({
                        "page": page["page"],
                        "data": table
                    })
        
        result = {
            "source_file": str(self.pdf_path.name),
            "total_pages": len(pages),
            "full_text": full_text,
            "pages": pages,
            "structure": structure,
            "tables": all_tables
        }
        
        # Save extracted data
        output_file = self.output_dir / f"{self.pdf_path.stem}_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted {len(pages)} pages")
        print(f"Found {len(structure['chapters'])} chapters, {len(structure['sections'])} sections")
        print(f"Found {len(all_tables)} tables")
        print(f"Saved to: {output_file}")
        
        return result


def main():
    """Extract text from all COBOL PDFs."""
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent
    output_dir = docs_dir / "data" / "extracted"
    
    pdf_files = [
        docs_dir / "cics-api-reference.pdf",
        docs_dir / "lrmvs.pdf",
        docs_dir / "pgmvs.pdf"
    ]
    
    all_extractions = {}
    
    for pdf_file in pdf_files:
        if not pdf_file.exists():
            print(f"Warning: {pdf_file} not found, skipping...")
            continue
        
        extractor = PDFExtractor(pdf_file, output_dir)
        result = extractor.extract()
        all_extractions[pdf_file.stem] = result
    
    # Save combined extraction summary
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_files": len(all_extractions),
            "files": list(all_extractions.keys()),
            "extraction_date": str(Path(__file__).stat().st_mtime)
        }, f, indent=2)
    
    print(f"\nExtraction complete! Summary saved to: {summary_file}")
    return all_extractions


if __name__ == "__main__":
    main()
