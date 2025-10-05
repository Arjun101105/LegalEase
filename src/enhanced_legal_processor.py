#!/usr/bin/env python3
"""
Enhanced Legal Document Processor for LegalEase
Handles different sections of legal documents with appropriate processing
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedLegalDocumentProcessor:
    """Enhanced processor for complete legal documents"""
    
    def __init__(self):
        self.section_patterns = {
            'case_header': [
                r'IN THE.*?COURT OF.*?(?:\n|$)',
                r'CASE NO\..*?(?:\n|$)',
                r'CIVIL WRIT PETITION NO\..*?(?:\n|$)',
                r'CRIMINAL APPEAL NO\..*?(?:\n|$)',
                r'SLP.*?NO\..*?(?:\n|$)'
            ],
            'parties': [
                r'(.+?)\s+\.{3,}\s+(?:Petitioner|Appellant|Plaintiff)',
                r'(.+?)\s+[Vv]ersus\s+(.+?)\s+\.{3,}\s+(?:Respondent|Defendant)',
                r'BETWEEN:?\s*(.+?)\s+\.{3,}\s+(?:Petitioner|Appellant)',
                r'AND:?\s*(.+?)\s+\.{3,}\s+(?:Respondent|Defendant)'
            ],
            'legal_content': [
                r'WHEREAS.*?(?=\n\n|\nAND WHEREAS|\nNOW THEREFORE|$)',
                r'The .+? filed .+? petition .+?[.!?]',
                r'[Tt]he [Cc]ourt .+?[.!?]',
                r'[Ii]t is .+? that .+?[.!?]',
                r'[Tt]he [Ll]earned .+?[.!?]'
            ],
            'orders': [
                r'ORDERED:.*?(?=\n\n|$)',
                r'IT IS ORDERED.*?(?=\n\n|$)',
                r'The Court .+?(?:directs?|orders?|holds?) .+?[.!?]',
                r'Accordingly.+?[.!?]'
            ],
            'procedural': [
                r'Having heard .+?[.!?]',
                r'After hearing .+?[.!?]',
                r'On .+? date .+?[.!?]',
                r'List .+?(?:after|on) .+?[.!?]'
            ]
        }
        
        # Different simplification strategies for each section type
        self.section_strategies = {
            'case_header': 'preserve',     # Keep as-is for reference
            'parties': 'simplify_names',   # Simplify roles but keep names
            'legal_content': 'full_simplify',  # Full legal simplification
            'orders': 'moderate_simplify',  # Moderate simplification
            'procedural': 'minimal_simplify'  # Light touch
        }
    
    def analyze_document_structure(self, text: str) -> Dict[str, List[str]]:
        """Analyze and categorize different sections of the legal document"""
        sections = {
            'case_header': [],
            'parties': [],
            'legal_content': [],
            'orders': [],
            'procedural': [],
            'other': []
        }
        
        # Split text into paragraphs for analysis
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            categorized = False
            
            # Try to categorize each paragraph
            for section_type, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, paragraph, re.IGNORECASE | re.MULTILINE):
                        sections[section_type].append(paragraph)
                        categorized = True
                        break
                if categorized:
                    break
            
            # If not categorized, put in 'other' or try to guess based on content
            if not categorized:
                if self._is_likely_legal_content(paragraph):
                    sections['legal_content'].append(paragraph)
                else:
                    sections['other'].append(paragraph)
        
        return sections
    
    def _is_likely_legal_content(self, text: str) -> bool:
        """Determine if text contains legal content that needs simplification"""
        legal_indicators = [
            'petition', 'mandamus', 'writ', 'article', 'constitution',
            'appellant', 'respondent', 'plaintiff', 'defendant',
            'whereas', 'therefore', 'pursuant', 'statutory',
            'provisions', 'obligations', 'compliance'
        ]
        
        text_lower = text.lower()
        legal_count = sum(1 for indicator in legal_indicators if indicator in text_lower)
        
        # If contains 2+ legal terms and is substantial, likely legal content
        return legal_count >= 2 and len(text.split()) > 10
    
    def process_legal_document(self, text: str, simplifier) -> Dict[str, any]:
        """Process a complete legal document with section-aware simplification"""
        # Analyze document structure
        sections = self.analyze_document_structure(text)
        
        processed_sections = {}
        summary = {
            'total_sections': sum(len(v) for v in sections.values()),
            'section_counts': {k: len(v) for k, v in sections.items()},
            'processing_strategy': {}
        }
        
        # Process each section type differently
        for section_type, content_list in sections.items():
            if not content_list:
                continue
                
            strategy = self.section_strategies.get(section_type, 'full_simplify')
            summary['processing_strategy'][section_type] = strategy
            
            processed_content = []
            
            for content in content_list:
                if strategy == 'preserve':
                    # Keep case headers as-is for reference
                    processed_content.append({
                        'original': content,
                        'processed': content,
                        'processing': 'preserved'
                    })
                
                elif strategy == 'simplify_names':
                    # Simplify legal roles but keep names
                    simplified = self._simplify_party_info(content)
                    processed_content.append({
                        'original': content,
                        'processed': simplified,
                        'processing': 'names_simplified'
                    })
                
                elif strategy == 'full_simplify':
                    # Full legal text simplification
                    simplified = simplifier.simplify_text(content)
                    processed_content.append({
                        'original': content,
                        'processed': simplified,
                        'processing': 'fully_simplified'
                    })
                
                elif strategy == 'moderate_simplify':
                    # Moderate simplification for orders
                    simplified = self._moderate_simplify(content, simplifier)
                    processed_content.append({
                        'original': content,
                        'processed': simplified,
                        'processing': 'moderately_simplified'
                    })
                
                else:  # minimal_simplify
                    # Light touch for procedural content
                    simplified = self._minimal_simplify(content)
                    processed_content.append({
                        'original': content,
                        'processed': simplified,
                        'processing': 'minimally_simplified'
                    })
            
            processed_sections[section_type] = processed_content
        
        return {
            'sections': processed_sections,
            'summary': summary,
            'readable_summary': self._create_readable_summary(processed_sections)
        }
    
    def _simplify_party_info(self, text: str) -> str:
        """Simplify party information while preserving names"""
        # Simple replacements for legal roles
        replacements = {
            r'\.{3,}\s*Petitioner': ' (person filing the case)',
            r'\.{3,}\s*Appellant': ' (person appealing)',
            r'\.{3,}\s*Respondent': ' (other party)',
            r'\.{3,}\s*Defendant': ' (person being sued)',
            r'[Vv]ersus': 'vs',
            r'[Bb]etween:?': 'Between',
            r'[Aa]nd:?': 'and'
        }
        
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)
        
        return result.strip()
    
    def _moderate_simplify(self, text: str, simplifier) -> str:
        """Apply moderate simplification - simpler than full but more than minimal"""
        # For orders and court directions, use targeted simplification
        if len(text.split()) < 50:  # Short text, use manual simplification
            return self._minimal_simplify(text)
        else:  # Longer text, use AI simplification
            return simplifier.enhanced_manual_simplification(text)
    
    def _minimal_simplify(self, text: str) -> str:
        """Apply minimal simplification for procedural content"""
        # Just replace a few common legal terms but keep structure
        simple_replacements = {
            'Having heard': 'After hearing',
            'learned counsel': 'lawyers',
            'perused': 'reviewed',
            'List after': 'Schedule for',
            'matter': 'case'
        }
        
        result = text
        for legal_term, simple_term in simple_replacements.items():
            result = re.sub(legal_term, simple_term, result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def _create_readable_summary(self, processed_sections: Dict) -> str:
        """Create a readable summary of the document"""
        summary_parts = []
        
        # Extract case information
        if 'case_header' in processed_sections and processed_sections['case_header']:
            summary_parts.append("📋 CASE INFORMATION:")
            for item in processed_sections['case_header'][:2]:  # Show first 2 headers
                summary_parts.append(f"   {item['processed']}")
        
        # Extract parties
        if 'parties' in processed_sections and processed_sections['parties']:
            summary_parts.append("\n👥 PARTIES:")
            for item in processed_sections['parties']:
                summary_parts.append(f"   {item['processed']}")
        
        # Main legal content summary
        if 'legal_content' in processed_sections and processed_sections['legal_content']:
            summary_parts.append("\n⚖️  MAIN LEGAL CONTENT:")
            for i, item in enumerate(processed_sections['legal_content'][:3]):  # Show first 3
                summary_parts.append(f"   {i+1}. {item['processed'][:200]}...")
        
        # Court orders
        if 'orders' in processed_sections and processed_sections['orders']:
            summary_parts.append("\n📜 COURT ORDERS:")
            for item in processed_sections['orders']:
                summary_parts.append(f"   • {item['processed']}")
        
        return '\n'.join(summary_parts)
    
    def save_processed_document(self, processed_doc: Dict, output_path: Path) -> None:
        """Save the processed document in multiple formats"""
        # Save detailed JSON
        json_path = output_path / "detailed_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_doc, f, indent=2, ensure_ascii=False)
        
        # Save readable summary
        summary_path = output_path / "readable_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(processed_doc['readable_summary'])
        
        # Save simplified content only
        simplified_path = output_path / "simplified_content.txt"
        with open(simplified_path, 'w', encoding='utf-8') as f:
            f.write("SIMPLIFIED LEGAL DOCUMENT\n")
            f.write("=" * 40 + "\n\n")
            
            for section_type, items in processed_doc['sections'].items():
                if items:
                    f.write(f"{section_type.upper().replace('_', ' ')}:\n")
                    f.write("-" * 30 + "\n")
                    for item in items:
                        f.write(f"{item['processed']}\n\n")
        
        print(f"💾 Saved processed document to:")
        print(f"   📄 Readable summary: {summary_path}")
        print(f"   📊 Detailed analysis: {json_path}")
        print(f"   ✨ Simplified content: {simplified_path}")