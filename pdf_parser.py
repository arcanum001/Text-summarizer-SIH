import fitz
import re
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)

def is_block_a_heading(block: Dict, last_font_size: float, max_heading_word_count: int) -> bool:
    """Determine if a text block is likely a heading based on formatting and content"""
    if "lines" not in block or not block.get('lines'): 
        return False
    
    # Extract text from block
    block_text = " ".join(
        span['text'] for line in block['lines'] 
        for span in line.get('spans', [])
    ).strip()
    
    if not block_text or len(block_text.split()) > max_heading_word_count or block_text.endswith('.'):
        return False
    
    # Check formatting characteristics
    first_span = block['lines'][0]['spans'][0] if block['lines'][0].get('spans') else {}
    font_size = first_span.get('size', 10)
    is_bold = bool(first_span.get('flags', 0) & 16)
    
    # Consider it a heading if it's bold or significantly larger font
    if is_bold or font_size > last_font_size + 1: 
        return True
    
    return False

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common PDF artifacts
    cleaned = re.sub(r'[^\w\s\.,;:!?()-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def process_pdf_to_chunks(pdf_path: str, min_chunk_word_count: int, max_heading_word_count: int) -> List[Dict]:
    """Process a PDF file and extract text chunks with metadata"""
    doc_name = os.path.basename(pdf_path)
    logger.info(f"Processing PDF: {doc_name}")
    chunks = []
    
    try:
        doc = fitz.open(pdf_path)
        current_section_title = doc_name.replace('.pdf', '')
        last_font_size = 10.0
        
        for page_num, page in enumerate(doc):
            try:
                blocks = page.get_text("dict").get("blocks", [])
                
                for block in blocks:
                    if "lines" not in block or not block.get('lines'): 
                        continue
                    
                    # Extract text from block
                    block_text = " ".join(
                        span['text'] for line in block['lines'] 
                        for span in line.get('spans', [])
                    ).strip()
                    
                    cleaned_text = clean_text(block_text)
                    if not cleaned_text: 
                        continue
                    
                    # Check if this block is a heading
                    if is_block_a_heading(block, last_font_size, max_heading_word_count):
                        current_section_title = cleaned_text
                        logger.debug(f"Found heading: {current_section_title}")
                    else:
                        # Process as content if it meets minimum word count
                        if len(cleaned_text.split()) >= min_chunk_word_count:
                            chunk = {
                                'text': cleaned_text,
                                'metadata': { 
                                    'doc_name': doc_name, 
                                    'page_number': page_num + 1, 
                                    'section_title': current_section_title 
                                }
                            }
                            chunks.append(chunk)
                    
                    # Update font size tracking
                    if block['lines'][0].get('spans'):
                        last_font_size = block['lines'][0]['spans'][0].get('size', last_font_size)
                        
            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1} in {doc_name}: {e}")
                continue
                
        doc.close()
        
    except Exception as e:
        logger.error(f"Failed to process {doc_name}: {e}")
        return []
    
    logger.info(f"Extracted {len(chunks)} chunks from {doc_name}")
    return chunks