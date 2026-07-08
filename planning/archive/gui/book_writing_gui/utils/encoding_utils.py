import os
import logging
from pathlib import Path

def fix_file_encoding(file_path):
    """
    Fix file encoding issues by reading the file in binary mode,
    removing any null bytes, and saving it back as UTF-8.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if the file was fixed successfully, False otherwise
    """
    try:
        # Read the file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try to decode with different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        decoded_content = None
        
        for encoding in encodings:
            try:
                # Try to decode with the current encoding
                decoded_content = content.decode(encoding, errors='replace')
                logging.debug(f"Successfully decoded {file_path} with {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if decoded_content is None:
            # If all decodings fail, use latin-1 as a fallback
            decoded_content = content.decode('latin-1', errors='replace')
            logging.warning(f"Falling back to latin-1 with error replacement for {file_path}")
        
        # Remove any null bytes
        if '\x00' in decoded_content:
            decoded_content = decoded_content.replace('\x00', '')
            logging.info(f"Removed null bytes from {file_path}")
        
        # Write the file back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(decoded_content)
        
        logging.info(f"Successfully fixed encoding for {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error fixing encoding for {file_path}: {e}")
        return False

def ensure_template_encoding():
    """
    Ensure all template files have proper UTF-8 encoding without null bytes.
    This function scans the templates directory and fixes encoding issues.
    
    Returns:
        int: Number of files fixed
    """
    # Get the templates directory
    base_dir = Path(__file__).resolve().parent.parent
    templates_dir = base_dir / "templates"
    
    if not templates_dir.exists():
        logging.warning(f"Templates directory not found: {templates_dir}")
        return 0
    
    fixed_count = 0
    
    # Process all HTML files in the templates directory
    for template_file in templates_dir.glob("*.html"):
        if fix_file_encoding(template_file):
            fixed_count += 1
    
    return fixed_count

# Add this to make the function available for import
__all__ = ['fix_file_encoding', 'ensure_template_encoding']
