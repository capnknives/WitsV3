import os
import logging

def fix_file_encoding(file_path):
    """
    Fix file encoding issues by reading the file in binary mode,
    removing any null bytes, and saving it back as UTF-8.
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
                print(f"Successfully decoded {file_path} with {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if decoded_content is None:
            # If all decodings fail, use latin-1 as a fallback
            decoded_content = content.decode('latin-1', errors='replace')
            print(f"Falling back to latin-1 with error replacement for {file_path}")
        
        # Remove any null bytes
        decoded_content = decoded_content.replace('\x00', '')
        
        # Write the file back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(decoded_content)
        
        print(f"Successfully fixed encoding for {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing encoding for {file_path}: {e}")
        return False

def main():
    # Get the templates directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(base_dir, "templates")
    
    if not os.path.exists(templates_dir):
        print(f"Templates directory not found: {templates_dir}")
        return 0
    
    # Fix encoding for index.html
    template_file = os.path.join(templates_dir, "index.html")
    if os.path.exists(template_file):
        fix_file_encoding(template_file)
    else:
        print(f"Template file not found: {template_file}")

if __name__ == "__main__":
    main()
