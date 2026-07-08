import os
import shutil
import sys

def remove_null_bytes(directory):
    """
    Scan all Python files in the directory and its subdirectories,
    remove null bytes, and save the cleaned files.
    """
    fixed_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Skip this script itself
                if file == 'fix_null_bytes.py':
                    continue
                
                try:
                    # Read the file in binary mode to detect null bytes
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    # Check if the file contains null bytes
                    if b'\x00' in content:
                        print(f"Found null bytes in {file_path}")
                        
                        # Create a backup
                        backup_path = file_path + '.bak'
                        shutil.copy2(file_path, backup_path)
                        print(f"Created backup at {backup_path}")
                        
                        # Remove null bytes
                        cleaned_content = content.replace(b'\x00', b'')
                        
                        # Write the cleaned content back to the file
                        with open(file_path, 'wb') as f:
                            f.write(cleaned_content)
                        
                        print(f"Fixed {file_path}")
                        fixed_files.append(file_path)
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return fixed_files

if __name__ == "__main__":
    # Use the current directory if no directory is specified
    directory = '.'
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    print(f"Scanning directory: {os.path.abspath(directory)}")
    fixed_files = remove_null_bytes(directory)
    
    if fixed_files:
        print("\nFixed the following files:")
        for file in fixed_files:
            print(f"- {file}")
        print(f"\nTotal files fixed: {len(fixed_files)}")
    else:
        print("\nNo files with null bytes found.")
