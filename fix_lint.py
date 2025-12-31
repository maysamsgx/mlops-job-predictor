import os

files_to_fix = [
    "train_pipeline.py", 
    "inference_service.py",
    "workflow.py",
    "src/config.py",
    "tests/smoke_test.py", 
    "tests/test_hashing.py", 
    "tests/test_integration.py", 
    "tests/test_pipeline.py"
]

for file_path in files_to_fix:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Strip trailing whitespace from each line
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Reconstruct content
        content = "\n".join(cleaned_lines)
        
        # Ensure single newline at end
        content = content.strip() + "\n"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Fixed {file_path}")
