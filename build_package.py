import subprocess
import sys

def build_container():
    """
    Build & Package Stage: Packages model and serving code into a container.
    """
    image_name = "job-role-predictor:latest"
    print(f"Building Docker image: {image_name}...")
    
    try:
        # Build the container
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "."],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Docker build successful.")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Docker command not found. Please ensure Docker is installed and in PATH.")
        sys.exit(1)

if __name__ == "__main__":
    build_container()
