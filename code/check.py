import importlib

# List of required packages and their versions
required_packages = {
    "absl-py": "2.0.0", "addict": "2.4.0", "aiohttp": "3.9.3", "aiosignal": "1.3.1", "anyio": "4.3.0",
    "arrow": "1.3.0", "astor": "0.8.1", "astunparse": "1.6.3", "async-timeout": "4.0.3", "attrs": "23.2.0",
    "basicsr": "1.4.2", "beautifulsoup4": "4.12.3", "blessed": "1.20.0", "boto3": "1.34.49",
    "botocore": "1.34.49", "cached-property": "1.5.2", "cachetools": "5.3.2", "certifi": "2022.12.7",
    "charset-normalizer": "2.1.1", "chex": "0.1.7", "click": "8.1.7", "clu": "0.0.11",
    "contextlib2": "21.6.0", "contourpy": "1.1.1", "croniter": "1.3.15", "cycler": "0.12.1",
    "dateutils": "0.6.12", "deepdiff": "6.7.1", "dm-tree": "0.1.8", "easydict": "1.11",
    "editor": "1.6.6", "einops": "0.7.0", "etils": "1.3.0", "exceptiongroup": "1.2.0",
    "fastapi": "0.88.0", "filelock": "3.9.0", "flatbuffers": "24.3.6", "flax": "0.7.2",
    "fonttools": "4.47.0", "frozenlist": "1.4.1", "fsspec": "2023.12.2", "future": "1.0.0",
    "fvcore": "0.1.5.post20221221", "gast": "0.3.3", "gdown": "5.1.0", "gitdb": "4.0.11",
    "GitPython": "3.1.42", "google-auth": "2.26.1", "google-auth-oauthlib": "1.0.0",
    "google-pasta": "0.2.0", "grpcio": "1.60.0", "h11": "0.14.0", "h5py": "2.10.0",
    "huggingface-hub": "0.20.3", "idna": "3.4", "imageio": "2.33.1", "importlib-metadata": "7.0.1",
    "importlib-resources": "6.1.1", "inquirer": "3.2.4", "iopath": "0.1.10", "itsdangerous": "2.1.2",
    "jax": "0.4.13", "jaxlib": "0.4.13", "Jinja2": "3.1.2", "jmespath": "1.0.1",
    "joblib": "1.3.2", "keras": "2.13.1", "kiwisolver": "1.4.5", "kornia": "0.7.1",
    "lazy_loader": "0.3", "libclang": "16.0.6", "lightning": "1.9.0", "llvmlite": "0.41.1",
    "lmdb": "1.4.1", "Markdown": "3.5.1", "matplotlib": "3.7.4", "ml-collections": "0.1.0",
    "numpy": "1.22.4", "opencv-python": "4.6.0.66", "optax": "0.1.8", "pandas": "2.0.3",
    "protobuf": "3.20.1", "py-cpuinfo": "9.0.0", "pydantic": "1.10.14", "scikit-learn": "1.3.2",
    "scipy": "1.10.1", "seaborn": "0.13.2", "tensorboard": "2.0.0", "torch": "2.2.1",
    "torchvision": "0.17.1", "torchaudio": "2.2.1", "tqdm": "4.64.1", "ultralytics": "8.1.26"
}

# Check installed packages
missing_packages = []
mismatched_versions = []

for package, required_version in required_packages.items():
    try:
        # Try importing the package
        module = importlib.import_module(package.replace("-", "_"))
        # Get the installed version
        installed_version = getattr(module, "__version__", None)
        if installed_version != required_version:
            mismatched_versions.append((package, installed_version, required_version))
    except ImportError:
        # If the package is not found, add it to missing list
        missing_packages.append(package)

# Display results
if missing_packages:
    print("Missing Packages:")
    for pkg in missing_packages:
        print(f" - {pkg}")

if mismatched_versions:
    print("\nMismatched Versions:")
    for pkg, installed, required in mismatched_versions:
        print(f" - {pkg}: Installed: {installed}, Required: {required}")

if not missing_packages and not mismatched_versions:
    print("All required packages are correctly installed!")
