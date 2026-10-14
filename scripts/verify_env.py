"""
Verifies local Python and Apple Silicon / MLX runtime environment.
"""

import platform


def check_env() -> bool:
    """Check and display local environment configuration.

    Returns:
        bool: True if environment inspection succeeded.
    """
    print(f"Python Version: {platform.python_version()}")
    print(f"OS: {platform.system()} ({platform.machine()})")

    is_mac = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_mac:
        print("[+] Detected Apple Silicon (M-series). MLX engine supported.")
        try:
            import mlx.core as mx

            print(f"[+] MLX installed successfully: {mx.__version__}")
        except ImportError:
            print(
                "[!] MLX not yet installed. Run: pip install -r "
                "requirements-mac.txt"
            )
    else:
        print("[i] Non-Mac or x86 system. HF/CUDA companion engine supported.")
    return True


if __name__ == "__main__":
    check_env()
