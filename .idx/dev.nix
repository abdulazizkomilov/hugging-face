{ pkgs, ... }: {
  # Specify the Nixpkgs channel.
  channel = "stable-24.05"; # Use "unstable" if you need newer packages.

  # List of packages required for the environment.
  packages = [
    pkgs.python311        # Use Python 3.11.
    pkgs.gcc              # Include GCC for compiling C/C++ dependencies.
    pkgs.libffi           # Required for ctypes and FFI support.
    pkgs.ffmpeg           # For audio/video processing.
    pkgs.poetry           # Dependency management tool.
    pkgs.redis            # Redis for caching and task queues.
  ];

  idx = {
    # Extensions to install in your IDE (Visual Studio Code / IDX).
    extensions = [
      "ms-python.python"             # Python extension for VS Code.
      "rangav.vscode-thunder-client" # Thunder Client for API testing.
    ];

    workspace = {
      # Commands to run when the workspace is created for the first time.
      onCreate = {
        install = ''
          python -m venv .venv 
          source .venv/bin/activate 
          pip install -r requirements.txt
        '';
        # Default files to open in the editor.
        default.openFiles = [ "README.md" "src/index.html" "main.py" ];
      };

      # Commands to run every time the workspace starts or restarts.
      onStart = {
        run-server = "./devserver.sh"; # Run your development server script.
      };
    };
  };
}
