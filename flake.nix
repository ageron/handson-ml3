{
  description = "Hands-on Machine Learning 3rd Edition - Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python310
            pkgs.micromamba
          ];

          shellHook = ''
            echo "===================================================="
            echo "Hands-on Machine Learning 3rd Edition Environment"
            echo "===================================================="
            echo ""

            # Set up micromamba
            export MAMBA_ROOT_PREFIX="$HOME/.micromamba"
            eval "$(micromamba shell hook --shell bash)"

            # Check if homl3 environment exists
            if ! micromamba env list | grep -q "homl3"; then
              echo "Creating homl3 environment from environment.yml..."
              micromamba env create -f environment.yml -y
              echo "Environment created successfully!"
              echo ""
            else
              echo "homl3 environment already exists."
              echo ""
            fi

            # Activate the environment
            echo "Activating homl3 environment..."
            micromamba activate homl3

            # Install IPython kernel if not already installed
            if ! jupyter kernelspec list | grep -q "python3"; then
              echo ""
              echo "Installing IPython kernel..."
              python -m ipykernel install --user --name=python3
              echo "IPython kernel installed!"
            fi

            echo ""
            echo "===================================================="
            echo "Environment activated!"
            echo "Python version: $(python --version)"
            echo ""
            echo "Ready to start Jupyter:"
            echo "  jupyter notebook"
            echo "===================================================="
          '';
        };
      }
    );
}
