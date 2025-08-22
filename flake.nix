{
  description = "A flake for a FNN with backpropagation built from scratch in C and Python.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (system: f system);
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              pkgs.gcc

              pkgs.python3

              (pkgs.python3.withPackages (p: [
                p.numpy
                p.pandas
                p.scipy
              ]))

              # C/C++ build tools
              pkgs.cmake
              pkgs.gdb
            ];

            shellHook = ''
              # Informative message for the user
              echo "Welcome to the FNN development shell!"
              echo "-------------------------------------"
              echo "Available tools:"
              echo "  - gcc (C compiler)"
              echo "  - python3 (with numpy, pandas, scipy, matplotlib)"
              echo "  - cmake"
              echo "  - gdb (debugger)"
              echo ""
              echo "To run the Python code: 'python3 your_script.py'"
              echo "To build the C code: 'cmake . && make'"
              echo "-------------------------------------"
            '';
          };
        });
    };
}
