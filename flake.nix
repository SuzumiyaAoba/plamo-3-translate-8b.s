{
  description = "Handwritten AArch64 PLaMo 3 Translate inference for Apple Silicon";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-26.05-darwin";

  outputs =
    { nixpkgs, ... }:
    let
      system = "aarch64-darwin";
      pkgs = import nixpkgs { inherit system; };
      plamo = pkgs.callPackage ./nix/package.nix { };
    in
    {
      packages.${system} = {
        default = plamo;
        inherit plamo;
      };

      apps.${system} = {
        default = {
          type = "app";
          program = "${plamo}/bin/plamo";
          meta.description = "Translate with separately downloaded original BF16 weights";
        };
        download-model = {
          type = "app";
          program = "${plamo}/bin/plamo-download";
          meta.description = "Download and verify original model files outside the Nix store";
        };
        serve = {
          type = "app";
          program = "${plamo}/bin/plamo-server";
          meta.description = "Run a local HTTP API with one persistent model worker";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        inputsFrom = [ plamo ];
        packages = [
          pkgs.python3
          pkgs.git
        ];
        shellHook = ''
          export PLAMO_DATA_DIR="''${PLAMO_DATA_DIR:-$PWD}"
        '';
      };

      checks.${system}.model-free = plamo;
      formatter.${system} = pkgs.nixfmt;
    };
}
