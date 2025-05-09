let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
        gmp
    ];
    nativeBuildInputs = with pkgs; [
        pkg-config
    ];
    gmp = pkgs.gmp;
  }

