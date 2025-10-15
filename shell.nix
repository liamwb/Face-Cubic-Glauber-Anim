{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.11") {} }:

pkgs.mkShellNoCC {
  packages = with pkgs; [
    python3Packages.numpy
    python3Packages.matplotlib
  ];
  shellHook = "fish";
}

