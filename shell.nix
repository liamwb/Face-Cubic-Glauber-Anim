{ pkgs ? import <nixpkgs> {} }: let
  libraryPath = with pkgs;
    lib.makeLibraryPath [
      # add other library packages here if needed
      stdenv.cc.cc
      stdenv.cc.libc
      glibc_multi
    ];
in
pkgs.mkShellNoCC {

  packages = with pkgs; [
    python3Packages.numpy
    python3Packages.matplotlib
    python3Packages.debugpy
    python3Packages.snakeviz  # profiler visualisation
  ];
  shellHook = ''
    # export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libraryPath}"
    fish
  '';
}

