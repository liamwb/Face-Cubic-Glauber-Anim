{
  pkgs ? import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/80c24eeb9ff46aa99617844d0c4168659e35175f.tar.gz";
    sha256 = "sha256:0a2cws2hhdi4j5ipn1rsk5k7b6vw1l29ibb020bvgl9mza51psgl";
  }) { },
}:
let
  libraryPath =
    with pkgs;
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
    python3Packages.snakeviz # profiler visualisation
  ];
  shellHook = ''
    # export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libraryPath}"
    fish
  '';
}
