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
  buildInputs = with pkgs; [
      qt5.full
      qtcreator
    ];

  packages = with pkgs; [
    python3Packages.numpy
    python3Packages.matplotlib
    python3Packages.debugpy
    python3Packages.snakeviz  # profiler visualisation

    python3Packages.vispy
    python3Packages.pyqt5


    ## https://stackoverflow.com/questions/79082181/unable-to-use-pyside6-on-nixos-qt-qpa-plugin-from-6-5-0-xcb-cursor0-or-libxc
    # To make vispy work
    xorg.libxcb
    xorg.xcbutilwm
    xorg.xcbutilimage
    xorg.xcbutilkeysyms
    xorg.xcbutilrenderutil
    xcb-util-cursor
  ];
  shellHook = ''
    # export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libraryPath}"
    export QT_QPA_PLATFORM=xcb
    export QT_DEBUG_PLUGINS=1
    fish
  '';
}

