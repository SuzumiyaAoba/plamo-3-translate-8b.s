{
  lib,
  stdenv,
  python3,
  bash,
  cacert,
  shellcheck,
}:

stdenv.mkDerivation {
  pname = "plamo-3-translate-8b";
  version = "0.2.0";

  # Explicit inputs keep model files and local build artifacts out of the store,
  # including when the flake is evaluated from a non-Git directory.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Makefile
      ../src
      ../runtime
      ../tests
      ../tools
      ../nix
      ../docs/config.json
      ../docs/API.md
      ../README.md
      ../LICENSE
      ../LICENSES
      ../THIRD_PARTY_NOTICES.md
    ];
  };

  nativeBuildInputs = [
    python3
    shellcheck
  ];
  dontConfigure = true;
  makeFlags = [ "CC=${stdenv.cc.targetPrefix}cc" ];
  doCheck = true;
  __darwinAllowLocalNetworking = true; # Model-free HTTP tests use loopback only.

  checkPhase = ''
    runHook preCheck
    test ! -e models
    make CC="$CC" test test-tools
    sha256sum src/weight_schema.s src/model_layout.s src/rope_constants.s > build/generated.sha256
    python3 tools/weight_schema.py
    python3 tools/model_layout.py
    python3 tools/rope_constants.py
    sha256sum -c build/generated.sha256
    test -z "$(nm -u build/plamo)"
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 build/plamo "$out/libexec/plamo"
    install -Dm755 build/plamo-worker "$out/libexec/plamo-worker"
    install -Dm644 tools/download.py "$out/share/plamo/download.py"
    install -Dm644 tools/server.py "$out/share/plamo/server.py"
    install -Dm644 tools/model-manifest.json "$out/share/plamo/model-manifest.json"
    mkdir -p "$out/bin"
    substitute nix/plamo.sh.in "$out/bin/plamo" \
      --subst-var-by shell '${bash}/bin/bash' \
      --subst-var-by engine "$out/libexec/plamo"
    substitute nix/plamo-download.sh.in "$out/bin/plamo-download" \
      --subst-var-by shell '${bash}/bin/bash' \
      --subst-var-by python '${python3}/bin/python3' \
      --subst-var-by script "$out/share/plamo/download.py" \
      --subst-var-by caBundle '${cacert}/etc/ssl/certs/ca-bundle.crt'
    substitute nix/plamo-server.sh.in "$out/bin/plamo-server" \
      --subst-var-by shell '${bash}/bin/bash' \
      --subst-var-by python '${python3}/bin/python3' \
      --subst-var-by script "$out/share/plamo/server.py" \
      --subst-var-by worker "$out/libexec/plamo-worker"
    chmod +x "$out/bin/plamo" "$out/bin/plamo-download" "$out/bin/plamo-server"
    shellcheck "$out/bin/plamo" "$out/bin/plamo-download" "$out/bin/plamo-server"
    install -Dm644 LICENSE "$out/share/doc/plamo/LICENSE"
    install -Dm644 THIRD_PARTY_NOTICES.md "$out/share/doc/plamo/THIRD_PARTY_NOTICES.md"
    install -Dm644 docs/API.md "$out/share/doc/plamo/API.md"
    install -Dm644 LICENSES/PLaMo-Community.en.txt "$out/share/doc/plamo/LICENSES/PLaMo-Community.en.txt"
    install -Dm644 LICENSES/PLaMo-Community.ja.txt "$out/share/doc/plamo/LICENSES/PLaMo-Community.ja.txt"
    runHook postInstall
  '';

  meta = {
    description = "PLaMo 3 Translate 8B in handwritten AArch64 assembly, without bundled weights";
    homepage = "https://github.com/SuzumiyaAoba/plamo-3-translate-8b.s";
    license = lib.licenses.mit;
    platforms = [ "aarch64-darwin" ];
    mainProgram = "plamo";
  };
}
