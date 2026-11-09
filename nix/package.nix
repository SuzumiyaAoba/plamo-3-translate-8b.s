{
  lib,
  stdenv,
  python3,
  bash,
  cacert,
  shellcheck,
}:

stdenv.mkDerivation {
  pname = "plamo-3-translate";
  version = "0.3.0";

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
      ../docs/MODELS.md
      ../docs/benchmarks/multi-model.json
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
  postBuild = ''
    make CC="$CC" models
  '';

  checkPhase = ''
    runHook preCheck
    test ! -e models
    make CC="$CC" test test-tools
    sha256sum src/weight_schema.s src/model_layout.s src/rope_constants.s > build/generated.sha256
    python3 tools/weight_schema.py
    python3 tools/model_layout.py
    python3 tools/rope_constants.py
    sha256sum -c build/generated.sha256
    for variant in build build/0.8b build/2b build/31b; do
      test -z "$(nm -u "$variant/plamo")"
    done
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 build/plamo "$out/libexec/plamo"
    install -Dm755 build/plamo-worker "$out/libexec/plamo-worker"
    install -Dm755 build/plamo "$out/libexec/plamo-8b"
    install -Dm755 build/plamo-worker "$out/libexec/plamo-worker-8b"
    for model in 0.8b 2b 31b; do
      install -Dm755 "build/$model/plamo" "$out/libexec/plamo-$model"
      install -Dm755 "build/$model/plamo-worker" "$out/libexec/plamo-worker-$model"
    done
    install -Dm644 tools/download.py "$out/share/plamo/download.py"
    install -Dm644 tools/server.py "$out/share/plamo/server.py"
    install -Dm644 tools/model-manifest.json "$out/share/plamo/model-manifest.json"
    install -Dm644 tools/model_catalog.py "$out/share/plamo/model_catalog.py"
    cp -r tools/profiles "$out/share/plamo/profiles"
    mkdir -p "$out/bin"
    substitute nix/plamo.sh.in "$out/bin/plamo" \
      --subst-var-by shell '${bash}/bin/bash' \
      --subst-var-by engines "$out/libexec" \
      --subst-var-by modelProfiles "$(cat nix/model-profiles.sh)"
    substitute nix/plamo-download.sh.in "$out/bin/plamo-download" \
      --subst-var-by shell '${bash}/bin/bash' \
      --subst-var-by python '${python3}/bin/python3' \
      --subst-var-by script "$out/share/plamo/download.py" \
      --subst-var-by caBundle '${cacert}/etc/ssl/certs/ca-bundle.crt'
    substitute nix/plamo-server.sh.in "$out/bin/plamo-server" \
      --subst-var-by shell '${bash}/bin/bash' \
      --subst-var-by python '${python3}/bin/python3' \
      --subst-var-by script "$out/share/plamo/server.py" \
      --subst-var-by workers "$out/libexec"
    chmod +x "$out/bin/plamo" "$out/bin/plamo-download" "$out/bin/plamo-server"
    shellcheck "$out/bin/plamo" "$out/bin/plamo-download" "$out/bin/plamo-server"
    install -Dm644 LICENSE "$out/share/doc/plamo/LICENSE"
    install -Dm644 THIRD_PARTY_NOTICES.md "$out/share/doc/plamo/THIRD_PARTY_NOTICES.md"
    install -Dm644 docs/API.md "$out/share/doc/plamo/API.md"
    install -Dm644 docs/MODELS.md "$out/share/doc/plamo/MODELS.md"
    install -Dm644 docs/benchmarks/multi-model.json "$out/share/doc/plamo/benchmarks/multi-model.json"
    install -Dm644 LICENSES/PLaMo-Community.en.txt "$out/share/doc/plamo/LICENSES/PLaMo-Community.en.txt"
    install -Dm644 LICENSES/PLaMo-Community.ja.txt "$out/share/doc/plamo/LICENSES/PLaMo-Community.ja.txt"
    runHook postInstall
  '';

  meta = {
    description = "PLaMo 3 Translate 0.8B/2B/8B/31B in handwritten AArch64 assembly, without bundled weights";
    homepage = "https://github.com/SuzumiyaAoba/plamo-3-translate.s";
    license = lib.licenses.mit;
    platforms = [ "aarch64-darwin" ];
    mainProgram = "plamo";
  };
}
