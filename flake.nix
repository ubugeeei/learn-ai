{
  description = "From-scratch LLM and AI agent hands-on in Scala 3";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forEachSystem = function:
        nixpkgs.lib.genAttrs supportedSystems (system:
          function (import nixpkgs { inherit system; }));
    in
    {
      devShells = forEachSystem (pkgs:
        let
          jdk = pkgs.jdk21;
        in
        {
          default = pkgs.mkShell {
            packages = [
              jdk
              pkgs.sbt
            ];

            JAVA_HOME = jdk.home;

            shellHook = ''
              echo "learn-ai: Java $(java -version 2>&1 | head -n 1)"
              echo "learn-ai: run 'sbt check' to verify the workspace"
            '';
          };
        });
    };
}
