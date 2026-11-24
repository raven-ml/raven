open Tolk

let here arch =
  match Tolk_engine.compiles Rig.host with
  | Ok () -> ()
  | Error reason when Rig.Image.links Rig.host -> Windtrap.skip ~reason ()
  | Error _ -> (
      (* Where rig links no host code, nothing answers for the host: Clang is
         asked for [arch] itself. *)
      match Renderer.Compiler.compile (Compiler_cpu.clang arch) "" with
      | (_ : string) -> ()
      | exception Renderer.Compiler.Compile_error reason ->
          Windtrap.skip ~reason ())
