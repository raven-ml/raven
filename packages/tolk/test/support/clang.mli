(** Tests that compile with Clang. *)

val here : string -> unit
(** [here arch] skips the running test ({!Windtrap.skip}) where the host's
    programs cannot compile ({!Tolk_engine.compiles}), with the reason, as where
    Clang is missing or {!Tolk.Compiler_cpu.clang} refuses it. Where rig links
    no host code, it skips where {!Tolk.Compiler_cpu.clang}[ arch] does not
    compile an empty source instead. Where the host compiles, a test whose Clang
    lacks [arch]'s backend or processor fails, unless it skips on that itself. A
    test of the refusals themselves, under a stand-in Clang, must not call it.
*)
