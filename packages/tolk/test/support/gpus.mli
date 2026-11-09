(** The GPUs the suites run on.

    Each GPU is opened once, at its first use. A test on one skips when the
    machine has none, or when its compiler's library is absent (NVRTC's where
    [NVRTC_PATH] is unset, comgr's where [COMGR_PATH] is), since the GPU then
    renders nothing. *)

val kinds : string list
(** [kinds] names the GPUs, in order: ["Metal"], ["CUDA"], ["NV"] (an NVIDIA
    GPU through its kernel driver) and ["AMD"]. *)

val gpu : string -> Rig.t
(** [gpu kind] is the first GPU of [kind], opened, or skips the running test
    ({!Windtrap.skip}) without one or without its compiler.

    Raises [Not_found] if [kind] is not one of {!kinds}. *)
