(* Copyright (c) 2026 The Raven authors. ISC License. *)

(** CUDA queue encoding for the shared host submission compiler. *)

val encode : string -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [encode device u] replaces CUDA compute and copy submissions with ordinary
    C calls and linked argument storage. Returns [None] for unrelated nodes. *)

val lower : string -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [lower device u] replaces timeline polling with the CUDA fault-aware host
    helper. Returns [None] for unrelated nodes. *)
