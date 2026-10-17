(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SafeTensors files of tensors, the implementation of
    {!Nx_io.load_safetensors} and {!Nx_io.save_safetensors} over the format
    of {!Safetensors}. *)

val load_safetensors : string -> Archive.t
val save_safetensors : ?overwrite:bool -> string -> Archive.t -> unit
