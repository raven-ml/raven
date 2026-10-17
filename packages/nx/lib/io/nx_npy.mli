(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NPY and NPZ files of tensors, the implementation of {!Nx_io.load_npy},
    {!Nx_io.save_npy}, {!Nx_io.load_npz}, {!Nx_io.load_npz_entry} and
    {!Nx_io.save_npz}.

    Each function returns its errors as {!Error.t}, where {!Nx_io} raises
    them. A malformed file is a [Format_error] and a failed system call an
    [Io_error]. Every other exception is caught and returned as [Other]
    with its text. *)

val load_npy : string -> (Nx.packed, Error.t) result

val save_npy :
  ?overwrite:bool -> string -> ('a, 'b) Nx.t -> (unit, Error.t) result

val load_npz : string -> (Archive.t, Error.t) result

val load_npz_entry : name:string -> string -> (Nx.packed, Error.t) result
(** [load_npz_entry ~name path] is {!Nx_io.load_npz_entry}[ ~name path]. A
    missing entry is [Missing_entry name]. *)

val save_npz :
  ?overwrite:bool -> string -> Archive.t -> (unit, Error.t) result
