(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type tensor_meta = { encoded_path : string; dtype : string; shape : int array }

val save : base_path:string -> Snapshot.t -> unit
(** Persist a snapshot to disk using the provided [base_path]. This writes four
    files: [<base>.structure.json], [<base>.scalars.json],
    [<base>.tensors.safetensors], and [<base>.tensors.json]. *)

val load : base_path:string -> (Snapshot.t, string) result
(** Load a snapshot previously stored with {!save}. *)
