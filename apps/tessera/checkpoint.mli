(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A training run's state on disk.

    A run is a function of its key, its options and the stream of corpus
    positions it reads: every draw, dropout's masks included, is keyed by the
    run's key and a position. Its state after a step is thus its trained values,
    its optimizer's state, its key and the first position it has not read, and a
    run resumed from them with the same options takes the steps the run would
    have taken. *)

val save :
  string ->
  (module U : Nx.Ptree.S) ->
  (('v, 's, 'd) Nx.t, 'd) U.t -> key:Nx.host Nx.Rng.key -> position:int -> unit
(** [save file (module U) x ~key ~position] writes to the SafeTensors file
    [file] the state [x], [key] and [position]. After a crash, [file] holds the
    previous state or the new one ([Nx_io.save_safetensors]). *)

val load :
  string ->
  (module U : Nx.Ptree.S) ->
  like:(('v, 's, 'd) Nx.t, 'd) U.t ->
  (('v, 's, 'd) Nx.t, 'd) U.t * Nx.host Nx.Rng.key * int
(** [load file (module U) ~like] is the state, the key and the position {!save}
    wrote to [file], the state of [like]'s structure, each array placed where
    [like]'s lies.

    Raises [Failure] as [Nx_io.load_safetensors] and [Nx_io.Archive.to_value]
    do. *)

val fresh : ('s, 'd) Nx.Ptree.t -> 's -> 's
(** [fresh s x] is [x] with each array stored afresh, in C order on 16 bytes, as
    a compiled call stores its results: [Nx.copy] of each, by a [Rune.jit], so
    that it copies on a set without kernels too, as a GPU set is. A state
    {!load} reads lies where the file's records lie; stored afresh, it lies as a
    compiled step's results do. *)
