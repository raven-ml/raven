(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's files on the Hugging Face Hub.

    The transformer's weights and each codec's live in one Hub repository,
    {!repo}, a codec's under [codecs/NAME/]. Every file is read at one commit
    of it, {!revision}, through {!Kaun_hf}'s cache: the first read of a file
    downloads it, and later reads find it in the cache. aion-large's weights
    live in a repository of their own ({!weights}), beside the same codec
    files. *)

val repo : string
(** [repo] is ["polymathic-ai/aion-base"], AION-1's base model. *)

val revision : string
(** [revision] is the commit of {!repo} whose files are read. *)

val codec :
  ?cache_dir:string -> ?offline:bool -> string -> Nx.host Nx_io.Archive.t
(** [codec name] is the tensors of [codecs/name/model.safetensors] in {!repo}
    at {!revision}, fetched as {!Kaun_hf.download_file} fetches with
    [cache_dir] and [offline].

    Raises [Failure] if the file cannot be fetched or read. *)

(** The type for AION-1's released models. *)
type model =
  | Base  (** aion-base, 300M parameters, in {!repo}. *)
  | Large  (** aion-large, in polymathic-ai/aion-large. *)

val weights :
  ?cache_dir:string -> ?offline:bool -> model -> Nx.host Nx_io.Archive.t
(** [weights m] is the tensors of [m]'s file [model.safetensors], at the commit
    of its repository whose files are read, fetched as {!Kaun_hf.download_file}
    fetches with [cache_dir] and [offline].

    Raises [Failure] if the file cannot be fetched or read. *)
