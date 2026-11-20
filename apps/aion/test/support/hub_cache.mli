(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Hub caches planted with test files. *)

val plant :
  ?repo:string ->
  ?revision:string ->
  string ->
  string ->
  Nx.host Nx_io.Archive.t ->
  unit
(** [plant ~repo ~revision dir file a] saves [a] as [repo]'s [file] at
    [revision] in the Hub cache [dir], where [Kaun_hf] looks for it, making the
    directories it needs. [repo] and [revision] default to aion-base's:
    {!Aion.Hub.repo} at {!Aion.Hub.revision}. *)
