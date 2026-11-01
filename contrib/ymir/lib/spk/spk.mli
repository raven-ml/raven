(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SPK files: the implementation of {!Ymir_spk.Spk}. *)

type segment = {
  name : string;
  target : int;
  center : int;
  frame : int;
  kind : int;
  start : (Ymir.Time.tdb, Nx.host) Ymir.Time.t;
  stop : (Ymir.Time.tdb, Nx.host) Ymir.Time.t;
}

type 'd t

val read : at:'d Nx.Placement.t -> string -> ('d t, string) result

val of_string :
  at:'d Nx.Placement.t -> name:string -> string -> ('d t, string) result

val segments : 'd t -> segment list

val state :
  'd t ->
  target:int ->
  center:int ->
  (Ymir.Time.tdb, 'd) Ymir.Time.t ->
  (float, Nx.float64_elt, 'd) Nx.t Ymir.Quantity.t
  * (float, Nx.float64_elt, 'd) Nx.t Ymir.Quantity.t

val position :
  'd t ->
  target:int ->
  center:int ->
  (Ymir.Time.tdb, 'd) Ymir.Time.t ->
  (float, Nx.float64_elt, 'd) Nx.t Ymir.Quantity.t
