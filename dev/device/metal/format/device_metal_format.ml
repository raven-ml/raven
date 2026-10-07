(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type dispatch = {
  pipeline : nativeint;
  offset : int;
  groups : int * int * int;
  threads : int * int * int;
}

type icb = {
  handle : nativeint;
  commands : nativeint array;
  release : unit -> unit;
}

type t = {
  icb : nativeint -> dispatch array -> (icb, string) result;
  split : nativeint;
}

let key = Type.Id.make ()
