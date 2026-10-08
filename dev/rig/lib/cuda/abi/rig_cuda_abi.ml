(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kernel = {
  func : int;
  grid : int * int * int;
  block : int * int * int;
  shared : int;
  args : string;
}

type graph = {
  handle : nativeint;
  nodes : nativeint array;
  release : unit -> unit;
}

type t = {
  symbol : string -> nativeint option;
  graph : kernel array -> (graph, string) result;
}

let key = Type.Id.make ()
