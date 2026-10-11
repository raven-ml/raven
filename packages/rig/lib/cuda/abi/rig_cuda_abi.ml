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

type graph = { handle : nativeint; nodes : nativeint array }

let graph ~handle ~nodes ~collect =
  let g = { handle; nodes } in
  Gc.finalise (fun _ -> collect ()) g;
  g

type t = {
  symbol : string -> nativeint option;
  graph : kernel array -> (graph, string) result;
}

let key : t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)
