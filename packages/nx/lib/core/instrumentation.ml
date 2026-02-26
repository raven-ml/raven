(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type event = {
  op : string;
  kind : [ `Info | `Warn | `Error ];
  msg : string;
  attrs : (string * string) list;
}

type hook = {
  enabled : bool;
  with_span :
    'a.
    op:string -> ?attrs:(unit -> (string * string) list) -> (unit -> 'a) -> 'a;
  emit : event -> unit;
}

let null_hook =
  {
    enabled = false;
    with_span = (fun ~op:_ ?attrs:_ f -> f ());
    emit = (fun _ -> ());
  }

let current_hook : hook ref = ref null_hook
let set_hook h = current_hook := Option.value ~default:null_hook h

let with_hook h f =
  let prev = !current_hook in
  set_hook h;
  Fun.protect f ~finally:(fun () -> current_hook := prev)
