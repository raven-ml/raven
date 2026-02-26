(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status = Starting | Idle | Busy | Shutting_down

type event =
  | Output of { cell_id : Cell.id; output : Cell.output }
  | Finished of { cell_id : Cell.id; success : bool }
  | Status_changed of status

type t = {
  execute : cell_id:Cell.id -> code:string -> unit;
  interrupt : unit -> unit;
  complete : code:string -> pos:int -> string list;
  status : unit -> status;
  shutdown : unit -> unit;
}
