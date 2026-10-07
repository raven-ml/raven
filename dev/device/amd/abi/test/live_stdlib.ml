(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A value of each Stdlib module the AMD library links, so that both probes link
   and initialise the same ones. *)

let linked =
  [
    Obj.repr Array.length;
    Obj.repr Bool.to_int;
    Obj.repr Buffer.create;
    Obj.repr Bytes.create;
    Obj.repr Char.code;
    Obj.repr Either.left;
    Obj.repr Hashtbl.hash;
    Obj.repr Iarray.length;
    Obj.repr Int.max;
    Obj.repr List.length;
    Obj.repr Option.get;
    Obj.repr Printf.sprintf;
    Obj.repr Result.get_ok;
    Obj.repr Seq.empty;
    Obj.repr String.length;
    Obj.repr Type.Id.make;
    Obj.repr Uchar.of_int;
  ]
