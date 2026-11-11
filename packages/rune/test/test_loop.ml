(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A loop's body as {!Rune.Loop.cut} cuts it: what the call is passed of a
   part of the body that reads storage alone. *)

open Windtrap
open Tolk

let device = "GPU"

(* A step that adds [w], a float32 buffer of 16 elements narrowed to
   bfloat16 and widened back, to its carry, or that passes it to a call of
   the body copying it into the carry: the dtypes of what the loop's call is
   passed, where [host] says whether the device's kernels compile for the
   host's processors. *)
let passed ~host ~called =
  let w = Ops.new_buffer device 16 Float32 in
  let widened = Ops.cast (Ops.cast w Bfloat16) Float32 in
  let param slot = Shape.param ~shape:[ Ops.Int 16 ] ~device slot Float32 in
  let carry = param 0 in
  let own = Ops.Tbl.create 1 in
  Ops.Tbl.replace own carry ();
  let step =
    if called then
      let copy = Ops.sink [ Ops.store (param 0) (param 1) ] in
      Ops.call copy [ carry; widened ]
    else Ops.store carry (Ops.add carry widened)
  in
  let count = ref 0 and args = ref [] in
  let next () =
    incr count;
    !count
  in
  ignore (Rune.Loop.cut ~host:(fun _ -> host) ~own next args (Ops.sink [ step ]));
  List.map (fun (_, u) -> Dtype.name (Ops.dtype u)) !args

let widenings =
  cases "a widening is passed narrow, but on the host and to a call"
    ~id:(fun (host, called, _) ->
      String.concat ", "
        ((if host then "the host" else "a device")
        :: (if called then [ "read by a call" ] else [])))
    [
      (false, false, [ "bf16" ]);
      (true, false, [ "f32" ]);
      (false, true, [ "f32" ]);
      (true, true, [ "f32" ]);
    ]
    (fun (host, called, dtypes) ->
      equal (list string) dtypes (passed ~host ~called))

let () = exit (Windtrap.run "Rune.Loop" [ widenings ])
