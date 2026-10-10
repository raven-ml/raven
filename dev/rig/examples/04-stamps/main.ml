(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Memory across devices.

   Each memory records the point of its last write and, per device, the point of
   its last use: its stamps. Work that reads memory waits for its last write;
   work that writes it waits for every use. So a program orders work on several
   devices by the memory the work touches, and the host does the same before it
   reads or writes ([Buffer.wait]).

   Two memory devices stand for two GPUs here. Their work is done when [submit]
   returns, so no wait ever blocks; the stamps and the data flow are those of
   devices whose work runs late. *)

open Rig

let int32s b = Buffer.bigarray Bigarray.int32 b

let show name b =
  Buffer.wait b Read;
  let a = int32s b in
  let xs = List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}) in
  Printf.printf "%-3s on %s [%s]\n" name
    (Rig.name (Buffer.device b))
    (String.concat "; " xs)

let copy src dst =
  { Submission.queue = "COPY:0"; after = [||]; work = Copy { src; dst } }

let () =
  let a = Result.get_ok (memory_device "A") in
  let b = Result.get_ok (memory_device "B") in
  Printf.printf "A's work reaches B's memory once borrowed: %b\n" (reaches a b);

  (* The host writes [src] on A. It first waits for every use of [src], so that
     no device's work still reads what it overwrites. *)
  let src = Buffer.create a 16 and x = Buffer.create a 16 in
  Buffer.wait src Read_write;
  List.iteri (fun i v -> (int32s src).{i} <- v) [ 5l; 6l; 7l; 8l ];

  (* A's work writes [x]: [x]'s last write is A's point. *)
  let run = Submission.Run.make () in
  let p =
    submit (Submission.make a [| copy src x |]) ~run ~buffers:[||] ~waits:[||]
  in
  Format.printf "A wrote x at %a@." Point.pp p;

  (* B's work addresses its own memory. [x] is A's, so a part of B that names it
     is refused; B borrows [x], a mapping that shares [x]'s stamps. *)
  let y = Buffer.create b 16 in
  (match Submission.make b [| copy x y |] with
  | _ -> ()
  | exception Invalid_argument msg -> print_endline msg);
  let x_on_b = Option.get (Buffer.borrow b x) in

  (* B's copy reads [x], so its submit waits for A's write first. A point the
     submit waits for orders it after any other work as well, here A's last
     value. *)
  let s = Submission.make b [| copy x_on_b y |] in
  Format.printf "B read x at %a@." Point.pp
    (submit s ~run ~buffers:[||] ~waits:[| p |]);
  show "y" y;

  (* A host copy orders itself the same way: it waits for [y]'s last write and
     for every use of its destination. *)
  let z = Buffer.create host 16 in
  Buffer.copy ~src:y ~dst:z;
  show "z" z;
  Printf.printf "A submitted %d, B submitted %d\n" (submitted a) (submitted b)
