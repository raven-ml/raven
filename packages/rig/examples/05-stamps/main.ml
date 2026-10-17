(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Memory across devices.

   Each allocation records the point of its last write and, for each device,
   the point of its last use. These points are its stamps. Work that reads
   memory follows its last write, and work that writes it follows every use.
   So work on several devices is ordered by the memory it touches, with no
   point named, and the host's reads and writes ([Buffer.read],
   [Buffer.write]) are ordered the same way.

   Two memory devices stand for two GPUs here. Their work is done when
   [submit] returns, so no wait ever blocks. The stamps and the order are
   those of devices whose work runs later. *)

open Rig

(* [a]'s elements, joined. *)
let ints a =
  String.concat "; "
    (List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}))

let show name b =
  Buffer.read b Bigarray.int32 (fun a ->
      Printf.printf "%-3s on %s [%s]\n" name
        (Rig.name (Buffer.device b))
        (ints a))

let () =
  let a = Result.get_ok (memory_device "A") in
  let b = Result.get_ok (memory_device "B") in
  Printf.printf "A's work reaches B's memory once borrowed: %b\n" (reaches a b);

  (* The host writes [src] on A once every use of [src] is done, so that no
     device's work still reads what it overwrites. *)
  let src = Buffer.create a 16 and x = Buffer.create a 16 in
  Buffer.write src Bigarray.int32 (fun w ->
      List.iteri (fun i v -> w.{i} <- v) [ 5l; 6l; 7l; 8l ]);

  (* A's work writes [x], so [x]'s last write is A's point. *)
  let run = Submission.Run.make () in
  let s = Submission.(make a [| copy ~src:(Buffer src) ~dst:(Buffer x) |]) in
  Format.printf "A wrote x at %a@." Point.pp (submit s ~run ~buffers:[||]);

  (* B's work addresses its own memory. [x] is A's, so a part of B that names
     it is refused. A borrow of [x] on B shares [x]'s stamps. *)
  let y = Buffer.create b 16 in
  (match Submission.(make b [| copy ~src:(Buffer x) ~dst:(Buffer y) |]) with
  | _ -> ()
  | exception Invalid_argument msg -> print_endline msg);
  let x_on_b = Option.get (Buffer.borrow b x) in

  (* B's copy reads [x], so it follows A's write by [x]'s stamps alone. *)
  let s = Submission.(make b [| copy ~src:(Buffer x_on_b) ~dst:(Buffer y) |]) in
  Format.printf "B read x at %a@." Point.pp (submit s ~run ~buffers:[||]);
  show "y" y;

  (* A host copy is ordered the same way. It follows [y]'s last write and
     every use of its destination. *)
  let z = Buffer.create host 16 in
  Buffer.copy ~src:y ~dst:z;
  show "z" z;
  Printf.printf "A submitted %d, B submitted %d\n" (submitted a) (submitted b)
