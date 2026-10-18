(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* to_device of views on a GPU whose memory the host does not address: the GPU
   packs them, or the host gathers their runs. It skips without such a GPU. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module B = Rig.Buffer

let layout = Testable.make ~pp:L.pp ~equal:L.equal

let gpu = lazy (Result.to_option (Rig_gpus.first ()))

let off_host () =
  match Lazy.force gpu with
  | Some d when not (Rig.reaches Rig.host d) -> d
  | Some _ -> skip ~reason:"the host addresses the GPU's memory" ()
  | None -> skip ~reason:"no GPU" ()

let slice rs a =
  Option.get
    (A.move
       (M.Slice
          (Iarray.of_list
             (List.map
                (fun (start, count, step) -> { M.start; count; step })
                rs)))
       a)

let iota dt s =
  let n = Array.fold_left ( * ) 1 s in
  A.of_array dt s
    (Array.init n (fun i -> D.of_float dt (float_of_int (i mod 7))))

(* [v] on [d] has [want]'s layout and [v]'s elements, read on the host. *)
let moves name d want v =
  let c = A.to_device d v in
  equal ~msg:(name ^ ": layout") layout want (A.layout c);
  let back x = A.to_array (A.to_device Rig.host x) in
  equal ~msg:(name ^ ": elements") (array int) (back v) (back c)

let test_views () =
  let d = off_host () in
  let m = A.to_device d (iota D.Int16 [| 8; 8 |]) in
  let column = slice [ (0, 8, 1); (2, 1, 1) ] m in
  let views =
    [
      ("column", L.v ~strides:[| 1; 0 |] [| 8; 1 |], column);
      ( "transposed slice",
        L.v ~strides:[| 1; 3 |] [| 3; 4 |],
        Option.get
          (A.move (M.Permute [| 1; 0 |]) (slice [ (0, 4, 2); (1, 3, 2) ] m)) );
      ( "broadcast of a column",
        L.v ~strides:[| 1; 0 |] [| 8; 4 |],
        Option.get (A.move (M.Broadcast [| 8; 4 |]) column) );
      ( "reversed column",
        L.v ~offset:7 ~strides:[| -1; 0 |] [| 8; 1 |],
        slice [ (7, 8, -1); (5, 1, 1) ] m );
    ]
  in
  List.iter
    (fun (name, want, v) ->
      moves (name ^ " to the host") Rig.host want v;
      moves (name ^ " on the GPU") d want v)
    views;
  let q = A.to_device d (iota D.Int4 [| 4; 4 |]) in
  let want = L.v ~strides:[| 1; 0 |] [| 4; 1 |] in
  moves "int4 column to the host" Rig.host want
    (slice [ (0, 4, 1); (1, 1, 1) ] q);
  moves "int4 column on the GPU" d want (slice [ (0, 4, 1); (1, 1, 1) ] q)

let () =
  exit
    (run "nx_array on a GPU"
       [ test "to_device packs views of a GPU's memory" test_views ])
