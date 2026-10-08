(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The array layer's own costs, each row beside the floor that bounds it: the
   per-op cost of an array kernel over rig's buffer, a movement, the door, and
   bulk element access against the OCaml and Bigarray allocations it fills. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module B = Rig.Buffer

external read_3 : ('v, 's) A.t -> ('v, 's) A.t -> ('v, 's) A.t -> int
  = "nx_array_bench_read_3"
[@@noalloc]

let mib = 1024 * 1024
let row name setup f = Thumper.bench_with_setup ~setup name f
let f32 = D.Float32
let one () = A.of_array f32 [| 1 |] [| 1. |]
let rank4 = [| 2; 3; 4; 5 |]

(* A kernel of one element: the result made, three operands read through the
   door and coalesced, one add. *)
let array_rows =
  Thumper.group "array"
    [
      row "add-1"
        (fun () -> (one (), one ()))
        (fun (x, y) ->
          let z = A.create Rig.host f32 [| 1 |] in
          ignore (Nx_array_support.add_noalloc z x y));
      row "add-1-layout-shared"
        (fun () -> (one (), one ()))
        (fun (x, y) ->
          let z = A.v f32 (A.layout x) (B.create Rig.host 4) in
          ignore (Nx_array_support.add_noalloc z x y));
      Thumper.bench "host-create-16" (fun () -> ignore (B.create Rig.host 16));
    ]

let layout_rows =
  let l = L.contiguous rank4 in
  let p = M.Permute [| 3; 1; 2; 0 |] in
  Thumper.group "layout"
    [
      Thumper.bench "permute-4" (fun () -> L.move p (Sys.opaque_identity l));
      row "equal-4"
        (fun () -> (L.contiguous rank4, L.contiguous rank4))
        (fun (a, b) -> L.equal a b);
      Thumper.bench "dim-4" (fun () ->
          let l = Sys.opaque_identity l in
          L.dim l 0 + L.dim l 1 + L.dim l 2 + L.dim l 3 + L.rank l);
    ]

let door_rows =
  let operand () = A.create Rig.host f32 rank4 in
  Thumper.group "door"
    [
      row "read-3"
        (fun () -> (operand (), operand (), operand ()))
        (fun (z, x, y) -> read_3 z x y);
    ]

let access_rows =
  let n = mib in
  Thumper.group "access"
    [
      row "to_array-f32-1M"
        (fun () -> A.of_array f32 [| n |] (Array.make n 1.5))
        A.to_array;
      row "to_array-i32-1M"
        (fun () -> A.of_array D.Int32 [| n |] (Array.make n 7l))
        A.to_array;
      Thumper.bench "create-f32-1M" (fun () -> A.create Rig.host f32 [| n |]);
      row "bigarray-f32-1M"
        (fun () -> A.of_array f32 [| n |] (Array.make n 1.5))
        (A.bigarray Bigarray.float32);
      (* Twins: the OCaml arrays [to_array] fills, and the bigarray of
         [create]'s bytes. *)
      Thumper.bench "float-array-1M" (fun () -> Array.create_float n);
      Thumper.bench "int32-array-1M" (fun () ->
          Array.init n (fun i -> Int32.of_int (Sys.opaque_identity i)));
      Thumper.bench "bigarray-create-f32-1M" (fun () ->
          Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n);
    ]

let () =
  exit
  @@ Thumper.run "nx_array" [ array_rows; layout_rows; door_rows; access_rows ]
