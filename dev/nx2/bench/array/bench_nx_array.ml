(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The array layer's own costs, each row beside the floor that bounds it: the
   store rule and the conversions a kernel runs per element, against loops of
   the same bytes; movements and the coalescer; the per-op cost of an array
   kernel over rig's buffer; the door; and bulk element access against the OCaml
   and Bigarray allocations it fills. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module B = Rig.Buffer
module A1 = Bigarray.Array1

external read_3 : ('v, 's) A.t -> ('v, 's) A.t -> ('v, 's) A.t -> int
  = "nx_array_bench_read_3"
[@@noalloc]

external loop_3 : ('v, 's) A.t -> ('v, 's) A.t -> ('v, 's) A.t -> int
  = "nx_array_bench_loop_3"
[@@noalloc]

external claim_3 : B.t -> B.t -> B.t -> unit = "nx_array_bench_claim_3"
[@@noalloc]

let mib = 1024 * 1024
let row name setup f = Thumper.bench_with_setup ~setup name f
let f32 = D.Float32
let one () = A.of_array f32 [| 1 |] [| 1. |]
let rank4 = [| 2; 3; 4; 5 |]
let no_alloc = [ Thumper.Budget.at_most ~metric:Thumper.Metric.alloc_words 0. ]

(* Dtypes *)

type ('v, 's) vec = ('v, 's, Bigarray.c_layout) A1.t
type f32 = (float, Bigarray.float32_elt) vec
type f64 = (float, Bigarray.float64_elt) vec
type u16 = (int, Bigarray.int16_unsigned_elt) vec
type u8 = (int, Bigarray.int8_unsigned_elt) vec
type i32 = (int32, Bigarray.int32_elt) vec

external f32_to_f16 : f32 -> u16 -> unit = "nx_array_bench_f32_to_f16"
[@@noalloc]

external f16_to_f32 : u16 -> f32 -> unit = "nx_array_bench_f16_to_f32"
[@@noalloc]

external f32_to_bf16 : f32 -> u16 -> unit = "nx_array_bench_f32_to_bf16"
[@@noalloc]

external bf16_to_f32 : u16 -> f32 -> unit = "nx_array_bench_bf16_to_f32"
[@@noalloc]

external f32_to_e4m3fn : f32 -> u8 -> unit = "nx_array_bench_f32_to_e4m3fn"
[@@noalloc]

external e4m3fn_to_f32 : u8 -> f32 -> unit = "nx_array_bench_e4m3fn_to_f32"
[@@noalloc]

external f32_to_e5m2 : f32 -> u8 -> unit = "nx_array_bench_f32_to_e5m2"
[@@noalloc]

external e5m2_to_f32 : u8 -> f32 -> unit = "nx_array_bench_e5m2_to_f32"
[@@noalloc]

external f32_to_e2m1fn : f32 -> u8 -> unit = "nx_array_bench_f32_to_e2m1fn"
[@@noalloc]

external e2m1fn_to_f32 : u8 -> f32 -> unit = "nx_array_bench_e2m1fn_to_f32"
[@@noalloc]

external f64_to_f16 : f64 -> u16 -> unit = "nx_array_bench_f64_to_f16"
[@@noalloc]

external f64_to_i32 : f64 -> i32 -> unit = "nx_array_bench_f64_to_i32"
[@@noalloc]

external f64_to_f16_run : f64 -> u16 -> unit = "nx_array_bench_f64_to_f16_run"
[@@noalloc]

external f16_to_f64_run : u16 -> f64 -> unit = "nx_array_bench_f16_to_f64_run"
[@@noalloc]

external f64_to_bf16_run : f64 -> u16 -> unit = "nx_array_bench_f64_to_bf16_run"
[@@noalloc]

external bf16_to_f64_run : u16 -> f64 -> unit = "nx_array_bench_bf16_to_f64_run"
[@@noalloc]

external floor_f32_to_u16 : f32 -> u16 -> unit
  = "nx_array_bench_floor_f32_to_u16"
[@@noalloc]

external floor_f32_to_u8 : f32 -> u8 -> unit = "nx_array_bench_floor_f32_to_u8"
[@@noalloc]

external floor_u16_to_f32 : u16 -> f32 -> unit
  = "nx_array_bench_floor_u16_to_f32"
[@@noalloc]

external floor_u8_to_f32 : u8 -> f32 -> unit = "nx_array_bench_floor_u8_to_f32"
[@@noalloc]

external floor_f64_to_u16 : f64 -> u16 -> unit
  = "nx_array_bench_floor_f64_to_u16"
[@@noalloc]

external floor_f64_to_i32 : f64 -> i32 -> unit
  = "nx_array_bench_floor_f64_to_i32"
[@@noalloc]

let vec k = A1.create k Bigarray.c_layout mib
let float32 = Bigarray.float32
let float64 = Bigarray.float64

(* 1 Mi values drawn uniformly from [-bound, bound], the same in every trial.
   [bound] defaults to 100: normal in every float format but e2m1fn, which takes
   8 to stay mostly in range. *)
let values ?(bound = 100.) k =
  let st = Random.State.make [| 32 |] and x = vec k in
  for i = 0 to mib - 1 do
    A1.unsafe_set x i (Random.State.float st (2. *. bound) -. bound)
  done;
  x

(* [f] from the values in [k] into a vector of [k']. *)
let convert ?bound name k k' f =
  row name (fun () -> (values ?bound k, vec k')) (fun (x, y) -> f x y)

(* [f] from the codes [codes] stores from the binary32 values into a vector of
   [k]. *)
let decode ?bound name k' ~codes k f =
  let setup () =
    let c = vec k' in
    codes (values ?bound float32) c;
    (c, vec k)
  in
  row name setup (fun (c, y) -> f c y)

let dtype_rows =
  let u16 = Bigarray.int16_unsigned and u8 = Bigarray.int8_unsigned in
  let of_float name dt =
    Thumper.bench ("of_float-" ^ name) (fun () ->
        D.of_float dt (Thumper.black_box 0.1))
  in
  Thumper.group "dtype"
    [
      of_float "f32" f32;
      of_float "f16" D.Float16;
      of_float "bf16" D.Bfloat16;
      of_float "e4m3fn" D.Float8_e4m3fn;
      of_float "i32" D.Int32;
      of_float "u8" D.Uint8;
      convert "f32-to-f16-1M" float32 u16 f32_to_f16;
      convert "f32-to-bf16-1M" float32 u16 f32_to_bf16;
      convert "floor-f32-to-u16-1M" float32 u16 floor_f32_to_u16;
      decode "f16-to-f32-1M" u16 ~codes:f32_to_f16 float32 f16_to_f32;
      decode "bf16-to-f32-1M" u16 ~codes:f32_to_bf16 float32 bf16_to_f32;
      decode "floor-u16-to-f32-1M" u16 ~codes:f32_to_f16 float32
        floor_u16_to_f32;
      convert "f32-to-e4m3fn-1M" float32 u8 f32_to_e4m3fn;
      convert "f32-to-e5m2-1M" float32 u8 f32_to_e5m2;
      convert ~bound:8. "f32-to-e2m1fn-1M" float32 u8 f32_to_e2m1fn;
      convert "floor-f32-to-u8-1M" float32 u8 floor_f32_to_u8;
      decode "e4m3fn-to-f32-1M" u8 ~codes:f32_to_e4m3fn float32 e4m3fn_to_f32;
      decode "e5m2-to-f32-1M" u8 ~codes:f32_to_e5m2 float32 e5m2_to_f32;
      decode ~bound:8. "e2m1fn-to-f32-1M" u8 ~codes:f32_to_e2m1fn float32
        e2m1fn_to_f32;
      decode "floor-u8-to-f32-1M" u8 ~codes:f32_to_e4m3fn float32
        floor_u8_to_f32;
      convert "f64-to-f16-1M" float64 u16 f64_to_f16;
      convert "f64-to-f16-run-1M" float64 u16 f64_to_f16_run;
      convert "f64-to-bf16-run-1M" float64 u16 f64_to_bf16_run;
      convert "floor-f64-to-u16-1M" float64 u16 floor_f64_to_u16;
      decode "f16-to-f64-run-1M" u16 ~codes:f32_to_f16 float64 f16_to_f64_run;
      decode "bf16-to-f64-run-1M" u16 ~codes:f32_to_bf16 float64 bf16_to_f64_run;
      convert "f64-to-i32-1M" float64 Bigarray.int32 f64_to_i32;
      convert "floor-f64-to-i32-1M" float64 Bigarray.int32 floor_f64_to_i32;
      row "floor-copy-f32-1M"
        (fun () -> (values float32, vec float32))
        (fun (x, y) -> A1.blit x y);
    ]

(* Movements and layouts, of a rank-4 layout of 120 elements. *)

let move_rows =
  let s = Thumper.black_box rank4 in
  Thumper.group "move"
    [
      Thumper.bench "shape-permute-4" (fun () ->
          M.shape (M.Permute [| 3; 1; 2; 0 |]) s);
      Thumper.bench "shape-reshape-4" (fun () ->
          M.shape (M.Reshape [| 6; 20 |]) s);
    ]

let layout_rows =
  let l = L.contiguous rank4 in
  let t = Option.get (L.move (M.Permute [| 3; 1; 2; 0 |]) l) in
  let c = L.contiguous (L.shape t) in
  let move name m =
    Thumper.bench name (fun () -> L.move m (Thumper.black_box l))
  in
  let range count = { M.start = 0; count; step = 1 } in
  Thumper.group "layout"
    [
      Thumper.bench "contiguous-4" (fun () ->
          L.contiguous (Thumper.black_box rank4));
      Thumper.bench "v-4" (fun () ->
          L.v ~strides:[| 1; 2; 6; 24 |] (Thumper.black_box rank4));
      move "permute-4" (M.Permute [| 3; 1; 2; 0 |]);
      move "reshape-4" (M.Reshape [| 6; 20 |]);
      move "broadcast-4" (M.Broadcast [| 7; 2; 3; 4; 5 |]);
      move "slice-4"
        (M.Slice
           [| range 2; range 3; { start = 3; count = 2; step = -2 }; range 5 |]);
      row "equal-4"
        (fun () -> (L.contiguous rank4, L.contiguous rank4))
        (fun (a, b) -> L.equal a b);
      Thumper.bench ~budgets:no_alloc "hash-4" (fun () ->
          L.hash (Thumper.black_box l));
      Thumper.bench ~budgets:no_alloc "dim-4" (fun () ->
          let l = Thumper.black_box l in
          L.dim l 0 + L.dim l 1 + L.dim l 2 + L.dim l 3 + L.rank l);
      Thumper.bench "coalesce-3" (fun () ->
          L.coalesce (Thumper.black_box [| l; l; l |]));
      Thumper.bench "coalesce-3-transposed" (fun () ->
          L.coalesce (Thumper.black_box [| t; c; c |]));
    ]

(* Arrays and the door. A kernel's operands are rank-4 float32 arrays of 120
   elements on the host; [t] is the transposed view of one. *)

let ok name e = if e <> 0 then failwith (name ^ ": the door refused")
let operand ?(shape = rank4) () = A.create Rig.host f32 shape

(* [a] with its axes reversed. *)
let transpose a =
  let r = L.rank (A.layout a) in
  Option.get (A.move (M.Permute (Array.init r (fun i -> r - 1 - i))) a)

(* A kernel of one element: the result made, three operands read through the
   door and coalesced, one add. Its floor is rig's buffer. *)
let array_rows =
  Thumper.group "array"
    [
      row "add-1"
        (fun () -> (one (), one ()))
        (fun (x, y) ->
          let z = A.create Rig.host f32 [| 1 |] in
          ok "add-1" (Nx_array_support.add z x y));
      row "add-1-layout-shared"
        (fun () -> (one (), one ()))
        (fun (x, y) ->
          let z = A.v f32 (A.layout x) (B.create Rig.host 4) in
          ok "add-1-layout-shared" (Nx_array_support.add z x y));
      Thumper.bench "create-1" (fun () -> A.create Rig.host f32 [| 1 |]);
      Thumper.bench "floor-host-create-16" (fun () -> B.create Rig.host 16);
      Thumper.bench "move-permute-4"
        (let a = operand () in
         fun () -> A.move (M.Permute [| 3; 1; 2; 0 |]) (Thumper.black_box a));
    ]

let door_rows =
  let three () = (operand (), operand (), operand ()) in
  Thumper.group "door"
    [
      row "read-3" three (fun (z, x, y) -> ok "read-3" (read_3 z x y));
      row "loop-3" three (fun (z, x, y) -> loop_3 z x y);
      row "loop-3-transposed"
        (fun () ->
          let shape = [| 5; 4; 3; 2 |] in
          (operand ~shape (), transpose (operand ()), operand ~shape ()))
        (fun (z, x, y) -> loop_3 z x y);
      row "floor-claim-3"
        (fun () ->
          let z, x, y = three () in
          (A.buffer z, A.buffer x, A.buffer y))
        (fun (z, x, y) -> claim_3 z x y);
    ]

(* Element and bulk access over 1 Mi float32 elements, each beside the
   allocation it fills or the copy that bounds it. *)
let access_rows =
  let n = mib in
  let host x = A.of_array f32 [| n |] (Array.make n x) in
  let i4 = [| 1; 2; 3; 4 |] in
  let int4 () = A.of_array D.Int4 rank4 (Array.make 120 3) in
  let square = [| 512; 512 |] in
  Thumper.group "access"
    [
      row "get-f32-4" (fun () -> operand ()) (fun a -> A.get a i4);
      row "set-f32-4" (fun () -> operand ()) (fun a -> A.set a i4 1.5);
      row "get-i4-4" int4 (fun a -> A.get a i4);
      row "set-i4-4" int4 (fun a -> A.set a i4 5);
      row "to_array-f32-1M" (fun () -> host 1.5) A.to_array;
      row "to_array-f16-1M"
        (fun () -> A.of_array D.Float16 [| n |] (Array.make n 1.5))
        A.to_array;
      row "to_array-i32-1M"
        (fun () -> A.of_array D.Int32 [| n |] (Array.make n 7l))
        A.to_array;
      row "of_array-f32-1M"
        (fun () -> Array.make n 1.5)
        (A.of_array f32 [| n |]);
      Thumper.bench "create-f32-1M" (fun () -> A.create Rig.host f32 [| n |]);
      row "copy-f32-1M" (fun () -> host 1.5) A.copy;
      row "copy-transposed-512x512"
        (fun () ->
          transpose (A.of_array f32 square (Array.make (512 * 512) 1.5)))
        A.copy;
      row "bigarray-f32-1M" (fun () -> host 1.5) (A.bigarray Bigarray.float32);
      row "of_bigarray-f32-1M"
        (fun () ->
          Bigarray.genarray_of_array1
            (Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n))
        (A.of_bigarray f32);
      (* Floors: the OCaml arrays [to_array] fills, the bigarray of [create]'s
         bytes, and a copy of 4 MiB into a fresh bigarray. *)
      Thumper.bench "floor-float-array-1M" (fun () -> Array.create_float n);
      Thumper.bench "floor-int32-array-1M" (fun () ->
          Array.init n (fun i -> Int32.of_int (Sys.opaque_identity i)));
      Thumper.bench "floor-bigarray-create-f32-1M" (fun () ->
          A1.create Bigarray.float32 Bigarray.c_layout n);
      row "floor-bigarray-copy-f32-1M"
        (fun () -> values float32)
        (fun x ->
          let y = vec float32 in
          A1.blit x y;
          y);
    ]

let () =
  exit
  @@ Thumper.run "nx_array"
       [
         dtype_rows; move_rows; layout_rows; array_rows; door_rows; access_rows;
       ]
