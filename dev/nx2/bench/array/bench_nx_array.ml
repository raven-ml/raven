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

external loop_3 : ('v, 's) A.t -> ('v, 's) A.t -> ('v, 's) A.t -> int
  = "nx_array_bench_loop_3"

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

external f64_to_e4m3fn : f64 -> u8 -> unit = "nx_array_bench_f64_to_e4m3fn"
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

(* A vector of 1 Mi elements over a rig host buffer, which starts on a page as
   an array's does: a loop's speed can depend on where its data starts. *)
let vec k =
  B.bigarray k (B.create Rig.host (mib * Bigarray.kind_size_in_bytes k))

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
      convert "f64-to-e4m3fn-1M" float64 u8 f64_to_e4m3fn;
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
  (* Four layouts for the contiguity test: [l]; its second block along axis 0,
     contiguous at an offset; [t]; and a broadcast. *)
  let four =
    let all n = { M.start = 0; count = n; step = 1 } in
    let block =
      M.Slice [| { start = 1; count = 1; step = 1 }; all 3; all 4; all 5 |]
    in
    let b = Option.get (L.move (M.Broadcast [| 7; 2; 3; 4; 5 |]) l) in
    [| l; Option.get (L.move block l); t; b |]
  in
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
      Thumper.bench ~budgets:no_alloc "is_contiguous-4x4" (fun () ->
          let ls = Thumper.black_box four in
          let n = ref 0 in
          for i = 0 to 3 do
            if L.is_contiguous (Array.unsafe_get ls i) then incr n
          done;
          !n);
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

(* A kernel's OCaml wrapper: a code other than [NX_OK] is a refusal. *)
let add z x y =
  let e = Nx_array_support.add z x y in
  if e <> 0 then A.refused "add" e [ A.Any z; A.Any x; A.Any y ]

(* A kernel of one element: the result made, three operands read through the
   door and coalesced, one add. Its floor is rig's buffer. The views and
   [expect] are what a kernel library calls around its kernels. *)
let array_rows =
  let narrowed () = Option.get (A.bitcast D.Uint8 (operand ())) in
  (* One element broadcast to [4; 8]: zero strides, which a reshape reads axis
     by axis. *)
  let broadcast () = Option.get (A.move (M.Broadcast [| 4; 8 |]) (one ())) in
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
      row "move-reshape-broadcast" broadcast (A.move (M.Reshape [| 2; 16 |]));
      row "bitcast-f32-i32-4" (fun () -> operand ()) (A.bitcast D.Int32);
      row "bitcast-f32-u8-4" (fun () -> operand ()) (A.bitcast D.Uint8);
      row "bitcast-u8-f32-4" narrowed (A.bitcast f32);
      row "expect-4" (fun () -> A.Any (operand ())) (A.expect f32);
    ]

let door_rows =
  let three () = (operand (), operand (), operand ()) in
  Thumper.group "door"
    [
      row "read-3" three (fun (z, x, y) -> ok "read-3" (read_3 z x y));
      row "loop-3" three (fun (z, x, y) -> loop_3 z x y);
      row "loop-3-scalar"
        (fun () ->
          let s = A.create Rig.host f32 [||] in
          let s = Option.get (A.move (M.Broadcast rank4) s) in
          let z, x, _ = three () in
          if loop_3 z x s <> 1 then failwith "loop-3-scalar: not one run";
          (z, x, s))
        (fun (z, x, s) -> loop_3 z x s);
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
   allocation it fills or the copy that bounds it. Narrow floats store and load
   1 Mi values drawn as the dtype rows draw them. Copies of sub-byte views and
   of windows take the shapes their names give. *)
let access_rows =
  let n = mib in
  let host x = A.of_array f32 [| n |] (Array.make n x) in
  let drawn () =
    let st = Random.State.make [| 32 |] in
    Array.init n (fun _ -> Random.State.float st 200. -. 100.)
  in
  let i4 = [| 1; 2; 3; 4 |] in
  let int4 () = A.of_array D.Int4 rank4 (Array.make 120 3) in
  let int4s k =
    A.of_array D.Int4 [| k |] (Array.init k (fun i -> (i land 15) - 8))
  in
  let filled s =
    A.of_array f32 s (Array.make (Array.fold_left ( * ) 1 s) 1.5)
  in
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
      row "to_array-e4m3fn-1M"
        (fun () -> A.of_array D.Float8_e4m3fn [| n |] (drawn ()))
        A.to_array;
      row "to_array-transposed-512x512"
        (fun () ->
          transpose (A.of_array f32 square (Array.make (512 * 512) 1.5)))
        A.to_array;
      row "of_array-f32-1M"
        (fun () -> Array.make n 1.5)
        (A.of_array f32 [| n |]);
      row "of_array-bf16-1M" drawn (A.of_array D.Bfloat16 [| n |]);
      row "of_array-e4m3fn-1M" drawn (A.of_array D.Float8_e4m3fn [| n |]);
      Thumper.bench "create-f32-1M" (fun () -> A.create Rig.host f32 [| n |]);
      row "copy-f32-1M" (fun () -> host 1.5) A.copy;
      row "copy-transposed-512x512"
        (fun () ->
          transpose (A.of_array f32 square (Array.make (512 * 512) 1.5)))
        A.copy;
      row "copy-i4-1M" (fun () -> int4s n) A.copy;
      (* The source starts inside a byte, the copy on one. *)
      row "copy-i4-1M-offset1"
        (fun () ->
          let all = int4s (n + 1) in
          Option.get
            (A.move (M.Slice [| { start = 1; count = n; step = 1 } |]) all))
        A.copy;
      row "copy-bit-transposed-1024x1024"
        (fun () ->
          let k = 1024 * 1024 in
          transpose
            (A.of_array D.Bit [| 1024; 1024 |]
               (Array.init k (fun i -> i mod 3 = 0))))
        A.copy;
      (* The 2x2 windows of a pooling layer, the column window's axis before
         the row window's. *)
      row "copy-2x2-windows-32x16x26x26"
        (fun () ->
          let a = filled [| 32; 16; 26; 26 |] in
          let w axis = { M.axis; size = 2; step = 2; dilation = 1 } in
          let v = Option.get (A.move (M.Window [| w 2; w 3 |]) a) in
          Option.get (A.move (M.Permute [| 0; 1; 2; 3; 5; 4 |]) v))
        A.copy;
      (* The 3x3 patches of a convolution in im2col order: batch, channel, the
         kernel's rows and columns, then the windows. *)
      row "copy-window-3x3-8x3x64x64"
        (fun () ->
          let a = filled [| 8; 3; 64; 64 |] in
          let w axis = { M.axis; size = 3; step = 1; dilation = 1 } in
          let v = Option.get (A.move (M.Window [| w 2; w 3 |]) a) in
          Option.get (A.move (M.Permute [| 0; 1; 4; 5; 2; 3 |]) v))
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
          let y = A1.create float32 Bigarray.c_layout mib in
          A1.blit x y;
          y);
    ]

(* Copies to a device, here the host: the bytes a layout reaches, between its
   first and last position, with no kernel. *)
let placement_rows =
  let n = mib in
  Thumper.group "placement"
    [
      row "to_device-f32-1" one (A.to_device Rig.host);
      row "to_device-f32-1M"
        (fun () -> A.of_array f32 [| n |] (Array.make n 1.5))
        (A.to_device Rig.host);
      row "to_device-transposed-512x512"
        (fun () ->
          transpose (A.of_array f32 [| 512; 512 |] (Array.make (512 * 512) 1.5)))
        (A.to_device Rig.host);
    ]

(* A kernel after device work: a submit on Late writes [z], and the door waits
   for it under its claims before the kernel runs. Beside it, the submit and the
   wait alone, and the kernel over the same operands with nothing pending. *)
let kernel_rows =
  let opened = ref 0 in
  let late () =
    incr opened;
    let name = Printf.sprintf "nx2-bench-late:%d" !opened in
    let d, _ = Nx_array_support.Late.open_ name in
    let z = A.create d f32 [| 1 |] and x = A.create d f32 [| 1 |] in
    let y = A.create d f32 [| 1 |] in
    (Rig.Submission.make ~reads:0 ~writes:1 d [||], z, x, y)
  in
  let writes s b =
    ignore (Rig.submit s ~reads:[||] ~writes:[| b |] ~waits:[||])
  in
  let pending () =
    let ((s, z, x, y) as env) = late () in
    writes s (A.buffer z);
    add z x y;
    env
  in
  Thumper.group "kernel"
    [
      row "add-1-pending" pending (fun (s, z, x, y) ->
          writes s (A.buffer z);
          add z x y);
      row "floor-submit-wait-1" late (fun (s, z, _, _) ->
          writes s (A.buffer z);
          B.wait (A.buffer z) B.Read_write);
      row "add-1-reached" late (fun (_, z, x, y) -> add z x y);
    ]

let () =
  exit
  @@ Thumper.run "nx_array"
       [
         dtype_rows;
         move_rows;
         layout_rows;
         array_rows;
         door_rows;
         access_rows;
         placement_rows;
         kernel_rows;
       ]
