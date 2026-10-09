(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Kernel libraries' rows, each through Nx_kernel.S and named
   <backend>/<op>-<case>: the same rows for every backend, its copies and casts,
   a copy of one element and the cost of a call among them. Each row states its
   work, and each backend's support turns that into the floors that bound it on
   its device: nx.cpu's are in cpu/. nx.cpu runs at the target table it starts
   with. *)

module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module F = Nx_cpu_floors

let strf = Printf.sprintf
let row name setup f = Thumper.bench_with_setup ~setup name f

let ok = function
  | A.Done -> ()
  | _ -> failwith "the kernels refused a bench operand"

let kib = 1024
let mib = 1024 * kib

let count n =
  if n mod mib = 0 then strf "%dM" (n / mib)
  else if n mod kib = 0 then strf "%dK" (n / kib)
  else string_of_int n

(* The element counts of the size classes: 4 KiB, 256 KiB, 4 MiB and 64 MiB of
   float32, from L1 to memory. *)
let sizes = [ kib; 64 * kib; mib; 16 * mib ]

let short (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Float64 -> "f64"
  | D.Float32 -> "f32"
  | D.Float16 -> "f16"
  | D.Bfloat16 -> "bf16"
  | D.Float8_e4m3fn -> "e4m3"
  | D.Float8_e5m2 -> "e5m2"
  | D.Float4_e2m1fn -> "e2m1"
  | D.Int64 -> "i64"
  | D.Uint64 -> "u64"
  | D.Int32 -> "i32"
  | D.Uint32 -> "u32"
  | D.Int16 -> "i16"
  | D.Uint16 -> "u16"
  | D.Int8 -> "i8"
  | D.Uint8 -> "u8"
  | D.Int4 -> "i4"
  | D.Uint4 -> "u4"
  | D.Complex128 -> "c128"
  | D.Complex64 -> "c64"
  | D.Bool -> "bool"
  | D.Bit -> "bit"

(* An array of [dt] and shape [s] holding stores of 251 values in [-46.25,
   46.25], as data a kernel meets. *)
let filled (type v s) (dt : (v, s) D.t) s : (v, s) A.t =
  let n = Array.fold_left ( * ) 1 s in
  let x i = float_of_int ((i * 7919 mod 251) - 125) *. 0.37 in
  let src = A.of_array D.Float64 s (Array.init n x) in
  let a = A.create Rig.host dt s in
  ok (Nx_cpu.apply1 Nx_kernel.Prog.Cast ~dst:a src);
  a

(* A row of a backend's kernels and the work its floors bound. *)
type row = { bench : (module Nx_kernel.S) -> Thumper.bench; work : F.work list }

(* A copy of [a ()], [bytes] long when its floors bound it. *)
let copy ?bytes name a =
  let bench (module K : Nx_kernel.S) =
    row name
      (fun () ->
        let a = a () in
        (a, A.create Rig.host (A.dtype a) (A.Layout.shape (A.layout a))))
      (fun (a, dst) -> ok (K.apply1 Nx_kernel.Prog.Copy ~dst a))
  in
  { bench; work = Option.to_list (Option.map (fun n -> F.Copy n) bytes) }

(* A cast of [n] elements of [s] into [d]. *)
let cast (type v s w r) (s : (v, s) D.t) (d : (w, r) D.t) n =
  let bench (module K : Nx_kernel.S) =
    row
      (strf "cast-%s-%s-%s" (short s) (short d) (count n))
      (fun () -> (filled s [| n |], A.create Rig.host d [| n |]))
      (fun (a, dst) -> ok (K.apply1 Nx_kernel.Prog.Cast ~dst a))
  in
  { bench; work = [ F.Cast (D.Any s, D.Any d, n) ] }

let f32 = D.Float32

let copy_rows =
  [ copy "copy-f32-1" (fun () -> filled f32 [| 1 |]) ]
  @ List.map
      (fun n ->
        copy ~bytes:(4 * n)
          ("copy-f32-" ^ count n)
          (fun () -> filled f32 [| n |]))
      sizes
  @ [
      copy
        ~bytes:(4 * 512 * 512)
        "copy-transposed-512x512"
        (fun () ->
          Option.get (A.move (M.Permute [| 1; 0 |]) (filled f32 [| 512; 512 |])));
      copy
        ~bytes:(4 * 4096 * 4096)
        "copy-transposed-4096x4096"
        (fun () ->
          Option.get
            (A.move (M.Permute [| 1; 0 |]) (filled f32 [| 4096; 4096 |])));
      (* Past the M1 Max's 48 MiB system cache and TLB reach, as 4096x4096 is,
         with rows that are no power of two apart. *)
      copy
        ~bytes:(4 * 4000 * 4000)
        "copy-transposed-4000x4000"
        (fun () ->
          Option.get
            (A.move (M.Permute [| 1; 0 |]) (filled f32 [| 4000; 4000 |])));
      (* A row broadcast over rows. *)
      copy ~bytes:(4 * mib) "copy-broadcast-1024x1024-1024" (fun () ->
          Option.get
            (A.move (M.Broadcast [| 1024; 1024 |]) (filled f32 [| 1024 |])));
      (* The 2x2 windows of a pooling layer, the column window's axis before the
         row window's. *)
      copy
        ~bytes:(4 * 32 * 16 * 26 * 26)
        "copy-2x2-windows-32x16x26x26"
        (fun () ->
          let a = filled f32 [| 32; 16; 26; 26 |] in
          let w axis = { M.axis; size = 2; step = 2; dilation = 1 } in
          let v = Option.get (A.move (M.Window [| w 2; w 3 |]) a) in
          Option.get (A.move (M.Permute [| 0; 1; 2; 3; 5; 4 |]) v));
      copy ~bytes:(mib / 2) "copy-i4-1M" (fun () -> filled D.Int4 [| mib |]);
    ]

(* float32 to float16 and to int32 at every size, then one cast per class of
   pairs at 1 Mi elements: narrow floats encoded and decoded, floats to integers
   and back, widening and narrowing, a double to a narrow float, 64-bit integers
   to floats, sub-byte unpacking and packing, complex and boolean. *)
let cast_rows =
  let m = mib in
  List.concat_map (fun n -> [ cast f32 D.Float16 n; cast f32 D.Int32 n ]) sizes
  @ [
      cast D.Float16 f32 m;
      cast f32 D.Bfloat16 m;
      cast D.Bfloat16 f32 m;
      cast f32 D.Float8_e4m3fn m;
      cast D.Float8_e4m3fn f32 m;
      cast D.Int32 f32 m;
      cast f32 D.Int8 m;
      cast f32 D.Uint8 m;
      cast D.Int8 f32 m;
      cast D.Int64 f32 m;
      cast D.Uint64 f32 m;
      cast f32 D.Float64 m;
      cast D.Float64 f32 m;
      cast D.Float64 D.Float16 m;
      cast D.Int32 D.Int64 m;
      cast D.Int64 D.Int32 m;
      cast D.Int4 D.Int8 m;
      cast D.Int8 D.Int4 m;
      cast D.Complex64 D.Complex128 m;
      cast f32 D.Bool m;
      cast D.Bool f32 m;
    ]

let rows = copy_rows @ cast_rows

(* A backend: its rows, then the floors its support derives from their work. *)
let backend name kernels floors =
  Thumper.group name
    (List.map (fun r -> r.bench kernels) rows
    @ floors (List.concat_map (fun r -> r.work) rows))

let () =
  exit
  @@ Thumper.run "nx_kernels"
       [ backend "cpu" (module Nx_cpu : Nx_kernel.S) F.rows ]
