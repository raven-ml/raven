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

(* Kinds of no, two and three operands into a fresh C-contiguous array. *)
let apply name dst run =
  let bench (module K : Nx_kernel.S) =
    row name (fun () -> dst ()) (fun x -> ok (run (module K : Nx_kernel.S) x))
  in
  { bench; work = [] }

let fill n =
  let k = Nx_kernel.Prog.Fill (Nx_kernel.Prog.bits f32 1.5) in
  apply ("fill-f32-" ^ count n)
    (fun () -> A.create Rig.host f32 [| n |])
    (fun (module K) dst -> K.apply0 k ~dst)

let binary name k dt n rd =
  apply name
    (fun () -> (filled dt [| n |], filled dt [| n |], A.create Rig.host rd [| n |]))
    (fun (module K) (x, y, dst) -> K.apply2 k ~dst x y)

let apply_rows =
  let m = mib in
  [
    fill 1;
    fill m;
    apply "iota-i64-1M"
      (fun () -> A.create Rig.host D.Int64 [| m |])
      (fun (module K) dst -> K.apply0 (Iota 0) ~dst);
  ]
  @ List.map
      (fun n -> binary ("add-f32-" ^ count n) (Binary Add) f32 n f32)
      [ 1; kib; m ]
  @ [
      apply "add-f32-1M-transposed"
        (fun () ->
          let x = filled f32 [| 1024; 1024 |] in
          ( Option.get (A.move (M.Permute [| 1; 0 |]) x),
            filled f32 [| 1024; 1024 |],
            A.create Rig.host f32 [| 1024; 1024 |] ))
        (fun (module K) (x, y, dst) -> K.apply2 (Binary Add) ~dst x y);
      binary "add-i8-1M" (Binary Add) D.Int8 m D.Int8;
      binary "less-f32-1M" (Compare Less) f32 m D.Bool;
      apply "where-f32-1M"
        (fun () ->
          let c = A.create Rig.host D.Bool [| m |] in
          ok
            (Nx_cpu.apply2 (Compare Less) ~dst:c (filled f32 [| m |])
               (filled f32 [| m |]));
          (c, filled f32 [| m |], filled f32 [| m |], A.create Rig.host f32 [| m |]))
        (fun (module K) (c, x, y, dst) -> K.apply3 Where ~dst c x y);
      apply "fma-f32-1M"
        (fun () ->
          ( filled f32 [| m |],
            filled f32 [| m |],
            filled f32 [| m |],
            A.create Rig.host f32 [| m |] ))
        (fun (module K) (x, y, z, dst) -> K.apply3 Fma ~dst x y z);
    ]

(* Contractions: [a] and [b] laid out by [layout] from C-contiguous arrays of
   the shapes it is given, contracted over [contracting], with [init] if
   given; the floor runs their flops at the host's peak and, with [streams],
   reads that many bytes. *)
let contract ?streams ?(init = false) ~acc name ~sa ~sb ~layout ~contracting
    ~flops =
  let (D.Any dt) = acc in
  let bench (module K : Nx_kernel.S) =
    let spec =
      Nx_kernel.Spec.contract ~batch:[||] ~contracting ~acc ~out:acc ~init
    in
    let shape (A.Any x) = A.Layout.shape (A.layout x) in
    row name
      (fun () ->
        let a, b = layout (A.Any (filled dt sa)) (A.Any (filled dt sb)) in
        let no_init =
          Nx_kernel.Spec.contract ~batch:[||] ~contracting ~acc ~out:acc
            ~init:false
        in
        let y =
          match Nx_kernel.Spec.shapes no_init [| shape a; shape b |] with
          | Ok [| y |] -> y
          | _ -> failwith "a contraction row's shapes do not fit"
        in
        let ops = [ a; b ] @ if init then [ A.Any (filled dt y) ] else [] in
        (A.Any (A.create Rig.host dt y), Array.of_list ops))
      (fun (dst, ops) -> ok (K.contract spec ~dst ops))
  in
  let read = Option.to_list (Option.map (fun n -> F.Read n) streams) in
  { bench; work = F.Fma (acc, flops) :: read }

let plain a b = (a, b)

(* The product of [m × k] and [k × n] C-contiguous matrices. *)
let gemm ?init ~acc name m n k =
  contract ?init ~acc name ~sa:[| m; k |] ~sb:[| k; n |] ~layout:plain
    ~contracting:[| (1, 0) |] ~flops:(2 * m * n * k)

let f64 = D.Float64

let contract_rows =
  let acc32 = D.Any f32 and acc64 = D.Any f64 in
  let square acc dt n = gemm ~acc (Printf.sprintf "contract-%s-%d" dt n) n n n in
  List.map (square acc32 "f32") [ 64; 128; 256; 512; 1024; 2048; 4096 ]
  @ List.map (square acc64 "f64") [ 256; 1024 ]
  @ [
      gemm ~init:true ~acc:acc32 "contract-f32-1024-init" 1024 1024 1024;
      (* a transposed: the [k × m] array viewed as [m × k]. *)
      contract ~acc:acc32 "contract-f32-1024-tn" ~sa:[| 1024; 1024 |]
        ~sb:[| 1024; 1024 |]
        ~layout:(fun (A.Any a) b ->
          (A.Any (Option.get (A.move (M.Permute [| 1; 0 |]) a)), b))
        ~contracting:[| (1, 0) |] ~flops:(2 * 1024 * 1024 * 1024);
      (* A dot of 10 Mi elements. *)
      contract ~acc:acc32 "contract-dot-f32-10M" ~sa:[| 10 * mib |]
        ~sb:[| 10 * mib |] ~layout:plain
        ~contracting:[| (0, 0) |] ~flops:(2 * 10 * mib)
        ~streams:(2 * 4 * 10 * mib);
    ]
  (* Decoding: [m] rows against a 4096 × 4096 weight stored as [n × k]. *)
  @ List.map
      (fun m ->
        contract ~acc:acc32
          (Printf.sprintf "contract-f32-m%dx4096x4096" m)
          ~sa:[| m; 4096 |] ~sb:[| 4096; 4096 |] ~layout:plain
          ~contracting:[| (1, 1) |]
          ~flops:(2 * m * 4096 * 4096)
          ~streams:(4 * 4096 * 4096))
      [ 1; 8; 32; 128 ]

let rows = copy_rows @ cast_rows @ apply_rows @ contract_rows

(* A backend: its rows, then the floors its support derives from their work. *)
let backend name kernels floors =
  Thumper.group name
    (List.map (fun r -> r.bench kernels) rows
    @ floors (List.concat_map (fun r -> r.work) rows))

let () =
  exit
  @@ Thumper.run "nx_kernels"
       [ backend "cpu" (module Nx_cpu : Nx_kernel.S) F.rows ]
