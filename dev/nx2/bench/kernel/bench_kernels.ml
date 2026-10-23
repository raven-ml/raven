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

(* A call of [map] on the one node [n] over [loads], read plain, into
   [dst]: the program and the arrays built once, as a caller holds them. *)
let one ins n dst loads =
  ( Nx_kernel.Prog.of_node ~ins:(Iarray.of_array ins) n,
    [| A.Any dst |],
    Array.map (fun (A.Any x) -> Nx_kernel.Plain x) loads )

let call (module K : Nx_kernel.S) (p, dsts, loads) = ok (K.map p ~dsts loads)

(* [src] cast into [dst] by nx.cpu. *)
let cast_into ~dst src =
  let s = D.Any (A.dtype src) and d = D.Any (A.dtype dst) in
  call (module Nx_cpu) (one [| s |] (Op1 (Cast, d, 0)) dst [| A.Any src |])

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
  cast_into ~dst:a src;
  a

(* A row of a backend's kernels and the work its floors bound. *)
type row = { bench : (module Nx_kernel.S) -> Thumper.bench; work : F.work list }

(* A copy of [a ()], [bytes] long when its floors bound it. *)
let copy ?bytes name a =
  let bench (module K : Nx_kernel.S) =
    row name
      (fun () ->
        let a = a () in
        let d = D.Any (A.dtype a) in
        let dst = A.create Rig.host (A.dtype a) (A.Layout.shape (A.layout a)) in
        one [| d |] (Op1 (Copy, d, 0)) dst [| A.Any a |])
      (call (module K))
  in
  { bench; work = Option.to_list (Option.map (fun n -> F.Copy n) bytes) }

(* A cast of [n] elements of [s] into [d]. *)
let cast (type v s w r) (s : (v, s) D.t) (d : (w, r) D.t) n =
  let bench (module K : Nx_kernel.S) =
    row
      (strf "cast-%s-%s-%s" (short s) (short d) (count n))
      (fun () ->
        one [| D.Any s |]
          (Op1 (Cast, D.Any d, 0))
          (A.create Rig.host d [| n |])
          [| A.Any (filled s [| n |]) |])
      (call (module K))
  in
  { bench; work = [ F.Cast (D.Any s, D.Any d, n) ] }

(* A cast of the transpose of [n × n] elements of [s] into [d]: a narrow
   float's strided elements go through the stage's buffer. *)
let transposed_cast (type v s w r) (s : (v, s) D.t) (d : (w, r) D.t) n =
  let bench (module K : Nx_kernel.S) =
    row
      (strf "cast-%s-%s-%dx%d-transposed" (short s) (short d) n n)
      (fun () ->
        let a =
          Option.get (A.move (M.Permute [| 1; 0 |]) (filled s [| n; n |]))
        in
        one [| D.Any s |]
          (Op1 (Cast, D.Any d, 0))
          (A.create Rig.host d [| n; n |])
          [| A.Any a |])
      (call (module K))
  in
  { bench; work = [ F.Cast (D.Any s, D.Any d, n * n) ] }

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
      (* float64's square blocks, 32 MiB: past fudge's and nonnormal's
         largest caches. *)
      copy
        ~bytes:(8 * 2048 * 2048)
        "copy-transposed-f64-2048x2048"
        (fun () ->
          Option.get
            (A.move (M.Permute [| 1; 0 |]) (filled D.Float64 [| 2048; 2048 |])));
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
      transposed_cast D.Bfloat16 f32 1024;
      transposed_cast D.Float8_e4m3fn f32 1024;
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

(* One-node maps into a fresh C-contiguous array, the call built by
   [setup]. *)
let apply ?work name setup =
  let bench (module K : Nx_kernel.S) = row name setup (call (module K)) in
  { bench; work = Option.to_list work }

(* The floor of an elementwise kind over [n] elements. *)
let stream ins inb outb n = F.Stream { ins; inb; outb; n }

let fill n =
  let f = D.Any f32 in
  apply ~work:(stream 0 4 4 n) ("fill-f32-" ^ count n) (fun () ->
      one [||]
        (Const (f, Nx_kernel.Prog.bits f32 1.5))
        (A.create Rig.host f32 [| n |])
        [||])

let binary name k dt n rd =
  apply
    ~work:(stream 2 (D.bits dt / 8) (D.bits rd / 8) n)
    name
    (fun () ->
      one [| D.Any dt; D.Any dt |]
        (Op2 (k, 0, 1))
        (A.create Rig.host rd [| n |])
        [| A.Any (filled dt [| n |]); A.Any (filled dt [| n |]) |])

(* An add of [n] elements of a dtype whose stream has no floor row. *)
let narrow name dt n =
  apply name (fun () ->
      one [| D.Any dt; D.Any dt |]
        (Op2 (Binary Add, 0, 1))
        (A.create Rig.host dt [| n |])
        [| A.Any (filled dt [| n |]); A.Any (filled dt [| n |]) |])

(* An array of [dt] and shape [s] drawn from N(0, 1), as activations are:
   Box-Muller over a fixed sequence. *)
let normal (type v s) (dt : (v, s) D.t) s : (v, s) A.t =
  let n = Array.fold_left ( * ) 1 s in
  let u i = (float_of_int ((i * 7919 mod 65521) + 1)) /. 65522. in
  let x i =
    let r = sqrt (-2. *. log (u (2 * i))) in
    r *. cos (2. *. Float.pi *. u ((2 * i) + 1))
  in
  let src = A.of_array D.Float64 s (Array.init n x) in
  let a = A.create Rig.host dt s in
  cast_into ~dst:a src;
  a

(* A kind of one operand over [n] elements of [dt]: [filled]'s, or with
   [`Normal] [normal]'s, the row's name then ending in -normal. *)
let unary ?(values = `Filled) name u dt n =
  let w = D.bits dt / 8 in
  let x () =
    match values with
    | `Filled -> filled dt [| n |]
    | `Normal -> normal dt [| n |]
  in
  let suffix = match values with `Filled -> "" | `Normal -> "-normal" in
  apply ~work:(stream 1 w w n)
    (strf "%s-%s-%s%s" name (short dt) (count n) suffix)
    (fun () ->
      one [| D.Any dt |]
        (Op1 (Unary u, D.Any dt, 0))
        (A.create Rig.host dt [| n |])
        [| A.Any (x ()) |])

(* A map of the program [p] over [n] elements of its loads, into fresh
   arrays of its outputs' dtypes. *)
let map ?work name (p : Nx_kernel.Prog.t) n =
  apply ?work name (fun () ->
      ( p,
        Array.map
          (fun o ->
            let (D.Any d) = Nx_kernel.Prog.dtype p o in
            A.Any (A.create Rig.host d [| n |]))
          (Iarray.to_array p.outs),
        Array.map
          (fun (D.Any d) -> Nx_kernel.Plain (filled d [| n |]))
          (Iarray.to_array p.ins) ))

(* Maps of several nodes: six float32 nodes over one load, a program the
   interpreter runs block by block in L1; and a Mul by a constant, a
   repeated element to its row. *)
let map_rows =
  let module P = Nx_kernel.Prog in
  let m = mib and f = D.Any f32 in
  let six =
    P.v ~ins:[| f |]
      P.
        [|
          In 0;
          Op2 (Binary Mul, 0, 0);
          Op2 (Binary Add, 1, 0);
          Op2 (Binary Mul, 2, 0);
          Op2 (Binary Sub, 3, 1);
          Op2 (Binary Maximum, 4, 0);
          Op2 (Binary Mul, 5, 2);
        |]
      ~outs:[| 6 |]
  in
  let scale =
    P.v ~ins:[| f |]
      [| P.In 0; Const (f, P.bits f32 0.5); Op2 (Binary Mul, 0, 1) |]
      ~outs:[| 2 |]
  in
  [
    map ~work:(stream 1 4 4 m) "map-6-f32-1M" six m;
    map ~work:(stream 1 4 4 m) "map-scale-f32-1M" scale m;
  ]

let apply_rows =
  let m = mib in
  [
    fill 1;
    fill m;
    apply ~work:(stream 0 8 8 m) "iota-i64-1M" (fun () ->
        one [||] (Coord 0) (A.create Rig.host D.Int64 [| m |]) [||]);
  ]
  @ List.map
      (fun n -> binary ("add-f32-" ^ count n) (Binary Add) f32 n f32)
      [ 1; kib; m ]
  @ [
      apply ~work:(stream 2 4 4 m) "add-f32-1M-transposed" (fun () ->
          let x = filled f32 [| 1024; 1024 |] in
          one [| D.Any f32; D.Any f32 |]
            (Op2 (Binary Add, 0, 1))
            (A.create Rig.host f32 [| 1024; 1024 |])
            [|
              A.Any (Option.get (A.move (M.Permute [| 1; 0 |]) x));
              A.Any (filled f32 [| 1024; 1024 |]);
            |]);
      binary "add-i8-1M" (Binary Add) D.Int8 m D.Int8;
      (* Narrow dtypes through their carriers, bound by add-f32-1M's and
         add-i8-1M's element rates rather than a floor of their own. *)
      narrow "add-bf16-1M" D.Bfloat16 m;
      narrow "add-e4m3-1M" D.Float8_e4m3fn m;
      narrow "add-i4-1M" D.Int4 m;
      binary "less-f32-1M" (Compare Less) f32 m D.Bool;
      binary "mul-c64-1M" (Binary Mul) D.Complex64 m D.Complex64;
      binary "mul-c128-1M" (Binary Mul) D.Complex128 m D.Complex128;
      binary "maximum-f32-1M" (Binary Maximum) f32 m f32;
      apply ~work:(F.Select { inb = 4; n = m }) "where-f32-1M" (fun () ->
          let c = A.create Rig.host D.Bool [| m |] in
          let f = D.Any f32 in
          call (module Nx_cpu)
            (one [| f; f |]
               (Op2 (Compare Less, 0, 1))
               c
               [| A.Any (filled f32 [| m |]); A.Any (filled f32 [| m |]) |]);
          one [| D.Any D.Bool; f; f |]
            (Op3 (Where, 0, 1, 2))
            (A.create Rig.host f32 [| m |])
            [| A.Any c; A.Any (filled f32 [| m |]); A.Any (filled f32 [| m |]) |]);
      apply ~work:(stream 3 4 4 m) "fma-f32-1M" (fun () ->
          let f = D.Any f32 in
          one [| f; f; f |]
            (Op3 (Fma, 0, 1, 2))
            (A.create Rig.host f32 [| m |])
            (Array.init 3 (fun _ -> A.Any (filled f32 [| m |]))));
    ]
  (* A unary kind that moves bits, then the transcendental kinds. *)
  @ [ unary "neg" Neg f32 m ]
  @ List.concat_map
      (fun (name, u) -> [ unary name u f32 m; unary name u D.Float64 m ])
      Nx_kernel.Prog.
        [
          ("exp", Exp); ("log", Log); ("sin", Sin); ("tanh", Tanh); ("erf", Erf);
        ]
  (* erf over its near range most of all, as GELU's activations are. *)
  @ List.map
      (fun (D.Any dt) -> unary ~values:`Normal "erf" Erf dt m)
      [ D.Any f32; D.Any D.Float64 ]

(* Contractions: [a] and [b] laid out by [layout] from C-contiguous arrays of
   the shapes it is given, contracted over [contracting] in [order], with
   [init] if given; the floor runs their flops at the host's peak and, with
   [streams], reads that many bytes. *)
(* A contraction in [acc] of [operands] into a dst of [out], both [acc]
   by default, from [init] of [out] where given. *)
let contract ?streams ?(init = false) ?operands ?out ~acc ~order name ~sa ~sb
    ~layout ~contracting ~flops =
  let (D.Any ot) = Option.value operands ~default:acc in
  let (D.Any yt) = Option.value out ~default:acc in
  let bench (module K : Nx_kernel.S) =
    let shape (A.Any x) = A.Layout.shape (A.layout x) in
    row name
      (fun () ->
        let a, b = layout (A.Any (filled ot sa)) (A.Any (filled ot sb)) in
        let y =
          match
            Nx_kernel.Shape.contract ~batch:[||] ~contracting ~segments:None
              (shape a, None) (shape b, None)
          with
          | Ok y -> y
          | Error _ -> failwith "a contraction row's shapes do not fit"
        in
        let init = if init then Some (filled yt y) else None in
        let a : Nx_kernel.operand = { array = a; scale = None } in
        let b : Nx_kernel.operand = { array = b; scale = None } in
        (A.create Rig.host yt y, init, a, b))
      (fun (dst, init, a, b) ->
        ok
          (K.contract ~order ~batch:[||] ~contracting ~segments:None ~acc ~dst
             ~init a b))
  in
  let read = Option.to_list (Option.map (fun n -> F.Read n) streams) in
  { bench; work = F.Fma (acc, flops) :: read }

let plain a b = (a, b)

(* [m] float32 rows against an MXFP4 weight: float4 codes stored
   [4096 × 4096] as [n × k], a code of 2^(e - 127) per 32 along k. *)
let mxfp4 ~order name m =
  let acc = D.Any D.Float32 and contracting : _ iarray = [| (1, 1) |] in
  let n = 4096 and k = 4096 in
  let bench (module K : Nx_kernel.S) =
    row name
      (fun () ->
        let codes =
          A.of_array D.Uint8 [| n; k / 32 |]
            (Array.init (n * k / 32) (fun i -> 120 + (i mod 15)))
        in
        let a : Nx_kernel.operand =
          { array = A.Any (filled D.Float32 [| m; k |]); scale = None }
        in
        let b : Nx_kernel.operand =
          { array = A.Any (filled D.Float4_e2m1fn [| n; k |]);
            scale = Some (Exponent codes) }
        in
        (A.create Rig.host D.Float32 [| m; n |], a, b))
      (fun (dst, a, b) ->
        ok
          (K.contract ~order ~batch:[||] ~contracting ~segments:None ~acc ~dst
             ~init:None a b))
  in
  { bench; work = [ F.Fma (acc, 2 * m * n * k); F.Read (n * k / 2) ] }

(* [t] float32 tokens, each routed to 8 of [g] MXFP4 experts' weights of
   [n × k] stored as [mxfp4]'s: the [8t] routed rows, sorted by expert,
   cut into the experts' groups by Rows segments. Token [i]'s experts are
   [(5i + 4j) mod g] for [j] below 8, fixed so the groups' sizes are. *)
let mxfp4_experts name t g =
  let acc = D.Any D.Float32 and contracting : _ iarray = [| (1, 2) |] in
  let n = 1024 and k = 4096 and m = 8 * t in
  let bench (module K : Nx_kernel.S) =
    row name
      (fun () ->
        let counts = Array.make g 0 in
        for i = 0 to t - 1 do
          for j = 0 to 7 do
            let e = ((5 * i) + (4 * j)) mod g in
            counts.(e) <- counts.(e) + 1
          done
        done;
        let offsets = Array.make (g + 1) 0L in
        Array.iteri
          (fun e c -> offsets.(e + 1) <- Int64.add offsets.(e) (Int64.of_int c))
          counts;
        let codes =
          A.of_array D.Uint8 [| g; n; k / 32 |]
            (Array.init (g * n * k / 32) (fun i -> 120 + (i mod 15)))
        in
        let a : Nx_kernel.operand =
          { array = A.Any (filled D.Float32 [| m; k |]); scale = None }
        in
        let b : Nx_kernel.operand =
          { array = A.Any (filled D.Float4_e2m1fn [| g; n; k |]);
            scale = Some (Exponent codes) }
        in
        let offsets = A.of_array D.Int64 [| g + 1 |] offsets in
        let segments = Some (Nx_kernel.Rows { offsets; a_axis = 0; b_axis = 0 }) in
        (A.create Rig.host D.Float32 [| m; n |], a, b, segments))
      (fun (dst, a, b, segments) ->
        ok
          (K.contract ~order:Chain ~batch:[||] ~contracting
             ~segments ~acc ~dst ~init:None a b))
  in
  { bench; work = [ F.Fma (acc, 2 * m * n * k); F.Read (g * n * k / 2) ] }

(* The product of [m × k] and [k × n] C-contiguous matrices in [order]. *)
let gemm ?init ~acc ~order name m n k =
  contract ?init ~acc ~order name ~sa:[| m; k |] ~sb:[| k; n |] ~layout:plain
    ~contracting:[| (1, 0) |] ~flops:(2 * m * n * k)

let f64 = D.Float64

(* The 100³ float32 product after one by 111 columns of subnormals, whose
   edge sliver of b, packed into a thread's buffer, leaves them where the
   100-column product's edge sliver has its padding. x86 kernels mask
   their loads of b past n; without the masks, each FMA of a subnormal
   takes a microcode assist. The row guards x86 only (on the M1
   it times a plain 100³ product), and only when the worker that packed
   the 111-column edge takes the 100-column edge: always on one thread,
   by chance where threads steal units. Both products sum in chain order,
   as nx gives them. *)
let after_subnormals =
  let acc = D.Any f32 and contracting : _ iarray = [| (1, 0) |] in
  let bench (module K : Nx_kernel.S) =
    let plain a : Nx_kernel.operand = { array = A.Any a; scale = None } in
    let product dst a b =
      ok
        (K.contract ~order:Nx_kernel.Chain ~batch:[||] ~contracting
           ~segments:None ~acc ~dst ~init:None a b)
    in
    row "contract-f32-100-after-subnormals"
      (fun () ->
        let a = filled f32 [| 100; 100 |] in
        let tiny = A.of_array f32 [| 100; 111 |] (Array.make 11100 1e-40) in
        product (A.create Rig.host f32 [| 100; 111 |]) (plain a) (plain tiny);
        ( A.create Rig.host f32 [| 100; 100 |],
          plain a,
          plain (filled f32 [| 100; 100 |]) ))
      (fun (dst, a, b) -> product dst a b)
  in
  { bench; work = [ F.Fma (acc, 2 * 100 * 100 * 100) ] }

let contract_rows =
  let acc32 = D.Any f32 and acc64 = D.Any f64 in
  let chain = Nx_kernel.Chain and lane = Nx_kernel.Lane in
  (* Squares of 8 or more: chain order. *)
  let square acc dt n =
    gemm ~acc ~order:chain (Printf.sprintf "contract-%s-%d" dt n) n n n
  in
  (* 8³ is a call's fixed cost: the view, the door, the jobs. *)
  List.map (square acc32 "f32") [ 8; 64; 128; 256; 512; 1024; 2048; 4096 ]
  @ List.map (square acc64 "f64") [ 256; 1024 ]
  @ [
      (* Edge tiles and partial blocks: 40, 50 and 100 are no multiple of
         a tile's side, and their rows lie off 64 bytes. *)
      square acc32 "f32" 40;
      square acc32 "f32" 50;
      square acc32 "f32" 100;
      after_subnormals;
      square acc64 "f64" 40;
      square acc64 "f64" 50;
      square acc64 "f64" 100;
      (* Few columns, many rows, [4096 × 256] by [256 × 64]: computed as
         its transpose, whose columns lie apart in dst. *)
      gemm ~acc:acc32 ~order:chain "contract-f32-4096x64x256" 4096 64 256;
      (* A factorisation's update: a triangular solve's half by 32 columns
         of its right-hand sides. *)
      gemm ~acc:acc64 ~order:chain "contract-f64-512x32x512" 512 32 512;
      gemm ~init:true ~acc:acc32 ~order:chain "contract-f32-1024-init" 1024
        1024 1024;
      (* a transposed: the [k × m] array viewed as [m × k]. *)
      contract ~acc:acc32 ~order:chain "contract-f32-1024-tn"
        ~sa:[| 1024; 1024 |]
        ~sb:[| 1024; 1024 |]
        ~layout:(fun (A.Any a) b ->
          (A.Any (Option.get (A.move (M.Permute [| 1; 0 |]) a)), b))
        ~contracting:[| (1, 0) |] ~flops:(2 * 1024 * 1024 * 1024);
      (* Results of another dtype than acc, from an init: float32 summed in
         int4, and complex64 in complex128, whose copies hold the most kept
         buffers at once. *)
      contract ~init:true ~operands:acc32 ~out:acc32 ~acc:(D.Any D.Int4)
        ~order:chain "contract-f32-i4-256-init" ~sa:[| 256; 256 |]
        ~sb:[| 256; 256 |] ~layout:plain ~contracting:[| (1, 0) |]
        ~flops:(2 * 256 * 256 * 256);
      contract ~init:true ~operands:(D.Any D.Complex64)
        ~out:(D.Any D.Complex64) ~acc:(D.Any D.Complex128) ~order:chain
        "contract-c64-c128-256-init" ~sa:[| 256; 256 |] ~sb:[| 256; 256 |]
        ~layout:plain ~contracting:[| (1, 0) |]
        ~flops:(8 * 256 * 256 * 256);
      (* A dot of 10 Mi elements. *)
      contract ~acc:acc32 ~order:lane "contract-dot-f32-10M"
        ~sa:[| 10 * mib |]
        ~sb:[| 10 * mib |] ~layout:plain
        ~contracting:[| (0, 0) |] ~flops:(2 * 10 * mib)
        ~streams:(2 * 4 * 10 * mib);
    ]
  (* Decoding: [m] rows against a 4096 × 4096 weight stored as [n × k]. *)
  @ List.map
      (fun (m, order) ->
        contract ~acc:acc32 ~order
          (Printf.sprintf "contract-f32-m%dx4096x4096" m)
          ~sa:[| m; 4096 |] ~sb:[| 4096; 4096 |] ~layout:plain
          ~contracting:[| (1, 1) |]
          ~flops:(2 * m * 4096 * 4096)
          ~streams:(4 * 4096 * 4096))
      [ (1, lane); (4, lane); (8, chain); (32, chain); (128, chain) ]
  (* One row against the weight stored as [k × n]. *)
  @ [
      contract ~acc:acc32 ~order:lane "contract-f32-m1x4096x4096-kn"
        ~sa:[| 1; 4096 |]
        ~sb:[| 4096; 4096 |] ~layout:plain ~contracting:[| (1, 0) |]
        ~flops:(2 * 4096 * 4096) ~streams:(4 * 4096 * 4096);
      (* A token, and 8 tokens, through an MXFP4 weight: lane order's dot
         stages the codes along k, chain order's few rows pack them. *)
      mxfp4 ~order:lane "contract-mxfp4-m1x4096x4096" 1;
      mxfp4 ~order:chain "contract-mxfp4-m8x4096x4096" 8;
      (* 64 tokens through 8 of 32 experts, a mixture of experts' layer. *)
      mxfp4_experts "contract-mxfp4-experts-64x8of32" 64 32;
    ]

(* Reductions and scans of one operand by a monoid, its extreme with its
   position, its logsumexp, mean and variance, and its running mean and
   variance. A reduction's floor reads its operand's bytes; a scan's copies
   them once per result, since it writes as many. *)

let identity dt = Nx_kernel.Prog.v ~ins:[| dt |] [| In 0 |] ~outs:[| 0 |]

let reduce name (r : Nx_kernel.reduction) ~axes a (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x) = a () in
      let dt = D.Any (A.dtype x) in
      let s = A.Layout.shape (A.layout x) in
      let y =
        Array.of_list
          (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list s))
      in
      let like () = A.Any (A.create Rig.host (A.dtype x) y) in
      let dsts =
        match r with
        | Monoid _ | Logsumexp -> [| like () |]
        | Moments -> [| like (); like () |]
        | Arg _ -> [| like (); A.Any (A.create Rig.host D.Int64 y) |]
      in
      (identity dt, s, [| (r, 0) |], [| Nx_kernel.Plain x |], dsts))
    (fun (p, shape, rs, loads, dsts) ->
      ok (K.reduce p ~shape ~axes:(Iarray.of_array axes) rs ~dsts loads))

let scan name r ~axis a (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x) = a () in
      let dt = D.Any (A.dtype x) in
      let s = A.Layout.shape (A.layout x) in
      let results =
        match r with Nx_kernel.Monoid _ | Logsumexp -> 1 | _ -> 2
      in
      ( identity dt,
        (r, 0),
        [| Nx_kernel.Plain x |],
        Array.init results (fun _ -> A.Any (A.create Rig.host (A.dtype x) s))
      ))
    (fun (p, r, loads, dsts) -> ok (K.scan p ~axis r ~dsts loads))

(* [run] of the log of 1M positive float32 terms but the last, negative,
   its log NaN, into a destination of shape [dst]: a sum or a scan of the
   program finds its first NaN term past all the others. *)
let late_nan name ~dst run (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let n = mib in
      let x =
        A.of_array D.Float32 [| n |]
          (Array.init n (fun i ->
               if i = n - 1 then -1. else 1. +. float_of_int (i mod 1000)))
      in
      let p =
        let d = D.Any f32 in
        Nx_kernel.Prog.v ~ins:[| d |] [| In 0; Op1 (Unary Log, d, 0) |]
          ~outs:[| 1 |]
      in
      (p, [| Nx_kernel.Plain x |], [| A.Any (A.create Rig.host f32 dst) |]))
    (fun (p, loads, dsts) -> ok (run (module K : Nx_kernel.S) p ~dsts loads))

let reduce_rows =
  let bytes dt s = D.bits dt / 8 * Array.fold_left ( * ) 1 s in
  let any dt s () = A.Any (filled dt s) in
  let along name dt s axes r =
    { bench = reduce name r ~axes (any dt s); work = [ F.Read (bytes dt s) ] }
  in
  let stated r name =
    [
      along (Printf.sprintf "reduce-%s-f32-1M" name) f32 [| mib |] [| 0 |] r;
      along
        (Printf.sprintf "reduce-%s-f32-1024x1024-axis1" name)
        f32 [| 1024; 1024 |] [| 1 |] r;
    ]
  in
  let running name dt s axis =
    {
      bench = scan name (Monoid Sum) ~axis (any dt s);
      work = [ F.Copy (bytes dt s) ];
    }
  in
  let moments name dt s axis =
    {
      bench = scan name Moments ~axis (any dt s);
      work = [ F.Copy (bytes dt s); F.Copy (bytes dt s) ];
    }
  in
  let m = mib in
  [
    along "reduce-sum-f32-1M" f32 [| m |] [| 0 |] (Monoid Sum);
    along "reduce-sum-f32-16M" f32 [| 16 * m |] [| 0 |] (Monoid Sum);
    along "reduce-sum-f64-16M" f64 [| 16 * m |] [| 0 |] (Monoid Sum);
    along "reduce-sum-i32-16M" D.Int32 [| 16 * m |] [| 0 |] (Monoid Sum);
    along "reduce-max-f32-16M" f32 [| 16 * m |] [| 0 |] (Monoid Max);
    (* Along an axis whose outputs lie side by side: rows of outputs. *)
    along "reduce-sum-f32-1024x1024-axis0" f32 [| 1024; 1024 |] [| 0 |] (Monoid Sum);
    along "reduce-sum-f32-4096x4096-axis0" f32 [| 4096; 4096 |] [| 0 |] (Monoid Sum);
    (* Along the innermost axis: each output's run into lanes. *)
    along "reduce-sum-f32-1024x1024-axis1" f32 [| 1024; 1024 |] [| 1 |] (Monoid Sum);
    along "reduce-sum-f32-4096x4096-axis1" f32 [| 4096; 4096 |] [| 1 |] (Monoid Sum);
    (* Four terms an output. *)
    along "reduce-sum-f32-1Mx4-axis1" f32 [| m; 4 |] [| 1 |] (Monoid Sum);
    (* Exact monoids of bytes: their runs and blocks along the innermost
       axis, their columns across rows; booleans' Max is any, Min all. *)
    along "reduce-sum-i8-1024x1024-axis1" D.Int8 [| 1024; 1024 |] [| 1 |] (Monoid Sum);
    along "reduce-prod-i8-1024x1024-axis1" D.Int8 [| 1024; 1024 |] [| 1 |] (Monoid Prod);
    along "reduce-sum-i8-1024x1024-axis0" D.Int8 [| 1024; 1024 |] [| 0 |] (Monoid Sum);
    along "reduce-prod-i8-1024x1024-axis0" D.Int8 [| 1024; 1024 |] [| 0 |] (Monoid Prod);
    along "reduce-max-bool-1024x1024-axis1" D.Bool [| 1024; 1024 |] [| 1 |] (Monoid Max);
    along "reduce-min-bool-1024x1024-axis0" D.Bool [| 1024; 1024 |] [| 0 |] (Monoid Min);
    (* No term: the identity filled into each output. *)
    {
      bench =
        reduce "reduce-sum-f32-1Mx0-axis1" (Monoid Sum) ~axes:[| 1 |]
          (any f32 [| m; 0 |]);
      work = [ F.Copy (4 * m) ];
    };
    (* The maximum of each 2x2 window of a pooling layer. *)
    {
      bench =
        reduce "reduce-max-f32-2x2-windows-32x16x26x26" (Monoid Max) ~axes:[| 4; 5 |]
          (fun () ->
            let w axis = { M.axis; size = 2; step = 2; dilation = 1 } in
            let a = filled f32 [| 32; 16; 26; 26 |] in
            A.Any (Option.get (A.move (M.Window [| w 2; w 3 |]) a)));
      work = [ F.Read (bytes f32 [| 32; 16; 26; 26 |]) ];
    };
    running "scan-sum-f64-5M" f64 [| 5 * m |] 0;
    running "scan-sum-f32-4096x1024-axis1" f32 [| 4096; 1024 |] 1;
    running "scan-sum-f32-1024x4096-axis0" f32 [| 1024; 4096 |] 0;
    moments "scan-moments-f32-4096x1024-axis1" f32 [| 4096; 1024 |] 1;
    moments "scan-moments-f32-1024x4096-axis0" f32 [| 1024; 4096 |] 0;
    moments "scan-moments-f32-1M" f32 [| m |] 0;
    moments "scan-moments-f32-5x768x5-axis1" f32 [| 5; 768; 5 |] 1;
    (* A NaN last: the fast pass meets it, and the exact one runs. *)
    {
      bench =
        scan "scan-moments-f32-1M-late-nan" Moments ~axis:0 (fun () ->
            let x = filled f32 [| m |] in
            A.set x [| m - 1 |] Float.nan;
            A.Any x);
      work = [ F.Copy (bytes f32 [| m |]); F.Copy (bytes f32 [| m |]) ];
    };
    {
      bench =
        late_nan "reduce-sum-log-f32-1M-late-nan" ~dst:[||]
          (fun (module K : Nx_kernel.S) p ~dsts l ->
            K.reduce p ~shape:[| mib |] ~axes:[| 0 |] [| (Monoid Sum, 0) |]
              ~dsts l);
      work = [ F.Read (bytes f32 [| m |]) ];
    };
    {
      bench =
        late_nan "scan-sum-log-f32-1M-late-nan" ~dst:[| mib |]
          (fun (module K : Nx_kernel.S) p ~dsts l ->
            K.scan p ~axis:0 (Monoid Sum, 0) ~dsts l);
      work = [ F.Copy (bytes f32 [| m |]) ];
    };
  ]
  (* The reductions computed by a state of each block. *)
  @ stated (Arg Max) "argmax"
  @ stated Logsumexp "logsumexp"
  @ stated Moments "moments"
  @ [ along "reduce-moments-f64-1M" f64 [| m |] [| 0 |] Moments ]

(* Gathers and scatters. A gather's floor copies its result's bytes and reads
   its positions'; a scatter's reads its positions and updates and copies
   [into]. Positions are random unless monotone. *)

let positions n bound f =
  A.of_array D.Int64 [| n |]
    (Array.init n (fun i -> Int64.of_int (f i mod bound)))

let random i = (i * 2654435761) lsr 7

(* The setups give the call as a closure over its operands, whose dtype the row
   does not know. *)
let gather_row name ~axis x idx (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x) = x () in
      let idx = idx () in
      let dst = A.create Rig.host (A.dtype x) (A.Layout.shape (A.layout idx)) in
      fun () -> ok (K.gather ~axis ~dst idx x))
    (fun call -> call ())

let scatter_row name c ~axis into idx u (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any into) = into () in
      let u = A.expect (A.dtype into) (u ()) and idx = idx () in
      let dst =
        A.create Rig.host (A.dtype into) (A.Layout.shape (A.layout into))
      in
      fun () -> ok (K.scatter c ~unique:false ~axis ~dst ~into idx u))
    (fun call -> call ())

let index_rows =
  let m = mib in
  let any dt s () = A.Any (filled dt s) in
  let flat n bound f () = positions n bound f in
  let rows n w bound () =
    Option.get
      (A.move
         (M.Broadcast [| n; w |])
         (Option.get (A.move (M.Reshape [| n; 1 |]) (positions n bound random))))
  in
  let gather name ~axis x idx ~out ~read =
    { bench = gather_row name ~axis x idx; work = [ F.Copy out; F.Read read ] }
  in
  let scatter name c ~axis into idx u ~into_bytes ~read =
    {
      bench = scatter_row name c ~axis into idx u;
      work = [ F.Copy into_bytes; F.Read read ];
    }
  in
  let along n () =
    A.of_array D.Int64 [| n; n |]
      (Array.init (n * n) (fun i -> Int64.of_int (random i mod n)))
  in
  [
    (* A row take, an embedding's lookup: rows of 8 float32. *)
    gather "gather-f32-rows-1Mx8" ~axis:0
      (any f32 [| m; 8 |])
      (rows m 8 m) ~out:(32 * m) ~read:(40 * m);
    gather "gather-f32-16M-monotone" ~axis:0
      (any f32 [| 16 * m |])
      (flat (16 * m) (16 * m) Fun.id)
      ~out:(64 * m) ~read:(192 * m);
    gather "gather-f64-16M-random" ~axis:0
      (any f64 [| 16 * m |])
      (flat (16 * m) (16 * m) random)
      ~out:(128 * m) ~read:(256 * m);
    gather "gather-f32-along-axis1-2048x2048" ~axis:1
      (any f32 [| 2048; 2048 |])
      (along 2048) ~out:(16 * m) ~read:(48 * m);
    scatter "scatter-add-f64-16M-into-1M" Add ~axis:0 (any f64 [| m |])
      (flat (16 * m) m random)
      (any f64 [| 16 * m |])
      ~into_bytes:(8 * m) ~read:(256 * m);
    scatter "scatter-add-f64-16M-into-128" Add ~axis:0 (any f64 [| 128 |])
      (flat (16 * m) 128 random)
      (any f64 [| 16 * m |])
      ~into_bytes:1024 ~read:(256 * m);
    scatter "scatter-set-f64-16M-into-16M" Set ~axis:0
      (any f64 [| 16 * m |])
      (flat (16 * m) (16 * m) random)
      (any f64 [| 16 * m |])
      ~into_bytes:(128 * m) ~read:(256 * m);
    scatter "scatter-max-f64-16M-into-1M" Max ~axis:0 (any f64 [| m |])
      (flat (16 * m) m random)
      (any f64 [| 16 * m |])
      ~into_bytes:(8 * m) ~read:(256 * m);
    scatter "scatter-add-f16-16M-into-1M" Add ~axis:0 (any D.Float16 [| m |])
      (flat (16 * m) m random)
      (any D.Float16 [| 16 * m |])
      ~into_bytes:(2 * m) ~read:(160 * m);
    (* An embedding's gradient: rows of 512 float32 into a table of 32 Ki. *)
    scatter "scatter-add-f32-rows-64Kx512" Add ~axis:0
      (any f32 [| 32 * kib; 512 |])
      (rows (64 * kib) 512 (32 * kib))
      (any f32 [| 64 * kib; 512 |])
      ~into_bytes:(64 * m)
      ~read:((128 * m) + (512 * kib));
  ]

(* Sorts. A sort's floor copies its keys and positions once per radix pass of
   its keys' span. *)

let sort_row name ~axis ~descending ~k x (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x) = x () in
      let y = Array.copy (A.Layout.shape (A.layout x)) in
      Option.iter (fun k -> y.(axis) <- k) k;
      let values = A.create Rig.host (A.dtype x) y in
      let positions = A.create Rig.host D.Int64 y in
      fun () -> ok (K.sort ~axis ~descending ~values ~positions x))
    (fun call -> call ())

let sort_rows =
  let m = mib in
  let any dt s () = A.Any (filled dt s) in
  (* Keys over the whole of int64's range: SplitMix64's outputs. *)
  let wide n () =
    let mix i =
      let open Int64 in
      let z = mul (of_int (i + 1)) 0x9E3779B97F4A7C15L in
      let z = mul (logxor z (shift_right_logical z 30)) 0xBF58476D1CE4E5B9L in
      let z = mul (logxor z (shift_right_logical z 27)) 0x94D049BB133111EBL in
      logxor z (shift_right_logical z 31)
    in
    A.Any (A.of_array D.Int64 [| n |] (Array.init n mix))
  in
  let sort name ?(descending = false) ?k ~axis x ~passes ~n =
    {
      bench = sort_row name ~axis ~descending ~k x;
      work = [ F.Copy (passes * n * 16) ];
    }
  in
  [
    sort "sort-f32-1M" ~axis:0 (any f32 [| m |]) ~passes:4 ~n:m;
    (* 251 values: one byte of span. *)
    sort "sort-i64-1M" ~axis:0 (any D.Int64 [| m |]) ~passes:1 ~n:m;
    sort "sort-i64-1M-wide" ~axis:0 (wide m) ~passes:8 ~n:m;
    sort "sort-f32-rows-512x512" ~axis:1
      (any f32 [| 512; 512 |])
      ~passes:4 ~n:(512 * 512);
    (* A mixture of experts' router: the best 4 of 32 per token. *)
    sort "topk-4-of-32-512-rows" ~descending:true ~k:4 ~axis:1
      (any f32 [| 512; 32 |])
      ~passes:1 ~n:(512 * 32);
  ]

(* Assemblies and folds. An assembly's floor copies its result's bytes; a fold's
   reads its operand and copies its result once per tap, as its boxes do. *)

let assemble_row name ~shape ~fill regions pieces (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let pieces = Array.combine regions (pieces ()) in
      let dst = A.create Rig.host (A.dtype (snd pieces.(0))) shape in
      fun () -> ok (K.assemble ~fill ~dst pieces))
    (fun call -> call ())

let fold_row name ~shape pad x (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x) = x () in
      let dst = A.create Rig.host (A.dtype x) shape in
      fun () -> ok (K.fold pad ~dst x))
    (fun call -> call ())

let assembly_rows =
  let any s () = A.Any (filled f32 s) in
  let whole d = { M.start = 0; count = d; step = 1 } in
  let n = 512 in
  let conv =
    let w axis = { M.axis; size = 3; step = 1; dilation = 1 } in
    {
      Nx_kernel.lo = [| 0; 0; 1; 1 |];
      hi = [| 0; 0; 1; 1 |];
      interior = [| 0; 0; 0; 0 |];
      windows = [| w 2; w 3 |];
    }
  in
  let image = [| 32; 64; 56; 56 |] in
  let bytes s = 4 * Array.fold_left ( * ) 1 s in
  [
    {
      bench =
        assemble_row "assemble-concat-2x-512x512-f32"
          ~shape:[| 2 * n; n |]
          ~fill:0.
          [|
            [| { M.start = 0; count = n; step = 1 }; whole n |];
            [| { M.start = n; count = n; step = 1 }; whole n |];
          |]
          (fun () -> [| filled f32 [| n; n |]; filled f32 [| n; n |] |]);
      work = [ F.Copy (bytes [| 2 * n; n |]) ];
    };
    {
      bench =
        assemble_row "assemble-pad-1-of-1022x1022-f32" ~shape:[| 1024; 1024 |]
          ~fill:0.
          [|
            [|
              { M.start = 1; count = 1022; step = 1 };
              { M.start = 1; count = 1022; step = 1 };
            |];
          |]
          (fun () -> [| filled f32 [| 1022; 1022 |] |]);
      work = [ F.Copy (bytes [| 1024; 1024 |]) ];
    };
    {
      bench =
        fold_row "fold-3x3-pad1-f32-32x64x56x56" ~shape:image conv (fun () ->
            any [| 32; 64; 56; 56; 3; 3 |] ());
      work =
        [ F.Read (bytes [| 32; 64; 56; 56; 3; 3 |]); F.Copy (9 * bytes image) ];
    };
  ]

(* Factorisations and triangular solves. An operand is symmetric, values in
   [-1, 1] off its diagonal and n on it, so positive definite and well
   conditioned; a solve's triangle is its Cholesky factor; eig's operand
   has the same values without the symmetry. A pivoting LU's operand has
   values in [-1, 1] from a multiplicative hash of the position and no
   large diagonal, so that its rows interchange: the operand's values
   repeat every 251 rows, which would make it singular. The floors are the peak of
   fused multiply-adds at LAPACK's flop counts: n³/3 for Cholesky, 2n³/3
   for LU, 4n³/3 for QR and as many to form Q, 4n³/3 for eigh's reduction
   and 2n³ to apply it to the vectors, 8n³/3 for svd's bidiagonalisation
   and 4n³ to apply it to the vectors, 25n³ for eig's Schur form and
   vectors (Golub and Van Loan), n² r for a solve, four real flops a
   complex one. *)

let operand (type v s) ?(dominant = true) ~symmetric (dt : (v, s) D.t) ~batch n
    : (v, s) A.t =
  let v i j = float_of_int ((((i * n) + j) * 7919 mod 251) - 125) /. 125. in
  let hashed k =
    float_of_int (((k * 0x9E3779B1) land 0xFFFFFFFF mod 2001) - 1000) /. 1000.
  in
  let a =
    Array.init (n * n) (fun k ->
        let i = k / n and j = k mod n in
        if not dominant then hashed k
        else if i = j then float_of_int n
        else if symmetric then v (min i j) (max i j)
        else v i j)
  in
  let one = A.of_array D.Float64 [| 1; n; n |] a in
  let src = Option.get (A.move (M.Broadcast [| batch; n; n |]) one) in
  let x = A.create Rig.host dt [| batch; n; n |] in
  cast_into ~dst:x src;
  x

let definite dt ~batch n = operand ~symmetric:true dt ~batch n

(* A routine's row: [call] allocates the destinations of its operands and
   gives the call. *)
let linalg name call (module K : Nx_kernel.S) =
  row name (fun () -> call (module K : Nx_kernel.S)) (fun call -> ok (call ()))

let shape x = A.Layout.shape (A.layout x)
let like x s = A.create Rig.host (A.dtype x) s
let positions_of s = A.create Rig.host D.Int64 s

let linalg_rows =
  let flops (type v s) (dt : (v, s) D.t) n =
    match dt with
    | D.Complex64 -> F.Fma (D.Any f32, 4 * n)
    | D.Complex128 -> F.Fma (D.Any f64, 4 * n)
    | _ -> F.Fma (D.Any dt, n)
  in
  let routine ?(symmetric = true) ?(dominant = true) name call dt ?(batch = 1)
      n ~count =
    let name =
      if batch = 1 then strf "%s-%s-%d" name (short dt) n
      else strf "%s-%s-%dx%d" name (short dt) batch n
    in
    let name = if dominant then name else name ^ "-pivoting" in
    let call k = call k (operand ~dominant ~symmetric dt ~batch n) in
    { bench = linalg name call; work = [ flops dt (batch * count) ] }
  in
  let cholesky dt ?batch n =
    routine "cholesky"
      (fun (module K : Nx_kernel.S) a ->
        let dst = like a (Result.get_ok (Nx_kernel.Shape.cholesky (shape a))) in
        fun () -> K.cholesky Lower ~dst a)
      dt ?batch n ~count:(n * n * n / 3)
  in
  let lu ?(dominant = true) dt ?batch n =
    routine ~symmetric:dominant ~dominant "lu"
      (fun (module K : Nx_kernel.S) a ->
        let y, p, q = Result.get_ok (Nx_kernel.Shape.lu (shape a)) in
        let lu = like a y and pivots = positions_of p and perm = positions_of q in
        fun () -> K.lu ~lu ~pivots ~perm a)
      dt ?batch n ~count:(2 * n * n * n / 3)
  in
  let qr dt ?batch n =
    routine "qr"
      (fun (module K : Nx_kernel.S) a ->
        let q, r = Result.get_ok (Nx_kernel.Shape.qr Reduced (shape a)) in
        let q = like a q and r = like a r in
        fun () -> K.qr ~q ~r a)
      dt ?batch n ~count:(8 * n * n * n / 3)
  in
  let eigh dt ?batch n =
    routine "eigh"
      (fun (module K : Nx_kernel.S) a ->
        let w, v = Result.get_ok (Nx_kernel.Shape.eigh (Vectors ()) (shape a)) in
        let w = like a w and v = like a v in
        fun () -> K.eigh ~w ~v a)
      dt ?batch n ~count:(10 * n * n * n / 3)
  in
  let eigvalsh dt ?batch n =
    routine "eigvalsh"
      (fun (module K : Nx_kernel.S) a ->
        let w, () = Result.get_ok (Nx_kernel.Shape.eigh No_vectors (shape a)) in
        let w = like a w in
        fun () -> K.eigh ~w a)
      dt ?batch n ~count:(4 * n * n * n / 3)
  in
  let svd dt ?batch n =
    routine "svd"
      (fun (module K : Nx_kernel.S) a ->
        let s, (u, vh) =
          Result.get_ok (Nx_kernel.Shape.svd (Vectors Reduced) (shape a))
        in
        let s = like a s and u = like a u and vh = like a vh in
        fun () -> K.svd ~s ~vectors:(~u, ~vh) a)
      dt ?batch n ~count:(20 * n * n * n / 3)
  in
  let svdvals dt ?batch n =
    routine "svdvals"
      (fun (module K : Nx_kernel.S) a ->
        let s, () = Result.get_ok (Nx_kernel.Shape.svd No_vectors (shape a)) in
        let s = like a s in
        fun () -> K.svd ~s a)
      dt ?batch n ~count:(8 * n * n * n / 3)
  in
  (* float64 alone: its results are complex128. *)
  let eig ?batch n =
    routine ~symmetric:false "eig"
      (fun (module K : Nx_kernel.S) a ->
        let w, v = Result.get_ok (Nx_kernel.Shape.eig (Vectors ()) (shape a)) in
        let make s = A.create Rig.host D.Complex128 s in
        let w = make w and v = make v in
        fun () -> K.eig ~w ~v a)
      f64 ?batch n ~count:(25 * n * n * n)
  in
  let solve n r =
    let call (module K : Nx_kernel.S) =
      let l = A.create Rig.host f64 [| 1; n; n |] in
      ok (Nx_cpu.cholesky Lower ~dst:l (definite f64 ~batch:1 n));
      let b = filled f64 [| 1; n; r |] in
      let y =
        Result.get_ok (Nx_kernel.Shape.solve_triangular (shape l) (shape b))
      in
      let dst = like b y in
      fun () ->
        K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false ~dst l b
    in
    {
      bench = linalg (strf "trsm-f64-%dx%d" n r) call;
      work = [ F.Fma (D.Any f64, n * n * r) ];
    }
  in
  let qr_complete batch m n =
    let call (module K : Nx_kernel.S) =
      let a = filled f64 [| batch; m; n |] in
      let q, r = Result.get_ok (Nx_kernel.Shape.qr Complete (shape a)) in
      let q = like a q and r = like a r in
      fun () -> K.qr ~q ~r a
    in
    let k = min m n in
    let r = (m * n * k) - (k * k * k / 3) in
    let q = (2 * m * m * k) - (2 * m * k * k) + (2 * k * k * k / 3) in
    {
      bench = linalg (strf "qr-f64-%dx%dx%d-complete" batch m n) call;
      work = [ F.Fma (D.Any f64, batch * (r + q)) ];
    }
  in
  [
    cholesky f64 32;
    cholesky f64 256;
    cholesky f64 1024;
    cholesky f32 1024;
    cholesky D.Complex128 256;
    cholesky f64 ~batch:1024 32;
    cholesky f64 ~batch:32 32;
    cholesky D.Complex128 ~batch:8 32;
    solve 256 16;
    solve 1024 1024;
    lu f64 256;
    lu f64 1024;
    lu ~dominant:false f64 1024;
    lu f32 1024;
    lu D.Complex128 256;
    lu f64 ~batch:16 32;
    qr f64 256;
    qr f64 1024;
    qr D.Complex128 256;
    qr f32 ~batch:16 32;
    qr_complete 8 64 16;
    eigh f64 64;
    eigh f64 256;
    eigh f64 1024;
    eigh f64 ~batch:64 32;
    eigvalsh f64 1024;
    eigvalsh D.Complex128 512;
    svd f64 64;
    svd f64 256;
    svd f64 1024;
    svd f64 ~batch:64 32;
    svd D.Complex128 512;
    svdvals f64 1024;
    svdvals D.Complex128 512;
    eig 64;
    eig 256;
    eig ~batch:64 32;
  ]

let rows =
  copy_rows @ cast_rows @ apply_rows @ map_rows @ contract_rows @ reduce_rows
  @ index_rows @ sort_rows @ assembly_rows @ linalg_rows

(* A contraction's axes grouped as a kernel library reads them: Contract_view
   of the bf16 4096 call, a and b [1; 4096; 4096] over the batch pair (0, 0)
   and the contracting pair (2, 2). *)
let view =
  let bf16 = D.Bfloat16 in
  row "spec/contract-view-bf16-4096"
    (fun () ->
      let x () = A.create Rig.host bf16 [| 1; 4096; 4096 |] in
      (([| (0, 0) |] : _ iarray), ([| (2, 2) |] : _ iarray), x (), x (), x ()))
    (fun (batch, contracting, dst, a, b) ->
      ignore
        (Nx_kernel.Contract_view.v ~batch ~contracting ~dst ~init:None a b))

(* A backend: its rows, then the floors its support derives from their work. *)
let backend name kernels floors =
  Thumper.group name
    (List.map (fun r -> r.bench kernels) rows
    @ floors (List.concat_map (fun r -> r.work) rows))

let () =
  exit
  @@ Thumper.run "nx_kernels"
       [ backend "cpu" (module Nx_cpu : Nx_kernel.S) F.rows; view ]
