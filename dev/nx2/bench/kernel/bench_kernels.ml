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
let apply ?work name dst run =
  let bench (module K : Nx_kernel.S) =
    row name (fun () -> dst ()) (fun x -> ok (run (module K : Nx_kernel.S) x))
  in
  { bench; work = Option.to_list work }

(* The floor of an elementwise kind over [n] elements. *)
let stream ins inb outb n = F.Stream { ins; inb; outb; n }

let fill n =
  let k = Nx_kernel.Prog.Fill (Nx_kernel.Prog.bits f32 1.5) in
  apply ~work:(stream 0 4 4 n) ("fill-f32-" ^ count n)
    (fun () -> A.create Rig.host f32 [| n |])
    (fun (module K) dst -> K.apply0 k ~dst)

let binary name k dt n rd =
  apply
    ~work:(stream 2 (D.bits dt / 8) (D.bits rd / 8) n)
    name
    (fun () -> (filled dt [| n |], filled dt [| n |], A.create Rig.host rd [| n |]))
    (fun (module K) (x, y, dst) -> K.apply2 k ~dst x y)

(* An add of [n] elements of a dtype whose stream has no floor row. *)
let narrow name dt n =
  apply name
    (fun () -> (filled dt [| n |], filled dt [| n |], A.create Rig.host dt [| n |]))
    (fun (module K) (x, y, dst) -> K.apply2 (Binary Add) ~dst x y)

(* A kind of one operand over [n] elements of [dt]. *)
let unary name u dt n =
  let w = D.bits dt / 8 in
  apply ~work:(stream 1 w w n)
    (strf "%s-%s-%s" name (short dt) (count n))
    (fun () -> (filled dt [| n |], A.create Rig.host dt [| n |]))
    (fun (module K) (x, dst) -> K.apply1 (Unary u) ~dst x)

(* A map of the program [p] over [n] elements of its loads, into fresh
   arrays of its outputs' dtypes. *)
let map ?work name p n =
  let module P = Nx_kernel.Prog in
  let s = Nx_kernel.Spec.map p ~loads:(Array.map (fun _ -> Nx_kernel.Spec.Plain) (P.ins p)) in
  apply ?work name
    (fun () ->
      ( Array.map (fun (D.Any d) -> A.Any (filled d [| n |])) (P.ins p),
        Array.map
          (fun o ->
            let (D.Any d) = P.dtype p o in
            A.Any (A.create Rig.host d [| n |]))
          (P.outs p) ))
    (fun (module K) (ops, dsts) -> K.map s ~dsts ops)

(* Maps: one Add, as apply2 runs it; six float32 nodes over one load, a
   program the interpreter runs block by block in L1; a Mul by a constant,
   a repeated element to its row; and a bfloat16 Add, decoded and encoded
   in its slots. *)
let map_rows =
  let module P = Nx_kernel.Prog in
  let m = mib and f = D.Any f32 in
  let add dt = P.v ~ins:[| dt; dt |] [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |] ~outs:[| 2 |] in
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
    map ~work:(stream 2 4 4 m) "map-add-f32-1M" (add f) m;
    map ~work:(stream 1 4 4 m) "map-6-f32-1M" six m;
    map ~work:(stream 1 4 4 m) "map-scale-f32-1M" scale m;
    map "map-add-bf16-1M" (add (D.Any D.Bfloat16)) m;
  ]

let apply_rows =
  let m = mib in
  [
    fill 1;
    fill m;
    apply ~work:(stream 0 8 8 m) "iota-i64-1M"
      (fun () -> A.create Rig.host D.Int64 [| m |])
      (fun (module K) dst -> K.apply0 (Iota 0) ~dst);
  ]
  @ List.map
      (fun n -> binary ("add-f32-" ^ count n) (Binary Add) f32 n f32)
      [ 1; kib; m ]
  @ [
      apply ~work:(stream 2 4 4 m) "add-f32-1M-transposed"
        (fun () ->
          let x = filled f32 [| 1024; 1024 |] in
          ( Option.get (A.move (M.Permute [| 1; 0 |]) x),
            filled f32 [| 1024; 1024 |],
            A.create Rig.host f32 [| 1024; 1024 |] ))
        (fun (module K) (x, y, dst) -> K.apply2 (Binary Add) ~dst x y);
      binary "add-i8-1M" (Binary Add) D.Int8 m D.Int8;
      (* Narrow dtypes through their carriers, bound by add-f32-1M's and
         add-i8-1M's element rates rather than a floor of their own. *)
      narrow "add-bf16-1M" D.Bfloat16 m;
      narrow "add-e4m3-1M" D.Float8_e4m3fn m;
      narrow "add-i4-1M" D.Int4 m;
      binary "less-f32-1M" (Compare Less) f32 m D.Bool;
      apply ~work:(F.Select { inb = 4; n = m }) "where-f32-1M"
        (fun () ->
          let c = A.create Rig.host D.Bool [| m |] in
          ok
            (Nx_cpu.apply2 (Compare Less) ~dst:c (filled f32 [| m |])
               (filled f32 [| m |]));
          (c, filled f32 [| m |], filled f32 [| m |], A.create Rig.host f32 [| m |]))
        (fun (module K) (c, x, y, dst) -> K.apply3 Where ~dst c x y);
      apply ~work:(stream 3 4 4 m) "fma-f32-1M"
        (fun () ->
          ( filled f32 [| m |],
            filled f32 [| m |],
            filled f32 [| m |],
            A.create Rig.host f32 [| m |] ))
        (fun (module K) (x, y, z, dst) -> K.apply3 Fma ~dst x y z);
    ]
  (* A unary kind that moves bits, then the transcendental kinds. *)
  @ [ unary "neg" Neg f32 m ]
  @ List.concat_map
      (fun (name, u) -> [ unary name u f32 m; unary name u D.Float64 m ])
      Nx_kernel.Prog.
        [
          ("exp", Exp); ("log", Log); ("sin", Sin); ("tanh", Tanh); ("erf", Erf);
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

(* Reductions and scans of one operand by a monoid. A reduction's floor reads
   its operand's bytes; a scan's copies them, since it writes as many. *)

let identity dt = Nx_kernel.Prog.v ~ins:[| dt |] [| In 0 |] ~outs:[| 0 |]

let reduce name m ~axes a (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x as a) = a () in
      let dt = D.Any (A.dtype x) in
      let s = A.Layout.shape (A.layout x) in
      let y =
        Array.of_list
          (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list s))
      in
      let spec =
        Nx_kernel.Spec.reduce (identity dt) ~loads:[| Plain |] ~axes
          [| (Monoid m, 0, dt) |]
      in
      (spec, a, A.Any (A.create Rig.host (A.dtype x) y)))
    (fun (spec, a, dst) -> ok (K.reduce spec ~dsts:[| dst |] [| a |]))

let scan name m ~axis a (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x as a) = a () in
      let dt = D.Any (A.dtype x) in
      let spec =
        Nx_kernel.Spec.scan (identity dt) ~loads:[| Plain |] ~axis
          (Monoid m, 0, dt)
      in
      let s = A.Layout.shape (A.layout x) in
      (spec, a, A.Any (A.create Rig.host (A.dtype x) s)))
    (fun (spec, a, dst) -> ok (K.scan spec ~dsts:[| dst |] [| a |]))

let fold_rows =
  let bytes dt s = D.bits dt / 8 * Array.fold_left ( * ) 1 s in
  let any dt s () = A.Any (filled dt s) in
  let along name dt s axes m =
    { bench = reduce name m ~axes (any dt s); work = [ F.Read (bytes dt s) ] }
  in
  let running name dt s axis =
    { bench = scan name Sum ~axis (any dt s); work = [ F.Copy (bytes dt s) ] }
  in
  let m = mib in
  [
    along "reduce-sum-f32-1M" f32 [| m |] [| 0 |] Sum;
    along "reduce-sum-f32-16M" f32 [| 16 * m |] [| 0 |] Sum;
    along "reduce-sum-f64-16M" f64 [| 16 * m |] [| 0 |] Sum;
    along "reduce-sum-i32-16M" D.Int32 [| 16 * m |] [| 0 |] Sum;
    along "reduce-max-f32-16M" f32 [| 16 * m |] [| 0 |] Max;
    (* Along an axis whose outputs lie side by side: rows of outputs. *)
    along "reduce-sum-f32-1024x1024-axis0" f32 [| 1024; 1024 |] [| 0 |] Sum;
    along "reduce-sum-f32-4096x4096-axis0" f32 [| 4096; 4096 |] [| 0 |] Sum;
    (* Along the innermost axis: each output's run into lanes. *)
    along "reduce-sum-f32-1024x1024-axis1" f32 [| 1024; 1024 |] [| 1 |] Sum;
    along "reduce-sum-f32-4096x4096-axis1" f32 [| 4096; 4096 |] [| 1 |] Sum;
    (* Four terms an output. *)
    along "reduce-sum-f32-1Mx4-axis1" f32 [| m; 4 |] [| 1 |] Sum;
    (* The maximum of each 2x2 window of a pooling layer. *)
    {
      bench =
        reduce "reduce-max-f32-2x2-windows-32x16x26x26" Max ~axes:[| 4; 5 |]
          (fun () ->
            let w axis = { M.axis; size = 2; step = 2; dilation = 1 } in
            let a = filled f32 [| 32; 16; 26; 26 |] in
            A.Any (Option.get (A.move (M.Window [| w 2; w 3 |]) a)));
      work = [ F.Read (bytes f32 [| 32; 16; 26; 26 |]) ];
    };
    running "scan-sum-f64-5M" f64 [| 5 * m |] 0;
    running "scan-sum-f32-4096x1024-axis1" f32 [| 4096; 1024 |] 1;
    running "scan-sum-f32-1024x4096-axis0" f32 [| 1024; 4096 |] 0;
  ]

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
      let s = Nx_kernel.Spec.gather ~axis in
      fun () -> ok (K.gather s ~dst idx x))
    (fun call -> call ())

let scatter_row name c ~axis into idx u (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any into) = into () in
      let u = A.expect (A.dtype into) (u ()) and idx = idx () in
      let dst =
        A.create Rig.host (A.dtype into) (A.Layout.shape (A.layout into))
      in
      let s = Nx_kernel.Spec.scatter c ~unique:false ~axis in
      fun () -> ok (K.scatter s ~dst ~into idx u))
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
   its key's bytes. *)

let sort_row name ~axis ~descending ~k x (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x) = x () in
      let y = Array.copy (A.Layout.shape (A.layout x)) in
      Option.iter (fun k -> y.(axis) <- k) k;
      let values = A.create Rig.host (A.dtype x) y in
      let positions = A.create Rig.host D.Int64 y in
      let s = Nx_kernel.Spec.sort ~axis ~descending ~k in
      fun () -> ok (K.sort s ~values ~positions x))
    (fun call -> call ())

let sort_rows =
  let m = mib in
  let any dt s () = A.Any (filled dt s) in
  let sort name ?(descending = false) ?k ~axis x ~passes ~n =
    {
      bench = sort_row name ~axis ~descending ~k x;
      work = [ F.Copy (passes * n * 16) ];
    }
  in
  [
    sort "sort-f32-1M" ~axis:0 (any f32 [| m |]) ~passes:4 ~n:m;
    sort "sort-i64-1M" ~axis:0 (any D.Int64 [| m |]) ~passes:8 ~n:m;
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
      let pieces = pieces () in
      let (A.Any p) = pieces.(0) in
      let dt = A.dtype p in
      let pieces = Array.map (A.expect dt) pieces in
      let dst = A.create Rig.host dt shape in
      let s = Nx_kernel.Spec.assemble ~shape ~fill regions in
      fun () -> ok (K.assemble s ~dst pieces))
    (fun call -> call ())

let fold_row name ~shape pad x (module K : Nx_kernel.S) =
  row name
    (fun () ->
      let (A.Any x) = x () in
      let dst = A.create Rig.host (A.dtype x) shape in
      let s = Nx_kernel.Spec.fold ~shape pad in
      fun () -> ok (K.fold s ~dst x))
    (fun call -> call ())

let assembly_rows =
  let zero = Nx_kernel.Prog.bits f32 0. in
  let any s () = A.Any (filled f32 s) in
  let whole d = { M.start = 0; count = d; step = 1 } in
  let n = 512 in
  let conv =
    let w axis = { M.axis; size = 3; step = 1; dilation = 1 } in
    {
      Nx_kernel.Spec.lo = [| 0; 0; 1; 1 |];
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
          ~fill:zero
          [|
            [| { M.start = 0; count = n; step = 1 }; whole n |];
            [| { M.start = n; count = n; step = 1 }; whole n |];
          |]
          (fun () -> [| any [| n; n |] (); any [| n; n |] () |]);
      work = [ F.Copy (bytes [| 2 * n; n |]) ];
    };
    {
      bench =
        assemble_row "assemble-pad-1-of-1022x1022-f32" ~shape:[| 1024; 1024 |]
          ~fill:zero
          [|
            [|
              { M.start = 1; count = 1022; step = 1 };
              { M.start = 1; count = 1022; step = 1 };
            |];
          |]
          (fun () -> [| any [| 1022; 1022 |] () |]);
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

let rows =
  copy_rows @ cast_rows @ apply_rows @ map_rows @ contract_rows @ fold_rows
  @ index_rows @ sort_rows @ assembly_rows

(* A contraction's axes grouped as a GPU planner reads them: Spec.Contract_view
   of the bf16 4096 call, a and b [1; 4096; 4096] over the batch pair (0, 0)
   and the contracting pair (2, 2). It allocates nothing. *)
let view =
  let bf16 = D.Bfloat16 in
  row "spec/contract-view-bf16-4096"
    (fun () ->
      let spec =
        Nx_kernel.Spec.contract ~batch:[| (0, 0) |] ~contracting:[| (2, 2) |]
          ~acc:(D.Any f32) ~out:(D.Any bf16) ~init:false
      in
      let x () = A.Any (A.create Rig.host bf16 [| 1; 4096; 4096 |]) in
      (Nx_kernel.Spec.Contract_view.make (), spec, x (), [| x (); x () |]))
    (fun (v, spec, dst, ops) ->
      ignore (Nx_kernel.Spec.Contract_view.fill v spec ~dst ops))

(* A backend: its rows, then the floors its support derives from their work. *)
let backend name kernels floors =
  Thumper.group name
    (List.map (fun r -> r.bench kernels) rows
    @ floors (List.concat_map (fun r -> r.work) rows))

let () =
  exit
  @@ Thumper.run "nx_kernels"
       [ backend "cpu" (module Nx_cpu : Nx_kernel.S) F.rows; view ]
