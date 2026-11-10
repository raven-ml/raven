(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The CUDA rows: each prepares its operands on the GPU, checks its result once,
   and gives the run to time, with the floor kernel that moves its bytes in its
   directions where it is memory-bound. A row of nx.cuda's kernels has a spec,
   which bench_cuda.exe compare hands to compare.py, so that PyTorch times
   the same computation. *)

module S = Nx_cuda_support
module D = Nx_array.Dtype
module A = Nx_array

let strf = Printf.sprintf

type work = Bytes of int | Flops of float
type prepared = { run : S.run; work : work; floor : S.run option }

(* Specs *)

(* The axis of an operand of a contraction that is contiguous: its
   contracted one ([`K]) or its free one ([`Free]). *)
type layout = [ `K | `Free ]

(* A computation of nx.cuda's: y[z, i, j] = Σ_k a[z, i, k] · b[z, j, k] with
   a [m; k] and b [n; k] of [dtype] per batch element, summed in [acc], y of
   [out]; a reduction of [dtype] by [monoid] along [axes] of [shape]; or a
   scan along [axis]. *)
type spec =
  | Contract of {
      dtype : D.any;
      acc : D.any;
      out : D.any;
      batch : int;
      m : int;
      n : int;
      k : int;
      la : layout;
      lb : layout;
    }
  | Reduce of {
      dtype : D.any;
      monoid : Nx_kernel.monoid;
      shape : int array;
      axes : int list;
    }
  | Scan of {
      dtype : D.any;
      monoid : Nx_kernel.monoid;
      shape : int array;
      axis : int;
    }
  | Loop of { op : string; dtype : D.any; shape : int array }
      (** The loop compare.py knows by [op], over [dtype] operands of
          [shape]. *)
  | Sort of {
      dtype : D.any;
      shape : int array;
      axis : int;
      descending : bool;
      k : int option;
      range : int option;
    }
      (** A stable sort along [axis], keeping the first [k] along it, of
          integers in \[0, [range]) or floats in \[-1, 1). *)
  | Unique of { n : int; w : int; range : int option }
      (** A numbering of the rows of int64 keys of [[n; w]], in \[0,
          [range]). *)
  | Move of { op : string; dtype : D.any }
      (** The gather, scatter, assembly or fold compare.py knows by [op] (the
          moves' ROWS.md), of [dtype] elements. *)
  | Fft of { kind : string; dtype : D.any; shape : int array; axes : int list }
      (** A forward transform ["c2c"], ["r2c"] or ["c2r"] (the inverse of a
          real one) along [axes] of an operand of the complex [dtype] or of
          its real precision, of [shape], a [c2r]'s that of its result. *)
  | Matrix of {
      routine : string;
      dtype : D.any;
      shape : int array;
      mode : string;
    }
      (** A matrix routine compare.py knows by [routine] on [dtype] operands
          of [shape] (batch, then the matrix's extents; a solve's right-hand
          sides last), with [mode] ["values"], ["vectors"] or ["complete"]
          for its vectors. *)
  | Attention of {
      dtype : D.any;
      b : int;
      hq : int;
      hk : int;
      tq : int;
      tk : int;
      d : int;
      mask : string;
    }
      (** [Nx.attention] of a query [[b; hq; tq; d]] of [dtype] over dense
          keys and values [[b; hk; tk; d]], with [mask] ["full"],
          ["causal"] ([Nx.causal], at [tq = tk]) or ["lengths"]: batch
          element [i]'s keys cut at [(i + 1) tk / b]. *)

let dname (D.Any d) = D.name d

let monoid_name : Nx_kernel.monoid -> string = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"

let ints sep xs = String.concat sep (List.map string_of_int xs)
let dims shape = ints "x" (Array.to_list shape)
let layout_name = function `K -> "k" | `Free -> "f"

(* [s] as compare.py reads it: its kind, then its fields separated by
   spaces, a shape's extents by [x] and axes by [,]:

   contract DTYPE ACC OUT BATCH M N K LALB   (LALB: kk, kf, fk or ff)
   reduce DTYPE MONOID SHAPE AXES
   scan DTYPE MONOID SHAPE AXIS
   loop OP DTYPE SHAPE
   sort DTYPE SHAPE AXIS DESCENDING K RANGE   (K -1: every element;
                                               RANGE 0: the dtype's)
   unique N W RANGE
   move OP DTYPE
   fft KIND DTYPE SHAPE AXES
   matrix ROUTINE DTYPE SHAPE MODE
   attention DTYPE B HQ HK TQ TK D MASK *)
let spec_text = function
  | Contract c ->
      strf "contract %s %s %s %d %d %d %d %s%s" (dname c.dtype) (dname c.acc)
        (dname c.out) c.batch c.m c.n c.k (layout_name c.la) (layout_name c.lb)
  | Reduce r ->
      strf "reduce %s %s %s %s" (dname r.dtype) (monoid_name r.monoid)
        (dims r.shape) (ints "," r.axes)
  | Scan s ->
      strf "scan %s %s %s %d" (dname s.dtype) (monoid_name s.monoid)
        (dims s.shape) s.axis
  | Loop l -> strf "loop %s %s %s" l.op (dname l.dtype) (dims l.shape)
  | Sort o ->
      strf "sort %s %s %d %b %d %d" (dname o.dtype) (dims o.shape) o.axis
        o.descending
        (Option.value o.k ~default:(-1))
        (Option.value o.range ~default:0)
  | Unique u -> strf "unique %d %d %d" u.n u.w (Option.value u.range ~default:0)
  | Move m -> strf "move %s %s" m.op (dname m.dtype)
  | Fft f ->
      strf "fft %s %s %s %s" f.kind (dname f.dtype) (dims f.shape)
        (ints "," f.axes)
  | Matrix m ->
      strf "matrix %s %s %s %s" m.routine (dname m.dtype) (dims m.shape) m.mode
  | Attention a ->
      strf "attention %s %d %d %d %d %d %d %s" (dname a.dtype) a.b a.hq a.hk
        a.tq a.tk a.d a.mask

type row = { name : string; make : S.gpu -> prepared; spec : spec option }

(* [n] as a row names a size: 64K, 16M. *)
let size n =
  if n >= 1 lsl 20 then strf "%dM" (n lsr 20) else strf "%dK" (n lsr 10)

(* Floors *)

(* A row of [work] whose run [run] makes, with no floor. *)
let row name work run =
  { name; make = (fun g -> { run = run g; work; floor = None }); spec = None }

let harness g launches = S.record (S.harness g) launches

let copy n =
  row (strf "floor/copy-%sB" (size n)) (Bytes (2 * n)) @@ fun g ->
  harness g [ S.floor_copy ~ins:[ S.buffer g n ] ~out:(S.buffer g n) ]

(* The driver's copy, which a copy floor should not trail. *)
let copy_driver n =
  row (strf "floor/copy-driver-%sB" (size n)) (Bytes (2 * n)) @@ fun g ->
  S.driver_copy ~src:(S.buffer g n) ~dst:(S.buffer g n)

let read n =
  row (strf "floor/read-%sB" (size n)) (Bytes n) @@ fun g ->
  harness g [ S.floor_read g (S.buffer g n) ]

let launch =
  row "floor/launch" (Bytes 0) (fun g ->
      harness g [ S.launch "empty" ~grid:(1, 1, 1) ~block:1 [] ])

(* A peak row: 4 blocks of 8 warps per multiprocessor, each warp [rounds] rounds
   of 16 instructions of [flops] each, per lane for an fma. *)
let peak name ~rounds ~flops =
  let kernel = String.map (fun c -> if c = '-' then '_' else c) name in
  let make g =
    let blocks = 4 * S.sms g in
    let o = S.buffer g (8 * blocks * 8) in
    let total = float (blocks * 8 * rounds * 16) *. flops in
    let launch =
      S.launch kernel ~grid:(blocks, 1, 1) ~block:256 [ A o; W rounds ]
    in
    { run = harness g [ launch ]; work = Flops total; floor = None }
  in
  { name = "floor/" ^ name ^ "-peak"; make; spec = None }

let peaks =
  [
    peak "mma-bf16" ~rounds:2000 ~flops:4096.;
    peak "mma-f16" ~rounds:2000 ~flops:4096.;
    peak "mma-s8" ~rounds:2000 ~flops:8192.;
    peak "fma-f32" ~rounds:20000 ~flops:64.;
    peak "fma-f64" ~rounds:500 ~flops:64.;
  ]

(* Copies at L2 sizes and from DRAM; reads off the L2's 64 MB edge. *)
let floors =
  let copies = [ 1 lsl 18; 1 lsl 22; 1 lsl 26; 1 lsl 28 ] in
  [ launch ]
  @ List.concat_map (fun n -> [ copy n; copy_driver n ]) copies
  @ List.map read [ 1 lsl 22; 1 lsl 25; 1 lsl 28 ]
  @ peaks

(* Contract *)

(* The operands of y[z, i, j] = Σ_k a[z, i, k] · b[z, j, k], a [m; k] and b [n;
   k] per batch element of [d], in the layouts [la] and [lb] ([`K] or [`Free]
   axis contiguous), y [m; n] of [o], and the plan's run. *)
let plan g ?(batch = 1) ?(la = `K) ?(lb = `K) (D.Any d) (D.Any c) (D.Any o) m n
    k =
  let operand (type v s) (d : (v, s) D.t) rows cols layout : S.operand =
    let buffer = S.buffer g (D.bits d / 8 * batch * rows * cols) in
    let draw = if D.is D.Float d then S.Uniform else S.Small in
    (* a complex operand's parts drawn as floats *)
    (match d with
    | D.Complex64 -> S.generate g buffer D.Float32 draw ~seed:rows
    | D.Complex128 -> S.generate g buffer D.Float64 draw ~seed:rows
    | _ -> S.generate g buffer d draw ~seed:rows);
    let s = if layout = `K then [| cols; 1 |] else [| 1; rows |] in
    let strides = Array.append [| rows * cols |] s in
    let shape = [| batch; rows; cols |] in
    { buffer; dtype = D.code d; shape; strides; first = 0 }
  in
  let a = operand d m k la and b = operand d n k lb and y = operand o m n `K in
  let contracting = [ (2, 2) ] and acc = D.code c in
  match S.contract g ~a ~b ~y ~batch:[ (0, 0) ] ~contracting ~acc () with
  | Some run -> (a, b, y, run)
  | None -> failwith "the plan declines"

(* The contraction, checked once at 512 outputs against the error
   bound, but for complex operands, which the reference does not take. A
   product with m <= 16 is memory-bound: its floor reads its operands;
   complex products count their real flops. *)
let contract ?(batch = 1) ?(la = `K) ?(lb = `K) (D.Any d as dt) (D.Any c as acc)
    out m n k =
  let name =
    strf "contract/%s-%s%dx%dx%d%s" (D.name d)
      (if batch > 1 then strf "%dx" batch else "")
      m n k
      (if la = `K && lb = `K then "" else "-" ^ layout_name la ^ layout_name lb)
  in
  let make g =
    let a, b, y, run = plan g ~batch ~la ~lb dt acc out m n k in
    S.run g run;
    let view (x : S.operand) =
      {
        Nx_gpu_ref.bytes = S.read x.buffer;
        dtype = x.dtype;
        strides = x.strides;
      }
    in
    let complex = D.is D.Complex d in
    if not complex then begin
      let r =
        Nx_gpu_ref.contract ~a:(view a) ~b:(view b) ~y:(view y) ~batch ~m ~n ~k
          ~acc:(D.code c) ~samples:512 ()
      in
      if r.wrong > 0 || not (r.worst <= 1.) then
        failwith (strf "%s: %g of the bound, %d wrong" name r.worst r.wrong)
    end;
    let bytes = D.bits d / 8 * batch * (m + n) * k in
    if m > 16 then
      {
        run;
        work =
          Flops
            ((if complex then 8. else 2.) *. float (batch * m * n) *. float k);
        floor = None;
      }
    else
      let operands = S.buffer g (16 * ((bytes + 15) / 16)) in
      let floor = harness g [ S.floor_read g operands ] in
      { run; work = Bytes bytes; floor = Some floor }
  in
  let spec = Contract { dtype = dt; acc; out; batch; m; n; k; la; lb } in
  { name; make; spec = Some spec }

let contracts =
  let open D in
  let bf16 = Any Bfloat16 and f16 = Any Float16 and f32 = Any Float32 in
  let i8 = Any Int8 and i32 = Any Int32 and f64 = Any Float64 in
  let i16 = Any Int16 and i64 = Any Int64 and e4m3 = Any Float8_e4m3fn in
  let squares dt =
    List.map
      (fun s -> contract dt f32 dt s s s)
      [ 256; 512; 1024; 2048; 4096; 8192 ]
  in
  let layouts dt =
    List.map
      (fun (la, lb) -> contract ~la ~lb dt f32 dt 4096 4096 4096)
      [ (`K, `Free); (`Free, `K); (`Free, `Free) ]
  in
  (* Many small products, each one batch element *)
  let batched dt acc = contract ~batch:256 dt acc dt 64 64 64 in
  List.concat_map squares [ bf16; f16; f32 ]
  @ List.concat_map layouts [ bf16; f16; f32 ]
  @ List.map (fun s -> contract f64 f64 f64 s s s) [ 256; 512; 2048 ]
  @ [ batched bf16 f32; batched f16 f32; batched f32 f32; batched f64 f64 ]
  @ [
      contract i8 i32 i32 4096 4096 4096;
      contract bf16 f32 bf16 512 5120 2880;
      contract bf16 f32 bf16 512 2880 4096;
      contract bf16 f32 bf16 512 201088 2880;
      contract bf16 f32 bf16 1 5120 2880;
      contract f16 f32 f16 1 5120 2880;
      (* float8 weights, packed to bfloat16 *)
      contract e4m3 f32 bf16 1 5120 2880;
      contract bf16 f32 bf16 1 201088 2880;
      contract bf16 f32 bf16 4096 14336 4096;
      (* long sums, k cut into 2 ranges: few outputs in chain order; a full
         grid in chain order, whose partials (128 MiB) pass the kept
         workspace; many rows of one column in lane order *)
      contract bf16 f32 bf16 256 256 65536;
      contract bf16 f32 bf16 4096 4096 65536;
      contract bf16 f32 bf16 4096 1 4096;
      contract ~batch:64 bf16 f32 bf16 512 512 512;
      contract f32 f32 f32 1 5120 2880;
      (* float64 and integer sums, which the core computes *)
      contract f64 f64 f64 1024 1024 1024;
      contract f64 f64 f64 1 5120 2880;
      contract i16 i64 i64 1024 1024 1024;
      contract i32 i32 i32 1024 1024 1024;
      contract i8 i32 i32 1 5120 2880;
      contract i16 i64 i64 1 5120 2880;
      contract i32 i32 i32 1 5120 2880;
      (* Chain order at 5 to 16 rows, on the SIMT kernels *)
      contract bf16 f32 bf16 8 5120 2880;
      contract f16 f32 f16 8 5120 2880;
      contract f32 f32 f32 8 5120 2880;
      contract f32 f32 f32 16 5120 2880;
      contract f64 f64 f64 8 5120 2880;
      contract i32 i32 i32 8 5120 2880;
      (* A @ x, in the order nx.cuda chooses: lane, on the skinny kernels. *)
      contract f32 f32 f32 20480 1 2880;
    ]

(* Reduce and scan *)

(* Element [i] of [s], the bytes of a [dt] array, as a float, exact: the
   rows reduce float32, float64 and int32. *)
let value (D.Any dt) =
  match dt with
  | D.Float32 -> fun s i -> Int32.float_of_bits (String.get_int32_le s (4 * i))
  | D.Float64 -> fun s i -> Int64.float_of_bits (String.get_int64_le s (8 * i))
  | D.Int32 -> fun s i -> Int32.to_float (String.get_int32_le s (4 * i))
  | D.Bfloat16 ->
      fun s i ->
        Int32.float_of_bits
          (Int32.shift_left (Int32.of_int (String.get_uint16_le s (2 * i))) 16)
  | _ -> invalid_arg ("Rows.value: " ^ D.name dt)

(* A reduction's terms so far, as a check folds them: their sum, exact for
   int32 terms, of magnitudes, and extreme. *)
type acc = {
  mutable terms : int;
  mutable sum : float;
  mutable isum : int;
  mutable mag : float;
  mutable extreme : float;
}

let acc (m : Nx_kernel.monoid) =
  let extreme =
    match m with Max -> Float.neg_infinity | Min -> Float.infinity | _ -> 0.
  in
  { terms = 0; sum = 0.; isum = 0; mag = 0.; extreme }

let add (m : Nx_kernel.monoid) a v =
  a.terms <- a.terms + 1;
  a.sum <- a.sum +. v;
  a.isum <- a.isum + Float.to_int v;
  a.mag <- a.mag +. Float.abs v;
  a.extreme <-
    (match m with
    | Max -> Float.max a.extreme v
    | Min -> Float.min a.extreme v
    | _ -> 0.)

(* Whether [y] is [a]'s terms reduced by [m] in [dt]: an extreme exactly, an
   int32 sum wrapped, a float sum within any order's error, [terms] units of
   roundoff of the magnitudes' sum, and the check's own in double, then a
   bfloat16 one's rounding into its destination. *)
let agrees (D.Any dt) (m : Nx_kernel.monoid) a y =
  match m with
  | Max | Min -> y = a.extreme
  | Prod -> invalid_arg "Rows.agrees: a product"
  | Sum when D.is D.Float dt ->
      let u = match dt with D.Float64 -> 0x1p-53 | _ -> 0x1p-24 in
      let store =
        match dt with D.Bfloat16 -> 0x1p-8 *. Float.abs y | _ -> 0.
      in
      Float.abs (y -. a.sum) <= (3. *. float a.terms *. u *. a.mag) +. store
  | Sum -> y = Int32.to_float (Int32.of_int a.isum)

(* C strides of [shape], in elements. *)
let strides shape =
  let n = Array.length shape in
  let s = Array.make n 1 in
  for i = n - 2 downto 0 do
    s.(i) <- s.(i + 1) * shape.(i + 1)
  done;
  s

let numel shape = Array.fold_left ( * ) 1 shape

(* An operand of [shape] of [dt] on [g], laid out in C order, drawn if
   [draw]. *)
let operand g (D.Any d) shape ~draw : S.operand =
  let buffer = S.buffer g (Int.max 1 (D.bits d / 8 * numel shape)) in
  if draw then S.generate g buffer d S.Uniform ~seed:(numel shape);
  { buffer; dtype = D.code d; shape; strides = strides shape; first = 0 }

(* [n] of [outputs] a check samples: the first, the last and some
   between. *)
let samples_of outputs n =
  let n = Int.min outputs n in
  List.sort_uniq compare
    (List.init n (fun i -> if n = 1 then 0 else i * (outputs - 1) / (n - 1)))

(* The terms of output [o] of the reduction of [shape] along [axes], as
   offsets into x in C order, given to [f]. *)
let iter_terms shape axes o f =
  let rank = Array.length shape in
  let st = strides shape in
  let kept =
    List.filter (fun a -> not (List.mem a axes)) (List.init rank Fun.id)
  in
  let base = ref 0 and rest = ref o in
  List.iter
    (fun a ->
      let inner =
        List.fold_left (fun p b -> if b > a then p * shape.(b) else p) 1 kept
      in
      base := !base + (!rest / inner * st.(a));
      rest := !rest mod inner)
    kept;
  let rec go = function
    | [] -> fun off -> f off
    | a :: more ->
        fun off ->
          for i = 0 to shape.(a) - 1 do
            go more (off + (i * st.(a)))
          done
  in
  go axes !base

let reduce dt monoid shape axes =
  let name =
    strf "reduce/%s-%s-%s-axes%s" (dname dt) (monoid_name monoid) (dims shape)
      (ints "." axes)
  in
  let make g =
    let x = operand g dt shape ~draw:true in
    let yshape =
      Array.of_list
        (List.filteri (fun i _ -> not (List.mem i axes)) (Array.to_list shape))
    in
    let y = operand g dt yshape ~draw:false in
    let run =
      match S.reduce g monoid ~axes ~x ~y with
      | Some run -> run
      | None -> failwith (name ^ ": nx.cuda declines it")
    in
    S.run g run;
    let xs = S.read x.buffer and ys = S.read y.buffer in
    let get = value dt in
    List.iter
      (fun o ->
        let a = acc monoid in
        iter_terms shape axes o (fun i -> add monoid a (get xs i));
        let got = get ys o in
        if not (agrees dt monoid a got) then
          failwith (strf "%s: output %d is %h" name o got))
      (samples_of (numel yshape) 64);
    let bytes = String.length xs in
    let floor = harness g [ S.floor_read g x.buffer ] in
    { run; work = Bytes bytes; floor = Some floor }
  in
  { name; make; spec = Some (Reduce { dtype = dt; monoid; shape; axes }) }

(* A scan writes as many bytes as it reads: its floor copies. *)
let scan dt monoid shape axis =
  let name =
    strf "scan/%s-%s-%s-axis%d" (dname dt) (monoid_name monoid) (dims shape)
      axis
  in
  let make g =
    let x = operand g dt shape ~draw:true
    and y = operand g dt shape ~draw:false in
    let run =
      match S.scan g monoid ~axis ~x ~y with
      | Some run -> run
      | None -> failwith (name ^ ": nx.cuda declines it")
    in
    S.run g run;
    let xs = S.read x.buffer and ys = S.read y.buffer in
    let get = value dt in
    let lines = numel shape / shape.(axis) in
    (* Line [l]: the elements along [axis] at the [l]th index of the other
       axes, each checked against the running reduction. *)
    List.iter
      (fun l ->
        let a = acc monoid in
        iter_terms shape [ axis ] l (fun off ->
            add monoid a (get xs off);
            let got = get ys off in
            if not (agrees dt monoid a got) then
              failwith (strf "%s: element %d is %h" name off got)))
      (samples_of lines 8);
    let bytes = String.length xs in
    let floor =
      harness g [ S.floor_copy ~ins:[ x.buffer ] ~out:(S.buffer g bytes) ]
    in
    { run; work = Bytes (2 * bytes); floor = Some floor }
  in
  { name; make; spec = Some (Scan { dtype = dt; monoid; shape; axis }) }

(* Each at 4 MiB, which the L2 holds, and 256 MiB, which it does not:
   whole-array, row and column sums and a row maximum, float64 and int32
   whole-array sums, sums of rows of 64 terms, and scans along rows, along
   columns and over a whole vector. *)
let reductions =
  let open D in
  let f32 = Any Float32 and f64 = Any Float64 and i32 = Any Int32 in
  let sized (small, large) f = [ f small; f large ] in
  let square = ([| 1024; 1024 |], [| 8192; 8192 |]) in
  List.concat
    [
      sized square (fun s -> reduce f32 Sum s [ 0; 1 ]);
      sized square (fun s -> reduce f32 Sum s [ 1 ]);
      sized square (fun s -> reduce f32 Sum s [ 0 ]);
      sized square (fun s -> reduce f32 Max s [ 1 ]);
      sized
        ([| 512; 1024 |], [| 4096; 8192 |])
        (fun s -> reduce f64 Sum s [ 0; 1 ]);
      sized square (fun s -> reduce i32 Sum s [ 0; 1 ]);
      sized
        ([| 16384; 64 |], [| 1048576; 64 |])
        (fun s -> reduce f32 Sum s [ 1 ]);
      sized square (fun s -> scan f32 Sum s 1);
      sized square (fun s -> scan f32 Sum s 0);
      sized ([| 1048576 |], [| 67108864 |]) (fun s -> scan f32 Sum s 0);
    ]

(* Loops *)

module K = Nx_kernel
module P = Nx_kernel.Prog

(* What a loop row computes from its program's outputs: a map, a reduction
   along [axes] by [rs], or a scan along [axis]. *)
type loop =
  | Map
  | Reduce of { axes : int list; rs : (K.reduction * int) list }
  | Scan of { axis : int; r : K.reduction * int }

(* [l] of [p] over [loads] into [dsts], a loop of shape [shape], by the
   kernels [Kn]. *)
let apply (module Kn : K.S) p l ~shape ~dsts loads =
  let dsts = Iarray.of_list dsts and loads = Iarray.of_list loads in
  match l with
  | Map -> Kn.map p ~dsts loads
  | Reduce { axes; rs } ->
      Kn.reduce p ~shape:(Iarray.of_array shape) ~axes:(Iarray.of_list axes)
        (Iarray.of_list rs) ~dsts loads
  | Scan { axis; r } -> Kn.scan p ~axis r ~dsts loads

let loads_arrays loads =
  List.map
    (function K.Plain x -> A.Any x | K.Padded { x; _ } -> A.Any x)
    loads

let host_load : K.load -> K.load = function
  | Plain x -> Plain (A.to_device Rig.host x)
  | Padded { x; fill; pad } -> Padded { x = A.to_device Rig.host x; fill; pad }

let fresh_like d (A.Any a) =
  A.Any (A.create d (A.dtype a) (A.Layout.shape (A.layout a)))

let bytes_of (A.Any a) = S.read (A.buffer a)

(* The first element at which [x] and [y], the bytes of two [dt] arrays,
   differ. *)
let first_difference (A.Any a) x y =
  let w = Int.max 1 (D.bits (A.dtype a) / 8) in
  let rec go i =
    if String.sub x i w <> String.sub y i w then i / w else go (i + w)
  in
  go 0

(* [p]'s one output over [loads], a loop of [shape], mapped by nx.cuda
   into a fresh array on [d]. *)
let mapped d p ~shape loads =
  let (D.Any od) = P.dtype p (Iarray.get p.outs 0) in
  let t = A.Any (A.create d od (Iarray.of_array shape)) in
  match apply (module Nx_cuda) p Map ~shape ~dsts:[ t ] loads with
  | A.Done -> t
  | _ -> failwith "Rows.mapped: nx.cuda does not map it"

(* A loop row: its program's [l] over [loads] into [dsts], checked once
   against nx.cpu's bits, which nx.cuda's loops promise; a scan in
   nx.cuda's own order ([own]) against nx.cuda's scan of the output mapped
   first, whose bits nx_cuda.mli promises. [bytes] are the bytes it moves,
   [floor] the floor kernel that moves them. *)
let loop_run ?(own = false) g name p l ~shape ~dsts loads =
  let arrays = dsts @ loads_arrays loads in
  let run =
    match
      S.kernel_call ~what:name ~launches:S.loop_launches arrays (fun () ->
          apply (module Nx_cuda) p l ~shape ~dsts loads)
    with
    | Some run -> run
    | None -> failwith (name ^ ": nx.cuda declines it")
  in
  S.run g run;
  let who, want =
    if own then
      let d = S.device g in
      let (A.Any t) = mapped d p ~shape loads in
      let id =
        P.v
          ~ins:(Iarray.of_list [ D.Any (A.dtype t) ])
          (Iarray.of_list [ P.In 0 ])
          ~outs:(Iarray.of_list [ 0 ])
      in
      let want = List.map (fresh_like d) dsts in
      match apply (module Nx_cuda) id l ~shape ~dsts:want [ K.Plain t ] with
      | A.Done -> ("its mapped output's", want)
      | _ -> failwith (name ^ ": nx.cuda declines its mapped output")
    else
      let host = List.map (fresh_like Rig.host) dsts in
      match
        apply (module Nx_cpu) p l ~shape ~dsts:host (List.map host_load loads)
      with
      | A.Done -> ("nx.cpu's", host)
      | _ -> failwith (name ^ ": nx.cpu does not compute it")
  in
  List.iteri
    (fun k (d, h) ->
      let x = bytes_of d and y = bytes_of h in
      if x <> y then
        failwith
          (strf "%s: destination %d differs from %s at element %d" name k who
             (first_difference d x y)))
    (List.combine dsts want);
  run

(* A fresh C-ordered [dt] array of [shape] on [g], drawn from [seed] if
   given. *)
let fresh g ?seed (D.Any d) shape =
  let a = A.create (S.device g) d (Iarray.of_array shape) in
  Option.iter (fun seed -> S.generate g (A.buffer a) d S.Uniform ~seed) seed;
  A.Any a

let plain (A.Any a) = K.Plain a
let buf (A.Any a) = A.buffer a
let any_dtype (A.Any a) = D.Any (A.dtype a)
let ins xs = Iarray.of_list (List.map any_dtype xs)

(* [view a ~strides shape]: [a]'s buffer through another layout. *)
let view (A.Any a) ~strides shape =
  A.Any
    (A.v (A.dtype a)
       (A.Layout.v ~strides:(Iarray.of_array strides) (Iarray.of_array shape))
       (A.buffer a))

let prog xs nodes out =
  P.v ~ins:(ins xs) (Iarray.of_list nodes) ~outs:(Iarray.of_list [ out ])

let f32_const v = P.Const (D.Any D.Float32, P.bits D.Float32 v)

(* The k12 rows (ROWS.md of the loops' plan): each op is a program and a
   loop over operands of [shape], with the bytes it moves and its floor. *)
let loop_op op (D.Any d as dt) shape g =
  let n = numel shape in
  let esize = D.bits d / 8 in
  let copy_floor ins out =
    Some (harness g [ S.floor_copy ~ins:(List.map buf ins) ~out:(buf out) ])
  in
  let read_floor x = Some (harness g [ S.floor_read g (buf x) ]) in
  let rows_of () = Array.sub shape 0 (Array.length shape - 1) in
  let two () = (fresh g ~seed:1 dt shape, fresh g ~seed:2 dt shape) in
  match op with
  | "add" | "where_lt" | "fused4" ->
      let x, y = two () in
      let z = fresh g dt shape in
      let nodes, out =
        match op with
        | "add" -> ([ P.In 0; In 1; Op2 (Binary Add, 0, 1) ], 2)
        | "where_lt" ->
            ([ P.In 0; In 1; Op2 (Compare Less, 0, 1); Op3 (Where, 2, 0, 1) ], 3)
        | _ ->
            (* x*y + exp(x) - 0.5 *)
            ( [
                P.In 0;
                In 1;
                Op2 (Binary Mul, 0, 1);
                Op1 (Unary Exp, dt, 0);
                Op2 (Binary Add, 2, 3);
                f32_const 0.5;
                Op2 (Binary Sub, 4, 5);
              ],
              6 )
      in
      let run =
        loop_run g op
          (prog [ x; y ] nodes out)
          Map ~shape ~dsts:[ z ]
          [ plain x; plain y ]
      in
      (run, 3 * n * esize, copy_floor [ x; y ] z)
  | "exp" | "cast_bf16" ->
      let x = fresh g ~seed:1 dt shape in
      let to_ = if op = "exp" then dt else D.Any D.Bfloat16 in
      let z = fresh g to_ shape in
      let k = if op = "exp" then P.Unary Exp else P.Cast in
      let run =
        loop_run g op
          (prog [ x ] [ P.In 0; Op1 (k, to_, 0) ] 1)
          Map ~shape ~dsts:[ z ]
          [ plain x ]
      in
      let (D.Any t) = to_ in
      (run, n * (esize + (D.bits t / 8)), copy_floor [ x ] z)
  | "bcast_add" | "t_add" ->
      let rows = shape.(0) and cols = shape.(1) in
      let x = fresh g ~seed:1 dt shape in
      let y, ybytes =
        if op = "bcast_add" then
          let y = fresh g ~seed:2 dt [| cols |] in
          (view y ~strides:[| 0; 1 |] shape, cols * esize)
        else
          (* y stored [cols; rows], read transposed *)
          let y = fresh g ~seed:2 dt [| cols; rows |] in
          (view y ~strides:[| 1; rows |] shape, n * esize)
      in
      let z = fresh g dt shape in
      let run =
        loop_run g op
          (prog [ x; y ] [ P.In 0; In 1; Op2 (Binary Add, 0, 1) ] 2)
          Map ~shape ~dsts:[ z ]
          [ plain x; plain y ]
      in
      (run, (2 * n * esize) + ybytes, copy_floor [ x; y ] z)
  | "pad9" ->
      (* The 9 taps of a 3x3 window over the last two axes, zero past the
         edges: load (dy, dx) is x shifted by (dy - 1, dx - 1). *)
      let x = fresh g ~seed:1 dt shape in
      let z = fresh g dt shape in
      let (A.Any xa) = x in
      let rank = Array.length shape in
      let zeros = Iarray.of_list (List.init rank (fun _ -> 0)) in
      let tap dy dx =
        let shift sy sx =
          Iarray.of_list
            (List.init rank (fun j ->
                 if j = rank - 2 then sy else if j = rank - 1 then sx else 0))
        in
        let lo = shift (1 - dy) (1 - dx) and hi = shift (dy - 1) (dx - 1) in
        let fill = D.of_float (A.dtype xa) 0. in
        K.Padded
          {
            x = xa;
            fill;
            pad = { lo; hi; interior = zeros; windows = Iarray.of_list [] };
          }
      in
      let loads =
        List.concat_map (fun dy -> List.init 3 (tap dy)) [ 0; 1; 2 ]
      in
      let nodes =
        List.init 9 (fun i -> P.In i)
        @ List.init 8 (fun i ->
            P.Op2 (Binary Add, (if i = 0 then 0 else 8 + i), i + 1))
      in
      let p =
        P.v
          ~ins:(Iarray.of_list (List.init 9 (fun _ -> dt)))
          (Iarray.of_list nodes) ~outs:(Iarray.of_list [ 16 ])
      in
      let run = loop_run g op p Map ~shape ~dsts:[ z ] loads in
      (run, 2 * n * esize, copy_floor [ x ] z)
  | "argmax" | "logsumexp" | "moments" ->
      let x = fresh g ~seed:1 dt shape in
      let ys = rows_of () in
      let r, dsts =
        match op with
        | "argmax" -> (K.Arg Max, [ fresh g dt ys; fresh g (D.Any D.Int64) ys ])
        | "logsumexp" -> (K.Logsumexp, [ fresh g dt ys ])
        | _ -> (K.Moments, [ fresh g dt ys; fresh g dt ys ])
      in
      let last = Array.length shape - 1 in
      let run =
        loop_run g op (prog [ x ] [ P.In 0 ] 0)
          (Reduce { axes = [ last ]; rs = [ (r, 0) ] })
          ~shape ~dsts
          [ plain x ]
      in
      (run, n * esize, read_floor x)
  | "sqdiff" ->
      let x, y = two () in
      let z = fresh g dt (rows_of ()) in
      let nodes =
        [ P.In 0; In 1; Op2 (Binary Sub, 0, 1); Op2 (Binary Mul, 2, 2) ]
      in
      let last = Array.length shape - 1 in
      let run =
        loop_run g op
          (prog [ x; y ] nodes 3)
          (Reduce { axes = [ last ]; rs = [ (Monoid Sum, 0) ] })
          ~shape ~dsts:[ z ]
          [ plain x; plain y ]
      in
      (run, 2 * n * esize, None)
  | "window2" ->
      (* Sums of the 2x2 windows, at steps of 2, of the last two axes. *)
      let x = fresh g ~seed:1 dt shape in
      let (A.Any xa) = x in
      let rank = Array.length shape in
      let zeros = Iarray.of_list (List.init rank (fun _ -> 0)) in
      let w axis = { Nx_array.Move.axis; size = 2; step = 2; dilation = 1 } in
      let pad =
        {
          K.lo = zeros;
          hi = zeros;
          interior = zeros;
          windows = Iarray.of_list [ w (rank - 2); w (rank - 1) ];
        }
      in
      let wshape =
        Array.concat
          [
            Array.sub shape 0 (rank - 2);
            [| shape.(rank - 2) / 2; shape.(rank - 1) / 2; 2; 2 |];
          ]
      in
      let z = fresh g dt (Array.sub wshape 0 rank) in
      let load = K.Padded { x = xa; fill = D.of_float (A.dtype xa) 0.; pad } in
      let run =
        loop_run g op (prog [ x ] [ P.In 0 ] 0)
          (Reduce { axes = [ rank; rank + 1 ]; rs = [ (Monoid Sum, 0) ] })
          ~shape:wshape ~dsts:[ z ] [ load ]
      in
      (run, n * esize * 5 / 4, None)
  | "logcumsumexp" | "sqcumsum" ->
      let x = fresh g ~seed:1 dt shape in
      let z = fresh g dt shape in
      let p, r =
        if op = "sqcumsum" then
          (prog [ x ] [ P.In 0; Op2 (Binary Mul, 0, 0) ] 1, K.Monoid Sum)
        else (prog [ x ] [ P.In 0 ] 0, K.Logsumexp)
      in
      let run =
        loop_run ~own:(op = "sqcumsum") g op p
          (Scan { axis = Array.length shape - 1; r = (r, 0) })
          ~shape ~dsts:[ z ]
          [ plain x ]
      in
      (run, 2 * n * esize, copy_floor [ x ] z)
  | _ -> invalid_arg ("Rows.loop_op: " ^ op)

let loop_row kind op dt shape =
  let name = strf "%s/%s-%s-%s" kind op (dname dt) (dims shape) in
  let make g =
    let run, bytes, floor = loop_op op dt shape g in
    { run; work = Bytes bytes; floor }
  in
  { name; make; spec = Some (Loop { op; dtype = dt; shape }) }

(* The loops' rows, each at an L2 size and a DRAM size. *)
let loops =
  let open D in
  let f32 = Any Float32 and bf16 = Any Bfloat16 in
  let sized kind op dt (small, large) =
    [ loop_row kind op dt small; loop_row kind op dt large ]
  in
  let vec = ([| 1 lsl 20 |], [| 1 lsl 26 |]) in
  let square = ([| 1024; 1024 |], [| 8192; 8192 |]) in
  List.concat
    [
      sized "map" "add" f32 vec;
      sized "map" "exp" f32 vec;
      sized "map" "add" bf16 ([| 1 lsl 21 |], [| 1 lsl 27 |]);
      sized "map" "cast_bf16" f32 vec;
      sized "map" "bcast_add" f32 square;
      sized "map" "t_add" f32 square;
      sized "map" "fused4" f32 vec;
      sized "map" "pad9" f32 ([| 1; 64; 64; 64 |], [| 1; 64; 512; 512 |]);
      sized "map" "where_lt" f32 vec;
      sized "reduce" "argmax" f32 square;
      sized "reduce" "logsumexp" f32 square;
      sized "reduce" "moments" f32 square;
      sized "reduce" "window2" f32
        ([| 1; 64; 256; 256 |], [| 1; 64; 2048; 2048 |]);
      sized "reduce" "sqdiff" f32 square;
      sized "scan" "logcumsumexp" f32 ([| 1024; 1024 |], [| 4096; 4096 |]);
      sized "scan" "sqcumsum" f32 square;
    ]
  @ List.concat
      [
        (let s = ([| 2048; 1024 |], [| 16384; 8192 |]) in
         [ reduce bf16 Sum (fst s) [ 1 ]; reduce bf16 Sum (snd s) [ 1 ] ]);
        [ scan f32 Max (fst square) 1; scan f32 Max (snd square) 1 ];
      ]

(* Sorts and numberings *)

(* The sort rows of the sort plan's ROWS.md. A row's [x] is drawn on the
   device, its integers masked into \[0, [range]) by a map, and its result
   checked once against nx.cpu's bits where it has at most 2^24 elements:
   past them the check's host copies take several GiB. *)

let int64 = D.Any D.Int64

(* [x], of an integer dtype, masked in place into \[0, [range]), a power
   of two. *)
let mask g x range =
  let dt = any_dtype x in
  let (D.Any d) = dt in
  (* [range - 1]'s bytes, little-endian, the host's order *)
  let m =
    String.init
      (D.bits d / 8)
      (fun i -> Char.chr (((range - 1) lsr (8 * i)) land 255))
  in
  let p = prog [ x ] [ P.In 0; Const (dt, m); Op2 (Binary And, 0, 1) ] 2 in
  let run =
    match
      S.kernel_call ~what:"mask" ~launches:S.loop_launches [ x ] (fun () ->
          Nx_cuda.map p ~dsts:[| x |] [| plain x |])
    with
    | Some r -> r
    | None -> failwith "mask: nx.cuda declines it"
  in
  S.run g run

(* A drawn [dt] array of [shape]: floats in \[-1, 1), complex numbers of
   such parts, integers over their range or [range]. *)
let drawn g (D.Any d as dt) shape range =
  match dt with
  | D.Any D.Complex64 ->
      let a = A.create (S.device g) d (Iarray.of_array shape) in
      S.generate g (A.buffer a) D.Float32 S.Uniform ~seed:1;
      A.Any a
  | _ ->
      let x = fresh g ~seed:1 dt shape in
      Option.iter (mask g x) range;
      x

(* The run of [f] on [dsts] and [x] by [Kn], checked against nx.cpu's
   bits for a small enough [x]. *)
let checked_run g name ~check f ~dsts x =
  let run =
    match
      S.kernel_call ~what:name ~launches:S.loop_launches (dsts @ [ x ])
        (fun () -> f (module Nx_cuda : K.S) dsts x)
    with
    | Some r -> r
    | None -> failwith (name ^ ": nx.cuda declines it")
  in
  S.run g run;
  if check then begin
    let host = List.map (fresh_like Rig.host) dsts in
    let (A.Any xa) = x in
    (match f (module Nx_cpu : K.S) host (A.Any (A.to_device Rig.host xa)) with
    | A.Done -> ()
    | _ -> failwith (name ^ ": nx.cpu does not compute it"));
    List.iteri
      (fun k (d, h) ->
        let a = bytes_of d and b = bytes_of h in
        if a <> b then
          failwith
            (strf "%s: destination %d differs from nx.cpu's at element %d" name
               k (first_difference d a b)))
      (List.combine dsts host)
  end;
  run

let index (A.Any a) : (int64, D.int64_elt) A.t =
  match D.equal_witness (A.dtype a) D.Int64 with
  | Some Equal -> a
  | None -> invalid_arg "Rows.index"

let sort_row ?(axis = 1) ?(descending = false) ?k ?range name dt shape =
  let make g =
    let x = drawn g dt shape range in
    let kept = Array.copy shape in
    Option.iter (fun k -> kept.(axis) <- k) k;
    let dsts = [ fresh g dt kept; fresh g int64 kept ] in
    let sort (module Kn : K.S) dsts (A.Any x) =
      match dsts with
      | [ A.Any v; p ] -> (
          match D.equal_witness (A.dtype v) (A.dtype x) with
          | Some Equal ->
              Kn.sort ~axis ~descending ~values:v ~positions:(index p) x
          | None -> invalid_arg "Rows.sort_row")
      | _ -> invalid_arg "Rows.sort_row"
    in
    let check = numel shape <= 1 lsl 24 in
    let run = checked_run g name ~check sort ~dsts x in
    let (D.Any d) = dt in
    let bytes = numel shape * D.bits d / 8 in
    { run; work = Bytes (2 * bytes); floor = None }
  in
  {
    name = "sort/" ^ name;
    make;
    spec = Some (Sort { dtype = dt; shape; axis; descending; k; range });
  }

let unique_row ?range name n w =
  let make g =
    let x = drawn g int64 [| n; w |] range in
    let dsts =
      [
        fresh g int64 [| n |];
        fresh g int64 [| n |];
        fresh g int64 [| n |];
        fresh g int64 [||];
      ]
    in
    let unique (module Kn : K.S) dsts x =
      match List.map index dsts with
      | [ ids; first; counts; count ] ->
          Kn.unique ~ids ~first ~counts ~count (index x)
      | _ -> invalid_arg "Rows.unique_row"
    in
    let run = checked_run g name ~check:true unique ~dsts x in
    { run; work = Bytes (8 * n * w); floor = None }
  in
  { name = "sort/" ^ name; make; spec = Some (Unique { n; w; range }) }

let sorts =
  let open D in
  let f32 = Any Float32 and f64 = Any Float64 and f16 = Any Float16 in
  let i64 = Any Int64 and i32 = Any Int32 and u8 = Any Uint8 in
  let c64 = Any Complex64 and bf16 = Any Bfloat16 in
  let m16 = 1 lsl 24 and m256 = 1 lsl 28 in
  [
    sort_row "sort-f32-1x16M" f32 [| 1; m16 |];
    sort_row "sort-f32-1x256M" f32 [| 1; m256 |];
    sort_row "sort-f64-1x16M" f64 [| 1; m16 |];
    sort_row ~range:(1 lsl 20) "sort-i64-1x16M-range2^20" i64 [| 1; m16 |];
    sort_row "sort-c64-1x16M" c64 [| 1; m16 |];
    sort_row "sort-f16-1x16M" f16 [| 1; m16 |];
    sort_row "sort-u8-1x256M" u8 [| 1; m256 |];
    sort_row "sort-f32-65536x128" f32 [| 65536; 128 |];
    sort_row ~descending:true ~range:(1 lsl 30) "sort-i32-65536x128-desc" i32
      [| 65536; 128 |];
    sort_row "sort-f32-4096x4096" f32 [| 4096; 4096 |];
    sort_row "sort-f64-4096x4096" f64 [| 4096; 4096 |];
    sort_row "sort-f32-256x65536" f32 [| 256; 65536 |];
    sort_row ~axis:0 "sort-f32-4096x4096-axis0" f32 [| 4096; 4096 |];
    sort_row ~axis:0 "sort-f32-65536x256-axis0" f32 [| 65536; 256 |];
    sort_row ~descending:true ~k:8 "topk-f32-1x16M-k8" f32 [| 1; m16 |];
    sort_row ~descending:true ~k:1000 "topk-f32-1x16M-k1000" f32 [| 1; m16 |];
    sort_row ~descending:true ~k:50 "topk-f32-64x131072-k50" f32
      [| 64; 131072 |];
    sort_row ~descending:true ~k:64 "topk-bf16-32x262144-k64" bf16
      [| 32; 262144 |];
    sort_row ~descending:true ~k:8 "topk-f32-65536x128-k8" f32 [| 65536; 128 |];
    unique_row ~range:(1 lsl 20) "unique-i64-16Mx1-range2^20" m16 1;
    unique_row "unique-i64-16Mx1-full" m16 1;
    unique_row ~range:16 "unique-i64-4Mx4-range16" (1 lsl 22) 4;
  ]

(* Moves, transforms and matrices *)

(* The rows of the moves', transforms' and matrices' ROWS.md files, under
   their names there. Operands are drawn on the device; index arrays are
   drawn on the host from fixed seeds. A move whose result nx.cuda promises
   bit for bit (all but its sums) is checked once against nx.cpu's bits
   where its destination has at most 2^24 elements; the suites hold the
   sums, transforms and matrix routines to their bounds. *)

let host x = A.to_device Rig.host x
let host_any (A.Any x) = A.Any (host x)

(* A [d] array of [shape] on [g] drawn from [seed]: floats in \[-1, 1),
   complex numbers of such parts. *)
let draw (type v s) g ?(seed = 1) (d : (v, s) D.t) shape : (v, s) A.t =
  let a = A.create (S.device g) d (Iarray.of_array shape) in
  let b = A.buffer a in
  (match d with
  | D.Complex64 -> S.generate g b D.Float32 S.Uniform ~seed
  | D.Complex128 -> S.generate g b D.Float64 S.Uniform ~seed
  | _ -> S.generate g b d S.Uniform ~seed);
  a

let empty g d shape = A.create (S.device g) d (Iarray.of_array shape)

(* [values] as an int64 array of [shape] on [g], in C order. *)
let ints_on g shape values =
  let b = Bytes.create (8 * Array.length values) in
  Array.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) values;
  let a = empty g D.Int64 shape in
  S.write (A.buffer a) (Bytes.unsafe_to_string b);
  a

let random ~seed n bound =
  let st = Random.State.make [| seed |] in
  Array.init n (fun _ -> Random.State.int st bound)

let permutation ~seed n =
  let st = Random.State.make [| seed |] in
  let p = Array.init n Fun.id in
  for i = n - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let t = p.(i) in
    p.(i) <- p.(j);
    p.(j) <- t
  done;
  p

(* [a]'s buffer through the layout of [strides] and [shape]. *)
let restride a ~strides shape =
  A.v (A.dtype a)
    (A.Layout.v ~strides:(Iarray.of_array strides) (Iarray.of_array shape))
    (A.buffer a)

(* The run of [f], nx.cuda's kernel [what] over [t], whose arrays [all]
   lists, its first [dsts] destinations; with [host], which copies [t] to
   the host, checked once against nx.cpu's bits. A call submits at most
   [launches] launches. *)
let kernel_run g what ~launches ?host ~dsts ~all t f =
  let run =
    match
      S.kernel_call ~what ~launches (all t) (fun () ->
          f (module Nx_cuda : K.S) t)
    with
    | Some r -> r
    | None -> failwith (what ^ ": nx.cuda declines it")
  in
  S.run g run;
  Option.iter
    (fun host ->
      let h = host t in
      (match f (module Nx_cpu : K.S) h with
      | A.Done -> ()
      | _ -> failwith (what ^ ": nx.cpu does not compute it"));
      let firsts l = List.filteri (fun i _ -> i < dsts) l in
      List.iteri
        (fun k (d, c) ->
          let x = bytes_of d and y = bytes_of c in
          if x <> y then
            failwith
              (strf "%s: destination %d differs from nx.cpu's at element %d"
                 what k (first_difference d x y)))
        (List.combine (firsts (all t)) (firsts (all h))))
    host;
  run

(* Moves *)

(* A move's call: one submission, of a few launches, an assembly one a
   group of pieces. *)
let move_launches = S.loop_launches

let move_row name op dt ~bytes run =
  {
    name = "move/" ^ name;
    make = (fun g -> { run = run g; work = Bytes bytes; floor = None });
    spec = Some (Move { op; dtype = dt });
  }

let small n = n <= 1 lsl 24

(* x [65536; 1024] at 16384 random rows, ids [16384] broadcast along the
   rows *)
let gather_emb (type v s) (d : (v, s) D.t) g =
  let x = draw g d [| 65536; 1024 |] in
  let ids = ints_on g [| 16384; 1 |] (random ~seed:2 16384 65536) in
  let idx = restride ids ~strides:[| 1; 0 |] [| 16384; 1024 |] in
  let dst = empty g d [| 16384; 1024 |] in
  kernel_run g "gather_emb" ~launches:move_launches
    ~host:(fun (dst, idx, x) -> (host dst, host idx, host x))
    ~dsts:1
    ~all:(fun (dst, idx, x) -> [ A.Any dst; A.Any idx; A.Any x ])
    (dst, idx, x)
    (fun (module Kn : K.S) (dst, idx, x) -> Kn.gather ~axis:0 ~dst idx x)

let gather_rand g =
  let n = 1 lsl 24 in
  let x = draw g D.Float32 [| n |] in
  let idx = ints_on g [| n |] (random ~seed:2 n n) in
  let dst = empty g D.Float32 [| n |] in
  kernel_run g "gather_rand" ~launches:move_launches
    ~host:(fun (dst, idx, x) -> (host dst, host idx, host x))
    ~dsts:1
    ~all:(fun (dst, idx, x) -> [ A.Any dst; A.Any idx; A.Any x ])
    (dst, idx, x)
    (fun (module Kn : K.S) (dst, idx, x) -> Kn.gather ~axis:0 ~dst idx x)

(* 8192 runs of 0 to 31 rows of x [262144; 256], from random starts: their
   ends and the rows they gather. *)
let ranges =
  let k = 8192 in
  let lens = random ~seed:3 k 32 and starts = random ~seed:4 k (262144 - 32) in
  let ends = Array.make k 0 in
  Array.iteri
    (fun i l -> ends.(i) <- (l + if i = 0 then 0 else ends.(i - 1)))
    lens;
  (starts, ends, ends.(k - 1))

let gather_ranges g =
  let starts, ends, size = ranges in
  let k = Array.length starts in
  let x = draw g D.Float32 [| 262144; 256 |] in
  let starts = ints_on g [| k |] starts and ends = ints_on g [| k |] ends in
  let dst = empty g D.Float32 [| size; 256 |] in
  kernel_run g "gather_ranges" ~launches:move_launches
    ~host:(fun (dst, s, e, x) -> (host dst, host s, host e, host x))
    ~dsts:1
    ~all:(fun (dst, s, e, x) -> [ A.Any dst; A.Any s; A.Any e; A.Any x ])
    (dst, starts, ends, x)
    (fun (module Kn : K.S) (dst, starts, ends, x) ->
      Kn.gather_ranges ~dst ~starts ~ends x)

(* [n] float32 updates combined by [c] into [targets] at the positions
   [idx] gives; checked where nx.cuda promises nx.cpu's bits: a [Set] or
   [Max], or a sum of one update a target ([exact]). *)
let scatter_run (c : K.combine) ~unique ?(exact = false) ~targets ~n idx g =
  let into = draw g D.Float32 [| targets |] in
  let u = draw g ~seed:2 D.Float32 [| n |] in
  let idx = ints_on g [| n |] (idx ()) in
  let dst = empty g D.Float32 [| targets |] in
  let check = (c <> Add || exact) && small n in
  kernel_run g "scatter" ~launches:move_launches
    ?host:
      (if check then
         Some
           (fun (dst, into, idx, u) -> (host dst, host into, host idx, host u))
       else None)
    ~dsts:1
    ~all:(fun (dst, into, idx, u) ->
      [ A.Any dst; A.Any into; A.Any idx; A.Any u ])
    (dst, into, idx, u)
    (fun (module Kn : K.S) (dst, into, idx, u) ->
      Kn.scatter c ~unique ~axis:0 ~dst ~into idx u)

(* g [16384; 1024] added into emb [32768; 1024] at 16384 random rows *)
let scatter_emb g =
  let into = draw g D.Float32 [| 32768; 1024 |] in
  let u = draw g ~seed:2 D.Float32 [| 16384; 1024 |] in
  let ids = ints_on g [| 16384; 1 |] (random ~seed:2 16384 32768) in
  let idx = restride ids ~strides:[| 1; 0 |] [| 16384; 1024 |] in
  let dst = empty g D.Float32 [| 32768; 1024 |] in
  kernel_run g "scatter_emb" ~launches:move_launches ~dsts:1
    ~all:(fun (dst, into, idx, u) ->
      [ A.Any dst; A.Any into; A.Any idx; A.Any u ])
    (dst, into, idx, u)
    (fun (module Kn : K.S) (dst, into, idx, u) ->
      Kn.scatter Add ~unique:false ~axis:0 ~dst ~into idx u)

let whole n = { Nx_array.Move.start = 0; count = n; step = 1 }

(* [pieces], each with its region, into a fresh float32 [shape] *)
let assemble_run ~fill shape pieces g =
  let pieces =
    Array.of_list
      (List.mapi
         (fun i (region, s) ->
           (Iarray.of_list region, draw g ~seed:(i + 1) D.Float32 s))
         pieces)
  in
  let dst = empty g D.Float32 shape in
  let all (dst, pieces) =
    A.Any dst :: Array.to_list (Array.map (fun (_, p) -> A.Any p) pieces)
  in
  let check = small (numel shape) in
  kernel_run g "assemble" ~launches:move_launches
    ?host:
      (if check then
         Some
           (fun (dst, pieces) ->
             (host dst, Array.map (fun (r, p) -> (r, host p)) pieces))
       else None)
    ~dsts:1 ~all (dst, pieces)
    (fun (module Kn : K.S) (dst, pieces) ->
      Kn.assemble ~fill ~dst (Iarray.of_array pieces))

(* The adjoint of 3x3 windows at padding 1 over [32; 64; 64; 64], channels
   before rows: their 9 taps summed into each element. *)
let fold_col2im g =
  let w axis = { Nx_array.Move.axis; size = 3; step = 1; dilation = 1 } in
  let pad =
    {
      K.lo = Iarray.of_list [ 0; 0; 1; 1 ];
      hi = Iarray.of_list [ 0; 0; 1; 1 ];
      interior = Iarray.of_list [ 0; 0; 0; 0 ];
      windows = Iarray.of_list [ w 2; w 3 ];
    }
  in
  let x = draw g D.Float32 [| 32; 64; 64; 64; 3; 3 |] in
  let dst = empty g D.Float32 [| 32; 64; 64; 64 |] in
  kernel_run g "fold" ~launches:move_launches ~dsts:1
    ~all:(fun (dst, x) -> [ A.Any dst; A.Any x ])
    (dst, x)
    (fun (module Kn : K.S) (dst, x) -> Kn.fold pad ~dst x)

let moves =
  let f32 = D.Any D.Float32 and bf16 = D.Any D.Bfloat16 in
  let m = 1 lsl 24 and mi = 1 lsl 20 in
  (* a scatter's bytes: its updates, their positions, and its targets read
     and written *)
  let sbytes ~n ~targets = (12 * n) + (8 * targets) in
  let scatter name op c ?unique ?exact ~targets idx =
    let unique = Option.value unique ~default:false in
    move_row name op f32 ~bytes:(sbytes ~n:m ~targets)
      (scatter_run c ~unique ?exact ~targets ~n:m idx)
  in
  let perm () = permutation ~seed:5 m in
  let emb = 16384 * 1024 in
  let _, _, size = ranges in
  [
    move_row "gather_emb_f32" "gather_emb" f32 ~bytes:(8 * emb)
      (gather_emb D.Float32);
    move_row "gather_emb_bf16" "gather_emb" bf16 ~bytes:(4 * emb)
      (gather_emb D.Bfloat16);
    move_row "gather_rand_f32" "gather_rand" f32 ~bytes:(16 * m) gather_rand;
    move_row "ranges_f32" "ranges" f32 ~bytes:(2 * 1024 * size) gather_ranges;
    scatter "scatter_add_perm_f32" "scatter_add_perm" Add ~targets:m perm;
    scatter "scatter_add_perm_unique_f32" "scatter_add_perm" Add ~unique:true
      ~exact:true ~targets:m perm;
    scatter "scatter_add_one_f32" "scatter_add_one" Add ~targets:1024 (fun () ->
        Array.make m 7);
    scatter "scatter_add_rand_f32" "scatter_add_rand" Add ~targets:m (fun () ->
        random ~seed:6 m m);
    scatter "scatter_add_rand1m_f32" "scatter_add_rand1m" Add ~targets:mi
      (fun () -> random ~seed:7 m mi);
    move_row "scatter_add_emb_f32" "scatter_add_emb" f32
      ~bytes:((12 * emb) + (8 * 32768 * 1024))
      scatter_emb;
    scatter "scatter_set_perm_f32" "scatter_set_perm" Set ~targets:m perm;
    scatter "scatter_max_rand1m_f32" "scatter_max_rand1m" Max ~targets:mi
      (fun () -> random ~seed:7 m mi);
    move_row "assemble_cat_f32" "assemble_cat" f32
      ~bytes:(8 * 16384 * 4096)
      (assemble_run ~fill:0. [| 16384; 4096 |]
         (List.init 8 (fun i ->
              ( [
                  { Nx_array.Move.start = 2048 * i; count = 2048; step = 1 };
                  whole 4096;
                ],
                [| 2048; 4096 |] ))));
    move_row "assemble_pad_f32" "assemble_pad" f32
      ~bytes:(4 * ((8192 * 8192) + (8200 * 8200)))
      (assemble_run ~fill:1.5 [| 8200; 8200 |]
         [
           ( [
               { Nx_array.Move.start = 4; count = 8192; step = 1 };
               { start = 4; count = 8192; step = 1 };
             ],
             [| 8192; 8192 |] );
         ]);
    move_row "fold_col2im_f32" "fold_col2im" f32
      ~bytes:(4 * 32 * 64 * 64 * 64 * 10)
      fold_col2im;
  ]

(* Transforms *)

(* A transform's call: one submission of at most 4 launches. *)
let fft_launches = S.loop_launches

let c2c_run (type s) (d : (Complex.t, s) D.t) shape axes g =
  let x = draw g d shape and dst = empty g d shape in
  kernel_run g "c2c" ~launches:fft_launches ~dsts:1
    ~all:(fun (dst, x) -> [ A.Any dst; A.Any x ])
    (dst, x)
    (fun (module Kn : K.S) (dst, x) -> Kn.c2c Forward ~axes ~dst x)

(* [shape] with its last extent [n] cut to the [n/2 + 1] bins of a real
   transform *)
let bins shape =
  let s = Array.copy shape in
  let l = Array.length s - 1 in
  s.(l) <- (s.(l) / 2) + 1;
  s

let r2c_run (type r c) (rd : (float, r) D.t) (cd : (Complex.t, c) D.t) shape
    axes g =
  let x = draw g rd shape and dst = empty g cd (bins shape) in
  kernel_run g "r2c" ~launches:fft_launches ~dsts:1
    ~all:(fun (dst, x) -> [ A.Any dst; A.Any x ])
    (dst, x)
    (fun (module Kn : K.S) (dst, x) -> Kn.r2c ~axes ~dst x)

let c2r_run (type r c) (rd : (float, r) D.t) (cd : (Complex.t, c) D.t) shape
    axes g =
  let x = draw g cd (bins shape) and dst = empty g rd shape in
  kernel_run g "c2r" ~launches:fft_launches ~dsts:1
    ~all:(fun (dst, x) -> [ A.Any dst; A.Any x ])
    (dst, x)
    (fun (module Kn : K.S) (dst, x) -> Kn.c2r ~axes ~dst x)

(* The row [name] of [kind] at complex64 or complex128 ([double]) over
   [shape], its bytes the operand's and the result's. *)
let fft_row ~double kind name shape axes =
  let cdt = if double then D.Any D.Complex128 else D.Any D.Complex64 in
  let part = if double then 8 else 4 in
  let n = numel shape and nb = numel (bins shape) in
  let bytes, run =
    let axes' = Iarray.of_list axes in
    match (kind, double) with
    | "c2c", false -> (4 * part * n, c2c_run D.Complex64 shape axes')
    | "c2c", true -> (4 * part * n, c2c_run D.Complex128 shape axes')
    | "r2c", false ->
        (part * (n + (2 * nb)), r2c_run D.Float32 D.Complex64 shape axes')
    | "r2c", true ->
        (part * (n + (2 * nb)), r2c_run D.Float64 D.Complex128 shape axes')
    | "c2r", false ->
        (part * (n + (2 * nb)), c2r_run D.Float32 D.Complex64 shape axes')
    | "c2r", true ->
        (part * (n + (2 * nb)), c2r_run D.Float64 D.Complex128 shape axes')
    | _ -> invalid_arg ("Rows.fft_row: " ^ kind)
  in
  {
    name = "fft/" ^ name;
    make = (fun g -> { run = run g; work = Bytes bytes; floor = None });
    spec = Some (Fft { kind; dtype = cdt; shape; axes });
  }

let transforms =
  let p20 = 1 lsl 20 in
  List.concat_map
    (fun double ->
      let p = if double then "c128" else "c64" in
      let row kind rest shape axes =
        fft_row ~double kind (strf "%s-%s-%s" kind p rest) shape axes
      in
      [
        row "c2c" "4096x1024" [| 4096; 1024 |] [ 1 ];
        row "c2c" "2^20" [| p20 |] [ 0 ];
        row "c2c" "2^22" [| 1 lsl 22 |] [ 0 ];
        row "c2c" "2^24" [| 1 lsl 24 |] [ 0 ];
        row "c2c" "2048x2048-axes01" [| 2048; 2048 |] [ 0; 1 ];
        row "r2c" "4096x1024" [| 4096; 1024 |] [ 1 ];
        row "c2r" "4096x513" [| 4096; 1024 |] [ 1 ];
        row "r2c" "2^20" [| p20 |] [ 0 ];
        row "r2c" "2048x2048-axes01" [| 2048; 2048 |] [ 0; 1 ];
        row "c2c" "4096x1009" [| 4096; 1009 |] [ 1 ];
        row "c2c" "2^20+7" [| p20 + 7 |] [ 0 ];
      ])
    [ false; true ]

(* Matrices *)

(* A routine's call: of up to hundreds of launches on a large matrix, so
   one call a round. *)
let matrix_launches = 512

(* [a]'s matrices, [shape] [[b; n; n]] of [d], with [n] added to the real
   part of each diagonal element: positive definite as the lower triangle
   and the diagonal's real parts name it, which is all a Cholesky
   factorisation reads, and well conditioned for a triangular solve. *)
let dominant (type v s) g (d : (v, s) D.t) shape : (v, s) A.t =
  let a = draw g d shape in
  let b = shape.(0) and n = shape.(1) in
  let parts = if D.is D.Complex d then 2 else 1 in
  let part = D.bits d / 8 / parts in
  let bytes = Bytes.of_string (S.read (A.buffer a)) in
  let add at =
    if part = 4 then
      let v = Int32.float_of_bits (Bytes.get_int32_le bytes at) in
      Bytes.set_int32_le bytes at (Int32.bits_of_float (v +. float n))
    else
      let v = Int64.float_of_bits (Bytes.get_int64_le bytes at) in
      Bytes.set_int64_le bytes at (Int64.bits_of_float (v +. float n))
  in
  for m = 0 to b - 1 do
    for i = 0 to n - 1 do
      add (((m * n * n) + (i * (n + 1))) * parts * part)
    done
  done;
  S.write (A.buffer a) (Bytes.unsafe_to_string bytes);
  a

type routine = Lu | Qr | Chol | Solve | Eigh | Svd | Eig
type vectors = Values | Vectors | Complete

let routine_name = function
  | Lu -> "lu"
  | Qr -> "qr"
  | Chol -> "chol"
  | Solve -> "solve"
  | Eigh -> "eigh"
  | Svd -> "svd"
  | Eig -> "eig"

let vectors_name = function
  | Values -> "values"
  | Vectors -> "vectors"
  | Complete -> "complete"

(* The real and complex dtypes of [d]'s precision *)
let real_of (type v s) (d : (v, s) D.t) =
  match d with
  | D.Float32 | D.Complex64 -> D.Any D.Float32
  | _ -> D.Any D.Float64

let complex_of (type v s) (d : (v, s) D.t) =
  match d with
  | D.Float32 | D.Complex64 -> D.Any D.Complex64
  | _ -> D.Any D.Complex128

let empty_any g (D.Any d) shape = A.Any (empty g d shape)

(* [r] on [a] into [dsts] by [Kn], a solve's right-hand sides [rhs]. *)
let matrix_call (module Kn : K.S) r (dsts, A.Any a, rhs) =
  let like x = A.expect (A.dtype a) x in
  let bad () = invalid_arg ("Rows.matrix_call: " ^ routine_name r) in
  match (r, dsts, rhs) with
  | Lu, [ lu; pivots; perm ], _ ->
      Kn.lu ~lu:(like lu) ~pivots:(index pivots) ~perm:(index perm) a
  | Qr, [ q; r ], _ -> Kn.qr ~q:(like q) ~r:(like r) a
  | Chol, [ l ], _ -> Kn.cholesky Lower ~dst:(like l) a
  | Solve, [ x ], [ b ] ->
      Kn.solve_triangular Lower ~adjoint:false ~unit_diagonal:false
        ~dst:(like x) a (like b)
  | Eigh, A.Any w :: v, _ -> (
      let v = Option.map like (List.nth_opt v 0) in
      match A.dtype w with
      | D.Float32 -> Kn.eigh ~w ?v a
      | D.Float64 -> Kn.eigh ~w ?v a
      | _ -> bad ())
  | Svd, A.Any s :: uv, _ -> (
      let vectors =
        match uv with
        | [ u; vh ] -> Some (~u:(like u), ~vh:(like vh))
        | _ -> None
      in
      match A.dtype s with
      | D.Float32 -> Kn.svd ~s ?vectors a
      | D.Float64 -> Kn.svd ~s ?vectors a
      | _ -> bad ())
  | Eig, A.Any w :: v, _ -> (
      let v = Option.map (A.expect (A.dtype w)) (List.nth_opt v 0) in
      match A.dtype w with
      | D.Complex64 -> Kn.eig ~w ?v a
      | D.Complex128 -> Kn.eig ~w ?v a
      | _ -> bad ())
  | _ -> bad ()

(* The row of [r] on [dt] matrices of [shape]: [b; m; n], [b; n] for a
   Cholesky factorisation, [b; n; k] for a solve of [k] right-hand sides,
   with [vectors] for eigh, svd and eig. Its work is LAPACK's count of the
   factorisation's flops (qr's without forming q, eigh's tridiagonal
   reduction, svd's bidiagonal one, eig's Hessenberg QR at 10 n^3), real
   flops on complex matrices. *)
let matrix_row ?(vectors = Vectors) r (D.Any d as dt) shape =
  let short = function
    | D.Any D.Float32 -> "f32"
    | D.Any D.Float64 -> "f64"
    | D.Any D.Complex64 -> "c64"
    | _ -> "c128"
  in
  let mode =
    match (r, vectors) with
    | (Eigh | Svd | Eig), Values -> "-values"
    | Svd, Complete -> "-complete"
    | _ -> ""
  in
  let name =
    strf "linalg/%s/%s/%s%s" (routine_name r) (short dt) (dims shape) mode
  in
  let b = shape.(0) in
  let m = shape.(1) in
  let n = if Array.length shape > 2 then shape.(2) else m in
  let k = Int.min m n in
  let cx = if D.is D.Complex d then 4. else 1. in
  let fm = float m and fn = float n and fk = float k in
  let flops =
    float b *. cx
    *.
    match r with
    | Lu ->
        2.
        *. ((fm *. fn *. fk)
           -. ((fm +. fn) *. fk *. fk /. 2.)
           +. (fk *. fk *. fk /. 3.))
    | Qr -> (2. *. Float.max fm fn *. fk *. fk) -. (2. *. fk *. fk *. fk /. 3.)
    | Chol -> fm *. fm *. fm /. 3.
    | Solve -> fm *. fm *. fn
    | Eigh -> 4. *. fm *. fm *. fm /. 3.
    | Svd -> 4. *. fm *. fn *. fk
    | Eig -> 10. *. fm *. fm *. fm
  in
  let make g =
    let a =
      match r with
      | Chol -> A.Any (dominant g d [| b; m; m |])
      | Solve -> A.Any (dominant g d [| b; m; m |])
      | _ -> A.Any (draw g d [| b; m; n |])
    in
    let e = empty_any g in
    let i64 = D.Any D.Int64 in
    let real = real_of d and cplx = complex_of d in
    let dsts, rhs =
      match (r, vectors) with
      | Lu, _ -> ([ e dt [| b; m; n |]; e i64 [| b; k |]; e i64 [| b; m |] ], [])
      | Qr, _ -> ([ e dt [| b; m; k |]; e dt [| b; k; n |] ], [])
      | Chol, _ -> ([ e dt [| b; m; m |] ], [])
      | Solve, _ ->
          ([ e dt [| b; m; n |] ], [ A.Any (draw g ~seed:2 d [| b; m; n |]) ])
      | Eigh, Values -> ([ e real [| b; m |] ], [])
      | Eigh, _ -> ([ e real [| b; m |]; e dt [| b; m; m |] ], [])
      | Svd, Values -> ([ e real [| b; k |] ], [])
      | Svd, Vectors ->
          ([ e real [| b; k |]; e dt [| b; m; k |]; e dt [| b; k; n |] ], [])
      | Svd, Complete ->
          ([ e real [| b; k |]; e dt [| b; m; m |]; e dt [| b; n; n |] ], [])
      | Eig, Values -> ([ e cplx [| b; m |] ], [])
      | Eig, _ -> ([ e cplx [| b; m |]; e cplx [| b; m; m |] ], [])
    in
    let run =
      kernel_run g name ~launches:matrix_launches ~dsts:(List.length dsts)
        ~all:(fun (dsts, a, rhs) -> dsts @ (a :: rhs))
        (dsts, a, rhs)
        (fun kn t -> matrix_call kn r t)
    in
    { run; work = Flops flops; floor = None }
  in
  {
    name;
    make;
    spec =
      Some
        (Matrix
           {
             routine = routine_name r;
             dtype = dt;
             shape;
             mode = vectors_name vectors;
           });
  }

let matrices =
  let open D in
  let f32 = Any Float32 and f64 = Any Float64 in
  let c64 = Any Complex64 and c128 = Any Complex128 in
  let batched =
    List.concat_map
      (fun dt ->
        List.concat_map
          (fun (b, n) ->
            [
              matrix_row Lu dt [| b; n; n |];
              matrix_row Qr dt [| b; n; n |];
              matrix_row Chol dt [| b; n |];
              matrix_row Solve dt [| b; n; n |];
            ])
          [ (65536, 8); (4096, 32); (64, 256) ])
      [ f32; f64; c64; c128 ]
  in
  let single = [| 1; 4096; 4096 |] and single8 = [| 1; 8192; 8192 |] in
  let tall = [| 1; 65536; 64 |] in
  batched
  @ [
      matrix_row Lu f32 single;
      matrix_row Chol f32 [| 1; 4096 |];
      matrix_row Qr f32 single;
      matrix_row Solve f32 single;
      matrix_row Solve f32 [| 1; 4096; 1 |];
      matrix_row Lu f32 single8;
      matrix_row Chol f32 [| 1; 8192 |];
      matrix_row Qr f32 single8;
      matrix_row Qr f32 tall;
      matrix_row Lu f64 single;
      matrix_row Chol f64 [| 1; 4096 |];
      matrix_row Qr f64 single;
      matrix_row Solve f64 single;
      matrix_row Solve f64 [| 1; 4096; 1 |];
      matrix_row Lu f64 single8;
      matrix_row Chol f64 [| 1; 8192 |];
      matrix_row Qr f64 single8;
      matrix_row Qr f64 tall;
      matrix_row Lu c64 single;
      matrix_row Chol c64 [| 1; 4096 |];
      matrix_row Qr c64 single;
      matrix_row Qr c64 tall;
      matrix_row Qr c128 tall;
      (* the integration's additions: complex packs in the chain
         products, P <= 16 panels over a long k, a transposed operand *)
      matrix_row Lu c64 [| 1; 1024; 1024 |];
      matrix_row Qr f32 [| 1; 2048; 1024 |];
    ]
  @ [
      matrix_row Eigh f32 [| 16384; 8; 8 |];
      matrix_row Eigh f64 [| 16384; 8; 8 |];
      matrix_row Eigh c64 [| 16384; 8; 8 |];
      matrix_row ~vectors:Values Eigh f32 [| 16384; 8; 8 |];
      matrix_row Eigh f32 [| 1024; 32; 32 |];
      matrix_row Eigh f64 [| 1024; 32; 32 |];
      matrix_row Eigh c128 [| 1024; 32; 32 |];
      matrix_row Eigh f32 [| 64; 256; 256 |];
      matrix_row Eigh f64 [| 64; 256; 256 |];
      matrix_row ~vectors:Values Eigh f32 [| 64; 256; 256 |];
      matrix_row Eigh f32 [| 1; 2048; 2048 |];
      matrix_row Eigh f64 [| 1; 2048; 2048 |];
      matrix_row Eigh c64 [| 1; 2048; 2048 |];
      matrix_row Eigh f32 single;
      matrix_row ~vectors:Values Eigh f32 single;
      matrix_row Svd f32 [| 16384; 8; 8 |];
      matrix_row Svd f64 [| 16384; 8; 8 |];
      matrix_row ~vectors:Values Svd f32 [| 16384; 8; 8 |];
      matrix_row Svd f32 [| 1024; 32; 32 |];
      matrix_row Svd c64 [| 1024; 32; 32 |];
      matrix_row ~vectors:Complete Svd f32 [| 1024; 32; 32 |];
      matrix_row Svd f32 [| 64; 256; 256 |];
      matrix_row Svd f64 [| 64; 256; 256 |];
      matrix_row ~vectors:Values Svd f32 [| 64; 256; 256 |];
      matrix_row ~vectors:Values Svd f32 [| 1; 2048; 2048 |];
      matrix_row Svd f32 tall;
      matrix_row Svd f64 tall;
      matrix_row ~vectors:Values Svd f32 tall;
      matrix_row Eig f32 [| 16384; 8; 8 |];
      matrix_row Eig f32 [| 1024; 32; 32 |];
      matrix_row Eig c64 [| 1024; 32; 32 |];
      matrix_row Eig f64 [| 1024; 32; 32 |];
      matrix_row ~vectors:Values Eig f32 [| 1024; 32; 32 |];
      matrix_row Eig f32 [| 64; 256; 256 |];
    ]

(* Attention *)

(* Calls of Nx.attention as nx.cuda receives them: a query [[b; hq; tq; d]]
   over dense keys and values [[b; hk; tk; d]], scale 1/√d, o in the
   query's dtype and lse in float32, float64 for float64 queries, the pair
   program Nx makes for the mask. Not checked: the suites hold attention
   to its bounds. *)

(* An attention's call: one submission of two launches, or the map of a
   program that does not fit before it. *)
let attention_launches = S.loop_launches

(* Nx.attention's program without lengths over a score in [acc]: a pair's
   length is the constant [tk], output 0 Nx.causal's [k <= q + (len - tq)]
   or [true], output 1 the score. *)
let pair_program acc ~tq ~tk ~causal =
  let int64 x = P.Const (D.Any D.Int64, P.bits D.Int64 (Int64.of_int x)) in
  let pair = [ P.In 0; int64 tk; Coord 2; Coord 3 ] in
  let mask =
    if causal then
      [
        int64 tq;
        Op2 (Binary Sub, 1, 4);
        Op2 (Binary Add, 2, 5);
        Op2 (Compare Less_equal, 3, 6);
      ]
    else [ Const (D.Any D.Bool, P.bits D.Bool true) ]
  in
  let nodes = pair @ mask in
  P.v ~ins:(Iarray.of_list [ acc ]) (Iarray.of_list nodes)
    ~outs:(Iarray.of_list [ List.length nodes - 1; 0 ])

(* Each batch element's keys under the mask ["lengths"]. *)
let key_lengths ~b ~tk = Array.init b (fun i -> (i + 1) * tk / b)

(* The run of attention over [d] operands of the extents
   [(b, hq, hk, tq, tk, d)] in [acc], under [mask]. *)
let attend (type v s c) g (d : (v, s) D.t) (acc : (float, c) D.t) ~mask
    (b, hq, hk, tq, tk, dh) =
  let q = draw g d [| b; hq; tq; dh |] in
  let k = draw g ~seed:2 d [| b; hk; tk; dh |] in
  let v = draw g ~seed:3 d [| b; hk; tk; dh |] in
  let o = empty g d [| b; hq; tq; dh |] and lse = empty g acc [| b; hq; tq |] in
  let lengths =
    if mask = "lengths" then Some (ints_on g [| b |] (key_lengths ~b ~tk))
    else None
  in
  let p = pair_program (D.Any acc) ~tq ~tk ~causal:(mask = "causal") in
  let scale = 1. /. Float.sqrt (float dh) in
  let arrays =
    [ A.Any o; A.Any lse; A.Any q; A.Any k; A.Any v ]
    @ Option.to_list (Option.map (fun l -> A.Any l) lengths)
  in
  let call () =
    Nx_cuda.attention p ~scale ~o ~lse ?lengths q
      (K.Dense { k; v })
      (Iarray.of_list [])
  in
  match
    S.kernel_call ~what:"attention" ~launches:attention_launches arrays call
  with
  | Some r -> r
  | None -> failwith "attention: nx.cuda declines it"

(* The row of [dt] queries under [mask] at [b; hq; hk; tq; tk; d]. Its work
   is 4d flops a visible pair, q·k's and the value product's. *)
let attention_row (D.Any d as dt) mask ((b, hq, hk, tq, tk, dh) as s) =
  let short = function
    | D.Any D.Float16 -> "f16"
    | D.Any D.Bfloat16 -> "bf16"
    | D.Any D.Float32 -> "f32"
    | _ -> "f64"
  in
  let name =
    strf "attention/%s-%s-b%d-h%dk%d-%dx%d-d%d" (short dt) mask b hq hk tq tk dh
  in
  let visible =
    match mask with
    | "causal" -> b * hq * ((tq * (tk - tq)) + (tq * (tq + 1) / 2))
    | "lengths" -> hq * tq * Array.fold_left ( + ) 0 (key_lengths ~b ~tk)
    | _ -> b * hq * tq * tk
  in
  let flops = 4. *. float dh *. float visible in
  let make g =
    let run =
      match d with
      | D.Float64 -> attend g d D.Float64 ~mask s
      | _ -> attend g d D.Float32 ~mask s
    in
    { run; work = Flops flops; floor = None }
  in
  {
    name;
    make;
    spec = Some (Attention { dtype = dt; b; hq; hk; tq; tk; d = dh; mask });
  }

(* A prefill (tq = tk = 2048, 32 heads, d 128), a mid shape (tq = tk = 512,
   16 heads, d 64, a batch of 4) and a decode step (one row a head against
   8192 keys, 32 query heads in groups of 4, a batch of 8). *)
let attentions =
  let open D in
  let f16 = Any Float16 and bf16 = Any Bfloat16 in
  let f32 = Any Float32 and f64 = Any Float64 in
  let prefill = (1, 32, 32, 2048, 2048, 128) in
  let mid = (4, 16, 16, 512, 512, 64) in
  let decode = (8, 32, 8, 1, 8192, 128) in
  [
    attention_row f16 "causal" prefill;
    attention_row f16 "full" prefill;
    attention_row bf16 "causal" prefill;
    attention_row f32 "causal" prefill;
    attention_row f32 "full" prefill;
    attention_row f16 "causal" mid;
    attention_row f32 "full" mid;
    attention_row f64 "causal" mid;
    attention_row f16 "full" decode;
    attention_row bf16 "full" decode;
    attention_row f32 "full" decode;
    attention_row f16 "lengths" decode;
  ]

(* Contraction rows of the contraction probe's names: b's free axis
   contiguous ([-kf]), as the probe lays it out, and complex64. *)
let probe_contracts =
  let open D in
  let f32 = Any Float32 and f64 = Any Float64 and c64 = Any Complex64 in
  let kf ?batch dt m n k = contract ?batch ~lb:`Free dt dt dt m n k in
  List.map (fun s -> kf f32 s s s) [ 256; 512; 1024; 2048 ]
  @ [
      kf f32 1 4096 4096;
      kf f32 8 4096 4096;
      kf ~batch:256 f32 64 64 64;
      kf f32 256 256 65536;
    ]
  @ List.map (fun s -> kf f64 s s s) [ 256; 512; 1024; 2048; 4096 ]
  @ [ kf f64 1 4096 4096; kf f64 8 4096 4096; kf ~batch:256 f64 64 64 64 ]
  @ [ kf c64 1024 1024 1024 ]

(* The bench's suite: floors, contractions, reductions, scans and loops. *)
let all = floors @ contracts @ reductions @ loops

(* What compare and gate also time: every family against PyTorch. *)
let compared =
  all @ probe_contracts @ sorts @ moves @ transforms @ matrices @ attentions
