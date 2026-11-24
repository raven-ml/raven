(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Metal rows: each prepares its operands on the GPU, checks its result
   against nx.cpu's bits, which nx.metal's loops promise for normal operands,
   where asked, and gives the call to time, with the copy that moves its bytes
   where it is memory-bound. A run is a call of nx.metal's, as a user makes it;
   the timing waits for the last of its runs. *)

module A = Nx_array
module D = Nx_array.Dtype
module K = Nx_kernel
module P = Nx_kernel.Prog

let strf = Printf.sprintf

type prepared = {
  run : unit -> unit;  (** Issues the call; returns before it is done. *)
  wait : unit -> unit;  (** Returns once every call issued is done. *)
}

type row = {
  name : string;
  bytes : int;  (** The bytes a call moves, its floor's copy. *)
  slow : int;
      (** Its large size's time over its floor's, measured on an M1 Max and
          rounded, the small size taking the large one's: a case issues that
          many fewer calls, so it stays milliseconds; the case's name carries
          its calls. *)
  make : check:bool -> Rig.t -> prepared;
      (** Its operands on the device, the call checked where [check]. *)
  spec : string;
      (** The computation as compare.py reads it, its fields separated by
          spaces, a shape's extents by [x] and axes by [,]:
          [loop OP DTYPE SHAPE], [reduce DTYPE MONOID SHAPE AXES],
          [scan DTYPE MONOID SHAPE AXIS],
          [contract DTYPE BDTYPE ACC OUT BATCH M N K LALB], a's dtype then b's,
          [LALB] each operand's contracted axis last ([k]) or first ([f]),
          [rows DTYPE G T N K LALB IDS], a Rows call of [T] rows by [G] groups,
          row i by group i of [IDS],
          [decode FORMAT DTYPE EXPERTS ROWS N K LALB [IDS]], a product by a
          weight stored as [FORMAT] ([mxfp4], [q8_0], [q6_k], GGUF's blocks, or
          [q6_k_int8], int8 quants by float32 scales a block of 16, as
          Nx_quant's product unpacks Q6_K), of experts where [EXPERTS > 0], row
          i by expert i of [IDS], [fft DTYPE SHAPE AXES [HELD]], a forward
          transform, of the leading part of a C-ordered array of [HELD] where
          given, or [sort DTYPE SHAPE AXIS KEEP], a stable ascending sort's
          values and int64 positions, the first [KEEP] of each slice, or
          [move OP DTYPE], the gather, range gather or assembly nx.cuda's moves
          rows name [OP], or [ranges_elem], a range gather of cells of one
          element, or [attention DTYPE B HQ HK TQ TK D MASK], a query
          [[B; HQ; TQ; D]] over dense keys and values [[B; HK; TK; D]], [MASK]
          [full], [causal] (Nx.causal), [window128] (Nx.causal and the 128 keys
          up to the query's position) or [lengths] (batch element i's keys cut
          at [(i + 1) TK / B]). *)
}

let dname (D.Any d) = D.name d

(* A shape in a row's name: its extents, or "scalar" of no axis. *)
let dims shape =
  if Array.length shape = 0 then "scalar"
  else String.concat "x" (List.map string_of_int (Array.to_list shape))

let numel shape = Array.fold_left ( * ) 1 shape

(* What a row computes from its program's outputs: a map, a reduction along
   [axes] by [rs], or a scan along [axis]. *)
type loop =
  | Map
  | Reduce of { axes : int list; rs : (K.reduction * int) list }
  | Scan of { axis : int; r : K.reduction * int }

let apply (module Kn : K.S) p l ~shape ~dsts loads =
  let dsts = Iarray.of_list dsts and loads = Iarray.of_list loads in
  match l with
  | Map -> Kn.map p ~dsts loads
  | Reduce { axes; rs } ->
      Kn.reduce p ~shape:(Iarray.of_array shape) ~axes:(Iarray.of_list axes)
        (Iarray.of_list rs) ~dsts loads
  | Scan { axis; r } -> Kn.scan p ~axis r ~dsts loads

let done_ what = function
  | A.Done -> ()
  | a ->
      failwith
        (strf "%s: %s" what (Format.asprintf "%a" Nx_array_support.pp_answer a))

let any_dtype (A.Any a) = D.Any (A.dtype a)
let ins xs = Iarray.of_list (List.map any_dtype xs)

let prog xs nodes out =
  P.v ~ins:(ins xs) (Iarray.of_list nodes) ~outs:(Iarray.of_list [ out ])

let f32_const v = P.Const (D.Any D.Float32, P.bits D.Float32 v)
let plain (A.Any a) = K.Plain a

(* Operands *)

(* A fresh C-ordered [dt] array of [shape] on [g], its elements, if [seed], sin
   (0.7548776662 i + seed) at index [i] in C order, computed by nx.metal and
   cast: a [complex64]'s at its parts' indices, as the float32 pairs it
   holds. *)
let fresh g ?seed (D.Any d as dt) shape =
  let a = A.create g d (Iarray.of_array shape) in
  Option.iter
    (fun seed ->
      let n = numel shape in
      let (D.Any fd as dt), n =
        if D.is D.Complex d then (D.Any D.Float32, 2 * n) else (dt, n)
      in
      let flat =
        A.v fd (A.Layout.contiguous (Iarray.of_list [ n ])) (A.buffer a)
      in
      let f32 = D.Any D.Float32 in
      let p =
        P.v ~ins:(Iarray.of_list [])
          (Iarray.of_list
             [
               P.Coord 0;
               Op1 (Cast, f32, 0);
               f32_const 0.7548776662;
               Op2 (Binary Mul, 1, 2);
               f32_const (float seed);
               Op2 (Binary Add, 3, 4);
               Op1 (Unary Sin, f32, 5);
               Op1 (Cast, dt, 6);
             ])
          ~outs:(Iarray.of_list [ 7 ])
      in
      done_ "fresh"
        (Nx_metal.map p
           ~dsts:(Iarray.of_list [ A.Any flat ])
           (Iarray.of_list [])))
    seed;
  A.Any a

let view (A.Any a) ~strides shape =
  A.Any
    (A.v (A.dtype a)
       (A.Layout.v ~strides:(Iarray.of_array strides) (Iarray.of_array shape))
       (A.buffer a))

let host_load : K.load -> K.load = function
  | Plain x -> Plain (A.to_device Rig.host x)
  | Padded { x; fill; pad } -> Padded { x = A.to_device Rig.host x; fill; pad }

let bytes_of (A.Any a) =
  let h = A.copy (A.to_device Rig.host a) in
  let b = A.buffer h in
  let s = Bytes.create (Rig.Buffer.length b) in
  Rig.Buffer.blit_to_bytes b 0 s 0 (Bytes.length s);
  Bytes.to_string s

let first_difference (A.Any a) x y =
  let w = Int.max 1 (D.bits (A.dtype a) / 8) in
  let rec go i =
    if String.sub x i w <> String.sub y i w then i / w else go (i + w)
  in
  go 0

let wait_all dsts () =
  List.iter (fun (A.Any a) -> Rig.Buffer.wait (A.buffer a) Read) dsts

(* The copy of [bytes] on [g], from one fresh buffer into another, by nx.metal:
   what a call of nx.metal's costs that only moves a row's bytes. *)
let copy_parts g bytes =
  let n = bytes / 8 in
  let x = fresh g ~seed:3 (D.Any D.Float32) [| n |] in
  let y = fresh g (D.Any D.Float32) [| n |] in
  (prog [ x ] [ P.In 0 ] 0, x, y)

let copy_floor g bytes =
  let p, x, y = copy_parts g bytes in
  let run () =
    done_ "floor"
      (Nx_metal.map p ~dsts:(Iarray.of_list [ y ]) (Iarray.of_list [ plain x ]))
  in
  (run, wait_all [ y ])

(* Fails unless [run] gives nx.cpu's bits of [p]'s [l] over [loads] in
   [dsts]. *)
let check_cpu name run p l ~shape ~dsts loads =
  run ();
  let host =
    List.map
      (fun (A.Any a) ->
        A.Any (A.create Rig.host (A.dtype a) (A.Layout.shape (A.layout a))))
      dsts
  in
  done_ (name ^ " on nx.cpu")
    (apply (module Nx_cpu) p l ~shape ~dsts:host (List.map host_load loads));
  List.iteri
    (fun k (d, h) ->
      let x = bytes_of d and y = bytes_of h in
      if x <> y then
        failwith
          (strf "%s: destination %d differs from nx.cpu's at element %d" name k
             (first_difference d x y)))
    (List.combine dsts host)

(* A loop row: its program's [l] over [loads] into [dsts], checked against
   nx.cpu's bits where [check]. *)
let loop_run ~check name p l ~shape ~dsts loads =
  let run () = done_ name (apply (module Nx_metal) p l ~shape ~dsts loads) in
  if check then check_cpu name run p l ~shape ~dsts loads;
  (run, wait_all dsts)

(* The target of a cast row's op. *)
let cast_target = function
  | "cast_bf16" -> Some (D.Any D.Bfloat16)
  | "cast_f16" -> Some (D.Any D.Float16)
  | "cast_f32" -> Some (D.Any D.Float32)
  | "cast_i8" -> Some (D.Any D.Int8)
  | "cast_e4m3" -> Some (D.Any D.Float8_e4m3fn)
  | "cast_i32" -> Some (D.Any D.Int32)
  | "cast_i64" -> Some (D.Any D.Int64)
  | "cast_u8" -> Some (D.Any D.Uint8)
  | _ -> None

(* [t_cast_X]'s target: [cast_X]'s, cast from a transposed view. *)
let t_cast_target op =
  if String.starts_with ~prefix:"t_" op then
    cast_target (String.sub op 2 (String.length op - 2))
  else None

(* Loops: nx.cuda's loop rows, of the same programs and shapes. *)
let loop_op ~check op dt shape g =
  let rows_of () = Array.sub shape 0 (Array.length shape - 1) in
  let two () = (fresh g ~seed:1 dt shape, fresh g ~seed:2 dt shape) in
  let run_of = loop_run ~check op in
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
      run_of
        (prog [ x; y ] nodes out)
        Map ~shape ~dsts:[ z ]
        [ plain x; plain y ]
  | "copy_at1" ->
      (* The elements from position 1 on, a copy of no loop: its first byte lies
         1, 2 or 4 bytes past the destination's modulo 16, so the run joins two
         aligned loads a thread. *)
      let (A.Any xa) = fresh g ~seed:1 dt [| numel shape + 1 |] in
      let l =
        A.Layout.v ~offset:1 ~strides:(Iarray.of_list [ 1 ])
          (Iarray.of_array shape)
      in
      let x = A.Any (A.v (A.dtype xa) l (A.buffer xa)) in
      let z = fresh g dt shape in
      run_of (prog [ x ] [ P.In 0 ] 0) Map ~shape ~dsts:[ z ] [ plain x ]
  | "copy_rows" | "copy_step2" ->
      (* Elements apart, by copy: the first [cols] of each row of
         [row_elements], two axes; or every other element, one axis, a position
         with no division. *)
      let row_elements = 256 in
      let strides, span =
        if op = "copy_rows" then ([ row_elements; 1 ], shape.(0) * row_elements)
        else ([ 2 ], 2 * shape.(0))
      in
      let (A.Any xa) = fresh g ~seed:1 dt [| span |] in
      let l =
        A.Layout.v ~strides:(Iarray.of_list strides) (Iarray.of_array shape)
      in
      let x = A.Any (A.v (A.dtype xa) l (A.buffer xa)) in
      let z = fresh g dt shape in
      run_of (prog [ x ] [ P.In 0 ] 0) Map ~shape ~dsts:[ z ] [ plain x ]
  | "exp" | "cast_bf16" | "cast_f16" | "cast_f32" | "cast_i8" | "cast_e4m3"
  | "cast_i32" | "cast_i64" | "cast_u8" ->
      let x = fresh g ~seed:1 dt shape in
      let to_ = Option.value (cast_target op) ~default:dt in
      let z = fresh g to_ shape in
      let k = if op = "exp" then P.Unary Exp else P.Cast in
      run_of
        (prog [ x ] [ P.In 0; Op1 (k, to_, 0) ] 1)
        Map ~shape ~dsts:[ z ]
        [ plain x ]
  | "bcast_add" | "t_add" ->
      let rows = shape.(0) and cols = shape.(1) in
      let x = fresh g ~seed:1 dt shape in
      let y =
        if op = "bcast_add" then
          view (fresh g ~seed:2 dt [| cols |]) ~strides:[| 0; 1 |] shape
        else
          view
            (fresh g ~seed:2 dt [| cols; rows |])
            ~strides:[| 1; rows |] shape
      in
      let z = fresh g dt shape in
      run_of
        (prog [ x; y ] [ P.In 0; In 1; Op2 (Binary Add, 0, 1) ] 2)
        Map ~shape ~dsts:[ z ]
        [ plain x; plain y ]
  | op when t_cast_target op <> None ->
      (* A cast of a transposed view: its load reads a column a row. *)
      let rows = shape.(0) and cols = shape.(1) in
      let x =
        view (fresh g ~seed:1 dt [| cols; rows |]) ~strides:[| 1; rows |] shape
      in
      let to_ = Option.get (t_cast_target op) in
      let z = fresh g to_ shape in
      run_of
        (prog [ x ] [ P.In 0; Op1 (P.Cast, to_, 0) ] 1)
        Map ~shape ~dsts:[ z ]
        [ plain x ]
  | "pad1" | "pad9" ->
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
        K.Padded
          {
            x = xa;
            fill = D.of_float (A.dtype xa) 0.;
            pad = { lo; hi; interior = zeros; windows = Iarray.of_list [] };
          }
      in
      (* pad1 is one load shifted by a row and a column; pad9 the sum of the 9
         taps of a 3x3 window. *)
      if op = "pad1" then
        let p =
          P.v ~ins:(Iarray.of_list [ dt ])
            (Iarray.of_list [ P.In 0 ])
            ~outs:(Iarray.of_list [ 0 ])
        in
        run_of p Map ~shape ~dsts:[ z ] [ tap 0 0 ]
      else
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
        run_of p Map ~shape ~dsts:[ z ] loads
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
      run_of (prog [ x ] [ P.In 0 ] 0)
        (Reduce { axes = [ last ]; rs = [ (r, 0) ] })
        ~shape ~dsts
        [ plain x ]
  | "sqdiff" ->
      let x, y = two () in
      let z = fresh g dt (rows_of ()) in
      let nodes =
        [ P.In 0; In 1; Op2 (Binary Sub, 0, 1); Op2 (Binary Mul, 2, 2) ]
      in
      let last = Array.length shape - 1 in
      run_of
        (prog [ x; y ] nodes 3)
        (Reduce { axes = [ last ]; rs = [ (Monoid Sum, 0) ] })
        ~shape ~dsts:[ z ]
        [ plain x; plain y ]
  | "window2" ->
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
      run_of (prog [ x ] [ P.In 0 ] 0)
        (Reduce { axes = [ rank; rank + 1 ]; rs = [ (Monoid Sum, 0) ] })
        ~shape:wshape ~dsts:[ z ] [ load ]
  | "logcumsumexp" | "sqcumsum" ->
      let x = fresh g ~seed:1 dt shape in
      let z = fresh g dt shape in
      let p, r =
        if op = "sqcumsum" then
          (prog [ x ] [ P.In 0; Op2 (Binary Mul, 0, 0) ] 1, K.Monoid Sum)
        else (prog [ x ] [ P.In 0 ] 0, K.Logsumexp)
      in
      run_of p
        (Scan { axis = Array.length shape - 1; r = (r, 0) })
        ~shape ~dsts:[ z ]
        [ plain x ]
  | "expcumsum_axis0" ->
      let x = fresh g ~seed:1 dt shape in
      let z = fresh g dt shape in
      run_of
        (prog [ x ] [ P.In 0; Op1 (Unary Exp, dt, 0) ] 1)
        (Scan { axis = 0; r = (Monoid Sum, 0) })
        ~shape ~dsts:[ z ]
        [ plain x ]
  | "cummoments_explog16" ->
      let x = fresh g ~seed:1 dt shape in
      let mean = fresh g dt shape and var = fresh g dt shape in
      (* 16 Exps, each followed by a Log: values stay near the load's. *)
      let nodes =
        P.In 0
        :: List.init 32 (fun i ->
            P.Op1 (Unary (if i mod 2 = 0 then Exp else Log), dt, i))
      in
      run_of (prog [ x ] nodes 32)
        (Scan { axis = Array.length shape - 1; r = (Moments, 0) })
        ~shape ~dsts:[ mean; var ]
        [ plain x ]
  | "maxchain1024" ->
      (* A chain of 1024 Maximums, each of the last and the load: a program
         whose nodes' dtypes each follow the chain back. *)
      let x = fresh g ~seed:1 dt shape in
      let z = fresh g dt shape in
      let nodes =
        P.In 0 :: List.init 1024 (fun i -> P.Op2 (Binary Maximum, i, 0))
      in
      run_of (prog [ x ] nodes 1024) Map ~shape ~dsts:[ z ] [ plain x ]
  | _ -> invalid_arg ("Rows.loop_op: " ^ op)

let loop_row kind op dt shape =
  let name = strf "%s/%s-%s-%s" kind op (dname dt) (dims shape) in
  let make ~check g =
    let run, wait = loop_op ~check op dt shape g in
    { run; wait }
  in
  (* The bytes follow from the shapes alone: a row on no GPU. *)
  let bytes =
    let (D.Any d) = dt in
    let n = numel shape and esize = D.bits d / 8 in
    match op with
    | "add" | "where_lt" | "fused4" | "t_add" -> 3 * n * esize
    | "copy_at1" | "copy_rows" | "copy_step2" | "exp" | "pad1" | "pad9"
    | "logcumsumexp" | "sqcumsum" | "sqdiff" | "expcumsum_axis0"
    | "maxchain1024" ->
        2 * n * esize
    | "cummoments_explog16" -> 3 * n * esize
    | op when cast_target op <> None ->
        let (D.Any t) = Option.get (cast_target op) in
        n * (D.bits d + D.bits t) / 8
    | "bcast_add" -> (2 * n * esize) + (shape.(1) * esize)
    | op when t_cast_target op <> None ->
        let (D.Any t) = Option.get (t_cast_target op) in
        n * (D.bits d + D.bits t) / 8
    | "window2" -> n * esize * 5 / 4
    | "argmax" | "logsumexp" | "moments" -> n * esize
    | op -> invalid_arg ("Rows.loop_row: no bytes for " ^ op)
  in
  (* The large row's time over its floor's, measured on an M1 Max and rounded.
     Padded loads divide by every axis an element; a scan folds a chunk's terms
     on one thread; a transpose's time follows its elements, whatever their
     bytes. t_add's 4 at 8 bits and 2 at 16 are that known gap, not a target:
     the faster of PyTorch and MLX runs float16's 8192^2 near its floor, 0.71 of
     this row's time. A slowness past the row's leaves a case too few calls:
     each of its samples then holds the GPU's start from idle over a fraction of
     a millisecond of work. *)
  let slow =
    let (D.Any d) = dt in
    match op with
    | "add" | "copy_at1" | "copy_rows" | "copy_step2" -> 1
    | "t_add" -> ( match D.bits d with 8 -> 4 | 16 -> 2 | _ -> 1)
    | "exp" | "where_lt" | "bcast_add" -> 2
    | op when t_cast_target op <> None -> 2
    | op when cast_target op <> None -> 2
    | "fused4" -> 3
    | "argmax" -> 1
    | "sqdiff" -> 25
    | "logsumexp" | "moments" -> 2
    | "sqcumsum" -> 65
    | "logcumsumexp" -> 90
    | "expcumsum_axis0" -> 50
    | "cummoments_explog16" -> 120
    | "pad1" -> 4
    | "pad9" -> 100
    | "window2" -> 250
    | "maxchain1024" -> 100
    | op -> invalid_arg ("Rows.loop_row: no slowness for " ^ op)
  in
  let spec = strf "loop %s %s %s" op (dname dt) (dims shape) in
  { name; bytes; slow; make; spec }

(* The reductions and scans of a monoid over one operand. *)
let monoid_row ?slow kind (D.Any d as dt) (m : K.monoid) shape axes =
  let mname =
    match m with Sum -> "sum" | Prod -> "prod" | Max -> "max" | Min -> "min"
  in
  let name =
    strf "%s/%s-%s-%s-%s%s" kind (dname dt) mname (dims shape)
      (if kind = "reduce" then "axes" else "axis")
      (String.concat "," (List.map string_of_int axes))
  in
  let make ~check g =
    let x = fresh g ~seed:1 dt shape in
    let carrier = K.carrier dt in
    let l, out_shape =
      if kind = "reduce" then
        ( Reduce { axes; rs = [ (Monoid m, 0) ] },
          Array.of_list
            (List.filteri
               (fun i _ -> not (List.mem i axes))
               (Array.to_list shape)) )
      else (Scan { axis = List.hd axes; r = (Monoid m, 0) }, shape)
    in
    let z = fresh g carrier out_shape in
    let run, wait =
      loop_run ~check name (prog [ x ] [ P.In 0 ] 0) l ~shape ~dsts:[ z ]
        [ plain x ]
    in
    { run; wait }
  in
  let read = numel shape * D.bits d / 8 in
  let bytes = if kind = "reduce" then read else 2 * read in
  (* As [loop_row]'s: a bfloat16 sum's, 12.6, and a float32 max scan's, 71,
     unless given. *)
  let slow =
    match slow with Some s -> s | None -> if kind = "reduce" then 12 else 70
  in
  let spec =
    strf "%s %s %s %s %s" kind (dname dt) mname (dims shape)
      (String.concat "," (List.map string_of_int axes))
  in
  { name; bytes; slow; make; spec }

let loops =
  let open D in
  let f32 = Any Float32 and bf16 = Any Bfloat16 and f16 = Any Float16 in
  let u8 = Any Uint8 and u16 = Any Uint16 in
  let i8 = Any Int8 and e4m3 = Any Float8_e4m3fn in
  let i64 = Any Int64 and bool = Any Bool in
  let u4 = Any Uint4 and e2m1 = Any Float4_e2m1fn in
  let sized kind op dt (small, large) =
    [ loop_row kind op dt small; loop_row kind op dt large ]
  in
  let vec = ([| 1 lsl 20 |], [| 1 lsl 26 |]) in
  let square = ([| 1024; 1024 |], [| 8192; 8192 |]) in
  List.concat
    [
      sized "map" "add" f32 vec;
      (* One element, of one axis and of none: the cost of a call itself, its
         planning and its submission. *)
      [ loop_row "map" "add" f32 [| 1 |]; loop_row "map" "add" f32 [||] ];
      sized "map" "exp" f32 vec;
      sized "map" "copy_at1" u8 vec;
      sized "map" "copy_at1" u16 vec;
      sized "map" "copy_at1" f32 vec;
      (* 13.8 MB of uint8 rows, 210 of each 256 bytes, and every other byte of
         32 MB. *)
      [
        loop_row "map" "copy_rows" u8 [| 1 lsl 16; 210 |];
        loop_row "map" "copy_step2" u8 [| 1 lsl 24 |];
      ];
      sized "map" "add" bf16 ([| 1 lsl 21 |], [| 1 lsl 27 |]);
      sized "map" "cast_bf16" f32 vec;
      (* Casts of a byte into a wider float, by their kernels: a float8's
         decode, an integer's conversion and a store of 2 bytes from 1. *)
      sized "map" "cast_f16" e4m3 vec;
      sized "map" "cast_f32" i8 vec;
      sized "map" "cast_bf16" u8 vec;
      (* A cast for each class of the call sites that cast into integers and
         float8: a float saturated into a byte, a float8's encode, an integer
         narrowed and widened, a mask counted, and 4-bit codes unpacked into
         bytes and into float32, a store of 4 bytes from half a byte. *)
      sized "map" "cast_i8" f32 vec;
      sized "map" "cast_e4m3" f32 vec;
      sized "map" "cast_i32" i64 vec;
      sized "map" "cast_i64" u8 vec;
      sized "map" "cast_i64" bool vec;
      sized "map" "cast_u8" u4 vec;
      sized "map" "cast_f32" e2m1 vec;
      sized "map" "cast_i32" bool vec;
      sized "map" "bcast_add" f32 square;
      sized "map" "t_add" f32 square;
      sized "map" "t_add" f16 square;
      sized "map" "t_add" u8 square;
      (* A cast whose load is transposed, on the tape's tiles: the cast's kernel
         takes none and reads a line an element. *)
      sized "map" "t_cast_f16" f32 square;
      (* A transposed cast whose tape takes one element a thread. *)
      sized "map" "t_cast_i32" f32 square;
      sized "map" "fused4" f32 vec;
      sized "map" "pad1" f32 ([| 1; 64; 64; 64 |], [| 1; 64; 512; 512 |]);
      sized "map" "pad9" f32 ([| 1; 64; 64; 64 |], [| 1; 64; 512; 512 |]);
      (* Not lean: one element a thread, the build of loads read first holding
         planes that cut the groups a core holds. *)
      [ loop_row "map" "pad9" (Any Int64) [| 1024; 1024 |] ];
      sized "map" "where_lt" f32 vec;
      sized "reduce" "argmax" f32 square;
      sized "reduce" "logsumexp" f32 square;
      sized "reduce" "moments" f32 square;
      sized "reduce" "window2" f32 ([| 1; 16; 256; 256 |], [| 1; 64; 512; 512 |]);
      sized "reduce" "sqdiff" f32 square;
      sized "scan" "logcumsumexp" f32 ([| 1024; 1024 |], [| 4096; 4096 |]);
      (* One slice of 2^22 terms: a network over its blocks of 4096 terms,
         chained between an up and a down pass. *)
      [ loop_row "scan" "logcumsumexp" f32 [| 1 lsl 22 |] ];
      (* A scan's thread folds its terms in microseconds each: 4096^2 keeps a
         trial within thumper's deadline. *)
      sized "scan" "sqcumsum" f32 ([| 1024; 1024 |], [| 4096; 4096 |]);
      (* A costly scan whose results do not step by one, its chunk spread over
         threads by its program. *)
      [ loop_row "scan" "expcumsum_axis0" f32 [| 1024; 1024 |] ];
      (* Moments of a program of 32 costly nodes, its slices centered in a
         launch of their own: 64^2 is mostly the host's planning of the
         launches, 1024^2 their kernels. *)
      sized "scan" "cummoments_explog16" f32 ([| 64; 64 |], [| 1024; 1024 |]);
      (* A map of a long chain at 64^2: the host's planning of its launches. *)
      [ loop_row "map" "maxchain1024" f32 [| 64; 64 |] ];
      [
        monoid_row "reduce" bf16 Sum [| 2048; 1024 |] [ 1 ];
        monoid_row "reduce" bf16 Sum [| 16384; 8192 |] [ 1 ];
        (* Outputs of one term, which run as maps, at a copy's speed. *)
        monoid_row ~slow:1 "reduce" f32 Sum [| 1 |] [ 0 ];
        monoid_row ~slow:1 "reduce" f32 Sum [| 2; 1 |] [ 1 ];
        monoid_row ~slow:1 "reduce" f32 Sum [| 4096; 1 |] [ 1 ];
        monoid_row ~slow:1 "reduce" f32 Sum [| 1 lsl 20; 1 |] [ 1 ];
        monoid_row ~slow:1 "scan" f32 Max [| 1; 1 |] [ 1 ];
        monoid_row "scan" f32 Max [| 1024; 1024 |] [ 1 ];
        monoid_row "scan" f32 Max [| 4096; 4096 |] [ 1 ];
      ];
    ]

(* Contractions *)

(* The M1 Max's rates, by which a row's launches are estimated against its GPU
   time: 340 GB/s and 32 cores of 128 lanes at 1.296 GHz, each lane at
   nx.metal's fraction of an operation a cycle ({!Nx_metal.Plan.op_rate}). *)
let max_bandwidth = 340e9
let max_cores = 32
let max_clock = 1.296e9

let max_op_rate =
  max_clock *. Nx_metal.Plan.op_rate /. Nx_metal.Plan.nominal_clock

type contraction = {
  batch : int;
  m : int;
  n : int;
  k : int;
  dt : D.any;  (** [a]'s dtype, and [dst]'s where [b]'s is the same. *)
  db : D.any;  (** [b]'s dtype. *)
  nk : bool;  (** [b] stored [n] x [k], as in [x @ Wᵀ]. *)
}

(* Whether [c]'s operands are of two dtypes, a call of their casts into its
   stage's; its [dst] is then float32. *)
let mixed c = c.dt <> c.db
let dst_dtype c = if mixed c then D.Any D.Float32 else c.dt

let shapes c =
  let lead = if c.batch > 1 then [| c.batch |] else [||] in
  let s a b = Array.append lead [| a; b |] in
  (s c.m c.k, (if c.nk then s c.n c.k else s c.k c.n), s c.m c.n)

let pairs c =
  let r = if c.batch > 1 then 1 else 0 in
  let batch = if c.batch > 1 then [ (0, 0) ] else [] in
  (Iarray.of_list batch, Iarray.of_list [ (r + 1, if c.nk then r + 1 else r) ])

let operand x = { K.array = x; scale = None }
let complex c = dname c.dt = "complex64"

(* A row's accumulator: complex64 for complex operands, else float32. *)
let acc c = if complex c then D.Any D.Complex64 else D.Any D.Float32

let contract_on (module Kn : K.S) c (A.Any dst) a b =
  let batch, contracting = pairs c in
  Kn.contract ~batch ~contracting ~segments:None ~acc:(acc c) ~dst ~init:None
    (operand a) (operand b)

(* [x] as float32 on the host, through nx.cpu's cast. *)
let host_f32 ?(abs = false) (A.Any x) =
  let h = A.to_device Rig.host x in
  let shape = A.Layout.shape (A.layout h) in
  let y = A.create Rig.host D.Float32 shape in
  let f32 = D.Any D.Float32 in
  let nodes =
    P.In 0
    :: Op1 (Cast, f32, 0)
    :: (if abs then [ P.Op1 (Unary Abs, f32, 1) ] else [])
  in
  done_ "host_f32"
    (Nx_cpu.map
       (prog [ A.Any h ] nodes (List.length nodes - 1))
       ~dsts:(Iarray.of_list [ A.Any y ])
       (Iarray.of_list [ K.Plain h ]));
  y

(* The complex64 [x]'s |re| + |im|, as float32 on the host, through nx.cpu. A
   product of two such bounds each part's magnitudes in the product of the two
   numbers: |re re'| + |im im'| and |re im'| + |im re'|. *)
let host_parts (A.Any x) =
  let h = A.copy (A.to_device Rig.host x) in
  let shape = A.Layout.shape (A.layout h) in
  let r = Iarray.length shape in
  let pairs = Option.get (A.bitcast D.Float32 h) in
  let whole i = { A.Move.start = 0; count = Iarray.get shape i; step = 1 } in
  let part p =
    let at = { A.Move.start = p; count = 1; step = 1 } in
    let one = Array.append (Array.init r whole) [| at |] in
    let x = Option.get (A.move (A.Move.Slice (Iarray.of_array one)) pairs) in
    A.Any (Option.get (A.move (A.Move.Permute (Iarray.init r Fun.id)) x))
  in
  let re = part 0 and im = part 1 in
  let y = A.create Rig.host D.Float32 shape in
  let f32 = D.Any D.Float32 in
  let nodes =
    [
      P.In 0;
      In 1;
      Op1 (Unary Abs, f32, 0);
      Op1 (Unary Abs, f32, 1);
      Op2 (Binary Add, 2, 3);
    ]
  in
  done_ "host_parts"
    (Nx_cpu.map
       (prog [ re; im ] nodes 4)
       ~dsts:(Iarray.of_list [ A.Any y ])
       (Iarray.of_list [ plain re; plain im ]));
  y

(* Fails unless each part of nx.metal's complex64 [dst] is within nx_metal.mli's
   bound of nx.cpu's, which is within its own: [γ(4K + 1)] each of the sum over
   k of (|re a| + |im a|)(|re b| + |im b|), which bounds the part's magnitudes,
   and the rounding into [dst] of each. *)
let check_complex name c ~dst a b =
  let host (A.Any x) = A.Any (A.to_device Rig.host x) in
  let _, _, ys = shapes c in
  let shape = Iarray.of_array ys in
  let want = A.create Rig.host D.Complex64 shape in
  let mag = A.create Rig.host D.Float32 shape in
  done_ (name ^ " on nx.cpu")
    (contract_on (module Nx_cpu) c (A.Any want) (host a) (host b));
  let parts x = A.Any (host_parts x) in
  done_ (name ^ " magnitudes")
    (contract_on
       (module Nx_cpu)
       { c with dt = D.Any D.Float32; db = D.Any D.Float32 }
       (A.Any mag) (parts a) (parts b));
  let u = 0x1p-24 in
  let gamma n = Float.of_int n *. u /. (1. -. (Float.of_int n *. u)) in
  let (A.Any d) = dst in
  let got =
    A.to_array (A.expect D.Complex64 (A.Any (A.to_device Rig.host d)))
  in
  let want = A.to_array want and mag = A.to_array mag in
  let check i part y x =
    let bound =
      (2. *. gamma ((4 * c.k) + 1) *. mag.(i))
      +. (u *. (Float.abs y +. Float.abs x))
    in
    if not (Float.abs (x -. y) <= bound) then
      failwith
        (strf "%s: output %d's %s part is %h, nx.cpu's %h, past %h apart" name i
           part x y bound)
  in
  Array.iteri
    (fun i (y : Complex.t) ->
      check i "real" y.re got.(i).Complex.re;
      check i "imaginary" y.im got.(i).im)
    want

(* Fails unless each output of nx.metal's [dst], of [k] terms, is within
   nx_metal.mli's bound of nx.cpu's [want], which is within its own: [γ(2K + 1)]
   each of [Σ|a||b|], nx.cpu's [mag], and the rounding into [dst] of each. *)
let within name ~k ~dst want mag =
  let u_dst =
    match dname (any_dtype dst) with
    | "float16" -> 0x1p-11
    | "bfloat16" -> 0x1p-8
    | _ -> 0x1p-24
  in
  let u = 0x1p-24 in
  let gamma n = Float.of_int n *. u /. (1. -. (Float.of_int n *. u)) in
  let got = A.to_array (host_f32 dst) in
  let want = A.to_array want and mag = A.to_array mag in
  Array.iteri
    (fun i y ->
      let bound =
        (2. *. gamma ((2 * k) + 1) *. mag.(i))
        +. (u_dst *. (Float.abs y +. Float.abs got.(i)))
      in
      if not (Float.abs (got.(i) -. y) <= bound) then
        failwith
          (strf "%s: output %d is %h, nx.cpu's %h, past %h apart" name i got.(i)
             y bound))
    want

let check_real name c ~dst a b =
  let a32 = A.Any (host_f32 a) and b32 = A.Any (host_f32 b) in
  let _, _, ys = shapes c in
  let fresh_host () = A.create Rig.host D.Float32 (Iarray.of_array ys) in
  let want = fresh_host () and mag = fresh_host () in
  done_ (name ^ " on nx.cpu")
    (contract_on (module Nx_cpu) c (A.Any want) a32 b32);
  done_ (name ^ " magnitudes")
    (contract_on
       (module Nx_cpu)
       c (A.Any mag)
       (A.Any (host_f32 ~abs:true a))
       (A.Any (host_f32 ~abs:true b)));
  within name ~k:c.k ~dst want mag

let check_contraction name c ~dst a b =
  if complex c then check_complex name c ~dst a b
  else check_real name c ~dst a b

(* The GPU time of one submission of [run]'s call, in nanoseconds. *)
let submission run wait =
  let (), events =
    Rig.Profile.take (fun () ->
        run ();
        wait ())
  in
  List.fold_left
    (fun t -> function Rig.Profile.Work w -> t + (w.stop - w.start) | _ -> t)
    0 events

(* The calls a warm-up issues before it waits, as a timed case chains them.
   Calls that each wait for the one before leave the GPU idle between them, and
   from idle it takes about 0.45 s of such calls to reach its sustained clock;
   0.1 s of chained ones reach it. *)
let chained = 8

(* The chained calls, in nanoseconds, before [gpu_time]'s submissions. *)
let warm_up = 200_000_000

(* The GPU time of [run]'s submission, the least of [n] after [warm_up], in
   seconds: at the sustained clock, at which the estimates are stated. A command
   buffer's GPU span holds any other client's work the GPU runs meanwhile: one
   verify after another process's GPU run read 0.178 ms for a row of 0.069 in
   all three submissions. {!guarded} takes a reading past its estimate again
   [repause] later, which outlasts such a burst. *)
let gpu_time ?(n = 3) run wait =
  let rec warm until =
    for _ = 1 to chained do
      run ()
    done;
    wait ();
    if Rig.Profile.now () < until then warm until
  in
  warm (Rig.Profile.now () + warm_up);
  Float.of_int
    (List.fold_left Int.min max_int
       (List.init n (fun _ -> submission run wait)))
  *. 1e-9

(* The launch costs timed so far, by device. *)
let launch_costs = ref []

(* Seconds between two readings of one GPU time. *)
let repause = 0.5

(* Readings a launch's fixed cost takes the least of. *)
let cost_readings = 3

(* A launch's fixed cost on [g], which no estimate counts: the GPU time of
   nx.metal's copy of one element, the least of [cost_readings] readings
   [repause] apart. It is the device's, so it is timed at [g]'s first estimate
   and reused, and a guarded row times only its own call. A burst of another
   client's work in one reading would raise every estimate and could leave rows
   unguarded; the readings apart outlast it. *)
let launch_cost g =
  match List.assq_opt g !launch_costs with
  | Some cost -> cost
  | None ->
      let x = fresh g ~seed:3 (D.Any D.Float32) [| 1 |] in
      let y = fresh g (D.Any D.Float32) [| 1 |] in
      let p = prog [ x ] [ P.In 0 ] 0 in
      let run () =
        done_ "launch cost"
          (Nx_metal.map p ~dsts:(Iarray.of_list [ y ])
             (Iarray.of_list [ plain x ]))
      in
      let reading i =
        if i > 0 then Unix.sleepf repause;
        gpu_time ~n:5 run (wait_all [ y ])
      in
      let cost =
        List.fold_left Float.min infinity (List.init cost_readings reading)
      in
      launch_costs := (g, cost) :: !launch_costs;
      cost

(* Whether [l] is a vector kernel's launch. *)
let is_vector (l : Nx_metal.Plan.launch) =
  String.starts_with ~prefix:"contract_mv_" (Nx_metal.Kernels.name l.kernel)

(* The estimate of [plan]'s launches at the M1 Max's rates, each by its own cost
   and with a launch's fixed cost, and those fixed costs: a vector call's, whose
   launches state their plan's costs. *)
let own_estimate g (plan : Nx_metal.Plan.t) =
  let work =
    Array.fold_left
      (fun s (l : Nx_metal.Plan.launch) ->
        s
        +. Nx_metal.Plan.estimate ~bandwidth:max_bandwidth ~cores:max_cores
             ~op_rate:max_op_rate
             ~threads:(Nx_metal.Kernels.threads l.kernel)
             ~groups:l.groups ~resident:l.resident l.cost)
      0. plan.launches
  in
  let fixed = Float.of_int (Array.length plan.launches) *. launch_cost g in
  (work +. fixed, fixed)

(* The estimate of [c]'s launches at the M1 Max's rates, each with a launch's
   fixed cost, and those fixed costs: a vector call's by {!own_estimate}, a
   tiled one's tiles as their plan states them, a mixed one's as its casts' and
   the call of the casts'. *)
let rec contraction_estimate name g c ~dst a b =
  let batch, contracting = pairs c in
  let (A.Any d) = dst in
  match
    Nx_metal.Contract.call ~ops:[| A.Any d; a; b |] ~batch ~contracting
      ~segments:None ~acc:(acc c) ~dst:(A.Any d) ~init:None (operand a)
      (operand b)
  with
  | Ok (Planned (Some plan)) when Array.exists is_vector plan.launches ->
      own_estimate g plan
  | Ok (Recast { a = ca; b = cb; temp = _ }) ->
      mixed_estimate name g c ~dst (a, ca) (b, cb)
  | Ok (Planned _) | Error _ -> tiled_contraction_estimate name g c ~dst a b

(* The estimate of a call of the casts of [a] and [b] into the dtypes [ca] and
   [cb] where given: each cast's map, with the device's own bound on a group's
   threads, and the call of the casts. *)
and mixed_estimate name g c ~dst (a, ca) (b, cb) =
  let cast (A.Any x as ax) = function
    | None -> (ax, (0., 0.))
    | Some (D.Any d as dt) -> (
        let y = A.create (A.device x) d (A.Layout.shape (A.layout x)) in
        let p =
          K.Prog.v
            ~ins:(Iarray.of_list [ D.Any (A.dtype x) ])
            (Iarray.of_list K.Prog.[ In 0; Op1 (Cast, dt, 0) ])
            ~outs:(Iarray.of_list [ 1 ])
        in
        let xs =
          Nx_metal.Operands.v
            ~dsts:(Iarray.of_list [ A.Any y ])
            (Iarray.of_list [ plain ax ])
        in
        match Nx_metal.Loop.map ~holds:Nx_metal.Kernels.threads p xs with
        | Ok (Some plan) -> (A.Any y, own_estimate g plan)
        | Ok None | Error _ -> failwith (name ^ ": a cast plans nothing"))
  in
  let a, (wa, fa) = cast a ca and b, (wb, fb) = cast b cb in
  let (A.Any x) = a in
  let staged = { c with dt = D.Any (A.dtype x); db = D.Any (A.dtype x) } in
  let w, f = contraction_estimate name g staged ~dst a b in
  (w +. wa +. wb, f +. fa +. fb)

and tiled_contraction_estimate name g c ~dst a b =
  let module C = Nx_metal.Contract in
  let stage =
    match dname c.dt with "float16" -> C.F16 | "bfloat16" -> Bf16 | _ -> F32
  in
  let source = 0 in
  (* A complex call plans its real form: twice the columns and terms. *)
  let parts = if complex c then 2 else 1 in
  let nk = c.nk in
  let e =
    {
      Nx_metal.Contract.batch = c.batch;
      m = c.m;
      n = parts * c.n;
      k = parts * c.k;
      stage;
      nk;
      complex = complex c;
      source;
    }
  in
  let p =
    match C.plan e with Some p -> p | None -> failwith (name ^ ": no plan")
  in
  let tiles_kernel =
    if complex c then C.complex_kernel ~nk p.tile (C.fit e p)
    else C.kernel stage ~nk p.tile (C.fit e p)
  in
  let tiles =
    Array.fold_left
      (fun s l ->
        s
        +. C.estimate e p ~bandwidth:max_bandwidth ~cores:max_cores
             ~op_rate:max_op_rate l)
      0. p.launches
  in
  let batch, contracting = pairs c in
  let call =
    let (A.Any dst) = dst in
    C.call ~ops:[| A.Any dst; a; b |] ~batch ~contracting ~segments:None
      ~acc:(acc c) ~dst:(A.Any dst) ~init:None (operand a) (operand b)
  in
  let others =
    match call with
    | Ok (Nx_metal.Contract.Planned (Some (plan : Nx_metal.Plan.t))) ->
        Array.fold_left
          (fun s (l : Nx_metal.Plan.launch) ->
            if
              Nx_metal.Kernels.family l.kernel
              = Nx_metal.Kernels.family tiles_kernel
            then s
            else
              s
              +. Nx_metal.Plan.estimate ~bandwidth:max_bandwidth
                   ~cores:max_cores ~op_rate:max_op_rate
                   ~threads:(Nx_metal.Kernels.threads l.kernel)
                   ~groups:l.groups ~resident:l.resident l.cost)
          0. plan.launches
    | Ok (Planned None | Recast _) | Error _ ->
        failwith (name ^ ": the call plans nothing")
  in
  let launches =
    match call with
    | Ok (Nx_metal.Contract.Planned (Some plan)) -> Array.length plan.launches
    | _ -> 0
  in
  let fixed = Float.of_int launches *. launch_cost g in
  (tiles +. others +. fixed, fixed)

(* Readings of a guarded row past its estimate taken again, each [repause] after
   the one before. *)
let retimes = 2

(* The GPU time of [run]'s call against its estimate. A row whose GPU time
   passes it fails, the estimate erring high by design, unless the fixed costs
   are a tenth of it or more: a tile's launch starts slower than a copy's, by
   microseconds no estimate states. A reading past the estimate is taken again,
   up to [retimes] times, and the row fails only where every reading is past it.
   This Mac's GPU slows by 1.2 to 2 times for a few seconds at a time with no
   client of its own in sight: in 21 verifies, 15 rows read so in a whole
   reading, and each read its usual time on the same buffers 0.5 to 2.5 s later.
   A kernel past its estimate is past it at every reading, so it still fails. *)
let guarded name (estimate, fixed) run wait =
  let guarded = fixed < estimate /. 10. in
  let rec readings i acc =
    let past = List.for_all (fun g -> g > estimate) acc in
    if acc <> [] && ((not guarded) || (not past) || i > retimes) then
      List.rev acc
    else begin
      if i > 0 then Unix.sleepf repause;
      readings (i + 1) (gpu_time run wait :: acc)
    end
  in
  let all = readings 0 [] in
  let gpu = List.fold_left Float.min infinity all in
  let ms l =
    String.concat ", " (List.map (fun g -> strf "%.3f" (g *. 1e3)) l)
  in
  Printf.printf "%s: GPU %.3f ms, estimate %.3f ms on the M1 Max%s%s\n%!" name
    (gpu *. 1e3) (estimate *. 1e3)
    (if guarded then "" else ", its launches' fixed cost unguarded")
    (if List.length all > 1 then strf " (readings %s)" (ms all) else "");
  if guarded && gpu > estimate then
    failwith
      (strf "%s: GPU %s ms past its estimate %.3f ms" name (ms all)
         (estimate *. 1e3))

let guard name g c ~dst a b run wait =
  guarded name (contraction_estimate name g c ~dst a b) run wait

(* The low clock

   plan.ml: a launch after the GPU idles starts at down to about 0.3 of its
   sustained clock, so its work takes up to about 3 times its estimate, which
   the time budget's margin leaves room for. The check times one call of a
   compute-bound row and one of a copy, each after [idle] seconds without work,
   against its estimate over [low]. *)
let low = 0.3
let idle = 3.

let cold name run wait estimate =
  run ();
  wait ();
  Unix.sleepf idle;
  let gpu = Float.of_int (submission run wait) *. 1e-9 in
  let most = estimate /. low in
  Printf.printf "%s after %.0f s idle: GPU %.3f ms, at most %.3f ms\n%!" name
    idle (gpu *. 1e3) (most *. 1e3);
  if gpu > most then
    failwith
      (strf "%s: GPU %.3f ms after idling, past %.3f ms" name (gpu *. 1e3)
         (most *. 1e3))

let low_clock g =
  let c =
    {
      batch = 1;
      m = 1024;
      n = 1024;
      k = 1024;
      dt = D.Any D.Float32;
      db = D.Any D.Float32;
      nk = false;
    }
  in
  let sa, sb, sy = shapes c in
  let a = fresh g ~seed:1 c.dt sa and b = fresh g ~seed:2 c.dt sb in
  let dst = fresh g c.dt sy in
  let run () = done_ "low clock" (contract_on (module Nx_metal) c dst a b) in
  let estimate, _ = contraction_estimate "low clock" g c ~dst a b in
  cold "contract/float32-1024x1024x1024" run (wait_all [ dst ]) estimate;
  let bytes = 32 lsl 20 in
  let p, x, y = copy_parts g bytes in
  let dsts = Iarray.of_list [ y ] and loads = Iarray.of_list [ plain x ] in
  let run () = done_ "low clock copy" (Nx_metal.map p ~dsts loads) in
  let estimate =
    match Nx_metal.Moves.map p (Nx_metal.Operands.v ~dsts loads) with
    | Ok (Some plan) ->
        Array.fold_left
          (fun s (l : Nx_metal.Plan.launch) ->
            s
            +. Nx_metal.Plan.estimate ~bandwidth:max_bandwidth ~cores:max_cores
                 ~op_rate:max_op_rate
                 ~threads:(Nx_metal.Kernels.threads l.kernel)
                 ~groups:l.groups ~resident:l.resident l.cost)
          0. plan.launches
        +. (Float.of_int (Array.length plan.launches) *. launch_cost g)
    | Ok None | Error _ -> failwith "low clock: the copy plans nothing"
  in
  cold (strf "floor/copy-%d" bytes) run (wait_all [ y ]) estimate

let contract_row ?(batch = 1) ?(nk = false) ?db ~slow dt m n k =
  let db = Option.value db ~default:dt in
  let c = { batch; m; n; k; dt; db; nk } in
  let name =
    strf "contract/%s-%s%s%dx%dx%d%s" (dname dt)
      (if mixed c then dname db ^ "-" else "")
      (if batch > 1 then strf "%dx" batch else "")
      m n k
      (if nk then "-nk" else "")
  in
  let sa, sb, sy = shapes c in
  let make ~check g =
    let a = fresh g ~seed:1 dt sa and b = fresh g ~seed:2 db sb in
    let dst = fresh g (dst_dtype c) sy in
    let run () = done_ name (contract_on (module Nx_metal) c dst a b) in
    let wait = wait_all [ dst ] in
    if check then begin
      run ();
      wait ();
      check_contraction name c ~dst a b;
      guard name g c ~dst a b run wait
    end;
    { run; wait }
  in
  let bytes_of n (D.Any d) = n * D.bits d / 8 in
  let bytes =
    bytes_of (numel sa) dt
    + bytes_of (numel sb) db
    + bytes_of (numel sy) (dst_dtype c)
  in
  let spec =
    strf "contract %s %s %s %s %d %d %d %d k%s" (dname dt) (dname db)
      (dname (acc c))
      (dname (dst_dtype c))
      c.batch c.m c.n c.k
      (if c.nk then "k" else "f")
  in
  { name; bytes; slow; make; spec }

(* The bars at 4096³ at each staged dtype and b's layout; mid sizes; 768³ on
   both sides of the 64 x 64 tiles' threshold; small cubes, whose time is the
   host's; batches; 16, 8 and 1 rows by a 4096² weight, as a decode step reads
   it; 1000³, whose tiles take the edge function; and 64 ranges of k at 64 x 64
   x 65536; complex64 at 1024³ and 16 rows by a 4096² weight stored n x k. Each
   [slow] is as [loop_row]'s. *)
let contractions =
  let open D in
  let f32 = Any Float32 and f16 = Any Float16 and bf16 = Any Bfloat16 in
  let c64 = Any Complex64 in
  let e4m3 = Any Float8_e4m3fn and i8 = Any Int8 and i64 = Any Int64 in
  let cube ?nk ~slow dt n = contract_row ?nk ~slow dt n n n in
  [
    cube ~slow:27 f32 4096;
    cube ~slow:49 f16 4096;
    cube ~slow:60 bf16 4096;
    cube ~nk:true ~slow:27 f32 4096;
    cube ~nk:true ~slow:50 f16 4096;
    cube ~nk:true ~slow:61 bf16 4096;
    cube ~slow:16 f32 2048;
    cube ~slow:8 f32 1024;
    cube ~slow:14 f16 1024;
    cube ~slow:7 f32 768;
    cube ~slow:10 f16 768;
    cube ~slow:13 bf16 768;
    cube ~slow:3 f32 512;
    cube ~slow:3 f32 256;
    cube ~slow:3 f32 128;
    cube ~slow:2 f32 64;
    contract_row ~batch:64 ~slow:2 f32 256 256 256;
    contract_row ~batch:32 ~slow:9 f16 512 512 64;
    contract_row ~slow:1 f32 16 4096 4096;
    contract_row ~nk:true ~slow:1 f32 16 4096 4096;
    contract_row ~slow:1 f32 8 4096 4096;
    contract_row ~nk:true ~slow:1 f32 8 4096 4096;
    contract_row ~slow:1 f32 1 4096 4096;
    contract_row ~nk:true ~slow:1 f32 1 4096 4096;
    cube ~slow:9 f32 1000;
    contract_row ~slow:1 f32 64 64 65536;
    cube ~slow:17 c64 1024;
    contract_row ~nk:true ~slow:14 c64 16 4096 4096;
    contract_row ~db:f16 ~slow:40 e4m3 1024 1024 1024;
    contract_row ~db:bf16 ~slow:20 i8 1024 1024 1024;
    contract_row ~db:f32 ~slow:20 i64 1024 1024 1024;
    contract_row ~nk:true ~db:e4m3 ~slow:4 f16 16 4096 4096;
  ]

(* The estimate of [plan]'s launches at the M1 Max's rates, each with a launch's
   fixed cost, and those fixed costs: the launches [tile] picks as the plan of
   [e] states them, grouped where [groups] is not 0, the others by their own
   costs. *)
let tiled_estimate name g e ~groups ~tile (plan : Nx_metal.Plan.t) =
  let module C = Nx_metal.Contract in
  let p =
    match if groups > 0 then C.grouped e ~groups else C.plan e with
    | Some p -> p
    | None -> failwith (name ^ ": no plan")
  in
  let tiles =
    Array.fold_left
      (fun s l ->
        s
        +. C.estimate e p ~bandwidth:max_bandwidth ~cores:max_cores
             ~op_rate:max_op_rate l)
      0. p.launches
  in
  let others =
    Array.fold_left
      (fun s (l : Nx_metal.Plan.launch) ->
        if tile l then s
        else
          s
          +. Nx_metal.Plan.estimate ~bandwidth:max_bandwidth ~cores:max_cores
               ~op_rate:max_op_rate
               ~threads:(Nx_metal.Kernels.threads l.kernel)
               ~groups:l.groups ~resident:l.resident l.cost)
      0. plan.launches
  in
  let fixed = Float.of_int (Array.length plan.launches) *. launch_cost g in
  (tiles +. others +. fixed, fixed)

let plan_of name = function
  | Ok (Nx_metal.Contract.Planned (Some plan)) -> plan
  | Ok (Planned None | Recast _) | Error _ ->
      failwith (name ^ ": the call plans nothing")

(* Rows segments *)

(* Ids as a spec gives them, separated by [,]. *)
let ids_of ids = String.concat "," (List.map string_of_int (Array.to_list ids))

(* [t] ids in [0, g), drawn by x' = 1103515245 x + 12345 mod 2^31 from x = 1,
   each its bits from 16 on mod g: the low bits of such a generator repeat with
   short periods. *)
let random_ids ~t ~g =
  let x = ref 1 in
  Array.init t (fun _ ->
      x := ((1103515245 * !x) + 12345) land 0x7fff_ffff;
      (!x lsr 16) mod g)

(* A Rows call of [dt]: a's [t] rows of [k] terms, row i by group ids.(i) of [g]
   groups of b, each [k] x [n], or [n] x [k] where [nk]. *)
type grouped = { gdt : D.any; t : int; g : int; gn : int; gk : int; gnk : bool }

let grouped_on (module Kn : K.S) c ~dst ~ids a b =
  let (A.Any dst) = dst in
  Kn.contract ~batch:(Iarray.of_list [])
    ~contracting:(Iarray.of_list [ (1, if c.gnk then 2 else 1) ])
    ~segments:(Some (K.Rows { ids; a_axis = 0; b_axis = 0 }))
    ~acc:(D.Any D.Float32) ~dst ~init:None (operand a) (operand b)

let grouped_row ~slow c =
  let ids = random_ids ~t:c.t ~g:c.g in
  let name =
    strf "contract/%s-rows-of-%d-%dx%dx%d%s" (dname c.gdt) c.g c.t c.gn c.gk
      (if c.gnk then "-nk" else "")
  in
  let b_shape =
    if c.gnk then [| c.g; c.gn; c.gk |] else [| c.g; c.gk; c.gn |]
  in
  let make ~check g =
    let a = fresh g ~seed:1 c.gdt [| c.t; c.gk |] in
    let b = fresh g ~seed:2 c.gdt b_shape in
    let (D.Any d) = c.gdt in
    let dst = A.Any (A.create g d (Iarray.of_list [ c.t; c.gn ])) in
    let ids =
      A.to_device g
        (A.of_array D.Int64 (Iarray.of_list [ c.t ])
           (Array.map Int64.of_int ids))
    in
    let run () = done_ name (grouped_on (module Nx_metal) c ~dst ~ids a b) in
    let wait = wait_all [ dst ] in
    if check then begin
      run ();
      wait ();
      let host (A.Any x) = A.Any (A.to_device Rig.host x) in
      let hids = A.to_device Rig.host ids in
      let fresh_host () =
        A.create Rig.host D.Float32 (Iarray.of_list [ c.t; c.gn ])
      in
      let want = fresh_host () and mag = fresh_host () in
      done_ (name ^ " on nx.cpu")
        (grouped_on
           (module Nx_cpu)
           c ~dst:(A.Any want) ~ids:hids (host a) (host b));
      done_ (name ^ " magnitudes")
        (grouped_on
           (module Nx_cpu)
           c ~dst:(A.Any mag) ~ids:hids
           (A.Any (host_f32 ~abs:true a))
           (A.Any (host_f32 ~abs:true b)));
      within name ~k:c.gk ~dst want mag;
      let plan =
        plan_of name
          (let (A.Any dst) = dst in
           Nx_metal.Contract.call
             ~ops:[| A.Any dst; a; b; A.Any ids |]
             ~batch:(Iarray.of_list [])
             ~contracting:(Iarray.of_list [ (1, if c.gnk then 2 else 1) ])
             ~segments:(Some (K.Rows { ids; a_axis = 0; b_axis = 0 }))
             ~acc:(D.Any D.Float32) ~dst:(A.Any dst) ~init:None (operand a)
             (operand b))
      in
      let e =
        {
          Nx_metal.Contract.batch = 1;
          m = c.t;
          n = c.gn;
          k = c.gk;
          stage = F32;
          nk = c.gnk;
          complex = false;
          source = 0;
        }
      in
      let tile (l : Nx_metal.Plan.launch) =
        String.starts_with ~prefix:"contract_rows_"
          (Nx_metal.Kernels.name l.kernel)
      in
      guarded name (tiled_estimate name g e ~groups:c.g ~tile plan) run wait
    end;
    { run; wait }
  in
  (* The bytes a call moves: a, dst, ids, and the groups its ids name, each
     once. *)
  let (D.Any d) = c.gdt in
  let width = D.bits d / 8 in
  let named = List.length (List.sort_uniq compare (Array.to_list ids)) in
  let bytes =
    (named * c.gn * c.gk * width) + (c.t * (c.gk + c.gn) * width) + (8 * c.t)
  in
  let spec =
    strf "rows %s %d %d %d %d k%s %s" (dname c.gdt) c.g c.t c.gn c.gk
      (if c.gnk then "k" else "f")
      (ids_of ids)
  in
  { name; bytes; slow; make; spec }

(* A decode step over a mixture of 384 experts of 256 x 256: 128 rows, each its
   own expert drawn at random, both weight layouts. *)
let groupeds =
  let f32 = D.Any D.Float32 in
  List.map
    (fun gnk ->
      grouped_row ~slow:1
        { gdt = f32; t = 128; g = 384; gn = 256; gk = 256; gnk })
    [ false; true ]

(* Weights as a model stores them *)

type format = Mxfp4 | Q8_0 | Q6_k

(* A decode step's product by a weight of [format], as Nx_quant's products call
   it: [dt]'s rows by the weight's [n] rows of [k] terms, b stored [n] x [k], or
   [k] x [n] where [kn], as no model stores it; or, of [experts] such weights,
   each row by the expert [ids] names, in a Rows call. *)
type decode = {
  format : format;
  dt : D.any;
  experts : int;  (** 0 for an unsegmented call. *)
  ids : int array;
      (** Each row's expert; of an unsegmented call, one entry a row, or [[||]]
          and one row. *)
  n : int;
  k : int;
  kn : bool;
}

let rows_of c = Int.max 1 (Array.length c.ids)
let lead c = if c.experts > 0 then [ c.experts ] else []

(* A string of [bytes] bytes, byte [i] the low 8 bits of [f i]. *)
let pattern bytes f =
  let s = Bytes.create bytes in
  for i = 0 to bytes - 1 do
    Bytes.unsafe_set s i (Char.unsafe_chr (f i land 0xff))
  done;
  Bytes.unsafe_to_string s

(* [pattern]'s bytes of [experts]' parts of [per] bytes each, in that order:
   part [e] is bytes [e * per] to [(e + 1) * per - 1] of the whole. *)
let pattern_of experts per f =
  let s = Bytes.create (List.length experts * per) in
  List.iteri
    (fun at e ->
      for i = 0 to per - 1 do
        Bytes.unsafe_set s
          ((at * per) + i)
          (Char.unsafe_chr (f ((e * per) + i) land 0xff))
      done)
    experts;
  Bytes.unsafe_to_string s

let bytes_on g shape s =
  let x = A.create g D.Uint8 (Iarray.of_list shape) in
  Rig.Buffer.blit_from_string s 0 (A.buffer x) 0 (String.length s);
  x

(* [c]'s weights on [g], as nx_quant stores them, and their scale: MXFP4's
   float4_e2m1fn codes two a byte, the first in the low bits, by an Exponent
   code a block of 32 terms; Q8_0's blocks of 34 bytes, a positive float16 scale
   then 32 int8 quants, the quants a view between the scales, or stored k x n
   its quants [k/32; 32; n] and scales [k/32; 1; n] apart; Q6_K's as Nx_quant's
   product unpacks them, int8 quants less 32 by a positive float32 scale a block
   of 16 terms. Where [abs], each code or quant is its magnitude. With
   [~experts], of a call of experts, only theirs, in that order, each as the
   whole weight holds it. *)
let weight ?experts g c ~abs =
  let experts =
    match experts with
    | Some es -> es
    | None -> List.init (Int.max 1 c.experts) Fun.id
  in
  let lead = if c.experts > 0 then [ List.length experts ] else [] in
  let picked per f = pattern_of experts per f in
  match c.format with
  | Mxfp4 ->
      let code i = (i * 37) + (i / 89) in
      let code i = if abs then code i land 0x77 else code i in
      let cols, terms = if c.kn then (c.k, c.n) else (c.n, c.k) in
      let packed =
        bytes_on g (lead @ [ cols; terms / 2 ]) (picked (c.n * c.k / 2) code)
      in
      let b = Option.get (A.bitcast D.Float4_e2m1fn packed) in
      let b =
        Option.get
          (A.move (Reshape (Iarray.of_list (lead @ [ cols; terms ]))) b)
      in
      let exponent j = 122 + (j mod 11) in
      let scale = if c.kn then [ c.k / 32; c.n ] else [ c.n; c.k / 32 ] in
      let codes =
        bytes_on g (lead @ scale) (picked (c.n * c.k / 32) exponent)
      in
      (A.Any b, K.Exponent codes)
  | Q8_0 when c.kn ->
      let nb = c.k / 32 in
      let quant i =
        let q = (((i * 37) + (i / 251)) mod 255) - 127 in
        if abs then Int.abs q else q
      in
      let quants =
        bytes_on g (lead @ [ nb; 32; c.n ]) (picked (c.n * c.k) quant)
      in
      let b = Option.get (A.bitcast D.Int8 quants) in
      let byte i =
        let half = (15 - (i / 2 mod 5)) lsl 10 in
        if i mod 2 = 0 then half else half lsr 8
      in
      let halves =
        bytes_on g (lead @ [ nb; 1; c.n; 2 ]) (picked (c.n * nb * 2) byte)
      in
      let d = Option.get (A.bitcast D.Float16 halves) in
      let shape = Iarray.of_list (lead @ [ nb; 1; c.n ]) in
      let d = Option.get (A.move (Reshape shape) d) in
      (A.Any b, K.Value d)
  | Q8_0 ->
      let nb = c.k / 32 in
      let byte i =
        let block = i / 34 in
        let half = (15 - (block mod 5)) lsl 10 in
        match i mod 34 with
        | 0 -> half
        | 1 -> half lsr 8
        | _ ->
            let q = (((i * 37) + block) mod 255) - 127 in
            if abs then Int.abs q else q
      in
      let blocks =
        bytes_on g (lead @ [ c.n; nb; 34 ]) (picked (c.n * nb * 34) byte)
      in
      let whole =
        List.map
          (fun e -> { A.Move.start = 0; count = e; step = 1 })
          (lead @ [ c.n; nb ])
      in
      let slice start count =
        let at = { A.Move.start; count; step = 1 } in
        Option.get (A.move (Slice (Iarray.of_list (whole @ [ at ]))) blocks)
      in
      let b = Option.get (A.bitcast D.Int8 (slice 2 32)) in
      let d = Option.get (A.bitcast D.Float16 (slice 0 2)) in
      let d =
        Option.get (A.move (Reshape (Iarray.of_list (lead @ [ c.n; nb; 1 ]))) d)
      in
      (A.Any b, K.Value d)
  | Q6_k when c.kn -> invalid_arg "Rows.weight: Q6_K stored k x n"
  | Q6_k ->
      let quant i =
        let q = (((i * 37) + (i / 251)) mod 64) - 32 in
        if abs then Int.abs q else q
      in
      let b = A.create g D.Int8 (Iarray.of_list (lead @ [ c.n; c.k ])) in
      let quants = picked (c.n * c.k) quant in
      let bytes = String.length quants in
      Rig.Buffer.blit_from_string quants 0 (A.buffer b) 0 bytes;
      let power j = Float.ldexp 1. ((j mod 7) - 10) in
      let per = c.n * c.k / 16 in
      let scales =
        Array.concat
          (List.map
             (fun e -> Array.init per (fun j -> power ((e * per) + j)))
             experts)
      in
      let d =
        A.to_device g
          (A.of_array D.Float32
             (Iarray.of_list (lead @ [ c.n; c.k / 16 ]))
             scales)
      in
      (A.Any b, K.Value d)

let inner c =
  match c.format with Mxfp4 | Q6_k -> [ c.k ] | Q8_0 -> [ c.k / 32; 32 ]

(* [c]'s terms, a's axes after its row and b's after its rows, and its
   segments. *)
let contracting c =
  let q = List.length (inner c) and at = List.length (lead c) in
  let b_at = if c.kn then at else 1 + at in
  Iarray.of_list (List.init q (fun i -> (1 + i, b_at + i)))

let segments c ids =
  if c.experts = 0 then None else Some (K.Rows { ids; a_axis = 0; b_axis = 0 })

let decode_on (module Kn : K.S) c ~dst ~ids a (b, scale) =
  let (A.Any dst) = dst in
  Kn.contract ~batch:(Iarray.of_list []) ~contracting:(contracting c)
    ~segments:(segments c ids) ~acc:(D.Any D.Float32) ~dst ~init:None
    (operand a)
    { K.array = b; scale = Some scale }

(* The arrays a decode reads, after [dst], as the door numbers them. *)
let decode_ops c ~dst ~ids a b scale =
  let ids = if c.experts = 0 then [] else [ A.Any ids ] in
  let scale =
    match scale with K.Exponent e -> A.Any e | K.Value v -> A.Any v
  in
  Array.of_list ((dst :: a :: b :: ids) @ [ scale ])

let decode_plan name c ~dst ~ids a (b, scale) =
  plan_of name
    (Nx_metal.Contract.call
       ~ops:(decode_ops c ~dst ~ids a b scale)
       ~batch:(Iarray.of_list []) ~contracting:(contracting c)
       ~segments:(segments c ids) ~acc:(D.Any D.Float32) ~dst ~init:None
       (operand a)
       { K.array = b; scale = Some scale })

(* Fails unless nx.metal's [dst] is within its bound of nx.cpu's product of the
   same weight, its magnitudes those of the weight of each value's magnitude,
   its scales positive. *)
let check_decode name c ~dst a =
  let host (A.Any x) = A.Any (A.to_device Rig.host x) in
  (* The weights of the experts the rows name only, renumbered in order: of a
     mixture of 128 experts, 8 rows read 8 of them, and the whole weight took
     0.9 s to make for each of the two products. *)
  let named = List.sort_uniq compare (Array.to_list c.ids) in
  let experts = if c.experts = 0 then None else Some named in
  let renamed e = Option.get (List.find_index (Int.equal e) named) in
  let r =
    if c.experts = 0 then c
    else { c with experts = List.length named; ids = Array.map renamed c.ids }
  in
  let ids =
    A.of_array D.Int64
      (Iarray.of_list [ rows_of c ])
      (if c.experts > 0 then Array.map Int64.of_int r.ids
       else Array.make (rows_of c) 0L)
  in
  let fresh_host () =
    A.create Rig.host D.Float32 (Iarray.of_list [ rows_of c; c.n ])
  in
  let want = fresh_host () and mag = fresh_host () in
  done_ (name ^ " on nx.cpu")
    (decode_on
       (module Nx_cpu)
       r ~dst:(A.Any want) ~ids (host a)
       (weight ?experts Rig.host c ~abs:false));
  done_ (name ^ " magnitudes")
    (decode_on
       (module Nx_cpu)
       r ~dst:(A.Any mag) ~ids
       (A.Any (host_f32 ~abs:true a))
       (weight ?experts Rig.host c ~abs:true));
  within name ~k:c.k ~dst want mag

let is_loader (l : Nx_metal.Plan.launch) =
  let name = Nx_metal.Kernels.name l.kernel in
  List.exists
    (fun prefix -> String.starts_with ~prefix name)
    [ "contract_ld_"; "contract_rows_ld_" ]

(* Fails unless [plan] reads the weight through loaders or the vector kernels
   with no copy, widening or scale pass. *)
let check_reads name (plan : Nx_metal.Plan.t) =
  let moves =
    Array.exists
      (fun (l : Nx_metal.Plan.launch) ->
        match l.kernel with
        | Nx_metal.Kernels.Copy | Copy_8 | Copy_16 | Copy_32 | Copy_64 | Widen
        | Contract_scale ->
            true
        | _ -> false)
      plan.launches
  in
  let reads =
    Array.exists (fun l -> is_loader l || is_vector l) plan.launches
  in
  if moves || not reads then
    failwith (name ^ ": the weight is not read where it lies")

let decode_estimate name g c (plan : Nx_metal.Plan.t) =
  if Array.exists is_vector plan.launches then own_estimate g plan
  else
    let (D.Any d) = c.dt in
    let e =
      {
        Nx_metal.Contract.batch = 1;
        m = rows_of c;
        n = c.n;
        k = c.k;
        stage = F32;
        nk = true;
        complex = false;
        source = Int.max 1 (D.bits d / 8);
      }
    in
    tiled_estimate name g e ~groups:c.experts ~tile:is_loader plan

let decode_row ~slow c =
  let rows = rows_of c in
  let name =
    strf "contract/%s-%s-%s%dx%dx%d"
      (match c.format with Mxfp4 -> "mxfp4" | Q8_0 -> "q8_0" | Q6_k -> "q6_k")
      (dname c.dt)
      (if c.experts > 0 then strf "rows-of-%d-" c.experts else "")
      rows c.n c.k
    ^ if c.kn then "-kn" else ""
  in
  let make ~check g =
    let a = fresh g ~seed:1 c.dt (Array.of_list (rows :: inner c)) in
    let b = weight g c ~abs:false in
    let (D.Any d) = c.dt in
    let dst = A.Any (A.create g d (Iarray.of_list [ rows; c.n ])) in
    let ids =
      A.to_device g
        (A.of_array D.Int64 (Iarray.of_list [ rows ])
           (if c.experts > 0 then Array.map Int64.of_int c.ids
            else Array.make rows 0L))
    in
    let run () = done_ name (decode_on (module Nx_metal) c ~dst ~ids a b) in
    let wait = wait_all [ dst ] in
    if check then begin
      run ();
      wait ();
      check_decode name c ~dst a;
      let plan = decode_plan name c ~dst ~ids a b in
      check_reads name plan;
      guarded name (decode_estimate name g c plan) run wait
    end;
    { run; wait }
  in
  (* The bytes a call moves: a, dst, and the weights and scales of the experts
     its rows name, each once. *)
  let experts =
    if c.experts = 0 then 1
    else List.length (List.sort_uniq compare (Array.to_list c.ids))
  in
  let weight_bytes =
    match c.format with
    | Mxfp4 -> (c.n * c.k / 2) + (c.n * c.k / 32)
    | Q8_0 -> c.n * c.k / 32 * 34
    | Q6_k -> (c.n * c.k) + (c.n * c.k / 16 * 4)
  in
  let (D.Any d) = c.dt in
  let bytes = (experts * weight_bytes) + (rows * (c.k + c.n) * D.bits d / 8) in
  let spec =
    strf "decode %s %s %d %d %d %d k%s%s"
      (match c.format with
      | Mxfp4 -> "mxfp4"
      | Q8_0 -> "q8_0"
      | Q6_k -> "q6_k_int8")
      (dname c.dt) c.experts rows c.n c.k
      (if c.kn then "f" else "k")
      (if c.experts = 0 then "" else " " ^ ids_of c.ids)
  in
  { name; bytes; slow; make; spec }

(* Decode steps of the models' weights: gpt-oss-20b's MXFP4 experts at bfloat16,
   one expert's gate and up projection (5760 x 2880) as an unsegmented call, and
   a token's top 4 of 32 experts' down projections (2880 x 2880); a dense
   4096-wide model's Q8_0 projection at float32, and a mixture of 128 experts'
   top 8 of their 768 x 2048 Q8_0 gates; a 4096-wide Q6_K projection at float16,
   as its product unpacks it: the row times the contraction alone, its unpack
   untimed, and [quants]' row the whole step. The Q8_0 projection and the gate
   and up projection at 64 rows, a prompt's, time the tiles' loaders, which
   divide each element's term by its scale's and block's. At 16 rows, a batch of
   decodes', past the vector kernels' bounds, the gate and up projection stored
   k x n and n x k and the Q8_0 projection time the loaders; the Q8_0 projection
   stored k x n times the pass that writes its products with its scales, then
   the tiles. Each [slow] is as [loop_row]'s, measured on the M1 Max. *)
let decodes =
  let bf16 = D.Any D.Bfloat16
  and f16 = D.Any D.Float16
  and f32 = D.Any D.Float32 in
  [
    decode_row ~slow:2
      {
        format = Mxfp4;
        dt = bf16;
        experts = 0;
        ids = [||];
        n = 5760;
        k = 2880;
        kn = false;
      };
    decode_row ~slow:2
      {
        format = Mxfp4;
        dt = bf16;
        experts = 32;
        ids = [| 3; 11; 19; 27 |];
        n = 2880;
        k = 2880;
        kn = false;
      };
    decode_row ~slow:1
      {
        format = Q8_0;
        dt = f32;
        experts = 0;
        ids = [||];
        n = 4096;
        k = 4096;
        kn = false;
      };
    decode_row ~slow:1
      {
        format = Q8_0;
        dt = f32;
        experts = 128;
        ids = [| 5; 21; 37; 53; 69; 85; 101; 117 |];
        n = 768;
        k = 2048;
        kn = false;
      };
    decode_row ~slow:2
      {
        format = Q6_k;
        dt = f16;
        experts = 0;
        ids = [||];
        n = 4096;
        k = 4096;
        kn = false;
      };
    decode_row ~slow:20
      {
        format = Q8_0;
        dt = f32;
        experts = 0;
        ids = Array.make 64 0;
        n = 4096;
        k = 4096;
        kn = false;
      };
    decode_row ~slow:50
      {
        format = Mxfp4;
        dt = bf16;
        experts = 0;
        ids = Array.make 64 0;
        n = 5760;
        k = 2880;
        kn = false;
      };
    decode_row ~slow:16
      {
        format = Mxfp4;
        dt = bf16;
        experts = 0;
        ids = Array.make 16 0;
        n = 5760;
        k = 2880;
        kn = true;
      };
    decode_row ~slow:16
      {
        format = Mxfp4;
        dt = bf16;
        experts = 0;
        ids = Array.make 16 0;
        n = 5760;
        k = 2880;
        kn = false;
      };
    decode_row ~slow:3
      {
        format = Q8_0;
        dt = f32;
        experts = 0;
        ids = Array.make 16 0;
        n = 4096;
        k = 4096;
        kn = false;
      };
    decode_row ~slow:7
      {
        format = Q8_0;
        dt = f32;
        experts = 0;
        ids = Array.make 16 0;
        n = 4096;
        k = 4096;
        kn = true;
      };
  ]

(* A decode step as a user calls it, Nx_quant's product of a Q6_K weight in
   GGUF's packed blocks: each call unpacks the quants as int8 and the
   sub-blocks' scales at float32 through nx's operations, then contracts them,
   as [decodes]' Q6_K row times alone. A block's float16 [d] is 1/16, its scales
   and quants the pattern's; nx.cpu's product of the same blocks checks it. On
   the M1 Max a call takes 31.9 ms, 103 us of it the contraction: the unpack's
   33 launches take the rest, 99.7%. Its [slow] is that over its floor. *)
let quant_q6_k ~slow ~n ~k =
  let name = strf "quant/q6_k-float16-1x%dx%d" n k in
  let block = 210 and per = 256 in
  let packed = n * (k / per) * block in
  let byte i =
    match i mod block with
    | 208 -> 0x00
    | 209 -> 0x2C
    | _ -> (i * 37) + (i / 251)
  in
  let blocks_s = pattern packed byte in
  let make ~check g =
    let module M = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let on_set (type d) (v : d Nx.Devices.t) dev =
      let b = A.create dev D.Uint8 (Iarray.of_list [ n; k / per * block ]) in
      Rig.Buffer.blit_from_string blocks_s 0 (A.buffer b) 0 packed;
      Nx_quant.q6_k (Nx.Repr.of_array v b)
    in
    let w = on_set M.v g in
    let (A.Any x0) = fresh g ~seed:1 (D.Any D.Float16) [| 1; k |] in
    let x = Nx.Repr.of_array M.v (A.expect D.Float16 (A.Any x0)) in
    let last = ref None in
    let run () = last := Some (Nx_quant.apply w x) in
    let wait () = match !last with None -> () | Some y -> Nx.sync y in
    if check then begin
      run ();
      wait ();
      let wh = on_set Nx.Host.v Rig.host in
      let xh =
        Nx.Repr.of_array Nx.Host.v
          (A.to_device Rig.host (A.expect D.Float16 (A.Any x0)))
      in
      let x32 = Nx.cast Nx.float32 xh in
      let dq = Nx.matrix_transpose (Nx_quant.dequant Nx.float32 wh) in
      let want = Nx.matmul x32 dq
      and mag = Nx.matmul (Nx.abs x32) (Nx.abs dq) in
      let by = "Nx_quant.apply" in
      let host t = Nx.Repr.read ~by t (fun a -> A.to_device Rig.host a) in
      let got = Nx.Repr.read ~by (Option.get !last) (fun a -> A.Any a) in
      within name ~k ~dst:got (host want) (host mag)
    end;
    { run; wait }
  in
  let bytes = packed + ((k + n) * 2) in
  let spec = strf "decode q6_k float16 0 1 %d %d kk" n k in
  { name; bytes; slow; make; spec }

let quants = [ quant_q6_k ~slow:720 ~n:4096 ~k:4096 ]

(* Transforms *)

(* Fails unless each line of [y], [x]'s forward transform along [axis] of a
   matrix of [lines] lines of [n] points, rows where [axis] is 1, columns where
   it is 0, lies within the bound nx_metal.mli states of the transform nx.cpu
   computes at complex128, whose error lies far below it: [e 2^-24 √n] times the
   line's norm, [e = 5.35 log₂ n] for a 7-smooth [n] and [91 log₂ n + 236] for
   another, plus [c n 2^-126] for the flushing of subnormals, [c] the line's
   class's. *)
let check_bound name (x : (Complex.t, D.complex32_elt) A.t) y ~axis ~lines ~n =
  (* Each array's parts as floats, read through a bigarray: as arrays of
     Complex.t, the three of 2^24 points took 1.8 s to make. *)
  let module B = Bigarray.Array1 in
  let parts (type s) (d : (float, s) D.t) a =
    let g = A.bigarray (Option.get (A.bitcast d a)) in
    Bigarray.reshape_1 g (numel (Bigarray.Genarray.dims g))
  in
  let host a = A.copy (A.to_device Rig.host a) in
  let xs = parts D.Float32 (host x) and ys = parts D.Float32 (host y) in
  let shape = A.Layout.shape (A.layout x) in
  let wide = A.create Rig.host D.Complex128 shape in
  let ws = parts D.Float64 wide in
  for i = 0 to B.dim ws - 1 do
    B.unsafe_set ws i (B.unsafe_get xs i)
  done;
  let exact = A.create Rig.host D.Complex128 shape in
  done_ (name ^ " on nx.cpu")
    (Nx_cpu.c2c Forward ~axes:(Iarray.of_list [ axis ]) ~dst:exact wide);
  let es = parts D.Float64 exact in
  let fn = float n in
  let module F = Nx_metal.Fft in
  let factor, c =
    match (F.line n ~real:false).how with
    | Smooth -> (5.35 *. Float.log2 fn, F.flush_scale)
    | Rader -> ((91. *. Float.log2 fn) +. 236., F.rader_flush_scale)
    | Bluestein _ -> ((91. *. Float.log2 fn) +. 236., F.bluestein_flush_scale n)
  in
  let e = factor *. Float.ldexp 1. (-24) *. Float.sqrt fn in
  let flush = c *. fn *. Float.ldexp 1. (-126) in
  let at l j = if axis = 1 then (l * n) + j else (j * lines) + l in
  for l = 0 to lines - 1 do
    let norm = ref 0. and err = ref 0. in
    for j = 0 to n - 1 do
      let i = 2 * at l j in
      let re = B.unsafe_get xs i and im = B.unsafe_get xs (i + 1) in
      norm := !norm +. ((re *. re) +. (im *. im));
      let re = B.unsafe_get ys i -. B.unsafe_get es i in
      let im = B.unsafe_get ys (i + 1) -. B.unsafe_get es (i + 1) in
      err := !err +. ((re *. re) +. (im *. im))
    done;
    let bound = (e *. Float.sqrt !norm) +. flush and err = Float.sqrt !err in
    if not (err <= bound) then
      failwith
        (strf "%s: line %d lies %g from the exact transform, past %g" name l err
           bound)
  done

(* A forward c2c through Nx.fft of [lines] lines of [n] complex64 points, the
   rows of a [lines] by [n] matrix, or with [~columns] the columns of an [n] by
   [lines] one, or with [~pitch] the first [lines] columns of an [n] by [pitch]
   one; each point sin (0.7548776662 i + 1) at its [i]th float in C order of the
   matrix, on a set of [g] that nx.metal computes, into a fresh value each call,
   as a user calls it. *)
let fft_row ?(columns = false) ?pitch ~slow lines n =
  let axis = if columns then 0 else 1 in
  let shape = if columns then [| n; lines |] else [| lines; n |] in
  let held = match pitch with Some p -> [| n; p |] | None -> shape in
  let name =
    strf "fft/c2c-complex64-%s%s" (dims shape)
      (match pitch with
      | Some p -> strf "-columns-of-%dx%d" n p
      | None -> if columns then "-columns" else "")
  in
  let make ~check g =
    let module G = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let (A.Any floats) =
      fresh g ~seed:1 (D.Any D.Float32) [| held.(0); 2 * held.(1) |]
    in
    let x =
      A.v D.Complex64
        (A.Layout.v
           ~strides:(Iarray.of_list [ held.(1); 1 ])
           (Iarray.of_array shape))
        (A.buffer floats)
    in
    let xv = Nx.Repr.of_array G.v x in
    let last = ref xv in
    let run () = last := Nx.fft ~axes:[ axis ] xv in
    let wait () = Nx.sync !last in
    if check then begin
      run ();
      Nx.Repr.read ~by:"Nx.fft" !last (fun y ->
          check_bound name x y ~axis ~lines ~n)
    end;
    { run; wait }
  in
  (* The operand's bytes and the result's. *)
  let spec =
    strf "fft complex64 %s %d%s" (dims shape) axis
      (match pitch with Some _ -> " " ^ dims held | None -> "")
  in
  { name; bytes = 2 * 8 * lines * n; slow; make; spec }

(* The factor [e] and flushing factor [c] nx_metal.mli states for an [r2c]
   ([~inverse:false]) or a [c2r] along the last axis of [n] points: those of the
   line of [n / 2] points, plus 11 or 20, for an even [n]. *)
let real_factors ~inverse n =
  let module F = Nx_metal.Fft in
  let h = if n mod 2 = 0 then n / 2 else n in
  let fh = float h in
  let factor, c =
    match (F.line h ~real:(n mod 2 = 0)).how with
    | Smooth -> (5.35 *. Float.log2 fh, F.flush_scale)
    | Rader -> ((91. *. Float.log2 fh) +. 236., F.rader_flush_scale)
    | Bluestein _ -> ((91. *. Float.log2 fh) +. 236., F.bluestein_flush_scale h)
  in
  let more = if n mod 2 <> 0 then 0. else if inverse then 20. else 11. in
  (factor +. more, c)

(* Fails unless each of [lines] lines lies within [e 2^-24 √n ‖x‖₂ + c n 2^-126]
   of its exact values, scaled by [scale], [norm l] the squared norm of line
   [l]'s operand and [err l] the squared distance of its result. A scale other
   than 1 is a product of its own, whose output parts below [2^-126] flush: up
   to [2^-126] a point, [√n 2^-126] over a line. *)
let check_lines name ~lines ~n ~inverse ~scale ~norm ~err =
  let fn = float n in
  let factor, c = real_factors ~inverse n in
  let e = factor *. Float.ldexp 1. (-24) *. Float.sqrt fn in
  let flush = c *. fn *. Float.ldexp 1. (-126) in
  let scaled =
    if scale = 1. then 0. else Float.sqrt fn *. Float.ldexp 1. (-126)
  in
  for l = 0 to lines - 1 do
    let bound = (scale *. ((e *. Float.sqrt (norm l)) +. flush)) +. scaled in
    let err = Float.sqrt (err l) in
    if not (err <= bound) then
      failwith
        (strf "%s: line %d lies %g from the exact transform, past %g" name l err
           bound)
  done

(* An r2c through Nx.rfft of [lines] rows of [n] float32 points, drawn as
   [fft_row]'s, into a fresh value each call; checked against nx.cpu's
   complex128 c2c of the points, its first [n / 2 + 1] bins. *)
let rfft_row ~slow lines n =
  let shape = [| lines; n |] and bins = (n / 2) + 1 in
  let name = strf "fft/r2c-float32-%s" (dims shape) in
  let make ~check g =
    let module G = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let (A.Any x) = fresh g ~seed:1 (D.Any D.Float32) shape in
    let x = A.expect D.Float32 (A.Any x) in
    let xv = Nx.Repr.of_array G.v x in
    let last = ref (Nx.rfft Nx.complex64 ~axes:[ 1 ] xv) in
    let run () = last := Nx.rfft Nx.complex64 ~axes:[ 1 ] xv in
    let wait () = Nx.sync !last in
    if check then begin
      run ();
      Nx.Repr.read ~by:"Nx.rfft" !last (fun y ->
          let xs = A.to_array (A.copy (A.to_device Rig.host x)) in
          let wide =
            A.of_array D.Complex128 (Iarray.of_array shape)
              (Array.map (fun re -> { Complex.re; im = 0. }) xs)
          in
          let exact = A.create Rig.host D.Complex128 (Iarray.of_array shape) in
          done_ (name ^ " on nx.cpu")
            (Nx_cpu.c2c Forward ~axes:(Iarray.of_list [ 1 ]) ~dst:exact wide);
          let es = A.to_array exact and ys = A.to_array y in
          let sum f =
            let s = ref 0. in
            for j = 0 to n - 1 do
              s := !s +. f j
            done;
            !s
          in
          check_lines name ~lines ~n ~inverse:false ~scale:1.
            ~norm:(fun l -> sum (fun j -> xs.((l * n) + j) ** 2.))
            ~err:(fun l ->
              sum (fun j ->
                  if j >= bins then 0.
                  else
                    Complex.norm2
                      (Complex.sub ys.((l * bins) + j) es.((l * n) + j)))))
    end;
    { run; wait }
  in
  let spec = strf "rfft float32 %s 1" (dims shape) in
  { name; bytes = (4 * lines * n) + (8 * lines * bins); slow; make; spec }

(* A c2r through Nx.irfft of [lines] rows of [n / 2 + 1] complex64 bins, drawn
   as [fft_row]'s, into [n] float32 points a row, scaled by [1 / n] as
   Nx.irfft's [Backward]; checked against nx.cpu's complex128 c2c of the bins
   completed by [X(n - k) = conj X(k)], bin 0's and bin [n / 2]'s imaginary
   parts dropped, which never contribute. [n] is a power of two, so the scale
   rounds nothing. *)
let irfft_row ~slow lines n =
  let bins = (n / 2) + 1 in
  let shape = [| lines; bins |] in
  let name = strf "fft/c2r-complex64-%s-%d" (dims shape) n in
  let make ~check g =
    let module G = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let (A.Any x) = fresh g ~seed:1 (D.Any D.Complex64) shape in
    let x = A.expect D.Complex64 (A.Any x) in
    let xv = Nx.Repr.of_array G.v x in
    let irfft () = Nx.irfft Nx.float32 ~axes:[ 1 ] ~s:[ n ] xv in
    let last = ref (irfft ()) in
    let run () = last := irfft () in
    let wait () = Nx.sync !last in
    if check then begin
      run ();
      Nx.Repr.read ~by:"Nx.irfft" !last (fun y ->
          let xs = A.to_array (A.copy (A.to_device Rig.host x)) in
          let at l k =
            let b =
              if k < bins then xs.((l * bins) + k)
              else Complex.conj xs.((l * bins) + n - k)
            in
            if k = 0 || 2 * k = n then { b with im = 0. } else b
          in
          let full = Array.init (lines * n) (fun i -> at (i / n) (i mod n)) in
          let wide =
            A.of_array D.Complex128 (Iarray.of_list [ lines; n ]) full
          in
          let exact =
            A.create Rig.host D.Complex128 (Iarray.of_list [ lines; n ])
          in
          done_ (name ^ " on nx.cpu")
            (Nx_cpu.c2c Inverse ~axes:(Iarray.of_list [ 1 ]) ~dst:exact wide);
          let es = A.to_array exact and ys = A.to_array y in
          let fn = float n in
          let sum f =
            let s = ref 0. in
            for j = 0 to n - 1 do
              s := !s +. f j
            done;
            !s
          in
          check_lines name ~lines ~n ~inverse:true ~scale:(1. /. fn)
            ~norm:(fun l -> sum (fun j -> Complex.norm2 full.((l * n) + j)))
            ~err:(fun l ->
              sum (fun j ->
                  (ys.((l * n) + j) -. (es.((l * n) + j).re /. fn)) ** 2.)))
    end;
    { run; wait }
  in
  let spec = strf "irfft complex64 %s 1 %d" (dims shape) n in
  { name; bytes = (8 * lines * bins) + (4 * lines * n); slow; make; spec }

(* Fails unless [y], [x]'s forward transform along both axes of an [n] by [n]
   matrix, lies within the bound nx_metal.mli states of each axis's transform,
   summed over the two, of nx.cpu's complex128 transform: [2 · 5.35 log₂ n 2^-24
   n] times [x]'s norm, plus the flushing term of each axis's [n] lines; to
   first order, as the second axis's error on the first's, [(5.35 log₂ n 2^-24)²
   n] times the norm, is left out. *)
let check_bound2 name (x : (Complex.t, D.complex32_elt) A.t) y ~n =
  let x = A.copy (A.to_device Rig.host x) in
  let wide =
    A.of_array D.Complex128 (A.Layout.shape (A.layout x)) (A.to_array x)
  in
  let exact = A.create Rig.host D.Complex128 (A.Layout.shape (A.layout x)) in
  done_ (name ^ " on nx.cpu")
    (Nx_cpu.c2c Forward ~axes:(Iarray.of_list [ 0; 1 ]) ~dst:exact wide);
  let xs = A.to_array x and ys = A.to_array y and es = A.to_array exact in
  let fn = float n in
  let e = 2. *. 5.35 *. Float.log2 fn *. Float.ldexp 1. (-24) *. fn in
  let flush =
    2. *. fn *. Nx_metal.Fft.flush_scale *. fn *. Float.ldexp 1. (-126)
  in
  let norm = ref 0. and err = ref 0. in
  Array.iteri
    (fun i x ->
      norm := !norm +. Complex.norm2 x;
      err := !err +. Complex.norm2 (Complex.sub ys.(i) es.(i)))
    xs;
  let bound = (e *. Float.sqrt !norm) +. flush and err = Float.sqrt !err in
  if not (err <= bound) then
    failwith
      (strf "%s lies %g from the exact transform, past %g" name err bound)

(* A forward c2c through Nx.fft along both axes of an [n] by [n] matrix of
   complex64 points, drawn as [fft_row]'s, its columns side by side. *)
let fft2_row ~slow n =
  let shape = [| n; n |] in
  let name = strf "fft/c2c2-complex64-%s" (dims shape) in
  let make ~check g =
    let module G = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let (A.Any floats) = fresh g ~seed:1 (D.Any D.Float32) [| n; 2 * n |] in
    let x =
      A.v D.Complex64
        (A.Layout.contiguous (Iarray.of_array shape))
        (A.buffer floats)
    in
    let xv = Nx.Repr.of_array G.v x in
    let last = ref xv in
    let run () = last := Nx.fft ~axes:[ 0; 1 ] xv in
    let wait () = Nx.sync !last in
    if check then begin
      run ();
      Nx.Repr.read ~by:"Nx.fft" !last (fun y -> check_bound2 name x y ~n)
    end;
    { run; wait }
  in
  let spec = strf "fft complex64 %s 0,1" (dims shape) in
  { name; bytes = 2 * 8 * n * n; slow; make; spec }

(* Rows: about 1.1 times the floor's time, which rounds to 1, though the row
   misses its bar, the floor's time over 0.95: mostly by the gap between
   consecutive calls' kernels, which the floor pays too. Columns of 2 KiB and of
   8 KiB, the column step's widest, run its two launches; columns 128 KiB apart
   run the transposes' three, about 3 times the floor. A cut moved down slows
   the 8 KiB row about 1.9 times, one moved up the 128 KiB row about 1.14 times.
   A lone line of 1024 points and 4 columns of 512 run fewer lines than a group
   holds: 1 of 2 and, side by side, 4 of 8. *)
let transforms =
  [
    fft_row ~slow:9 1 1024;
    fft_row ~columns:true ~slow:9 4 512;
    fft_row ~slow:4 64 1024;
    fft_row ~slow:4 1024 1024;
    fft_row ~slow:2 16384 1024;
    fft_row ~slow:2 16384 64;
    fft_row ~slow:3 1024 1000;
    fft_row ~slow:3 64 1009;
    fft_row ~slow:3 64 1031;
    fft_row ~slow:3 1 65537;
    rfft_row ~slow:3 1024 1024;
    rfft_row ~slow:3 64 2062;
    irfft_row ~slow:3 1024 1024;
    fft_row ~columns:true ~slow:2 16384 64;
    fft2_row ~slow:8 512;
    fft2_row ~slow:8 1024;
    fft_row ~slow:3 1 1_000_000;
    fft_row ~slow:3 1 (1 lsl 24);
    fft_row ~slow:1 256 4096;
    fft_row ~columns:true ~slow:1 256 4096;
    fft_row ~columns:true ~slow:2 1024 4096;
    fft_row ~columns:true ~pitch:16384 ~slow:3 256 4096;
  ]

(* Factorisations *)

(* LU through Nx.lu of an [m] by [n] float32 matrix of fresh values, on a set of
   [g] that nx.metal computes, into fresh values each call, as a user calls it:
   its panels' rows over ⌈m / 128⌉ groups a matrix. The check: L U against the
   operand's rows that perm names, within n² ε max |U|, since |L| <= 1. *)
let lu_row ~slow m n =
  let name = strf "lu/float32-%s" (dims [| m; n |]) in
  let make ~check g =
    let module G = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let (A.Any floats) = fresh g ~seed:1 (D.Any D.Float32) [| m; n |] in
    let x =
      A.v D.Float32
        (A.Layout.v
           ~strides:(Iarray.of_list [ n; 1 ])
           (Iarray.of_array [| m; n |]))
        (A.buffer floats)
    in
    let xv = Nx.Repr.of_array G.v x in
    let last = ref (Nx.lu xv) in
    let run () = last := Nx.lu xv in
    let wait () =
      let p, l, u = !last in
      Nx.sync p;
      Nx.sync l;
      Nx.sync u
    in
    if check then begin
      (* On the host, which gathers A's rows. *)
      let host t =
        Nx.Repr.of_array Nx.Host.v
          (Nx.Repr.read ~by:"Nx.lu" t (fun a -> A.to_device Rig.host a))
      in
      let p, l, u = !last in
      let p = host p and l = host l and u = host u in
      let most x = Nx.item [] (Nx.max (Nx.abs x)) in
      let err = most (Nx.sub (Nx.matmul l u) (Nx.take ~axis:0 p (host xv))) in
      let bound = float (n * n) *. 0x1p-23 *. most u in
      if not (err <= bound) then
        failwith (strf "%s: |L U - P A| is %g, past %g" name err bound)
    end;
    { run; wait }
  in
  (* The operand's bytes and the factors'. *)
  let spec = strf "lu float32 %s" (dims [| m; n |]) in
  { name; bytes = 2 * 4 * m * n; slow; make; spec }

(* A tall LU, whose panel spans groups: leaves of 8 columns, each step over 64
   groups of about 1,600 rows. Its slow: 7.1 ms over the 0.092 ms of a copy of
   its bytes (the 48 MiB floor's, 1.36 ms for 15 copies). *)
let factorisations = [ lu_row ~slow:77 100_000 64 ]

(* Sorts *)

(* A fresh array of [d] on [g] whose bytes are drawn at random: integers of
   every value, floats of every bit pattern, NaNs included. *)
let random_bits g d shape =
  let a = A.create g d (Iarray.of_array shape) in
  let b = A.buffer a in
  let st = Random.State.make [| 5 |] in
  let n = Rig.Buffer.length b in
  let s = String.init n (fun _ -> Char.chr (Random.State.int st 256)) in
  Rig.Buffer.blit_from_string s 0 b 0 n;
  a

(* A sort of [dt] [shape] along [axis] by nx.metal, keeping [keep] of each slice
   where given, values and int64 positions, checked against nx.cpu's bits. A
   float32 element is sin (0.7548776662 i + 1) at its index [i] in C order;
   another dtype's bytes are drawn at random. Its floor's bytes: x read, then
   the values and positions kept written. *)
let sort_row ?(dt = D.Any D.Float32) ?keep ~slow shape axis =
  let (D.Any d) = dt in
  let n = shape.(axis) in
  let keep = Option.value keep ~default:n in
  let kept = Array.copy shape in
  kept.(axis) <- keep;
  let name =
    strf "sort/%s-%s%s%s" (D.name d) (dims shape)
      (if axis = Array.length shape - 1 then "" else strf "-axis%d" axis)
      (if keep = n then "" else strf "-keep%d" keep)
  in
  let make ~check g =
    let x =
      if dt = D.Any D.Float32 then A.expect d (fresh g ~seed:1 dt shape)
      else random_bits g d shape
    in
    let values = A.create g d (Iarray.of_array kept) in
    let positions = A.create g D.Int64 (Iarray.of_array kept) in
    let run () =
      done_ name (Nx_metal.sort ~axis ~descending:false ~values ~positions x)
    in
    let wait = wait_all [ A.Any values; A.Any positions ] in
    if check then begin
      run ();
      wait ();
      let hv = A.create Rig.host d (Iarray.of_array kept) in
      let hp = A.create Rig.host D.Int64 (Iarray.of_array kept) in
      done_ (name ^ " on nx.cpu")
        (Nx_cpu.sort ~axis ~descending:false ~values:hv ~positions:hp
           (A.to_device Rig.host x));
      List.iter
        (fun (d, h) ->
          let a = bytes_of d and b = bytes_of h in
          if a <> b then
            failwith
              (strf "%s: differs from nx.cpu's at element %d" name
                 (first_difference d a b)))
        [ (A.Any values, A.Any hv); (A.Any positions, A.Any hp) ]
    end;
    { run; wait }
  in
  let e = Int.max 1 (D.bits d / 8) in
  let bytes = (e * numel shape) + ((e + 8) * numel kept) in
  let spec = strf "sort %s %s %d %d" (D.name d) (dims shape) axis keep in
  { name; bytes; slow; make; spec }

(* Rows: each path, the block at one element, a slice and many slices, the radix
   at one and many slices of 16384 and at 2^20 and 2^24, the block along the
   first axis of a matrix, and selects of 8 and 1000; and 2^20 elements of 8, 2
   and 1 bytes by radix. *)
let sorts =
  [
    sort_row ~slow:3 [| 1 |] 0;
    sort_row ~slow:39 [| 1024 |] 0;
    sort_row ~slow:31 [| 1024; 1024 |] 1;
    sort_row ~slow:18 [| 4096; 4096 |] 0;
    sort_row ~slow:49 [| 64; 16384 |] 1;
    sort_row ~slow:18 [| 1024; 16384 |] 1;
    sort_row ~slow:53 [| 1 lsl 20 |] 0;
    sort_row ~slow:32 [| 1 lsl 24 |] 0;
    sort_row ~keep:8 ~slow:45 [| 64; 16384 |] 1;
    sort_row ~keep:1000 ~slow:18 [| 1 lsl 24 |] 0;
    sort_row ~dt:(D.Any D.Int64) ~slow:69 [| 1 lsl 20 |] 0;
    sort_row ~dt:(D.Any D.Float16) ~slow:34 [| 1 lsl 20 |] 0;
    sort_row ~dt:(D.Any D.Uint8) ~slow:21 [| 1 lsl 20 |] 0;
  ]

(* Moves *)

(* Gathers, range gathers and assemblies, at nx.cuda's moves rows' shapes, seeds
   and bytes, and a range gather of cells of one element, where a position's
   search costs most. *)

let ints_on g shape values =
  A.to_device g
    (A.of_array D.Int64 (Iarray.of_array shape) (Array.map Int64.of_int values))

let random ~seed n bound =
  let st = Random.State.make [| seed |] in
  Array.init n (fun _ -> Random.State.int st bound)

let typed d g ?seed shape = A.expect d (fresh g ?seed (D.Any d) shape)
let on_host x = A.to_device Rig.host x

(* A move's call [f] over [t] on nx.metal into [dst t]; checked once against
   nx.cpu's bits over [host t] where [check]. *)
let move_run ~check name ~dst ~host t f =
  let run () = done_ name (f (module Nx_metal : K.S) t) in
  if check then begin
    run ();
    let h = host t in
    done_ (name ^ " on nx.cpu") (f (module Nx_cpu : K.S) h);
    let x = bytes_of (dst t) and y = bytes_of (dst h) in
    if x <> y then
      failwith
        (strf "%s: differs from nx.cpu's at element %d" name
           (first_difference (dst t) x y))
  end;
  { run; wait = wait_all [ dst t ] }

let move_row ~slow name op dt ~bytes make =
  {
    name = "move/" ^ name;
    bytes;
    slow;
    make;
    spec = strf "move %s %s" op (dname dt);
  }

let fresh_like (a : (_, _) A.t) =
  A.create Rig.host (A.dtype a) (A.Layout.shape (A.layout a))

(* A gather along axis 0 of [x] at [idx] into [dst]. *)
let gather_run ~check name (dst, idx, x) ~axis =
  move_run ~check name
    ~dst:(fun (d, _, _) -> A.Any d)
    ~host:(fun (d, i, x) -> (fresh_like d, on_host i, on_host x))
    (dst, idx, x)
    (fun (module Kn : K.S) (dst, idx, x) -> Kn.gather ~axis ~dst idx x)

(* x [65536; 1024] at 16384 random rows, ids [16384] broadcast along the rows *)
let gather_emb (type v s) (d : (v, s) D.t) ~check g =
  let x = typed d g ~seed:1 [| 65536; 1024 |] in
  let ids = ints_on g [| 16384; 1 |] (random ~seed:2 16384 65536) in
  let idx =
    A.v D.Int64
      (A.Layout.v
         ~strides:(Iarray.of_list [ 1; 0 ])
         (Iarray.of_list [ 16384; 1024 ]))
      (A.buffer ids)
  in
  let dst = A.create g d (Iarray.of_list [ 16384; 1024 ]) in
  gather_run ~check "gather_emb" (dst, idx, x) ~axis:0

let gather_rand ~check g =
  let n = 1 lsl 24 in
  let x = typed D.Float32 g ~seed:1 [| n |] in
  let idx = ints_on g [| n |] (random ~seed:2 n n) in
  let dst = A.create g D.Float32 (Iarray.of_list [ n ]) in
  gather_run ~check "gather_rand" (dst, idx, x) ~axis:0

(* x [4096; 4096] at 64 random positions of each row: a top-k's reorder. *)
let take_along ~check g =
  let x = typed D.Float32 g ~seed:1 [| 4096; 4096 |] in
  let idx = ints_on g [| 4096; 64 |] (random ~seed:2 (4096 * 64) 4096) in
  let dst = A.create g D.Float32 (Iarray.of_list [ 4096; 64 ]) in
  gather_run ~check "take_along" (dst, idx, x) ~axis:1

(* [k] runs of 0 to 31 of [n] positions, from random starts: their starts, their
   ends and the positions they gather. *)
let ranges ~k n =
  let lens = random ~seed:3 k 32 and starts = random ~seed:4 k (n - 32) in
  let ends = Array.make k 0 in
  Array.iteri
    (fun i l -> ends.(i) <- (l + if i = 0 then 0 else ends.(i - 1)))
    lens;
  (starts, ends, ends.(k - 1))

(* 8192 runs of rows of x [262144; 256]. *)
let row_ranges = ranges ~k:8192 262144

(* The positions that [ranges ~k] gathers, without building its runs. *)
let ranges_size ~k =
  let st = Random.State.make [| 3 |] and size = ref 0 in
  for _ = 1 to k do
    size := !size + Random.State.int st 32
  done;
  !size

(* 2^20 runs of elements of x [2^24]: cells of one element. Built by the row
   that gathers them, so that no other row's process carries their 3 x 2^20
   words through each major collection. *)
let element_runs = 1 lsl 20
let element_ranges () = ranges ~k:element_runs (1 lsl 24)

let gather_ranges (starts, ends, size) xs ~check g =
  let k = Array.length starts in
  let x = typed D.Float32 g ~seed:1 xs in
  let starts = ints_on g [| k |] starts and ends = ints_on g [| k |] ends in
  let ds = Array.copy xs in
  ds.(0) <- size;
  let dst = A.create g D.Float32 (Iarray.of_array ds) in
  move_run ~check "gather_ranges"
    ~dst:(fun (d, _, _, _) -> A.Any d)
    ~host:(fun (d, s, e, x) -> (fresh_like d, on_host s, on_host e, on_host x))
    (dst, starts, ends, x)
    (fun (module Kn : K.S) (dst, starts, ends, x) ->
      Kn.gather_ranges ~dst ~starts ~ends x)

let whole n = { A.Move.start = 0; count = n; step = 1 }

(* [pieces], each with its region, into a fresh float32 [shape] *)
let assemble_run ~fill shape pieces ~check g =
  let pieces =
    Array.of_list
      (List.mapi
         (fun i (region, s) ->
           (Iarray.of_list region, typed D.Float32 g ~seed:(i + 1) s))
         pieces)
  in
  let dst = A.create g D.Float32 (Iarray.of_array shape) in
  move_run ~check "assemble"
    ~dst:(fun (d, _) -> A.Any d)
    ~host:(fun (d, ps) ->
      (fresh_like d, Array.map (fun (r, p) -> (r, on_host p)) ps))
    (dst, pieces)
    (fun (module Kn : K.S) (dst, pieces) ->
      Kn.assemble ~fill ~dst (Iarray.of_array pieces))

(* The fold into a fresh float32 [ys] of square windows of [size] taps at [step]
   along axes 2 and 3, padded by [by] on each side of them. *)
let fold_run ~size ~step ~by ys ~check g =
  let w axis = { A.Move.axis; size; step; dilation = 1 } in
  let pad =
    {
      K.lo = Iarray.of_list [ 0; 0; by; by ];
      hi = Iarray.of_list [ 0; 0; by; by ];
      interior = Iarray.of_list [ 0; 0; 0; 0 ];
      windows = Iarray.of_list [ w 2; w 3 ];
    }
  in
  let xs =
    match K.Shape.padded pad ys with
    | Ok xs -> xs
    | Error e -> invalid_arg ("Rows.fold_run: " ^ e)
  in
  let x = typed D.Float32 g ~seed:1 xs in
  let dst = A.create g D.Float32 (Iarray.of_array ys) in
  move_run ~check "fold"
    ~dst:(fun (d, _) -> A.Any d)
    ~host:(fun (d, x) -> (fresh_like d, on_host x))
    (dst, x)
    (fun (module Kn : K.S) (dst, x) -> Kn.fold pad ~dst x)

(* The adjoint of 3x3 windows at padding 1 over [32; 64; 64; 64], channels
   before rows: their 9 taps summed into each element. *)
let fold_col2im = fold_run ~size:3 ~step:1 ~by:1 [| 32; 64; 64; 64 |]

(* The patches of 16x16 at step 16 of [16; 64; 256; 256] put back in place, as a
   vision model's patches become an image: each element one tap's, moved into
   place. *)
let fold_patch = fold_run ~size:16 ~step:16 ~by:0 [| 16; 64; 256; 256 |]

(* The adjoint of 3x3 windows at step 2 and padding 1 over [32; 64; 128; 128], a
   strided convolution's gradient: each element 1, 2 or 4 taps', found at once
   since the dilation divides the step. *)
let fold_s2 = fold_run ~size:3 ~step:2 ~by:1 [| 32; 64; 128; 128 |]

(* Slows: each row's time over its floor's on the M1 Max, rounded down, at most
   the measured ratio, so that a case issues enough calls: gather_rand's a
   random read of each of 16M elements, take_along's of 256K, ranges' the
   positions' search and traffic. ranges_elem's 1 lies below its measured 10. *)
let moves =
  let f32 = D.Any D.Float32 and bf16 = D.Any D.Bfloat16 in
  let m = 1 lsl 24 and emb = 16384 * 1024 in
  let _, _, size = row_ranges and elements = ranges_size ~k:element_runs in
  [
    move_row ~slow:1 "gather_emb_f32" "gather_emb" f32 ~bytes:(8 * emb)
      (gather_emb D.Float32);
    move_row ~slow:1 "gather_emb_bf16" "gather_emb" bf16 ~bytes:(4 * emb)
      (gather_emb D.Bfloat16);
    move_row ~slow:20 "gather_rand_f32" "gather_rand" f32 ~bytes:(16 * m)
      gather_rand;
    move_row ~slow:3 "take_along_f32" "take_along" f32
      ~bytes:(4096 * 64 * 16)
      take_along;
    move_row ~slow:2 "ranges_f32" "ranges" f32
      ~bytes:(2 * 1024 * size)
      (gather_ranges row_ranges [| 262144; 256 |]);
    move_row ~slow:1 "ranges_elem_f32" "ranges_elem" f32
      ~bytes:(2 * 4 * elements)
      (fun ~check g ->
        gather_ranges (element_ranges ()) [| 1 lsl 24 |] ~check g);
    move_row ~slow:1 "assemble_cat_f32" "assemble_cat" f32
      ~bytes:(8 * 16384 * 4096)
      (assemble_run ~fill:0. [| 16384; 4096 |]
         (List.init 8 (fun i ->
              ( [
                  { A.Move.start = 2048 * i; count = 2048; step = 1 }; whole 4096;
                ],
                [| 2048; 4096 |] ))));
    move_row ~slow:1 "assemble_pad_f32" "assemble_pad" f32
      ~bytes:(4 * ((8192 * 8192) + (8200 * 8200)))
      (assemble_run ~fill:1.5 [| 8200; 8200 |]
         [
           ( [
               { A.Move.start = 4; count = 8192; step = 1 };
               { start = 4; count = 8192; step = 1 };
             ],
             [| 8192; 8192 |] );
         ]);
    move_row ~slow:1 "fold_col2im_f32" "fold_col2im" f32
      ~bytes:(4 * 32 * 64 * 64 * 64 * 10)
      fold_col2im;
    move_row ~slow:1 "fold_patch_f32" "fold_patch" f32
      ~bytes:(4 * 16 * 64 * 256 * 256 * 2)
      fold_patch;
    move_row ~slow:6 "fold_s2_f32" "fold_s2" f32
      ~bytes:(4 * ((32 * 64 * 128 * 128) + (32 * 64 * 64 * 64 * 9)))
      fold_s2;
  ]

(* Spectral routines *)

(* γ(m) of float32, m u / (1 − m u). *)
let gamma32 m =
  let mu = float m *. 0x1p-24 in
  mu /. (1. -. mu)

(* [x]'s Frobenius norm, in float64. *)
let frobenius x =
  Float.sqrt (Nx.item [] (Nx.sum (Nx.square (Nx.cast Nx.float64 x))))

(* [t] on the host, [by] naming the call that made it. *)
let to_host ~by t =
  Nx.Repr.of_array Nx.Host.v
    (Nx.Repr.read ~by t (fun a -> A.to_device Rig.host a))

(* A float32 [m] by [n] matrix of fresh values on [g], as [lu_row]'s. *)
let fresh_matrix g m n =
  let (A.Any floats) = fresh g ~seed:1 (D.Any D.Float32) [| m; n |] in
  A.v D.Float32
    (A.Layout.v ~strides:(Iarray.of_list [ n; 1 ]) (Iarray.of_array [| m; n |]))
    (A.buffer floats)

(* eigh through Nx.eigh, or Nx.eigvalsh where [values], of an [n] by [n] float32
   matrix of fresh values, its lower triangle read, into fresh values each call,
   as a user calls it. The check, by nx_metal.mli's bounds with g = γ(16n): with
   vectors, ‖a − v diag w vᵀ‖_F <= g ‖a‖_F and ‖vᵀv − I‖_F <= g; the values' sum
   within n g ‖a‖_F of a's trace, as each value is within g ‖a‖₂ of a's. The
   bounds' h = ν(·) term is dropped: below 2^-100 of them for fresh values near
   1. *)
let eigh_row ~values ~slow n =
  let name =
    strf "eigh/float32-%s%s" (dims [| n; n |]) (if values then "-w" else "")
  in
  let make ~check g =
    let module G = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let xv = Nx.Repr.of_array G.v (fresh_matrix g n n) in
    let call () =
      if values then (Nx.eigvalsh Nx.float32 xv, None)
      else
        let w, v = Nx.eigh Nx.float32 xv in
        (w, Some v)
    in
    let last = ref (call ()) in
    let run () = last := call () in
    let wait () =
      let w, v = !last in
      Nx.sync w;
      Option.iter Nx.sync v
    in
    if check then begin
      let x = to_host ~by:"Nx.eigh" xv in
      let lower = Nx.tril ~k:(-1) x in
      let a =
        Nx.cast Nx.float64 (Nx.add (Nx.tril x) (Nx.matrix_transpose lower))
      in
      let g = gamma32 (16 * n) and fa = frobenius a in
      let w, v = !last in
      let w = Nx.cast Nx.float64 (to_host ~by:"Nx.eigh" w) in
      let fail what got bound =
        if not (got <= bound) then
          failwith (strf "%s: %s is %g, past %g" name what got bound)
      in
      fail "|Σ w − tr a|"
        (Float.abs (Nx.item [] (Nx.sum w) -. Nx.item [] (Nx.trace a)))
        (float n *. g *. fa);
      Option.iter
        (fun v ->
          let v = Nx.cast Nx.float64 (to_host ~by:"Nx.eigh" v) in
          let vt = Nx.matrix_transpose v in
          let back = Nx.matmul (Nx.mul v (Nx.reshape [| 1; n |] w)) vt in
          fail "‖a − v diag w vᵀ‖_F" (frobenius (Nx.sub a back)) (g *. fa);
          fail "‖vᵀv − I‖_F"
            (frobenius (Nx.sub (Nx.matmul vt v) (Nx.eye Nx.float64 n)))
            g)
        v
    end;
    { run; wait }
  in
  (* The operand's bytes and the results'. *)
  let bytes = 4 * ((n * n) + n + if values then 0 else n * n) in
  let spec =
    strf "%s float32 %s"
      (if values then "eigvalsh" else "eigh")
      (dims [| n; n |])
  in
  { name; bytes; slow; make; spec }

(* svd's values through Nx.svdvals of an [m] by [n] float32 matrix of fresh
   values, into fresh values each call, as a user calls it. The check, by
   nx_metal.mli's bound with g = γ(16 max(m, n) + 100): each value within g ‖a‖₂
   of a's, so the squares' sum within k (2 g + g²) ‖a‖_F² of ‖a‖_F². The bound's
   h = ν(·) term is dropped: below 2^-100 of it for fresh values near 1. *)
let svdvals_row ~slow m n =
  let name = strf "svdvals/float32-%s" (dims [| m; n |]) in
  let k = Int.min m n in
  let make ~check g =
    let module G = (val Nx.devices ~kernels:(module Nx_metal) [ g ]) in
    let xv = Nx.Repr.of_array G.v (fresh_matrix g m n) in
    let last = ref (Nx.svdvals Nx.float32 xv) in
    let run () = last := Nx.svdvals Nx.float32 xv in
    let wait () = Nx.sync !last in
    if check then begin
      let fa = frobenius (to_host ~by:"Nx.svdvals" xv) in
      let s = Nx.cast Nx.float64 (to_host ~by:"Nx.svdvals" !last) in
      let g = gamma32 ((16 * Int.max m n) + 100) in
      let got = Float.abs (Nx.item [] (Nx.sum (Nx.square s)) -. (fa *. fa)) in
      let bound = float k *. ((2. *. g) +. (g *. g)) *. fa *. fa in
      if not (got <= bound) then
        failwith (strf "%s: |Σ s² − ‖a‖_F²| is %g, past %g" name got bound)
    end;
    { run; wait }
  in
  let spec = strf "svdvals float32 %s" (dims [| m; n |]) in
  { name; bytes = 4 * ((m * n) + k); slow; make; spec }

(* eigh of order 2048 at most: a call of order 4096 takes 2-3 s, which a case's
   20 samples cannot fit in its 10 s. svdvals of 1024² takes past 0.5 s for the
   same reason, so the square row is 768². 2^20 x 8 sums its columns in chunks.
   Its words vary by up to about 90 a call: rig's cache takes back a call's
   result buffer only once the collector has collected it, which so few words a
   call do not bring about between every two calls, and a call that comes first
   allocates from the driver. *)
let spectrals =
  [
    eigh_row ~values:true ~slow:4100 256;
    eigh_row ~values:false ~slow:3700 256;
    eigh_row ~values:true ~slow:6000 1024;
    eigh_row ~values:false ~slow:4200 1024;
    eigh_row ~values:true ~slow:7900 2048;
    eigh_row ~values:false ~slow:4000 2048;
    svdvals_row ~slow:6600 20_000 256;
    svdvals_row ~slow:26000 768 768;
    svdvals_row ~slow:190 (1 lsl 20) 8;
  ]

(* Attention *)

(* Calls of attention as Nx.attention makes them: a query [[b; hq; tq; d]] over
   keys and values [[b; hk; tk; d]], or in pools of pages, scale 1/√d, o in the
   query's dtype and lse in float32, the pair program Nx makes for the mask. *)

(* Nx.attention's pair program over a float32 score, of no load: a pair's length
   is the constant [tk], output 1 the score, output 0 the mask's and output 2
   [true], every visible pair kept. A mask is ["full"] or ["lengths"], of none;
   ["causal"], Nx.causal's [k <= q + (len - tq)]; or ["window128"], the window
   of 128 keys of Nx.attention's docs, [causal p] and [q + (len - tq) < k +
   128], whose position Nx records once more. *)
let pair_program ~tq ~tk mask =
  let int64 x = P.Const (D.Any D.Int64, P.bits D.Int64 (Int64.of_int x)) in
  let pair = [ P.In 0; int64 tk; Coord 2; Coord 3 ] in
  (* The query's position at node [at + 2], from the length (1) and the row
     (2). *)
  let position at =
    [ int64 tq; P.Op2 (Binary Sub, 1, at); Op2 (Binary Add, 2, at + 1) ]
  in
  let causal = position 4 @ [ P.Op2 (Compare Less_equal, 3, 6) ] in
  let mask =
    match mask with
    | "causal" -> causal
    | "window128" ->
        causal @ position 8
        @ [
            int64 128;
            Op2 (Binary Add, 3, 11);
            Op2 (Compare Less, 10, 12);
            Op2 (Binary And, 7, 13);
          ]
    | _ -> [ Const (D.Any D.Bool, P.bits D.Bool true) ]
  in
  let nodes = pair @ mask @ [ Const (D.Any D.Bool, P.bits D.Bool true) ] in
  let n = List.length nodes in
  P.v
    ~ins:(Iarray.of_list [ D.Any D.Float32 ])
    (Iarray.of_list nodes)
    ~outs:(Iarray.of_list [ n - 2; 0; n - 1 ])

(* Each batch element's keys under the mask ["lengths"]. *)
let key_lengths ~b ~tk = Array.init b (fun i -> (i + 1) * tk / b)

let int64s_on g shape xs =
  A.to_device g
    (A.of_array D.Int64 (Iarray.of_array shape) (Array.map Int64.of_int xs))

(* The keys nx.metal reads in one part at most, where its result has nx.cpu's
   bits. *)
let one_part = 448

(* The bounds on the difference of nx.metal's o and lse from nx.cpu's: twice the
   bound Nx.attention states for each against the exact result. The operands are
   of [dt], drawn within [-1, 1], so [M ≤ 1], [a ≤ √dk] and [|lse| ≤ a + ln tk].
   [ε] and [o]'s sum take a merge after each block of 64 keys, nx.metal's most,
   which bounds nx.cpu's none too. Neither library rounds a weight to the
   values' dtype. *)
let attention_bounds dt ~tk ~dk =
  let u = 0x1p-24 in
  let u' = u /. (1. -. u) in
  let uq =
    match dname dt with "float16" -> 0x1p-11 | "bfloat16" -> 0x1p-8 | _ -> u
  in
  let gamma n = float n *. 2. *. u /. (1. -. (float n *. 2. *. u)) in
  let a = Float.sqrt (float dk) in
  let eta = ((((1. +. u) ** 2.) *. (1. +. gamma (dk + 1))) -. 1.) *. a in
  let blocks = (tk + 63) / 64 in
  let merges = blocks - 1 in
  let ln_t = Float.log (float tk) in
  let e = Float.exp (3. *. u) in
  let eps =
    (e *. (float ((13 * (blocks + (2 * merges))) + 9) +. ln_t) *. u')
    +. ((e -. 1.) *. ln_t)
    +. (u' *. (a +. ln_t))
  in
  let w =
    ((1. +. u) *. ((2. *. eta) +. eps))
    +. (((2. *. a) +. ln_t) *. u)
    +. (5. *. u')
  in
  let ew = Float.exp w in
  let o = uq +. ((1. +. uq) *. (ew -. 1. +. (gamma (tk + 1 + merges) *. ew))) in
  (2. *. o, 2. *. (eta +. eps))

(* Fails unless nx.metal's [o] and [lse] have nx.cpu's bits, up to [one_part]
   keys, or else lie within [attention_bounds] of them. *)
let check_attention name p ~scale ~o ~lse ?lengths q kv ~tk ~dk =
  let host x = A.to_device Rig.host x in
  let kv_h =
    match kv with
    | K.Dense { k; v } -> K.Dense { k = host k; v = host v }
    | Paged { k; v; table } ->
        Paged { k = host k; v = host v; table = host table }
  in
  let shape x = A.Layout.shape (A.layout x) in
  let o_h = A.create Rig.host (A.dtype o) (shape o) in
  let lse_h = A.create Rig.host D.Float32 (shape lse) in
  done_ (name ^ " on nx.cpu")
    (Nx_cpu.attention p ~scale ~o:o_h ~lse:lse_h
       ?lengths:(Option.map host lengths) (host q) kv_h (Iarray.of_list []));
  if tk <= one_part then
    List.iter
      (fun (what, x, y) ->
        let bx = bytes_of x and by = bytes_of y in
        if bx <> by then
          failwith
            (strf "%s: %s differs from nx.cpu's at element %d" name what
               (first_difference x bx by)))
      [ ("o", A.Any o, A.Any o_h); ("lse", A.Any lse, A.Any lse_h) ]
  else begin
    let o_tol, lse_tol = attention_bounds (D.Any (A.dtype q)) ~tk ~dk in
    let close what tol got want =
      Array.iteri
        (fun i y ->
          let x = got.(i) in
          if not (x = y || Float.abs (x -. y) <= tol) then
            failwith
              (strf "%s: %s %d is %h, nx.cpu's %h, past %h apart" name what i x
                 y tol))
        want
    in
    close "o" o_tol
      (A.to_array (host_f32 (A.Any o)))
      (A.to_array (host_f32 (A.Any o_h)));
    close "lse" lse_tol (A.to_array (host lse)) (A.to_array lse_h)
  end

(* A row's call over [d] operands of the extents [(b, hq, hk, tq, tk, d)] under
   [mask], with [~page] keys and values in pools of pages of [page] slots, the
   table's entries spread through the pool: entry [i] of its [n] pages is page
   [37 i mod n], every page once where 37 does not divide [n]. *)
let attend (type v s) ~check name g (d : (v, s) D.t) ~mask ?page
    (b, hq, hk, tq, tk, dh) =
  let on ?seed shape = A.expect d (fresh g ?seed (D.Any d) shape) in
  let q = on ~seed:1 [| b; hq; tq; dh |] in
  let kv =
    match page with
    | None ->
        let s = [| b; hk; tk; dh |] in
        K.Dense { k = on ~seed:2 s; v = on ~seed:3 s }
    | Some page ->
        let pages = b * tk / page in
        let pool = [| pages; hk; page; dh |] in
        let table =
          int64s_on g
            [| b; tk / page |]
            (Array.init pages (fun i -> 37 * i mod pages))
        in
        K.Paged { k = on ~seed:2 pool; v = on ~seed:3 pool; table }
  in
  let o = on [| b; hq; tq; dh |] in
  let lse = A.create g D.Float32 (Iarray.of_list [ b; hq; tq ]) in
  let lengths =
    if mask = "lengths" then Some (int64s_on g [| b |] (key_lengths ~b ~tk))
    else None
  in
  let p = pair_program ~tq ~tk mask in
  let scale = 1. /. Float.sqrt (float dh) in
  let run () =
    done_ name
      (Nx_metal.attention p ~scale ~o ~lse ?lengths q kv (Iarray.of_list []))
  in
  let wait = wait_all [ A.Any o; A.Any lse ] in
  if check then begin
    run ();
    wait ();
    check_attention name p ~scale ~o ~lse ?lengths q kv ~tk ~dk:dh
  end;
  { run; wait }

(* The row of [dt] queries under [mask] at [(b, hq, hk, tq, tk, d)]. Its bytes
   are the query's, the keys', the values', o's and lse's, and the table's or
   the lengths'. A paged row's spec is its dense row's: no reference reads paged
   keys. *)
let attention_row ?page ~slow dt mask ((b, hq, hk, tq, tk, dh) as s) =
  let name =
    strf "attention/%s-%s-b%d-h%dk%d-%dx%d-d%d%s" (dname dt) mask b hq hk tq tk
      dh
      (match page with Some p -> strf "-page%d" p | None -> "")
  in
  let (D.Any d) = dt in
  let make ~check g = attend ~check name g d ~mask ?page s in
  let w = D.bits d / 8 in
  let index =
    match page with
    | Some p -> 8 * b * (tk / p)
    | None -> if mask = "lengths" then 8 * b else 0
  in
  let bytes =
    (w * 2 * ((b * hq * tq * dh) + (b * hk * tk * dh)))
    + (4 * b * hq * tq)
    + index
  in
  let spec =
    strf "attention %s %d %d %d %d %d %d %s" (dname dt) b hq hk tq tk dh mask
  in
  { name; bytes; slow; make; spec }

(* gpt-oss-20b's decode over 512, 1024 and its window's 128 keys, the 512 in
   pools of pages of 1 and of 16 slots, and its prefill of 512, causal and in
   its window; whisper turbo's encoder at float32 and float16, its
   cross-attention, and its decoder's self-attention at its 448 keys; prefills
   of 2048 at 32 heads of 128, mid shapes, decode steps over 8192 keys, of 32
   query heads in groups of 4, at every dtype, one under lengths; and prefills
   at d = 80 and d = 110. Each [slow] is estimated, not measured: twice MLX's
   time on an M1 Max (mlx 0.32.4, lazy, chained) over the bytes' time at 400
   GB/s. *)
let attentions =
  let open D in
  let f16 = Any Float16 and bf16 = Any Bfloat16 and f32 = Any Float32 in
  let oss t = (1, 64, 8, 1, t, 64) and oss_prefill = (1, 64, 8, 512, 512, 64) in
  let enc = (1, 20, 20, 1500, 1500, 64) and cross = (1, 20, 20, 1, 1500, 64) in
  let prefill = (1, 32, 32, 2048, 2048, 128) in
  let mid = (4, 16, 16, 512, 512, 64) in
  let decode = (8, 32, 8, 1, 8192, 128) in
  [
    attention_row ~slow:18 bf16 "full" (oss 512);
    attention_row ~page:1 ~slow:18 bf16 "full" (oss 512);
    attention_row ~page:16 ~slow:18 bf16 "full" (oss 512);
    attention_row ~slow:14 bf16 "full" (oss 1024);
    attention_row ~slow:39 bf16 "full" (oss 128);
    attention_row ~slow:38 bf16 "causal" oss_prefill;
    attention_row ~slow:67 bf16 "window128" oss_prefill;
    attention_row ~slow:59 f32 "full" enc;
    attention_row ~slow:83 f16 "full" enc;
    attention_row ~slow:2 f32 "full" cross;
    attention_row ~slow:3 f16 "full" cross;
    attention_row ~slow:3 f32 "full" (1, 20, 20, 1, 448, 64);
    attention_row ~slow:62 f16 "causal" prefill;
    attention_row ~slow:81 bf16 "causal" prefill;
    attention_row ~slow:49 f32 "causal" prefill;
    attention_row ~slow:94 f32 "full" prefill;
    attention_row ~slow:19 f16 "causal" mid;
    attention_row ~slow:19 f32 "full" mid;
    attention_row ~slow:3 f16 "full" decode;
    attention_row ~slow:3 bf16 "full" decode;
    attention_row ~slow:3 f32 "full" decode;
    attention_row ~slow:3 f16 "lengths" decode;
    attention_row ~slow:36 f16 "causal" (1, 32, 32, 1024, 1024, 80);
    attention_row ~slow:32 bf16 "full" (4, 4, 4, 256, 256, 110);
  ]

let all =
  loops @ contractions @ groupeds @ decodes @ quants @ transforms
  @ factorisations @ sorts @ moves @ spectrals @ attentions
