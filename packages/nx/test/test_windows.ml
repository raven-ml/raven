(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The window law, through Nx: each device of a set computes its window of a
   result from the windows of the operands it holds, so an operation over
   values cut along axes its result keeps gives the result it gives on one
   device, bitwise but for which NaN a NaN result is, which no operation on
   numbers specifies. *)

open Windtrap
open Nx_support
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Prog = Nx_kernel.Prog

let m = Nx_support.memory
let numel s = Array.fold_left ( * ) 1 s
let axes rank = List.init rank Fun.id

module Set1 = (val Nx.devices [ m 0 ])
module Set2 = (val Nx.devices [ m 0; m 1 ])
module Set3 = (val Nx.devices [ m 0; m 1; m 2 ])
module Set4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh4 = Nx.Mesh.v Set4.v [ ("a", 2); ("b", 2) ]

(* Cases *)

(* Where an operation's operands lie on a set ['d]: [place x] at the
   placement the case draws, keeping its cuts of the axes [keep] holds;
   [at] that placement. *)
type 'd placer = {
  place :
    'v 's. ?keep:(int -> bool) -> ('v, 's, Nx.host) Nx.t -> ('v, 's, 'd) Nx.t;
  at : 'd Nx.Placement.t;
}

type run = { run : 'd. 'd placer -> 'd Nx.packed list }

(* An operation of a family over host operands, and the axes of its first
   result a placement may cut: an operand that holds such an axis holds it
   at the same position. *)
type case = {
  family : string;
  show : Format.formatter -> unit;
  free : int list;
  run : run;
}

(* Float32 terms whose sums depend on their association, signed zeros and
   NaNs of both signs. *)
let terms = [ 1e7; -1e7; 1.; 0.1; 3.3; -2.5e-3; 0.; -0.; nan; -.nan ]

(* [terms] but NaN: a routine's iterations converge on them. *)
let finite = List.filter (fun x -> not (Float.is_nan x)) terms

let floats pool shape =
  Gen.map (Nx.create Nx.float32 shape)
    (Gen.array ~size:(Gen.constant (numel shape)) (Gen.of_list pool))

let f32 = floats terms
let matrix = floats finite

let i32 shape =
  Gen.map (Nx.create Nx.int32 shape)
    (Gen.array
       ~size:(Gen.constant (numel shape))
       (Gen.of_list [ 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int ]))

let c64 shape =
  let complex re = { Complex.re; im = -.re } in
  Gen.map
    (fun re -> Nx.create Nx.complex64 shape (Array.map complex re))
    (Gen.array ~size:(Gen.constant (numel shape)) (Gen.of_list terms))

(* Positions from [lo] to [hi] inclusive. *)
let positions lo hi shape =
  Gen.map
    (fun p -> Nx.create Nx.int64 shape (Array.map Int64.of_int p))
    (Gen.array ~size:(Gen.constant (numel shape)) (Gen.int_range lo hi))

let shape_of rank = Gen.array ~size:(Gen.constant rank) (Gen.int_range 0 6)
let axis_of rank = Gen.int_range 0 (rank - 1)
let without a rank = List.filter (( <> ) a) (axes rank)
let pp_x = Nx.pp

(* Elementwise and creation *)

let map_case =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* x = f32 s in
  let+ y = f32 s in
  {
    family = "Map";
    show = (fun ppf -> Format.fprintf ppf "add %a %a" pp_x x pp_x y);
    free = axes rank;
    run = { run = (fun pl -> [ Nx.P (Nx.add (pl.place x) (pl.place y)) ]) };
  }

(* An identity offset by [k]: its elements show each window's offset. *)
let creation_case =
  let open Gen in
  let* n = int_range 0 6 in
  let* m = int_range 0 6 in
  let+ k = int_range (-3) 3 in
  {
    family = "creation";
    show = (fun ppf -> Format.fprintf ppf "eye ~m:%d ~k:%d %d" m k n);
    free = [ 0; 1 ];
    run = { run = (fun pl -> [ Nx.P (Nx.eye_at ~m ~k pl.at Nx.float32 n) ]) };
  }

let copy_case =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let+ x = f32 s in
  {
    family = "Copy";
    show = (fun ppf -> Format.fprintf ppf "copy %a" pp_x x);
    free = axes rank;
    run = { run = (fun pl -> [ Nx.P (Nx.copy (pl.place x)) ]) };
  }

(* Reductions and scans *)

(* Axes of a loop of [rank] to reduce, at least one, none of them [kept]. *)
let reduced_of kept rank =
  let open Gen in
  let+ mask = array ~size:(constant rank) bool in
  let others = without kept rank in
  match List.filter (fun a -> mask.(a)) others with
  | [] -> [ List.hd others ]
  | r -> r

(* A sum, and where every reduced axis has an element, a maximum and the
   first maximum's position along the first. *)
let reduce_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* kept = axis_of rank in
  let* reduced = reduced_of kept rank in
  let+ x = f32 s in
  let some = List.for_all (fun a -> s.(a) > 0) reduced in
  let run pl =
    let x = pl.place x in
    let sum = Nx.P (Nx.sum ~axes:reduced x) in
    if not some then [ sum ]
    else
      [
        sum;
        Nx.P (Nx.max ~axes:reduced x);
        Nx.P (Nx.argmax ~axis:(List.hd reduced) x);
      ]
  in
  {
    family = "Reduce";
    show =
      (fun ppf ->
        Format.fprintf ppf "reduce along %a %a" pp_ints (Array.of_list reduced)
          pp_x x);
    free = List.filter (fun a -> not (List.mem a reduced)) (axes rank);
    run = { run };
  }

let scan_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* a = axis_of rank in
  let+ x = f32 s in
  let run pl =
    let x = pl.place x in
    [ Nx.P (Nx.cumsum ~axis:a x); Nx.P (Nx.cummax ~axis:a x) ]
  in
  {
    family = "Scan";
    show = (fun ppf -> Format.fprintf ppf "scan along %d %a" a pp_x x);
    free = without a rank;
    run = { run };
  }

(* Padded loops

   No function reads one operand through a pad: a compiler fuses a pad into
   the loop that reads it. Prim builds such loops as a compiler does. *)

(* A pad of an operand of shape [s] along one axis, and that axis: crops,
   pads, an interior and possibly a window of 2, with [hi] grown where the
   padded extent would hold no window or be negative. *)
let pad_of s =
  let open Gen in
  let rank = Array.length s in
  let* a = axis_of rank in
  let* lo = int_range (-1) 2 in
  let* hi = int_range (-1) 2 in
  let* interior = int_range 0 1 in
  let+ windowed = bool in
  let d = s.(a) in
  let extent = lo + hi + d + (interior * max 0 (d - 1)) in
  let hi = hi + max 0 ((if windowed then 2 else 0) - extent) in
  let at v = Iarray.init rank (fun i -> if i = a then v else 0) in
  let windows : Nx_array.Move.window iarray =
    if windowed then [| { axis = a; size = 2; step = 1; dilation = 1 } |]
    else [||]
  in
  ({ Nx_kernel.lo = at lo; hi = at hi; interior = at interior; windows }, a)

let pp_pad ppf (p : Nx_kernel.pad) =
  let ints a = Iarray.to_array a in
  Format.fprintf ppf "pad lo %a hi %a interior %a%s" pp_ints (ints p.lo)
    pp_ints (ints p.hi) pp_ints (ints p.interior)
    (if Iarray.length p.windows > 0 then " windowed" else "")

(* An operand of rank 2 or 3, its pad along one axis, that axis, and its
   loop's shape. *)
let padded_operand =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* pad, a = pad_of s in
  let+ x = f32 s in
  match Nx_kernel.Shape.padded pad s with
  | Ok loop -> (x, pad, a, loop)
  | Error e -> failwith ("pad_of drew a pad its operand refuses: " ^ e)

let one = Prog.v ~ins:[| D.Any D.Float32 |] [| In 0 |] ~outs:[| 0 |]

(* [x0 · Coord 0 + x1] and [x0 + x1]: coordinates make each window's offset
   show. *)
let coords =
  Prog.v
    ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
    [|
      In 0;
      In 1;
      Coord 0;
      Op1 (Cast, D.Any D.Float32, 2);
      Op2 (Binary Mul, 0, 3);
      Op2 (Binary Add, 4, 1);
      Op2 (Binary Add, 0, 1);
    |]
    ~outs:[| 5; 6 |]

let layout loop = L.contiguous (Iarray.of_array loop)

let padded_load pl x pad : _ Nx.Prim.load =
  Nx.Prim.Padded { x = pl.place x; fill = 0.5; pad }

let padded_map_case =
  let open Gen in
  let* x, pad, a, loop = padded_operand in
  let+ y = f32 loop in
  let run pl =
    let a, (b, ()) =
      Nx.Prim.eval ~by:"t"
        (Nx.Prim.Map
           {
             layout = layout loop;
             prog = coords;
             outs = Nx.Prim.[ Nx.float32; Nx.float32 ];
             loads = [| padded_load pl x pad; Nx.Prim.Plain (pl.place y) |];
             at = None;
           })
    in
    [ Nx.P a; Nx.P b ]
  in
  {
    family = "padded Map";
    show =
      (fun ppf ->
        Format.fprintf ppf "map of %a by %a and %a" pp_x x pp_pad pad pp_x y);
    free = without a (Array.length (Nx.shape x));
    run = { run };
  }

(* A sum, and where every reduced axis has an element the first maximum and
   its position, along axes of the loop beside a kept one. *)
let padded_reduce_case =
  let open Gen in
  let* x, pad, a, loop = padded_operand in
  let rank = Array.length (Nx.shape x) in
  let* kept = of_list (without a rank) in
  let+ reduced = reduced_of kept (Array.length loop) in
  let some = List.for_all (fun b -> loop.(b) > 0) reduced in
  let run pl =
    let layout = layout loop and axes = Iarray.of_list reduced in
    let loads : _ iarray = [| padded_load pl x pad |] in
    if some then
      let sum, ((max, at), ()) =
        Nx.Prim.eval ~by:"t"
          (Nx.Prim.Reduce
             {
               layout;
               axes;
               prog = one;
               reductions =
                 Nx.Prim.
                   [ Monoid (Sum, 0, Nx.float32); Arg (Max, 0, Nx.float32) ];
               loads;
             })
      in
      [ Nx.P sum; Nx.P max; Nx.P at ]
    else
      let sum, () =
        Nx.Prim.eval ~by:"t"
          (Nx.Prim.Reduce
             {
               layout;
               axes;
               prog = one;
               reductions = Nx.Prim.[ Monoid (Sum, 0, Nx.float32) ];
               loads;
             })
      in
      [ Nx.P sum ]
  in
  {
    family = "padded Reduce";
    show =
      (fun ppf ->
        Format.fprintf ppf "reduce along %a of %a by %a" pp_ints
          (Array.of_list reduced) pp_x x pp_pad pad);
    free =
      List.filter (fun b -> b <> a && not (List.mem b reduced)) (axes rank);
    run = { run };
  }

(* A running sum, or a running first maximum and its position. *)
let padded_scan_case =
  let open Gen in
  let* x, pad, a, loop = padded_operand in
  let rank = Array.length (Nx.shape x) in
  let* kept = of_list (without a rank) in
  let* axis = of_list (without kept (Array.length loop)) in
  let+ arg = bool in
  let run pl =
    let layout = layout loop in
    let loads : _ iarray = [| padded_load pl x pad |] in
    if arg then
      let max, at =
        Nx.Prim.eval ~by:"t"
          (Nx.Prim.Scan
             {
               layout;
               axis;
               prog = one;
               reduction = Nx.Prim.Arg (Max, 0, Nx.float32);
               loads;
               carry = None;
             })
      in
      [ Nx.P max; Nx.P at ]
    else
      [
        Nx.P
          (Nx.Prim.eval ~by:"t"
             (Nx.Prim.Scan
                {
                  layout;
                  axis;
                  prog = one;
                  reduction = Nx.Prim.Monoid (Sum, 0, Nx.float32);
                  loads;
                  carry = None;
                }));
      ]
  in
  {
    family = "padded Scan";
    show =
      (fun ppf ->
        Format.fprintf ppf "%s along %d of %a by %a"
          (if arg then "running argmax" else "running sum")
          axis pp_x x pp_pad pad);
    free = List.filter (fun b -> b <> a && b <> axis) (axes rank);
    run = { run };
  }

(* Gathers, scatters and sorts *)

(* Positions from one before the axis to one past it, which read zero. *)
let gather_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* d = int_range 1 6 in
  let* idx = positions (-1) d s in
  let+ x = f32 (Array.mapi (fun a e -> if a = axis then d else e) s) in
  let run pl =
    [ Nx.P (Nx.take_along_axis ~axis (pl.place idx) (pl.place x)) ]
  in
  {
    family = "Gather";
    show =
      (fun ppf ->
        Format.fprintf ppf "take_along_axis ~axis:%d %a %a" axis pp_x idx pp_x
          x);
    free = without axis rank;
    run = { run };
  }

let scatter_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* n = int_range 0 6 in
  let su = Array.mapi (fun a e -> if a = axis then n else e) s in
  let* idx = positions 0 (max 0 (s.(axis) - 1)) su in
  let* u = f32 su in
  let+ into = f32 s in
  let run pl =
    let idx = pl.place idx and u = pl.place u and into = pl.place into in
    [ Nx.P (Nx.scatter ~combine:Add ~axis idx u into) ]
  in
  {
    family = "Scatter";
    show =
      (fun ppf ->
        Format.fprintf ppf "scatter ~combine:Add ~axis:%d %a %a %a" axis pp_x
          idx pp_x u pp_x into);
    free = without axis rank;
    run = { run };
  }

(* A sort, or its first [k] along the axis. *)
let sort_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* descending = bool in
  let* first = bool in
  let* k = int_range 0 s.(axis) in
  let+ x = f32 s in
  let show ppf =
    if first then Format.fprintf ppf "top_k ~k:%d ~axis:%d %a" k axis pp_x x
    else
      Format.fprintf ppf "sort ~descending:%b ~axis:%d %a" descending axis pp_x
        x
  in
  let run pl =
    let v, i =
      if first then Nx.top_k ~k ~axis (pl.place x)
      else Nx.sort ~descending ~axis (pl.place x)
    in
    [ Nx.P v; Nx.P i ]
  in
  {
    family = (if first then "Sort k" else "Sort");
    show;
    free = without axis rank;
    run = { run };
  }

(* Patches summed back over one spatial axis, of one or two lead axes. *)
let fold_case =
  let open Gen in
  let* lead = int_range 1 2 in
  let* ls = array ~size:(constant lead) (int_range 0 4) in
  let* o = int_range 0 6 in
  let* k = int_range 1 3 in
  let* stride = int_range 1 2 in
  let* dilation = int_range 1 2 in
  let* lo = int_range 0 2 in
  let* hi = int_range 0 2 in
  let span = (dilation * (k - 1)) + 1 and padded = o + lo + hi in
  let windows = if span > padded then 0 else ((padded - span) / stride) + 1 in
  let+ p = f32 (Array.append ls [| k; windows |]) in
  let run pl =
    [
      Nx.P
        (Nx.combine_patches ~output_size:[| o |] ~kernel_size:[| k |]
           ~stride:[| stride |] ~dilation:[| dilation |]
           ~padding:[| (lo, hi) |] (pl.place p));
    ]
  in
  {
    family = "Fold";
    show =
      (fun ppf ->
        Format.fprintf ppf
          "combine_patches ~output_size:%d ~kernel_size:%d ~stride:%d \
           ~dilation:%d ~padding:(%d, %d) %a"
          o k stride dilation lo hi pp_x p);
    free = axes lead;
    run = { run };
  }

(* Transforms *)

let c2c_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* inverse = bool in
  let+ x = c64 s in
  let run pl =
    let x = pl.place x in
    if inverse then [ Nx.P (Nx.ifft ~axes:[ axis ] x) ]
    else [ Nx.P (Nx.fft ~axes:[ axis ] x) ]
  in
  {
    family = "C2c";
    show =
      (fun ppf ->
        Format.fprintf ppf "%s ~axes:[%d] %a"
          (if inverse then "ifft" else "fft")
          axis pp_x x);
    free = without axis rank;
    run = { run };
  }

let r2c_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let+ x = f32 s in
  let run pl = [ Nx.P (Nx.rfft Nx.complex64 ~axes:[ axis ] (pl.place x)) ] in
  {
    family = "R2c";
    show = (fun ppf -> Format.fprintf ppf "rfft ~axes:[%d] %a" axis pp_x x);
    free = without axis rank;
    run = { run };
  }

(* A real inverse of [n] points from its [n/2 + 1] bins. *)
let c2r_case =
  let open Gen in
  let* rank = int_range 2 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* n = int_range 0 6 in
  let bins = Array.mapi (fun a d -> if a = axis then (n / 2) + 1 else d) s in
  let+ x = c64 bins in
  let run pl =
    [ Nx.P (Nx.irfft Nx.float32 ~axes:[ axis ] ~s:[ n ] (pl.place x)) ]
  in
  {
    family = "C2r";
    show =
      (fun ppf ->
        Format.fprintf ppf "irfft ~axes:[%d] ~s:[%d] %a" axis n pp_x x);
    free = without axis rank;
    run = { run };
  }

(* Routines, over a batch of matrices *)

let routine family show run = { family; show; free = [ 0 ]; run }

let lu_case =
  let open Gen in
  let* batch = int_range 0 4 in
  let* n = int_range 1 4 in
  let+ a = f32 [| batch; n; n |] in
  let run pl =
    let perm, l, u = Nx.lu (pl.place a) in
    [ Nx.P perm; Nx.P l; Nx.P u ]
  in
  routine "Lu" (fun ppf -> Format.fprintf ppf "lu %a" pp_x a) { run }

let solve_case =
  let open Gen in
  let* batch = int_range 0 3 in
  let* n = int_range 1 4 in
  let* r = int_range 0 3 in
  let* lower = bool in
  let* transpose = of_list [ None; Some Nx.Transpose; Some Nx.Adjoint ] in
  let* unit_diag = bool in
  let* a = matrix [| batch; n; n |] in
  let+ b = matrix [| batch; n; r |] in
  let triangle : Nx.triangle = if lower then Lower else Upper in
  let run pl =
    let a = pl.place a and b = pl.place b in
    [ Nx.P (Nx.solve_triangular ~triangle ?transpose ~unit_diag a b) ]
  in
  let show ppf =
    Format.fprintf ppf "solve_triangular lower %b%s unit_diag %b %a %a" lower
      (match transpose with
      | None -> ""
      | Some Transpose -> " transposed"
      | Some Adjoint -> " adjoint")
      unit_diag pp_x a pp_x b
  in
  routine "Solve_triangular" show { run }

let qr_case =
  let open Gen in
  let* batch = int_range 0 3 in
  let* rows = int_range 1 4 in
  let* cols = int_range 1 4 in
  let* complete = bool in
  let+ a = matrix [| batch; rows; cols |] in
  let factors : Nx.factors = if complete then Complete else Reduced in
  let run pl =
    let q, r = Nx.qr ~factors (pl.place a) in
    [ Nx.P q; Nx.P r ]
  in
  routine "Qr"
    (fun ppf -> Format.fprintf ppf "qr complete %b %a" complete pp_x a)
    { run }

let svd_case =
  let open Gen in
  let* batch = int_range 0 3 in
  let* rows = int_range 1 4 in
  let* cols = int_range 1 4 in
  let* vectors = bool in
  let+ a = matrix [| batch; rows; cols |] in
  let run pl =
    let u, s, vh = Nx.svd ~vectors Nx.float32 (pl.place a) in
    [ Nx.P u; Nx.P s; Nx.P vh ]
  in
  routine "Svd"
    (fun ppf -> Format.fprintf ppf "svd vectors %b %a" vectors pp_x a)
    { run }

let eigh_case =
  let open Gen in
  let* batch = int_range 0 3 in
  let* n = int_range 1 4 in
  let* vectors = bool in
  let+ a = matrix [| batch; n; n |] in
  let run pl =
    let w, v = Nx.eigh ~vectors Nx.float32 (pl.place a) in
    [ Nx.P w; Nx.P v ]
  in
  routine "Eigh"
    (fun ppf -> Format.fprintf ppf "eigh vectors %b %a" vectors pp_x a)
    { run }

(* Contractions *)

(* An int32 matrix product, batched or not, of rows and columns either side
   of where nx.cpu's order changes. A placement cuts [a]'s batch and rows and
   [b]'s batch and columns, which lie at their result axes' positions. *)
let contract_case =
  let open Gen in
  let* nb = int_range 0 1 in
  let extent = of_list [ 0; 1; 3; 5; 8; 9 ] in
  let* rows = extent in
  let* cols = extent in
  let* k = of_list [ 0; 1; 3; 17; 40 ] in
  let batch = if nb = 0 then [||] else [| 3 |] in
  let* a = i32 (Array.append batch [| rows; k |]) in
  let+ b = i32 (Array.append batch [| k; cols |]) in
  let run pl =
    let a = pl.place ~keep:(fun i -> i <= nb) a in
    let b = pl.place ~keep:(fun i -> i < nb || i = nb + 1) b in
    [ Nx.P (Nx.matmul a b) ]
  in
  {
    family = "Contract";
    show = (fun ppf -> Format.fprintf ppf "matmul %a %a" pp_x a pp_x b);
    free = axes (nb + 2);
    run = { run };
  }

let cases =
  [
    map_case;
    creation_case;
    padded_map_case;
    reduce_case;
    padded_reduce_case;
    scan_case;
    padded_scan_case;
    gather_case;
    scatter_case;
    sort_case;
    fold_case;
    c2c_case;
    r2c_case;
    c2r_case;
    lu_case;
    solve_case;
    qr_case;
    svd_case;
    eigh_case;
    copy_case;
    contract_case;
  ]

let families =
  [
    "Map"; "creation"; "padded Map"; "Reduce"; "padded Reduce"; "Scan";
    "padded Scan"; "Gather"; "Scatter"; "Sort"; "Sort k"; "Fold"; "C2c";
    "R2c"; "C2r"; "Lu"; "Solve_triangular"; "Qr"; "Svd"; "Eigh"; "Copy";
    "Contract";
  ]

(* Placements *)

(* Where operands lie: split over two to four devices; a 2 x 2 mesh cutting
   one axis over both its axes or two axes over one each; or one axis over
   its minor axis, the major holding copies. *)
type where = Split of int * int | Both of int | Two of int * int | Minor of int

let pp_where ppf = function
  | Split (n, a) -> Format.fprintf ppf "split %d over %d" a n
  | Both a -> Format.fprintf ppf "cut %d over 2x2" a
  | Two (a, b) -> Format.fprintf ppf "cuts %d/a %d/b" a b
  | Minor a -> Format.fprintf ppf "cut %d/b" a

let where_of free =
  let open Gen in
  let axis = of_list free in
  let split =
    let* n = int_range 2 4 in
    let+ a = axis in
    Split (n, a)
  in
  let two =
    let* a = axis in
    let+ b = of_list (List.filter (( <> ) a) free) in
    Two (a, b)
  in
  let one =
    [ split; map (fun a -> Both a) axis; map (fun a -> Minor a) axis ]
  in
  one_of (if List.length free < 2 then one else two :: one)

let pp_case ppf (c, w) = Format.fprintf ppf "%t at %a" c.show pp_where w

let case =
  Gen.with_pp pp_case
    (Gen.bind (Gen.one_of cases) (fun c ->
         Gen.map (fun w -> (c, w)) (where_of c.free)))

let placer (type d) (cut : (int -> bool) -> d Nx.Placement.t) : d placer =
  {
    place = (fun ?(keep = fun _ -> true) x -> Nx.place (cut keep) x);
    at = cut (fun _ -> true);
  }

(* [cuts]' entries whose axis [keep] holds, over [mesh4], or whole on each
   device without one. *)
let meshed keep cuts =
  match List.filter (fun (a, _) -> keep a) cuts with
  | [] -> Set4.on
  | cuts -> Nx.Placement.mesh mesh4 cuts

(* The law *)

let windows (type v s d) (x : (v, s, d) Nx.t) =
  Array.map
    (fun a -> Iarray.to_array (L.shape (Nx_array.layout a)))
    (Iarray.to_array (Nx.Repr.shards x))

let same (type d e) i (Nx.P x : d Nx.packed) (Nx.P y : e Nx.packed) =
  let msg = Printf.sprintf "result %d" i in
  equal ~msg:(msg ^ ", shape") (array int) (Nx.shape x) (Nx.shape y);
  match D.equal_witness (Nx.dtype x) (Nx.dtype y) with
  | None ->
      failf "%s: %s, not %s" msg (D.name (Nx.dtype y)) (D.name (Nx.dtype x))
  | Some Type.Equal ->
      equal ~msg
        (elements (Nx.dtype x))
        (Array.to_list (Nx.to_array x))
        (Array.to_list (Nx.to_array y))

(* Covers the windows the first of [results] lies in. *)
let cover_windows family (Nx.P first : _ Nx.packed) =
  let whole = Nx.shape first and ws = windows first in
  let cut = Array.exists (fun s -> s <> whole) ws in
  List.iter
    (fun f -> cover ("a cut window of " ^ f) (cut && family = f))
    families;
  cover "an empty window of a result with elements"
    (numel whole > 0 && Array.exists (fun s -> numel s = 0) ws);
  cover "windows of two extents, none empty"
    (Array.for_all (fun s -> numel s > 0) ws
    && Array.exists (fun s -> s <> ws.(0)) ws)

let law (c, w) =
  cover "a split over three devices"
    (match w with Split (3, _) -> true | _ -> false);
  cover "a cut over both mesh axes" (match w with Both _ -> true | _ -> false);
  cover "two axes cut" (match w with Two _ -> true | _ -> false);
  cover "a cut beside copies" (match w with Minor _ -> true | _ -> false);
  let expected = c.run.run (placer (fun _ -> Set1.on)) in
  let check : type d. d Nx.packed list -> unit =
   fun actual ->
    cover_windows c.family (List.hd actual);
    equal ~msg:"results" int (List.length expected) (List.length actual);
    List.iteri (fun i (e, a) -> same i e a) (List.combine expected actual)
  in
  let split (type d) (on : d Nx.Placement.t)
      (cut : axis:int -> d Nx.Placement.t) a =
    placer (fun keep -> if keep a then cut ~axis:a else on)
  in
  match w with
  | Split (2, a) -> check (c.run.run (split Set2.on Set2.split a))
  | Split (3, a) -> check (c.run.run (split Set3.on Set3.split a))
  | Split (_, a) -> check (c.run.run (split Set4.on Set4.split a))
  | Both a ->
      check (c.run.run (placer (fun keep -> meshed keep [ (a, [ "a"; "b" ]) ])))
  | Two (a, b) ->
      let cuts = [ (a, [ "a" ]); (b, [ "b" ]) ] in
      check (c.run.run (placer (fun keep -> meshed keep cuts)))
  | Minor a ->
      check (c.run.run (placer (fun keep -> meshed keep [ (a, [ "b" ]) ])))

(* NaNs of both signs, as a negated NaN gives, in a transform's lines and a
   contraction's terms. *)

let test_fft_nans () =
  let re =
    [|
      3.3; 3.3; 1.; -1e7; nan; 3.3; -1e7; 0.; 1.; -1e7; 1.; nan; -0.; nan; 0.;
      0.1; nan; 1.; 0.; nan; -2.5e-3; 1.; -2.5e-3; -0.; 1e7; nan; 1.; -0.;
      -1e7; 0.;
    |]
  in
  let x =
    Nx.create Nx.complex64 [| 5; 2; 3 |]
      (Array.map (fun re -> { Complex.re; im = -.re }) re)
  in
  let p = Nx.Placement.mesh mesh4 [ (1, [ "a" ]); (2, [ "b" ]) ] in
  same 0
    (Nx.P (Nx.fft ~axes:[ 0 ] (Nx.place Set1.on x)))
    (Nx.P (Nx.fft ~axes:[ 0 ] (Nx.place p x)))

let test_contract_nans () =
  let a =
    Nx.create Nx.float32 [| 8; 3 |]
      (Array.init 24 (fun i -> [| nan; -.nan; 1. |].(i mod 3)))
  in
  let b = Nx.create Nx.float32 [| 3; 8 |] (Array.make 24 1.) in
  let rows = Nx.Placement.mesh mesh4 [ (0, [ "a" ]) ] in
  let cols = Nx.Placement.mesh mesh4 [ (1, [ "b" ]) ] in
  same 0
    (Nx.P (Nx.matmul (Nx.place Set1.on a) (Nx.place Set1.on b)))
    (Nx.P (Nx.matmul (Nx.place rows a) (Nx.place cols b)))

let () =
  exit
    (run "nx windows"
       [
         group "laws"
           [
             prop ~count:1000
               "an operation over values cut along axes its result keeps \
                gives its result on one device"
               case law;
             test "a transform over cut NaNs of both signs" test_fft_nans;
             test "a contraction over cut NaNs of both signs"
               test_contract_nans;
           ];
       ])
