(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The window law, through the engine's private Prim, Route and Exec, copied
   here: on one device, an operation over the windows its rule gives its
   operands equals that window of its whole result, bitwise but for which NaN
   a NaN result is, which no operation on numbers specifies; a float
   contraction's window is within its bound instead. Each device of a set
   computes its window so, so this is the law a set's results rest on. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog

type b
type op = Op : 'r Value.prim -> op

let m = Nx_support.memory
let set : b Devices.t = Devices.mint ~by:"t" [ m 0 ]
let at = Devices.one set 0
let numel s = Array.fold_left ( * ) 1 s
let full d = { M.start = 0; count = d; step = 1 }

let value (type v s) (dt : (v, s) D.t) shape (data : v array) :
    (v, s, b) Value.t =
  Value.Array
    { at; a = A.to_device (m 0) (A.of_array dt shape data); dead = Value.Live }

(* Float32 terms whose sums depend on their association, signed zeros and a
   NaN. *)
let term =
  Gen.of_list ~pp:Format.pp_print_float
    [ 1e7; -1e7; 1.; 0.1; 3.3; -2.5e-3; 0.; -0.; Float.nan ]

let f32 shape =
  Gen.map
    (fun data -> value D.Float32 shape data)
    (Gen.array ~size:(Gen.constant (numel shape)) term)

(* Matrices of [term]s but NaN: a routine's iterations converge on them. *)
let matrix shape =
  let finite =
    Gen.of_list ~pp:Format.pp_print_float
      [ 1e7; -1e7; 1.; 0.1; 3.3; -2.5e-3; 0.; -0. ]
  in
  Gen.map
    (fun data -> value D.Float32 shape data)
    (Gen.array ~size:(Gen.constant (numel shape)) finite)

let i32 shape =
  let pp ppf x = Format.fprintf ppf "%ld" x in
  Gen.map
    (fun data -> value D.Int32 shape data)
    (Gen.array
       ~size:(Gen.constant (numel shape))
       (Gen.of_list ~pp [ 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int ]))

let c64 shape =
  Gen.map
    (fun re ->
      let data = Array.map (fun re -> { Complex.re; im = -.re }) re in
      value D.Complex64 shape data)
    (Gen.array ~size:(Gen.constant (numel shape)) term)

let positions shape bound =
  Gen.map
    (fun data -> value D.Int64 shape (Array.map Int64.of_int data))
    (Gen.array
       ~size:(Gen.constant (numel shape))
       (Gen.int_range 0 (max 0 (bound - 1))))

let shape_of rank = Gen.array ~size:(Gen.constant rank) (Gen.int_range 0 6)
let axis_of rank = Gen.int_range 0 (rank - 1)

(* Operations *)

let one = P.v ~ins:[| D.Any D.Float32 |] [| In 0 |] ~outs:[| 0 |]

(* [x0 · Coord 0 + x1] and [x0 + x1]: coordinates make each window's offset
   show. *)
let coords =
  P.v
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

let map_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* x = f32 s in
  let+ y = f32 s in
  Op
    (Value.Map
       {
         layout = L.contiguous s;
         prog = coords;
         outs = Value.[ D.Float32; D.Float32 ];
         loads = [| Plain x; Plain y |];
       })

(* A pad of an operand of shape [s] along one axis: crops, pads, an interior
   and possibly a window of 2, with [hi] grown where the padded extent would
   hold no window or be negative. *)
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
  {
    Nx_kernel.lo = at lo;
    hi = at hi;
    interior = at interior;
    windows =
      (if windowed then [| { M.axis = a; size = 2; step = 1; dilation = 1 } |]
       else [||]);
  }

(* The loop's shape a padded load by [pad] of [s] gives. *)
let loop_of pad s =
  match Nx_kernel.Shape.padded pad s with
  | Ok loop -> loop
  | Error e -> failwith ("pad_of drew a pad its operand refuses: " ^ e)

(* A loop's shape and its load of an operand of rank 1 to 3, padded or
   plain. *)
let operand ~padded =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  if not padded then
    let+ x = f32 s in
    (s, Value.Plain x)
  else
    let* pad = pad_of s in
    let+ x = f32 s in
    (loop_of pad s, Value.Padded { x; fill = 0.5; pad })

let padded_map_op =
  let open Gen in
  let* loop, load = operand ~padded:true in
  let+ y = f32 loop in
  Op
    (Value.Map
       {
         layout = L.contiguous loop;
         prog = coords;
         outs = Value.[ D.Float32; D.Float32 ];
         loads = [| load; Plain y |];
       })

(* The loop's coordinate along [axis], a program of no input. *)
let coord axis =
  P.v ~ins:[||] [| Coord axis; Op1 (Cast, D.Any D.Float32, 0) |] ~outs:[| 1 |]

(* A loop of rank 1 to 3 of no load, and its program: its coordinate along one
   axis. *)
let unloaded =
  let open Gen in
  let* rank = int_range 1 3 in
  let* loop = shape_of rank in
  let+ axis = axis_of rank in
  (loop, coord axis)

(* A sum and an argmax of [prog] along some axes of a loop of shape [loop]; the
   sum alone where an axis reduced has no element, of which Arg finds no
   extreme. *)
let reduce_of loop prog loads =
  let open Gen in
  let rank = Array.length loop in
  let+ keep = array ~size:(constant rank) bool in
  let axes =
    Array.of_list (List.filter (fun a -> not keep.(a)) (List.init rank Fun.id))
  in
  let axes = if axes = [||] then [| rank - 1 |] else axes in
  let layout = L.contiguous loop in
  if Array.for_all (fun a -> loop.(a) > 0) axes then
    Op
      (Value.Reduce
         {
           layout;
           axes = Iarray.of_array axes;
           prog;
           reductions =
             Value.[ Monoid (Sum, 0, D.Float32); Arg (Max, 0, D.Float32) ];
           loads;
         })
  else
    Op
      (Value.Reduce
         {
           layout;
           axes = Iarray.of_array axes;
           prog;
           reductions = Value.[ Monoid (Sum, 0, D.Float32) ];
           loads;
         })

let reduce_op ~padded =
  let open Gen in
  let* loop, load = operand ~padded in
  reduce_of loop one [| load |]

let unloaded_reduce_op =
  let open Gen in
  let* loop, prog = unloaded in
  reduce_of loop prog [||]

(* A running sum or argmax of [prog] over a loop of shape [loop]. *)
let scan_of loop prog loads =
  let open Gen in
  let* axis = axis_of (Array.length loop) in
  let+ arg = bool in
  let layout = L.contiguous loop in
  if arg then
    Op
      (Value.Scan
         {
           layout;
           axis;
           prog;
           reduction = Arg (Max, 0, D.Float32);
           loads;
           carry = None;
         })
  else
    Op
      (Value.Scan
         {
           layout;
           axis;
           prog;
           reduction = Monoid (Sum, 0, D.Float32);
           loads;
           carry = None;
         })

let scan_op ~padded =
  let open Gen in
  let* loop, load = operand ~padded in
  scan_of loop one [| load |]

let unloaded_scan_op =
  let open Gen in
  let* loop, prog = unloaded in
  scan_of loop prog [||]

let gather_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* d = int_range 1 6 in
  let sx = Array.mapi (fun a e -> if a = axis then d else e) s in
  let* idx = positions s d in
  let+ x = f32 sx in
  Op (Value.Gather { axis; idx; x })

let scatter_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* n = int_range 0 6 in
  let su = Array.mapi (fun a e -> if a = axis then n else e) s in
  let* idx = positions su s.(axis) in
  let* updates = f32 su in
  let+ into = f32 s in
  Op (Value.Scatter { combine = Add; unique = false; axis; idx; updates; into })

(* A sort, or its first [k] along the axis. *)
let sort_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* descending = bool in
  let* first = bool in
  let* k = int_range 0 s.(axis) in
  let+ x = f32 s in
  let k = if first then Some k else None in
  Op (Value.Sort { axis; descending; k; x })

let fold_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* pad = pad_of s in
  let+ x = f32 (loop_of pad s) in
  Op (Value.Fold { shape = Iarray.of_array s; pad; x })

let fft_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* inverse = bool in
  let+ x = c64 s in
  let direction : Nx_kernel.direction = if inverse then Inverse else Forward in
  Op (Value.Fft (C2c { direction; axes = [| axis |]; x }))

let r2c_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let+ x = f32 s in
  Op (Value.Fft (R2c { dtype = D.Complex64; axes = [| axis |]; x }))

(* A real inverse of [n] points from its [n/2 + 1] bins. *)
let c2r_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* n = int_range 0 6 in
  let s = Array.mapi (fun a d -> if a = axis then (n / 2) + 1 else d) s in
  let+ x = c64 s in
  Op (Value.Fft (C2r { dtype = D.Float32; n; axes = [| axis |]; x }))

let lu_op =
  let open Gen in
  let* batch = int_range 0 4 in
  let* n = int_range 1 4 in
  let+ a = f32 [| batch; n; n |] in
  Op (Value.Linalg (Lu a))

let solve_op =
  let open Gen in
  let* batch = int_range 0 3 in
  let* n = int_range 1 4 in
  let* r = int_range 0 3 in
  let* lower = bool in
  let* adjoint = bool in
  let* unit_diagonal = bool in
  let* a = matrix [| batch; n; n |] in
  let+ b = matrix [| batch; n; r |] in
  let triangle : Nx_kernel.triangle = if lower then Lower else Upper in
  Op
    (Value.Linalg
       (Solve_triangular { triangle; adjoint; unit_diagonal; a; b }))

(* A routine of two results of other shapes than its operand's. *)
let qr_op =
  let open Gen in
  let* batch = int_range 0 3 in
  let* rows = int_range 1 4 in
  let* cols = int_range 1 4 in
  let* complete = bool in
  let+ a = matrix [| batch; rows; cols |] in
  let factors : Nx_kernel.factors = if complete then Complete else Reduced in
  Op (Value.Linalg (Qr { factors; a }))

(* Singular values, with or without the reduced vectors beside them. *)
let svd_op =
  let open Gen in
  let* batch = int_range 0 3 in
  let* rows = int_range 1 4 in
  let* cols = int_range 1 4 in
  let* vectors = bool in
  let+ a = matrix [| batch; rows; cols |] in
  let dtype = D.Float32 in
  if vectors then
    Op (Value.Linalg (Svd { factors = Nx_kernel.Reduced; dtype; a }))
  else Op (Value.Linalg (Svd_values { dtype; a }))

(* Eigenvalues of the symmetric matrix each lower triangle names, with or
   without their vectors. *)
let eigh_op =
  let open Gen in
  let* batch = int_range 0 3 in
  let* n = int_range 1 4 in
  let* vectors = bool in
  let+ a = matrix [| batch; n; n |] in
  let dtype = D.Float32 in
  if vectors then Op (Value.Linalg (Eigh { dtype; a }))
  else Op (Value.Linalg (Eigh_values { dtype; a }))

let copy_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let+ x = f32 s in
  Op (Value.Copy x)

(* Axes of [extents], tagged, in some order: the shape and where each tag
   lies. *)
let permuted extents =
  Gen.map
    (fun order ->
      let shape = Array.of_list (List.map snd order) in
      let at tag = Option.get (List.find_index (fun (t, _) -> t = tag) order) in
      (shape, at))
    (Gen.permutation extents)

(* A contraction of [dt] operands that [operand] draws, of a batch, rows and
   columns of [extents] and one or two contracted pairs, the first of
   extents from 0 to 40, each operand's axes in any order. *)
let contract_of (type v s) (dt : (v, s) D.t) operand extents =
  let open Gen in
  let* nb = int_range 0 1 in
  let* nc = int_range 1 2 in
  let extent = of_list extents in
  let* rows = extent in
  let* cols = extent in
  let* k = of_list [ 0; 1; 3; 17; 40 ] in
  let* k2 = of_list [ 0; 1; 2; 5 ] in
  let* init = bool in
  let batch_axes = if nb = 0 then [] else [ (`Batch, 3) ] in
  let ks = if nc = 1 then [ (`K 0, k) ] else [ (`K 0, k); (`K 1, k2) ] in
  let* sa, at_a = permuted (batch_axes @ ((`Free, rows) :: ks)) in
  let* sb, at_b = permuted (batch_axes @ ks @ [ (`Free, cols) ]) in
  let pair tag = (at_a tag, at_b tag) in
  let batch = Array.of_list (List.map (fun (t, _) -> pair t) batch_axes) in
  let contracting = Array.of_list (List.map (fun (t, _) -> pair t) ks) in
  let* (a : (v, s, b) Value.t) = operand sa in
  let* b = operand sb in
  let result = Array.of_list (List.map snd batch_axes @ [ rows; cols ]) in
  let+ init =
    if init then map Option.some (operand result) else constant None
  in
  Op
    (Value.Contract
       {
         batch = Iarray.of_array batch;
         contracting = Iarray.of_array contracting;
         acc = D.Any dt;
         out = dt;
         a;
         b;
         init;
         a_scale = None;
         b_scale = None;
         segments = None;
       })

let contract_op = contract_of D.Int32 i32 [ 0; 1; 3; 5; 8; 9 ]

(* Rows and columns either side of where nx.cpu's order changes: 4 or
   fewer, or fewer than 64 outputs, in lane order; more in chain order. *)
let float_contract_op = contract_of D.Float32 f32 [ 0; 1; 3; 5; 8; 9; 16 ]

(* The operations' families. *)
let ops =
  [
    map_op;
    padded_map_op;
    reduce_op ~padded:false;
    reduce_op ~padded:true;
    unloaded_reduce_op;
    scan_op ~padded:false;
    scan_op ~padded:true;
    unloaded_scan_op;
    scatter_op;
    sort_op;
    fold_op;
    fft_op;
    r2c_op;
    c2r_op;
    contract_op;
    gather_op;
    lu_op;
    solve_op;
    qr_op;
    svd_op;
    eigh_op;
    copy_op;
  ]

(* Each case's family, as the covers name it. *)
let family (Op op) =
  let loop kind loads =
    let padded = function Value.Padded _ -> true | Value.Plain _ -> false in
    if Iarray.exists padded loads then "padded " ^ kind else kind
  in
  match op with
  | Value.Map { loads; _ } -> loop "Map" loads
  | Value.Reduce { loads; _ } -> loop "Reduce" loads
  | Value.Scan { loads; _ } -> loop "Scan" loads
  | Value.Sort { k = Some _; _ } -> "Sort k"
  | Value.Fft (C2c _) -> "C2c"
  | Value.Fft (R2c _) -> "R2c"
  | Value.Fft (C2r _) -> "C2r"
  | Value.Linalg l -> Prim.routine_name l
  | op -> Prim.name op

let families =
  [
    "Map"; "padded Map"; "Reduce"; "padded Reduce"; "Scan"; "padded Scan";
    "Gather"; "Scatter"; "Sort"; "Sort k"; "Fold"; "C2c"; "R2c"; "C2r"; "Lu";
    "Solve_triangular"; "Qr"; "Svd"; "Eigh"; "Copy"; "Contract";
  ]

(* Covers [what] of each family, demanded whatever the case's family. *)
let cover_families what o cond =
  let f = family o in
  List.iter (fun g -> cover (what ^ " " ^ g) (cond && f = g)) families

(* The law *)

let shape_of_results (Op op) =
  let first = ref [||] in
  ignore
    (Prim.results ~by:"t"
       (fun k (f : (_, _, _) Value.form) ->
         if k = 0 then first := L.shape f.layout;
         Value.Array
           {
             at = Devices.rebrand at;
             a = A.create Rig.host f.dtype [| 0 |];
             dead = Value.Live;
           })
       op);
  !first

(* Whether a scan runs along the results' axis [a] of [rule]: the law holds of
   no window that cuts it. *)
let scans (rule : Route.rule) a =
  match rule with
  | Elementwise -> false
  | Axes t | Into t -> Array.exists (Array.mem (Route.Scanned a)) t

(* Whether [rule]'s results keep their axis [a] whole: no operand's axis
   becomes it, scans along it or regroups into it. *)
let keeps (rule : Route.rule) a =
  match rule with
  | Elementwise -> false
  | Axes t | Into t ->
      let into : Route.axis -> bool = function
        | Becomes b | Scanned b | Regroups { axis = b; _ } -> b = a
        | Whole | Reduced _ -> false
      in
      not (Array.exists (Array.exists into) t)

(* A window of [o]'s results, whole along an axis its rule keeps or scans. *)
let window_of (Op op as o) =
  let open Gen in
  let rule = Prim.rule op and s = shape_of_results o in
  let+ w =
    let range a d =
      if keeps rule a || scans rule a || d = 0 then constant (full d)
      else
        let* start = int_range 0 (d - 1) in
        let+ count = int_range 0 (d - start) in
        { M.start; count; step = 1 }
    in
    let rec all a =
      if a = Array.length s then constant []
      else
        let* r = range a s.(a) in
        let+ rest = all (a + 1) in
        r :: rest
    in
    Gen.map Array.of_list (all 0)
  in
  (o, w)

let case = Gen.bind (Gen.one_of ops) window_of

(* A contraction's window of all its result but the first row, which, of
   enough rows and columns, nx.cpu sums in chain order; or any window. *)
let wide_window_of (Op op as o) =
  let nb =
    match op with
    | Value.Contract { batch; _ } -> Iarray.length batch
    | _ -> invalid_arg "wide_window_of: a contraction"
  in
  let row i d =
    if i = nb && d > 1 then { M.start = 1; count = d - 1; step = 1 }
    else full d
  in
  let wide = (o, Array.mapi row (shape_of_results o)) in
  Gen.frequency [ (2, window_of o); (1, Gen.constant wide) ]

let pp_case ppf (Op op, (w : M.range array)) =
  Format.fprintf ppf "%a at [%s]" Prim.pp op
    (String.concat "; "
       (Array.to_list
          (Array.map
             (fun (r : M.range) -> Printf.sprintf "%d+%d" r.start r.count)
             w)))

(* [a] with each NaN the quiet NaN of a clear sign bit. *)
let any_nan (type v s) (a : (v, s) A.t) : (v, s) A.t =
  let s = L.shape (A.layout a) in
  let f x = if Float.is_nan x then Float.nan else x in
  let z (c : Complex.t) = { Complex.re = f c.re; im = f c.im } in
  match A.dtype a with
  | D.Float32 -> A.of_array D.Float32 s (Array.map f (A.to_array a))
  | D.Float64 -> A.of_array D.Float64 s (Array.map f (A.to_array a))
  | D.Complex64 -> A.of_array D.Complex64 s (Array.map z (A.to_array a))
  | D.Complex128 -> A.of_array D.Complex128 s (Array.map z (A.to_array a))
  | _ -> a

(* [x] computed at [at] where it is a constant and [at] is given. *)
let reach (type v s d) at (x : (v, s, d) Value.t) =
  match at with None -> x | Some p -> Exec.at (Devices.rebrand p) x

(* Each result's bytes on the host, in C order, at [windows], each NaN made
   one: a constant result computed at [at], or else on the host. *)
let bytes (type r) ?at windows (op : r Value.prim) (r : r) =
  let host : b Devices.placement = Devices.rebrand (Devices.one Devices.host 0) in
  let arrays = ref [] in
  Prim.iter_results
    (fun x ->
      let x = reach at x in
      let x = Exec.run ~by:"t" (Value.Place (Devices.rebrand host, x)) in
      arrays := A.Any (Iarray.get (Prim.arrays x) 0) :: !arrays)
    op r;
  Array.mapi
    (fun i (A.Any a) ->
      let a =
        Option.get
          (A.move
             (Slice (Iarray.of_array (windows i (L.shape (A.layout a)))))
             a)
      in
      let a = any_nan (A.copy a) in
      A.to_array (Option.get (A.bitcast D.Uint8 (A.copy a))))
    (Array.of_list (List.rev !arrays))

let agrees (Op op, w) =
  let rule = Prim.rule op in
  let whole = Exec.run ~by:"t" op in
  let view i x =
    let win = Route.window rule ~operand:i (Prim.shape x) w in
    let a =
      Option.get
        (A.move (Slice (Iarray.of_array win)) (Iarray.get (Prim.arrays x) 0))
    in
    Value.Array { at = Prim.placement x; a; dead = Value.Live }
  in
  let local = Prim.local (Prim.mapi view op) w in
  let part = Exec.run ~by:"t" local in
  (* Each result is cut where the first is, and whole along any other axis: a
     routine's pivots or [r] factor share the first's batch alone. *)
  let first = shape_of_results (Op op) in
  let window _ s =
    Array.mapi
      (fun a d ->
        if a < Array.length w && w.(a) <> full first.(a) then w.(a) else full d)
      s
  in
  let whole = bytes window op whole in
  let part = bytes (fun _ s -> Array.map full s) local part in
  Array.iteri
    (fun i expected ->
      equal (array int) ~msg:(Printf.sprintf "result %d" i) expected part.(i))
    whole

let unloaded_loop (Op op) =
  match op with
  | Value.Reduce { loads = [||]; _ } | Value.Scan { loads = [||]; _ } -> true
  | _ -> false

let law (o, w) =
  let cut =
    Array.exists2 (fun (r : M.range) d -> r.count <> d) w (shape_of_results o)
  in
  cover_families "a cut window of" o cut;
  cover "a cut window of a loop of no load" (cut && unloaded_loop o);
  agrees (o, w)

(* [x], float32, as float64 values where it lies, their magnitudes where
   [abs]. *)
let widened (type v s d) ~abs (x : (v, s, d) Value.t) :
    (float, D.float64_elt, d) Value.t =
  match D.equal_witness (Prim.dtype x) D.Float32 with
  | Some Type.Equal ->
      let xs = A.to_array (Iarray.get (Prim.arrays x) 0) in
      let xs = if abs then Array.map Float.abs xs else xs in
      let a = A.to_device (m 0) (A.of_array D.Float64 (Prim.shape x) xs) in
      Value.Array { at = Prim.placement x; a; dead = Value.Live }
  | None -> invalid_arg "widened: not float32"

(* [x]'s elements on the host at the window [w], in C order. *)
let floats_at (type s d) (w : M.range array) (x : (float, s, d) Value.t) =
  let host = Devices.rebrand (Devices.one Devices.host 0) in
  let x = Exec.run ~by:"t" (Value.Place (host, x)) in
  let a = Iarray.get (Prim.arrays x) 0 in
  A.to_array (A.copy (Option.get (A.move (Slice (Iarray.of_array w)) a)))

let gamma n u = Float.of_int n *. u /. (1. -. (Float.of_int n *. u))

(* A float contraction's window is an ordinary contraction of the operands'
   windows, its library's order chosen from the window's extents: within
   the bound {!Nx.contract} states of the exact sum, and the same on a
   second run. Its float64 reference sums float32 products, each exact. *)
let bound_law ((Op op as o), (w : M.range array)) =
  match op with
  | Value.Contract { batch; contracting; a; b; init; out; _ } -> (
      match D.equal_witness out D.Float32 with
      | None -> invalid_arg "bound_law: a float32 contraction"
      | Some Type.Equal ->
          let k =
            Iarray.fold_left (fun n (i, _) -> n * Prim.dim a i) 1 contracting
          in
          let s = shape_of_results o and nb = Iarray.length batch in
          let cut = Array.exists2 (fun (r : M.range) d -> r.count <> d) w s in
          let rows = w.(nb).count and cols = w.(nb + 1).count in
          let all p = Array.for_all p w and some p = Array.exists p w in
          cover "a cut window" cut;
          cover "a cut window of 4 rows or fewer"
            (cut && rows > 0 && rows <= 4);
          cover "a cut window of 64 outputs or more, rows and columns above 4"
            (cut && rows > 4 && cols > 4 && rows * cols >= 64);
          cover "an empty window" (some (fun (r : M.range) -> r.count = 0));
          cover "a window of one output"
            (all (fun (r : M.range) -> r.count = 1));
          let view i x =
            let win = Route.window (Prim.rule op) ~operand:i (Prim.shape x) w in
            let a = Iarray.get (Prim.arrays x) 0 in
            let a = Option.get (A.move (Slice (Iarray.of_array win)) a) in
            Value.Array { at = Prim.placement x; a; dead = Value.Live }
          in
          let local = Prim.local (Prim.mapi view op) w in
          let run () = Exec.run ~by:"t" local in
          let whole y = floats_at (Array.map full (Prim.shape y)) y in
          let got = whole (run ()) in
          let bits = Array.map Int32.bits_of_float in
          equal (array int32) ~msg:"a second run" (bits got)
            (bits (whole (run ())));
          let wide ~abs =
            floats_at w
              (Exec.run ~by:"t"
                 (Value.Contract
                    {
                      batch;
                      contracting;
                      acc = D.Any D.Float64;
                      out = D.Float64;
                      a = widened ~abs a;
                      b = widened ~abs b;
                      init = Option.map (widened ~abs) init;
                      a_scale = None;
                      b_scale = None;
                      segments = None;
                    }))
          in
          let exact = wide ~abs:false and mag = wide ~abs:true in
          cover "a NaN result" (Array.exists Float.is_nan exact);
          let u = Float.ldexp 1. (-24) and u64 = epsilon_float /. 2. in
          let g = gamma (k + 1) (2. *. u) +. (2. *. gamma (k + 1) (2. *. u64))
          in
          let show x =
            if Float.is_nan x then "nan" else Printf.sprintf "%h" x
          in
          Array.iteri
            (fun i e ->
              let msg = Printf.sprintf "output %d against %h" i e in
              if Float.is_nan e then equal string ~msg "nan" (show got.(i))
              else
                less float_exact ~msg ~than:((g *. mag.(i)) +. Float.min_float)
                  (Float.abs (got.(i) -. e)))
            exact)
  | _ -> invalid_arg "bound_law: a contraction"

(* NaNs of both signs, as a negated NaN gives, in a transform's lines and a
   contraction's terms. *)
let range start count = { M.start; count; step = 1 }

let test_fft_nans () =
  let re =
    [|
      3.3; 3.3; 1.; -1e7; nan; 3.3; -1e7; 0.; 1.; -1e7; 1.; nan; -0.; nan; 0.;
      0.1; nan; 1.; 0.; nan; -2.5e-3; 1.; -2.5e-3; -0.; 1e7; nan; 1.; -0.;
      -1e7; 0.;
    |]
  in
  let x =
    value D.Complex64 [| 5; 2; 3 |]
      (Array.map (fun re -> { Complex.re; im = -.re }) re)
  in
  let op = Value.Fft (C2c { direction = Forward; axes = [| 0 |]; x }) in
  agrees (Op op, [| full 5; range 1 1; range 1 2 |])

let test_contract_nans () =
  let a =
    value D.Float32 [| 8; 3 |]
      (Array.init 24 (fun i -> [| nan; -.nan; 1. |].(i mod 3)))
  in
  let b = value D.Float32 [| 3; 8 |] (Array.make 24 1.) in
  let batch : _ iarray = [||] and contracting : _ iarray = [| (1, 0) |] in
  let op =
    Value.Contract
      {
        batch;
        contracting;
        acc = D.Any D.Float32;
        out = D.Float32;
        a;
        b;
        init = None;
        a_scale = None;
        b_scale = None;
        segments = None;
      }
  in
  agrees (Op op, [| range 3 4; range 1 5 |])

(* Reductions over cut axes *)

let quad : b Devices.t = Devices.mint ~by:"t" [ m 0; m 1; m 2; m 3 ]
let mesh = Devices.mesh_v ~by:"t" quad [ ("a", 2); ("b", 2) ]

(* Where loads lie: split over the four devices, cut over both mesh axes, two
   axes cut over one each, or one cut over the minor mesh axis, the major
   holding copies. *)
type where = Split of int | Both of int | Two of int * int | Minor of int

let placement_of = function
  | Split a -> Devices.split ~by:"t" ~axis:a quad
  | Both a -> Devices.mesh ~by:"t" mesh [ (a, [ "a"; "b" ]) ]
  | Two (a, a') -> Devices.mesh ~by:"t" mesh [ (a, [ "a" ]); (a', [ "b" ]) ]
  | Minor a -> Devices.mesh ~by:"t" mesh [ (a, [ "b" ]) ]

let cut_axes = function
  | Split a | Both a | Minor a -> [ a ]
  | Two (a, a') -> [ a; a' ]

(* Floats and int8s of few values, so that extremes tie across windows, with
   NaNs of three bit patterns, signed zeros and infinities. *)
let tied_floats =
  [
    1.; 1.; 2.; 0.; -0.; Float.nan; -.Float.nan;
    Int32.float_of_bits 0x7fc00123l; Float.infinity; Float.neg_infinity;
  ]

let tied_ints = [ 0; 1; 1; -1; 127; -128 ]

type cut_case = {
  shape : int array;
  axes : int array;
  where : where;
  xs : float array;
  ys : int array;
}

let pp_cut_case ppf c =
  let ints a = String.concat ";" (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "[%s] along [%s], cut %s: [%s] [%s]" (ints c.shape)
    (ints c.axes)
    (String.concat "," (List.map string_of_int (cut_axes c.where)))
    (String.concat ";"
       (Array.to_list (Array.map (fun x -> Printf.sprintf "%h" x) c.xs)))
    (ints c.ys)

let cut_case =
  let open Gen in
  with_pp pp_cut_case
  @@ let* rank = int_range 2 3 in
     let* shape = array ~size:(constant rank) (int_range 1 5) in
     let axis = int_range 0 (rank - 1) in
     let* where =
       one_of
         [
           map (fun a -> Split a) axis;
           map (fun a -> Both a) axis;
           map
             (fun (a, a') -> if a = a' then Both a else Two (a, a'))
             (pair axis axis);
           map (fun a -> Minor a) axis;
         ]
     in
     (* The first cut axis reduced, and any other axis or not. *)
     let* extra = array ~size:(constant rank) bool in
     let axes =
       Array.of_list
         (List.filter
            (fun a -> a = List.hd (cut_axes where) || extra.(a))
            (List.init rank Fun.id))
     in
     let n = numel shape in
     let* xs = array ~size:(constant n) (of_list tied_floats) in
     let+ ys = array ~size:(constant n) (of_list tied_ints) in
     { shape; axes; where; xs; ys }

(* A reduction of every kind over a float32 and an int8 load: the floats'
   first maximum and its position, minimum, sum and moments; the int8s' sum
   into int32, rounding from int8, and first minimum and its position. The
   results from [rounding] on, the float sum's and the moments, round their
   partials: their bits are a function of the placement, which test_exec
   bounds. *)
let mixed (x : (float, D.float32_elt, b) Value.t)
    (y : (int, D.int8_signed_elt, b) Value.t)
    shape axes =
  Value.Reduce
    {
      layout = L.contiguous shape;
      axes = Iarray.of_array axes;
      prog =
        P.v ~ins:[| D.Any D.Float32; D.Any D.Int8 |] [| In 0; In 1 |]
          ~outs:[| 0; 1 |];
      reductions =
        Value.
          [
            Arg (Max, 0, D.Float32);
            Monoid (Min, 0, D.Float32);
            Monoid (Sum, 1, D.Int32);
            Arg (Min, 1, D.Int8);
            Monoid (Sum, 0, D.Float32);
            Moments (0, D.Float32);
          ];
      loads = [| Plain x; Plain y |];
    }

let rounding = 6

let cut_law c =
  let along a = Array.mem a c.axes in
  let cut = cut_axes c.where in
  let reduced = List.filter along cut in
  cover "a cut reduced axis after the first reduced one"
    (List.exists (fun a -> a > c.axes.(0)) reduced);
  cover "a cut axis kept beside a cut reduced one"
    (List.exists (fun a -> not (along a)) cut);
  cover "an empty window along a reduced axis"
    (List.exists
       (fun a ->
         let t = match c.where with Two _ | Minor _ -> 2 | _ -> 4 in
         c.shape.(a) < t)
       reduced);
  cover "a reduced axis cut, the other mesh axis holding copies"
    (match c.where with Minor _ -> true | _ -> false);
  let x = value D.Float32 c.shape c.xs and y = value D.Int8 c.shape c.ys in
  let one = mixed x y c.shape c.axes in
  let whole _ s = Array.map full s in
  let expected = bytes whole one (Exec.run ~by:"t" one) in
  let p = Devices.rebrand (placement_of c.where) in
  let place v = Exec.run ~by:"t" (Value.Place (p, v)) in
  let cut = mixed (place x) (place y) c.shape c.axes in
  let actual = bytes whole cut (Exec.run ~by:"t" cut) in
  let again = bytes whole cut (Exec.run ~by:"t" cut) in
  Array.iteri
    (fun i e ->
      let msg = Printf.sprintf "result %d" i in
      if i < rounding then equal (array int) ~msg e actual.(i);
      equal (array int) ~msg:(msg ^ ", run again") actual.(i) again.(i))
    expected

let () =
  exit
    (run "nx windows"
       [
         group "laws"
           [
             prop ~count:1000
               "an operation over its operands' windows is that window"
               (Gen.with_pp pp_case case) law;
             prop ~count:300
               "a float contraction's window is within its bound and the \
                same on every run"
               (Gen.with_pp pp_case
                  (Gen.bind float_contract_op wide_window_of))
               bound_law;
             test "a transform's window over NaNs of both signs"
               test_fft_nans;
             test "a contraction's window over NaNs of both signs"
               test_contract_nans;
             prop ~count:1000
               "a reduction over loads cut along its axes gives an exact \
                kind's bits on one device, each kind's on every run"
               cut_case cut_law;
           ];
       ])
