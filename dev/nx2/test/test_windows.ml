(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The window law, through the engine's private Prim, Route and Exec, copied
   here: on one device, an operation over the windows its rule gives its
   operands equals that window of its whole result, bitwise. Each device of a
   set computes its window so, so this is the law a set's results rest on. *)

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

let pad_of rank =
  let open Gen in
  let* a = axis_of rank in
  let* lo = int_range (-1) 2 in
  let* hi = int_range (-1) 2 in
  let* interior = int_range 0 1 in
  let+ windowed = bool in
  let at v = Array.init rank (fun i -> if i = a then v else 0) in
  ( a,
    {
      Nx_kernel.lo = at lo;
      hi = at hi;
      interior = at interior;
      windows =
        (if windowed then [| { M.axis = a; size = 2; step = 1; dilation = 1 } |]
         else [||]);
    } )

(* A padded shape its pad admits, or [None]. *)
let padded pad s = Result.to_option (Nx_kernel.Shape.padded pad s)

let padded_map_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* _, pad = pad_of rank in
  match padded pad s with
  | None -> map_op
  | Some loop ->
      let* x = f32 s in
      let+ y = f32 loop in
      Op
        (Value.Map
           {
             layout = L.contiguous loop;
             prog = coords;
             outs = Value.[ D.Float32; D.Float32 ];
             loads = [| Padded { x; fill = 0.5; pad }; Plain y |];
           })

let reduce_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s =
    (* Arg finds no extreme of no term. *)
    array ~size:(constant rank) (int_range 1 6)
  in
  let* keep = array ~size:(constant rank) bool in
  let axes =
    Array.of_list (List.filter (fun a -> not keep.(a)) (List.init rank Fun.id))
  in
  let axes = if axes = [||] then [| rank - 1 |] else axes in
  let+ x = f32 s in
  Op
    (Value.Reduce
       {
         layout = L.contiguous s;
         axes;
         prog = one;
         reductions =
           Value.[ Monoid (Sum, 0, D.Float32); Arg (Max, 0, D.Float32) ];
         loads = [| Plain x |];
       })

let scan_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let+ x = f32 s in
  Op
    (Value.Scan
       {
         layout = L.contiguous s;
         axis;
         prog = one;
         reduction = Monoid (Sum, 0, D.Float32);
         loads = [| Plain x |];
       })

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

let sort_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let* descending = bool in
  let+ x = f32 s in
  Op (Value.Sort { axis; descending; k = None; x })

let fold_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* _, pad = pad_of rank in
  match padded pad s with
  | None -> sort_op
  | Some loop ->
      let+ x = f32 loop in
      Op (Value.Fold { shape = s; pad; x })

let fft_op =
  let open Gen in
  let* rank = int_range 1 3 in
  let* s = shape_of rank in
  let* axis = axis_of rank in
  let+ re = array ~size:(constant (numel s)) term in
  let x =
    value D.Complex64 s (Array.map (fun re -> { Complex.re; im = -.re }) re)
  in
  Op (Value.Fft (C2c { direction = Forward; axes = [| axis |]; x }))

let lu_op =
  let open Gen in
  let* batch = int_range 0 4 in
  let* n = int_range 1 4 in
  let+ a = f32 [| batch; n; n |] in
  Op (Value.Linalg (Lu a))

(* A contraction of a batch, rows, columns and contracted extents from 0 to 9,
   in the order nx gives it: windows of rows and columns cross the extents
   that order is chosen by. *)
let contract_op =
  let open Gen in
  let* nb = int_range 0 1 in
  let extent = of_list [ 0; 1; 3; 5; 8; 9 ] in
  let* rows = extent in
  let* cols = extent in
  let* k = of_list [ 0; 1; 3; 17; 40 ] in
  let* init = bool in
  let shape = if nb = 0 then [||] else [| 3 |] in
  let batch = if nb = 0 then [||] else [| (0, 0) |] in
  let contracting = [| (nb + 1, nb) |] in
  let* a = f32 (Array.append shape [| rows; k |]) in
  let* b = f32 (Array.append shape [| k; cols |]) in
  let+ init =
    if init then map Option.some (f32 (Array.append shape [| rows; cols |]))
    else constant None
  in
  let order = Prim.contract_order ~batch ~contracting a b in
  Op
    (Value.Contract
       {
         order;
         batch;
         contracting;
         acc = D.Any D.Float32;
         out = D.Float32;
         a;
         b;
         init;
       })

(* The law *)

(* Whether the results keep their axis [a] whole by [rule]: every device's
   window holds all of it. *)
let kept (rule : Route.rule) a =
  match rule with
  | Elementwise | Gather _ | Contract _ -> false
  | Reduce { axes; whole } ->
      List.exists
        (fun b ->
          (not (Array.mem b axes))
          && b - List.length (List.filter (fun c -> c < b) (Array.to_list axes))
             = a)
        (Array.to_list whole)
  | Whole axes -> Array.mem a axes
  | Into axis -> a = axis
  | Move _ | Replicated -> true

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

let case =
  let open Gen in
  let* (Op op as o) =
    one_of
      [
        map_op;
        padded_map_op;
        reduce_op;
        scan_op;
        gather_op;
        scatter_op;
        sort_op;
        fold_op;
        fft_op;
        lu_op;
        contract_op;
      ]
  in
  let rule = Prim.rule op and s = shape_of_results o in
  let+ w =
    let range a d =
      if kept rule a || d = 0 then constant (full d)
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

let pp_case ppf (Op op, (w : M.range array)) =
  Format.fprintf ppf "%a at [%s]" Prim.pp op
    (String.concat "; "
       (Array.to_list
          (Array.map
             (fun (r : M.range) -> Printf.sprintf "%d+%d" r.start r.count)
             w)))

(* Each result's bytes on the host, in C order, at [windows]: a constant
   result, an expansion's of no element, computed there. *)
let bytes (type r) windows (op : r Value.prim) (r : r) =
  let host : b Devices.placement = Devices.rebrand (Devices.one Devices.host 0) in
  let arrays = ref [] in
  Prim.iter_results
    (fun x ->
      let x = Exec.run ~by:"t" (Value.Place (Devices.rebrand host, x)) in
      arrays := A.Any (Iarray.get (Prim.arrays x) 0) :: !arrays)
    op r;
  Array.mapi
    (fun i (A.Any a) ->
      let a =
        Option.get (A.move (Slice (windows i (L.shape (A.layout a)))) a)
      in
      A.to_array (Option.get (A.bitcast D.Uint8 (A.copy a))))
    (Array.of_list (List.rev !arrays))

let agrees (Op op, w) =
  let rule = Prim.rule op in
  let whole = Exec.run ~by:"t" op in
  let view i x =
    let win = Route.window rule ~operand:i (Prim.shape x) w in
    let a = Option.get (A.move (Slice win) (Iarray.get (Prim.arrays x) 0)) in
    Value.Array { at = Prim.placement x; a; dead = Value.Live }
  in
  let local = Prim.local (Prim.mapi view op) w in
  let part = Exec.run ~by:"t" local in
  (* A result of another rank than the first, a routine's pivots, has the
     window's batch and is whole after it. *)
  let window i s =
    if i = 0 || Array.length s = Array.length w then
      Array.mapi (fun a d -> if a < Array.length w then w.(a) else full d) s
    else Array.mapi (fun a d -> if a = 0 then w.(0) else full d) s
  in
  let whole = bytes window op whole in
  let part =
    bytes (fun _ s -> Array.map full s) local part
  in
  Array.iteri
    (fun i expected ->
      equal (array int) ~msg:(Printf.sprintf "result %d" i) expected part.(i))
    whole

(* Rows of [1e7 + 0.1 - 1e7]: a window of 4 of 8 rows sums each in the order of
   8. *)
let law ((Op op as o), w) =
  let cut =
    Array.exists2 (fun (r : M.range) d -> r.count <> d) w (shape_of_results o)
  in
  cover "a cut window" cut;
  cover "a cut contraction"
    (cut && match op with Value.Contract _ -> true | _ -> false);
  agrees (o, w)

let test_contract_rows () =
  let a =
    value D.Float32 [| 8; 3 |]
      (Array.init 24 (fun i -> [| 1e7; 0.1; -1e7 |].(i mod 3)))
  in
  let b = value D.Float32 [| 3; 8 |] (Array.make 24 1.) in
  let batch = [||] and contracting = [| (1, 0) |] in
  let op =
    Value.Contract
      {
        order = Prim.contract_order ~batch ~contracting a b;
        batch;
        contracting;
        acc = D.Any D.Float32;
        out = D.Float32;
        a;
        b;
        init = None;
      }
  in
  let w = [| { M.start = 0; count = 4; step = 1 }; full 8 |] in
  agrees (Op op, w)

let () =
  exit
    (run "nx windows"
       [
         group "laws"
           [
             prop ~count:1000
               "an operation over its operands' windows is that window"
               (Gen.with_pp pp_case case) law;
             test "a contraction's rows keep the whole's order"
               test_contract_rows;
           ];
       ])
