(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Operations as data, through the engine's private Prim, copied here: each rule
   against the array layer that computes what it states. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog

type b
type op = Op : 'r Value.prim -> op

let m = Nx_support.memory
let s2 : b Devices.t = Devices.mint ~by:"t" [ m 0; m 1 ]
let at0 = Devices.one s2 0
let layout = Testable.make ~pp:L.pp ~equal:L.equal
let invalid f = raises_match (Exn.invalid_arg ~substring:"Nx.f: ") f
let floats shape = Array.init (Array.fold_left ( * ) 1 shape) float_of_int

let value d a : ('v, 's, b) Value.t =
  Value.Array
    { at = Devices.one s2 d; a = A.to_device (m d) a; dead = Value.Live }

let f32 ?(d = 0) shape =
  value d (A.of_array D.Float32 (Iarray.of_array shape) (floats shape))

(* A maker that records the forms it is given in [forms], and one that must not
   run. *)
let recording forms k (f : (_, _, _) Value.form) =
  forms := (k, D.Any f.dtype, f.layout, f.placement) :: !forms;
  let a = A.create Rig.host f.dtype (L.shape f.layout) in
  Value.Array { at = Devices.one Devices.host 0; a; dead = Value.Live }

let refusing _ _ = fail "the maker ran before the rule refused"

let forms (Op op) =
  let forms = ref [] in
  ignore (Prim.results ~by:"Nx.f" (fun k f -> recording forms k f) op);
  List.rev !forms

let one_form op =
  match forms op with
  | [ (_, dt, l, (p : b Devices.placement)) ] -> (dt, l, p)
  | fs -> failf "%d results where one was expected" (List.length fs)

let refuses (Op op) = invalid (fun () -> Prim.results ~by:"Nx.f" refusing op)

(* Maps *)

(* [x0 + x1] and [x0 * x1], over two float32 operands. *)
let add_mul =
  P.v
    ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
    [| In 0; In 1; Op2 (Binary Add, 0, 1); Op2 (Binary Mul, 0, 1) |]
    ~outs:[| 2; 3 |]

let two = Value.[ D.Float32; D.Float32 ]

(* The coordinate along axis 0: a program of no input. *)
let coordinate = P.v ~ins:[||] [| Coord 0 |] ~outs:[| 0 |]

let split_x () : (float, D.float32_elt, b) Value.t =
  Value.Shards
    {
      at = Devices.split ~by:"t" ~axis:0 s2;
      shape = [| 4 |];
      arrays =
        Iarray.of_list
          [
            A.to_device (m 0) (A.of_array D.Float32 [| 2 |] [| 0.; 1. |]);
            A.to_device (m 1) (A.of_array D.Float32 [| 2 |] [| 2.; 3. |]);
          ];
      dead = Value.Live;
    }

let i32 : (int32, D.int32_elt, b) Value.t =
  value 0 (A.of_array D.Int32 [| 2 |] [| 1l; 2l |])

let maps =
  group "maps"
    [
      test
        "a map's results are C-contiguous of its shape, at its loads' placement"
        (fun () ->
          let x = f32 [| 2; 3 |] and y = f32 [| 2; 3 |] in
          let fs =
            forms
              (Op
                 (Map
                    {
                      layout = L.contiguous [| 2; 3 |];
                      prog = add_mul;
                      outs = two;
                      loads = [| Plain x; Plain y |];
                      at = None;
                    }))
          in
          equal (list int) [ 0; 1 ] (List.map (fun (k, _, _, _) -> k) fs);
          List.iter
            (fun (_, _, l, p) ->
              equal layout (L.contiguous [| 2; 3 |]) l;
              equal bool true (Devices.equal p at0))
            fs);
      test "a map over split loads is split" (fun () ->
          let x = split_x () in
          List.iter
            (fun (_, _, _, p) ->
              equal bool true
                (Devices.equal p (Devices.split ~by:"t" ~axis:0 s2)))
            (forms
               (Op
                  (Map
                     {
                       layout = L.contiguous [| 4 |];
                       prog = add_mul;
                       outs = two;
                       loads = [| Plain x; Plain x |];
                       at = None;
                     }))));
      test "a creation lies where its map names" (fun () ->
          let one =
            P.v ~ins:[||]
              [| Const (D.Any D.Float32, P.bits D.Float32 1.) |]
              ~outs:[| 0 |]
          in
          let lies at =
            let _, _, p =
              one_form
                (Op
                   (Map
                      {
                        layout = L.contiguous [| 4 |];
                        prog = one;
                        outs = [ D.Float32 ];
                        loads = [||];
                        at = Some at;
                      }))
            in
            equal bool true (Devices.equal p at)
          in
          lies at0;
          lies (Devices.split ~by:"t" ~axis:0 s2));
      cases "a map refuses, before any result is made" ~name:fst
        [
          ( "one load for two operands",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = two;
                   loads = [| Plain (f32 [| 2 |]) |];
                   at = None;
                 }) );
          ( "a load of another dtype",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = two;
                   loads = [| Plain (f32 [| 2 |]); Plain i32 |];
                   at = None;
                 }) );
          ( "loads of another shape",
            Op
              (Map
                 {
                   layout = L.contiguous [| 3 |];
                   prog = add_mul;
                   outs = two;
                   loads = [| Plain (f32 [| 2 |]); Plain (f32 [| 2 |]) |];
                   at = None;
                 }) );
          ( "a result of another dtype",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = [ D.Float32; D.Float64 ];
                   loads = [| Plain (f32 [| 2 |]); Plain (f32 [| 2 |]) |];
                   at = None;
                 }) );
          ( "one result for two outputs",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = [ D.Float32 ];
                   loads = [| Plain (f32 [| 2 |]); Plain (f32 [| 2 |]) |];
                   at = None;
                 }) );
          ( "a layout that is not C-contiguous",
            Op
              (Map
                 {
                   layout = L.v ~offset:0 ~strides:[| 2 |] [| 2 |];
                   prog = coordinate;
                   outs = [ D.Int64 ];
                   loads = [||];
                   at = Some at0;
                 }) );
          ( "a map of no loads that names no placement",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = coordinate;
                   outs = [ D.Int64 ];
                   loads = [||];
                   at = None;
                 }) );
          ( "a map with loads that names a placement",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = two;
                   loads = [| Plain (f32 [| 2 |]); Plain (f32 [| 2 |]) |];
                   at = Some at0;
                 }) );
          ( "a creation at a placement that cuts an axis its shape lacks",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = coordinate;
                   outs = [ D.Int64 ];
                   loads = [||];
                   at = Some (Devices.split ~by:"t" ~axis:1 s2);
                 }) );
        ]
        (fun (_, op) -> refuses op);
      test "a map prints its program as expressions of its operands" (fun () ->
          let x = f32 [| 2 |] in
          equal string
            "Map [add(x0, x1); mul(x0, x1)] (x0: float32 [2] at m0) (x1: \
             float32 [2] at m0)"
            (Format.asprintf "%a" Prim.pp
               (Value.Map
                  {
                    layout = L.contiguous [| 2 |];
                    prog = add_mul;
                    outs = two;
                    loads = [| Plain x; Plain x |];
                    at = None;
                  })));
      test "a node read twice prints once, named by its index" (fun () ->
          let prog =
            P.v
              ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
              [| In 0; In 1; Op2 (Binary Add, 0, 1); Op2 (Binary Mul, 2, 2) |]
              ~outs:[| 3 |]
          in
          let x = f32 [| 2 |] in
          equal string
            "Map n2 = add(x0, x1); [mul(n2, n2)] (x0: float32 [2] at m0) (x1: \
             float32 [2] at m0)"
            (Format.asprintf "%a" Prim.pp
               (Value.Map
                  {
                    layout = L.contiguous [| 2 |];
                    prog;
                    outs = Value.[ D.Float32 ];
                    loads = [| Plain x; Plain x |];
                    at = None;
                  })));
      test "a chain of shared nodes prints in its program's size" (fun () ->
          (* Inline, 12 adds print about 36 KiB: a failure stays cheap. *)
          let adds = 12 in
          let nodes =
            Iarray.init (adds + 1) (fun i ->
                if i = 0 then P.In 0 else P.Op2 (Binary Add, i - 1, i - 1))
          in
          let prog = P.v ~ins:[| D.Any D.Float32 |] nodes ~outs:[| adds |] in
          let printed =
            Format.asprintf "%a" Prim.pp
              (Value.Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog;
                   outs = Value.[ D.Float32 ];
                   loads = [| Plain (f32 [| 2 |]) |];
                   at = None;
                 })
          in
          less ~than:2_000 int (String.length printed));
    ]

(* Movements and bitcasts, against nx.array's *)

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_move ppf = function
  | M.Reshape s -> Format.fprintf ppf "reshape %a" pp_ints (Iarray.to_array s)
  | Broadcast s -> Format.fprintf ppf "broadcast %a" pp_ints (Iarray.to_array s)
  | Permute p -> Format.fprintf ppf "permute %a" pp_ints (Iarray.to_array p)
  | Slice rs ->
      Format.fprintf ppf "slice %s"
        (String.concat " "
           (Iarray.to_list
              (Iarray.map
                 (fun (r : M.range) ->
                   Printf.sprintf "%d+%d/%d" r.start r.count r.step)
                 rs)))
  | Window ws ->
      Format.fprintf ppf "window %s"
        (String.concat " "
           (Iarray.to_list
              (Iarray.map
                 (fun (w : M.window) ->
                   Printf.sprintf "%d:%d/%d" w.axis w.size w.step)
                 ws)))

(* A value of a shape, transposed or not, and a movement drawn for its shape:
   valid or not. *)
let moved =
  let open Gen in
  with_pp
    (fun ppf (s, flip, mv) ->
      Format.fprintf ppf "%a%s by %a" pp_ints s
        (if flip then " transposed" else "")
        pp_move mv)
    (let* rank = int_range 1 3 in
     let* s0 = array ~size:(constant rank) (of_list [ 1; 2; 4 ]) in
     let* flip = bool in
     let s = if flip then Array.of_list (List.rev (Array.to_list s0)) else s0 in
     let n = Array.fold_left ( * ) 1 s in
     let+ mv =
       one_of
         [
           map
             (fun p -> M.Permute (Iarray.of_list p))
             (permutation (List.init rank Fun.id));
           constant (M.Reshape [| n |]);
           constant (M.Reshape [| n; 1 |]);
           constant (M.Broadcast (Iarray.of_array (Array.append [| 3 |] s)));
           constant
             (M.Broadcast
                (Iarray.of_array
                   (Array.map (fun d -> if d = 1 then 3 else d) s)));
           map
             (fun rev ->
               M.Slice
                 (Iarray.of_array
                    (Array.map
                       (fun d ->
                         if rev && d > 1 then
                           { M.start = d - 1; count = d / 2; step = -2 }
                         else { M.start = 0; count = d; step = 1 })
                       s)))
             bool;
           constant
             (M.Window [| { axis = 0; size = 1; step = 1; dilation = 1 } |]);
           constant (M.Reshape [| n + 1 |]);
         ]
     in
     (s0, flip, mv))

let array_of (type v s) (x : (v, s, b) Value.t) =
  require_match
    ~pp:(fun ppf _ -> Format.pp_print_string ppf "a value on two devices")
    (function
      | Value.Array { a; _ } -> Some a
      | Shards _ | Handle _ | Traced _ -> None)
    x

let moved_by mv x =
  match x with
  | Value.Array { at; a; _ } ->
      Value.Array { at; a = Option.get (A.move mv a); dead = Value.Live }
  | Shards _ | Handle _ | Traced _ -> x

let movements =
  group "movements"
    [
      prop "a movement's form is nx.array's move of the layout, or C order"
        moved (fun (s0, flip, mv) ->
          let x = f32 s0 in
          let rank = Array.length s0 in
          let x =
            if flip then
              moved_by (M.Permute (Iarray.init rank (fun i -> rank - 1 - i))) x
            else x
          in
          let s = Iarray.of_array (Prim.shape x) in
          match M.shape mv s with
          | exception Invalid_argument _ ->
              cover "refused" true;
              refuses (Op (Move (mv, x)))
          | s' -> (
              let _, l, p = one_form (Op (Move (mv, x))) in
              equal bool true (Devices.equal p at0);
              match A.move mv (array_of x) with
              | Some a' ->
                  cover "a view" true;
                  equal layout (A.layout a') l
              | None ->
                  cover "a copy" true;
                  equal layout (L.contiguous s') l));
      prop "a movement of a split value has its eager result's form" moved
        (fun (s0, _, mv) ->
          assume (s0.(0) mod 2 = 0);
          let host = Devices.one Devices.host 0 in
          let x =
            Place.value ~by:"t"
              (Devices.split ~by:"t" ~axis:0 s2)
              (Value.Array
                 {
                   at = host;
                   a = A.of_array D.Float32 (Iarray.of_array s0) (floats s0);
                   dead = Value.Live;
                 })
          in
          match Eval.run ~by:"t" (Move (mv, x)) with
          | exception Invalid_argument _ -> cover "refused" true
          | y ->
              let _, l, p = one_form (Op (Move (mv, x))) in
              let f = Prim.form y in
              cover "on two devices"
                (Devices.device f.placement = None);
              equal layout f.layout l;
              equal bool true
                (Devices.equal f.placement p));
    ]

(* A bitcast of [x] to [dt]: its form's layout against nx.array's bitcast of
   [x]'s array, or C order of the result's shape where nx.array has no view; a
   refusal where nx.array finds no result shape. *)
let bitcast (type v s w r) (dt : (w, r) D.t) (x : (v, s, b) Value.t) () =
  let a = array_of x in
  match A.bitcast dt a with
  | exception Invalid_argument _ -> refuses (Op (Bitcast (dt, x)))
  | expected -> (
      match one_form (Op (Bitcast (dt, x))) with
      | exception Invalid_argument msg ->
          (* No view and no shape: a trailing axis that is not the ratio. *)
          is_none ~msg expected
      | _, l, p -> (
          equal bool true (Devices.equal p at0);
          match expected with
          | Some a' -> equal layout (A.layout a') l
          | None -> equal bool true (L.is_contiguous l)))

let u8 shape =
  value 0
    (A.of_array D.Uint8 (Iarray.of_array shape)
       (Array.init (Array.fold_left ( * ) 1 shape) (fun i -> i mod 256)))

let bitcasts =
  group "bitcasts"
    [
      test "float32 to uint8 appends an axis of 4"
        (bitcast D.Uint8 (f32 [| 3; 2 |]));
      test "a transposed float32 to uint8"
        (bitcast D.Uint8 (moved_by (M.Permute [| 1; 0 |]) (f32 [| 3; 2 |])));
      test "uint8 to float32 removes a trailing axis of 4"
        (bitcast D.Float32 (u8 [| 2; 4 |]));
      test "a strided trailing axis widens through a copy"
        (bitcast D.Float32
           (moved_by
              (M.Slice
                 [|
                   { start = 0; count = 2; step = 1 };
                   { start = 0; count = 4; step = 2 };
                 |])
              (u8 [| 2; 8 |])));
      test "an offset float32 to uint8"
        (bitcast D.Uint8
           (moved_by
              (M.Slice
                 [|
                   { start = 1; count = 2; step = 1 };
                   { start = 0; count = 2; step = 1 };
                 |])
              (f32 [| 3; 2 |])));
      test "an offset uint8 to float32"
        (bitcast D.Float32
           (moved_by
              (M.Slice
                 [|
                   { start = 1; count = 1; step = 1 };
                   { start = 0; count = 4; step = 1 };
                 |])
              (u8 [| 2; 4 |])));
      test "float32 to int32 keeps the layout"
        (bitcast D.Int32 (moved_by (M.Permute [| 1; 0 |]) (f32 [| 3; 2 |])));
      test "a widening without a trailing axis of the ratio raises" (fun () ->
          refuses (Op (Bitcast (D.Float32, u8 [| 2; 3 |]))));
    ]

(* Places and checks *)

let others =
  group "places and checks"
    [
      test "a place lies where it is told, keeping a one-device layout"
        (fun () ->
          let x = moved_by (M.Permute [| 1; 0 |]) (f32 [| 3; 2 |]) in
          let _, l, p = one_form (Op (Place (Devices.one s2 1, x))) in
          equal bool true (Devices.equal p (Devices.one s2 1));
          equal layout (A.layout (array_of x)) l);
      test "a place split over two devices is C order of the whole" (fun () ->
          let split = Devices.split ~by:"t" ~axis:0 s2 in
          let _, l, p = one_form (Op (Place (split, f32 [| 4; 2 |]))) in
          equal bool true (Devices.equal p split);
          equal layout (L.contiguous [| 4; 2 |]) l);
      test "a place that splits an axis the value lacks raises" (fun () ->
          refuses (Op (Place (Devices.split ~by:"t" ~axis:1 s2, f32 [| 3 |]))));
      test "a check makes no result" (fun () ->
          let ok : (bool, D.bool_elt, b) Value.t =
            value 0 (A.of_array D.Bool [| 2 |] [| true; true |])
          in
          equal int 0
            (List.length
               (forms
                  (Op
                     (Check
                        {
                          ok;
                          data = [ f32 [| 2 |] ];
                          fail = (fun _ _ -> Exit);
                        })))));
      test "a check refuses data of another shape" (fun () ->
          let ok : (bool, D.bool_elt, b) Value.t =
            value 0 (A.of_array D.Bool [| 2 |] [| true; true |])
          in
          refuses
            (Op
               (Check
                  { ok; data = [ f32 [| 3 |] ]; fail = (fun _ _ -> Exit) })));
    ]

(* Operands and maps over them *)

let ops () =
  let x = f32 [| 2 |] in
  let ok : (bool, D.bool_elt, b) Value.t =
    value 0 (A.of_array D.Bool [| 2 |] [| true; false |])
  in
  [
    ( "Map",
      Op
        (Map
           {
             layout = L.contiguous [| 2 |];
             prog = add_mul;
             outs = two;
             loads = [| Plain x; Plain x |];
             at = None;
           }),
      2 );
    ("Copy", Op (Copy x), 1);
    ("Move", Op (Move (M.Reshape [| 1; 2 |], x)), 1);
    ("Bitcast", Op (Bitcast (D.Int32, x)), 1);
    ("Place", Op (Place (Devices.one s2 1, x)), 1);
    ("Check", Op (Check { ok; data = [ x ]; fail = (fun _ _ -> Exit) }), 2);
  ]

(* A matrix product of float32 [2; 3] and [3; 2], accumulated in [acc], from
   [init]. *)
let contractions =
  let a = f32 [| 2; 3 |] and b = f32 [| 3; 2 |] in
  let contract ?(contracting : _ iarray = [| (1, 0) |]) ?(acc = D.Any D.Float32)
      ?a_scale ?b_scale ?segments init : op =
    Op
      (Contract
         {
           batch = [||];
           contracting;
           acc;
           out = D.Float32;
           a;
           b;
           init;
           a_scale;
           b_scale;
           segments;
         })
  in
  let index shape xs =
    value 0 (A.of_array D.Int64 (Iarray.of_array shape) xs)
  in
  let codes shape =
    let n = Array.fold_left ( * ) 1 shape in
    value 0 (A.of_array D.Uint8 (Iarray.of_array shape) (Array.make n 127))
  in
  (* [a]'s scale a block per row, [b]'s one per column, [b]'s two columns
     the groups of [a]'s rows by [offsets]. *)
  let sa : b Value.scale = Value (f32 [| 2; 1 |])
  and sb : b Value.scale = Exponent (codes [| 1; 2 |])
  and offsets = index [| 3 |] [| 0L; 1L; 2L |] in
  let rows offsets : b Value.index Nx_kernel.segments =
    Rows { offsets; a_axis = 0; b_axis = 1 }
  in
  group "contractions"
    [
      test "a contraction gives its result's dtype and shape" (fun () ->
          let dt, l, _ = one_form (contract None) in
          equal bool true (dt = D.Any D.Float32);
          equal layout (L.contiguous [| 2; 2 |]) l);
      test "a scaled, segmented contraction's operands are a, b, init, the \
            scales and the offsets"
        (fun () ->
          let init = f32 [| 2 |] in
          let (Op op) =
            contract ~a_scale:sa ~b_scale:sb ~segments:(rows offsets)
              (Some init)
          in
          let arrays (Value.Any x) = A.buffer (array_of x) in
          let scale : b Value.scale -> b Value.any = function
            | Exponent x -> Any x
            | Value x -> Any x
          in
          let want =
            List.map arrays
              ([ Any a; Any b; Any init; scale sa; scale sb; Any offsets ]
                : b Value.any list)
          in
          let (Operands xs) = Prim.operands op in
          let buffer (Value.Any x) = A.buffer (Iarray.get (Prim.arrays x) 0) in
          equal (list bool) (List.map (fun _ -> true) want)
            (List.map2 (fun x w -> buffer x == w) xs want);
          let seen = ref [] in
          ignore
            (Prim.mapi
               (fun i x ->
                 seen := (i, A.buffer (Iarray.get (Prim.arrays x) 0)) :: !seen;
                 x)
               op);
          equal (list int) [ 0; 1; 2; 3; 4; 5 ] (List.rev_map fst !seen);
          equal (list bool) (List.map (fun _ -> true) want)
            (List.map2 (fun (_, x) w -> x == w) (List.rev !seen) want);
          let dt, l, _ = one_form (Op op) in
          equal bool true (dt = D.Any D.Float32);
          equal layout (L.contiguous [| 2 |]) l);
      cases "a contraction refuses" ~name:fst
        [
          ("an init of another shape", contract (Some (f32 [| 2; 3 |])));
          ("a float16 accumulator", contract ~acc:(D.Any D.Float16) None);
          ("a bool accumulator", contract ~acc:(D.Any D.Bool) None);
          ("pairs of other extents", contract ~contracting:[| (0, 0) |] None);
          ( "a scale whose extents do not divide",
            contract ~a_scale:(Value (f32 [| 2; 2 |])) None );
          ( "a scale of another rank",
            contract ~b_scale:(Exponent (codes [| 2 |])) None );
          ( "offsets of another shape",
            contract ~segments:(rows (index [| 2 |] [| 0L; 1L |])) None );
          ( "rows of a contracting axis",
            contract
              ~segments:(Rows { offsets; a_axis = 1; b_axis = 1 })
              None );
        ]
        (fun (_, op) ->
          refuses op;
          let (Op o) = op in
          invalid (fun () -> Prim.route ~by:"Nx.f" o));
    ]

(* Partials take a contraction cut along its contracting pairs, scaled in
   blocks of any shape or not, and one of rows by groups cut along [b]'s
   groups; a segmented one cut along its pairs raises. *)
let partials =
  let compute =
    Testable.make
      ~pp:(fun ppf c ->
        Format.pp_print_string ppf
          (match c with Route.Windows -> "windows" | Partials -> "partials"))
      ~equal:( = )
  in
  let placed p x = Eval.run ~by:"t" (Value.Place (p, x)) in
  let split axis = Devices.split ~by:"t" ~axis s2 in
  let every = Devices.on s2 in
  let codes shape =
    let n = Array.fold_left ( * ) 1 shape in
    placed every
      (value 0 (A.of_array D.Uint8 (Iarray.of_array shape) (Array.make n 127)))
  in
  (* [a] [[2; ks]] by [b] [[ks; 3]], cut along the first pair, [b] scaled by
     codes of [blocks]. *)
  let route ?segments ks blocks =
    let r = Array.length ks in
    let a = placed (split 1) (f32 (Array.append [| 2 |] ks)) in
    let b = placed (split 0) (f32 (Array.append ks [| 3 |])) in
    let contracting = Iarray.init r (fun i -> (i + 1, i)) in
    let segments =
      Option.map
        (fun g : b Value.index Nx_kernel.segments ->
          (* [Rows]' groups are [b]'s 3 columns. *)
          let os =
            match g with
            | `Rows -> [| 0L; 1L; 1L; 2L |]
            | `Contracting -> [| 0L; 8L |]
          in
          let offsets =
            placed every
              (value 0 (A.of_array D.Int64 [| Array.length os |] os))
          in
          match g with
          | `Rows -> Rows { offsets; a_axis = 0; b_axis = r }
          | `Contracting -> Contracting { offsets; pair = 0 })
        segments
    in
    let b_scale = Option.map (fun s -> Value.Exponent (codes s)) blocks in
    let op =
      Value.Contract
        {
          batch = [||];
          contracting;
          acc = D.Any D.Float32;
          out = D.Float32;
          a;
          b;
          init = None;
          a_scale = None;
          b_scale;
          segments;
        }
    in
    match (Prim.route ~by:"t" op).route with
    | Some r -> r.compute
    | None -> fail "a contraction of no route"
  in
  group "partials"
    [
      cases "a scaled contraction cut along its pairs"
        ~name:(fun (n, _, _, _) -> n)
        [
          ("unscaled", [| 64 |], None, Route.Partials);
          ("blocks of 32", [| 64 |], Some [| 2; 3 |], Partials);
          ("blocks of 3 in one of 4096", [| 96 |], Some [| 32; 3 |], Partials);
          ("a block of 64, all its terms", [| 64 |], Some [| 1; 3 |], Partials);
          ("blocks of 4096", [| 8192 |], Some [| 2; 3 |], Partials);
          ("blocks of 8192", [| 16384 |], Some [| 2; 3 |], Partials);
          ("blocks across two pairs", [| 8; 4 |], Some [| 4; 2; 3 |], Partials);
          ("blocks of a second pair", [| 8; 4 |], Some [| 8; 2; 3 |], Partials);
        ]
        (fun (_, ks, blocks, want) -> equal compute want (route ks blocks));
      (* Segments are computed uncut, so their pairs are read whole, which
         a device holding its tile of them does not hold. *)
      cases "a segmented contraction cut along its pairs raises"
        ~name:(fun (n, _) -> n)
        [ ("rows", `Rows); ("contracting", `Contracting) ]
        (fun (_, g) ->
          raises
            (Invalid_argument
               "t: operand 0 lies at split ~axis:1 [m0; m1] and is read at on \
                [m0; m1]; Nx.place (Placement.on s) x")
            (fun () -> route ~segments:g [| 64 |] None));
      (* [a] [[2; 4]] by [b] [[3; 4; 2]] split along its 3 groups, [b]'s
         scale a code per group and column too, the offsets whole on each
         device: [b]'s groups are a factor no axis of [a] shares. *)
      cases "rows by groups of b cut along them" ~name:fst
        [ ("unscaled", None); ("scaled per group", Some [| 3; 1; 2 |]) ]
        (fun (_, blocks) ->
          let a = placed every (f32 [| 2; 4 |]) in
          let b = placed (split 0) (f32 [| 3; 4; 2 |]) in
          let offsets =
            placed every
              (value 0 (A.of_array D.Int64 [| 4 |] [| 0L; 1L; 1L; 2L |]))
          in
          let b_scale =
            Option.map
              (fun s ->
                let n = Array.fold_left ( * ) 1 s in
                Value.Exponent
                  (placed (split 0)
                     (value 0
                        (A.of_array D.Uint8 (Iarray.of_array s)
                           (Array.make n 127)))))
              blocks
          in
          let op =
            Value.Contract
              {
                batch = [||];
                contracting = [| (1, 1) |];
                acc = D.Any D.Float32;
                out = D.Float32;
                a;
                b;
                init = None;
                a_scale = None;
                b_scale;
                segments = Some (Rows { offsets; a_axis = 0; b_axis = 0 });
              }
          in
          match (Prim.route ~by:"t" op).route with
          | Some r ->
              equal compute Route.Partials r.compute;
              equal bool ~msg:"the results whole on each device" true
                (Devices.equal every r.result)
          | None -> fail "a contraction of no route");
    ]

(* One-node programs *)

let unaries =
  P.
    [
      Neg; Recip; Abs; Sign; Sqrt; Exp; Exp2; Log; Log2; Log1p; Expm1; Sin;
      Cos; Tan; Asin; Acos; Atan; Sinh; Cosh; Tanh; Asinh; Acosh; Atanh; Erf;
      Floor; Ceil; Round; Trunc;
    ]

let binaries =
  P.
    [
      Add; Sub; Mul; Fdiv; Idiv; Mod; Pow; Atan2; Maximum; Minimum; And; Or;
      Xor; Threefry;
    ]

let compares = P.[ Equal; Not_equal; Less; Less_equal ]

let op1s =
  P.[ Copy; Cast; Bitcast ] @ List.map (fun u -> P.Unary u) unaries

let op2s =
  List.map (fun b -> P.Binary b) binaries
  @ List.map (fun c -> P.Compare c) compares

let name (D.Any dt) = D.name dt

(* [p]'s fields, each node by its kind and its fields. *)
let pp_prog ppf (p : P.t) =
  let list pp ppf xs =
    let sep ppf () = Format.fprintf ppf "; " in
    Format.pp_print_list ~pp_sep:sep pp ppf (Iarray.to_list xs)
  in
  let node ppf (n : P.node) =
    let k = Prim.kind n in
    match n with
    | In i | Coord i -> Format.fprintf ppf "%s %d" k i
    | Const (dt, _) -> Format.fprintf ppf "%s %s" k (name dt)
    | Op1 (_, dt, a) -> Format.fprintf ppf "%s %s %d" k (name dt) a
    | Op2 (_, a, b) -> Format.fprintf ppf "%s %d %d" k a b
    | Op3 (_, a, b, c) -> Format.fprintf ppf "%s %d %d %d" k a b c
  in
  let dtype ppf d = Format.pp_print_string ppf (name d) in
  Format.fprintf ppf "{ ins = [%a]; nodes = [%a]; outs = [%a] }" (list dtype)
    p.ins (list node) p.nodes (list Format.pp_print_int) p.outs

let prog = option (Testable.structural ~pp:pp_prog)
let pairs f = List.concat_map (fun x -> List.map (f x) D.all) D.all

(* Every key of each arity's programs: a kind and its operands' and result's
   dtypes. *)
let keys1 = List.concat_map (fun k -> pairs (fun x y -> (k, x, y))) op1s
let keys2 = List.concat_map (fun k -> List.map (fun x -> (k, x)) D.all) op2s

let keys3 =
  List.concat_map (fun k -> pairs (fun c x -> (k, c, x))) P.[ Where; Fma ]

let strf = Printf.sprintf

let name1 (k, x, y) =
  strf "%s %s to %s" (Prim.kind (Op1 (k, y, 0))) (name x) (name y)

let name2 (k, x) = strf "%s %s" (Prim.kind (Op2 (k, 0, 1))) (name x)

let name3 (k, c, x) =
  strf "%s %s %s" (Prim.kind (Op3 (k, 0, 1, 2))) (name c) (name x)

let test_node1 (k, D.Any x, D.Any y) =
  let want =
    if not (P.accepts1 k x y) then None
    else Some (P.of_node ~ins:[| D.Any x |] (Op1 (k, D.Any y, 0)))
  in
  equal prog want (Prim.node1 k x y)

let test_node2 (k, D.Any x) =
  let want =
    if not (P.accepts2 k x) then None
    else Some (P.of_node ~ins:[| D.Any x; D.Any x |] (Op2 (k, 0, 1)))
  in
  equal prog want (Prim.node2 k x)

let test_node3 (k, D.Any c, D.Any x) =
  let want =
    if not (P.accepts3 k c x) then None
    else
      Some (P.of_node ~ins:[| D.Any c; D.Any x; D.Any x |] (Op3 (k, 0, 1, 2)))
  in
  equal prog want (Prim.node3 k c x)

let test_node_found_allocates_nothing () =
  let add = P.Binary Add and cast = P.Cast in
  let calls () =
    ignore (Sys.opaque_identity (Prim.node1 cast D.Float32 D.Int8));
    ignore (Sys.opaque_identity (Prim.node2 add D.Float32));
    ignore (Sys.opaque_identity (Prim.node3 Where D.Bool D.Float32))
  in
  calls ();
  let before = Gc.minor_words () in
  for _ = 1 to 1000 do
    calls ()
  done;
  equal ~msg:"words" (float 0.5) 0. (Gc.minor_words () -. before)

(* The cache against a model, a map from each key to the program its first
   lookup gave, with lookups on two domains: each result must be the one some
   order of the calls gives. The keys span every kind and dtype, so that most
   lookups fill a slot first. *)

type key =
  | K1 of P.op1 * D.any * D.any
  | K2 of P.op2 * D.any
  | K3 of P.op3 * D.any * D.any

module Keys = Map.Make (struct
  type t = key

  let compare = compare
end)

let lookup = function
  | K1 (k, D.Any x, D.Any y) -> Prim.node1 k x y
  | K2 (k, D.Any x) -> Prim.node2 k x
  | K3 (k, D.Any c, D.Any x) -> Prim.node3 k c x

(* What prim.mli states the lookup of [key] gives. *)
let made = function
  | K1 (k, (D.Any x as dx), (D.Any y as dy)) ->
      if not (P.accepts1 k x y) then None
      else Some (P.of_node ~ins:[| dx |] (Op1 (k, dy, 0)))
  | K2 (k, (D.Any x as dx)) ->
      if not (P.accepts2 k x) then None
      else Some (P.of_node ~ins:[| dx; dx |] (Op2 (k, 0, 1)))
  | K3 (k, (D.Any c as dc), (D.Any x as dx)) ->
      if not (P.accepts3 k c x) then None
      else Some (P.of_node ~ins:[| dc; dx; dx |] (Op3 (k, 0, 1, 2)))

let model cache key =
  match Keys.find_opt key !cache with
  | Some p -> p
  | None ->
      let p = made key in
      cache := Keys.add key p !cache;
      p

let pp_key ppf = function
  | K1 (k, x, y) -> Format.pp_print_string ppf (name1 (k, x, y))
  | K2 (k, x) -> Format.pp_print_string ppf (name2 (k, x))
  | K3 (k, c, x) -> Format.pp_print_string ppf (name3 (k, c, x))

let keys f l =
  let a = Array.of_list (List.map f l) in
  Gen.with_pp pp_key
    (Gen.map (Array.get a) (Gen.int_range 0 (Array.length a - 1)))

let cache = abstract "c"

let cache_commands =
  let start () = ref Keys.empty and system () key = lookup key in
  [
    command "start" (Gen.unit @-> makes cache) start Fun.id;
    command "node1"
      (cache ^-> keys (fun (k, x, y) -> K1 (k, x, y)) keys1 @-> returns prog)
      model system;
    command "node2"
      (cache ^-> keys (fun (k, x) -> K2 (k, x)) keys2 @-> returns prog)
      model system;
    command "node3"
      (cache ^-> keys (fun (k, c, x) -> K3 (k, c, x)) keys3 @-> returns prog)
      model system;
  ]

(* Literal programs *)

(* What prim.mli states [literal2 ~swapped k dt c] gives: [k] of the operand
   and [c]'s bits. *)
let literal_prog (type v s) ~swapped k (dt : (v, s) D.t) (c : v) : P.t =
  let op : P.node = if swapped then Op2 (k, 1, 0) else Op2 (k, 0, 1) in
  P.v ~ins:[| D.Any dt |]
    [| In 0; Const (D.Any dt, P.bits dt c); op |]
    ~outs:[| 2 |]

let program = Testable.structural ~pp:pp_prog
let literal ~swapped c = Prim.literal2 ~by:"t" ~swapped (Binary Add) D.Float32 c

(* Literals one slot holds in turn: each equal to the one before as a float,
   or of other bits than it. *)
let literals =
  [
    0.; -0.; 0.; Float.nan; Int32.float_of_bits 0x7fc00001l; Float.nan;
    Float.infinity; 1.; 1.;
  ]

let law_literal_slot (swapped, c) =
  let want = literal_prog ~swapped (Binary Add) D.Float32 c in
  equal program want (literal ~swapped c)

let slot = abstract "s"

let literal_commands =
  let a = Array.of_list literals in
  let value =
    Gen.with_pp
      (fun ppf c -> Format.fprintf ppf "%h" c)
      (Gen.map (Array.get a) (Gen.int_range 0 (Array.length a - 1)))
  in
  let model () (swapped, c) =
    literal_prog ~swapped (Binary Add) D.Float32 c
  in
  let system () (swapped, c) = literal ~swapped c in
  [
    command "start" (Gen.unit @-> makes slot) Fun.id Fun.id;
    command "literal2"
      (slot ^-> Gen.pair Gen.bool value @-> returns program)
      model system;
  ]

let literal_programs =
  group "literal programs"
    [
      test "each literal's program holds its bits, after any other"
        (fun () ->
          List.iter
            (fun c ->
              law_literal_slot (false, c);
              law_literal_slot (true, c))
            literals);
      test "a repeated literal gives the last call of its slot's program"
        (fun () ->
          let p = literal ~swapped:false 1. in
          let q = literal ~swapped:true 1. in
          equal ~msg:"repeated" bool true (literal ~swapped:false 1. == p);
          equal ~msg:"repeated, swapped" bool true
            (literal ~swapped:true 1. == q);
          equal ~msg:"one side's slot" bool false (p == q);
          ignore (literal ~swapped:false 2.);
          equal ~msg:"after another literal" bool false
            (literal ~swapped:false 1. == p));
      test "a literal outside its dtype raises naming the caller" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"t: 300")
            (fun () ->
              Prim.literal2 ~by:"t" ~swapped:false (Binary Add) D.Int8 300));
      test "a kind refusing the dtype raises naming the caller" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"t: ")
            (fun () ->
              Prim.literal2 ~by:"t" ~swapped:false (Binary Threefry)
                D.Float32 1.));
      stateful ~domains:2 ~count:20 ~steps:8
        "literals on two domains give each call's bits" literal_commands;
    ]

let one_node_programs =
  group "one-node programs"
    [
      cases "node1 is of_node's program where its kind takes its dtypes"
        ~name:name1 keys1 test_node1;
      cases "node2 is of_node's program where its kind takes its dtype"
        ~name:name2 keys2 test_node2;
      cases "node3 is of_node's program where its kind takes its dtypes"
        ~name:name3 keys3 test_node3;
      test "a node found allocates nothing" test_node_found_allocates_nothing;
      stateful ~domains:2 ~count:20 ~steps:8
        "lookups on two domains give the model's programs" cache_commands;
    ]

(* A reduction of any result, and what it computes, the output it reads and
   its first result's dtype, as its constructor's fields state them. *)
type reduced = Reduced : ('d, 'a) Value.reduction -> reduced

let reduced =
  [
    (Reduced (Monoid (Max, 1, D.Int32)), (Nx_kernel.Monoid Max, 1, "int32"));
    (Reduced (Logsumexp (0, D.Float32)), (Logsumexp, 0, "float32"));
    (Reduced (Moments (2, D.Float64)), (Moments, 2, "float64"));
    (Reduced (Arg (Min, 3, D.Int8)), (Arg Min, 3, "int8"));
  ]

let kernel_reduction =
  Testable.make
    ~pp:(fun ppf r -> Format.pp_print_string ppf (Prim.reduction_name r))
    ~equal:( = )

let operations =
  group "operations"
    [
      cases "each operation names itself and lists its operands"
        ~name:(fun (n, _, _) -> n)
        (ops ())
        (fun (n, Op op, count) ->
          equal string n (Prim.name op);
          let (Operands xs) = Prim.operands op in
          equal int count (List.length xs));
      cases "a reduction is what it computes, its output and its dtype"
        ~name:(fun (_, (r, _, _)) -> Prim.reduction_name r)
        reduced
        (fun (Reduced r, want) ->
          let k, o, D.Any dt = Prim.reduction r in
          let got = (k, o, D.name dt) in
          equal (triple kernel_reduction int string) want got);
      test "a node is named by its innermost constructor" (fun () ->
          equal ~msg:"a unary" string "Exp"
            (Prim.kind (Op1 (Unary Exp, D.Any D.Float32, 0)));
          equal ~msg:"a comparison" string "Less_equal"
            (Prim.kind (Op2 (Compare Less_equal, 0, 1)));
          equal ~msg:"a ternary" string "Where"
            (Prim.kind (Op3 (Where, 0, 1, 2)));
          equal ~msg:"an input" string "In" (Prim.kind (In 0)));
      test "a coordinate is Iota and a constant Fill" (fun () ->
          equal ~msg:"a coordinate" string "Iota" (Prim.kind (Coord 0));
          let zero = String.make 4 '\000' in
          equal ~msg:"a constant" string "Fill"
            (Prim.kind (Const (D.Any D.Float32, zero))));
      cases "a map over operands visits each and keeps the rule"
        ~name:(fun (n, _, _) -> n)
        (ops ())
        (fun (_, Op op, count) ->
          let seen = ref 0 in
          let op' =
            Prim.map
              (fun x ->
                incr seen;
                x)
              op
          in
          equal int count !seen;
          equal int (List.length (forms (Op op))) (List.length (forms (Op op'))));
      test "arrays are each result's, in order" (fun () ->
          let x = f32 [| 2 |] in
          let op =
            Value.Map
              {
                layout = L.contiguous [| 2 |];
                prog = add_mul;
                outs = two;
                loads = [| Plain x; Plain x |];
                at = None;
              }
          in
          let r =
            Prim.results ~by:"Nx.f" (fun k f -> recording (ref []) k f) op
          in
          let u, (v, ()) = r in
          let arrays = Prim.result_arrays op r in
          equal int 2 (Array.length arrays);
          let same (Nx_array.Any a) y = A.buffer a == A.buffer (array_of y) in
          equal bool true (same arrays.(0).(0) u && same arrays.(1).(0) v));
    ]

let () =
  exit
    (run "nx prim"
       [
         maps;
         movements;
         bitcasts;
         others;
         contractions;
         partials;
         operations;
         one_node_programs;
         literal_programs;
       ])
