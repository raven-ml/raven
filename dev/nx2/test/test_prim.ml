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
    { at = Devices.one s2 d; a = A.to_device (m d) a; dead = Prim.live }

let f32 ?(d = 0) shape = value d (A.of_array D.Float32 shape (floats shape))

(* A maker that records the forms it is given, and one that must not run. *)
let recording () =
  let forms = ref [] in
  let make : type v s d. int -> (v, s, d) Value.form -> (v, s, d) Value.t =
   fun k f ->
    forms :=
      (k, D.Any f.dtype, f.layout, Option.map Devices.rebrand f.placement)
      :: !forms;
    let a = A.create Rig.host f.dtype (L.shape f.layout) in
    Value.Array
      { at = Devices.rebrand (Devices.one Devices.host 0); a; dead = Prim.live }
  in
  ({ Prim.make }, fun () -> List.rev !forms)

let refusing =
  { Prim.make = (fun _ _ -> fail "the maker ran before the rule refused") }

let forms (Op op) =
  let mk, forms = recording () in
  ignore (Prim.results ~by:"Nx.f" mk op);
  forms ()

let one_form op =
  match forms op with
  | [ (_, dt, l, (p : b Devices.placement option)) ] -> (dt, l, p)
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

let split_x () : (float, D.float32_elt, b) Value.t =
  Value.Shards
    {
      at = Devices.split ~by:"t" ~axis:0 s2;
      arrays =
        [|
          A.to_device (m 0) (A.of_array D.Float32 [| 2 |] [| 0.; 1. |]);
          A.to_device (m 1) (A.of_array D.Float32 [| 2 |] [| 2.; 3. |]);
        |];
      dead = Prim.live;
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
                    }))
          in
          equal (list int) [ 0; 1 ] (List.map (fun (k, _, _, _) -> k) fs);
          List.iter
            (fun (_, _, l, p) ->
              equal layout (L.contiguous [| 2; 3 |]) l;
              equal bool true (Devices.equal (Option.get p) at0))
            fs);
      test "a map over split loads is split" (fun () ->
          let x = split_x () in
          List.iter
            (fun (_, _, _, p) ->
              equal bool true
                (Devices.equal (Option.get p)
                   (Devices.split ~by:"t" ~axis:0 s2)))
            (forms
               (Op
                  (Map
                     {
                       layout = L.contiguous [| 4 |];
                       prog = add_mul;
                       outs = two;
                       loads = [| Plain x; Plain x |];
                     }))));
      test "a creation is of every set" (fun () ->
          let one =
            P.v ~ins:[||]
              [| Const (D.Any D.Float32, P.bits D.Float32 1.) |]
              ~outs:[| 0 |]
          in
          let _, _, p =
            one_form
              (Op
                 (Map
                    {
                      layout = L.contiguous [| 3 |];
                      prog = one;
                      outs = [ D.Float32 ];
                      loads = [||];
                    }))
          in
          equal bool true (p = None));
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
                 }) );
          ( "a load of another dtype",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = two;
                   loads = [| Plain (f32 [| 2 |]); Plain i32 |];
                 }) );
          ( "loads of another shape",
            Op
              (Map
                 {
                   layout = L.contiguous [| 3 |];
                   prog = add_mul;
                   outs = two;
                   loads = [| Plain (f32 [| 2 |]); Plain (f32 [| 2 |]) |];
                 }) );
          ( "a result of another dtype",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = [ D.Float32; D.Float64 ];
                   loads = [| Plain (f32 [| 2 |]); Plain (f32 [| 2 |]) |];
                 }) );
          ( "one result for two outputs",
            Op
              (Map
                 {
                   layout = L.contiguous [| 2 |];
                   prog = add_mul;
                   outs = [ D.Float32 ];
                   loads = [| Plain (f32 [| 2 |]); Plain (f32 [| 2 |]) |];
                 }) );
          ( "a layout that is not C-contiguous",
            Op
              (Map
                 {
                   layout = L.v ~offset:0 ~strides:[| 2 |] [| 2 |];
                   prog = add_mul;
                   outs = two;
                   loads = [||];
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
                  })));
    ]

(* Movements and bitcasts, against nx.array's *)

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_move ppf = function
  | M.Reshape s -> Format.fprintf ppf "reshape %a" pp_ints s
  | Broadcast s -> Format.fprintf ppf "broadcast %a" pp_ints s
  | Permute p -> Format.fprintf ppf "permute %a" pp_ints p
  | Slice rs ->
      Format.fprintf ppf "slice %s"
        (String.concat " "
           (Array.to_list
              (Array.map
                 (fun (r : M.range) ->
                   Printf.sprintf "%d+%d/%d" r.start r.count r.step)
                 rs)))
  | Window ws ->
      Format.fprintf ppf "window %s"
        (String.concat " "
           (Array.to_list
              (Array.map
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
             (fun p -> M.Permute (Array.of_list p))
             (permutation (List.init rank Fun.id));
           constant (M.Reshape [| n |]);
           constant (M.Reshape [| n; 1 |]);
           constant (M.Broadcast (Array.append [| 3 |] s));
           constant
             (M.Broadcast (Array.map (fun d -> if d = 1 then 3 else d) s));
           map
             (fun rev ->
               M.Slice
                 (Array.map
                    (fun d ->
                      if rev && d > 1 then
                        { M.start = d - 1; count = d / 2; step = -2 }
                      else { M.start = 0; count = d; step = 1 })
                    s))
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
      | Shards _ | Donated _ | Deferred _ | Traced _ -> None)
    x

let moved_by mv x =
  match x with
  | Value.Array { at; a; _ } ->
      Value.Array { at; a = Option.get (A.move mv a); dead = Prim.live }
  | Shards _ | Donated _ | Deferred _ | Traced _ -> x

let movements =
  group "movements"
    [
      prop "a movement's form is nx.array's move of the layout, or C order"
        moved (fun (s0, flip, mv) ->
          let x = f32 s0 in
          let rank = Array.length s0 in
          let x =
            if flip then
              moved_by (M.Permute (Array.init rank (fun i -> rank - 1 - i))) x
            else x
          in
          let s = Prim.shape x in
          match M.shape mv s with
          | exception Invalid_argument _ ->
              cover "refused" true;
              refuses (Op (Move (mv, x)))
          | s' -> (
              let _, l, p = one_form (Op (Move (mv, x))) in
              equal bool true (Devices.equal (Option.get p) at0);
              match A.move mv (array_of x) with
              | Some a' ->
                  cover "a view" true;
                  equal layout (A.layout a') l
              | None ->
                  cover "a copy" true;
                  equal layout (L.contiguous s') l));
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
          equal bool true (Devices.equal (Option.get p) at0);
          match expected with
          | Some a' -> equal layout (A.layout a') l
          | None -> equal bool true (L.is_contiguous l)))

let u8 shape =
  value 0
    (A.of_array D.Uint8 shape
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
          equal bool true (Devices.equal (Option.get p) (Devices.one s2 1));
          equal layout (A.layout (array_of x)) l);
      test "a place split over two devices is C order of the whole" (fun () ->
          let split = Devices.split ~by:"t" ~axis:0 s2 in
          let _, l, p = one_form (Op (Place (split, f32 [| 4; 2 |]))) in
          equal bool true (Devices.equal (Option.get p) split);
          equal layout (L.contiguous [| 4; 2 |]) l);
      test "a place whose split does not divide the shape raises" (fun () ->
          refuses (Op (Place (Devices.split ~by:"t" ~axis:0 s2, f32 [| 3 |]))));
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
                          data = [ Any (f32 [| 2 |]) ];
                          fail = (fun _ _ -> Exit);
                        })))));
      test "a check refuses data of another shape" (fun () ->
          let ok : (bool, D.bool_elt, b) Value.t =
            value 0 (A.of_array D.Bool [| 2 |] [| true; true |])
          in
          refuses
            (Op
               (Check
                  { ok; data = [ Any (f32 [| 3 |]) ]; fail = (fun _ _ -> Exit) })));
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
           }),
      2 );
    ("Copy", Op (Copy x), 1);
    ("Move", Op (Move (M.Reshape [| 1; 2 |], x)), 1);
    ("Bitcast", Op (Bitcast (D.Int32, x)), 1);
    ("Place", Op (Place (Devices.one s2 1, x)), 1);
    ("Check", Op (Check { ok; data = [ Any x ]; fail = (fun _ _ -> Exit) }), 2);
  ]

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
      cases "a map over operands visits each and keeps the rule"
        ~name:(fun (n, _, _) -> n)
        (ops ())
        (fun (_, Op op, count) ->
          let seen = ref 0 in
          let op' =
            Prim.map
              {
                map =
                  (fun x ->
                    incr seen;
                    x);
              }
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
              }
          in
          let r = Prim.results ~by:"Nx.f" (fst (recording ())) op in
          let u, (v, ()) = r in
          let arrays = Prim.arrays op r in
          equal int 2 (Array.length arrays);
          let same (Nx_array.Any a) y = A.buffer a == A.buffer (array_of y) in
          equal bool true (same arrays.(0).(0) u && same arrays.(1).(0) v));
    ]

let () = exit (run "nx prim" [ maps; movements; bitcasts; others; operations ])
