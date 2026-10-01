(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Assembly, indexed access and windows, lowered: each is checked against nx's
   eager result. The operations that move elements give eager's bits, [-0.] and
   NaN included. Scatter's additions and fold's overlaps are sums, checked
   exactly over values whose every sum is exact, so that the association a graph
   picks cannot show; their signed zeros are checked apart. *)

open Windtrap
open Nx_test
open Traces

(* [traced f] is the value of [f ()] traced, its operands captured. *)
let traced f =
  let s, y = trace f in
  value s y

let agrees f = exact (f ()) (traced f)
let f32 values = Nx.create Nx.float32 [| Array.length values |] values
let i32 values = Nx.create Nx.int32 [| Array.length values |] values

(* Drawn operations

   An operation is drawn at a dtype from values of that dtype, and checked at
   every float dtype, every integer dtype and booleans. *)

type 'r case = string * (unit -> 'r)

type op = {
  draw : 'a 'b. ('a, 'b) Nx.dtype -> 'a Gen.t -> ('a, 'b) Nx.t case Gen.t;
}

(* [case describe apply operands] draws [operands], shown by [describe], and
   applies [apply] to them. *)
let case describe apply operands =
  Gen.with_pp
    (fun ppf (shown, _) -> Format.pp_print_string ppf shown)
    (Gen.map
       (fun v -> (Format.asprintf "%a" describe v, fun () -> apply v))
       operands)

type float_dtype = F : string * (float, 'b) Nx.dtype -> float_dtype

let float_dtypes =
  [
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
  ]

(* [rows name ~floats op] checks [op] at each dtype, its floats drawn from
   [floats] and its integers from the values that break arithmetic. *)
let rows ?count name ~floats op =
  let check (_, f) = agrees f in
  group name
    (List.map
       (fun (F (n, dt)) -> prop ?count n (op.draw dt floats) check)
       float_dtypes
    @ List.map
        (fun (Int_dtype { name; dtype; bits; signed; of_i64; _ }) ->
          prop ?count name
            (op.draw dtype (Gen.map of_i64 (int_value ~bits ~signed)))
            check)
        int_dtypes
    @ [ prop ?count "bool" (op.draw Nx.bool Gen.bool) check ])

(* Floats a move must keep: signed zeros, NaN and infinities among any. *)
let element =
  Gen.frequency
    [
      (3, Gen.any_float); (1, Gen.of_list ~pp:Format.pp_print_float [ -0.; 0. ]);
    ]

(* Small integers and halves, whose sums are exact in every float dtype. *)
let summand = Gen.map (fun n -> float_of_int n /. 2.) (Gen.int_range (-8) 8)

let tensor dtype shape value =
  Gen.map (Nx.create dtype shape)
    (Gen.array ~size:(Gen.constant (Array.fold_left ( * ) 1 shape)) value)

let sizes ~rank ~least =
  Gen.bind (Gen.int_range 1 rank) (fun r ->
      Gen.array ~size:(Gen.constant r) (Gen.int_range least 4))

let with_axis shape axis n =
  Array.mapi (fun d s -> if d = axis then n else s) shape

let positions shape ~lo ~hi =
  tensor Nx.int32 shape (Gen.map Int32.of_int (Gen.int_range lo hi))

(* [each gens] draws from each of [gens], in order. *)
let each gens =
  List.fold_right
    (fun g rest ->
      let open Gen in
      let+ x = g and+ xs = rest in
      x :: xs)
    gens (Gen.constant [])

let pp_tensors ppf xs =
  Format.pp_print_list ~pp_sep:Format.pp_print_space Nx.pp ppf xs

(* Assembly *)

let pad =
  {
    draw =
      (fun dt value ->
        let open Gen in
        let operands =
          let* shape = array ~size:(int_range 0 3) (int_range 0 3) in
          let+ x = tensor dt shape value
          and+ widths =
            array
              ~size:(constant (Array.length shape))
              (pair (int_range 0 2) (int_range 0 2))
          and+ fill = value in
          (widths, fill, x)
        in
        case
          (fun ppf (_, fill, x) -> pp_tensors ppf [ Nx.scalar dt fill; x ])
          (fun (widths, fill, x) -> Nx.pad widths fill x)
          operands);
  }

let cat =
  {
    draw =
      (fun dt value ->
        let open Gen in
        let operands =
          let* shape = sizes ~rank:3 ~least:1 in
          let* axis = int_range 0 (Array.length shape - 1) in
          let+ pieces =
            list ~size:(int_range 1 4)
              (bind (int_range 0 3) (fun n ->
                   tensor dt (with_axis shape axis n) value))
          in
          (axis, pieces)
        in
        case
          (fun ppf (_, pieces) -> pp_tensors ppf pieces)
          (fun (axis, pieces) -> Nx.concatenate ~axis pieces)
          operands);
  }

let assembly =
  group "assembly"
    [
      rows "pad" ~floats:element pad;
      test "a pad of -0. keeps its sign" (fun () ->
          agrees (fun () -> Nx.pad [| (1, 2) |] (-0.) (f32 [| 1.; Float.nan |])));
      test "a pad of 0. is +0." (fun () ->
          agrees (fun () -> Nx.pad [| (2, 1) |] 0. (f32 [| -0.; 2. |])));
      rows "cat" ~floats:element cat;
      test "pieces of different lengths keep -0." (fun () ->
          agrees (fun () ->
              Nx.concatenate ~axis:0
                [
                  f32 [| -0. |];
                  f32 [| -0.; Float.nan |];
                  f32 [| -0.; 0.; -0. |];
                ]));
      test "empty pieces are skipped" (fun () ->
          agrees (fun () ->
              Nx.concatenate ~axis:1
                [
                  Nx.zeros Nx.float32 [| 2; 0 |];
                  Nx.full Nx.float32 [| 2; 1 |] (-0.);
                  Nx.zeros Nx.float32 [| 2; 0 |];
                ]));
      test "pieces all empty are empty" (fun () ->
          agrees (fun () ->
              Nx.concatenate ~axis:0
                [ Nx.zeros Nx.float32 [| 0 |]; Nx.zeros Nx.float32 [| 0 |] ]));
    ]

(* Indexed access *)

let gather =
  {
    draw =
      (fun dt value ->
        let open Gen in
        let operands =
          let* shape = sizes ~rank:3 ~least:1 in
          let* axis = int_range 0 (Array.length shape - 1) in
          let* m = int_range 0 4 in
          let n = shape.(axis) in
          let+ x = tensor dt shape value
          and+ indices =
            positions (with_axis shape axis m) ~lo:(-2) ~hi:(n + 1)
          in
          (axis, indices, x)
        in
        case
          (fun ppf (_, indices, x) ->
            pp_tensors ppf [ x ];
            Nx.pp ppf indices)
          (fun (axis, indices, x) -> Nx.take_along_axis ~axis ~indices x)
          operands);
  }

(* Updates at positions along an axis, some repeated and some outside it, or
   each position at most once when [unique]. *)
let scatter ~mode ~unique =
  {
    draw =
      (fun dt value ->
        let open Gen in
        let operands =
          let* shape = sizes ~rank:3 ~least:1 in
          let* axis = int_range 0 (Array.length shape - 1) in
          let n = shape.(axis) in
          let* m = int_range 0 (if unique then n else 4) in
          let at = with_axis shape axis m in
          let* indices =
            if unique then
              map
                (fun order ->
                  let first = Array.sub (Array.of_list order) 0 m in
                  let along =
                    Array.mapi (fun d s -> if d = axis then m else 1) shape
                  in
                  Nx.broadcast_to at
                    (Nx.reshape along (Nx.create Nx.int32 [| m |] first)))
                (permutation (List.init n Int32.of_int))
            else positions at ~lo:(-1) ~hi:n
          in
          let+ x = tensor dt shape value and+ values = tensor dt at value in
          (axis, indices, values, x)
        in
        case
          (fun ppf (_, indices, values, x) ->
            pp_tensors ppf [ x; values ];
            Nx.pp ppf indices)
          (fun (axis, indices, values, x) ->
            Nx.scatter ~mode ~unique_indices:unique ~axis ~indices ~values x)
          operands);
  }

(* A window of [x] at a start drawn along each axis, some past either end, which
   [Nx.set] clamps: the start is a value of the program. *)
let update =
  {
    draw =
      (fun dt value ->
        let open Gen in
        let operands =
          let* shape = sizes ~rank:3 ~least:1 in
          let* window =
            map Array.of_list
              (each (List.map (fun s -> int_range 1 s) (Array.to_list shape)))
          in
          let* starts =
            each (List.map (fun s -> int_range (-1) s) (Array.to_list shape))
          in
          let+ x = tensor dt shape value and+ v = tensor dt window value in
          (starts, window, v, x)
        in
        case
          (fun ppf (_, _, v, x) -> pp_tensors ppf [ x; v ])
          (fun (starts, window, v, x) ->
            Nx.set
              (List.mapi
                 (fun d start ->
                   Nx.D (Nx.scalar Nx.int32 (Int32.of_int start), window.(d)))
                 starts)
              v x)
          operands);
  }

let indexed =
  group "indexed access"
    [
      rows "gather" ~floats:element gather;
      test "a gathered -0. keeps its sign" (fun () ->
          agrees (fun () ->
              Nx.take ~indices:(i32 [| 1l; 0l; 1l |]) (f32 [| Float.nan; -0. |])));
      test "an index out of range reads +0." (fun () ->
          agrees (fun () ->
              Nx.take ~indices:(i32 [| -1l; 2l; 7l |]) (f32 [| -0.; -1. |])));
      test "a gather from an empty axis reads zeros" (fun () ->
          agrees (fun () ->
              Nx.take_along_axis ~axis:1
                ~indices:(Nx.zeros Nx.int32 [| 2; 3 |])
                (Nx.zeros Nx.float32 [| 2; 0 |])));
      rows "scatter set" ~floats:element (scatter ~mode:`Set ~unique:false);
      rows "scatter set unique" ~floats:element
        (scatter ~mode:`Set ~unique:true);
      rows "scatter add" ~floats:summand (scatter ~mode:`Add ~unique:false);
      test "the last of duplicate positions is set" (fun () ->
          agrees (fun () ->
              Nx.scatter ~axis:0
                ~indices:(i32 [| 1l; 1l; 0l; 1l |])
                ~values:(f32 [| 1.; 2.; -0.; -0. |])
                (f32 [| 5.; 6.; 7. |])));
      test "a position no update reaches keeps -0." (fun () ->
          agrees (fun () ->
              Nx.scatter ~mode:`Add ~axis:0
                ~indices:(i32 [| 0l; 3l |])
                ~values:(f32 [| 1.; 1. |])
                (f32 [| 2.; -0.; -0. |])));
      test "an update of -0. added to -0. is +0." (fun () ->
          agrees (fun () ->
              Nx.scatter ~mode:`Add ~axis:0 ~indices:(i32 [| 0l |])
                ~values:(f32 [| -0. |]) (f32 [| -0. |])));
      rows "update" ~floats:element update;
      test "a window at constant starts" (fun () ->
          let x = Nx.create Nx.float32 [| 3; 4 |] (Array.make 12 (-0.)) in
          let v =
            Nx.create Nx.float32 [| 2; 2 |] [| 1.; -0.; Float.nan; 4. |]
          in
          agrees (fun () -> Nx.set [ R (1, 3); R (2, 4) ] v x));
    ]

(* Windows *)

(* Windows over the last one or two axes, padded, strided and dilated; each
   spatial axis is long enough for one window. *)
let geometry =
  let open Gen in
  let* lead = array ~size:(int_range 0 2) (int_range 1 2) in
  let+ axes =
    list ~size:(int_range 1 2)
      (let+ kernel = int_range 1 3
       and+ stride = int_range 1 3
       and+ dilation = int_range 1 2
       and+ before = int_range 0 2
       and+ after = int_range 0 2
       and+ extra = int_range 0 3 in
       let reach = (dilation * (kernel - 1)) + 1 in
       let size = Int.max 1 (reach - before - after + extra) in
       (kernel, stride, dilation, (before, after), size))
  in
  let per_axis f = Array.of_list (List.map f axes) in
  ( lead,
    per_axis (fun (k, _, _, _, _) -> k),
    per_axis (fun (_, s, _, _, _) -> s),
    per_axis (fun (_, _, d, _, _) -> d),
    per_axis (fun (_, _, _, p, _) -> p),
    per_axis (fun (_, _, _, _, n) -> n) )

let unfold =
  {
    draw =
      (fun dt value ->
        let open Gen in
        let operands =
          let* ((lead, _, _, _, _, spatial) as g) = geometry in
          let+ x = tensor dt (Array.append lead spatial) value in
          (g, x)
        in
        case
          (fun ppf (_, x) -> Nx.pp ppf x)
          (fun ((_, kernel_size, stride, dilation, padding, _), x) ->
            Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x)
          operands);
  }

let fold =
  {
    draw =
      (fun dt value ->
        let open Gen in
        let operands =
          let* ((lead, kernel_size, stride, dilation, padding, output_size) as g)
              =
            geometry
          in
          let windows =
            Array.mapi
              (fun a n ->
                let before, after = padding.(a) in
                (n + before + after - (dilation.(a) * (kernel_size.(a) - 1)) - 1)
                / stride.(a)
                + 1)
              output_size
          in
          let product = Array.fold_left ( * ) 1 in
          let+ x =
            tensor dt
              (Array.append lead [| product kernel_size; product windows |])
              value
          in
          (g, x)
        in
        case
          (fun ppf (_, x) -> Nx.pp ppf x)
          (fun ((_, kernel_size, stride, dilation, padding, output_size), x) ->
            Nx.combine_patches ~output_size ~kernel_size ~stride ~dilation
              ~padding x)
          operands);
  }

let windows =
  group "windows"
    [
      rows "unfold" ~floats:element unfold;
      rows "fold" ~floats:summand fold;
      test "overlapping windows of -0. fold to +0." (fun () ->
          agrees (fun () ->
              Nx.combine_patches ~output_size:[| 3 |] ~kernel_size:[| 2 |]
                ~stride:[| 1 |] ~dilation:[| 1 |]
                ~padding:[| (0, 0) |]
                (Nx.full Nx.float32 [| 2; 2 |] (-0.))));
      test "a fold whose windows along an axis read only padding is zeros"
        (fun () ->
          agrees (fun () ->
              Nx.combine_patches ~output_size:[| 3; 1 |] ~kernel_size:[| 2; 2 |]
                ~stride:[| 1; 1 |] ~dilation:[| 2; 2 |]
                ~padding:[| (0, 0); (1, 1) |]
                (Nx.create Nx.float32 [| 4; 1 |] [| 1.; 2.; 3.; Float.nan |])));
      test "a fold with no window is zeros" (fun () ->
          exact
            (Nx.zeros Nx.float32 [| 1 |])
            (traced (fun () ->
                 Nx.combine_patches ~output_size:[| 1 |] ~kernel_size:[| 3 |]
                   ~stride:[| 1 |] ~dilation:[| 1 |]
                   ~padding:[| (0, 0) |]
                   (Nx.zeros Nx.float32 [| 3; 0 |]))));
      test "a single window of -0. folds to +0." (fun () ->
          agrees (fun () ->
              Nx.combine_patches ~output_size:[| 2 |] ~kernel_size:[| 1 |]
                ~stride:[| 1 |] ~dilation:[| 1 |]
                ~padding:[| (0, 0) |]
                (Nx.full Nx.float32 [| 1; 2 |] (-0.))));
    ]

(* Graph parity: the kernels tinygrad schedules for the same program. *)

let parity =
  let x shape = Nx.zeros Nx.float32 shape in
  let case file f =
    Golden.graph (file ^ ".golden") (fun () ->
        let args = f () in
        Programs.kernels (snd (trace (fun () -> args ()))))
  in
  group "graph parity"
    [
      case "pad_zero" (fun () ->
          let a = x [| 4; 4 |] in
          fun () -> Nx.pad [| (1, 2); (0, 3) |] 0. a);
      case "pad_value" (fun () ->
          let a = x [| 4; 4 |] in
          fun () -> Nx.pad [| (1, 2); (0, 3) |] 1.5 a);
      case "cat_equal" (fun () ->
          let a = x [| 2; 4 |] and b = x [| 2; 4 |] and c = x [| 2; 4 |] in
          fun () -> Nx.concatenate ~axis:0 [ a; b; c ]);
      case "unfold" (fun () ->
          let a = x [| 2; 5; 6 |] in
          fun () ->
            Nx.extract_patches ~kernel_size:[| 2; 3 |] ~stride:[| 1; 2 |]
              ~dilation:[| 2; 1 |]
              ~padding:[| (1, 0); (1, 1) |]
              a);
    ]

let () = exit @@ run "lower_index" [ assembly; indexed; windows; parity ]
