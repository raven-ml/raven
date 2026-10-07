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
let i64 values = Nx.create Nx.int64 [| Array.length values |] values

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

let far = 1 lsl 32

(* Positions in [lo, hi], some moved by 2^32, which an index narrowed by
   truncation would bring back to where it was. *)
let positions shape ~lo ~hi =
  let open Gen in
  tensor Nx.int64 shape
    (map Int64.of_int
       (frequency
          [
            (4, int_range lo hi);
            ( 1,
              let+ i = int_range lo hi and+ k = of_list [ -1; 1 ] in
              i + (k * far) );
          ]))

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
                    Array.mapi (fun d _ -> if d = axis then m else 1) shape
                  in
                  Nx.broadcast_to at
                    (Nx.reshape along (Nx.create Nx.int64 [| m |] first)))
                (permutation (List.init n Int64.of_int))
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
            each
              (List.map
                 (fun s ->
                   frequency
                     [ (4, int_range (-1) s); (1, of_list [ -far; far + 1 ]) ])
                 (Array.to_list shape))
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
                   Nx.D (Nx.scalar Nx.int64 (Int64.of_int start), window.(d)))
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
              Nx.take ~indices:(i64 [| 1L; 0L; 1L |]) (f32 [| Float.nan; -0. |])));
      test "an index out of range reads +0." (fun () ->
          agrees (fun () ->
              Nx.take ~indices:(i64 [| -1L; 2L; 7L |]) (f32 [| -0.; -1. |])));
      test "an index of 2^32 + 1 reads zero compiled, on an axis of 4"
        (fun () ->
          let x = f32 [| 1.; 2.; 3.; 4. |] in
          exact
            (f32 [| 0.; 2. |])
            (traced (fun () ->
                 Nx.take ~indices:(i64 [| 0x1_0000_0001L; 1L |]) x)));
      test "a gather from an empty axis reads zeros" (fun () ->
          agrees (fun () ->
              Nx.take_along_axis ~axis:1
                ~indices:(Nx.zeros Nx.int64 [| 2; 3 |])
                (Nx.zeros Nx.float32 [| 2; 0 |])));
      rows "scatter set" ~floats:element (scatter ~mode:`Set ~unique:false);
      rows "scatter set unique" ~floats:element
        (scatter ~mode:`Set ~unique:true);
      rows "scatter add" ~floats:summand (scatter ~mode:`Add ~unique:false);
      rows "scatter max" ~floats:element (scatter ~mode:`Max ~unique:false);
      rows "scatter min" ~floats:element (scatter ~mode:`Min ~unique:false);
      test "the last of duplicate positions is set" (fun () ->
          agrees (fun () ->
              Nx.scatter ~axis:0
                ~indices:(i64 [| 1L; 1L; 0L; 1L |])
                ~values:(f32 [| 1.; 2.; -0.; -0. |])
                (f32 [| 5.; 6.; 7. |])));
      test "a position no update reaches keeps -0." (fun () ->
          agrees (fun () ->
              Nx.scatter ~mode:`Add ~axis:0
                ~indices:(i64 [| 0L; 3L |])
                ~values:(f32 [| 1.; 1. |])
                (f32 [| 2.; -0.; -0. |])));
      test "an update of -0. added to -0. is +0." (fun () ->
          agrees (fun () ->
              Nx.scatter ~mode:`Add ~axis:0 ~indices:(i64 [| 0L |])
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
      test "an unfold whose windows along an axis read only padding is zeros"
        (fun () ->
          agrees (fun () ->
              Nx.extract_patches ~kernel_size:[| 2; 1 |] ~stride:[| 1; 2 |]
                ~dilation:[| 1; 1 |]
                ~padding:[| (0, 0); (1, 0) |]
                (Nx.create Nx.float32 [| 1; 3; 1 |] [| 1.; Float.nan; -0. |])));
      test "a fold with no window is zeros" (fun () ->
          agrees (fun () ->
              Nx.combine_patches ~output_size:[| 1 |] ~kernel_size:[| 3 |]
                ~stride:[| 1 |] ~dilation:[| 1 |]
                ~padding:[| (0, 0) |]
                (Nx.zeros Nx.float32 [| 3; 0 |])));
      test "an unfold with no window is empty" (fun () ->
          agrees (fun () ->
              Nx.extract_patches ~kernel_size:[| 3 |] ~stride:[| 1 |]
                ~dilation:[| 1 |]
                ~padding:[| (0, 0) |]
                (Nx.ones Nx.float32 [| 2; 1 |])));
      test "a single window of -0. folds to +0." (fun () ->
          agrees (fun () ->
              Nx.combine_patches ~output_size:[| 2 |] ~kernel_size:[| 1 |]
                ~stride:[| 1 |] ~dilation:[| 1 |]
                ~padding:[| (0, 0) |]
                (Nx.full Nx.float32 [| 1; 2 |] (-0.))));
    ]

(* Graph parity: the kernels tinygrad schedules for the same program. *)

let parity =
  (* Captures with storage of their own: a constant would fold. *)
  let x shape = Nx.copy (Nx.zeros Nx.float32 shape) in
  let case file f =
    Golden.graph
      ("golden/lower_index/" ^ file ^ ".golden")
      (fun () ->
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

(* Quantised products

   A product with a quantised weight gathers the codes of the experts its ids
   select, decodes them and multiplies. The decoding is integer operations on
   the code bytes, so no kernel stores the decoded stack. One token's few
   experts each take a block of their own: no kernel sorts them, and no kernel
   stores an index array wider than one element. GGUF's MXFP4 codes and scales
   are views of its blocks, which a product reads in place, each byte once per
   output. *)

let quantised =
  let experts = 8 and n = 16 and k = 64 in
  let codes =
    Nx.init Nx.uint8 [| experts; n; k / 2 |] (fun i -> (i.(1) * 37) + i.(2))
  in
  let scales = Nx.full Nx.uint8 [| experts; n; k / 32 |] 127 in
  let w = Nx_quant.mxfp4 ~scales codes in
  let routed ids x =
    Nx.map_segments ~segments:experts ids
      (fun owners rows ->
        let g = Nx.dim 0 rows and c = Nx.dim 1 rows in
        let m = Nx.dim 2 rows in
        Nx.reshape [| g; c; m; n |]
          (Nx_quant.apply
             (Nx_quant.take ~axis:0 ~indices:owners w)
             (Nx.reshape [| g; c * m; k |] rows)))
      x
  in
  (* The elements of each buffer a kernel of [y]'s program stores values of [dt]
     into. *)
  let stores dt y =
    let size u =
      List.find_map
        (fun v ->
          match (Tolk.Ops.op v, Tolk.Ops.arg v) with
          | Param, Tolk.Ops.Param p -> p.size
          | _ -> None)
        (Tolk.Ops.toposort ~calls:Enter (Tolk.Ops.nth u 0))
    in
    List.filter_map
      (fun u ->
        if
          Tolk.Op.equal (Tolk.Ops.op u) Store
          && Tolk.Dtype.equal (Tolk.Ops.dtype (Tolk.Ops.nth u 1)) dt
        then size u
        else None)
      (Tolk.Ops.toposort ~calls:Enter (Programs.kernels y))
  in
  let traced ids x =
    let s = scope () in
    within s (fun () -> routed (argument s ids) (argument s x))
  in
  (* The bytes of [y]'s uint8 loads for each output element: each load's bytes
     times the trips of the reductions it runs inside. *)
  let loaded y =
    let bytes k =
      List.fold_left
        (fun acc u ->
          if
            Tolk.Op.equal (Tolk.Ops.op u) Load
            && Tolk.Dtype.equal (Tolk.Ops.dtype u) Uint8
          then
            let trips r t =
              match Tolk.Ops.axis_type r with
              | Reduce -> t * (Tolk.Dtype.Value.to_int (Tolk.Ops.vmax r) + 1)
              | _ -> t
            in
            acc + Tolk.Ops.Nodes.fold trips (Tolk.Ops.ranges u) 1
          else acc)
        0
        (Tolk.Ops.toposort ~calls:Enter
           (Tolk.Codegen.full_rewrite_to_sink k (host Nx_device.host)))
    in
    List.fold_left
      (fun acc k -> acc + bytes k)
      0
      (Tolk.Ops.src (Programs.kernels y))
  in
  let product w =
    let s = Nx_quant.shape w in
    let x = Nx.ones Nx.float32 [| 1; s.(Array.length s - 1) |] in
    let s = scope () in
    within s (fun () ->
        Nx_quant.apply
          (Nx.Ptree.map Nx_quant.ptree (fun _ t -> argument s t) w)
          (argument s x))
  in
  let bytes shape =
    Nx.init Nx.uint8 shape (fun i -> ((i.(0) * 37) + i.(1)) land 255)
  in
  group "quantised products"
    [
      test "one token's routed product stores no int64 array of two elements"
        (fun () ->
          let ids = Nx.create Nx.int64 [| 1; 2 |] [| 3L; 5L |] in
          let x = Nx.ones Nx.float32 [| 1; 1; 3; k |] in
          equal (list int) []
            (List.filter (fun n -> n > 1) (stores Int64 (traced ids x))));
      test "a prompt's routed product stores no float array of the stack's size"
        (fun () ->
          let ids =
            Nx.create Nx.int64 [| 32; 2 |]
              (Array.init 64 (fun i -> Int64.of_int (i * 3 mod experts)))
          in
          let x = Nx.ones Nx.float32 [| 32; 1; 1; k |] in
          let stack = experts * n * k in
          equal (list int) []
            (List.filter (fun n -> n >= stack) (stores Float32 (traced ids x))));
      test "a product unpacks its codes at the width it decodes them in"
        (fun () ->
          (* A nibble masked and shifted as a byte, then widened, is slower on
             the host than the same operations on the widened byte. *)
          let w =
            Nx_quant.mxfp4
              ~scales:(bytes [| n; k / 32 |])
              (bytes [| n; k / 2 |])
          in
          let narrow k =
            List.length
              (List.filter
                 (fun u ->
                   (Tolk.Op.equal (Tolk.Ops.op u) And
                   || Tolk.Op.equal (Tolk.Ops.op u) Shr)
                   && Tolk.Dtype.equal (Tolk.Ops.dtype u) Uint8)
                 (Tolk.Ops.toposort ~calls:Enter
                    (Tolk.Codegen.full_rewrite_to_sink k (host Nx_device.host))))
          in
          equal (list int) [ 0 ]
            (List.map narrow (Tolk.Ops.src (Programs.kernels (product w)))));
      test "a product decodes a checkpoint's codes 16 at once on the host"
        (fun () ->
          (* Each byte's two codes are lanes of one vector, 8 bytes of them, a
             vector of 64 bytes at uint32: decoded one at a time, Clang branches
             on each code's exponent on x86. A scale is decoded once for its 16
             codes, alone. *)
          let w =
            Nx_quant.mxfp4
              ~scales:(bytes [| n; k / 32 |])
              (bytes [| n; k / 2 |])
          in
          let selects k =
            List.filter_map
              (fun u ->
                if
                  Tolk.Op.equal (Tolk.Ops.op u) Where
                  && Tolk.Dtype.equal (Tolk.Ops.dtype u) Uint32
                then Some (Tolk.Shape.shape u)
                else None)
              (Tolk.Ops.toposort ~calls:Enter
                 (Tolk.Codegen.full_rewrite_to_sink k (host Nx_device.host)))
          in
          let lanes = function
            | [ Tolk.Ops.Int n ] -> n
            | [] -> 1
            | _ -> invalid_arg "a select of several axes"
          in
          equal int 16
            (List.fold_left max 1
               (List.concat_map
                  (fun k -> List.map lanes (selects k))
                  (Tolk.Ops.src (Programs.kernels (product w))))));
      test "a product over GGUF's Q6_K blocks converts each scale once"
        (fun () ->
          (* A scale byte serves 16 values: converted in each of their lanes,
             the conversion and the products by the block's scale run 16
             times. *)
          let w = Nx_quant.q6_k (bytes [| n; 210 |]) in
          let sources u =
            let x = Tolk.Ops.nth u 0 in
            let x =
              if Tolk.Op.equal (Tolk.Ops.op x) Bitcast then Tolk.Ops.nth x 0
              else x
            in
            if Tolk.Op.equal (Tolk.Ops.op x) Stack then Tolk.Ops.src x
            else [ x ]
          in
          let converted k =
            List.concat_map
              (fun u ->
                if
                  Tolk.Op.equal (Tolk.Ops.op u) Cast
                  && Tolk.Dtype.equal (Tolk.Ops.dtype u) Float32
                  && Tolk.Dtype.equal (Tolk.Ops.dtype (Tolk.Ops.nth u 0)) Int8
                then sources u
                else [])
              (Tolk.Ops.toposort ~calls:Enter
                 (Tolk.Codegen.full_rewrite_to_sink k (host Nx_device.host)))
          in
          let lanes =
            List.concat_map converted
              (Tolk.Ops.src (Programs.kernels (product w)))
          in
          let distinct =
            List.fold_left
              (fun seen u -> if List.memq u seen then seen else u :: seen)
              [] lanes
          in
          equal int (List.length distinct) (List.length lanes));
      test "a product over GGUF's MXFP4 blocks loads each byte once" (fun () ->
          let w = Nx_quant.mxfp4_blocks (bytes [| n; k / 32 * 17 |]) in
          equal int (k / 32 * 17) (loaded (product w)));
    ]

let () =
  exit @@ run "lower_index" [ assembly; indexed; windows; parity; quantised ]
