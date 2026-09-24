(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantised weights: construction, the format's values, the product with and
   without ids, and the structure. *)

open Windtrap

let rng = Random.State.make [| 4 |]
let bytes shape f = Nx.init Nx.uint8 shape f
let random_bytes shape = bytes shape (fun _ -> Random.State.int rng 256)

(* Scale bytes of finite products, with NaN groups and subnormal ones. *)
let scale_bytes shape =
  bytes shape (fun _ ->
      match Random.State.int rng 20 with
      | 0 -> 255
      | 1 -> 0
      | _ -> 118 + Random.State.int rng 19)

let weight ?(scales = scale_bytes) shape =
  let r = Array.length shape in
  let part last = Array.append (Array.sub shape 0 (r - 1)) [| last |] in
  let k = shape.(r - 1) in
  Nx_quant.mxfp4 ~scales:(scales (part (k / 32))) (random_bytes (part (k / 2)))

let floats shape =
  Nx.init Nx.float32 shape (fun _ -> Random.State.float rng 2.0 -. 1.0)

let same a b = a = b || (Float.is_nan a && Float.is_nan b)

let check_floats msg expected actual =
  let n = Array.length expected in
  equal ~msg:(msg ^ ", length") int n (Array.length actual);
  Array.iteri
    (fun i e ->
      if not (same e actual.(i)) then
        fail
          (Printf.sprintf "%s: at %d, expected %h, got %h" msg i e actual.(i)))
    expected

let parts (Nx_quant.Mxfp4 { codes; scales }) = (codes, scales)

(* The format's values, from its definition. *)
let reference w =
  let codes, scales = parts w in
  let codes = Nx.to_array codes and scales = Nx.to_array scales in
  let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |] in
  Array.init
    (2 * Array.length codes)
    (fun i ->
      let byte = codes.(i / 2) in
      let code = if i mod 2 = 0 then byte land 15 else byte lsr 4 in
      let s = scales.(i / 32) in
      let magnitude = e2m1.(code land 7) *. Float.ldexp 1.0 (s - 127) in
      let v = if code land 8 = 0 then magnitude else -.magnitude in
      if s = 255 then Float.nan else Int32.float_of_bits (Int32.bits_of_float v))

(* Construction *)

let test_mxfp4_shapes () =
  let w =
    Nx_quant.mxfp4
      ~scales:(Nx.zeros Nx.uint8 [| 3; 4; 2 |])
      (Nx.zeros Nx.uint8 [| 3; 4; 32 |])
  in
  equal ~msg:"logical shape" (array int) [| 3; 4; 64 |] (Nx_quant.shape w);
  let refuses msg e ~scales codes =
    raises ~msg (Invalid_argument e) (fun () ->
        ignore
          (Nx_quant.mxfp4 ~scales:(Nx.zeros Nx.uint8 scales)
             (Nx.zeros Nx.uint8 codes)))
  in
  refuses "one axis"
    "Nx_quant.mxfp4: codes must have shape [...; n; k / 2] with k a multiple \
     of 32, got [32]"
    ~scales:[| 2 |] [| 32 |];
  refuses "k not a multiple of 32"
    "Nx_quant.mxfp4: codes must have shape [...; n; k / 2] with k a multiple \
     of 32, got [4; 8]"
    ~scales:[| 4; 1 |] [| 4; 8 |];
  refuses "a scale per 16 values"
    "Nx_quant.mxfp4: scales must have shape [4; 2], one per 32 values, got [4; \
     4]"
    ~scales:[| 4; 4 |] [| 4; 32 |];
  refuses "scales missing a leading axis"
    "Nx_quant.mxfp4: scales must have shape [2; 4; 1], one per 32 values, got \
     [4; 1]"
    ~scales:[| 4; 1 |] [| 2; 4; 16 |]

(* A device whose storage is host memory of its own and that counts the elements
   read from it, and a zero uint8 tensor placed on it with the count of its
   elements read. *)

type Nx_effect.storage +=
  | Bytes : (int, Nx.uint8_elt) Nx_buffer.t -> Nx_effect.storage

let counting reads =
  {
    Nx_effect.read =
      (fun (type a b) (r : (a, b) Nx_effect.resident) : (a, b) Nx_buffer.t ->
        match
          (r.r_cell.state, Nx_core.Dtype.equal_witness r.r_dtype Nx.uint8)
        with
        | Live (Bytes _), Some Type.Equal ->
            let n = Nx_core.View.numel r.r_view in
            reads := !reads + n;
            Nx_buffer.create Nx_buffer.uint8 n
        | _ -> assert false);
    place = (fun _ _ -> invalid_arg "the counting device places nothing");
  }

let deferred shape =
  let reads = ref 0 in
  let n = Array.fold_left ( * ) 1 shape in
  let engine = counting reads in
  let device = Nx_effect.Device.make "COUNTING" engine in
  let cell =
    Nx_effect.cell engine ~length:n (Bytes (Nx_buffer.create Nx_buffer.uint8 n))
  in
  ( Nx_effect.placed
      (Nx.Placement.device device)
      Nx.uint8
      (Nx_core.View.create shape)
      cell,
    reads )

let test_no_bytes_read () =
  let codes, code_fills = deferred [| 2; 8; 32 |] in
  let scales, scale_fills = deferred [| 2; 8; 2 |] in
  let w = Nx_quant.mxfp4 ~scales codes in
  equal ~msg:"shape" (array int) [| 2; 8; 64 |] (Nx_quant.shape w);
  let w = Nx.Ptree.map Nx_quant.ptree (fun _ t -> t) w in
  ignore (Nx.Ptree.map2 Nx_quant.ptree (fun _ a _ -> a) w w);
  ignore (Nx.Ptree.visits Nx_quant.ptree w);
  let bad, bad_fills = deferred [| 2; 8; 3 |] in
  raises ~msg:"a mismatch"
    (Invalid_argument
       "Nx_quant.mxfp4: scales must have shape [2; 8; 2], one per 32 values, \
        got [2; 8; 3]") (fun () -> ignore (Nx_quant.mxfp4 ~scales:bad codes));
  equal ~msg:"bytes read" int 0 (!code_fills + !scale_fills + !bad_fills)

(* A weight already where it is placed keeps its parts. *)
let test_place_host () =
  let w =
    Nx_quant.mxfp4
      ~scales:(Nx.full Nx.uint8 [| 8; 2 |] 127)
      (Nx.full Nx.uint8 [| 8; 32 |] 0x21)
  in
  let (Nx_quant.Mxfp4 { codes; scales }) = w in
  let (Nx_quant.Mxfp4 p) = Nx_quant.place Nx.Placement.host w in
  is_true ~msg:"codes" (p.codes == codes);
  is_true ~msg:"scales" (p.scales == scales)

(* Values *)

(* Every scale byte and every code byte, overflowing groups included. *)
let test_dequant_values () =
  let scales = bytes [| 256; 2 |] (fun i -> (i.(0) + (37 * i.(1))) mod 256) in
  let codes =
    bytes [| 256; 32 |] (fun i -> ((i.(0) * 32) + (7 * i.(1))) mod 256)
  in
  let w = Nx_quant.mxfp4 ~scales codes in
  let expected = reference w in
  is_true ~msg:"some values overflow"
    (Array.exists (fun v -> Float.abs v = Float.infinity) expected);
  let f32 = Nx_quant.dequant Nx.float32 w in
  equal ~msg:"shape" (array int) [| 256; 64 |] (Nx.shape f32);
  check_floats "float32" expected (Nx.to_array f32);
  check_floats "bfloat16" expected
    (Nx.to_array (Nx.cast Nx.float32 (Nx_quant.dequant Nx.bfloat16 w)))

let test_dequant_edges () =
  let group s code =
    Nx_quant.dequant Nx.float32
      (Nx_quant.mxfp4
         ~scales:(bytes [| 1; 1 |] (fun _ -> s))
         (bytes [| 1; 16 |] (fun _ -> code)))
  in
  let first s code = Nx.item [ 0; 0 ] (group s code) in
  equal ~msg:"byte 0 is 2^-127" float_exact (Float.ldexp 1.0 (-127))
    (first 0 0x22);
  equal ~msg:"byte 0 keeps halves" float_exact (Float.ldexp 1.0 (-128))
    (first 0 0x11);
  equal ~msg:"byte 127 is 1" float_exact (-6.0) (first 127 0xff);
  equal ~msg:"byte 254 is 2^127" float_exact (Float.ldexp 3.0 126)
    (first 254 0x33);
  equal ~msg:"2 at byte 254 overflows" float_exact Float.infinity
    (first 254 0x44);
  is_true ~msg:"byte 255 is NaN, zero codes included"
    (Array.for_all Float.is_nan (Nx.to_array (group 255 0)))

(* A weight of more than one chunk of values, whose chunk boundary falls inside
   a matrix. *)
let test_dequant_chunks () =
  let w = weight [| 3; 1500; 1024 |] in
  let expected = reference w in
  let f32 = Nx_quant.dequant Nx.float32 w in
  equal ~msg:"shape" (array int) [| 3; 1500; 1024 |] (Nx.shape f32);
  check_floats "float32" expected (Nx.to_array f32);
  check_floats "bfloat16" expected
    (Nx.to_array (Nx.cast Nx.float32 (Nx_quant.dequant Nx.bfloat16 w)))

(* Products *)

(* [close msg ~k expected bound actual] checks [actual] against [expected]
   within twice the error of a float32 sum of [k] terms whose magnitudes sum to
   [bound], and NaN exactly where [expected] is. *)
let close ?(relative = 0.0) msg ~k expected bound actual =
  let expected = Nx.to_array expected and bound = Nx.to_array bound in
  let actual = Nx.to_array (Nx.cast Nx.float32 actual) in
  equal ~msg:(msg ^ ", length") int (Array.length expected)
    (Array.length actual);
  Array.iteri
    (fun i e ->
      let a = actual.(i) in
      if Float.is_nan e || Float.is_nan a then
        begin if not (Float.is_nan e && Float.is_nan a) then
          fail (Printf.sprintf "%s: at %d, expected %g, got %g" msg i e a)
        end
      else
        let tol =
          (2.0 *. float_of_int k *. Float.ldexp 1.0 (-24) *. bound.(i))
          +. (relative *. Float.abs e)
        in
        if Float.abs (a -. e) > tol then
          fail (Printf.sprintf "%s: at %d, expected %g, got %g" msg i e a))
    expected

let reference_product ?(transpose = false) x dq =
  let x = Nx.cast Nx.float32 x in
  let side w = if transpose then w else Nx.matrix_transpose w in
  (Nx.matmul x (side dq), Nx.matmul (Nx.abs x) (side (Nx.abs dq)))

let check_apply msg w x =
  let k = (Nx_quant.shape w).(Array.length (Nx_quant.shape w) - 1) in
  let expected, bound = reference_product x (Nx_quant.dequant Nx.float32 w) in
  let y = Nx_quant.apply w x in
  equal ~msg:(msg ^ ", shape") (array int) (Nx.shape expected) (Nx.shape y);
  close msg ~k expected bound y;
  let y16 = Nx_quant.apply w (Nx.cast Nx.bfloat16 x) in
  equal ~msg:(msg ^ ", bfloat16 dtype") string "bfloat16"
    (Nx_core.Dtype.to_string (Nx.dtype y16));
  close ~relative:(Float.ldexp 1.0 (-8)) (msg ^ ", bfloat16") ~k
    (fst
       (reference_product (Nx.cast Nx.bfloat16 x)
          (Nx_quant.dequant Nx.float32 w)))
    bound y16

let test_apply_shapes () =
  check_apply "matrix" (weight [| 5; 64 |]) (floats [| 3; 4; 64 |]);
  check_apply "vector" (weight [| 5; 64 |]) (floats [| 64 |]);
  check_apply "broadcast batch"
    (weight [| 2; 5; 64 |])
    (floats [| 4; 1; 3; 64 |]);
  check_apply "unit weight batch" (weight [| 1; 5; 64 |]) (floats [| 3; 64 |]);
  equal ~msg:"vector shape" (array int) [| 2; 5 |]
    (Nx.shape (Nx_quant.apply (weight [| 2; 5; 64 |]) (floats [| 64 |])))

let test_apply_chunks () =
  check_apply "two chunks of rows"
    (weight [| 1100; 4096 |])
    (floats [| 2; 4096 |])

let test_apply_errors () =
  let w = weight [| 2; 5; 64 |] in
  let refuses msg e f =
    raises ~msg (Invalid_argument e) (fun () -> ignore (f ()))
  in
  refuses "x's last axis"
    "Nx_quant.apply: x's last axis is 32, the weight's inputs are 64" (fun () ->
      Nx_quant.apply w (floats [| 3; 32 |]));
  refuses "scalar x" "Nx_quant.apply: x must have at least one axis" (fun () ->
      Nx_quant.apply w (Nx.scalar Nx.float32 1.0));
  refuses "batch axes" "Nx_quant.apply: batch axes [3] and [2] do not broadcast"
    (fun () -> Nx_quant.apply w (floats [| 3; 1; 64 |]));
  let ids = Nx.zeros Nx.int32 [| 3 |] in
  refuses "no expert axis"
    "Nx_quant.apply: ids need a weight with an expert axis, got shape [5; 64]"
    (fun () -> Nx_quant.apply ~ids (weight [| 5; 64 |]) (floats [| 64 |]));
  refuses "ids without lanes"
    "Nx_quant.apply: ids of shape [] lack the weight's 1 leading axes"
    (fun () ->
      Nx_quant.apply ~ids:(Nx.scalar Nx.int32 0l)
        (weight [| 2; 3; 5; 64 |])
        (floats [| 64 |]))

let ints shape values = Nx.create Nx.int32 shape (Array.map Int32.of_int values)

(* [gathered dq lanes ids] is [w'] for the decoded weight [dq] of shape [[|
   lanes...; e; n; k |]] with [lanes] leading axes: ids clamped into range, one
   matrix per position of [ids] broadcast against the weight's lanes. *)
let gathered dq ~lanes ids =
  let ds = Nx.shape dq in
  let e = ds.(lanes) in
  let lane_shape = Array.sub ds 0 lanes in
  let is = Nx.shape ids in
  let wb =
    Array.append
      (Array.init lanes (fun a -> max lane_shape.(a) is.(a)))
      (Array.sub is lanes (Array.length is - lanes))
  in
  let ids = Nx.broadcast_to wb ids in
  let rec positions prefix a =
    if a = Array.length wb then [ List.rev prefix ]
    else
      List.concat_map
        (fun i -> positions (i :: prefix) (a + 1))
        (List.init wb.(a) Fun.id)
  in
  let matrix pos =
    let id = Int32.to_int (Nx.item pos ids) in
    let lane =
      List.mapi
        (fun a i -> if lane_shape.(a) = 1 then 0 else i)
        (List.filteri (fun a _ -> a < lanes) pos)
    in
    Nx.slice (List.map (fun i -> Nx.I i) (lane @ [ max 0 (min (e - 1) id) ])) dq
  in
  let n = ds.(lanes + 1) and k = ds.(lanes + 2) in
  match positions [] 0 with
  | [] -> Nx.zeros Nx.float32 (Array.append wb [| n; k |])
  | ps ->
      Nx.reshape
        (Array.append wb [| n; k |])
        (Nx.stack ~axis:0 (List.map matrix ps))

(* [check_ids] checks [apply ~ids w x] against the product over [w'], and exact
   zeros where an id is outside [0, e). *)
let check_ids ?(transpose = false) msg ~lanes w ids x =
  let e = (Nx_quant.shape w).(lanes) in
  let k = (Nx_quant.shape w).(if transpose then lanes + 1 else lanes + 2) in
  let wprime = gathered (Nx_quant.dequant Nx.float32 w) ~lanes ids in
  let expected, bound = reference_product ~transpose x wprime in
  let y = Nx_quant.Effect.perform w (Apply { ids = Some ids; x; transpose }) in
  equal ~msg:(msg ^ ", shape") (array int) (Nx.shape expected) (Nx.shape y);
  let valid =
    Nx.logical_and (Nx.greater_equal_s ids 0l) (Nx.less_s ids (Int32.of_int e))
  in
  let units = if Nx.ndim x = 1 then [| 1 |] else [| 1; 1 |] in
  let valid = Nx.reshape (Array.append (Nx.shape valid) units) valid in
  let valid = Nx.broadcast_to (Nx.shape expected) valid in
  close msg ~k (Nx.where valid expected (Nx.zeros_like expected)) bound y;
  let y = Nx.to_array y in
  Array.iteri
    (fun i v ->
      if not v then
        let a = y.(i) in
        if Int64.bits_of_float a <> 0L then
          fail (Printf.sprintf "%s: at %d, no expert gives %h" msg i a))
    (Nx.to_array valid)

let test_ids () =
  let w = weight [| 4; 6; 64 |] in
  let ids = ints [| 5; 2 |] [| 0; 3; -1; 2; 4; 1; 1; 1; 7; -5 |] in
  let x = floats [| 5; 1; 1; 64 |] in
  let poisoned =
    Nx.set [ I 4 ] (Nx.full Nx.float32 [| 1; 1; 64 |] Float.nan) x
  in
  let poisoned =
    Nx.set [ I 2; I 0; I 0; I 3 ] (Nx.scalar Nx.float32 Float.infinity) poisoned
  in
  check_ids "tokens" ~lanes:0 w ids x;
  check_ids "NaN and infinity in x" ~lanes:0 w ids poisoned;
  check_ids "one matrix per id" ~lanes:0 w ids (floats [| 5; 2; 3; 64 |]);
  check_ids "vector x" ~lanes:0 w ids (floats [| 64 |]);
  check_ids "lanes" ~lanes:1
    (weight [| 2; 3; 6; 64 |])
    (ints [| 2; 4 |] [| 0; 2; -1; 2; 1; 1; 3; 0 |])
    (floats [| 2; 4; 1; 64 |]);
  check_ids "ids broadcast over lanes" ~lanes:1
    (weight [| 2; 3; 6; 64 |])
    (ints [| 1; 4 |] [| 2; 0; -1; 0 |])
    (floats [| 2; 4; 1; 64 |]);
  check_ids "a lane broadcast over ids" ~lanes:1
    (weight [| 1; 3; 6; 64 |])
    (ints [| 2; 4 |] [| 2; 0; -1; 0; 1; 1; 2; 5 |])
    (floats [| 4; 2; 64 |])

let test_ids_chunks () =
  check_ids "two chunks of rows" ~lanes:0
    (weight [| 3; 1100; 4096 |])
    (ints [| 2; 2 |] [| 2; 0; 2; -1 |])
    (floats [| 2; 1; 1; 4096 |])

let test_ids_empty () =
  let w = weight [| 4; 6; 64 |] in
  let shape ids x = Nx.shape (Nx_quant.apply ~ids w x) in
  equal ~msg:"no tokens" (array int) [| 0; 2; 1; 6 |]
    (shape (Nx.zeros Nx.int32 [| 0; 2 |]) (floats [| 0; 1; 1; 64 |]));
  equal ~msg:"no experts per token" (array int) [| 3; 0; 1; 6 |]
    (shape (Nx.zeros Nx.int32 [| 3; 0 |]) (floats [| 3; 1; 1; 64 |]));
  equal ~msg:"no rows" (array int) [| 3; 2; 0; 6 |]
    (shape (Nx.zeros Nx.int32 [| 3; 2 |]) (floats [| 3; 1; 0; 64 |]))

let test_ids_read_nothing () =
  let codes, code_fills = deferred [| 4; 6; 32 |] in
  let scales, scale_fills = deferred [| 4; 6; 2 |] in
  let w = Nx_quant.mxfp4 ~scales codes in
  let y =
    Nx_quant.apply
      ~ids:(ints [| 3; 2 |] [| -1; 4; 9; -1; -3; 4 |])
      w
      (Nx.full Nx.float32 [| 3; 1; 1; 64 |] Float.nan)
  in
  check_floats "zeros" (Array.make 36 0.0) (Nx.to_array y);
  equal ~msg:"bytes read" int 0 (!code_fills + !scale_fills)

(* The transposed product, which only a reverse rule performs. *)

let test_transposed () =
  let check msg w x =
    let s = Nx_quant.shape w in
    let n = s.(Array.length s - 2) in
    let expected, bound =
      reference_product ~transpose:true x (Nx_quant.dequant Nx.float32 w)
    in
    let y =
      Nx_quant.Effect.perform w (Apply { ids = None; x; transpose = true })
    in
    equal ~msg:(msg ^ ", shape") (array int) (Nx.shape expected) (Nx.shape y);
    close msg ~k:n expected bound y
  in
  check "matrix" (weight [| 5; 64 |]) (floats [| 3; 4; 5 |]);
  check "vector" (weight [| 2; 5; 64 |]) (floats [| 5 |]);
  check "two chunks of rows" (weight [| 1100; 4096 |]) (floats [| 2; 1100 |]);
  let ids = ints [| 5; 2 |] [| 0; 3; -1; 2; 4; 1; 1; 1; 7; -5 |] in
  let x = floats [| 5; 1; 1; 6 |] in
  let poisoned =
    Nx.set [ I 4 ] (Nx.full Nx.float32 [| 1; 1; 6 |] Float.nan) x
  in
  check_ids ~transpose:true "ids" ~lanes:0 (weight [| 4; 6; 64 |]) ids poisoned;
  check_ids ~transpose:true "ids over lanes" ~lanes:1
    (weight [| 2; 3; 6; 64 |])
    (ints [| 1; 4 |] [| 2; 0; -1; 0 |])
    (floats [| 2; 4; 1; 6 |]);
  check_ids ~transpose:true "ids, two chunks of rows" ~lanes:0
    (weight [| 3; 1100; 4096 |])
    (ints [| 2; 2 |] [| 2; 0; 2; -1 |])
    (floats [| 2; 2; 3; 1100 |]);
  raises ~msg:"x's last axis"
    (Invalid_argument
       "Nx_quant.apply: x's last axis is 64, the weight's outputs are 5")
    (fun () ->
      ignore
        (Nx_quant.Effect.perform
           (weight [| 5; 64 |])
           (Apply { ids = None; x = floats [| 3; 64 |]; transpose = true })))

(* Views *)

let contiguous w = Nx.Ptree.map Nx_quant.ptree (fun _ t -> Nx.contiguous t) w

(* [check_views msg ?ids w x] checks that a weight whose parts are views gives
   the values and products of the same weight over contiguous parts. *)
let check_views msg ?ids w x =
  is_false ~msg:(msg ^ ", parts are views") (Nx.is_c_contiguous (fst (parts w)));
  let c = contiguous w in
  check_floats (msg ^ ", dequant")
    (Nx.to_array (Nx_quant.dequant Nx.float32 c))
    (Nx.to_array (Nx_quant.dequant Nx.float32 w));
  check_floats (msg ^ ", apply")
    (Nx.to_array (Nx_quant.apply c x))
    (Nx.to_array (Nx_quant.apply w x));
  Option.iter
    (fun ids ->
      check_floats (msg ^ ", apply ~ids")
        (Nx.to_array (Nx_quant.apply ~ids c x))
        (Nx.to_array (Nx_quant.apply ~ids w x)))
    ids

let test_views () =
  let rows t = Nx.slice [ A; R (2, 7); A ] t in
  check_views "rows of each matrix"
    ~ids:(ints [| 4; 2 |] [| 0; 2; -1; 1; 2; 2; 1; 0 |])
    (Nx.Ptree.map Nx_quant.ptree (fun _ t -> rows t) (weight [| 3; 10; 64 |]))
    (floats [| 4; 1; 1; 64 |]);
  let swap t = Nx.transpose ~axes:[ 1; 0; 2; 3 ] t in
  check_views "leading axes swapped"
    ~ids:(ints [| 3; 2 |] [| 3; 0; -1; 2; 1; 1 |])
    (Nx.Ptree.map Nx_quant.ptree (fun _ t -> swap t) (weight [| 4; 3; 5; 64 |]))
    (floats [| 3; 1; 1; 64 |]);
  let first_inputs t = Nx.slice [ A; A; R (0, Nx.dim 2 t / 2) ] t in
  check_views "the first half of the inputs"
    ~ids:(ints [| 2; 2 |] [| 1; 0; 2; -1 |])
    (Nx.Ptree.map Nx_quant.ptree
       (fun _ t -> first_inputs t)
       (weight [| 3; 5; 128 |]))
    (floats [| 2; 1; 1; 64 |])

(* An [x] that is a view whose batch axes do not merge, as a cotangent broadcast
   along the positions is. *)
let test_x_views () =
  let w = weight [| 4; 6; 64 |] and dense = weight [| 6; 64 |] in
  let ids = ints [| 5; 2 |] [| 0; 3; -1; 2; 1; 1; 3; 0; 2; 2 |] in
  let check msg x =
    is_false ~msg:(msg ^ ", x is a view") (Nx.is_c_contiguous x);
    check_floats (msg ^ ", apply")
      (Nx.to_array (Nx_quant.apply dense (Nx.contiguous x)))
      (Nx.to_array (Nx_quant.apply dense x));
    check_floats (msg ^ ", apply ~ids")
      (Nx.to_array (Nx_quant.apply ~ids w (Nx.contiguous x)))
      (Nx.to_array (Nx_quant.apply ~ids w x))
  in
  check "broadcast along the positions"
    (Nx.broadcast_to [| 5; 2; 1; 64 |] (floats [| 5; 1; 1; 64 |]));
  check "batch axes permuted"
    (Nx.transpose ~axes:[ 1; 0; 2; 3 ] (floats [| 2; 5; 1; 64 |]));
  check "the first two of three positions"
    (Nx.slice [ A; R (0, 2); A; A ] (floats [| 5; 3; 1; 64 |]))

(* The effect *)

let test_effect () =
  let w = weight [| 3; 4; 64 |] in
  let x = floats [| 2; 1; 1; 64 |]
  and ids = ints [| 2; 2 |] [| 0; 2; 1; -1 |] in
  let seen = ref [] in
  let answer : type a b. Nx_quant.t -> (a, b) Nx_quant.Effect.op -> (a, b) Nx.t
      =
   fun w' op ->
    is_true ~msg:"the weight" (w' == w);
    match op with
    | Apply { ids = ids'; x = x'; transpose } ->
        seen := "apply" :: !seen;
        is_false ~msg:"apply leaves transpose false" transpose;
        is_true ~msg:"the ids" (Option.equal ( == ) ids' (Some ids));
        check_floats "the x" (Nx.to_array x)
          (Nx.to_array (Nx.cast Nx.float32 x'));
        Nx.full (Nx.dtype x') [| 7 |] 42.0
    | Dequant dt ->
        seen := Nx_core.Dtype.to_string dt :: !seen;
        Nx.full dt [| 5 |] 7.0
  in
  let handled f =
    Effect.Deep.try_with f ()
      {
        effc =
          (fun (type c) (e : c Effect.t) ->
            match e with
            | Nx_quant.Effect.E_quant { w = w'; op } ->
                Some
                  (fun (k : (c, _) Effect.Deep.continuation) ->
                    Effect.Deep.continue k (answer w' op))
            | _ -> None);
      }
  in
  check_floats "apply returns the handler's answer" (Array.make 7 42.0)
    (Nx.to_array (handled (fun () -> Nx_quant.apply ~ids w x)));
  check_floats "dequant returns the handler's answer" (Array.make 5 7.0)
    (Nx.to_array
       (Nx.cast Nx.float32 (handled (fun () -> Nx_quant.dequant Nx.bfloat16 w))));
  equal ~msg:"operations seen" (list string) [ "apply"; "bfloat16" ]
    (List.rev !seen);
  let unrelated f =
    Effect.Deep.try_with f () { effc = (fun (type c) (_ : c Effect.t) -> None) }
  in
  let performed = ref 0 in
  let counted f =
    Effect.Deep.try_with f ()
      {
        effc =
          (fun (type c) (e : c Effect.t) ->
            match e with
            | Nx_quant.Effect.E_quant _ ->
                incr performed;
                None
            | _ -> None);
      }
  in
  raises ~msg:"shapes are checked"
    (Invalid_argument
       "Nx_quant.apply: x's last axis is 32, the weight's inputs are 64")
    (fun () -> ignore (counted (fun () -> Nx_quant.apply w (floats [| 32 |]))));
  equal ~msg:"before the effect is performed" int 0 !performed;
  check_floats "an unrelated handler falls through"
    (Nx.to_array (Nx_quant.apply ~ids w x))
    (Nx.to_array (unrelated (fun () -> Nx_quant.apply ~ids w x)))

(* Traversals *)

let test_visits () =
  let w = weight [| 3; 4; 64 |] in
  equal ~msg:"the case, then codes before scales" (list string)
    [ "the root: case \"mxfp4\""; "codes: a leaf"; "scales: a leaf" ]
    (List.map
       (Format.asprintf "%a" Nx.Ptree.pp_visit)
       (Nx.Ptree.visits Nx_quant.ptree w));
  let parts =
    Nx.Ptree.fold Nx_quant.ptree
      (fun _ t acc -> (Nx.shape t, Nx_core.Dtype.to_string (Nx.dtype t)) :: acc)
      w []
  in
  equal ~msg:"each part keeps its dtype"
    (list (pair (array int) string))
    [ ([| 3; 4; 32 |], "uint8"); ([| 3; 4; 2 |], "uint8") ]
    (List.rev parts);
  let (Nx_quant.Mxfp4 { codes; scales }) = w in
  let (Nx_quant.Mxfp4 r) =
    Nx.Ptree.rebuild Nx_quant.ptree ~like:w
      (fst (Nx.Ptree.flatten Nx_quant.ptree w))
  in
  is_true ~msg:"round trip" (r.codes == codes && r.scales == scales)

let test_walk_checks () =
  let w = weight [| 3; 4; 64 |] in
  (* Halving an axis of two leaves the scales one per 64 values. *)
  let halve (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    if Nx.dim (-1) t = 2 then Nx.slice [ A; A; R (0, 1) ] t else t
  in
  raises ~msg:"map"
    (Invalid_argument
       "Nx_quant.walk: scales must have shape [3; 4; 2], one per 32 values, \
        got [3; 4; 1]") (fun () ->
      ignore (Nx.Ptree.map Nx_quant.ptree (fun _ t -> halve t) w));
  raises ~msg:"map2 on a changed part"
    (Invalid_argument
       "Nx_quant.walk: scales must have shape [3; 4; 2], one per 32 values, \
        got [3; 4; 1]") (fun () ->
      ignore (Nx.Ptree.map2 Nx_quant.ptree (fun _ a _ -> halve a) w w))

let tests =
  [
    group "construction"
      [
        test "mxfp4 checks shapes" test_mxfp4_shapes;
        test "construction reads no bytes" test_no_bytes_read;
        test "place keeps parts already placed" test_place_host;
      ];
    group "dequant"
      [
        test "every code and scale byte" test_dequant_values;
        test "scale bytes 0, 254 and 255" test_dequant_edges;
        test "chunks" test_dequant_chunks;
      ];
    group "apply"
      [
        test "matmul's shapes" test_apply_shapes;
        test "chunks" test_apply_chunks;
        test "errors" test_apply_errors;
      ];
    group "ids"
      [
        test "experts, lanes and ids outside the experts" test_ids;
        test "chunks" test_ids_chunks;
        test "empty" test_ids_empty;
        test "no expert reads nothing" test_ids_read_nothing;
        test "transposed" test_transposed;
      ];
    group "views"
      [
        test "sliced and permuted parts" test_views;
        test "x's batch axes as views" test_x_views;
      ];
    group "effect" [ test "apply and dequant perform E_quant" test_effect ];
    group "traversals"
      [
        test "the case, then codes before scales" test_visits;
        test "rebuilt parts are checked" test_walk_checks;
      ];
  ]

let () = run "nx quant" tests
