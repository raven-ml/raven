(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtypes =
  Nx.
    [
      Dtype float64;
      Dtype float32;
      Dtype float16;
      Dtype bfloat16;
      Dtype float8_e4m3fn;
      Dtype float8_e5m2;
      Dtype float4_e2m1fn;
    ]

let dtype_name (Dtype dt) = Nx.Dtype.name dt
let floats t = Nx.to_array (Nx.cast Nx.float64 t)
let of_floats dt shape xs = Nx.cast dt (Nx.create Nx.float64 shape xs)

(* A tensor as its shape and its elements' bits. *)
let value = pair (array int) (array float_exact)
let value_of t = (Nx.shape t, floats t)
let linear64 = Nx.Ptree.instantiate (module Linear)
let grads_ok = function Ok () -> () | Error m -> fail m

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

(* Two software devices, minted when a test first needs them. *)
let two =
  lazy
    (Nx.devices
       (List.map
          (fun n -> Result.get_ok (Rig.software n))
          [ "kaun-linear-0"; "kaun-linear-1" ]))

(* A witness for placements of one set: each set has its own brand. *)
let placement () =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)

(* An initializer that records the fans it is applied with. *)
let recording fans key ~fan_in ~fan_out dt shape =
  fans := (fan_in, fan_out) :: !fans;
  Init.zeros key ~fan_in ~fan_out dt shape

(* A layer and an input whose terms are [± m 2^e], [m] 1 or 3, [|e| <= 3] (1 at
   float4, 10 at float64), each representable in the case's dtype, and now and
   then NaN, an infinity or [-0.]. float8_e4m3fn has no infinity and float4
   neither infinity nor NaN: there a special term is what [Nx.cast] makes of it,
   and the reference reads the cast elements back. A product needs at most 4
   bits and the exact sum at most 18 bits (products from 2^-6 to 576, six
   terms), 47 at float64: float64 holds every sum exactly and float32 often
   cannot, as float16, bfloat16, float8 and float4 often cannot hold theirs. *)

type case = {
  dtype : dtype;
  batch : int array;
  inputs : int;
  outputs : int;
  x : float array;
  w : float array;
  b : float array option;
}

let pp_floats = Format.(pp_print_array ~pp_sep:pp_print_space pp_print_float)
let pp_ints = Format.(pp_print_array ~pp_sep:pp_print_space pp_print_int)

let pp_case ppf c =
  Format.fprintf ppf
    "@[<v>%s, batch [%a], inputs %d, outputs %d@,x [%a]@,w [%a]@,b %a@]"
    (dtype_name c.dtype) pp_ints c.batch c.inputs c.outputs pp_floats c.x
    pp_floats c.w
    (Format.pp_print_option
       ~none:(fun ppf () -> Format.pp_print_string ppf "None")
       (fun ppf b -> Format.fprintf ppf "[%a]" pp_floats b))
    c.b

let term (Dtype dt) =
  let open Gen in
  let widest =
    if Nx.Dtype.equal dt Nx.float4_e2m1fn then 1
    else if Nx.Dtype.equal dt Nx.float64 then 10
    else 3
  in
  let finite =
    let+ negative = bool
    and+ m = of_list [ 1.; 3. ]
    and+ e = int_range (-widest) widest in
    (if negative then -.m else m) *. Float.ldexp 1. e
  in
  frequency [ (16, finite); (1, of_list [ nan; infinity; neg_infinity; -0. ]) ]

let case =
  let open Gen in
  with_pp pp_case
  @@ let* dtype =
       of_list
         ~pp:(fun ppf d -> Format.pp_print_string ppf (dtype_name d))
         dtypes
     in
     let* batch = array ~size:(int_range 0 4) (int_range 0 3)
     and+ inputs = int_range 1 5
     and+ outputs = int_range 1 5
     and+ bias = bool in
     let rows = Array.fold_left ( * ) 1 batch in
     let terms n = array ~size:(constant n) (term dtype) in
     let+ x = terms (rows * inputs)
     and+ w = terms (outputs * inputs)
     and+ b = terms outputs in
     {
       dtype;
       batch;
       inputs;
       outputs;
       x;
       w;
       b = (if bias then Some b else None);
     }

let layer_of dt c =
  {
    Linear.w = of_floats dt [| c.outputs; c.inputs |] c.w;
    b = Option.map (of_floats dt [| c.outputs |]) c.b;
  }

let input_of dt c = of_floats dt (Array.append c.batch [| c.inputs |]) c.x

(* [b + Σ x w] in float64 over the elements of [p] and [x], exact on the case's
   terms. As [Nx.contract] states, the products sum from [+0] and [b] is added
   to their sum, so [b = -0.] over products of [-0.] gives [+0.]. *)
let exact_sums p x =
  let w = floats p.Linear.w and x = floats x in
  let outputs, inputs = Nx.(dim 0 p.w, dim 1 p.w) in
  let rows = Array.length x / inputs in
  Array.init (rows * outputs) (fun k ->
      let r = k / outputs and o = k mod outputs in
      let s = ref 0. in
      for i = 0 to inputs - 1 do
        s := !s +. (x.((r * inputs) + i) *. w.((o * inputs) + i))
      done;
      match p.b with Some b -> !s +. (floats b).(o) | None -> !s)

(* The sum [1 + 2^-p + 2^-(p+1)]. At a dtype whose ulp at 1 is [2^-(p-1)],
   rounded per step, [1 + 2^-p] ties to even, [1], and adding [2^-(p+1)] rounds
   to [1] again; rounded once, the sum is [1 + 2^-(p-1)]. At a dtype that holds
   the sum, it is the sum, which an accumulator of fewer bits would round. *)
let one_plus_tie dtype ~p =
  {
    dtype;
    batch = [| 1 |];
    inputs = 2;
    outputs = 1;
    x = [| 1.; 1. |];
    w = [| 1.; Float.ldexp 1. (-p) |];
    b = Some [| Float.ldexp 1. (-p - 1) |];
  }

let apply_examples =
  [
    (* 0eeac2d1b8 *)
    one_plus_tie (Dtype Nx.bfloat16) ~p:8;
    one_plus_tie (Dtype Nx.float16) ~p:11;
    one_plus_tie (Dtype Nx.float8_e4m3fn) ~p:4;
    one_plus_tie (Dtype Nx.float8_e5m2) ~p:3;
    (* [b = -0.] added to a sum of [-0.] from [+0]: [+0.]. *)
    {
      dtype = Dtype Nx.float64;
      batch = [| 1 |];
      inputs = 1;
      outputs = 1;
      x = [| 1. |];
      w = [| -0. |];
      b = Some [| -0. |];
    };
    (* A sum float64 holds and float32 does not. *)
    one_plus_tie (Dtype Nx.float64) ~p:30;
    (* At float4, per step 4 + 1 ties to 4 and 4 + 0.5 rounds to 4; once, 5.5
       rounds to 6. *)
    {
      dtype = Dtype Nx.float4_e2m1fn;
      batch = [| 1 |];
      inputs = 2;
      outputs = 1;
      x = [| 1.; 1. |];
      w = [| 4.; 1. |];
      b = Some [| 0.5 |];
    };
    (* [x @ wᵀ + b], [w] a row per output. *)
    {
      dtype = Dtype Nx.float32;
      batch = [| 2 |];
      inputs = 2;
      outputs = 2;
      x = [| 1.; 1.; 0.; 2. |];
      w = [| 1.; 3.; 2.; 4. |];
      b = Some [| 10.; 20. |];
    };
  ]

let law_rounded_once c =
  let (Dtype dt) = c.dtype in
  let p = layer_of dt c and x = input_of dt c in
  let sums = exact_sums p x in
  let expected = of_floats dt (Array.append c.batch [| c.outputs |]) sums in
  let special v = Float.is_nan v || Float.abs v = infinity in
  List.iter (fun d -> cover ("at " ^ dtype_name d) (d == c.dtype)) dtypes;
  cover "x is 1-d" (Array.length c.batch = 0);
  cover "x has two or more batch axes" (Array.length c.batch >= 2);
  cover "x is 5-d" (Array.length c.batch = 4);
  cover "a NaN or an infinity"
    (List.exists (Array.exists special)
       [ c.x; c.w; Option.value c.b ~default:[||] ]);
  cover "a -0."
    (List.exists
       (Array.exists (fun v -> 1. /. v = neg_infinity))
       [ c.x; c.w; Option.value c.b ~default:[||] ]);
  cover "at float64, a sum float32 cannot hold"
    (Nx.Dtype.equal dt Nx.float64
    && Array.exists
         (fun s ->
           not (Float.equal s (Int32.float_of_bits (Int32.bits_of_float s))))
         sums);
  cover "a zero-size batch" (Array.exists (( = ) 0) c.batch);
  cover "one input" (c.inputs = 1);
  cover "one output" (c.outputs = 1);
  cover "no bias" (c.b = None);
  cover "the exact sum is not representable in the dtype"
    (Array.exists2 (fun s r -> not (Float.equal s r)) sums (floats expected));
  equal value (value_of expected) (value_of (Linear.apply p x))

(* Over the two devices: a layer of 4 inputs and 4 outputs with a bias, and an
   [x] of [[3; 4]]. *)
let device_case dtypes =
  let open Gen in
  with_pp pp_case
  @@ let* dtype = of_list dtypes in
     let terms n = array ~size:(constant n) (term dtype) in
     let+ x = terms 12 and+ w = terms 16 and+ b = terms 4 in
     { dtype; batch = [| 3 |]; inputs = 4; outputs = 4; x; w; b = Some b }

(* Which operands devices cut along the summed axis. *)
type cut = Both | W_alone | X_alone

let cut =
  Gen.of_list
    ~pp:(fun ppf c ->
      Format.pp_print_string ppf
        (match c with
        | Both -> "Both"
        | W_alone -> "W_alone"
        | X_alone -> "X_alone"))
    [ Both; W_alone; X_alone ]

(* Each device's term sums the products of its window of the summed axis, and
   the bias joins the sum once. At float32 and float64 the terms and their sum
   are exact on the case's terms, so the closed result has the exact sum's bits.
   At a narrower dtype each term rounds into it before the close, which nx
   bounds and does not make exact, so those dtypes are not drawn. *)
let law_sum_cut (cut, c) =
  let (module Two) = Lazy.force two in
  let (Dtype dt) = c.dtype in
  let p = layer_of dt c and x = input_of dt c in
  let summed = Two.split ~axis:1 in
  let w_at, x_at =
    match cut with
    | Both -> (summed, summed)
    | W_alone -> (summed, Two.on)
    | X_alone -> (Two.on, summed)
  in
  let placed =
    { Linear.w = Nx.place w_at p.w; b = Option.map (Nx.place Two.on) p.b }
  in
  cover "w and x are cut" (cut = Both);
  cover "w alone is cut" (cut = W_alone);
  cover "x alone is cut" (cut = X_alone);
  let y = Linear.apply ~sum:true placed (Nx.place x_at x) in
  let at = Nx.placement y in
  cover "the result is the devices' terms"
    (not (Nx.Placement.equal (Nx.Placement.closed at) at));
  equal value
    (value_of (of_floats dt [| 3; 4 |] (exact_sums p x)))
    (value_of (Nx.place (Nx.Placement.closed at) y))

let law_sum_uncut c =
  let (module Two) = Lazy.force two in
  let (Dtype dt) = c.dtype in
  let p = layer_of dt c and x = input_of dt c in
  let outputs = Two.split ~axis:0 in
  let placed =
    { Linear.w = Nx.place outputs p.w; b = Option.map (Nx.place outputs) p.b }
  in
  let x = Nx.place Two.on x in
  let at_value y = (Nx.placement y, value_of y) in
  equal
    (pair (placement ()) value)
    (at_value (Linear.apply placed x))
    (at_value (Linear.apply ~sum:true placed x))

(* t *)

let test_shapes () =
  let p = Linear.init ~key:(Nx.Rng.key 0) ~inputs:3 ~outputs:5 in
  equal
    (pair (array int) (option (array int)))
    ([| 5; 3 |], Some [| 5 |])
    (Nx.shape p.w, Option.map Nx.shape p.b)

let test_no_bias () =
  is_none
    (Linear.make ~key:(Nx.Rng.key 0) ~bias:false ~inputs:3 ~outputs:5 Nx.float32)
      .b

(* make *)

let test_fans () =
  let w_fans = ref [] and b_fans = ref [] in
  ignore
    (Linear.make ~w_init:(recording w_fans) ~bias_init:(recording b_fans)
       ~key:(Nx.Rng.key 0) ~inputs:3 ~outputs:5 Nx.float32);
  equal
    (pair (list (pair int int)) (list (pair int int)))
    ([ (3, 5) ], [ (3, 5) ])
    (!w_fans, !b_fans)

let test_split_keys () =
  let key = Nx.Rng.key 4 in
  let k = Nx.Rng.split key in
  let draw = Init.normal ~stddev:1. in
  let p =
    Linear.make ~w_init:draw ~bias_init:draw ~key ~inputs:3 ~outputs:5
      Nx.float32
  in
  equal (pair value value)
    ( value_of (draw k.(0) ~fan_in:3 ~fan_out:5 Nx.float32 [| 5; 3 |]),
      value_of (draw k.(1) ~fan_in:3 ~fan_out:5 Nx.float32 [| 5 |]) )
    (value_of p.w, value_of (Option.get p.b))

let test_defaults () =
  let key = Nx.Rng.key 5 in
  let k = Nx.Rng.split key in
  let p = Linear.make ~key ~inputs:3 ~outputs:5 Nx.float64 in
  equal (pair value value)
    ( value_of
        (Init.glorot_uniform k.(0) ~fan_in:3 ~fan_out:5 Nx.float64 [| 5; 3 |]),
      ([| 5 |], Array.make 5 0.) )
    (value_of p.w, value_of (Option.get p.b))

let test_bias_init_ignored () =
  let fans = ref [] in
  ignore
    (Linear.make ~bias_init:(recording fans) ~bias:false ~key:(Nx.Rng.key 0)
       ~inputs:3 ~outputs:5 Nx.float32);
  equal (list (pair int int)) [] !fans

let test_placed_key () =
  let (module Two) = Lazy.force two in
  let key = Nx.Rng.key 3 in
  let host = Linear.init ~key ~inputs:3 ~outputs:2 in
  let placed =
    Linear.init ~key:(Nx.Rng.place Two.on key) ~inputs:3 ~outputs:2
  in
  equal (placement ()) Two.on (Nx.placement placed.w);
  equal ~msg:"the values the host key draws" (array float_exact) (floats host.w)
    (floats placed.w)

let test_make_refuses =
  cases ~id:fst "raises if inputs or outputs is not positive"
    [ ("inputs 0", (0, 4)); ("outputs -1", (3, -1)) ]
    (fun (_, (inputs, outputs)) ->
      raises_match (Exn.invalid_arg ~sub:"Linear.make") (fun () ->
          Linear.make ~key:(Nx.Rng.key 0) ~inputs ~outputs Nx.float32))

(* init *)

let test_init () =
  let key = Nx.Rng.key 6 in
  let p = Linear.init ~key ~inputs:3 ~outputs:5 in
  let q = Linear.make ~key ~inputs:3 ~outputs:5 Nx.float32 in
  equal (pair value value)
    (value_of q.w, value_of (Option.get q.b))
    (value_of p.w, value_of (Option.get p.b))

(* apply *)

(* At float32 the accumulator is the dtype, so rounding once gives the bits of
   the product, then the bias added, at depths on either side of a block of the
   contraction's products (0eeac2d1b8). *)
let test_float32_bits =
  cases ~id:string_of_int
    "at float32, the result has the bits of the product plus the bias"
    [ 144; 300 ] (fun depth ->
      Nx.Rng.with_key (Nx.Rng.key 6) @@ fun () ->
      let x = Nx.Rng.normal Nx.float32 [| 64; depth |] in
      let w =
        Nx.copy (Nx.matrix_transpose (Nx.Rng.normal Nx.float32 [| depth; 40 |]))
      in
      let b = Nx.mul_s (Nx.Rng.normal Nx.float32 [| 40 |]) 10. in
      equal value
        (value_of (Nx.add (Nx.matmul x (Nx.matrix_transpose w)) b))
        (value_of (Linear.apply { Linear.w; b = Some b } x)))

let test_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 5) @@ fun () ->
  let x = Nx.Rng.normal Nx.float64 [| 4; 3 |] in
  let loss x p =
    let y = Linear.apply p x in
    Nx.sum (Nx.mul y y)
  in
  let p =
    Linear.make ~key:(Nx.Rng.next_key ()) ~inputs:3 ~outputs:2 Nx.float64
  in
  let no_bias =
    Linear.make ~key:(Nx.Rng.next_key ()) ~bias:false ~inputs:3 ~outputs:2
      Nx.float64
  in
  grads_ok (Rune.check_grads linear64 (loss x) p);
  grads_ok (Rune.check_grads linear64 (loss x) no_bias);
  grads_ok (Rune.check_grads Nx.Ptree.tensor (fun x -> loss x p) x)

let test_apply_refuses =
  cases ~id:fst "raises if x is 0-d or its last axis is not of size inputs"
    [ ("0-d", [||]); ("last axis 2", [| 4; 2 |]); ("last axis 4", [| 4 |]) ]
    (fun (_, shape) ->
      let p = Linear.init ~key:(Nx.Rng.key 0) ~inputs:3 ~outputs:2 in
      raises_match
        (fun e -> Exn.invalid_arg e)
        (fun () -> Linear.apply p (Nx.zeros Nx.float32 shape)))

(* walk *)

let test_visits =
  cases ~id:fst "visits w, then b, reporting whether b is present"
    [
      ("with a bias", (true, [ "w: a leaf"; "b: Some"; "b: a leaf" ]));
      ("without a bias", (false, [ "w: a leaf"; "b: None" ]));
    ]
    (fun (_, (bias, expected)) ->
      equal (list string) expected
        (visits linear64
           (Linear.make ~key:(Nx.Rng.key 0) ~bias ~inputs:3 ~outputs:2
              Nx.float64)))

(* The model of linear.mli's preamble. *)
module Mlp = struct
  type ('a, 'd) t = { l1 : ('a, 'd) Linear.t; l2 : ('a, 'd) Linear.t }

  let walk c { l1; l2 } =
    let open Nx.Ptree.Walk in
    let l1 = field c "l1" Linear.walk l1 in
    let l2 = field c "l2" Linear.walk l2 in
    { l1; l2 }
end

let test_saved_names () =
  let k = Nx.Rng.split (Nx.Rng.key 0) in
  let params =
    {
      Mlp.l1 = Linear.init ~key:k.(0) ~inputs:4 ~outputs:8;
      l2 = Linear.init ~key:k.(1) ~inputs:8 ~outputs:2;
    }
  in
  let mlp = Nx.Ptree.instantiate (module Mlp) in
  equal (list string)
    [ "l1.b"; "l1.w"; "l2.b"; "l2.w" ]
    (Nx_io.Archive.names (Nx_io.Archive.of_value mlp params))

let test_cast () =
  let p = Linear.init ~key:(Nx.Rng.key 0) ~inputs:3 ~outputs:2 in
  let half = Nx.Ptree.cast (module Linear) Nx.bfloat16 p in
  equal (list string) [ "bfloat16"; "bfloat16" ]
    (Nx.Ptree.fold
       (Nx.Ptree.instantiate (module Linear))
       (fun _ x acc -> Nx.Dtype.name (Nx.dtype x) :: acc)
       half [])

let () =
  exit
    (run "kaun linear"
       [
         group "t"
           [
             test "w is [outputs; inputs] and b is [outputs]" test_shapes;
             test "b is None for a layer made without a bias" test_no_bias;
           ];
         group "make"
           [
             test "w_init and bias_init get inputs and outputs as their fans"
               test_fans;
             test "the weight and the bias draw from the two keys split makes"
               test_split_keys;
             test "the defaults are Glorot-uniform weights and a zero bias"
               test_defaults;
             test "bias_init is ignored without a bias" test_bias_init_ignored;
             test "a layer drawn from a placed key lies there" test_placed_key;
             test_make_refuses;
           ];
         group "init" [ test "is make at float32" test_init ];
         group "apply"
           [
             prop ~count:200 ~examples:apply_examples
               "each output is b + Σ x w rounded once into the dtype" case
               law_rounded_once;
             test_float32_bits;
             prop
               "with ~sum:true, where devices cut the summed axis, the closed \
                result is b + Σ x w, the shift added once"
               (Gen.pair cut (device_case Nx.[ Dtype float64; Dtype float32 ]))
               law_sum_cut;
             prop
               "with ~sum:true, where no device cuts the summed axis, the \
                result is the same"
               (device_case dtypes) law_sum_uncut;
             test
               "gradients in the parameters and x agree with central \
                differences"
               test_gradients;
             test_apply_refuses;
           ];
         group "walk"
           [
             test_visits;
             test "the paths it gives name the leaves in a saved file"
               test_saved_names;
             test "Nx.Ptree.cast converts every leaf" test_cast;
           ];
       ])
