(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantised weights against the format's definition: dequant gives each value
   at its dtype, and apply is the product with the dequantised weight within the
   error of a float32 sum. *)

open Windtrap
open Nx_test
module S = Nx_dtype.Scalar

(* The format *)

let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |]

(* The exact values of [w] in row-major order: a code's magnitude, signed, times
   [2 ^ (s - 127)] for its group's scale byte [s], NaN for [255]. *)
let values (Nx_quant.Mxfp4 { codes; scales }) =
  let codes = Nx.to_array codes and scales = Nx.to_array scales in
  Array.init
    (2 * Array.length codes)
    (fun i ->
      let code = (codes.(i / 2) lsr (4 * (i mod 2))) land 15 in
      let s = scales.(i / 32) in
      let m = e2m1.(code land 7) *. Float.ldexp 1. (s - 127) in
      if s = 255 then Float.nan else if code < 8 then m else -.m)

(* A float dtype, the rounding of a value computed at float32 to it, and the
   error of that rounding: [rel] of the value, or [tiny] below its least
   normal. *)
type fdt =
  | F : {
      dtype : (float, 'b) Nx.dtype;
      round : float -> float;
      rel : float;
      tiny : float;
    }
      -> fdt

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let narrow dtype s rel tiny =
  let round x =
    if Float.is_nan x then x else S.decode s (S.encode s (to_f32 x))
  in
  F { dtype; round; rel = Float.ldexp 1. rel; tiny = Float.ldexp 1. tiny }

let fdts =
  [
    F { dtype = Nx.float32; round = to_f32; rel = 0.; tiny = 0. };
    narrow Nx.bfloat16 S.BFloat16 (-8) (-134);
    narrow Nx.float16 S.Float16 (-11) (-25);
    F { dtype = Nx.float64; round = to_f32; rel = 0.; tiny = 0. };
  ]

let fdt =
  Gen.of_list
    ~pp:(fun ppf (F d) ->
      Format.pp_print_string ppf (Nx_dtype.to_string d.dtype))
    fdts

(* Weights *)

(* A view of both parts that keeps them a weight, where the shape allows it. *)
type view = {
  view : string;
  fits : int array -> bool;
  apply : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t;
}

let views =
  let r = Array.length in
  let last t = List.init (Nx.ndim t - 1) (fun _ -> Nx.A) in
  [
    { view = "contiguous"; fits = (fun _ -> true); apply = Fun.id };
    {
      view = "every other row";
      fits = (fun s -> s.(r s - 2) > 0);
      apply =
        (fun t ->
          Nx.squeeze ~axes:[ -1 ]
            (Nx.sliding_window ~axis:(-2) ~window:1 ~step:2 t));
    };
    {
      view = "its leading axes swapped";
      fits = (fun s -> r s >= 4);
      apply = (fun t -> Nx.swapaxes 0 1 t);
    };
    {
      view = "the first half of its inputs";
      fits = (fun s -> s.(r s - 1) mod 64 = 0);
      apply = (fun t -> Nx.slice (last t @ [ R (0, Nx.dim (-1) t / 2) ]) t);
    };
  ]

let pp_weight ppf (view, (Nx_quant.Mxfp4 { codes; scales } as w)) =
  Format.fprintf ppf "@[<v>%a, %s@,codes %a@,scales %a@]" pp_shape
    (Nx_quant.shape w) view Nx.pp codes Nx.pp scales

(* A weight of shape [[| lead...; n; k |]] under a view, its scale bytes at the
   overflow, subnormal and NaN corners too. *)
let weight ?(lead = Gen.list ~size:(Gen.int_range 0 2) (Gen.int_range 0 3))
    ?(n = Gen.int_range 0 4) () =
  let open Gen in
  let scale =
    frequency
      [
        (4, int_range 118 136);
        (1, of_list ~pp:Format.pp_print_int [ 0; 1; 127; 253; 254; 255 ]);
      ]
  in
  with_pp pp_weight
    (let* lead, n, groups = triple lead n (int_range 1 4) in
     let m = List.fold_left ( * ) 1 lead * n in
     let+ codes = array ~size:(constant (m * groups * 16)) (int_range 0 255)
     and+ scales = array ~size:(constant (m * groups)) scale
     and+ v = of_list views in
     let part last = Nx.create Nx.uint8 (Array.of_list (lead @ [ n; last ])) in
     let w =
       Nx_quant.mxfp4 ~scales:(part groups scales) (part (groups * 16) codes)
     in
     if v.fits (Nx_quant.shape w) then
       (v.view, Nx.Ptree.map Nx_quant.ptree (fun _ t -> v.apply t) w)
     else ("contiguous", w))

let dims w =
  let s = Nx_quant.shape w in
  let r = Array.length s in
  (Array.sub s 0 (r - 2), s.(r - 2), s.(r - 1))

(* A weight of fixed shape with finite values, for large weights. *)
let random_weight shape =
  let rng = Random.State.make [| 4 |] and r = Array.length shape in
  let part last f =
    Nx.init Nx.uint8 (Array.append (Array.sub shape 0 (r - 1)) [| last |]) f
  in
  Nx_quant.mxfp4
    ~scales:(part (shape.(r - 1) / 32) (fun _ -> 118 + Random.State.int rng 19))
    (part (shape.(r - 1) / 2) (fun _ -> Random.State.int rng 256))

let random_floats shape =
  let rng = Random.State.make [| 5 |] in
  Nx.init Nx.float32 shape (fun _ -> Random.State.float rng 2. -. 1.)

(* Products *)

(* [agrees ~at ~k expected bound actual] holds when [actual] is [expected]
   rounded to [at] within twice the error of a float32 sum of [k] terms whose
   magnitudes sum to [bound], subnormal terms included, and the error of the
   rounding; non-finite values must be exact. *)
let agrees ?(at = List.hd fdts) ~k expected bound actual =
  let (F d) = at in
  equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape actual);
  let b = Nx.to_array bound and a = Nx.to_array (Nx.cast Nx.float64 actual) in
  let worst = ref 0. in
  Array.iteri
    (fun i e ->
      if not (Float.is_finite (d.round e)) then
        equal float_exact (d.round e) a.(i)
      else
        let tol =
          2. *. float_of_int k
          *. ((Float.ldexp 1. (-24) *. b.(i)) +. Float.ldexp 1. (-149))
          +. Float.max (d.rel *. Float.abs e) d.tiny
        in
        let err = Float.abs (a.(i) -. e) in
        let ratio = if err = 0. then 0. else err /. tol in
        worst :=
          Float.max !worst
            (if Float.is_nan ratio then Float.infinity else ratio))
    (Nx.to_array expected);
  at_most ~msg:"worst error over its bound" float_exact ~than:1. !worst

(* The product of [x] with each matrix of [w'], transposed, at float32, and the
   same product of magnitudes. *)
let product x w' =
  let x = Nx.cast Nx.float32 x in
  let t = Nx.matrix_transpose in
  (Nx.matmul x (t w'), Nx.matmul (Nx.abs x) (t (Nx.abs w')))

let broadcast a b =
  let n = Int.max (Array.length a) (Array.length b) in
  let dim s i =
    if i < n - Array.length s then 1 else s.(i - n + Array.length s)
  in
  Array.init n (fun i -> if dim a i = 1 then dim b i else dim a i)

(* [w'] of [apply ~ids] for the dequantised weight [dq] with [lanes] leading
   axes: each id's expert in its lane, ids clamped into the experts. *)
let gathered dq ~lanes ids =
  let ds = Nx.shape dq and is = Nx.shape ids in
  let wb =
    Array.append
      (broadcast (Array.sub ds 0 lanes) (Array.sub is 0 lanes))
      (Array.sub is lanes (Array.length is - lanes))
  in
  let ids = Nx.broadcast_to wb ids in
  let matrix i =
    let pos = unravel wb i in
    let id = Int64.to_int (Nx.item (Array.to_list pos) ids) in
    let lane = List.init lanes (fun a -> if ds.(a) = 1 then 0 else pos.(a)) in
    Nx.slice
      (List.map
         (fun i -> Nx.I i)
         (lane @ [ Int.max 0 (Int.min (ds.(lanes) - 1) id) ]))
      dq
  in
  let count = Ref.numel wb in
  Nx.reshape
    (Array.append wb [| ds.(lanes + 1); ds.(lanes + 2) |])
    (if count = 0 then Nx.zeros Nx.float32 [| 0 |]
     else Nx.stack ~axis:0 (List.init count matrix))

(* An input at one of the float dtypes. *)
type input = X : { at : fdt; x : (float, 'b) Nx.t } -> input

let flag = Gen.of_list ~pp:Format.pp_print_bool [ false; true ]

let usually =
  Gen.frequency [ (3, Gen.constant ~pp:Format.pp_print_bool true); (1, flag) ]

(* An input whose last axis is [inputs] and whose batch axes broadcast against
   [batch]: a vector, or rows behind [batch]'s last axes, each whole or of one,
   or behind all of them and an axis of its own in front. *)
let input ~batch ~inputs =
  let open Gen in
  let* shape =
    let+ vector = frequency [ (1, constant true); (4, constant false) ]
    and+ front = list ~size:(int_range 0 1) (int_range 1 2)
    and+ whole = list ~size:(constant (Array.length batch)) usually
    and+ drop = int_range 0 (Array.length batch)
    and+ m = int_range 0 3 in
    let own = List.mapi (fun i w -> if w then batch.(i) else 1) whole in
    if vector then [| inputs |]
    else
      Array.of_list
        ((if drop = 0 then front else [])
        @ List.filteri (fun i _ -> i >= drop) own
        @ [ m; inputs ])
  in
  let entry =
    frequency
      [
        (30, float_range (-1.) 1.);
        ( 1,
          of_list ~pp:Format.pp_print_float
            [ Float.nan; Float.infinity; Float.neg_infinity ] );
      ]
  in
  let+ at = fdt
  and+ xs = array ~size:(constant (Ref.numel shape)) entry
  and+ view =
    of_list ~pp:Format.pp_print_string [ "contiguous"; "broadcast"; "flipped" ]
  in
  let (F d) = at in
  let r = Array.length shape in
  let x = Nx.cast d.dtype (Nx.create Nx.float64 shape xs) in
  (* Views whose batch axes do not merge. *)
  let x =
    match view with
    | "broadcast" when r >= 3 && shape.(r - 3) > 0 ->
        Nx.broadcast_to shape
          (Nx.slice (List.init (r - 3) (fun _ -> Nx.A) @ [ R (0, 1) ]) x)
    | "flipped" -> Nx.flip x
    | _ -> x
  in
  X { at; x }

(* A weight, ids selecting its experts or none, and an input. *)
let products =
  let open Gen in
  let id e =
    frequency
      [
        (6, map Int64.of_int (int_range (-2) (e + 1)));
        ( 1,
          of_list
            ~pp:(fun ppf -> Format.fprintf ppf "%Ld")
            [ Int64.min_int; Int64.max_int; 0x1_0000_0000L ] );
      ]
  in
  let* experts, lanes =
    pair flag (list ~size:(int_range 0 1) (int_range 1 2))
  in
  let* view, w =
    if experts then
      weight ~lead:(map (fun e -> lanes @ [ e ]) (int_range 1 3)) ()
    else weight ()
  in
  let lead, _, k = dims w in
  let p = Array.length lead - 1 in
  let* ids, batch =
    if not experts then constant (None, lead)
    else
      let* whole, tokens =
        pair
          (list ~size:(constant p) usually)
          (list ~size:(int_range 0 2) (int_range 0 3))
      in
      let is =
        Array.append
          (Array.of_list
             (List.mapi (fun a w -> if w then lead.(a) else 1) whole))
          (Array.of_list tokens)
      in
      let+ ids = array ~size:(constant (Ref.numel is)) (id lead.(p)) in
      ( Some (Nx.create Nx.int64 is ids),
        Array.append
          (broadcast (Array.sub lead 0 p) (Array.sub is 0 p))
          (Array.of_list tokens) )
  in
  let+ x = input ~batch ~inputs:k in
  (view, w, ids, x)

let pp_product ppf (view, w, ids, X { x; _ }) =
  Format.fprintf ppf "@[<v>%a@,ids %a@,x %a@]" pp_weight (view, w)
    (Format.pp_print_option Nx.pp)
    ids Nx.pp x

(* [label] is [cover] in a property. *)
let product_law (_, w, ids, X { at; x }) =
  let lead, _, k = dims w in
  let dq = Nx_quant.dequant Nx.float32 w in
  let w' =
    match ids with
    | None -> dq
    | Some ids -> gathered dq ~lanes:(Array.length lead - 1) ids
  in
  let expected, bound = product x w' in
  let valid =
    match ids with
    | None -> Nx.full Nx.bool (Nx.shape expected) true
    | Some ids ->
        let e = Int64.of_int lead.(Array.length lead - 1) in
        let v = Nx.logical_and (Nx.greater_equal_s ids 0L) (Nx.less_s ids e) in
        let units = if Nx.ndim x = 1 then [| 1 |] else [| 1; 1 |] in
        Nx.broadcast_to (Nx.shape expected)
          (Nx.reshape (Array.append (Nx.shape v) units) v)
  in
  cover "an id that selects no expert" (Array.mem false (Nx.to_array valid));
  cover "an empty result" (Nx.numel expected = 0);
  let y = Nx_quant.apply ?ids w x in
  equal ~msg:"dtype" string
    (Nx_dtype.to_string (Nx.dtype x))
    (Nx_dtype.to_string (Nx.dtype y));
  agrees ~at ~k (Nx.where valid expected (Nx.zeros_like expected)) bound y;
  let y = Nx.to_array (Nx.cast Nx.float64 y) in
  Array.iteri
    (fun i v -> if not v then equal ~msg:"no expert" float_exact 0. y.(i))
    (Nx.to_array valid)

let values_and_products =
  group "values and products"
    [
      prop
        "dequant gives each value of the format, computed at float32, at its \
         dtype"
        (Gen.pair (weight ()) fdt)
        (fun ((_, w), F d) ->
          let exact = values w in
          cover "an infinite value at float32"
            (Array.exists (fun v -> Float.abs v > 0x1.fffffep127) exact);
          cover "a NaN group" (Array.exists Float.is_nan exact);
          let got = Nx_quant.dequant d.dtype w in
          equal
            (pair (array int) (array float_exact))
            (Nx_quant.shape w, Array.map d.round exact)
            (Nx.shape got, Nx.to_array (Nx.cast Nx.float64 got)));
      prop
        "apply is the product with the dequantised weight, or with each id's \
         expert and exactly zero where an id selects none"
        (Gen.with_pp pp_product products)
        product_law;
    ]

(* Construction and placement *)

open Devices

let errors =
  let w = random_weight [| 2; 5; 64 |] in
  let x = random_floats in
  let mxfp4 scales codes () =
    ignore
      (Nx_quant.mxfp4 ~scales:(Nx.zeros Nx.uint8 scales)
         (Nx.zeros Nx.uint8 codes))
  in
  let halve (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    if Nx.dim (-1) t = 2 then Nx.slice [ A; A; R (0, 1) ] t else t
  in
  let apply ?ids w x () = ignore (Nx_quant.apply ?ids w x) in
  cases "refuse, naming what is wrong"
    ~name:(fun (n, _, _) -> n)
    [
      ("mxfp4 codes of one axis", "codes", mxfp4 [| 2 |] [| 32 |]);
      ("mxfp4 k not a multiple of 32", "codes", mxfp4 [| 4; 1 |] [| 4; 8 |]);
      ("mxfp4 a scale per 16 values", "scales", mxfp4 [| 4; 4 |] [| 4; 32 |]);
      ( "mxfp4 scales without a leading axis",
        "scales",
        mxfp4 [| 4; 1 |] [| 2; 4; 16 |] );
      ( "a map that changes a part's shape",
        "Nx_quant.walk",
        fun () -> ignore (Nx.Ptree.map Nx_quant.ptree (fun _ t -> halve t) w) );
      ( "apply to a scalar",
        "at least one axis",
        apply w (Nx.scalar Nx.float32 1.) );
      ( "apply to a last axis of another size",
        "last axis",
        apply w (x [| 3; 32 |]) );
      ( "apply over batch axes that do not broadcast",
        "broadcast",
        apply w (x [| 3; 1; 64 |]) );
      ( "ids without an expert axis",
        "expert axis",
        apply
          ~ids:(Nx.zeros Nx.int64 [| 3 |])
          (random_weight [| 5; 64 |])
          (x [| 64 |]) );
      ( "ids without the weight's lanes",
        "leading axes",
        apply ~ids:(Nx.scalar Nx.int64 0L)
          (random_weight [| 2; 3; 5; 64 |])
          (x [| 64 |]) );
    ]
    (fun (_, part, f) -> raises_match (Exn.invalid_arg ~substring:part) f)

let placements =
  let drawn =
    let open Gen in
    let* ((_, weight) as w) =
      weight
        ~lead:(list ~size:(int_range 0 1) (int_range 1 4))
        ~n:(int_range 1 4) ()
    in
    let k = Array.length (Nx_quant.shape weight) - 1 in
    let+ devices =
      of_list
        ~pp:(fun ppf l -> Format.fprintf ppf "over %d devices" (List.length l))
        [ [ d1; d2 ]; [ d1; d2; d3; d4 ] ]
    and+ axis =
      frequency [ (1, constant (Some k)); (2, option (int_range 0 3)) ]
    in
    (w, devices, axis)
  in
  prop
    "place splits any axis Nx.place can, and k only between groups of 32 values"
    drawn (fun ((_, w), devices, axis) ->
      let s = Nx_quant.shape w in
      let r = Array.length s in
      let p =
        match axis with
        | None -> Nx.Placement.replicated devices
        | Some axis -> Nx.Placement.sharded ~axis devices
      in
      let refused =
        match axis with
        | None -> false
        | Some a ->
            a >= r
            || (if a = r - 1 then s.(a) / 32 else s.(a)) mod List.length devices
               <> 0
      in
      cover "a split of k" (axis = Some (r - 1) && not refused);
      cover "k refused" (axis = Some (r - 1) && refused);
      match Nx_quant.place p w with
      | exception Invalid_argument _ ->
          is_true ~msg:"refused only where a split cuts a group or an axis"
            refused
      | Nx_quant.Mxfp4 { codes; scales } as placed ->
          is_false ~msg:"placed where a split cuts a group or an axis" refused;
          equal
            (triple placement placement (array float_exact))
            (p, p, Nx.to_array (Nx_quant.dequant Nx.float32 w))
            ( Nx.placement codes,
              Nx.placement scales,
              Nx.to_array (Nx_quant.dequant Nx.float32 placed) ))

let others =
  group "weights"
    [
      test "construction, maps and visits read no byte" (fun () ->
          let place s = Nx.place (Nx.Placement.on d1) (Nx.zeros Nx.uint8 s) in
          let codes = place [| 4; 6; 32 |]
          and scales = place [| 4; 6; 2 |]
          and bad = place [| 4; 6; 3 |] in
          let sent = bytes_out () in
          let w =
            Nx.Ptree.map Nx_quant.ptree
              (fun _ t -> t)
              (Nx_quant.mxfp4 ~scales codes)
          in
          ignore (Nx.Ptree.map2 Nx_quant.ptree (fun _ a _ -> a) w w);
          raises_invalid_arg (fun () -> Nx_quant.mxfp4 ~scales:bad codes);
          equal (array int) [| 4; 6; 64 |] (Nx_quant.shape w);
          equal int 0 (bytes_out () - sent));
      test "ids that select no expert give zeros over NaN rows" (fun () ->
          let w = random_weight [| 4; 6; 64 |] in
          let ids =
            Nx.create Nx.int64 [| 3; 2 |] [| -1L; 4L; 9L; -1L; -3L; 4L |]
          in
          equal (tensor float_exact)
            (Nx.zeros Nx.float32 [| 3; 2; 1; 6 |])
            (Nx_quant.apply ~ids w
               (Nx.full Nx.float32 [| 3; 1; 1; 64 |] Float.nan)));
      test "dequant and apply read no value's elements" (fun () ->
          let w = random_weight [| 3; 4; 64 |] in
          let reads f =
            let seen = ref [] in
            let run : type r. r Nx.Op.t -> r =
             fun op ->
              (match op with Read { by; _ } -> seen := by :: !seen | _ -> ());
              Nx.Op.eval op
            in
            let claims : type r. r Nx.Op.t -> bool = function
              | Read _ -> true
              | _ -> false
            in
            Nx.Op.intercept { run; claims } (fun () -> ignore (f ()));
            List.sort_uniq String.compare !seen
          in
          equal (list string) []
            (reads (fun () -> Nx_quant.dequant Nx.float32 w));
          equal (list string) []
            (reads (fun () ->
                 Nx_quant.apply
                   ~ids:(Nx.create Nx.int64 [| 2 |] [| 0L; 2L |])
                   w
                   (random_floats [| 2; 1; 64 |]))));
      test "products live where their operands are" (fun () ->
          let p = Nx.Placement.on d1 in
          let w = Nx_quant.place p (random_weight [| 3; 4; 64 |]) in
          let x = Nx.place p (random_floats [| 2; 1; 64 |])
          and ids = Nx.place p (Nx.create Nx.int64 [| 2 |] [| 0L; 2L |]) in
          equal (pair placement placement) (p, p)
            ( Nx.placement (Nx_quant.dequant Nx.float32 w),
              Nx.placement (Nx_quant.apply ~ids w x) ));
      test
        "visits the case, then codes before scales; rebuild and place keep the \
         parts" (fun () ->
          let (Nx_quant.Mxfp4 { codes; scales } as w) =
            random_weight [| 3; 4; 64 |]
          in
          equal (list string)
            [ "the root: case \"mxfp4\""; "codes: a leaf"; "scales: a leaf" ]
            (List.map
               (Format.asprintf "%a" Nx.Ptree.pp_visit)
               (Nx.Ptree.visits Nx_quant.ptree w));
          let (Nx_quant.Mxfp4 r) =
            Nx.Ptree.rebuild Nx_quant.ptree ~like:w
              (fst (Nx.Ptree.flatten Nx_quant.ptree w))
          in
          is_true (r.codes == codes && r.scales == scales);
          let (Nx_quant.Mxfp4 p) = Nx_quant.place Nx.Placement.host w in
          is_true ~msg:"place keeps parts already placed"
            (p.codes == codes && p.scales == scales));
    ]

let () =
  exit (run "nx quant" [ values_and_products; errors; placements; others ])
