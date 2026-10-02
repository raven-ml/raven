(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantised products under rune. Compiled, Nx_quant.apply and dequant
   compute eager's values, which nx's suite checks against the format: on the
   host, on test devices over the host's memory with the weight, its routes and
   its rows placed, and on Metal (slow). The derivatives of apply in its rows,
   and its maps, are those of the product with the dequantised weight. *)

open Windtrap
open Nx_test

let rng = Random.State.make [| 7 |]

(* Scale bytes of finite values, with 0 and 1 (subnormal values) and 255 (NaN
   groups) among them. *)
let any_scale _ =
  match Random.State.int rng 16 with
  | 0 -> 255
  | 1 -> 0
  | 2 -> 1
  | _ -> 100 + Random.State.int rng 51

(* Scale bytes of normal float32 values, for Metal, which flushes subnormals,
   and for derivatives. *)
let moderate _ = 120 + Random.State.int rng 15

(* A weight of logical shape [shape] with random codes. *)
let weight ?(scale = any_scale) shape =
  let r = Array.length shape in
  let part last = Array.append (Array.sub shape 0 (r - 1)) [| last |] in
  let k = shape.(r - 1) in
  Nx_quant.mxfp4
    ~scales:(Nx.init Nx.uint8 (part (k / 32)) scale)
    (Nx.init Nx.uint8 (part (k / 2)) (fun _ -> Random.State.int rng 256))

let floats shape =
  Nx.init Nx.float32 shape (fun _ -> Random.State.float rng 2. -. 1.)

let ints shape v = Nx.create Nx.int64 shape (Array.map Int64.of_int v)

(* [poison ~at x] is [x] with a NaN and an infinity in its rows at the batch
   indices [at], positions that select no expert. *)
let poison ~at x =
  List.fold_left
    (fun x index ->
      let s = Nx.shape x in
      let row =
        Array.sub s (List.length index) (Array.length s - List.length index)
      in
      let bad =
        Nx.set [ I 0 ]
          (Nx.scalar Nx.float32 Float.infinity)
          (Nx.full Nx.float32 [| Array.fold_left ( * ) 1 row |] Float.nan)
      in
      Nx.set (List.map (fun i -> Nx.I i) index) (Nx.reshape row bad) x)
    x at

(* Agreement *)

(* [magnitudes w] is [w] with every code's sign cleared: the absolute values of
   [w]'s values. *)
let magnitudes (Nx_quant.Mxfp4 { codes; scales }) =
  Nx_quant.mxfp4 ~scales
    (Nx.bitwise_and codes (Nx.full Nx.uint8 (Nx.shape codes) 0x77))

type case = {
  name : string;
  w : Nx_quant.t;
  ids : Nx.int64_t option;
  x : Nx.float32_t;
  transpose : bool;
}

let case ?ids ?(transpose = false) name w x = { name; w; ids; x; transpose }

(* [product c w x] is [c]'s product of [w] and [x]. *)
let product c w x =
  Nx_quant.Effect.perform w
    (Nx_quant.Effect.Apply { ids = c.ids; x; transpose = c.transpose })

(* [agrees c expected actual] checks [actual] against eager's [expected] within
   the error of a float32 sum of [c]'s terms, each rounded once to [x]'s dtype:
   at a value whose terms' magnitudes sum to [b], within [2 k u b] plus [k]
   least normals, and one unit of [x]'s dtype in the last place of the value. A
   NaN is a NaN and an infinity itself; a position that selects no expert is
   exactly zero, as its bound is. *)
let agrees (type b) c (expected : (float, b) Nx.t) (actual : (float, b) Nx.t) =
  equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape actual);
  let s = Nx.shape c.x in
  let k = float_of_int s.(Array.length s - 1) in
  let bound =
    Nx.to_array (product c (magnitudes c.w) (Nx.abs (Nx.cast Nx.float32 c.x)))
  in
  let unit =
    match Nx.dtype expected with
    | Nx.Float16 -> Float.ldexp 1. (-11)
    | Nx.BFloat16 -> Float.ldexp 1. (-8)
    | _ -> 0.
  in
  let e = Nx.to_array (Nx.cast Nx.float32 expected)
  and a = Nx.to_array (Nx.cast Nx.float32 actual) in
  Array.iteri
    (fun i e ->
      let a = a.(i) in
      let tol =
        (2. *. k *. Float.ldexp 1. (-24) *. bound.(i))
        +. (k *. Float.ldexp 1. (-126))
        +. (2. *. unit *. Float.abs e)
      in
      if
        not
          ((Float.is_nan e && Float.is_nan a)
          || e = a
          || (Float.is_finite e && Float.abs (a -. e) <= tol))
      then failf "%s: at %d, expected %h, got %h (within %h)" c.name i e a tol)
    e

(* The products, from the old suite's Law 2 *)

let products ~scale =
  let w68 = weight ~scale [| 6; 8; 64 |]
  and w38 = weight ~scale [| 3; 8; 64 |] in
  let w48 = weight ~scale [| 4; 8; 64 |] in
  [
    case "without ids, matrices"
      (weight ~scale [| 5; 64 |])
      (floats [| 3; 4; 64 |]);
    case "without ids, batch axes broadcast"
      (weight ~scale [| 2; 5; 64 |])
      (floats [| 4; 1; 3; 64 |]);
    case "without ids, a vector"
      (weight ~scale [| 2; 5; 64 |])
      (floats [| 64 |]);
    case "fewer positions than experts"
      ~ids:(ints [| 2; 2 |] [| 3; -1; 6; 3 |])
      w68
      (poison ~at:[ [ 0; 1 ]; [ 1; 0 ] ] (floats [| 2; 2; 1; 64 |]));
    case "ids outside the experts, over a vector"
      ~ids:(ints [| 3 |] [| 5; -5; 0 |])
      w68 (floats [| 64 |]);
    case "ids 2^32 from an expert"
      ~ids:(ints [| 6; 1 |] [| 0; (1 lsl 32) + 2; 1; 3; 5 - (1 lsl 32); 4 |])
      w68
      (floats [| 6; 1; 2; 64 |]);
    case "a lane of experts per batch row"
      ~ids:(ints [| 2; 2 |] [| 2; -1; 0; 2 |])
      (weight ~scale [| 2; 3; 8; 64 |])
      (floats [| 2; 2; 1; 64 |]);
    case "as many positions as experts or more"
      ~ids:(ints [| 3; 2 |] [| 0; 2; -1; 1; 3; 2 |])
      w38
      (poison ~at:[ [ 1; 0 ] ] (floats [| 3; 2; 3; 64 |]));
    case "many routes of one row"
      ~ids:
        (ints [| 8; 2 |] [| 0; 3; -1; 2; 4; 3; 1; 1; 3; -5; 0; 2; 2; 2; 1; 0 |])
      w48
      (floats [| 8; 1; 1; 64 |]);
    case "transposed" ~transpose:true
      ~ids:(ints [| 3; 2 |] [| 0; 2; -1; 1; 3; 2 |])
      w38
      (poison ~at:[ [ 1; 0 ] ] (floats [| 3; 2; 2; 8 |]));
  ]

(* The largest scales: codes of magnitude 4 or more at scale byte 253, and of 2
   or more at 254, are infinite. *)
let largest =
  case "the largest scales"
    (weight ~scale:(fun i -> 253 + (i.(0) mod 2)) [| 4; 64 |])
    (floats [| 2; 64 |])

(* [compiled c] is [c]'s product compiled, its routes and rows arguments. *)
let compiled c =
  match c.ids with
  | None -> Rune.jit' (product c c.w) c.x
  | Some ids ->
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        (fun ids x -> product { c with ids = Some ids } c.w x)
        ids c.x

let values =
  group "values"
    [
      cases
        ~name:(fun c -> c.name)
        "compiled, a product is eager's"
        (largest :: products ~scale:any_scale)
        (fun c -> agrees c (product c c.w c.x) (compiled c));
      test "compiled, a product of a float16 x is eager's" (fun () ->
          let c =
            case "float16 x"
              ~ids:(ints [| 3; 2 |] [| 0; 2; -1; 1; 3; 2 |])
              (weight [| 3; 8; 64 |])
              (floats [| 3; 2; 3; 64 |])
          in
          let x = Nx.cast Nx.float16 c.x in
          let ids = Option.get c.ids in
          agrees c
            (Nx_quant.apply ~ids c.w x)
            (Rune.jit
               Nx.Ptree.(tensor @-> tensor @-> returns tensor)
               (fun ids x -> Nx_quant.apply ~ids c.w x)
               ids x));
      cases ~name:fst "compiled, dequant is eager's bit for bit"
        [
          ("float32", fun w -> Nx_quant.dequant Nx.float32 w);
          ( "bfloat16",
            fun w -> Nx.cast Nx.float32 (Nx_quant.dequant Nx.bfloat16 w) );
        ]
        (fun (_, f) ->
          let w = weight [| 3; 8; 64 |] in
          equal (tensor float_exact) (f w)
            (Rune.jit Nx.Ptree.(Nx_quant.ptree @-> returns tensor) f w));
      test "compiled, one token's four experts among 32 are its product"
        (fun () ->
          let c =
            case "one token"
              ~ids:(ints [| 1; 4 |] [| 3; 0; 2; 1 |])
              (weight ~scale:moderate [| 32; 64; 256 |])
              (floats [| 1; 1; 65; 256 |])
          in
          agrees c (product c c.w c.x) (compiled c));
    ]

(* Placements *)

let devices = Nx.Device.all [ Cpu 1; Cpu 2; Cpu 3; Cpu 4 ]

let pair = [ List.nth devices 0; List.nth devices 1 ]
let split ?(axis = 0) ds = Nx.Placement.sharded ~axis ds
let host t = Nx.place Nx.Placement.host t
let routed w ids x = Nx_quant.apply ~ids w x

let routed_compiled w =
  Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) (routed w)

let placements =
  let w = weight ~scale:moderate [| 4; 8; 64 |] in
  group "placements"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "routes and rows split over two devices give eager's product"
        [
          ("gathered", ints [| 2; 1 |] [| 3; -1 |], floats [| 2; 1; 1; 64 |]);
          ( "several rows per position",
            ints [| 2; 2 |] [| 3; -1; 0; 1 |],
            floats [| 2; 2; 2; 64 |] );
          ( "many routes",
            ints [| 8; 2 |] (Array.init 16 (fun i -> (i * 5 mod 6) - 1)),
            floats [| 8; 2; 1; 64 |] );
        ]
        (fun (name, ids, x) ->
          agrees (case ~ids name w x) (routed w ids x)
            (host
               (routed_compiled w
                  (Nx.place (split pair) ids)
                  (Nx.place (split pair) x))));
      test "experts split over two devices give eager's product" (fun () ->
          let ids = ints [| 8; 2 |] (Array.init 16 (fun i -> (i * 5 mod 6) - 1))
          and x = floats [| 8; 1; 1; 64 |] in
          agrees
            (case ~ids "split experts" w x)
            (routed w ids x)
            (host
               (Rune.jit
                  Nx.Ptree.(Nx_quant.ptree @-> returns tensor)
                  (fun w -> routed w ids x)
                  (Nx_quant.place (split pair) w))));
      test
        "experts split under routes split over two devices give eager's product"
        (fun () ->
          let ids = ints [| 4; 1 |] [| 3; 0; 1; 2 |]
          and x = floats [| 4; 1; 1; 64 |] in
          agrees
            (case ~ids "split experts and routes" w x)
            (routed w ids x)
            (host
               (Rune.jit
                  Nx.Ptree.(
                    Nx_quant.ptree @-> tensor @-> tensor @-> returns tensor)
                  routed
                  (Nx_quant.place (split pair) w)
                  (Nx.place (split pair) ids)
                  (Nx.place (split pair) x))));
      slow "sixteen experts over four devices, four each, give eager's product"
        (fun () ->
          let w = weight ~scale:moderate [| 16; 8; 64 |] in
          let ids =
            ints [| 8; 2 |]
              (Array.init 16 (fun i -> ((i * 7) + (i / 2)) mod 16))
          and x = floats [| 8; 1; 1; 64 |] in
          agrees
            (case ~ids "expert parallel" w x)
            (routed w ids x)
            (host
               (Rune.jit
                  Nx.Ptree.(
                    Nx_quant.ptree @-> tensor @-> tensor @-> returns tensor)
                  routed
                  (Nx_quant.place (split devices) w)
                  (Nx.place
                     (Nx.Placement.on devices)
                     ids)
                  (Nx.place
                     (Nx.Placement.on devices)
                     x))));
      test "dequant of a weight placed on a device is eager's bit for bit"
        (fun () ->
          let w = weight [| 2; 4; 128 |] in
          let p =
            Nx.Placement.on [ List.hd devices ]
          in
          equal (tensor float_exact)
            (Nx_quant.dequant Nx.float32 w)
            (host
               (Rune.jit
                  Nx.Ptree.(Nx_quant.ptree @-> returns tensor)
                  (Nx_quant.dequant Nx.float32)
                  (Nx_quant.place p w))));
    ]

(* Memory *)

(* [peak d f] is [f ()] and the most bytes [d] held while it ran beyond those it
   held before. *)
let peak d f =
  Gc.full_major ();
  Nx_device.synchronize d;
  let before = Nx_device.Stats.allocated (Nx_device.stats d) in
  let p = Nx_device.Profile.start () in
  match f () with
  | y ->
      let most =
        List.fold_left
          (fun most -> function
            | Nx_device.Profile.Allocation a when Nx_device.equal a.device d ->
                max most a.allocated
            | _ -> most)
          before
          (Nx_device.Profile.stop p)
      in
      (y, most - before)
  | exception e ->
      ignore (Nx_device.Profile.stop p);
      raise e

(* A routed product gathers each expert's codes and scales at its ids: compiled,
   the gather reads them in place, and an index it stores has the size of the
   ids. An index broadcast to the gathered codes, [16; 2; 64; 128] here, would
   take 2 MiB. *)
let memory =
  let d = Nx.Device.cpu 5 in
  let p = Nx.Placement.on [ d ] in
  group "memory"
    [
      test "compiled, a routed product holds its result and its ids' size"
        (fun () ->
          let w = Nx_quant.place p (weight ~scale:moderate [| 8; 64; 256 |]) in
          let ids =
            Nx.place p (ints [| 16; 2 |] (Array.init 32 (fun i -> i * 3 mod 9)))
          and x = Nx.place p (floats [| 16; 2; 1; 256 |]) in
          let f = routed_compiled w in
          (* The first call loads the program, whose code counts while [f]
             holds it. *)
          ignore (f ids x);
          let y, held = peak d (fun () -> f ids x) in
          at_least ~msg:"the result" ~than:(Nx.nbytes y) int held;
          at_most ~than:(Nx.nbytes y + Nx.nbytes ids) int held);
    ]

(* Rules *)

(* [dense ?ids w x] is the product as an ordinary matmul by the dequantised
   weight, its experts taken by [ids] and zero where an id names none: the
   function whose derivatives the rules must give. [w] has no lane axes. *)
let dense ?ids w x =
  let dq = Nx_quant.dequant Nx.float32 w in
  let w' =
    match ids with
    | None -> dq
    | Some ids ->
        let e = Nx.dim 0 dq in
        let named =
          Nx.logical_and
            (Nx.greater_equal ids (Nx.zeros_like ids))
            (Nx.less ids (Nx.full Nx.int64 (Nx.shape ids) (Int64.of_int e)))
        in
        let taken =
          Nx.reshape
            (Array.append (Nx.shape ids) (Array.sub (Nx.shape dq) 1 2))
            (Nx.take ~axis:0
               ~indices:(Nx.flatten (Nx.where named ids (Nx.zeros_like ids)))
               dq)
        in
        let mask = Nx.reshape (Array.append (Nx.shape ids) [| 1; 1 |]) named in
        Nx.where mask taken (Nx.zeros_like taken)
  in
  Nx.matmul x (Nx.matrix_transpose w')

(* [near expected actual] checks [actual] against [expected] within float32 sums
   in another order: each value within 2^-14 of [expected]'s largest magnitude,
   which bounds the terms of these products' sums. *)
let near ?msg expected actual =
  let largest = Nx.item [] (Nx.max (Nx.abs expected)) in
  equal ?msg
    (tensor (Nx_test.close ~abs:(Float.ldexp largest (-14)) ~rel:0. ()))
    expected actual

(* A loss weighting each value of a product differently, so that a gradient
   tells its positions apart. *)
let weighted y =
  let n = Nx.numel y in
  Nx.sum
    (Nx.mul y
       (Nx.reshape (Nx.shape y)
          (Nx.sin (Nx.arange_f Nx.float32 0. (float_of_int n) 1.))))

let rule_cases () =
  let w = weight ~scale:moderate [| 4; 8; 64 |] in
  [
    ("without ids", None, weight ~scale:moderate [| 5; 64 |], floats [| 3; 64 |]);
    ( "with ids",
      Some (ints [| 3; 2 |] [| 0; 3; -1; 2; 1; 1 |]),
      w,
      floats [| 3; 2; 1; 64 |] );
  ]

let rules =
  group "rules"
    [
      cases ~tags:[ "slow" ]
        ~name:(fun (n, _, _, _) -> n)
        "the gradient in x is the dense product's, eager and compiled"
        (rule_cases ())
        (fun (_, ids, w, x) ->
          let expected = Rune.grad' (fun x -> weighted (dense ?ids w x)) x in
          let f x = weighted (Nx_quant.apply ?ids w x) in
          near ~msg:"eager" expected (Rune.grad' f x);
          near ~msg:"compiled" expected (Rune.jit' (Rune.grad' f) x));
      test
        "the gradient through a sum over the positions is the dense product's"
        (fun () ->
          let w = weight ~scale:moderate [| 4; 8; 64 |] in
          let ids = ints [| 3; 4 |] [| 0; 3; -1; 2; 1; 1; 3; 0; 2; 4; 0; 1 |] in
          let x = floats [| 3; 1; 1; 64 |] in
          let mixed p x = weighted (Nx.sum ~axes:[ 1 ] (p x)) in
          let expected = Rune.grad' (mixed (dense ~ids w)) x in
          near ~msg:"eager" expected
            (Rune.grad' (mixed (Nx_quant.apply ~ids w)) x);
          near ~msg:"compiled" expected
            (Rune.jit' (Rune.grad' (mixed (Nx_quant.apply ~ids w))) x));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "the tangent in x is the dense product's, eager and compiled"
        (rule_cases ())
        (fun (_, ids, w, x) ->
          let t = floats (Nx.shape x) in
          let y, dy = Rune.jvp' (Nx_quant.apply ?ids w) x t in
          let y', dy' = Rune.jvp' (dense ?ids w) x t in
          near ~msg:"primal" y' y;
          near ~msg:"tangent" dy' dy;
          near ~msg:"compiled tangent" dy'
            (Rune.jit' (fun x -> snd (Rune.jvp' (Nx_quant.apply ?ids w) x t)) x));
      test "a map over x is each row's product, eager and compiled" (fun () ->
          let w = weight ~scale:moderate [| 4; 8; 64 |] in
          let ids = ints [| 2; 2 |] [| 0; 3; -1; 2 |] in
          let xs = floats [| 3; 2; 1; 1; 64 |] in
          let f = Nx_quant.apply ~ids w in
          let expected =
            Nx.stack (List.init 3 (fun i -> f (Nx.slice [ I i ] xs)))
          in
          near ~msg:"eager" expected (Rune.vmap' f xs);
          near ~msg:"compiled" expected (Rune.jit' (Rune.vmap' f) xs));
      test "a map over routes and rows is each one's product, compiled"
        (fun () ->
          let w = weight ~scale:moderate [| 4; 8; 64 |] in
          let ids =
            ints [| 3; 2; 2 |] [| 0; 3; -1; 2; 1; 1; 3; 0; 2; 4; 0; 1 |]
          in
          let xs = floats [| 3; 2; 1; 1; 64 |] in
          let f (ids, x) = Nx_quant.apply ~ids w x in
          let s = Nx.Ptree.(pair tensor tensor @-> returns tensor) in
          let expected =
            Nx.stack
              (List.init 3 (fun i ->
                   f (Nx.slice [ I i ] ids, Nx.slice [ I i ] xs)))
          in
          near expected (Rune.jit s (Rune.vmap s f) (ids, xs)));
      test "a map over weights is each weight's product, compiled" (fun () ->
          let ws = weight ~scale:moderate [| 3; 4; 8; 64 |] in
          let ids = ints [| 2 |] [| 0; 3 |] and x = floats [| 2; 1; 64 |] in
          let lane i =
            Nx.Ptree.map Nx_quant.ptree (fun _ t -> Nx.slice [ I i ] t) ws
          in
          let f w = Nx_quant.apply ~ids w x in
          let s = Nx.Ptree.(Nx_quant.ptree @-> returns tensor) in
          let expected = Nx.stack (List.init 3 (fun i -> f (lane i))) in
          near ~msg:"eager" expected (Rune.vmap s f ws);
          near ~msg:"compiled" expected (Rune.jit s (Rune.vmap s f) ws));
    ]

let empty =
  test "compiled, a product over no positions is empty" (fun () ->
      let w = weight [| 4; 8; 64 |] in
      let ids = ints [| 0; 2 |] [||] and x = floats [| 0; 1; 1; 64 |] in
      let r =
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> returns tensor)
          (fun ids x -> Nx_quant.apply ~ids w x)
          ids x
      in
      equal (array int) [| 0; 2; 1; 8 |] (Nx.shape r))

(* A weight built from a differentiated value takes no derivative: its parts are
   integers, whose casts carry none. *)
let undifferentiated =
  test "a weight built from a differentiated value contributes no derivative"
    (fun () ->
      let x = floats [| 2; 64 |] in
      let scales = Nx.full Nx.uint8 [| 5; 2 |] 127 in
      let built v = Nx_quant.mxfp4 ~scales (Nx.cast Nx.uint8 v) in
      let v = Nx.full Nx.float32 [| 5; 32 |] 17. in
      equal (tensor float_exact)
        (Nx.zeros Nx.float32 [| 5; 32 |])
        (Rune.grad' (fun v -> Nx.sum (Nx_quant.apply (built v) x)) v))

(* Metal *)

let metal =
  match Result.to_option (Nx.Device.get Metal) with
  | None -> slow "metal" (fun () -> skip ~reason:"no Metal device" ())
  | Some m ->
      let p =
        Nx.Placement.on [ m ]
      in
      cases
        ~name:(fun c -> c.name)
        "on Metal, a product is eager's" (products ~scale:moderate)
        (fun c ->
          let r =
            match c.ids with
            | None -> Rune.jit' (product c c.w) (Nx.place p c.x)
            | Some ids ->
                Rune.jit
                  Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                  (fun ids x -> product { c with ids = Some ids } c.w x)
                  (Nx.place p ids) (Nx.place p c.x)
          in
          agrees c (product c c.w c.x) (host r))

let () =
  exit
    (run "Rune.quant"
       [
         values;
         placements;
         memory;
         rules;
         empty;
         undifferentiated;
         group ~tags:[ "slow" ] "metal" [ metal ];
       ])
