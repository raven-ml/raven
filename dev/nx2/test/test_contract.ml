(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions: Nx.contract against a reference that reads the pattern's
   meaning off its names, within the stated bound; on the host, bit for bit
   against nx.cpu's stated summation order; einsum and matmul against
   contract. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let numel s = Array.fold_left ( * ) 1 s
let host dt s xs = Nx.Repr.of_array Nx.Host.v (A.of_array dt s xs)
let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

(* Rounding to float32, and float32's fused multiply-add: [a b] is exact in
   float64, and the sum rounded to odd in float64 then to float32 rounds once,
   since float64 holds two more bits than float32's significand. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let fma32 a b c =
  let p = a *. b in
  let s = p +. c in
  let e =
    let bb = s -. p in
    p -. (s -. bb) +. (c -. bb)
  in
  let odd =
    if e = 0. || Int64.logand (Int64.bits_of_float s) 1L = 1L then s
    else if e > 0. then Float.succ s
    else Float.pred s
  in
  f32 odd

(* The reference *)

(* A contraction by its meaning: [names] each with its extent, the result's
   names in order, the summed names in order, and each operand's names in
   order. *)
type meaning = {
  extent : string -> int;
  result : string list;
  summed : string list;
  a : string list;
  b : string list;
}

(* Each assignment of [names], in C order. *)
let assignments extent names =
  let rec go = function
    | [] -> [ [] ]
    | n :: rest ->
        let tails = go rest in
        List.concat_map
          (fun i -> List.map (fun t -> (n, i) :: t) tails)
          (List.init (extent n) Fun.id)
  in
  go names

let index extent names env =
  List.fold_left (fun acc n -> (acc * extent n) + List.assoc n env) 0 names

(* For each result index in C order, the list of its products' operand
   positions, in C order of the summed names. *)
let terms m =
  List.map
    (fun r ->
      List.map
        (fun s ->
          let env = r @ s in
          (index m.extent m.a env, index m.extent m.b env))
        (assignments m.extent m.summed))
    (assignments m.extent m.result)

(* The exact sum in float64 of float32 products, which is exact for operands
   whose products and partial sums fit 53 bits; with it, the bound [Σ|a||b|]. *)
let reference m xa xb =
  List.map
    (fun ts ->
      List.fold_left
        (fun (s, abs) (i, j) ->
          let p = xa.(i) *. xb.(j) in
          (s +. p, abs +. Float.abs p))
        (0., 0.) ts)
    (terms m)

let gamma n u = Float.of_int n *. u /. (1. -. (Float.of_int n *. u))

(* nx.cpu's float32 order (nx_cpu.mli): with at least 64 outputs per batch
   element a fused chain in order from [init]; with fewer, blocks of 1024, 16
   lanes each, the lanes' tree, the blocks' tree, then [init]. *)
let chain ~init products =
  List.fold_left (fun acc (x, y) -> fma32 x y acc) init products

let lanes products =
  let l = Array.make 16 0. in
  List.iteri (fun i (x, y) -> l.(i mod 16) <- fma32 x y l.(i mod 16)) products;
  List.iter
    (fun w ->
      for i = 0 to w - 1 do
        l.(i) <- f32 (l.(i) +. l.(i + w))
      done)
    [ 8; 4; 2; 1 ];
  l.(0)

let rec tree = function
  | [||] -> 0.
  | [| x |] -> x
  | xs ->
      let n = Array.length xs in
      let p = ref 1 in
      while !p * 2 < n do
        p := !p * 2
      done;
      f32 (tree (Array.sub xs 0 !p) +. tree (Array.sub xs !p (n - !p)))

let blocked ~init products =
  let a = Array.of_list products in
  let n = Array.length a in
  let blocks =
    Array.init
      ((n + 1023) / 1024)
      (fun k ->
        lanes
          (Array.to_list (Array.sub a (k * 1024) (min 1024 (n - (k * 1024))))))
  in
  f32 (tree blocks +. init)

let cpu_order ~outputs ~init m xa xb =
  List.mapi
    (fun o ts ->
      let products = List.map (fun (i, j) -> (xa.(i), xb.(j))) ts in
      let init = match init with Some x -> x.(o) | None -> 0. in
      if outputs >= 64 then chain ~init products else blocked ~init products)
    (terms m)

(* Cases *)

(* A pattern, its operands' shapes, and its meaning. *)
type case = {
  pattern : string;
  sa : int array;
  sb : int array;
  meaning : meaning;
  outputs : int;  (** Per batch element. *)
}

let extent_of l n = List.assoc n l

let case pattern ~a ~b ~result ~summed ~batch extents =
  let extent = extent_of extents in
  let shape names = Array.of_list (List.map extent names) in
  {
    pattern;
    sa = shape a;
    sb = shape b;
    meaning = { extent; result; summed; a; b };
    outputs =
      numel (shape (List.filter (fun n -> not (List.mem n batch)) result));
  }

let matmul_like m n k =
  case "i k, k j -> i j | k" ~a:[ "i"; "k" ] ~b:[ "k"; "j" ]
    ~result:[ "i"; "j" ] ~summed:[ "k" ] ~batch:[]
    [ ("i", m); ("j", n); ("k", k) ]

let cases_of_sizes =
  let open Gen in
  (let+ m = int_range 1 9
   and+ n = int_range 1 9
   and+ k = frequency [ (1, constant 0); (5, int_range 1 40) ]
   and+ which = int_range 0 5 in
   let c =
     match which with
     | 0 -> matmul_like m n k
     | 1 ->
         case "b i k, b j k -> b i j | k" ~a:[ "b"; "i"; "k" ]
           ~b:[ "b"; "j"; "k" ] ~result:[ "b"; "i"; "j" ] ~summed:[ "k" ]
           ~batch:[ "b" ]
           [ ("b", 2); ("i", m); ("j", n); ("k", k) ]
     | 2 ->
         case "k i, k j -> j i | k" ~a:[ "k"; "i" ] ~b:[ "k"; "j" ]
           ~result:[ "j"; "i" ] ~summed:[ "k" ] ~batch:[]
           [ ("i", m); ("j", n); ("k", k) ]
     | 3 ->
         case "i, j -> i j" ~a:[ "i" ] ~b:[ "j" ] ~result:[ "i"; "j" ]
           ~summed:[] ~batch:[]
           [ ("i", m); ("j", n) ]
     | 4 ->
         case "i j, i j -> | i j" ~a:[ "i"; "j" ] ~b:[ "i"; "j" ] ~result:[]
           ~summed:[ "i"; "j" ] ~batch:[]
           [ ("i", m); ("j", n + k) ]
     | _ ->
         case "h i k, k h j -> h j i | k" ~a:[ "h"; "i"; "k" ]
           ~b:[ "k"; "h"; "j" ] ~result:[ "h"; "j"; "i" ] ~summed:[ "k" ]
           ~batch:[ "h" ]
           [ ("h", 3); ("i", m); ("j", n); ("k", k) ]
   in
   c)
  |> with_pp (fun ppf c ->
      Format.fprintf ppf "%S a %a b %a" c.pattern pp_shape c.sa pp_shape c.sb)

(* Small integers times powers of two: every product and partial sum is exact in
   float64, so the reference is exact. *)
let operand n =
  Array.init n (fun _ ->
      Float.ldexp (Float.of_int (Random.int 33 - 16)) (Random.int 9 - 4))

(* Laws *)

let within_bound =
  prop "contract is within the bound of the exact sum" cases_of_sizes (fun c ->
      let k =
        numel (Array.of_list (List.map c.meaning.extent c.meaning.summed))
      in
      cover "no summed name" (c.meaning.summed = []);
      cover "an empty sum" (k = 0);
      let xa = operand (numel c.sa) and xb = operand (numel c.sb) in
      let a = host D.Float32 c.sa xa and b = host D.Float32 c.sb xb in
      let y = read (Nx.contract Nx.float32 (Nx.Pattern.v c.pattern) a b) in
      let u = Float.ldexp 1. (-24) in
      List.iteri
        (fun o (exact, abs) ->
          let bound = gamma (k + 1) (2. *. u) *. abs in
          less float_exact ~than:(bound +. Float.min_float)
            (Float.abs (y.(o) -. exact)))
        (reference c.meaning xa xb))

let bitwise =
  prop "on the host, contract sums in nx.cpu's stated order" cases_of_sizes
    (fun c ->
      cover "fewer than 64 outputs" (c.outputs < 64);
      cover "at least 64 outputs" (c.outputs >= 64);
      let xa =
        Array.map f32 (Array.init (numel c.sa) (fun _ -> Random.float 2. -. 1.))
      in
      let xb =
        Array.map f32 (Array.init (numel c.sb) (fun _ -> Random.float 2. -. 1.))
      in
      let a = host D.Float32 c.sa xa and b = host D.Float32 c.sb xb in
      let y = read (Nx.contract Nx.float32 (Nx.Pattern.v c.pattern) a b) in
      let want =
        Array.of_list (cpu_order ~outputs:c.outputs ~init:None c.meaning xa xb)
      in
      equal (array float_exact) want y)

let long_sum =
  test "a long sum takes blocks and lanes" (fun () ->
      let c = matmul_like 2 3 3000 in
      let xa =
        Array.map f32 (Array.init (numel c.sa) (fun _ -> Random.float 2. -. 1.))
      in
      let xb =
        Array.map f32 (Array.init (numel c.sb) (fun _ -> Random.float 2. -. 1.))
      in
      let init =
        Array.map f32 (Array.init 6 (fun _ -> Random.float 2. -. 1.))
      in
      let y =
        read
          (Nx.contract Nx.float32
             ~init:(host D.Float32 [| 2; 3 |] init)
             (Nx.Pattern.v c.pattern) (host D.Float32 c.sa xa)
             (host D.Float32 c.sb xb))
      in
      equal (array float_exact)
        (Array.of_list (cpu_order ~outputs:6 ~init:(Some init) c.meaning xa xb))
        y)

let einsum_is_contract =
  prop "einsum p a b is contract (dtype a) p a b" cases_of_sizes (fun c ->
      let a = host D.Float32 c.sa (operand (numel c.sa)) in
      let b = host D.Float32 c.sb (operand (numel c.sb)) in
      let p = Nx.Pattern.v c.pattern in
      equal (array float_exact)
        (read (Nx.contract Nx.float32 p a b))
        (read (Nx.einsum p a b)))

(* Groups, units and ellipses lower to the same contraction as the names they
   stand for. *)
let lowering =
  let x s = host D.Float32 s (operand (numel s)) in
  let same name ~p ~a ~b ~q ~a' ~b' =
    test name (fun () ->
        let a0 = x a and b0 = x b in
        let y = read (Nx.contract Nx.float32 (Nx.Pattern.v p) a0 b0) in
        let y' =
          read
            (Nx.contract Nx.float32 (Nx.Pattern.v q) (Nx.reshape a' a0)
               (Nx.reshape b' b0))
        in
        equal (array float_exact) y' y)
  in
  group "lowering"
    [
      same "a group splits an axis, its extent from the other operand"
        ~p:"b (kv g) i d, b kv j d -> b (kv g) i j | d" ~a:[| 2; 6; 3; 4 |]
        ~b:[| 2; 2; 5; 4 |] ~q:"b kv g i d, b kv j d -> b kv g i j | d"
        ~a':[| 2; 2; 3; 3; 4 |] ~b':[| 2; 2; 5; 4 |];
      same "an ellipsis is batch axes" ~p:"... i k, ... k j -> ... i j | k"
        ~a:[| 2; 3; 4; 5 |] ~b:[| 2; 3; 5; 6 |]
        ~q:"x y i k, x y k j -> x y i j | k" ~a':[| 2; 3; 4; 5 |]
        ~b':[| 2; 3; 5; 6 |];
      same "a unit axis drops from an operand" ~p:"i 1 k, k j -> i j | k"
        ~a:[| 3; 1; 4 |] ~b:[| 4; 2 |] ~q:"i k, k j -> i j | k" ~a':[| 3; 4 |]
        ~b':[| 4; 2 |];
      test "a unit axis and a group in the result" (fun () ->
          let a = x [| 3; 4 |] and b = x [| 4; 2 |] in
          let y =
            Nx.contract Nx.float32 (Nx.Pattern.v "i k, k j -> 1 (i j) | k") a b
          in
          equal (array int) [| 1; 6 |] (Nx.shape y);
          equal (array float_exact)
            (read
               (Nx.contract Nx.float32 (Nx.Pattern.v "i k, k j -> i j | k") a b))
            (read y));
      test "sizes give a group's inner extent" (fun () ->
          let a = x [| 12; 5 |] and b = x [| 5; 3 |] in
          let y =
            Nx.contract Nx.float32
              ~sizes:[ ("h", 4) ]
              (Nx.Pattern.v "(h d) k, k j -> h d j | k")
              a b
          in
          equal (array int) [| 4; 3; 3 |] (Nx.shape y));
    ]

(* Dtypes *)

let dtypes =
  group "dtypes"
    [
      test "bfloat16 operands accumulate in float32 and round once" (fun () ->
          (* 1 + 2^-8 + 2^-8: each addition rounds away in bfloat16, and the
             float32 sum 1 + 2^-7 is a bfloat16. *)
          let a =
            host D.Bfloat16 [| 1; 3 |]
              [| 1.; Float.ldexp 1. (-8); Float.ldexp 1. (-8) |]
          in
          let b = host D.Bfloat16 [| 3; 1 |] [| 1.; 1.; 1. |] in
          let y =
            Nx.contract Nx.bfloat16 (Nx.Pattern.v "i k, k j -> i j | k") a b
          in
          equal (array float_exact) [| 1. +. Float.ldexp 1. (-7) |] (read y));
      test "float64 accumulates in float64" (fun () ->
          let a =
            host D.Float64 [| 3 |]
              [| 1.; Float.ldexp 1. (-40); Float.ldexp 1. (-40) |]
          in
          let b = host D.Float64 [| 3 |] [| 1.; 1.; 1. |] in
          equal (array float_exact)
            [| 1. +. Float.ldexp 1. (-39) |]
            (read (Nx.contract Nx.float64 (Nx.Pattern.v "k, k -> | k") a b)));
      test "integers wrap in their dtype" (fun () ->
          let a =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Int8 [| 2 |] [| 100; 100 |])
          in
          let b =
            Nx.Repr.of_array Nx.Host.v (A.of_array D.Int8 [| 2 |] [| 1; 1 |])
          in
          equal (array int) [| -56 |]
            (read (Nx.contract Nx.int8 (Nx.Pattern.v "k, k -> | k") a b)));
      test "int8 operands into int32" (fun () ->
          let a =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Int8 [| 2 |] [| 100; 100 |])
          in
          let b =
            Nx.Repr.of_array Nx.Host.v (A.of_array D.Int8 [| 2 |] [| 1; 1 |])
          in
          equal (array int32) [| 200l |]
            (read (Nx.contract Nx.int32 (Nx.Pattern.v "k, k -> | k") a b)));
    ]

(* matmul *)

let matmul =
  let x s = host D.Float32 s (operand (numel s)) in
  let shape_of name sa sb want =
    test name (fun () ->
        equal (array int) want (Nx.shape (Nx.matmul (x sa) (x sb))))
  in
  group "matmul"
    [
      shape_of "two matrices" [| 2; 3 |] [| 3; 4 |] [| 2; 4 |];
      shape_of "leading axes broadcast" [| 5; 1; 2; 3 |] [| 4; 3; 6 |]
        [| 5; 4; 2; 6 |];
      shape_of "a leading 0 broadcasts against 1" [| 0; 2; 3 |] [| 1; 3; 4 |]
        [| 0; 2; 4 |];
      shape_of "a leading 0 against 1 on the right" [| 1; 2; 3 |]
        [| 0; 3; 4 |] [| 0; 2; 4 |];
      shape_of "a leading 0 beside a missing axis" [| 0; 2; 3 |] [| 3; 4 |]
        [| 0; 2; 4 |];
      shape_of "a 1-d left operand is a row" [| 3 |] [| 2; 3; 4 |] [| 2; 4 |];
      shape_of "a 1-d right operand is a column" [| 2; 2; 3 |] [| 3 |]
        [| 2; 2 |];
      shape_of "two 1-d operands give a 0-d product" [| 3 |] [| 3 |] [||];
      test "matmul is the batched contraction" (fun () ->
          let a = x [| 2; 3; 4 |] and b = x [| 4; 5 |] in
          equal (array float_exact)
            (read
               (Nx.contract Nx.float32
                  (Nx.Pattern.v "x i k, k j -> x i j | k")
                  a b))
            (read (Nx.matmul a b)));
      test "a 0-d operand raises" (fun () ->
          equal string
            "Nx.matmul: an operand is 0-d; float32 [] and float32 [3]"
            (message (fun () -> Nx.matmul (x [||]) (x [| 3 |]))));
      test "a boolean dtype raises" (fun () ->
          let t =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Bool [| 2; 2 |] [| true; false; true; true |])
          in
          equal string "Nx.matmul: the result's dtype is bool"
            (message (fun () -> Nx.matmul t t)));
      test "leading axes that do not broadcast raise" (fun () ->
          equal string
            "Nx.matmul: the leading axes do not broadcast; float32 [2; 2; 3] \
             and float32 [3; 3; 4]"
            (message (fun () ->
                 Nx.matmul (x [| 2; 2; 3 |]) (x [| 3; 3; 4 |]))));
      test "inner extents that differ raise" (fun () ->
          equal string
            "Nx.matmul: inner extents 3 and 4 differ; float32 [2; 3] and \
             float32 [4; 2]"
            (message (fun () -> Nx.matmul (x [| 2; 3 |]) (x [| 4; 2 |]))));
    ]

(* Placements *)

module S2 = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

let placements =
  let x s = host D.Float32 s (operand (numel s)) in
  let p = Nx.Pattern.v "i k, k j -> i j | k" in
  group "placements"
    [
      test "operands on every device give the host's result" (fun () ->
          let a = x [| 4; 3 |] and b = x [| 3; 2 |] in
          let y =
            Nx.contract Nx.float32 p (Nx.place S2.on a) (Nx.place S2.on b)
          in
          equal (array float_exact)
            (read (Nx.contract Nx.float32 p a b))
            (read y));
      test "operands split over two devices give the host's result" (fun () ->
          let a = x [| 4; 3 |] and b = x [| 3; 2 |] in
          let a' = Nx.place (S2.split ~axis:0) a in
          let b' = Nx.place (S2.split ~axis:1) b in
          equal (array float_exact)
            (read (Nx.contract Nx.float32 p a b))
            (read (Nx.contract Nx.float32 p a' b')));
    ]

(* Errors *)

let errors =
  let x s = host D.Float32 s (operand (numel s)) in
  let refuses name want f =
    test name (fun () -> equal string want (message f))
  in
  group "errors"
    [
      refuses "paired extents differ"
        {|Nx.contract: "i k, k j -> i j | k": k is 3 in a and 4 in b|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 4; 2 |]));
      refuses "a rank the pattern does not give"
        {|Nx.contract: "i k, k j -> i j | k" names 2 axes for a; a is float32 [2; 3; 4]|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3; 4 |])
            (x [| 4; 2 |]));
      refuses "a group that does not divide"
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) is 10 in a, which 4 does not divide|}
        (fun () ->
          Nx.contract Nx.float32
            ~sizes:[ ("h", 4) ]
            (Nx.Pattern.v "(h d) k, k j -> h d j | k")
            (x [| 10; 5 |])
            (x [| 5; 3 |]));
      refuses "a group with two unknowns"
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) in a leaves h and d unknown; give one in ~sizes|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "(h d) k, k j -> h d j | k")
            (x [| 12; 5 |])
            (x [| 5; 3 |]));
      refuses "a unit axis of another extent"
        {|Nx.contract: "i 1 k, k j -> i j | k": axis 1 of a is 1 in the pattern and 2 in a|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "i 1 k, k j -> i j | k")
            (x [| 3; 2; 4 |])
            (x [| 4; 2 |]));
      refuses "a size the pattern lacks"
        {|Nx.contract: "i k, k j -> i j | k": ~sizes names z, which it lacks|}
        (fun () ->
          Nx.contract Nx.float32
            ~sizes:[ ("z", 2) ]
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
      refuses "a one-operand pattern"
        {|Nx.contract: "i k -> k i" has one operand|} (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "i k -> k i")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
      refuses "an init of another shape"
        "Nx.contract: ~init is [3; 2] where the result is [2; 2]" (fun () ->
          Nx.contract Nx.float32
            ~init:(x [| 3; 2 |])
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
      refuses "a narrow accumulator"
        "Nx.contract: ~acc float16 is narrower than float32" (fun () ->
          Nx.contract Nx.float32 ~acc:Nx.float16
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
      refuses "an accumulator of another kind"
        "Nx.contract: ~acc int32 is of another kind than float32" (fun () ->
          Nx.contract Nx.float32 ~acc:Nx.int32
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
      refuses "an integer and a float operand without acc"
        "Nx.contract: int8 and float32 operands need ~acc" (fun () ->
          let i =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Int8 [| 2; 3 |] (Array.make 6 1))
          in
          Nx.contract Nx.float32
            (Nx.Pattern.v "i k, k j -> i j | k")
            i
            (x [| 3; 2 |]));
      refuses "a boolean result" "Nx.contract: the result's dtype is bool"
        (fun () ->
          Nx.contract Nx.bool
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
    ]

(* Init *)

let init =
  let a () = host D.Float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let b () = host D.Float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
  let c () = host D.Float32 [| 2; 2 |] [| 0.5; 0.25; 0.125; 2. |] in
  let p = Nx.Pattern.v "i k, k j -> i j | k" in
  let expected = [| 19.5; 22.25; 43.125; 52. |] in
  group "init"
    [
      test "the sum starts from init" (fun () ->
          equal (array float_exact) expected
            (read (Nx.contract ~init:(c ()) Nx.float32 p (a ()) (b ()))));
      test "a donated init is consumed" (fun () ->
          let c = c () in
          let y = Nx.contract ~init:(Nx.donate c) Nx.float32 p (a ()) (b ()) in
          equal (array float_exact) expected (read y);
          raises_match (Exn.invalid_arg ~substring:"was donated to Nx.contract")
            (fun () -> Nx.copy c));
    ]

let () =
  exit
    (run "nx contractions"
       [
         within_bound;
         bitwise;
         long_sum;
         einsum_is_contract;
         lowering;
         init;
         dtypes;
         matmul;
         placements;
         errors;
       ])
