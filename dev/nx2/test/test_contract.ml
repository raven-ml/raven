(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions: Nx.contract against a reference that reads the pattern's
   meaning off its names, within the stated bound; on the host, bit for bit
   against nx.cpu's stated summation order; einsum of two operands and matmul
   against contract, and of more against a sum over every index tuple. *)

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
  (* Extents to 12 make at least 64 outputs about one case in five (35 of
     144 pairs), so the order law reaches both of nx.cpu's orders in every
     run; to 9 it was 4 of 81, which 100 cases missed about once in 70 runs. *)
  (let+ m = int_range 1 12
   and+ n = int_range 1 12
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

(* Complex operands whose parts are {!operand}'s: each part of every product
   and partial sum is exact in float64. *)
let complex_within_bound =
  prop "a complex contract is within the bound of the exact sum"
    cases_of_sizes (fun c ->
      let k =
        numel (Array.of_list (List.map c.meaning.extent c.meaning.summed))
      in
      let draw n =
        let re = operand n and im = operand n in
        Array.init n (fun i -> { Complex.re = re.(i); im = im.(i) })
      in
      let xa = draw (numel c.sa) and xb = draw (numel c.sb) in
      let a = host D.Complex64 c.sa xa and b = host D.Complex64 c.sb xb in
      let y = read (Nx.contract Nx.complex64 (Nx.Pattern.v c.pattern) a b) in
      let u = Float.ldexp 1. (-24) in
      List.iteri
        (fun o ts ->
          let exact, abs =
            List.fold_left
              (fun (s, abs) (i, j) ->
                (Complex.add s (Complex.mul xa.(i) xb.(j)),
                 abs +. (Complex.norm xa.(i) *. Complex.norm xb.(j))))
              (Complex.zero, 0.) ts
          in
          let bound = gamma (k + 1) (2. *. u) *. abs in
          less float_exact ~than:(bound +. Float.min_float)
            (Complex.norm (Complex.sub y.(o) exact)))
        (terms c.meaning))

(* A result of another dtype than its accumulator is the contraction into the
   accumulator, cast once: bit for bit, from operands whose products round. *)
let widened =
  prop "a narrower result is the cast of the contraction into acc"
    cases_of_sizes (fun c ->
      let p = Nx.Pattern.v c.pattern in
      let draw n =
        Array.init n (fun _ ->
            { Complex.re = Random.float 2. -. 1.; im = Random.float 2. -. 1. })
      in
      let a = host D.Complex64 c.sa (draw (numel c.sa)) in
      let b = host D.Complex64 c.sb (draw (numel c.sb)) in
      let bits y =
        Array.concat
          (List.map
             (fun z ->
               [| Int64.bits_of_float z.Complex.re; Int64.bits_of_float z.im |])
             (Array.to_list (read y)))
      in
      equal (array int64)
        (bits
           (Nx.cast Nx.complex64
              (Nx.contract ~acc:D.Complex128 Nx.complex128 p a b)))
        (bits (Nx.contract ~acc:D.Complex128 Nx.complex64 p a b));
      let int8 n = Array.init n (fun _ -> Random.int 256 - 128) in
      let a = host D.Int8 c.sa (int8 (numel c.sa)) in
      let b = host D.Int8 c.sb (int8 (numel c.sb)) in
      equal (array int)
        (read (Nx.cast Nx.int8 (Nx.contract Nx.int32 p a b)))
        (read (Nx.contract ~acc:D.Int32 Nx.int8 p a b)))

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
  prop "einsum p [a; b] is contract (dtype a) p a b" cases_of_sizes (fun c ->
      let a = host D.Float32 c.sa (operand (numel c.sa)) in
      let b = host D.Float32 c.sb (operand (numel c.sb)) in
      let p = Nx.Pattern.v c.pattern in
      equal (array float_exact)
        (read (Nx.contract Nx.float32 p a b))
        (read (Nx.einsum p [ a; b ])))

(* Einsum *)

(* An axis of a generated layout: a name, two names grouped, or [1]. *)
type axis = N of string | G of string * string | U

let axis_text = function
  | N n -> n
  | G (a, b) -> "(" ^ a ^ " " ^ b ^ ")"
  | U -> "1"

let axis_names = function N n -> [ n ] | G (a, b) -> [ a; b ] | U -> []

(* A pattern of several operands with its operands' shapes, and its meaning:
   each operand's names and the result's in C order, those [...] covers
   first, and the summed names. *)
type problem = {
  text : string;
  shapes : int array list;
  extents : (string * int) list;
  ops : string list list;
  out : string list;
  sums : string list;
}

let pp_problem ppf p =
  Format.fprintf ppf "%S %a" p.text
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_shape)
    p.shapes

(* Two to five operands; free names, each held by one operand or more, and
   summed names, each by two or more; extents from 0 to 3; at times [...]
   over one or two leading axes, a group in an operand where its extents can
   be learned, and a group and a [1] in the result. *)
let problems =
  let open Gen in
  let extent =
    frequency [ (1, constant 0); (2, constant 1); (6, int_range 2 3) ]
  in
  let* n = int_range 2 5 in
  let ops = List.init n Fun.id in
  let held least =
    such_that (fun h -> List.length h >= least) (subsequence ops)
  in
  let* frees = list ~size:(int_range 0 3) (pair extent (held 1)) in
  let* sums = list ~size:(int_range 0 3) (pair extent (held 2)) in
  let* dots =
    frequency [ (2, constant []); (1, list ~size:(int_range 1 2) extent) ]
  in
  let named prefix l =
    List.mapi (fun i (e, h) -> (prefix ^ string_of_int i, e, h)) l
  in
  let frees = named "f" frees and sums = named "s" sums in
  let dots = List.mapi (fun i e -> ("e" ^ string_of_int i, e)) dots in
  let extents = dots @ List.map (fun (nm, e, _) -> (nm, e)) (frees @ sums) in
  let ext nm = List.assoc nm extents in
  let holders nm =
    List.find_map
      (fun (nm', _, h) -> if nm = nm' then Some h else None)
      (frees @ sums)
    |> Option.get
  in
  let* layouts =
    List.fold_right
      (fun i acc ->
        let* rest = acc in
        let* l =
          permutation
            (List.filter_map
               (fun (nm, _, h) -> if List.mem i h then Some nm else None)
               (frees @ sums))
        in
        constant (l :: rest))
      ops (constant [])
  in
  let* grouped = int_range (-1) (n - 1) in
  let* unit = bool in
  let* result = permutation (List.map (fun (nm, _, _) -> nm) frees) in
  let* result_group, result_unit = pair bool bool in
  (* A group in operand [i] whose extents another operand gives, or whose
     known extent is not 0 and divides out the other. *)
  let learned i nm = List.exists (fun j -> j <> i) (holders nm) in
  let group i = function
    | a :: b :: rest
      when (learned i a && learned i b)
           || (learned i a && ext a <> 0)
           || (learned i b && ext b <> 0) ->
        G (a, b) :: List.map (fun nm -> N nm) rest
    | l -> List.map (fun nm -> N nm) l
  in
  let layouts =
    List.mapi
      (fun i l ->
        let l =
          if i = grouped then group i l else List.map (fun nm -> N nm) l
        in
        if i = 0 && unit then l @ [ U ] else l)
      layouts
  in
  let result =
    match result with
    | a :: b :: rest when result_group ->
        G (a, b) :: List.map (fun nm -> N nm) rest
    | l -> List.map (fun nm -> N nm) l
  in
  let result = if result_unit then result @ [ U ] else result in
  let dot_names = List.map fst dots in
  let text l =
    String.concat " "
      ((if dots = [] then [] else [ "..." ]) @ List.map axis_text l)
  in
  let shape l =
    Array.of_list
      (List.map snd dots
      @ List.map
          (function N nm -> ext nm | G (a, b) -> ext a * ext b | U -> 1)
          l)
  in
  let flat l = dot_names @ List.concat_map axis_names l in
  let sum_names = List.map (fun (nm, _, _) -> nm) sums in
  constant
    {
      text =
        (String.concat ", " (List.map text layouts)
        ^ " -> " ^ text result
        ^ if sum_names = [] then "" else " | " ^ String.concat " " sum_names);
      shapes = List.map shape layouts;
      extents;
      ops = List.map flat layouts;
      out = flat result;
      sums = sum_names;
    }

(* The exact sum, in float64, of the products of [xs]'s elements at every
   index tuple for each result index in C order, with the sum of their
   magnitudes: exact while every product and partial sum fits 53 bits. *)
let sum_over_tuples p xs =
  let extent nm = List.assoc nm p.extents in
  List.map
    (fun r ->
      List.fold_left
        (fun (s, abs) t ->
          let env = r @ t in
          let x =
            List.fold_left2
              (fun x names xs -> x *. xs.(index extent names env))
              1. p.ops xs
          in
          (s +. x, abs +. Float.abs x))
        (0., 0.)
        (assignments extent p.sums))
    (assignments extent p.out)

(* The least work of a tree of contractions of [p]'s operands, and the
   greatest [N = Σ (K + 1)] among the trees doing it, by einsum's definitions:
   a contraction does the product of the extents of the names its parts'
   values hold, a value holds the names its operands hold that the result or
   an operand outside it needs, and a contraction sums [K] products, the
   product of the extents of the names it meets and its value drops. Every
   tree is drawn, over the few operands and names [problems] draws, all
   within the layout's ranks. *)
let least_work p =
  let ext nm = List.assoc nm p.extents in
  let ops = Array.of_list p.ops in
  let names =
    List.fold_left
      (fun acc nm -> if List.mem nm acc then acc else acc @ [ nm ])
      [] (List.concat p.ops)
  in
  let holds s nm = List.exists (fun i -> List.mem nm ops.(i)) s in
  let outside s nm =
    List.exists
      (fun i -> (not (List.mem i s)) && List.mem nm ops.(i))
      (List.init (Array.length ops) Fun.id)
  in
  let live s =
    List.filter
      (fun nm -> holds s nm && (List.mem nm p.out || outside s nm))
      names
  in
  let product = List.fold_left (fun k nm -> k * ext nm) 1 in
  let rec subsets = function
    | [] -> [ [] ]
    | x :: rest ->
        let s = subsets rest in
        s @ List.map (fun l -> x :: l) s
  in
  (* Each tree of the operands [s] as its work and [N]. *)
  let rec trees = function
    | [] | [ _ ] -> [ (0, 0) ]
    | first :: rest as s ->
        List.concat_map
          (fun sub ->
            let t = first :: sub in
            let u = List.filter (fun i -> not (List.mem i t)) s in
            if u = [] then []
            else
              let met =
                live t
                @ List.filter (fun nm -> not (List.mem nm (live t))) (live u)
              in
              let summed =
                List.filter (fun nm -> not (List.mem nm (live s))) met
              in
              List.concat_map
                (fun (wt, nt) ->
                  List.map
                    (fun (wu, nu) ->
                      (wt + wu + product met, nt + nu + product summed + 1))
                    (trees u))
                (trees t))
          (subsets rest)
  in
  let all = trees (List.init (Array.length ops) Fun.id) in
  let least = List.fold_left (fun w (w', _) -> min w w') max_int all in
  ( least,
    List.fold_left (fun n (w, n') -> if w = least then max n n' else n) 0 all )

type float_dtype = F : (float, 's) D.t -> float_dtype

let float_dtypes =
  Gen.of_list
    ~pp:(fun ppf (F dt) -> Format.pp_print_string ppf (D.name dt))
    [
      F D.Float64;
      F D.Float32;
      F D.Float16;
      F D.Bfloat16;
      F D.Float8_e4m3fn;
      F D.Float8_e5m2;
      F D.Float4_e2m1fn;
    ]

(* Small integers times small powers of two, held exactly in float64 through
   five products and their sums, drawn from [seed]. *)
let small seed n =
  let r = Random.State.make [| seed; n |] in
  Array.init n (fun _ ->
      Float.ldexp
        (Float.of_int (Random.State.int r 9 - 4))
        (Random.State.int r 5 - 2))

let einsum_within_bound =
  prop "einsum is within the bound of the sum over every index tuple"
    (Gen.triple (Gen.with_pp pp_problem problems) float_dtypes Gen.int)
    (fun (p, F dt, seed) ->
      let n = List.length p.ops in
      let holders nm = List.length (List.filter (List.mem nm) p.ops) in
      cover "five operands" (n = 5);
      cover "a name summed across three operands"
        (List.exists (fun nm -> holders nm >= 3) p.sums);
      cover "a batch name" (List.exists (fun nm -> holders nm >= 2) p.out);
      cover "a group" (String.contains p.text '(');
      cover "an ellipsis" (String.contains p.text '.');
      cover "a zero extent" (List.exists (fun (_, e) -> e = 0) p.extents);
      cover "a one-element axis" (List.exists (fun (_, e) -> e = 1) p.extents);
      let xs =
        List.mapi (fun i s -> host dt s (small (seed + i) (numel s))) p.shapes
      in
      let y = read (Nx.einsum (Nx.Pattern.v p.text) xs) in
      let _, big_n = least_work p in
      let u =
        match dt with
        | D.Float64 -> Float.ldexp 1. (-53)
        | _ -> Float.ldexp 1. (-24)
      in
      let g = gamma big_n (2. *. u) in
      List.iteri
        (fun o (exact, abs) ->
          (* Rounding is monotone: the result lies between the roundings of
             the bound's ends. *)
          let b = g *. abs in
          let lo = D.of_float dt (exact -. b)
          and hi = D.of_float dt (exact +. b) in
          at_most (float Float.min_float) ~than:hi y.(o);
          at_least (float Float.min_float) ~than:lo y.(o))
        (sum_over_tuples p (List.map read xs)))

let einsum_deterministic =
  prop "einsum's bits are the same whatever the order of independent calls"
    (Gen.pair
       (Gen.list ~size:(Gen.int_range 2 4) (Gen.with_pp pp_problem problems))
       Gen.int)
    (fun (ps, seed) ->
      let r = Random.State.make [| seed |] in
      let calls =
        List.map
          (fun p ->
            let xs =
              List.map
                (fun s ->
                  host D.Float32 s
                    (Array.init (numel s) (fun _ ->
                         Random.State.float r 2. -. 1.)))
                p.shapes
            in
            fun () -> read (Nx.einsum (Nx.Pattern.v p.text) xs))
          ps
      in
      let forward = List.map (fun f -> f ()) calls in
      let backward = List.rev (List.map (fun f -> f ()) (List.rev calls)) in
      List.iter2 (equal (array float_exact)) forward backward)

let einsum_order =
  let random dt s =
    host dt s (Array.init (numel s) (fun _ -> Random.float 2. -. 1.))
  in
  group "einsum's order"
    [
      test "a matrix, a matrix and a vector contract the vector first"
        (fun () ->
          (* (a b) v does 3·4·5 + 3·5 multiply-adds, a (b v) 4·5 + 3·4. The
             intermediate stays in float32 and the result rounds once. *)
          let a = random D.Bfloat16 [| 3; 4 |]
          and b = random D.Bfloat16 [| 4; 5 |]
          and v = random D.Bfloat16 [| 5 |] in
          let bv =
            Nx.contract Nx.float32 (Nx.Pattern.v "j k, k -> j | k") b v
          in
          equal (array float_exact)
            (read
               (Nx.contract Nx.bfloat16 (Nx.Pattern.v "i j, j -> i | j") a bv))
            (read
               (Nx.einsum (Nx.Pattern.v "i j, j k, k -> i | j k") [ a; b; v ])));
      test "sizes give a group's inner extent" (fun () ->
          let a = random D.Float32 [| 12; 5 |]
          and b = random D.Float32 [| 5; 3 |]
          and v = random D.Float32 [| 3 |] in
          let y =
            Nx.einsum
              ~sizes:[ ("h", 4) ]
              (Nx.Pattern.v "(h d) k, k j, j -> h d | k j")
              [ a; b; v ]
          in
          equal (array int) [| 4; 3 |] (Nx.shape y);
          equal (array float_exact)
            (read
               (Nx.einsum
                  (Nx.Pattern.v "h d k, k j, j -> h d | k j")
                  [ Nx.reshape [| 4; 3; 5 |] a; b; v ]))
            (read y));
      test "with equal work the operands contract from the left" (fun () ->
          let x () = random D.Float32 [| 4; 4 |] in
          let x1 = x () and x2 = x () and x3 = x () in
          let x12 =
            Nx.contract Nx.float32 (Nx.Pattern.v "i j, j k -> i k | j") x1 x2
          in
          equal (array float_exact)
            (read
               (Nx.contract Nx.float32
                  (Nx.Pattern.v "i k, k l -> i l | k")
                  x12 x3))
            (read
               (Nx.einsum
                  (Nx.Pattern.v "i j, j k, k l -> i l | j k")
                  [ x1; x2; x3 ])));
    ]

let einsum_errors =
  let x s = host D.Float32 s (operand (numel s)) in
  let refuses name want f =
    test name (fun () -> equal string want (message f))
  in
  let chain = Nx.Pattern.v "i j, j k, k -> i | j k" in
  group "einsum's errors"
    [
      refuses "a one-operand pattern"
        {|Nx.einsum: "i j -> j i" has one operand; use Nx.rearrange|} (fun () ->
          Nx.einsum (Nx.Pattern.v "i j -> j i") [ x [| 2; 3 |] ]);
      refuses "fewer values than operands"
        {|Nx.einsum: "i j, j k, k -> i | j k" has 3 operands but the list holds 2|}
        (fun () -> Nx.einsum chain [ x [| 2; 3 |]; x [| 3; 4 |] ]);
      refuses "more values than operands"
        {|Nx.einsum: "i j, j k -> i k | j" has 2 operands but the list holds 3|}
        (fun () ->
          Nx.einsum
            (Nx.Pattern.v "i j, j k -> i k | j")
            [ x [| 2; 3 |]; x [| 3; 4 |]; x [| 4 |] ]);
      refuses "one value"
        {|Nx.einsum: "i j, j k -> i k | j" has 2 operands but the list holds 1|}
        (fun () ->
          Nx.einsum (Nx.Pattern.v "i j, j k -> i k | j") [ x [| 2; 3 |] ]);
      refuses "no value"
        {|Nx.einsum: "i j, j k -> i k | j" has 2 operands but the list holds 0|}
        (fun () -> Nx.einsum (Nx.Pattern.v "i j, j k -> i k | j") []);
      refuses "extents that differ between operands"
        {|Nx.einsum: "i j, j k, k -> i | j k": k is 4 in x2 and 5 in x3|}
        (fun () -> Nx.einsum chain [ x [| 2; 3 |]; x [| 3; 4 |]; x [| 5 |] ]);
      refuses "a rank the pattern does not give"
        {|Nx.einsum: "i j, j k, k -> i | j k" names 1 axis for x3; x3 is float32 [4; 1]|}
        (fun () -> Nx.einsum chain [ x [| 2; 3 |]; x [| 3; 4 |]; x [| 4; 1 |] ]);
      refuses "boolean operands" "Nx.einsum: the operands are booleans"
        (fun () ->
          let t =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Bool [| 2; 2 |] [| true; false; true; true |])
          in
          Nx.einsum (Nx.Pattern.v "i j, j k -> i k | j") [ t; t ]);
      refuses "a narrow accumulator"
        "Nx.einsum: ~acc float16 cannot accumulate (no bool, no float narrower \
         than float32)" (fun () ->
          Nx.einsum ~acc:Nx.float16 chain
            [ x [| 2; 3 |]; x [| 3; 4 |]; x [| 4 |] ]);
      refuses "more operands than einsum plans"
        {|Nx.einsum: "a, a, a, a, a, a, a, a, a, a, a, a -> a" has 12 operands; einsum plans at most 11: contract parts in separate einsums|}
        (fun () ->
          Nx.einsum
            (Nx.Pattern.v
               (String.concat ", " (List.init 12 (fun _ -> "a")) ^ " -> a"))
            (List.init 12 (fun _ -> x [| 2 |])));
      refuses "an ellipsis over different numbers of axes"
        {|Nx.einsum: "... i, ... i, ... i -> ... | i": ... is 1 axis in x1 and 2 in x3|}
        (fun () ->
          Nx.einsum
            (Nx.Pattern.v "... i, ... i, ... i -> ... | i")
            [ x [| 2; 3 |]; x [| 2; 3 |]; x [| 2; 2; 3 |] ]);
      test "a pattern whose every order holds a value past the rank raises"
        (fun () ->
          (* Four operands, each pair sharing ten summed names: an operand
             has 30 axes, and the first contraction of any order keeps the
             40 names its two operands share with the other two. *)
          let pairs = [ (0, 1); (0, 2); (0, 3); (1, 2); (1, 3); (2, 3) ] in
          let names (i, j) =
            List.init 10 (fun k -> Printf.sprintf "p%d%d_%d" i j k)
          in
          let layout o =
            String.concat " "
              (List.concat_map
                 (fun (i, j) -> if i = o || j = o then names (i, j) else [])
                 pairs)
          in
          let text =
            String.concat ", " (List.init 4 layout)
            ^ " -> | "
            ^ String.concat " " (List.concat_map names pairs)
          in
          equal string
            (Printf.sprintf
               "Nx.einsum: %S: every order holds a value of more than 32 axes"
               text)
            (message (fun () ->
                 Nx.einsum (Nx.Pattern.v text)
                   (List.init 4 (fun _ -> x (Array.make 30 1))))));
      refuses "contract with three operands"
        {|Nx.contract: "i j, j k, k -> i | j k" has 3 operands; use Nx.einsum|}
        (fun () -> Nx.contract Nx.float32 chain (x [| 2; 3 |]) (x [| 3; 4 |]));
    ]

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

(* Operand shapes for matmul: leading axes where each operand holds the
   result's extent, holds 1 or lacks the axis, 0 among the extents, and 1-d
   operands, which have no leading axis. *)
let matmul_shapes =
  let open Gen in
  let extent hi = frequency [ (1, constant 0); (16, int_range 1 hi) ] in
  (* Both operands hold the extent, or one holds an extent other than 1 and
     the other 1. *)
  let alone = frequency [ (1, constant 0); (16, int_range 2 3) ] in
  let axis =
    frequency
      [
        (1, map (fun e -> (e, e)) (extent 3));
        (1, map (fun e -> (e, 1)) alone);
        (1, map (fun e -> (1, e)) alone);
      ]
  in
  let last n l = List.filteri (fun i _ -> i >= List.length l - n) l in
  let edge =
    frequency [ (1, constant None); (6, map Option.some (extent 4)) ]
  in
  let held = frequency [ (1, int_range 0 2); (2, constant 3) ] in
  (let+ axes =
     list ~size:held axis
   and+ na = held
   and+ nb = held
   and+ rows = edge
   and+ cols = edge
   and+ k = frequency [ (1, constant 0); (4, int_range 1 5) ] in
   let sa =
     match rows with
     | None -> [| k |]
     | Some m -> Array.of_list (List.map fst (last na axes) @ [ m; k ])
   in
   let sb =
     match cols with
     | None -> [| k |]
     | Some n -> Array.of_list (List.map snd (last nb axes) @ [ k; n ])
   in
   (sa, sb))
  |> with_pp (fun ppf (sa, sb) ->
      Format.fprintf ppf "a %a b %a" pp_shape sa pp_shape sb)

(* matmul's meaning: each result element is the sum along the inner axis, its
   leading index read in each operand along the axes it has, at 0 where its
   extent is 1. *)
let matmul_meaning sa xa sb xb =
  let ra = Array.length sa and rb = Array.length sb in
  let k = sa.(ra - 1) in
  let lead s r = if r = 1 then [||] else Array.sub s 0 (r - 2) in
  let la = lead sa ra and lb = lead sb rb in
  let l = max (Array.length la) (Array.length lb) in
  let at s i =
    let j = i - (l - Array.length s) in
    if j < 0 then 1 else s.(j)
  in
  let lead = Array.init l (fun i -> if at la i = 1 then at lb i else at la i) in
  let m = if ra = 1 then 1 else sa.(ra - 2) in
  let n = if rb = 1 then 1 else sb.(rb - 1) in
  let rows = if ra = 1 then [||] else [| m |] in
  let cols = if rb = 1 then [||] else [| n |] in
  let shape = Array.concat [ lead; rows; cols ] in
  (* The offset in [s]'s leading block of the result's leading index [r]. *)
  let offset s r =
    let off = ref 0 and r = ref r and stride = ref 1 in
    for i = l - 1 downto 0 do
      let x = !r mod lead.(i) in
      r := !r / lead.(i);
      let j = i - (l - Array.length s) in
      if j >= 0 then begin
        if s.(j) <> 1 then off := !off + (x * !stride);
        stride := !stride * s.(j)
      end
    done;
    !off
  in
  let y =
    Array.init (numel shape) (fun o ->
        let j = o mod n and i = o / n mod m and r = o / (m * n) in
        let a0 = ((offset la r * m) + i) * k in
        let b0 = (offset lb r * k * n) + j in
        let s = ref 0. in
        for kk = 0 to k - 1 do
          s := !s +. (xa.(a0 + kk) *. xb.(b0 + (kk * n)))
        done;
        !s)
  in
  (shape, y)

let matmul_is_its_meaning =
  prop "matmul is its meaning" matmul_shapes (fun (sa, sb) ->
      let ra = Array.length sa and rb = Array.length sb in
      let lead s r = Array.to_list (Array.sub s 0 (max 0 (r - 2))) in
      let la = lead sa ra and lb = lead sb rb in
      let alone ours theirs =
        let d = List.length ours - List.length theirs in
        List.exists Fun.id
          (List.mapi
             (fun i e ->
               e <> 1 && (i < d || List.nth theirs (i - d) = 1))
             ours)
      in
      cover "a 1-d operand" (ra = 1 || rb = 1);
      cover "operands with leading axes of different counts"
        (List.length la <> List.length lb);
      cover "leading axes of a alone and of b alone"
        (alone la lb && alone lb la);
      cover "a zero extent" (Array.mem 0 sa || Array.mem 0 sb);
      cover "an empty sum" (sa.(ra - 1) = 0);
      (* Exact in float64: {!operand}'s products and their sums of 5 fit. *)
      let xa = operand (numel sa) and xb = operand (numel sb) in
      let y = Nx.matmul (host D.Float64 sa xa) (host D.Float64 sb xb) in
      let shape, want = matmul_meaning sa xa sb xb in
      equal (array int) shape (Nx.shape y);
      equal (array float_exact) want (read y))

let matmul =
  let x s = host D.Float32 s (operand (numel s)) in
  let shape_of name sa sb want =
    test name (fun () ->
        equal (array int) want (Nx.shape (Nx.matmul (x sa) (x sb))))
  in
  group "matmul"
    [
      matmul_is_its_meaning;
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

(* dot against einsum *)

(* What a call gives: the result's shape and bits, or that it raises. *)
let outcome f =
  match f () with
  | y ->
      Format.asprintf "%a %s" pp_shape (Nx.shape y)
        (String.concat " "
           (Array.to_list
              (Array.map (fun x -> Int32.to_string (Int32.bits_of_float x))
                 (read y))))
  | exception Invalid_argument _ -> "raises"

let f32_operand n = Array.init n (fun _ -> f32 (Random.float 2. -. 1.))

(* The pattern of names [a] and [b] into [out], summing [k]. *)
let pattern a b out =
  let words = String.concat " " in
  Nx.Pattern.v
    (Printf.sprintf "%s, %s -> %s | k" (words a) (words b) (words out))

let named p n = List.init n (fun i -> p ^ string_of_int i)

(* Operands of 0 to 3 extents each, of any ranks dot takes, the summed
   extents unequal one draw in six. *)
let dot_cases =
  let open Gen in
  let ext = int_range 0 3 in
  (let+ la = list ~size:(int_range 0 2) ext
   and+ lb = list ~size:(int_range 0 2) ext
   and+ k = ext
   and+ k' = frequency [ (5, constant None); (1, map Option.some ext) ]
   and+ c = ext
   and+ vector = frequency [ (1, constant true); (3, constant false) ] in
   let kb = Option.value k' ~default:k in
   ( Array.of_list (la @ [ k ]),
     if vector then [| kb |] else Array.of_list (lb @ [ kb; c ]) ))
  |> with_pp (fun ppf (sa, sb) ->
      Format.fprintf ppf "a %a b %a" pp_shape sa pp_shape sb)

let dot_is_einsum =
  prop "dot gives einsum's bits over its pattern, and raises where it does"
    dot_cases (fun (sa, sb) ->
      let ra = Array.length sa and rb = Array.length sb in
      cover "a vector b" (rb = 1);
      cover "summed extents that differ" (sa.(ra - 1) <> sb.(max 0 (rb - 2)));
      cover "an empty result or sum" (Array.mem 0 sa || Array.mem 0 sb);
      let la = named "a" (ra - 1) and lb = named "b" (max 0 (rb - 2)) in
      let pb, out_b =
        if rb = 1 then ([ "k" ], []) else (lb @ [ "k"; "c" ], lb @ [ "c" ])
      in
      let p = pattern (la @ [ "k" ]) pb (la @ out_b) in
      let a = host D.Float32 sa (f32_operand (numel sa)) in
      let b = host D.Float32 sb (f32_operand (numel sb)) in
      equal string (outcome (fun () -> Nx.einsum p [ a; b ]))
        (outcome (fun () -> Nx.dot a b)))

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
      test "a split operand's rows keep their cut" (fun () ->
          let a = x [| 16; 3 |] and b = x [| 3; 8 |] in
          let split = S2.split ~axis:0 in
          let y =
            Nx.contract Nx.float32 p (Nx.place split a) (Nx.place S2.on b)
          in
          equal bool ~msg:"the result's cut" true
            (Nx.Placement.equal split (Option.get (Nx.placement y)));
          equal (array float_exact)
            (read (Nx.contract Nx.float32 p a b))
            (read y));
      test "operands alone on two devices raise" (fun () ->
          let a = x [| 4; 3 |] and b = x [| 3; 2 |] in
          let on k =
            Nx.place (Nx.Placement.device S2.v (Nx_support.memory k))
          in
          raises_match
            (Exn.invalid_arg ~substring:"alone; place one beside the other")
            (fun () -> Nx.contract Nx.float32 p (on 0 a) (on 1 b)));
    ]

(* Errors *)

let errors =
  let x s = host D.Float32 s (operand (numel s)) in
  let refuses name want f =
    test name (fun () -> equal string want (message f))
  in
  let units r = x (Array.make r 1) in
  let wide s = Nx.broadcast_to s (x [||]) in
  group "errors"
    [
      refuses "a dot whose result has more axes than a layout"
        "Nx.dot: a result of rank 33"
        (fun () -> Nx.dot (units 17) (units 18));
      refuses "a contraction whose result has more axes than a layout"
        "Nx.contract: a result of rank 33"
        (fun () ->
          let names p n = List.init n (fun i -> p ^ string_of_int i) in
          let p =
            Printf.sprintf "%s k, k %s -> %s %s | k"
              (String.concat " " (names "a" 16))
              (String.concat " " (names "b" 17))
              (String.concat " " (names "a" 16))
              (String.concat " " (names "b" 17))
          in
          Nx.contract Nx.float32 (Nx.Pattern.v p) (units 17) (units 18));
      refuses "a dot whose result has more elements than a layout"
        "Nx.dot: a contraction of shape [134217728; 134217728] has more than \
         9007199254740992 elements"
        (fun () ->
          Nx.dot (wide [| 1 lsl 27; 1 |]) (wide [| 1; 1 lsl 27 |]));
      refuses "paired extents differ"
        {|Nx.contract: "i k, k j -> i j | k": k is 3 in a and 4 in b|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 4; 2 |]));
      refuses "axes under ... that differ"
        {|Nx.contract: "... i k, ... k j -> ... i j | k": ...0 is 2 in a and 3 in b|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "... i k, ... k j -> ... i j | k")
            (x [| 2; 2; 3 |])
            (x [| 3; 3; 2 |]));
      refuses "a size an operand contradicts"
        {|Nx.contract: "i k, k j -> i j | k": k is 5 in ~sizes and 3 in a|}
        (fun () ->
          Nx.contract ~sizes:[ ("k", 5) ] Nx.float32
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
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
      refuses "a negative size"
        {|Nx.contract: "(h d) k, k j -> h d j | k": h = -2 in ~sizes is negative|}
        (fun () ->
          Nx.contract Nx.float32
            ~sizes:[ ("h", -2) ]
            (Nx.Pattern.v "(h d) k, k j -> h d j | k")
            (x [| 10; 5 |])
            (x [| 5; 3 |]));
      refuses "a result group past an int"
        {|Nx.contract: "h k, k d -> (h d) | k": (h d) in the result multiplies past an int|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "h k, k d -> (h d) | k")
            (x [| 1 lsl 31; 0 |])
            (x [| 0; 1 lsl 32 |]));
      refuses "a group whose names multiply past an int"
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) is 0 in a, and its names multiply to more than an int counts|}
        (fun () ->
          Nx.contract Nx.float32
            ~sizes:[ ("h", 1 lsl 31); ("d", 1 lsl 32) ]
            (Nx.Pattern.v "(h d) k, k j -> h d j | k")
            (x [| 0; 5 |])
            (x [| 5; 3 |]));
      test "a group of no element solves its last name as 0" (fun () ->
          let y =
            Nx.contract Nx.float32
              ~sizes:[ ("h", 3); ("d", 4) ]
              (Nx.Pattern.v "(h d u) k, k j -> h d u j | k")
              (x [| 0; 5 |])
              (x [| 5; 3 |])
          in
          equal (array int) [| 3; 4; 0; 3 |] (Nx.shape y));
      test "a group's quotient gives another group its last name" (fun () ->
          (* (a b) gives b = 6 / 2, then (b c) gives c = 12 / 3. *)
          let y =
            Nx.contract Nx.float32
              ~sizes:[ ("a", 2) ]
              (Nx.Pattern.v "(b c) k, (a b) k -> a b c | k")
              (x [| 12; 5 |])
              (x [| 6; 5 |])
          in
          equal (array int) [| 2; 3; 4 |] (Nx.shape y));
      refuses "a known zero in a group of elements"
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) is 10 in a, and its names multiply to 0|}
        (fun () ->
          Nx.contract Nx.float32
            ~sizes:[ ("h", 0) ]
            (Nx.Pattern.v "(h d) k, k j -> h d j | k")
            (x [| 10; 5 |])
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
      refuses "a size given twice"
        {|Nx.contract: "(h d) k, k j -> h d j | k": h is twice in ~sizes|}
        (fun () ->
          Nx.contract Nx.float32
            ~sizes:[ ("h", 2); ("h", 2) ]
            (Nx.Pattern.v "(h d) k, k j -> h d j | k")
            (x [| 10; 5 |])
            (x [| 5; 3 |]));
      refuses "too few axes under ..."
        {|Nx.contract: "... i k, ... k j -> ... i j | k" names at least 2 axes for a; a is float32 [3]|}
        (fun () ->
          Nx.contract Nx.float32
            (Nx.Pattern.v "... i k, ... k j -> ... i j | k")
            (x [| 3 |])
            (x [| 3; 2 |]));
      refuses "a one-operand pattern"
        {|Nx.contract: "i k -> k i" has one operand; use Nx.rearrange|}
        (fun () ->
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
        "Nx.contract: ~acc float16 cannot accumulate (no bool, no float \
         narrower than float32)" (fun () ->
          Nx.contract Nx.float32 ~acc:Nx.float16
            (Nx.Pattern.v "i k, k j -> i j | k")
            (x [| 2; 3 |])
            (x [| 3; 2 |]));
      refuses "a boolean accumulator"
        "Nx.contract: ~acc bool cannot accumulate (no bool, no float narrower \
         than float32)" (fun () ->
          let i s =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Int32 s (Array.make (numel s) 1l))
          in
          Nx.contract Nx.int32 ~acc:Nx.bool
            (Nx.Pattern.v "i k, k j -> i j | k")
            (i [| 2; 3 |])
            (i [| 3; 2 |]));
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

(* Scales and segments

   [Nx.Prim.Contract] with block scales and segments: on the host, nx.cpu
   against the operation's expansion within the bound; over devices, a cut
   placement against one device's bits. *)

module Scaled = struct
  module P = Nx.Prim

  type dt = F32 | F16 | F4
  type scale = Codes of int array | Values of dt * float array

  (* An operand: its view's shape, stored with its last two axes swapped
     where [swap], its values in stored order, and its scale's view shape and
     values, stored alike. *)
  type side = {
    dt : dt;
    view : int array;
    swap : bool;
    xs : float array;
    scale : (int array * scale) option;
  }

  type segs = Rows of int64 array | Contracting of int64 array

  (* [a], [[batch; m; k]], by [b], [[batch; g; k; n]], the batch axis where
     [batch] gives its extent and [g] for [Rows] alone, from [init], summed
     in float64 or float32. Offsets have the shape [[batch; g + 1]]. *)
  type case = {
    batch : int option;
    m : int;
    n : int;
    k : int;
    g : int;
    a : side;
    b : side;
    f64 : bool;
    init : float array option;
    segs : segs option;
  }

  let dt_name = function F32 -> "f32" | F16 -> "f16" | F4 -> "f4"

  let pp_floats ppf xs =
    Format.fprintf ppf "[%s]"
      (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%g") xs)))

  let pp_side ppf s =
    Format.fprintf ppf "%s %a%s %a" (dt_name s.dt) pp_shape s.view
      (if s.swap then " swapped" else "")
      pp_floats s.xs;
    match s.scale with
    | None -> ()
    | Some (sv, Codes cs) ->
        Format.fprintf ppf " codes %a %a" pp_shape sv pp_shape cs
    | Some (sv, Values (dt, vs)) ->
        Format.fprintf ppf " %s scale %a %a" (dt_name dt) pp_shape sv pp_floats
          vs

  let pp_case ppf c =
    let offsets ppf os =
      pp_shape ppf (Array.map Int64.to_int os)
    in
    Format.fprintf ppf "@[<v>%s, g %d@,a %a@,b %a@,init %s@,%s@]"
      (if c.f64 then "float64" else "float32")
      c.g pp_side c.a pp_side c.b
      (match c.init with
      | None -> "none"
      | Some xs -> Format.asprintf "%a" pp_floats xs)
      (match c.segs with
      | None -> "unsegmented"
      | Some (Rows os) -> Format.asprintf "rows %a" offsets os
      | Some (Contracting os) -> Format.asprintf "contracting %a" offsets os)

  let swapped s =
    let s = Array.copy s and r = Array.length s in
    if r >= 2 then begin
      let t = s.(r - 1) in
      s.(r - 1) <- s.(r - 2);
      s.(r - 2) <- t
    end;
    s

  let stored s v = if s.swap then swapped v else v
  let lead c = Option.to_list c.batch

  let result_shape c =
    let groups = match c.segs with Some (Contracting _) -> [ c.g ] | _ -> [] in
    Array.of_list (groups @ lead c @ [ c.m; c.n ])

  let offsets_shape c = Array.of_list (lead c @ [ c.g + 1 ])

  (* Values *)

  (* Where each operand lies: [put role x] places the host value [x], stored
     as made, for [role]. *)
  type role = A | B | A_scale | B_scale | Offsets | Init
  type 'e put = {
    put : 'v 's. role -> ('v, 's, Nx.host) Nx.t -> ('v, 's, 'e) Nx.t;
  }

  let on_host = { put = (fun _ x -> x) }

  (* An array of [dt] of the stored shape [shape] holding [xs], placed by
     [p] for [role], through its view. *)
  let made (type e) (p : e put) role dt shape swap xs : e P.any =
    let x = Nx.create Nx.float32 shape xs in
    let view x =
      let x = p.put role x in
      if swap then Nx.swapaxes (-1) (-2) x else x
    in
    match dt with
    | F32 -> Any (view x)
    | F16 -> Any (view (Nx.cast Nx.float16 x))
    | F4 -> Any (view (Nx.cast Nx.float4_e2m1fn x))

  let operand p role s = made p role s.dt (stored s s.view) s.swap s.xs

  let scale (type e) (p : e put) role s : e P.scale option =
    match s.scale with
    | None -> None
    | Some (sv, Codes cs) ->
        let x = p.put role (Nx.create Nx.uint8 (stored s sv) cs) in
        Some (Exponent (if s.swap then Nx.swapaxes (-1) (-2) x else x))
    | Some (sv, Values (dt, vs)) ->
        let (P.Any v) = made p role dt (stored s sv) s.swap vs in
        Some (Value v)

  let op (type r e) (p : e put) (acc : (float, r) Nx.dtype) c :
      (float, r, e) Nx.t P.t =
    let (P.Any a) = operand p A c.a in
    let (P.Any b) = operand p B c.b in
    let nb = List.length (lead c) in
    let batch : (int * int) iarray = if nb = 1 then [| (0, 0) |] else [||] in
    let bk = match c.segs with Some (Rows _) -> nb + 1 | _ -> nb in
    let contracting : (int * int) iarray = [| (nb + 1, bk) |] in
    let offsets os = p.put Offsets (Nx.create Nx.int64 (offsets_shape c) os) in
    let segments : _ Nx_kernel.segments option =
      match c.segs with
      | None -> None
      | Some (Rows os) ->
          Some (Rows { offsets = offsets os; a_axis = nb; b_axis = nb })
      | Some (Contracting os) ->
          Some (Contracting { offsets = offsets os; pair = 0 })
    in
    let init =
      Option.map
        (fun xs ->
          p.put Init (Nx.cast acc (Nx.create Nx.float64 (result_shape c) xs)))
        c.init
    in
    Contract
      {
        order = P.contract_order ~batch ~contracting a b;
        batch;
        contracting;
        acc = D.Any acc;
        out = acc;
        a;
        b;
        init;
        a_scale = scale p A_scale c.a;
        b_scale = scale p B_scale c.b;
        segments;
      }

  (* [c] with each value and scale its magnitude: its sums bound the
     roundings of [c]'s. *)
  let magnitudes c =
    let side s =
      let scale =
        match s.scale with
        | Some (sv, Values (dt, vs)) ->
            Some (sv, Values (dt, Array.map Float.abs vs))
        | sc -> sc
      in
      { s with xs = Array.map Float.abs s.xs; scale }
    in
    let init = Option.map (Array.map Float.abs) c.init in
    { c with a = side c.a; b = side c.b; init }

  (* [op] by its expansion, each other operation eagerly. *)
  let expanded op =
    let rule : type q. P.interpretation -> by:string -> q P.t -> q =
     fun i ~by op ->
      match op with
      | Contract _ -> (
          match P.expand i ~by ~repeat:Nx_support.folding op with
          | Some r -> r
          | None -> fail "a contraction without an expansion")
      | _ -> P.eval ~by op
    in
    P.interpret ~name:"test.expand" Extent rule (fun _ -> P.eval ~by:"t" op)

  (* The index [r] of the segmented axis is in group [e] of the offsets [o],
     as Nx_kernel.segments states. *)
  let group_of (o : int64 array) r =
    let g = Array.length o - 1 and r = Int64.of_int r in
    let rec find e =
      if e < 0 then None
      else if o.(e) <= r then Some e
      else find (e - 1)
    in
    if g > 0 && o.(0) <= r && r < o.(g) then find (g - 1) else None

  (* Each output's bits are within twice the bound of the exact sum of
     either: [2k + 8] roundings of [u] relative to the magnitudes' sum, and
     one least subnormal per term for what underflows. A NaN magnitude is a
     NaN term, which makes both NaNs; an infinite one is an overflow, which
     the bound bars. *)
  let check (type r) (acc : (float, r) Nx.dtype) c =
    let read y = Nx.to_array (Nx.cast Nx.float64 y) in
    let y = read (P.eval ~by:"t" (op on_host acc c)) in
    let want = read (expanded (op on_host acc c)) in
    let mag = read (expanded (op on_host acc (magnitudes c))) in
    let u, eta =
      if c.f64 then (epsilon_float /. 2., Float.of_string "0x1p-1074")
      else (Float.of_string "0x1p-24", Float.of_string "0x1p-149")
    in
    let n = (2 * c.k) + 8 in
    let gamma = Float.of_int n *. u /. (1. -. (Float.of_int n *. u)) in
    Array.iteri
      (fun i m ->
        let msg = Printf.sprintf "output %d of magnitude %g" i m in
        if Float.is_nan m then
          equal (pair bool bool) ~msg (true, true)
            (Float.is_nan want.(i), Float.is_nan y.(i))
        else if Float.is_finite m then
          let tol = (2. *. gamma *. m) +. (Float.of_int (c.k + 2) *. eta) in
          equal (float tol) ~msg want.(i) y.(i))
      mag;
    (y, mag)

  (* Generators *)

  let e2m1 =
    [ 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. ]
    @ [ -0.; -0.5; -1.; -1.5; -2.; -3.; -4.; -6. ]

  let finite = [ 0.; -0.; 0.5; -1.; 1.5; 3.; -3.; 1. ]

  let element dt =
    let open Gen in
    match dt with
    | F4 -> of_list e2m1
    | F32 | F16 -> frequency [ (12, of_list finite); (1, constant Float.nan) ]

  (* Codes of every kind: the neutral 127 and its neighbours most often, the
     subnormal 2^-127 of 0, the largest finite 254 and the NaN 255. *)
  let code =
    let open Gen in
    frequency
      [
        (10, of_list [ 126; 127; 128; 131 ]);
        (3, of_list [ 0; 1; 120; 253 ]);
        (1, of_list [ 254; 255 ]);
      ]

  let scale_value =
    let open Gen in
    frequency
      [ (10, of_list [ 0.5; 2.; -1.; 0.25; 3.; 0. ]); (1, constant Float.nan) ]

  (* Each element of [gs] drawn in order. *)
  let all gs =
    let open Gen in
    let rec go i =
      if i = Array.length gs then constant []
      else
        let* x = gs.(i) in
        let+ rest = go (i + 1) in
        x :: rest
    in
    map Array.of_list (go 0)

  (* The block counts an axis of [e] elements takes: [e] itself, one block,
     or blocks of 2, 3 or 32; none for an empty axis. *)
  let blocks e =
    if e = 0 then Gen.constant 0
    else
      Gen.of_list
        (List.sort_uniq compare
           (List.filter_map
              (fun r -> if e mod r = 0 then Some (e / r) else None)
              [ 1; 2; 3; 32; e ]))

  let scale_of sv =
    let open Gen in
    let size = constant (numel sv) in
    let* codes = bool in
    let+ sc =
      if codes then map (fun cs -> Codes cs) (array ~size code)
      else
        let* dt = of_list [ F32; F16 ] in
        let+ vs = array ~size scale_value in
        Values (dt, vs)
    in
    Some (sv, sc)

  let side dt view =
    let open Gen in
    let* swap = bool in
    let size = constant (numel view) in
    let* xs = array ~size (element dt) in
    let* scaled = frequency [ (1, constant false); (2, constant true) ] in
    let+ scale =
      if not scaled then constant None
      else
        let* sv = all (Array.map blocks view) in
        scale_of sv
    in
    { dt; view; swap; xs; scale }

  (* MXFP4: float4 stored [[…; n; k]], a code per 32 along [k]. *)
  let mxfp4 view =
    let open Gen in
    let r = Array.length view in
    let* xs = array ~size:(constant (numel view)) (element F4) in
    let sv = Array.mapi (fun i e -> if i = r - 2 then e / 32 else e) view in
    let+ cs = array ~size:(constant (numel sv)) code in
    { dt = F4; view; swap = true; xs; scale = Some (sv, Codes cs) }

  (* The offsets of each batch index, [g + 1] of them over an axis of [len]:
     increasing half the time, any values from [-1] to [len + 1] else. *)
  let offsets ~batch ~g len =
    let open Gen in
    let row =
      let* sorted = bool in
      let+ o = array ~size:(constant (g + 1)) (int_range (-1) (len + 1)) in
      if sorted then Array.sort compare o;
      Array.to_list (Array.map Int64.of_int o)
    in
    let+ rows = list ~size:(constant (max 1 batch)) row in
    let rows = if batch = 0 then [] else rows in
    Array.of_list (List.concat rows)

  (* [c] with a NaN in [a] at index 0 of the contracted axis, which
     [Contracting]'s groups leave out: one group from 1, the others empty. *)
  let planted c =
    match c.segs with
    | Some (Contracting os) when c.k > 1 && c.g > 0 ->
        let w = c.g + 1 in
        let os =
          Array.mapi
            (fun i _ -> if i mod w = 0 then 1L else Int64.of_int c.k)
            os
        in
        (* [k]'s index is the last of the view, before the last stored. *)
        let inner = if c.a.swap then c.m else 1 in
        let xs =
          Array.mapi
            (fun i x -> if i / inner mod c.k = 0 then Float.nan else x)
            c.a.xs
        in
        { c with a = { c.a with xs }; segs = Some (Contracting os) }
    | _ -> c

  let case =
    let open Gen in
    let* batch =
      frequency [ (3, constant None); (1, map Option.some (int_range 0 2)) ]
    in
    let* mx = frequency [ (3, constant false); (1, constant true) ] in
    let* m = of_list [ 0; 1; 2; 3; 5 ] in
    let* n = of_list [ 0; 1; 2; 4 ] in
    let* k =
      if mx then of_list [ 32; 64; 96 ]
      else of_list [ 0; 1; 2; 3; 4; 6; 32; 64 ]
    in
    let* kind = of_list [ `None; `Rows; `Contracting ] in
    let* g = int_range 0 3 in
    let lead = Option.to_list batch in
    let groups = if kind = `Rows then [ g ] else [] in
    let av = Array.of_list (lead @ [ m; k ]) in
    let bv = Array.of_list (lead @ groups @ [ k; n ]) in
    let* adt = of_list [ F32; F16 ] in
    let* a = side adt av in
    let* b =
      if mx then mxfp4 bv
      else
        let* bdt = of_list [ F32; F16; F4 ] in
        side bdt bv
    in
    let* f64 = bool in
    let nb = Option.value ~default:1 batch in
    let* segs =
      match kind with
      | `None -> constant None
      | `Rows -> map (fun o -> Some (Rows o)) (offsets ~batch:nb ~g m)
      | `Contracting ->
          map (fun o -> Some (Contracting o)) (offsets ~batch:nb ~g k)
    in
    let* plant = frequency [ (1, constant false); (2, constant true) ] in
    let c = { batch; m; n; k; g; a; b; f64; init = None; segs } in
    let c = if plant then planted c else c in
    let+ init =
      frequency
        [
          (1, constant None);
          ( 1,
            let size = constant (numel (result_shape c)) in
            map Option.some (array ~size (of_list finite)) );
        ]
    in
    { c with init }

  let case = Gen.with_pp pp_case case

  (* The element of [s] at the view's index [idx]. *)
  let value_at s idx =
    let idx = stored s idx and shape = stored s s.view in
    let flat = ref 0 in
    Array.iteri (fun i x -> flat := (!flat * shape.(i)) + x) idx;
    s.xs.(!flat)

  let offsets_at c os b =
    let w = c.g + 1 in
    Array.sub os (b * w) w

  let batches c =
    match c.batch with None -> [ None ] | Some e -> List.init e Option.some

  let with_batch bt rest = Array.of_list (Option.to_list bt @ rest)
  let range n = List.init n Fun.id

  (* Whether [Rows] leaves a row in no group that has outputs. *)
  let row_alone c =
    match c.segs with
    | Some (Rows os) ->
        c.n > 0
        && List.exists
             (fun bt ->
               let o = offsets_at c os (Option.value ~default:0 bt) in
               List.exists (fun r -> group_of o r = None) (range c.m))
             (batches c)
    | _ -> false

  let nan_at s bt idx = Float.is_nan (value_at s (with_batch bt idx))

  (* Whether a NaN of [a] or [b] lies at an index of the contracted axis in
     no group. *)
  let nan_alone c =
    match c.segs with
    | Some (Contracting os) ->
        List.exists
          (fun bt ->
            let o = offsets_at c os (Option.value ~default:0 bt) in
            List.exists
              (fun r ->
                group_of o r = None
                && (List.exists (fun i -> nan_at c.a bt [ i; r ]) (range c.m)
                   || List.exists (fun j -> nan_at c.b bt [ r; j ]) (range c.n)
                   ))
              (range c.k))
          (batches c)
    | _ -> false

  let law c =
    let codes s = match s.scale with Some (_, Codes _) -> true | _ -> false in
    let values s = match s.scale with Some (_, Values _) -> true | _ -> false in
    let cut s =
      match s.scale with
      | Some (sv, _) -> Array.exists2 (fun e d -> e > 1 && e < d) sv s.view
      | None -> false
    in
    cover "a scale of codes" (codes c.a || codes c.b);
    cover "a scale of values" (values c.a || values c.b);
    cover "a scale cutting an axis into blocks" (cut c.a || cut c.b);
    cover "an MXFP4 operand" (c.b.dt = F4 && codes c.b && c.k >= 32);
    let rows = match c.segs with Some (Rows _) -> true | _ -> false in
    let scaled = c.a.scale <> None || c.b.scale <> None in
    cover "rows by groups" rows;
    cover "groups of the contracted axis"
      (match c.segs with Some (Contracting _) -> c.g > 0 | _ -> false);
    cover "scaled and segmented" (scaled && c.segs <> None);
    cover "a row in no group" (row_alone c);
    let y, mag = if c.f64 then check Nx.float64 c else check Nx.float32 c in
    cover "a NaN output" (Array.exists Float.is_nan y);
    cover "a NaN in no group kept out"
      (nan_alone c && Array.exists Float.is_finite mag)

  (* Placements *)

  module S4 = (val Nx.devices (List.init 4 Nx_support.memory))

  (* [n] finite values, by position. *)
  let some n = Array.init n (fun i -> List.nth finite (i * 7 mod 8))
  let codes n = Array.init n (fun i -> [| 126; 127; 128; 131 |].(i * 5 mod 4))

  (* [a] [[m; k]] by MXFP4 [b] [[k; n]], stored [[n; k]], a code per 32 along
     [k] and per [per] along [n]; [a] scaled per two rows where [rows]. *)
  let mx ?(rows = false) ?segs ?(g = 0) ~m ~k ~n ~per () =
    let lead = match segs with Some (Rows _) -> [| g |] | _ -> [||] in
    let bv = Array.append lead [| k; n |] in
    let sv = Array.append lead [| k / 32; n / per |] in
    let a_scale =
      if not rows then None
      else
        let per_row i = [| 0.5; 2. |].(i mod 2) in
        Some ([| m / 2; 1 |], Values (F32, Array.init (m / 2) per_row))
    in
    let b_scale = Some (sv, Codes (codes (numel sv))) in
    {
      batch = None;
      m;
      n;
      k;
      g;
      a =
        {
          dt = F32;
          view = [| m; k |];
          swap = false;
          xs = some (m * k);
          scale = a_scale;
        };
      b =
        { dt = F4; view = bv; swap = true; xs = some (numel bv); scale = b_scale };
      f64 = false;
      init = None;
      segs;
    }

  (* [c] with [a]'s or [b]'s arrays placed by [where], the others on every
     device of the set of [on]: the result's placement, and its bits against
     one device's. *)
  let placed (type e) (on : e Nx.Placement.t) where c =
    let at role = Option.value ~default:on (where role) in
    let p = { put = (fun role x -> Nx.place (at role) x) } in
    let y = P.eval ~by:"t" (op p Nx.float32 c) in
    let one = P.eval ~by:"t" (op on_host Nx.float32 c) in
    equal (array float_exact) (read one) (read y);
    Option.get (Nx.placement y)

  (* A long MXFP4 product of more than one block of 4096, rows in no group,
     and a NaN outside every group of the contracted axis. *)
  let examples =
    let rows = Rows [| 1L; 2L; 2L; 3L |] in
    let groups = Contracting [| 0L; 4L; 4L |] in
    [
      mx ~m:1 ~k:4160 ~n:2 ~per:1 ();
      mx ~segs:rows ~g:3 ~m:4 ~k:32 ~n:2 ~per:1 ();
      planted
        { (mx ~segs:groups ~g:2 ~m:2 ~k:32 ~n:2 ~per:2 ()) with
          init = Some (some 8);
        };
    ]

  let within_bound =
    prop ~examples
      "a scaled or segmented contraction is within the bound of its expansion"
      case law

  let placements =
    let b_cut q = function
      | B | B_scale -> Some q
      | A | A_scale | Offsets | Init -> None
    in
    let a_cut ?(scale = true) q = function
      | A -> Some q
      | A_scale when scale -> Some q
      | A_scale | B | B_scale | Offsets | Init -> None
    in
    group "placements"
      [
        test "a cut over b's columns at its blocks gives one device's bits"
          (fun () ->
            let c = mx ~m:3 ~k:64 ~n:8 ~per:2 () in
            let y = placed S2.on (b_cut (S2.split ~axis:0)) c in
            equal bool ~msg:"the result's columns are cut" true
              (Nx.Placement.equal (S2.split ~axis:1) y));
        test "a cut inside b's blocks of columns is undone" (fun () ->
            let c = mx ~m:3 ~k:64 ~n:8 ~per:4 () in
            let y = placed S4.on (b_cut (S4.split ~axis:0)) c in
            equal bool ~msg:"the result's columns are cut" false
              (Nx.Placement.equal (S4.split ~axis:1) y));
        test "a cut over a's rows at its blocks gives one device's bits"
          (fun () ->
            let c = mx ~rows:true ~m:4 ~k:32 ~n:2 ~per:1 () in
            let y = placed S2.on (a_cut (S2.split ~axis:0)) c in
            equal bool ~msg:"the result's rows are cut" true
              (Nx.Placement.equal (S2.split ~axis:0) y));
        test "segmented rows cut over devices give one device's bits" (fun () ->
            let segs = Rows [| 0L; 3L; 3L; 6L |] in
            let c = mx ~segs ~g:3 ~m:6 ~k:32 ~n:4 ~per:2 () in
            ignore (placed S2.on (a_cut ~scale:false (S2.split ~axis:0)) c));
      ]

  (* Contractions cut along [k]: [a] [[m; k]] and [b], stored [[…; n; k]],
     along their last stored axis, the scales and offsets on every device. *)
  let partials =
    let k_cut ?(b = 1) (module S : Nx.Devices) = function
      | A -> Some (S.split ~axis:1)
      | B -> Some (S.split ~axis:b)
      | A_scale | B_scale | Offsets | Init -> None
    in
    let along ?b (module S : Nx.Devices) c =
      ignore (placed S.on (k_cut ?b (module S)) c)
    in
    (* [a] scaled in two blocks of [k / 2] per row. *)
    let halves c =
      let scale = Some ([| c.m; 2 |], Values (F32, some (2 * c.m))) in
      { c with a = { c.a with scale } }
    in
    group "partials"
      [
        test "a cut at blocks of 4096 products gives one device's bits"
          (fun () -> along (module S2) (mx ~m:1 ~k:8256 ~n:2 ~per:1 ()));
        test "a cut inside a block of 4096 products gives one device's bits"
          (fun () -> along (module S2) (mx ~m:8 ~k:6144 ~n:8 ~per:2 ()));
        test "a cut inside the scale's blocks gives one device's bits"
          (fun () -> along (module S4) (mx ~m:2 ~k:96 ~n:2 ~per:1 ()));
        test "scale blocks past 4096 products give one device's bits"
          (fun () ->
            along (module S2) (halves (mx ~m:2 ~k:16384 ~n:2 ~per:1 ())));
        test "groups of a cut contracted axis give one device's bits" (fun () ->
            let segs = Contracting [| 0L; 40L; 96L |] in
            along (module S2) (mx ~segs ~g:2 ~m:2 ~k:96 ~n:2 ~per:1 ()));
        test "rows by groups over a cut contracted axis give one device's bits"
          (fun () ->
            let segs = Rows [| 0L; 1L; 4L |] in
            along ~b:2 (module S2) (mx ~segs ~g:2 ~m:4 ~k:64 ~n:2 ~per:2 ()));
      ]
end

(* Convolution: Nx.conv against its sum over the taps computed in OCaml, on
   small integers whose sums every dtype holds exactly, and conv_transpose as
   its adjoint. *)
module Conv = struct
  type case = {
    n : int;
    spatial : int array;
    groups : int;
    cg : int;
    og : int;
    taps : int array;
    stride : int array;
    dilation : int array;
    padding : (int * int) array;
    xs : float array;
    ws : float array;
  }

  let pp_case ppf c =
    let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
    Format.fprintf ppf
      "n %d spatial [%s] groups %d cg %d og %d taps [%s] stride [%s] dilation [%s] padding [%s]"
      c.n (ints c.spatial) c.groups c.cg c.og (ints c.taps) (ints c.stride)
      (ints c.dilation)
      (String.concat "; " (Array.to_list (Array.map (fun (a, b) -> Printf.sprintf "(%d, %d)" a b) c.padding)))

  let numel = Array.fold_left ( * ) 1
  let x_shape c = Array.concat [ [| c.n |]; c.spatial; [| c.groups * c.cg |] ]
  let w_shape c = Array.concat [ [| c.groups * c.og |]; c.taps; [| c.cg |] ]

  (* The dilated kernel's extent along axis [i]: none without a tap. *)
  let dilated c i =
    if c.taps.(i) = 0 then 0 else (c.dilation.(i) * (c.taps.(i) - 1)) + 1

  (* The windows along axis [i] of an input of extent [d]. *)
  let count c i d =
    let lo, hi = c.padding.(i) in
    let span = dilated c i in
    let e = d + lo + hi in
    if e < span then 0 else ((e - span) / c.stride.(i)) + 1

  let windows c i = count c i c.spatial.(i)

  let case =
    Gen.with_pp pp_case
      (let open Gen in
       let* k = int_range 1 2 in
       let ints n lo hi = array ~size:(constant n) (int_range lo hi) in
       let* n = int_range 0 2 in
       let* spatial = ints k 0 6 in
       let* groups = int_range 1 2 in
       let* cg = int_range 0 2 in
       let* og = int_range 1 2 in
       let* taps = ints k 0 3 in
       let* stride = ints k 1 3 in
       let* dilation = ints k 1 2 in
       let* los = ints k (-1) 2 in
       let* his = ints k (-1) 2 in
       (* One case in ten makes axis 0 an input below the formula's extent:
          one window of one tap, stride 2, over an input of 1 padded by 1 on
          each side, which an input of 0 has too. Drawn freely it is about
          one case in a hundred, which some seeds never reach. *)
       let* below = int_range 0 9 in
       let at0 v a =
         if below = 0 then Array.mapi (fun i e -> if i = 0 then v else e) a
         else a
       in
       let spatial = at0 1 spatial and taps = at0 1 taps in
       let stride = at0 2 stride and dilation = at0 1 dilation in
       let los = at0 1 los and his = at0 1 his in
       (* One case in ten makes axis 0 a padding that cuts past the axis: an
          input of 1 cropped by 1 on each side. Drawn freely it is about one
          case in twenty, which some seeds never reach. *)
       let* past = int_range 0 9 in
       let cut v a =
         if past = 0 then Array.mapi (fun i e -> if i = 0 then v else e) a
         else a
       in
       let spatial = cut 1 spatial in
       let los = cut (-1) los and his = cut (-1) his in
       let c0 =
         { n; spatial; groups; cg; og; taps; stride; dilation;
           padding = Array.map2 (fun a b -> (a, b)) los his; xs = [||]; ws = [||] }
       in
       let small m = array ~size:(constant m) (map Float.of_int (int_range (-3) 3)) in
       let+ xs = small (numel (x_shape c0)) and+ ws = small (numel (w_shape c0)) in
       { c0 with xs; ws })

  let unravel s j =
    let i = Array.make (Array.length s) 0 and j = ref j in
    for a = Array.length s - 1 downto 0 do
      i.(a) <- !j mod s.(a);
      j := !j / s.(a)
    done;
    i

  let ravel s i =
    let j = ref 0 in
    Array.iteri (fun a e -> j := (!j * s.(a)) + e) i;
    !j

  (* The sum over the taps and the group's channels, at each output index. *)
  let reference c =
    let k = Array.length c.spatial in
    let t = Array.init k (windows c) in
    let o = c.groups * c.og in
    let out = Array.concat [ [| c.n |]; t; [| o |] ] in
    let xs_shape = x_shape c and ws_shape = w_shape c in
    Array.init (numel out) (fun flat ->
        let idx = unravel out flat in
        let b = idx.(0) and j = idx.(k + 1) in
        let g = j / c.og in
        let sum = ref 0. in
        for tap = 0 to numel c.taps - 1 do
          let q = unravel c.taps tap in
          let pos = Array.init k (fun i -> (idx.(1 + i) * c.stride.(i)) + (q.(i) * c.dilation.(i)) - fst c.padding.(i)) in
          if Array.for_all2 (fun p d -> p >= 0 && p < d) pos c.spatial then
            for i = 0 to c.cg - 1 do
              let xv = c.xs.(ravel xs_shape (Array.concat [ [| b |]; pos; [| (g * c.cg) + i |] ])) in
              let wv = c.ws.(ravel ws_shape (Array.concat [ [| j |]; q; [| i |] ])) in
              sum := !sum +. (xv *. wv)
            done
        done;
        !sum)

  let run (type v s) (dt : (v, s) D.t) c =
    let x = Nx.cast dt (Nx.create Nx.float64 (x_shape c) c.xs) in
    let w = Nx.cast dt (Nx.create Nx.float64 (w_shape c) c.ws) in
    let y = Nx.conv ~stride:c.stride ~padding:c.padding ~dilation:c.dilation ~groups:c.groups x w in
    Nx.to_array (Nx.cast Nx.float64 y)

  let law_conv =
    prop "conv is its sum over the taps, at every dtype" case (fun c ->
        let want = reference c in
        let axes f = Array.exists Fun.id (Array.init (Array.length c.spatial) f) in
        cover "an output" (Array.length want > 0);
        cover "two groups" (c.groups = 2);
        cover "no input channel" (c.cg = 0);
        cover "a negative padding" (axes (fun i -> fst c.padding.(i) < 0 || snd c.padding.(i) < 0));
        cover "a padding that cuts past the axis"
          (axes (fun i -> c.spatial.(i) + fst c.padding.(i) + snd c.padding.(i) < 0));
        cover "a stride and a dilation above 1"
          (axes (fun i -> c.stride.(i) > 1 && c.dilation.(i) > 1));
        cover "a kernel axis of no element, under an output"
          (Array.mem 0 c.taps && Array.length want > 0);
        equal ~msg:"float64" (array float_exact) want (run D.Float64 c);
        equal ~msg:"float32" (array float_exact) want (run D.Float32 c);
        equal ~msg:"float16, summed at float32" (array float_exact) want (run D.Float16 c);
        equal ~msg:"int32" (array float_exact) want (run D.Int32 c))

  (* The least input extent along axis [i] whose convolution has [t]
     windows, searched from the count, or [None] where no extent has. The
     count grows by at most one per element, so the search stops past
     [t]. *)
  let least c i t =
    let rec from d =
      let n = count c i d in
      if n = t then Some d else if n > t then None else from (d + 1)
    in
    from 0

  (* [x]'s spatial extents are [c.spatial], each a count [t]. *)
  let law_transpose =
    prop "conv_transpose is conv's adjoint" case (fun c ->
        let t = c.spatial in
        let least = Array.mapi (least c) t in
        cover "no input has the count" (Array.exists Option.is_none least);
        cover "a kernel axis of no element" (Array.mem 0 c.taps);
        cover "a padding past the span, cut in the transpose"
          (Array.exists Fun.id
             (Array.mapi
                (fun i (lo, _) -> lo > c.dilation.(i) * (c.taps.(i) - 1))
                c.padding));
        cover "an input below the formula's extent"
          (Array.exists Fun.id
             (Array.mapi
                (fun i d ->
                  let lo, hi = c.padding.(i) in
                  t.(i) > 0 && d = Some 0
                  && ((t.(i) - 1) * c.stride.(i)) + dilated c i - lo - hi < 0)
                least));
        let o = c.groups * c.og in
        let gs = Array.concat [ [| c.n |]; t; [| o |] ] in
        let g = Nx.init Nx.float64 gs (fun i -> Float.of_int ((Array.fold_left ( + ) 0 i mod 5) - 2)) in
        let w = Nx.create Nx.float64 (w_shape c) c.ws in
        let stride = c.stride and padding = c.padding and dilation = c.dilation in
        let groups = c.groups in
        if Array.exists Option.is_none least then
          raises_match (Exn.invalid_arg ~substring:"Nx.conv_transpose: ") (fun () ->
              Nx.conv_transpose ~stride ~padding ~dilation ~groups g w)
        else
          let d = Array.map Option.get least in
          let z_shape = Array.concat [ [| c.n |]; d; [| c.groups * c.cg |] ] in
          let z = Nx.init Nx.float64 z_shape (fun i -> Float.of_int ((Array.fold_left ( * ) 1 (Array.map succ i) mod 7) - 3)) in
          let y = Nx.conv_transpose ~stride ~padding ~dilation ~groups g w in
          equal ~msg:"shape" (array int) z_shape (Nx.shape y);
          (* Small integers: each dtype's sums are exact. *)
          let at (type v s) (dt : (v, s) D.t) =
            Nx.to_array
              (Nx.cast Nx.float64
                 (Nx.conv_transpose ~stride ~padding ~dilation ~groups (Nx.cast dt g)
                    (Nx.cast dt w)))
          in
          let want = Nx.to_array y in
          equal ~msg:"float32" (array float_exact) want (at D.Float32);
          equal ~msg:"float16" (array float_exact) want (at D.Float16);
          equal ~msg:"int32" (array float_exact) want (at D.Int32);
          let conv = Nx.conv ~stride ~padding ~dilation ~groups z w in
          let lhs = Nx.item [] (Nx.sum (Nx.mul conv g)) in
          let rhs = Nx.item [] (Nx.sum (Nx.mul z y)) in
          equal ~msg:"adjoint" float_exact lhs rhs)

  let invalid f = raises_match (Exn.invalid_arg ~substring:"Nx.conv") f

  let test_refusals () =
    let x = Nx.zeros Nx.float32 [| 1; 5; 4 |] and w = Nx.zeros Nx.float32 [| 2; 3; 4 |] in
    invalid (fun () -> Nx.conv (Nx.zeros Nx.float32 [| 5; 4 |]) w);
    invalid (fun () -> Nx.conv x (Nx.zeros Nx.float32 [| 2; 4 |]));
    invalid (fun () -> Nx.conv ~stride:[| 1; 1 |] x w);
    invalid (fun () -> Nx.conv ~stride:[| 0 |] x w);
    invalid (fun () -> Nx.conv ~dilation:[| 0 |] x w);
    invalid (fun () -> Nx.conv ~groups:3 x w);
    invalid (fun () -> Nx.conv ~groups:2 x w);
    invalid (fun () -> Nx.conv (Nx.zeros Nx.bool [| 1; 5; 4 |]) (Nx.zeros Nx.bool [| 2; 3; 4 |]));
    invalid (fun () -> Nx.conv_transpose x w);
    invalid (fun () -> Nx.conv_transpose ~padding:[| (3, 3) |] (Nx.zeros Nx.float32 [| 1; 0; 2 |]) w)

  (* Three products reach one output: 2048 then 1 and 1 sum to 2048 at
     float16, but to 2050 at float32, which float16 holds. *)
  let test_narrow () =
    let f16 s vs = Nx.create Nx.float16 s vs in
    let three = f16 [| 1; 3; 1 |] [| 2048.; 1.; 1. |] in
    equal (array float_exact) [| 2050. |]
      (Nx.to_array (Nx.conv three (f16 [| 1; 3; 1 |] [| 1.; 1.; 1. |])));
    let channels = f16 [| 1; 1; 3 |] [| 2048.; 1.; 1. |] in
    equal (array float_exact) [| 2050. |]
      (Nx.to_array (Nx.conv_transpose channels (f16 [| 3; 1; 1 |] [| 1.; 1.; 1. |])))

  let test_wrap () =
    let i8 s vs = Nx.create Nx.int8 s vs in
    equal (array int) [| -56 |]
      (Nx.to_array (Nx.conv (i8 [| 1; 2; 1 |] [| 100; 100 |]) (i8 [| 1; 2; 1 |] [| 1; 1 |])));
    equal (array int) [| -56 |]
      (Nx.to_array
         (Nx.conv_transpose (i8 [| 1; 1; 2 |] [| 100; 100 |]) (i8 [| 2; 1; 1 |] [| 1; 1 |])))

  (* A donated operand that another operation consumed is dead: each
     function names itself, whatever it computes through. *)
  let test_dead_names () =
    let dead s =
      let x = Nx.create Nx.float32 s (Array.make (numel s) 1.) in
      let _ = Nx.neg (Nx.donate x) in
      x
    in
    let live s = Nx.ones Nx.float32 s in
    let names by f =
      raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
    in
    names "Nx.conv" (fun () -> Nx.conv (dead [| 1; 4; 2 |]) (live [| 2; 2; 2 |]));
    names "Nx.conv" (fun () -> Nx.conv (live [| 1; 4; 2 |]) (dead [| 2; 2; 2 |]));
    names "Nx.conv_transpose" (fun () ->
        Nx.conv_transpose (dead [| 1; 3; 2 |]) (live [| 2; 2; 2 |]));
    names "Nx.conv_transpose" (fun () ->
        Nx.conv_transpose (live [| 1; 3; 2 |]) (dead [| 2; 2; 2 |]))

  (* A kernel of no tap along a spatial axis sums no term: +0 at each index
     of the result its formula shapes, an empty kernel's dilated extent being
     0 whatever the dilation: [t = s + 1] at stride 1 and [s = t - 1] for the
     transpose, on the operands' set. *)
  let test_no_taps () =
    let f32 s v = Nx.create Nx.float32 s (Array.make (numel s) v) in
    let w = f32 [| 2; 0; 4 |] 1. in
    let zeros s y =
      equal ~msg:"shape" (array int) s (Nx.shape y);
      equal ~msg:"+0" (array float_exact) (Array.make (numel s) 0.)
        (Nx.to_array y);
      equal ~msg:"on the host" bool true (Option.is_some (Nx.placement y))
    in
    zeros [| 1; 6; 2 |] (Nx.conv (f32 [| 1; 5; 4 |] 1.) w);
    zeros [| 1; 6; 2 |] (Nx.conv ~dilation:[| 2 |] (f32 [| 1; 5; 4 |] 1.) w);
    zeros [| 1; 3; 2 |] (Nx.conv ~stride:[| 2 |] (f32 [| 1; 5; 4 |] 1.) w);
    zeros [| 1; 0; 2 |]
      (Nx.conv ~dilation:[| 3 |] ~padding:[| (-2, 0) |] (f32 [| 1; 1; 4 |] 1.) w);
    zeros [| 1; 5; 4 |]
      (Nx.conv_transpose ~dilation:[| 3 |] (f32 [| 1; 6; 2 |] 1.) w);
    zeros [| 1; 5; 4 |] (Nx.conv_transpose (f32 [| 1; 6; 2 |] 1.) w)

  (* A [[1; 0; 1]] input at stride 3 and padding (1, 1) has one window: the
     transpose of that count has an input to find. *)
  let test_transpose_short () =
    let ones s = Nx.ones Nx.float64 s in
    let x = Nx.create Nx.float64 [| 1; 3; 1 |] [| 1.; 0.; 1. |] in
    let stride = [| 3 |] and padding = [| (1, 1) |] in
    let y = Nx.conv ~stride ~padding x (ones [| 1; 3; 1 |]) in
    equal (array int) [| 1; 1; 1 |] (Nx.shape y);
    let z = Nx.conv_transpose ~stride ~padding (ones [| 1; 1; 1 |]) (ones [| 1; 1; 1 |]) in
    equal (array int) [| 1; 0; 1 |] (Nx.shape z)

  let group =
    group "convolution"
      [
        law_conv;
        law_transpose;
        test "float16 sums at float32" test_narrow;
        test "integers wrap" test_wrap;
        test "conv_transpose finds an input shorter than its formula's"
          test_transpose_short;
        test "each refusal names its function" test_refusals;
        test "a kernel of no tap sums no term" test_no_taps;
        test "a dead operand names the function called" test_dead_names;
      ]
end

let () =
  exit
    (run "nx contractions"
       [
         within_bound;
         complex_within_bound;
         widened;
         bitwise;
         long_sum;
         einsum_is_contract;
         einsum_within_bound;
         einsum_deterministic;
         einsum_order;
         einsum_errors;
         lowering;
         init;
         dtypes;
         matmul;
         dot_is_einsum;
         placements;
         errors;
         group "scales and segments"
           [ Scaled.within_bound; Scaled.placements; Scaled.partials ];
         Conv.group;
       ])
