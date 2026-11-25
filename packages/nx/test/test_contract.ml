(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions: Nx.contract against a reference that reads the pattern's
   meaning off its names, within the stated bound; einsum of two operands and
   matmul against contract, and of more against a sum over every index tuple. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let numel s = Array.fold_left ( * ) 1 s

let host dt s xs =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array s) xs)

let read x =
  A.to_array (Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* A float32 operand of shape [s] whose elements are all [1]. *)
let ones s = host D.Float32 s (Array.make (numel s) 1.)

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

(* [β(n)] as nx_kernel.mli's error bounds count it: the roundings a term of a
   float sum of [n] terms passes, added in blocks of at most 256. *)
let beta n = (((5 * min n 256) + 3) / 4) + ((n + 127) / 128)

(* Cases *)

(* A pattern, its operands' shapes, and its meaning. *)
type case = {
  pattern : string;
  sa : int array;
  sb : int array;
  meaning : meaning;
}

let extent_of l n = List.assoc n l

let case pattern ~a ~b ~result ~summed extents =
  let extent = extent_of extents in
  let shape names = Array.of_list (List.map extent names) in
  {
    pattern;
    sa = shape a;
    sb = shape b;
    meaning = { extent; result; summed; a; b };
  }

let matmul_like m n k =
  case "i k, k j -> i j | k" ~a:[ "i"; "k" ] ~b:[ "k"; "j" ]
    ~result:[ "i"; "j" ] ~summed:[ "k" ]
    [ ("i", m); ("j", n); ("k", k) ]

let outer m n =
  case "i, j -> i j" ~a:[ "i" ] ~b:[ "j" ] ~result:[ "i"; "j" ] ~summed:[]
    [ ("i", m); ("j", n) ]

let cases_of_sizes =
  let open Gen in
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
           [ ("b", 2); ("i", m); ("j", n); ("k", k) ]
     | 2 ->
         case "k i, k j -> j i | k" ~a:[ "k"; "i" ] ~b:[ "k"; "j" ]
           ~result:[ "j"; "i" ] ~summed:[ "k" ]
           [ ("i", m); ("j", n); ("k", k) ]
     | 3 -> outer m n
     | 4 ->
         case "i j, i j -> | i j" ~a:[ "i"; "j" ] ~b:[ "i"; "j" ] ~result:[]
           ~summed:[ "i"; "j" ]
           [ ("i", m); ("j", n + k) ]
     | _ ->
         case "h i k, k h j -> h j i | k" ~a:[ "h"; "i"; "k" ]
           ~b:[ "k"; "h"; "j" ] ~result:[ "h"; "j"; "i" ] ~summed:[ "k" ]
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

(* Generated plain patterns *)

(* A plain pattern of two operands drawn from [st], with its operands' shapes,
   the [sizes] it needs and its meaning. Each name is a batch, summed, a-free or
   b-free name, and each layout and the sum list their names in an order of
   their own. At times every layout leads with [...] (in the meaning, batch
   names [e0], [e1]), a layout holds a unit axis, and a layout groups two
   adjacent names, an operand's group given its first name's extent. Operands
   have ranks 0 to 4. *)
type plain = {
  text : string;
  ellipsis : bool;
  unit : bool;
  grouped : bool;
  sizes : (string * int) list;
  pa : int array;
  pb : int array;
  pm : meaning;
}

let pp_plain ppf c =
  Format.fprintf ppf "%S ~sizes:[%s] a %a b %a" c.text
    (String.concat "; "
       (List.map (fun (n, e) -> Printf.sprintf "%s %d" n e) c.sizes))
    pp_shape c.pa pp_shape c.pb

let shuffled st l =
  List.map snd
    (List.sort compare (List.map (fun x -> (Random.State.bits st, x)) l))

(* A layout of [names]: its tokens, with a unit axis at [unit] and the names at
   [group] and after grouped, where given. *)
type token = Name of string | Group of string * string | Unit

let tokens ?unit ?group names =
  let rec go i = function
    | [] -> []
    | n :: m :: rest when group = Some i -> Group (n, m) :: go (i + 2) rest
    | n :: rest -> Name n :: go (i + 1) rest
  in
  let ts = go 0 names in
  match unit with
  | None -> ts
  | Some u ->
      List.filteri (fun i _ -> i < u) ts
      @ (Unit :: List.filteri (fun i _ -> i >= u) ts)

let layout ~ellipsis ts =
  String.concat " "
    ((if ellipsis then [ "..." ] else [])
    @ List.map
        (function
          | Name n -> n
          | Group (n, m) -> Printf.sprintf "(%s %s)" n m
          | Unit -> "1")
        ts)

let plain_of_seed seed =
  let st = Random.State.make [| seed |] in
  let int n = Random.State.int st n and chance k = Random.State.int st k = 0 in
  let rec draw () =
    let count () = int 3 in
    let nb = count () and ns = count () and nfa = count () and nfb = count () in
    let ellipsis = chance 4 in
    let lead = if ellipsis then int 3 else 0 in
    let fresh p k = List.init k (fun i -> Printf.sprintf "%s%d" p i) in
    let batch = fresh "t" nb and summed = fresh "s" ns in
    let free_a = fresh "f" nfa and free_b = fresh "g" nfb in
    let na = shuffled st (batch @ summed @ free_a)
    and nb' = shuffled st (batch @ summed @ free_b)
    and nr = shuffled st (batch @ free_a @ free_b) in
    let side names =
      let unit =
        if chance 5 then Some (int (List.length names + 1)) else None
      in
      let group =
        if List.length names >= 2 && chance 3 then
          Some (int (List.length names - 1))
        else None
      in
      tokens ?unit ?group names
    in
    let ta = side na and tb = side nb' and tr = side nr in
    let rank ts = lead + List.length ts in
    if rank ta > 4 || rank tb > 4 || rank tr > 4 then draw ()
    else
      let leads = List.init lead (fun _ -> 1 + int 3) in
      let firsts =
        List.concat_map
          (function Group (n, _) -> [ n ] | Name _ | Unit -> [])
          (ta @ tb)
      in
      let extents =
        List.map
          (fun n ->
            (n, if List.mem n firsts || not (chance 12) then 1 + int 3 else 0))
          (batch @ summed @ free_a @ free_b)
        @ List.mapi (fun i e -> (Printf.sprintf "e%d" i, e)) leads
      in
      let extent n = List.assoc n extents in
      let shape ts =
        Array.of_list
          (leads
          @ List.map
              (function
                | Name n -> extent n
                | Group (n, m) -> extent n * extent m
                | Unit -> 1)
              ts)
      in
      let es = List.init lead (Printf.sprintf "e%d") in
      let sizes =
        List.sort_uniq compare (List.map (fun n -> (n, extent n)) firsts)
      in
      let has f = List.exists f (ta @ tb @ tr) in
      {
        ellipsis;
        unit = has (( = ) Unit);
        grouped = has (function Group _ -> true | Name _ | Unit -> false);
        text =
          Printf.sprintf "%s, %s -> %s%s" (layout ~ellipsis ta)
            (layout ~ellipsis tb) (layout ~ellipsis tr)
            (if summed = [] then ""
             else " | " ^ String.concat " " (shuffled st summed));
        sizes;
        pa = shape ta;
        pb = shape tb;
        pm = { extent; result = es @ nr; summed; a = es @ na; b = es @ nb' };
      }
  in
  draw ()

let plains =
  Gen.with_pp pp_plain (Gen.map plain_of_seed (Gen.int_range 0 0x3FFFFFFF))

let int32_operand st n = Array.init n (fun _ -> Random.State.bits32 st)

type float_dtype = F : (float, 's) D.t -> float_dtype

let pp_float_dtype ppf (F dt) = Format.pp_print_string ppf (D.name dt)

(* A contraction *)

module S2 = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

(* [text] with each of its names [rename] maps written as its image. *)
let renamed rename text =
  let out = Buffer.create (String.length text) and word = Buffer.create 4 in
  let flush () =
    let w = Buffer.contents word in
    Buffer.add_string out (Option.value (List.assoc_opt w rename) ~default:w);
    Buffer.clear word
  in
  String.iter
    (fun ch ->
      match ch with
      | 'a' .. 'z' | '0' .. '9' -> Buffer.add_char word ch
      | _ ->
          flush ();
          Buffer.add_char out ch)
    text;
  flush ();
  Buffer.contents out

(* [text] with the names after its [|] listed in reverse. *)
let reversed_sum text =
  match String.index_opt text '|' with
  | None -> text
  | Some i ->
      let sum = String.sub text (i + 1) (String.length text - i - 1) in
      let names = String.split_on_char ' ' (String.trim sum) in
      String.sub text 0 (i + 1) ^ " " ^ String.concat " " (List.rev names)

(* [x]'s values through a layout of other strides: a transposed copy, transposed
   back. *)
let restrided x = Nx.transpose (Nx.copy (Nx.transpose x))

(* Whether some output's products, summed in float32 in C order and in reverse,
   give other bits: on a draw where every order rounds alike, one association
   cannot be told from another. *)
let orders_differ m xa xb =
  let sum ts =
    List.fold_left (fun s (i, j) -> f32 (s +. (xa.(i) *. xb.(j)))) 0. ts
  in
  List.exists
    (fun ts ->
      not
        (Int64.equal
           (Int64.bits_of_float (sum ts))
           (Int64.bits_of_float (sum (List.rev ts)))))
    (terms m)

(* The same call again, the summed names listed in another order, every name
   exchanged for another and operands of other strides: each is the same
   contraction of the same shapes and values, so each gives the same bits,
   through contract and einsum alike. *)
let law_shapes_and_values (c, F dt, seed) =
  let st = Random.State.make [| seed |] in
  let draw s =
    host dt s (Array.init (numel s) (fun _ -> Random.State.float st 2. -. 1.))
  in
  let a = draw c.pa and b = draw c.pb in
  let names = List.sort_uniq compare (c.pm.a @ c.pm.b) in
  let rename = List.combine names (shuffled st names) in
  cover "two summed names" (List.length c.pm.summed = 2);
  cover "an operand of two axes or more"
    (Array.length c.pa >= 2 || Array.length c.pb >= 2);
  cover "two association orders differ" (orders_differ c.pm (read a) (read b));
  let bits text sizes a b =
    let p = Nx.Pattern.v text in
    (read (Nx.contract ~sizes dt p a b), read (Nx.einsum ~sizes p [ a; b ]))
  in
  let results = pair (array float_exact) (array float_exact) in
  let want = bits c.text c.sizes a b in
  equal ~msg:"again" results want (bits c.text c.sizes a b);
  equal ~msg:"the summed names listed in reverse" results want
    (bits (reversed_sum c.text) c.sizes a b);
  equal ~msg:"every name exchanged for another" results want
    (bits (renamed rename c.text)
       (List.map (fun (n, e) -> (List.assoc n rename, e)) c.sizes)
       a b);
  equal ~msg:"operands of other strides" results want
    (bits c.text c.sizes (restrided a) (restrided b))

let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

let contraction =
  let x s = host D.Float32 s (operand (numel s)) in
  let p = Nx.Pattern.v "i k, k j -> i j | k" in
  group "a contraction"
    [
      prop ~count:200
        "has bits that are a function of its operands' shapes and values alone"
        Gen.(
          triple plains
            (of_list ~pp:pp_float_dtype [ F D.Float32; F D.Bfloat16 ])
            int)
        law_shapes_and_values;
      test
        "of float32 [1e7; -1e7; 0.1; 0] by ones is 0 on one device and 0.1 cut \
         two ways along its summed axis" (fun () ->
          (* On one device, lane order adds 1e7 and 0.1, which rounds to 1e7,
             and -1e7 and 0, then the two: 0. Cut, each device sums its two
             products, 0 and 0.1, and the closing sums the two. *)
          let a = host D.Float32 [| 1; 4 |] [| 1e7; -1e7; 0.1; 0. |] in
          let b = host D.Float32 [| 4; 1 |] [| 1.; 1.; 1.; 1. |] in
          let a' = Nx.place (S2.split ~axis:1) a in
          let b' = Nx.place (S2.split ~axis:0) b in
          equal ~msg:"one device" (array float_exact) [| 0. |]
            (read (Nx.contract Nx.float32 p a b));
          equal ~msg:"cut" (array float_exact)
            [| f32 0.1 |]
            (read (Nx.contract Nx.float32 p a' b')));
      test "of integers cut along its summed axis keeps its one-device bits"
        (fun () ->
          (* Draws over every int32, so that the sums wrap. *)
          let st = Random.State.make [| 1 |] in
          let a = host D.Int32 [| 3; 8 |] (int32_operand st 24) in
          let b = host D.Int32 [| 8; 2 |] (int32_operand st 16) in
          let cut =
            Nx.contract Nx.int32 p
              (Nx.place (S2.split ~axis:1) a)
              (Nx.place (S2.split ~axis:0) b)
          in
          equal (array int32) (read (Nx.contract Nx.int32 p a b)) (read cut));
      test "whose first windows hold every product has the one-device bits"
        (fun () ->
          (* The summed axis has one element: the first device's window holds it
             and the second's none, which adds -0 where there is an init. *)
          let random s =
            host D.Float32 s
              (Array.init (numel s) (fun _ -> Random.float 2. -. 1.))
          in
          let a = random [| 4; 1 |] and b = random [| 1; 3 |] in
          let init = random [| 4; 3 |] in
          let cut =
            Nx.contract ~init:(Nx.place S2.on init) Nx.float32 p
              (Nx.place (S2.split ~axis:1) a)
              (Nx.place (S2.split ~axis:0) b)
          in
          equal (array float_exact)
            (read (Nx.contract ~init Nx.float32 p a b))
            (read cut));
      test "of operands on every device has the host's bits" (fun () ->
          let a = x [| 4; 3 |] and b = x [| 3; 2 |] in
          let y =
            Nx.contract Nx.float32 p (Nx.place S2.on a) (Nx.place S2.on b)
          in
          equal (array float_exact)
            (read (Nx.contract Nx.float32 p a b))
            (read y));
      test
        "reads an operand whole on every device at the other's cut, which the \
         result keeps" (fun () ->
          let a = x [| 16; 3 |] and b = x [| 3; 8 |] in
          let split = S2.split ~axis:0 in
          let y =
            Nx.contract Nx.float32 p (Nx.place split a) (Nx.place S2.on b)
          in
          equal ~msg:"the result's cut" placement split (Nx.placement y);
          equal (array float_exact)
            (read (Nx.contract Nx.float32 p a b))
            (read y));
      (* Rows and columns cut over the same two devices would need four windows
         of the result: the second operand is read whole, which its devices do
         not hold. *)
      test
        "raises naming the place to write where rows and columns are cut over \
         the same devices" (fun () ->
          let a = Nx.place (S2.split ~axis:0) (x [| 4; 3 |]) in
          let b = Nx.place (S2.split ~axis:1) (x [| 3; 2 |]) in
          raises
            (Invalid_argument
               "Nx.contract: operand 1 lies at split ~axis:1 [m0; m1] and is \
                read at on [m0; m1]; Nx.place (Placement.on s) y") (fun () ->
              Nx.contract Nx.float32 p a b));
      test
        "raises naming the place to write where operands lie alone on two \
         devices" (fun () ->
          let a = x [| 4; 3 |] and b = x [| 3; 2 |] in
          let on k =
            Nx.place (Nx.Placement.device S2.v (Nx_support.memory k))
          in
          raises
            (Invalid_argument
               "Nx.contract: operand 1 lies at m1 and is read at m0; Nx.place \
                (Placement.device s m0) y") (fun () ->
              Nx.contract Nx.float32 p (on 0 a) (on 1 b)));
    ]

(* contract *)

let law_within_bound (c, with_init) =
  let k = numel (Array.of_list (List.map c.meaning.extent c.meaning.summed)) in
  cover "no summed name" (c.meaning.summed = []);
  cover "an empty sum" (k = 0);
  cover "an init" with_init;
  let xa = operand (numel c.sa) and xb = operand (numel c.sb) in
  let a = host D.Float32 c.sa xa and b = host D.Float32 c.sb xb in
  let shape = Array.of_list (List.map c.meaning.extent c.meaning.result) in
  let xi =
    if with_init then operand (numel shape) else Array.make (numel shape) 0.
  in
  let init = if with_init then Some (host D.Float32 shape xi) else None in
  let y = read (Nx.contract ?init Nx.float32 (Nx.Pattern.v c.pattern) a b) in
  let u = Float.ldexp 1. (-24) in
  let g = gamma (beta k) u in
  List.iteri
    (fun o (sum, abs) ->
      let r = xi.(o) +. sum in
      let rounding = if with_init then u *. Float.abs r else 0. in
      at_most float_exact
        ~than:(rounding +. ((1. +. u) *. g *. abs))
        (Float.abs (y.(o) -. r)))
    (reference c.meaning xa xb)

(* Complex operands whose parts are {!operand}'s: each part of every product and
   partial sum is exact in float64. *)
let law_complex_within_bound c =
  let k = numel (Array.of_list (List.map c.meaning.extent c.meaning.summed)) in
  let draw n =
    let re = operand n and im = operand n in
    Array.init n (fun i -> { Complex.re = re.(i); im = im.(i) })
  in
  let xa = draw (numel c.sa) and xb = draw (numel c.sb) in
  let a = host D.Complex64 c.sa xa and b = host D.Complex64 c.sb xb in
  let y = read (Nx.contract Nx.complex64 (Nx.Pattern.v c.pattern) a b) in
  let u = Float.ldexp 1. (-24) in
  let g = Float.sqrt 2. *. gamma (beta (2 * k)) u in
  List.iteri
    (fun o ts ->
      let exact, abs =
        List.fold_left
          (fun (s, abs) (i, j) ->
            ( Complex.add s (Complex.mul xa.(i) xb.(j)),
              abs +. (Complex.norm xa.(i) *. Complex.norm xb.(j)) ))
          (Complex.zero, 0.) ts
      in
      at_most float_exact
        ~than:((1. +. u) *. g *. abs)
        (Complex.norm (Complex.sub y.(o) exact)))
    (terms c.meaning)

(* [x] and [y] are the same value of [dt]: a float by its bits, NaNs as one
   class, a complex part by part. *)
let same (type v s) (dt : (v, s) D.t) (x : v) (y : v) =
  let float a b =
    Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
    || (Float.is_nan a && Float.is_nan b)
  in
  match D.kind dt with
  | Float -> float x y
  | Complex -> float x.Complex.re y.Complex.re && float x.im y.im
  | Signed | Unsigned | Boolean -> x = y

let value dt = Testable.make ~pp:(D.pp_value dt) ~equal:(same dt)
let values x = A.to_array (Option.get (Nx.Repr.array ~by:"test" x))

(* The elements of [dt] whose little-endian bytes are [bytes]. *)
let of_bytes (type v s) (dt : (v, s) D.t) bytes : (v, s, Nx.host) Nx.t =
  let n = Array.length bytes in
  let u = host D.Uint8 [| n |] bytes in
  let bits = D.bits dt in
  if bits < 8 then Nx.reshape [| n * 8 / bits |] (Nx.bitcast dt u)
  else if bits = 8 then Nx.bitcast dt u
  else Nx.bitcast dt (Nx.reshape [| n * 8 / bits; bits / 8 |] u)

(* Bit patterns that break a conversion that is not exact: the integer bounds
   and neighbours of each narrower dtype, the integers past each float's
   significand, and a half, the significand's last bit, the largest finite
   value, an infinity and a NaN of float32 and float64. *)
let patterns32 =
  [
    0L;
    1L;
    0xFFFFFFFFL;
    0x80000000L;
    0x7FFFFFFFL;
    0x1000001L;
    0x8000L;
    0x10000L;
    0x3F000000L;
    0x3F800001L;
    0x7F7FFFFFL;
    0x7F800000L;
    0x7FC00000L;
  ]

let patterns64 =
  [
    0L;
    1L;
    -1L;
    Int64.min_int;
    Int64.max_int;
    0x20000000000001L;
    0x80000000L;
    0x100000000L;
    0x3FE0000000000000L;
    0x3FF0000000000001L;
    0x7FEFFFFFFFFFFFFFL;
    0x7FF0000000000000L;
    0x7FF8000000000000L;
  ]

let bytes_of k p =
  List.init k (fun i ->
      Int64.(to_int (logand (shift_right_logical p (8 * i)) 0xFFL)))

(* Every value of a dtype of at most 16 bits, each bit pattern; for a wider one,
   the patterns above, a complex128's parts two neighbouring ones. *)
let probes (type v s) (dt : (v, s) D.t) : (v, s, Nx.host) Nx.t =
  let bytes ps k = Array.of_list (List.concat_map (bytes_of k) ps) in
  match D.bits dt with
  | _ when D.equal dt D.Bool -> of_bytes dt [| 0; 1 |]
  | 16 ->
      of_bytes dt
        (Array.init 131072 (fun i ->
             if i land 1 = 0 then (i lsr 1) land 0xFF else i lsr 9))
  | b when b < 16 -> of_bytes dt (Array.init 256 Fun.id)
  | 32 -> of_bytes dt (bytes patterns32 4)
  | 64 -> of_bytes dt (bytes patterns64 8)
  | _ ->
      let next = List.tl patterns64 @ [ List.hd patterns64 ] in
      let parts = List.map2 (fun a b -> [ a; b ]) patterns64 next in
      of_bytes dt (bytes (List.concat parts) 8)

(* A real number as two floats that no other number gives: an integer of
   magnitude below 2^64 as its multiple of 2^32 and the rest, in [[0, 2^32)],
   any other float as itself and [0.]. Under [Float.equal] every NaN is one
   number and [-0.] is [0]. *)
let real f =
  if Float.is_integer f && Float.abs f < 0x1p64 then
    let hi = Float.ldexp (Float.floor (Float.ldexp f (-32))) 32 in
    (hi, f -. hi)
  else (f, 0.)

(* An int64 [x] read signed or unsigned, as {!real} gives it: the upper 32 bits
   times 2^32 and the lower. *)
let low x = Int64.to_float (Int64.logand x 0xFFFFFFFFL)
let signed x = (Float.ldexp (Int64.to_float (Int64.shift_right x 32)) 32, low x)

let unsigned x =
  (Float.ldexp (Int64.to_float (Int64.shift_right_logical x 32)) 32, low x)

(* [x]'s number in [dt]: its real and imaginary parts, as {!real} gives them. *)
let number_of (type v s) (dt : (v, s) D.t) (x : v) :
    (float * float) * (float * float) =
  let zero = (0., 0.) in
  let small x = (signed (Int64.of_int x), zero) in
  match dt with
  | D.Float64 -> (real x, zero)
  | D.Float32 -> (real x, zero)
  | D.Float16 -> (real x, zero)
  | D.Bfloat16 -> (real x, zero)
  | D.Float8_e4m3fn -> (real x, zero)
  | D.Float8_e5m2 -> (real x, zero)
  | D.Float4_e2m1fn -> (real x, zero)
  | D.Int64 -> (signed x, zero)
  | D.Uint64 -> (unsigned x, zero)
  | D.Int32 -> (signed (Int64.of_int32 x), zero)
  | D.Uint32 -> (signed (Int64.logand (Int64.of_int32 x) 0xFFFFFFFFL), zero)
  | D.Int16 -> small x
  | D.Uint16 -> small x
  | D.Int8 -> small x
  | D.Uint8 -> small x
  | D.Int4 -> small x
  | D.Uint4 -> small x
  | D.Complex128 -> (real x.re, real x.im)
  | D.Complex64 -> (real x.re, real x.im)
  | D.Bool -> (real (if x then 1. else 0.), zero)
  | D.Bit -> (real (if x then 1. else 0.), zero)

let number =
  let part ppf (hi, lo) = Format.fprintf ppf "%h + %h" hi lo in
  let equal ((a, b), (c, d)) ((a', b'), (c', d')) =
    Float.equal a a' && Float.equal b b' && Float.equal c c' && Float.equal d d'
  in
  Testable.make ~equal ~pp:(fun ppf (re, im) ->
      Format.fprintf ppf "(%a) + (%a)i" part re part im)

let dtype_pairs =
  List.concat_map (fun a -> List.map (fun b -> (a, b)) D.all) D.all

let pair_name (D.Any a, D.Any b) = D.name a ^ " into " ^ D.name b

(* [Dtype.exact]'s statement read off {!Nx.cast}: a dtype is exact into another
   iff every probe keeps its number through the cast. *)
let law_exact (D.Any src, D.Any dt) =
  let x = probes src in
  let xs = values x and ys = values (Nx.cast dt x) in
  let changed =
    Seq.find_map
      (fun i ->
        let a = number_of src xs.(i) and b = number_of dt ys.(i) in
        if Testable.equal number a b then None else Some (i, a, b))
      (Seq.init (Array.length xs) Fun.id)
  in
  let first_change = option (triple int number number) in
  if D.exact src dt then equal first_change None changed
  else not_equal first_change None changed

(* Of an operand exact into [dt], [contract dt] is the contraction of its cast,
   bit for bit, over all its probes, one and none. *)
let law_exact_operand (D.Any src, D.Any dt) =
  let a = probes src in
  let b = Nx.cast dt (Nx.flip a) in
  let first k = Nx.slice [ Nx.R (0, k) ] in
  let check p a b =
    let p = Nx.Pattern.v p in
    equal
      (array (value dt))
      (values (Nx.contract dt p (Nx.cast dt a) b))
      (values (Nx.contract dt p a b))
  in
  check "k, k -> | k" a b;
  check "k, k -> | k" (first 1 a) (first 1 b);
  check "k, k -> | k" (first 0 a) (first 0 b);
  let n = min (Nx.numel a) 4 in
  check "i, j -> i j" (first n a) (first n b)

(* Rows of a source dtype, a result dtype, two operands as float64 values and
   their sum as the result's carrier gives it. *)
let carrier_rows =
  [
    (* 1 + 2^-25 is no float32. *)
    ( "float32 operands into float64 sum in float64",
      D.Any D.Float32,
      D.Any D.Float64,
      [| 1.; 0x1p-25 |],
      [| 1.; 1. |],
      1. +. 0x1p-25 );
    (* 1 + 2^-8 + 2^-8: each addition rounds away in bfloat16, and the float32
       sum 1 + 2^-7 is a bfloat16. *)
    ( "bfloat16 operands sum in float32 and round once",
      D.Any D.Bfloat16,
      D.Any D.Bfloat16,
      [| 1.; 0x1p-8; 0x1p-8 |],
      [| 1.; 1.; 1. |],
      1. +. 0x1p-7 );
    (* 1 + 2^-40 + 2^-40 is 1 in float32. *)
    ( "float64 operands sum in float64",
      D.Any D.Float64,
      D.Any D.Float64,
      [| 1.; 0x1p-40; 0x1p-40 |],
      [| 1.; 1.; 1. |],
      1. +. 0x1p-39 );
    (* 16384 + 16129 wraps in int8. *)
    ( "int8 operands into float32 sum in float32",
      D.Any D.Int8,
      D.Any D.Float32,
      [| -128.; 127. |],
      [| -128.; 127. |],
      32513. );
    (* 9e8 + 9e8 wraps in int16. *)
    ( "int16 operands into int32 sum in int32",
      D.Any D.Int16,
      D.Any D.Int32,
      [| 30000.; 30000. |],
      [| 30000.; 30000. |],
      1.8e9 );
    (* 100 + 100 is 200, which int8 wraps to -56. *)
    ( "int8 operands into int8 wrap",
      D.Any D.Int8,
      D.Any D.Int8,
      [| 100.; 100. |],
      [| 1.; 1. |],
      -56. );
  ]

let law_carrier (_, D.Any src, D.Any dt, xa, xb, want) =
  let operand xs = Nx.cast src (host D.Float64 [| Array.length xs |] xs) in
  let y =
    Nx.contract dt (Nx.Pattern.v "k, k -> | k") (operand xa) (operand xb)
  in
  equal (array float_exact) [| want |] (read (Nx.cast Nx.float64 y))

let narrow_floats =
  [
    F D.Float16;
    F D.Bfloat16;
    F D.Float8_e4m3fn;
    F D.Float8_e5m2;
    F D.Float4_e2m1fn;
  ]

(* A float result narrower than float32 sums in float32 and rounds once: from
   operands of its dtype it is the contraction of their float32 casts,
   rounded. *)
let law_narrow (c, F dt) =
  List.iter (fun (F d) -> cover (D.name d) (D.equal d dt)) narrow_floats;
  let p = Nx.Pattern.v c.pattern in
  let draw n = Array.init n (fun _ -> Random.float 2. -. 1.) in
  let a = host dt c.sa (draw (numel c.sa)) in
  let b = host dt c.sb (draw (numel c.sb)) in
  equal (array float_exact)
    (read (Nx.cast dt (Nx.contract Nx.float32 p a b)))
    (read (Nx.contract dt p a b))

(* At int32 the sum wraps, so every order of association gives the exact sum
   modulo 2^32: the result is the pattern's meaning, element for element, and
   with an [init] it is that sum plus [init]'s element, which reads the result's
   names in written order. *)
let law_int32 c =
  cover "a group" c.grouped;
  cover "an ellipsis" c.ellipsis;
  cover "a unit axis" c.unit;
  cover "a batch name"
    (List.exists (fun n -> List.mem n c.pm.b && List.mem n c.pm.result) c.pm.a);
  cover "a summed name" (c.pm.summed <> []);
  cover "an empty operand" (numel c.pa = 0 || numel c.pb = 0);
  cover "a 0-d operand" (c.pa = [||] || c.pb = [||]);
  cover "a rank-4 operand" (Array.length c.pa = 4 || Array.length c.pb = 4);
  let st = Random.State.make [| numel c.pa; numel c.pb |] in
  let xa = int32_operand st (numel c.pa)
  and xb = int32_operand st (numel c.pb) in
  let a = host D.Int32 c.pa xa and b = host D.Int32 c.pb xb in
  let p = Nx.Pattern.v c.text in
  let y = Nx.contract ~sizes:c.sizes Nx.int32 p a b in
  let want =
    Array.of_list
      (List.map
         (List.fold_left
            (fun s (i, j) -> Int32.add s (Int32.mul xa.(i) xb.(j)))
            0l)
         (terms c.pm))
  in
  equal (array int32) want (read y);
  let xi = int32_operand st (Array.length want) in
  let init = host D.Int32 (Nx.shape y) xi in
  equal ~msg:"with an init" (array int32)
    (Array.map2 Int32.add want xi)
    (read (Nx.contract ~sizes:c.sizes ~init Nx.int32 p a b))

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
      test "a group of no element solves its last name as 0" (fun () ->
          (* The quotient 0 / 12. *)
          let y =
            Nx.contract Nx.float32
              ~sizes:[ ("h", 3); ("d", 4) ]
              (Nx.Pattern.v "(h d u) k, k j -> h d u j | k")
              (x [| 0; 5 |])
              (x [| 5; 3 |])
          in
          equal (array int) [| 3; 4; 0; 3 |] (Nx.shape y));
    ]

let init =
  let a () = host D.Float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let b () = host D.Float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
  let c () = host D.Float32 [| 2; 2 |] [| 0.5; 0.25; 0.125; 2. |] in
  let p = Nx.Pattern.v "i k, k j -> i j | k" in
  group "init"
    [
      test "is added to the products' sum" (fun () ->
          equal (array float_exact)
            [| 19.5; 22.25; 43.125; 52. |]
            (read (Nx.contract ~init:(c ()) Nx.float32 p (a ()) (b ()))));
      (* Two products of [2^-24] and an init of [1]: added last, [init] meets
         their sum [2^-23] and gives [1 + 2^-23]; added first, each product
         would be a tie that rounds to [1]. Over 8 × 8 outputs, as every
         library's chain orders take them. *)
      test "joins after the products" (fun () ->
          let tiny = Nx.full Nx.float32 [| 8; 2 |] 0x1p-12 in
          let one = Nx.full Nx.float32 [| 8; 8 |] 1. in
          let y = Nx.contract ~init:one Nx.float32 p tiny (Nx.transpose tiny) in
          equal (array float_exact)
            (Array.make 64 (1. +. 0x1p-23))
            (Nx.to_array y));
      (* A row of [init] broadcast down the result: the bits of [init] given
         whole. *)
      test "broadcasts to the result's shape" (fun () ->
          let row = host D.Float32 [| 2 |] [| 0.5; 0.25 |] in
          let whole = Nx.copy (Nx.broadcast_to [| 2; 2 |] row) in
          equal (array float_exact)
            (read (Nx.contract ~init:whole Nx.float32 p (a ()) (b ())))
            (read (Nx.contract ~init:row Nx.float32 p (a ()) (b ()))));
    ]

(* The call [contract dt p] of float32 operands of ones of shapes [sa] and
   [sb]. *)
let contracting ?sizes dt p sa sb () =
  ignore (Nx.contract ?sizes dt (Nx.Pattern.v p) (ones sa) (ones sb))

let mm = "i k, k j -> i j | k"
let grouped = "(h d) k, k j -> h d j | k"
let named p n = List.init n (fun i -> p ^ string_of_int i)

(* A pattern whose result has [na + nb] axes. *)
let wide_result na nb =
  let words = String.concat " " in
  Printf.sprintf "%s k, k %s -> %s %s | k"
    (words (named "a" na))
    (words (named "b" nb))
    (words (named "a" na))
    (words (named "b" nb))

(* In nx.mli's order of contract's raises, then a result past a layout. The
   messages are baselines: nx.mli states the exception and the function. *)
let contract_refusals =
  [
    ( "a boolean result",
      contracting Nx.bool mm [| 2; 3 |] [| 3; 2 |],
      __POS_OF__ {|Nx.contract: the result's dtype is bool|} );
    ( "an operand not exact into the result",
      (fun () ->
        let i = host D.Int32 [| 3; 2 |] (Array.make 6 1l) in
        ignore (Nx.contract Nx.float32 (Nx.Pattern.v mm) (ones [| 2; 3 |]) i)),
      __POS_OF__
        {|Nx.contract: int32 b is not exact into float32 (Nx.Dtype.exact); write Nx.cast Nx.float32 b, or cast the result of a contraction into a wider dtype|}
    );
    ( "float operands into an integer result",
      contracting Nx.int32 mm [| 2; 3 |] [| 3; 2 |],
      __POS_OF__
        {|Nx.contract: float32 a is not exact into int32 (Nx.Dtype.exact); write Nx.cast Nx.int32 a, or cast the result of a contraction into a wider dtype|}
    );
    ( "one operand",
      contracting Nx.float32 "i k -> k i" [| 2; 3 |] [| 3; 2 |],
      __POS_OF__ {|Nx.contract: "i k -> k i" has one operand; use Nx.rearrange|}
    );
    ( "three operands",
      contracting Nx.float32 "i j, j k, k -> i | j k" [| 2; 3 |] [| 3; 4 |],
      __POS_OF__
        {|Nx.contract: "i j, j k, k -> i | j k" has 3 operands; use Nx.einsum|}
    );
    ( "a name an operand repeats",
      contracting Nx.float32 "i i, i -> i" [| 2; 2 |] [| 2 |],
      __POS_OF__
        {|Nx.contract: "i i, i -> i" takes a diagonal or sums within one operand; use Nx.einsum|}
    );
    ( "a summed name in one operand alone",
      contracting Nx.float32 "i j, k -> k | i j" [| 2; 3 |] [| 4 |],
      __POS_OF__
        {|Nx.contract: "i j, k -> k | i j" takes a diagonal or sums within one operand; use Nx.einsum|}
    );
    ( "a rank the pattern does not give",
      contracting Nx.float32 mm [| 2; 3; 4 |] [| 4; 2 |],
      __POS_OF__
        {|Nx.contract: "i k, k j -> i j | k" names 2 axes for a; a is float32 [2; 3; 4]|}
    );
    ( "fewer axes than the names beside ...",
      contracting Nx.float32 "... i k, ... k j -> ... i j | k" [| 3 |]
        [| 3; 2 |],
      __POS_OF__
        {|Nx.contract: "... i k, ... k j -> ... i j | k" names at least 2 axes for a; a is float32 [3]|}
    );
    ( "... over different numbers of axes",
      contracting Nx.float32 "... i, ... i -> ... | i" [| 2; 3 |] [| 2; 2; 3 |],
      __POS_OF__
        {|Nx.contract: "... i, ... i -> ... | i": ... is 1 axis in a and 2 in b|}
    );
    ( "paired extents that differ",
      contracting Nx.float32 mm [| 2; 3 |] [| 4; 2 |],
      __POS_OF__ {|Nx.contract: "i k, k j -> i j | k": k is 3 in a and 4 in b|}
    );
    ( "axes under ... that differ",
      contracting Nx.float32 "... i k, ... k j -> ... i j | k" [| 2; 2; 3 |]
        [| 3; 3; 2 |],
      __POS_OF__
        {|Nx.contract: "... i k, ... k j -> ... i j | k": ...0 is 2 in a and 3 in b|}
    );
    ( "a size an operand contradicts",
      contracting ~sizes:[ ("k", 5) ] Nx.float32 mm [| 2; 3 |] [| 3; 2 |],
      __POS_OF__
        {|Nx.contract: "i k, k j -> i j | k": k is 5 in ~sizes and 3 in a|} );
    ( "a group that does not divide",
      contracting ~sizes:[ ("h", 4) ] Nx.float32 grouped [| 10; 5 |] [| 5; 3 |],
      __POS_OF__
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) is 10 in a, which 4 does not divide|}
    );
    ( "a known zero in a group of elements",
      contracting ~sizes:[ ("h", 0) ] Nx.float32 grouped [| 10; 5 |] [| 5; 3 |],
      __POS_OF__
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) is 10 in a, and its names multiply to 0|}
    );
    ( "a group whose names multiply past an int",
      contracting
        ~sizes:[ ("h", 1 lsl 31); ("d", 1 lsl 32) ]
        Nx.float32 grouped [| 0; 5 |] [| 5; 3 |],
      __POS_OF__
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) is 0 in a, and its names multiply to more than an int counts|}
    );
    ( "a group with two unknowns",
      contracting Nx.float32 grouped [| 12; 5 |] [| 5; 3 |],
      __POS_OF__
        {|Nx.contract: "(h d) k, k j -> h d j | k": (h d) in a leaves h and d unknown; give one in ~sizes|}
    );
    ( "a unit axis of another extent",
      contracting Nx.float32 "i 1 k, k j -> i j | k" [| 3; 2; 4 |] [| 4; 2 |],
      __POS_OF__
        {|Nx.contract: "i 1 k, k j -> i j | k": axis 1 of a is 1 in the pattern and 2 in a|}
    );
    ( "a size the pattern lacks",
      contracting ~sizes:[ ("z", 2) ] Nx.float32 mm [| 2; 3 |] [| 3; 2 |],
      __POS_OF__
        {|Nx.contract: "i k, k j -> i j | k": ~sizes names z, which it lacks|}
    );
    ( "a size given twice",
      contracting
        ~sizes:[ ("h", 2); ("h", 2) ]
        Nx.float32 grouped [| 10; 5 |] [| 5; 3 |],
      __POS_OF__
        {|Nx.contract: "(h d) k, k j -> h d j | k": h is twice in ~sizes|} );
    ( "a negative size",
      contracting ~sizes:[ ("h", -2) ] Nx.float32 grouped [| 10; 5 |] [| 5; 3 |],
      __POS_OF__
        {|Nx.contract: "(h d) k, k j -> h d j | k": h = -2 in ~sizes is negative|}
    );
    ( "an init of another shape",
      (fun () ->
        ignore
          (Nx.contract
             ~init:(ones [| 3; 2 |])
             Nx.float32 (Nx.Pattern.v mm)
             (ones [| 2; 3 |])
             (ones [| 3; 2 |]))),
      __POS_OF__
        {|Nx.contract: ~init is [3; 2], which does not broadcast to the result's [2; 2]|}
    );
    ( "a result of more axes than a layout",
      contracting Nx.float32 (wide_result 16 17) (Array.make 17 1)
        (Array.make 18 1),
      __POS_OF__ {|Nx.contract: a result of rank 33|} );
    ( "a result group past an int",
      contracting Nx.float32 "h k, k d -> (h d) | k"
        [| 1 lsl 31; 0 |]
        [| 0; 1 lsl 32 |],
      __POS_OF__
        {|Nx.contract: "h k, k d -> (h d) | k": (h d) in the result multiplies past an int|}
    );
  ]

(* Raises [Invalid_argument] naming [name], with the message [text]. *)
let refuses name (_, f, text) =
  let m = message f in
  starts_with ~prefix:(name ^ ": ") m;
  expect m text

let refusal_id (id, _, _) = id

let contract =
  group "contract"
    [
      prop
        "is within u|r| + (1 + u) γ(β(K), u) Σ|a||b| of the exact r = init + Σ \
         a·b"
        ~examples:[ (matmul_like 2 3 0, true); (outer 2 3, false) ]
        Gen.(pair cases_of_sizes bool)
        law_within_bound;
      prop
        "of complex operands is within √2 γ(β(2K), u) Σ|a||b| of the exact sum"
        cases_of_sizes law_complex_within_bound;
      cases "Dtype.exact holds iff a cast keeps every probe's number"
        ~id:pair_name dtype_pairs law_exact;
      cases "contracts an operand exact into the result as its cast"
        ~id:pair_name
        (List.filter
           (fun (D.Any src, D.Any dt) ->
             D.exact src dt && not (D.is D.Boolean dt))
           dtype_pairs)
        law_exact_operand;
      cases "raises naming itself for an operand not exact into the result"
        ~id:pair_name
        (List.filter
           (fun (D.Any src, D.Any dt) ->
             (not (D.exact src dt)) && not (D.is D.Boolean dt))
           dtype_pairs)
        (fun (D.Any src, D.Any dt) ->
          let a = Nx.zeros src [| 1 |] in
          starts_with ~prefix:"Nx.contract: "
            (message (fun () ->
                 Nx.contract dt (Nx.Pattern.v "k, k -> | k") a (Nx.cast dt a))));
      cases "sums in its result's carrier"
        ~id:(fun (id, _, _, _, _, _) -> id)
        carrier_rows law_carrier;
      test
        "of float32 operands into complex64 is their sum with imaginary part +0"
        (fun () ->
          let a = host D.Float32 [| 2 |] [| 1.; 2. |] in
          let b = host D.Float32 [| 2 |] [| 3.; 4. |] in
          let z =
            read (Nx.contract Nx.complex64 (Nx.Pattern.v "k, k -> | k") a b)
          in
          equal (array float_exact) [| 11.; 0. |] [| z.(0).re; z.(0).im |]);
      test "of a real operand by a complex infinity has a NaN imaginary part"
        (fun () ->
          (* (2 + 0i)(inf + 0i) = 2 inf - 0·0 + (2·0 + 0·inf)i; as a real scale
             2 (inf + 0i) it would be inf + 0i. *)
          let a = host D.Float32 [| 1 |] [| 2. |] in
          let b =
            host D.Complex64 [| 1 |] [| { Complex.re = infinity; im = 0. } |]
          in
          let z =
            read (Nx.contract Nx.complex64 (Nx.Pattern.v "k, k -> | k") a b)
          in
          equal (array float_exact) [| infinity; nan |] [| z.(0).re; z.(0).im |]);
      prop ~count:150
        "into a float narrower than float32 is its float32 contraction rounded \
         once"
        Gen.(pair cases_of_sizes (of_list ~pp:pp_float_dtype narrow_floats))
        law_narrow;
      test
        "gives an output of no products init's element, -0 included, or +0 \
         without init" (fun () ->
          let a = host D.Float32 [| 2; 0 |] [||] in
          let b = host D.Float32 [| 0; 3 |] [||] in
          let init = host D.Float32 [| 2; 3 |] (Array.make 6 (-0.)) in
          let p = Nx.Pattern.v mm in
          equal ~msg:"an init of -0" (array float_exact) (Array.make 6 (-0.))
            (read (Nx.contract ~init Nx.float32 p a b));
          equal ~msg:"no init" (array float_exact) (Array.make 6 0.)
            (read (Nx.contract Nx.float32 p a b)));
      prop ~count:500
        "of int32 operands is a plain pattern's wrapping sum by index, then \
         init"
        plains law_int32;
      lowering;
      init;
      cases "raises Invalid_argument naming itself" ~id:refusal_id
        contract_refusals (refuses "Nx.contract");
    ]

(* einsum *)

(* An axis of a generated layout: a name, two names grouped, or [1]. *)
type axis = N of string | G of string * string | U

let axis_text = function
  | N n -> n
  | G (a, b) -> "(" ^ a ^ " " ^ b ^ ")"
  | U -> "1"

let axis_names = function N n -> [ n ] | G (a, b) -> [ a; b ] | U -> []

(* A pattern of several operands with its operands' shapes, and its meaning:
   each operand's names and the result's in C order, those [...] covers first,
   and the summed names. *)
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

(* One to five operands; free names and summed names, each held by one operand
   or more; extents from 0 to 3; at times [...] over one or two leading axes, a
   group in an operand where its extents can be learned, a name an operand
   repeats, and a group and a [1] in the result. *)
let problems =
  let open Gen in
  let extent =
    frequency [ (1, constant 0); (2, constant 1); (6, int_range 2 3) ]
  in
  let* n = int_range 1 5 in
  let ops = List.init n Fun.id in
  let held least =
    such_that (fun h -> List.length h >= least) (subsequence ops)
  in
  let* frees = list ~size:(int_range 0 3) (pair extent (held 1)) in
  let* sums = list ~size:(int_range 0 3) (pair extent (held 1)) in
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
  let* repeated = int_range (-1) (n - 1) in
  let* again = int_range 1 2 in
  let* unit = bool in
  let* result = permutation (List.map (fun (nm, _, _) -> nm) frees) in
  let* result_group, result_unit = pair bool bool in
  (* A group in operand [i] whose extents another operand gives, or whose known
     extent is not 0 and divides out the other. *)
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
        (* The first name again, once or twice, last. *)
        let l =
          match List.concat_map axis_names l with
          | nm :: _ when i = repeated -> l @ List.init again (fun _ -> N nm)
          | _ -> l
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

(* The exact sum, in float64, of the products of [xs]'s elements at every index
   tuple for each result index in C order, with the sum of their magnitudes:
   exact while every product and partial sum fits 53 bits. *)
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

(* The least work of a tree of contractions of [p]'s operands, and the greatest
   [N = Σ (K + 1)] among the trees doing it plus that of the sums within one
   operand, by einsum's definitions: a contraction does the product of the
   extents of the names its parts' values hold, a value holds the names its
   operands hold that the result or an operand outside it needs, and a
   contraction sums [K] products, the product of the extents of the names it
   meets and its value drops. Every tree is drawn, over the few operands and
   names [problems] draws, all within the layout's ranks. *)
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
  (* An operand's summed names that no other holds, summed in it first by a
     contraction with ones that sums [K] products; one operand alone always
     takes that contraction. *)
  let alone i =
    List.filter
      (fun nm -> (not (List.mem nm p.out)) && not (outside [ i ] nm))
      (List.filter (fun nm -> List.mem nm ops.(i)) names)
  in
  let first =
    List.fold_left
      (fun k i ->
        match alone i with
        | [] when Array.length ops > 1 -> k
        | l -> k + product l + 1)
      0
      (List.init (Array.length ops) Fun.id)
  in
  let all = trees (List.init (Array.length ops) Fun.id) in
  let least = List.fold_left (fun w (w', _) -> min w w') max_int all in
  ( least,
    first
    + List.fold_left (fun n (w, n') -> if w = least then max n n' else n) 0 all
  )

let float_dtypes =
  Gen.of_list ~pp:pp_float_dtype
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
  let r = Random.State.make [| seed |] in
  Array.init n (fun _ ->
      Float.ldexp
        (Float.of_int (Random.State.int r 9 - 4))
        (Random.State.int r 5 - 2))

let law_einsum_within_bound (p, F dt, seed) =
  let n = List.length p.ops in
  let holders nm = List.length (List.filter (List.mem nm) p.ops) in
  cover "one operand" (n = 1);
  cover "five operands" (n = 5);
  cover "a name summed in one operand"
    (List.exists (fun nm -> holders nm = 1) p.sums);
  let times l nm = List.length (List.filter (( = ) nm) l) in
  let repeated =
    List.concat_map (fun l -> List.filter (fun nm -> times l nm > 1) l) p.ops
  in
  cover "a name an operand repeats" (repeated <> []);
  cover "a name an operand repeats three times"
    (List.exists (fun l -> List.exists (fun nm -> times l nm > 2) l) p.ops);
  cover "a repeated name of extent 0 or 1"
    (List.exists (fun nm -> List.assoc nm p.extents <= 1) repeated);
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
      (* Rounding is monotone: the result lies between the roundings of the
         bound's ends. *)
      let b = g *. abs in
      let lo = D.of_float dt (exact -. b) and hi = D.of_float dt (exact +. b) in
      at_most (float Float.min_float) ~than:hi y.(o);
      at_least (float Float.min_float) ~than:lo y.(o))
    (sum_over_tuples p (List.map read xs))

(* The diagonal of one operand that repeats a name of extent [e]. *)
let diagonal_of e =
  {
    text = "f0 f0 -> f0";
    shapes = [ [| e; e |] ];
    extents = [ ("f0", e) ];
    ops = [ [ "f0"; "f0" ] ];
    out = [ "f0" ];
    sums = [];
  }

(* Of x1 [i; j], x2 [j; k] and x3 [k; l], (x1 x2) x3 does i·j·k + i·k·l
   multiply-adds and x1 (x2 x3) j·k·l + i·j·l; (x1 x3) x2 does 2 i·j·k·l, never
   fewer than (x1 x2) x3, which wins a tie with it. Each tree's reference is its
   two contractions, the first left in float32 as einsum leaves it. *)
let law_tree ((i, j, k, l), seed) =
  let st = Random.State.make [| seed |] in
  let x s =
    host D.Float32 s
      (Array.init (numel s) (fun _ -> Random.State.float st 2. -. 1.))
  in
  let x1 = x [| i; j |] and x2 = x [| j; k |] and x3 = x [| k; l |] in
  let contract p a b = Nx.contract Nx.float32 (Nx.Pattern.v p) a b in
  let left =
    read
      (contract "i k, k l -> i l | k" (contract "i j, j k -> i k | j" x1 x2) x3)
  in
  let right =
    read
      (contract "i j, j l -> i l | j" x1 (contract "j k, k l -> j l | k" x2 x3))
  in
  let wl = (i * j * k) + (i * k * l) and wr = (j * k * l) + (i * j * l) in
  cover "the left tree does less work" (wl < wr);
  cover "the right tree does less work" (wl > wr);
  cover "the trees do equal work" (wl = wr);
  cover "the two trees' results differ"
    (not (Testable.equal (array float_exact) left right));
  equal (array float_exact)
    (if wl <= wr then left else right)
    (read
       (Nx.einsum (Nx.Pattern.v "i j, j k, k l -> i l | j k") [ x1; x2; x3 ]))

(* Signed zeros, infinities and NaN in both parts of a complex64 [[m; n; n]]
   operand, moved by a transpose and a diagonal. *)
let law_copy (m, n) =
  let special = [| -0.; 0.; infinity; neg_infinity; nan; 1.5 |] in
  let pick k = special.(k mod Array.length special) in
  let c k = { Complex.re = pick k; im = pick (k + 1) } in
  let a = host D.Complex64 [| m; n; n |] (Array.init (m * n * n) c) in
  let bits y =
    Array.map
      (fun (z : Complex.t) ->
        (Int64.bits_of_float z.re, Int64.bits_of_float z.im))
      (read y)
  in
  let same msg want y =
    equal ~msg (array (pair int64 int64)) (bits want) (bits y)
  in
  let y = Nx.einsum (Nx.Pattern.v "b i j -> j i b") [ a ] in
  same "transpose" (Nx.transpose ~axes:[ 2; 1; 0 ] a) y;
  same "diagonal"
    (Nx.transpose (Nx.diagonal ~axis1:1 ~axis2:2 a))
    (Nx.einsum (Nx.Pattern.v "b i i -> i b") [ a ]);
  let buffer v = A.buffer (Option.get (Nx.Repr.array ~by:"test" v)) in
  let memory =
    Testable.make ~equal:( == ) ~pp:(fun ppf b ->
        Format.fprintf ppf "%d bytes at offset %d" (Rig.Buffer.length b)
          (Rig.Buffer.offset b))
  in
  satisfies ~claim:"memory apart from the operand's" memory
    (fun b -> not (Rig.Buffer.overlaps (buffer a) b))
    (buffer y)

(* The call [einsum p] of float32 operands of ones of [shapes]. *)
let einsumming ?sizes p shapes () =
  ignore (Nx.einsum ?sizes (Nx.Pattern.v p) (List.map ones shapes))

let chain = "i j, j k, k -> i | j k"
let mk = "i j, j k -> i k | j"

(* In nx.mli's order of einsum's raises. The messages are baselines: nx.mli
   states the exception and the function. *)
let einsum_refusals =
  [
    ( "more operands than einsum plans",
      einsumming
        (String.concat ", " (List.init 12 (fun _ -> "a")) ^ " -> a")
        (List.init 12 (fun _ -> [| 2 |])),
      __POS_OF__
        {|Nx.einsum: "a, a, a, a, a, a, a, a, a, a, a, a -> a" has 12 operands; einsum plans at most 11: contract parts in separate einsums|}
    );
    ( "fewer values than operands",
      einsumming chain [ [| 2; 3 |]; [| 3; 4 |] ],
      __POS_OF__
        {|Nx.einsum: "i j, j k, k -> i | j k" has 3 operands but the list holds 2|}
    );
    ( "more values than operands",
      einsumming mk [ [| 2; 3 |]; [| 3; 4 |]; [| 4 |] ],
      __POS_OF__
        {|Nx.einsum: "i j, j k -> i k | j" has 2 operands but the list holds 3|}
    );
    ( "one value",
      einsumming mk [ [| 2; 3 |] ],
      __POS_OF__
        {|Nx.einsum: "i j, j k -> i k | j" has 2 operands but the list holds 1|}
    );
    ( "no value",
      einsumming mk [],
      __POS_OF__
        {|Nx.einsum: "i j, j k -> i k | j" has 2 operands but the list holds 0|}
    );
    ( "boolean operands",
      (fun () ->
        let t = host D.Bool [| 2; 2 |] [| true; false; true; true |] in
        ignore (Nx.einsum (Nx.Pattern.v mk) [ t; t ])),
      __POS_OF__ {|Nx.einsum: the operands are booleans|} );
    ( "a rank the pattern does not give",
      einsumming chain [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 1 |] ],
      __POS_OF__
        {|Nx.einsum: "i j, j k, k -> i | j k" names 1 axis for x3; x3 is float32 [4; 1]|}
    );
    ( "an ellipsis over different numbers of axes",
      einsumming "... i, ... i, ... i -> ... | i"
        [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 2; 3 |] ],
      __POS_OF__
        {|Nx.einsum: "... i, ... i, ... i -> ... | i": ... is 1 axis in x1 and 2 in x3|}
    );
    ( "extents that differ within an operand",
      einsumming "i i -> i" [ [| 2; 3 |] ],
      __POS_OF__ {|Nx.einsum: "i i -> i": i is 2 in x1 and 3 in x1|} );
    ( "extents that differ between operands",
      einsumming chain [ [| 2; 3 |]; [| 3; 4 |]; [| 5 |] ],
      __POS_OF__
        {|Nx.einsum: "i j, j k, k -> i | j k": k is 4 in x2 and 5 in x3|} );
    ( "a group with two unknowns",
      einsumming "(h d) k, k -> h d | k" [ [| 12; 5 |]; [| 5 |] ],
      __POS_OF__
        {|Nx.einsum: "(h d) k, k -> h d | k": (h d) in x1 leaves h and d unknown; give one in ~sizes|}
    );
    ( "a unit axis of another extent",
      einsumming "i 1, i -> i" [ [| 2; 2 |]; [| 2 |] ],
      __POS_OF__
        {|Nx.einsum: "i 1, i -> i": axis 1 of x1 is 1 in the pattern and 2 in x1|}
    );
    ( "a size the pattern lacks",
      einsumming ~sizes:[ ("z", 2) ] mk [ [| 2; 3 |]; [| 3; 4 |] ],
      __POS_OF__
        {|Nx.einsum: "i j, j k -> i k | j": ~sizes names z, which it lacks|} );
  ]

let einsum =
  let x s = host D.Float32 s (operand (numel s)) in
  let random dt s =
    host dt s (Array.init (numel s) (fun _ -> Random.float 2. -. 1.))
  in
  group "einsum"
    [
      prop "of two operands is contract at their dtype" cases_of_sizes (fun c ->
          let a = host D.Float32 c.sa (operand (numel c.sa)) in
          let b = host D.Float32 c.sb (operand (numel c.sb)) in
          let p = Nx.Pattern.v c.pattern in
          equal (array float_exact)
            (read (Nx.contract Nx.float32 p a b))
            (read (Nx.einsum p [ a; b ])));
      prop "is within γ(N, 2u) Σ|x1|⋯|xn| of the sum over every index tuple"
        ~examples:
          [ (diagonal_of 0, F D.Float32, 0); (diagonal_of 1, F D.Float32, 0) ]
        (Gen.triple (Gen.with_pp pp_problem problems) float_dtypes Gen.int)
        law_einsum_within_bound;
      prop "takes the tree of fewest multiply-adds, the left one on a tie"
        ~examples:[ ((4, 4, 4, 4), 0) ]
        Gen.(
          let e = int_range 1 6 in
          pair (quad e e e e) int)
        law_tree;
      test "computes i j, j k, k -> i | j k as a (b v)" (fun () ->
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
               (Nx.cast Nx.bfloat16
                  (Nx.contract Nx.float32 (Nx.Pattern.v "i j, j -> i | j") a bv)))
            (read (Nx.einsum (Nx.Pattern.v chain) [ a; b; v ])));
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
      test "reads the diagonal of an operand that repeats a name" (fun () ->
          let z = x [| 2; 3; 3 |] in
          equal (array float_exact)
            (read (Nx.diagonal ~axis1:1 ~axis2:2 z))
            (read (Nx.einsum (Nx.Pattern.v "z a a -> z a") [ z ]));
          let t = x [| 3; 2; 3; 3 |] in
          equal (array float_exact)
            (read
               (Nx.diagonal ~axis1:1 ~axis2:2
                  (Nx.diagonal ~axis1:2 ~axis2:3
                     (Nx.transpose ~axes:[ 1; 0; 2; 3 ] t))))
            (read (Nx.einsum (Nx.Pattern.v "a b a a -> b a") [ t ])));
      test "sums a name one operand alone holds in it first" (fun () ->
          let a = x [| 2; 3 |] and v = x [| 4 |] in
          let s = Nx.sum ~axes:[ 1 ] a in
          equal (array float_exact)
            (read (Nx.mul (Nx.reshape [| 2; 1 |] s) (Nx.reshape [| 1; 4 |] v)))
            (read (Nx.einsum (Nx.Pattern.v "i j, k -> i k | j") [ a; v ]));
          let m = x [| 3; 3 |] in
          equal (array float_exact)
            (read (Nx.trace m))
            (read (Nx.einsum (Nx.Pattern.v "i i -> | i") [ m ])));
      prop
        "of one operand that sums nothing is a copy of its elements, moved bit \
         for bit"
        Gen.(pair (int_range 0 3) (int_range 0 3))
        law_copy;
      cases "raises Invalid_argument naming itself" ~id:refusal_id
        einsum_refusals (refuses "Nx.einsum");
      test "raises naming itself where every order holds a value past the rank"
        (fun () ->
          (* Four operands, each pair sharing ten summed names: an operand has
             30 axes, and the first contraction of any order keeps the 40 names
             its two operands share with the other two. *)
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
          let m =
            message (einsumming text (List.init 4 (fun _ -> Array.make 30 1)))
          in
          starts_with ~prefix:"Nx.einsum: " m;
          ends_with ~suffix:": every order holds a value of more than 32 axes" m);
    ]

(* matmul *)

(* Operand shapes for matmul: leading axes where each operand holds the result's
   extent, holds 1 or lacks the axis, 0 among the extents, and 1-d operands,
   which have no leading axis. *)
let matmul_shapes =
  let open Gen in
  let extent hi = frequency [ (1, constant 0); (16, int_range 1 hi) ] in
  (* Both operands hold the extent, or one holds an extent other than 1 and the
     other 1. *)
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
  (let+ axes = list ~size:held axis
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

let law_matmul (sa, sb) =
  let ra = Array.length sa and rb = Array.length sb in
  let lead s r = Array.to_list (Array.sub s 0 (max 0 (r - 2))) in
  let la = lead sa ra and lb = lead sb rb in
  let alone ours theirs =
    let d = List.length ours - List.length theirs in
    List.exists Fun.id
      (List.mapi
         (fun i e -> e <> 1 && (i < d || List.nth theirs (i - d) = 1))
         ours)
  in
  cover "a 1-d operand" (ra = 1 || rb = 1);
  cover "operands with leading axes of different counts"
    (List.length la <> List.length lb);
  cover "leading axes of a alone and of b alone" (alone la lb && alone lb la);
  cover "a zero extent" (Array.mem 0 sa || Array.mem 0 sb);
  cover "an empty sum" (sa.(ra - 1) = 0);
  (* Exact in float64: {!operand}'s products and their sums of 5 fit. *)
  let xa = operand (numel sa) and xb = operand (numel sb) in
  let y = Nx.matmul (host D.Float64 sa xa) (host D.Float64 sb xb) in
  let shape, want = matmul_meaning sa xa sb xb in
  equal (array int) shape (Nx.shape y);
  equal (array float_exact) want (read y)

(* The call [matmul] of float32 operands of ones of shapes [sa] and [sb]. *)
let matmulling sa sb () = ignore (Nx.matmul (ones sa) (ones sb))

(* In nx.mli's order of matmul's raises. The messages are baselines: nx.mli
   states the exception and the function. *)
let matmul_refusals =
  [
    ( "a 0-d operand",
      matmulling [||] [| 3 |],
      __POS_OF__ {|Nx.matmul: an operand is 0-d; float32 [] and float32 [3]|} );
    ( "inner extents that differ",
      matmulling [| 2; 3 |] [| 4; 2 |],
      __POS_OF__
        {|Nx.matmul: inner extents 3 and 4 differ; float32 [2; 3] and float32 [4; 2]|}
    );
    ( "leading axes that do not broadcast",
      matmulling [| 2; 2; 3 |] [| 3; 3; 4 |],
      __POS_OF__
        {|Nx.matmul: the leading axes do not broadcast; float32 [2; 2; 3] and float32 [3; 3; 4]|}
    );
    ( "a boolean dtype",
      (fun () ->
        let t = host D.Bool [| 2; 2 |] [| true; false; true; true |] in
        ignore (Nx.matmul t t)),
      __POS_OF__ {|Nx.matmul: the result's dtype is bool|} );
  ]

let matmul =
  group "matmul"
    [
      prop "is its meaning over the last two axes, leading axes broadcast"
        ~examples:
          [
            ([| 2; 3 |], [| 3; 4 |]);
            ([| 5; 1; 2; 3 |], [| 4; 3; 6 |]);
            ([| 0; 2; 3 |], [| 1; 3; 4 |]);
            ([| 1; 2; 3 |], [| 0; 3; 4 |]);
            ([| 0; 2; 3 |], [| 3; 4 |]);
            ([| 3 |], [| 2; 3; 4 |]);
            ([| 2; 2; 3 |], [| 3 |]);
            ([| 3 |], [| 3 |]);
            ([| 2; 0 |], [| 0; 3 |]);
          ]
        matmul_shapes law_matmul;
      test "accumulates as the batched contraction" (fun () ->
          let x s = host D.Float32 s (operand (numel s)) in
          let a = x [| 2; 3; 4 |] and b = x [| 4; 5 |] in
          equal (array float_exact)
            (read
               (Nx.contract Nx.float32
                  (Nx.Pattern.v "x i k, k j -> x i j | k")
                  a b))
            (read (Nx.matmul a b)));
      cases "raises Invalid_argument naming itself" ~id:refusal_id
        matmul_refusals (refuses "Nx.matmul");
    ]

(* dot *)

(* A result past a layout, as Values states it. The messages are baselines. *)
let dot_refusals =
  let wide s = Nx.broadcast_to s (ones [||]) in
  [
    ( "a result of more axes than a layout",
      (fun () ->
        ignore (Nx.dot (ones (Array.make 17 1)) (ones (Array.make 18 1)))),
      __POS_OF__ {|Nx.dot: a result of rank 33|} );
    ( "a result of more elements than a layout",
      (fun () ->
        ignore (Nx.dot (wide [| 1 lsl 27; 1 |]) (wide [| 1; 1 lsl 27 |]))),
      __POS_OF__
        {|Nx.dot: a contraction of shape [134217728; 134217728] has more than 9007199254740992 elements|}
    );
  ]

let dot =
  group "dot"
    [
      cases "raises Invalid_argument naming itself" ~id:refusal_id dot_refusals
        (refuses "Nx.dot");
    ]

(* Scales and segments

   [Nx.Prim.Contract] with block scales and segments: on the host, against the
   exact sums of its scaled terms within the bound; over devices, a cut
   placement against one device's bits. *)

module Scaled = struct
  module P = Nx.Prim
  module D = Nx_array.Dtype

  let pp_shape ppf s =
    Format.fprintf ppf "[%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int s)))

  let numel s = Array.fold_left ( * ) 1 s

  let read x =
    Nx_array.to_array
      (Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

  let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

  type dt = F32 | F16 | Bf16 | F4 | F64
  type scale = Codes of int array | Values of dt * float array

  (* An operand: its view's shape, stored with its last two axes swapped where
     [swap], its values in stored order, and its scale's view shape and values,
     stored alike. *)
  type side = {
    dt : dt;
    view : int array;
    swap : bool;
    xs : float array;
    scale : (int array * scale) option;
  }

  type segs = Rows of int64 array | Contracting of int64 array

  (* [a], [[batch; m; k]], by [b], [[batch; g; k; n]], the batch axis where
     [batch] gives its extent and [g] for [Rows] alone, from [init], summed in
     float64 or float32 and rounded into float16 where [half], [init] too. Ids
     have the shape [[batch; m]] for [Rows], [[batch; k]] for [Contracting]. *)
  type case = {
    batch : int option;
    m : int;
    n : int;
    k : int;
    g : int;
    a : side;
    b : side;
    f64 : bool;
    half : bool;
    init : float array option;
    segs : segs option;
  }

  let dt_name = function
    | F32 -> "f32"
    | F16 -> "f16"
    | Bf16 -> "bf16"
    | F4 -> "f4"
    | F64 -> "f64"

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
    let ids ppf os =
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map Int64.to_string os)))
    in
    Format.fprintf ppf "@[<v>%s%s, g %d@,a %a@,b %a@,init %s@,%s@]"
      (if c.f64 then "float64" else "float32")
      (if c.half then " into float16" else "")
      c.g pp_side c.a pp_side c.b
      (match c.init with
      | None -> "none"
      | Some xs -> Format.asprintf "%a" pp_floats xs)
      (match c.segs with
      | None -> "unsegmented"
      | Some (Rows os) -> Format.asprintf "rows %a" ids os
      | Some (Contracting os) -> Format.asprintf "contracting %a" ids os)

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

  (* Shape.contract's: [Contracting]'s groups, the batch, [a]'s free axis, then
     [b]'s, [Rows]' groups dropped. *)
  let result_shape c =
    let groups =
      match c.segs with Some (Contracting _) -> [ c.g ] | _ -> []
    in
    Array.of_list (groups @ lead c @ [ c.m; c.n ])

  (* The segmented axis's extent: [m] for [Rows], [k] for [Contracting]. *)
  let segmented c = match c.segs with Some (Rows _) -> c.m | _ -> c.k
  let ids_shape c = Array.of_list (lead c @ [ segmented c ])

  (* Values *)

  (* Where each operand lies: [put role x] places the host value [x], stored as
     made, for [role]. *)
  type role = A | B | A_scale | B_scale | Ids | Init

  type 'e put = {
    put : 'v 's. role -> ('v, 's, Nx.host) Nx.t -> ('v, 's, 'e) Nx.t;
  }

  let on_host = { put = (fun _ x -> x) }

  (* [roles] split by [q], the others by [rest]. *)
  let split_by roles q rest =
    let at role = if List.mem role roles then q else rest in
    { put = (fun role x -> Nx.place (at role) x) }

  (* An array of [dt] of the stored shape [shape] holding [xs], placed by [p]
     for [role], through its view. *)
  let made (type e) (p : e put) role dt shape swap xs : e P.any =
    let x = Nx.create Nx.float64 shape xs in
    let view x =
      let x = p.put role x in
      if swap then Nx.swapaxes (-1) (-2) x else x
    in
    match dt with
    | F32 -> Any (view (Nx.cast Nx.float32 x))
    | F16 -> Any (view (Nx.cast Nx.float16 x))
    | Bf16 -> Any (view (Nx.cast Nx.bfloat16 x))
    | F4 -> Any (view (Nx.cast Nx.float4_e2m1fn x))
    | F64 -> Any (view x)

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

  let op (type r o e) ?(sum = false) (p : e put) (acc : (float, r) Nx.dtype)
      (out : (float, o) Nx.dtype) c : (float, o, e) Nx.t P.t =
    let (P.Any a) = operand p A c.a in
    let (P.Any b) = operand p B c.b in
    let nb = List.length (lead c) in
    let batch : (int * int) iarray = if nb = 1 then [| (0, 0) |] else [||] in
    let bk = match c.segs with Some (Rows _) -> nb + 1 | _ -> nb in
    let contracting : (int * int) iarray = [| (nb + 1, bk) |] in
    let ids os = p.put Ids (Nx.create Nx.int64 (ids_shape c) os) in
    let segments : _ Nx_kernel.segments option =
      match c.segs with
      | None -> None
      | Some (Rows os) -> Some (Rows { ids = ids os; a_axis = nb; b_axis = nb })
      | Some (Contracting os) ->
          Some (Contracting { ids = ids os; groups = c.g; pair = 0 })
    in
    let init =
      Option.map
        (fun xs ->
          p.put Init (Nx.cast out (Nx.create Nx.float64 (result_shape c) xs)))
        c.init
    in
    Contract
      {
        batch;
        contracting;
        acc = D.Any acc;
        out;
        a;
        b;
        init;
        a_scale = scale p A_scale c.a;
        b_scale = scale p B_scale c.b;
        segments;
        sum;
      }

  (* The index [r] of the segmented axis is in group [ids.(r)] where that lies
     in [\[0, g)], as Nx_kernel.segments states. *)
  let group_of g (ids : int64 array) r =
    let e = ids.(r) in
    if Int64.compare e 0L >= 0 && Int64.compare e (Int64.of_int g) < 0 then
      Some (Int64.to_int e)
    else None

  let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

  (* Each element of [s]'s view as the accumulator, float64 where [f64], holds
     it, times its scale block's scale read into the accumulator, the product
     rounded once there, in C order of the view: what nx.cpu multiplies
     (nx_cpu.mli, Contractions). *)
  let scaled ~f64 role s =
    let (P.Any x) = operand on_host role s in
    let xs = Nx.to_array (Nx.cast Nx.float64 x) in
    let into_acc v = if f64 then v else f32 v in
    let sv = match s.scale with Some (sv, _) -> sv | None -> [||] in
    let ss =
      match scale on_host role s with
      | None -> [||]
      | Some (Exponent e) ->
          Array.map
            (fun c -> if c = 255 then Float.nan else Float.ldexp 1. (c - 127))
            (Nx.to_array e)
      | Some (Value v) ->
          Array.map into_acc (Nx.to_array (Nx.cast Nx.float64 v))
    in
    if s.scale = None then xs
    else
      let r = Array.length s.view in
      let idx = Array.make r 0 in
      Array.mapi
        (fun i x ->
          let rest = ref i and at = ref 0 in
          for a = r - 1 downto 0 do
            idx.(a) <- !rest mod s.view.(a);
            rest := !rest / s.view.(a)
          done;
          for a = 0 to r - 1 do
            at := (!at * sv.(a)) + (idx.(a) / (s.view.(a) / sv.(a)))
          done;
          into_acc (x *. ss.(!at)))
        xs

  let two_sum a b =
    let s = a +. b in
    let bb = s -. a in
    (s, a -. (s -. bb) +. (b -. bb))

  (* Each output of [c], in C order of [result_shape c], from [init]'s values as
     the result's dtype holds them: its exact sum [r] in double-double, as
     Nx_kernel.segments groups its products, the sum of its products'
     magnitudes, and their count [K]. *)
  let reference c init =
    let a = scaled ~f64:c.f64 A c.a and b = scaled ~f64:c.f64 B c.b in
    let nb = Option.value c.batch ~default:1 and w = segmented c in
    let gb = match c.segs with Some (Rows _) -> c.g | _ -> 1 in
    let groups = match c.segs with Some (Contracting _) -> c.g | _ -> 1 in
    let ids bi =
      match c.segs with
      | Some (Rows os | Contracting os) -> Array.sub os (bi * w) w
      | None -> [||]
    in
    let outs = groups * nb * c.m * c.n in
    let want = Array.make outs 0. and mag = Array.make outs 0. in
    let count = Array.make outs 0 in
    for o = 0 to outs - 1 do
      let j = o mod c.n and i = o / c.n mod c.m in
      let bi = o / (c.m * c.n) mod nb and e = o / (nb * c.m * c.n) in
      let t0 = match init with Some xs -> xs.(o) | None -> 0. in
      let hi = ref t0 and lo = ref 0. and m = ref 0. and terms = ref 0 in
      let term ~e t =
        let x = a.((((bi * c.m) + i) * c.k) + t)
        and y = b.((((((bi * gb) + e) * c.k) + t) * c.n) + j) in
        let p = x *. y in
        let s, err = two_sum !hi p in
        hi := s;
        lo := !lo +. err +. Float.fma x y (-.p);
        m := !m +. Float.abs p;
        incr terms
      in
      (match c.segs with
      | None ->
          for t = 0 to c.k - 1 do
            term ~e:0 t
          done
      | Some (Rows _) -> (
          match group_of c.g (ids bi) i with
          | Some e ->
              for t = 0 to c.k - 1 do
                term ~e t
              done
          | None -> ())
      | Some (Contracting _) ->
          for t = 0 to c.k - 1 do
            if group_of c.g (ids bi) t = Some e then term ~e:0 t
          done);
      want.(o) <- !hi +. !lo;
      mag.(o) <- !m;
      count.(o) <- !terms
    done;
    (want, mag, count)

  (* nx_kernel.mli's β of a library that adds in blocks (section Error
     bounds). *)
  let beta n =
    let cdiv a b = (a + b - 1) / b in
    cdiv (5 * Int.min n 256) 4 + cdiv n 128

  (* Each output is within nx.mli's bound of [contract] of the exact [r = init +
     Σ a·b] of its [K] products, [u|r| + (1 + u) γ(β(K)) Σ|a||b|], with one
     rounding more per term for each scaled operand, as nx.cpu states its
     scaling (nx_cpu.mli, Contractions), and one least subnormal per term and
     two more for what underflows, which the bound bars. The terms are the
     elements times their scales as the accumulator holds them, so the scaled
     operands' roundings hold for a library that scales products too. A NaN
     magnitude is a NaN term, which makes a NaN; an infinite one, or one past
     the accumulator's largest, is an overflow, which the bound bars. Rounded
     into float16, each sum takes one more rounding of float16's [2^-11]
     relative of [|r| + tol], which bounds the accumulator's sum, and half its
     least subnormal, [2^-25], an output near float16's largest, [65504],
     overflowing. *)
  let check (type r) (acc : (float, r) Nx.dtype) c =
    let to64 y = Nx.to_array (Nx.cast Nx.float64 y) in
    let run out =
      let init =
        Option.map
          (fun xs ->
            to64 (Nx.cast out (Nx.create Nx.float64 (result_shape c) xs)))
          c.init
      in
      let y = P.eval ~by:"t" (op on_host acc out c) in
      equal ~msg:"the result's shape" (array int) (result_shape c) (Nx.shape y);
      (read (Nx.cast Nx.float64 y), init)
    in
    let y, init = if c.half then run Nx.float16 else run acc in
    let want, mag, count = reference c init in
    let u, eta =
      if c.f64 then (epsilon_float /. 2., Float.of_string "0x1p-1074")
      else (Float.of_string "0x1p-24", Float.of_string "0x1p-149")
    in
    let scaled_operands =
      Bool.to_int (c.a.scale <> None) + Bool.to_int (c.b.scale <> None)
    in
    let gamma k =
      let n = Float.of_int (beta k + scaled_operands) in
      n *. u /. (1. -. (n *. u))
    in
    let u16 = Float.of_string "0x1p-11" and eta16 = Float.of_string "0x1p-25" in
    let largest =
      if c.f64 then Float.max_float else Float.of_string "0x1.fffffep127"
    in
    Array.iteri
      (fun i m ->
        let msg = Printf.sprintf "output %d of magnitude %g" i m in
        let k = count.(i) and r = Float.abs want.(i) in
        let from_init = if init = None then 0. else u *. r in
        let tol =
          from_init
          +. ((1. +. u) *. gamma k *. m)
          +. (Float.of_int (k + 2) *. eta)
        in
        let tol = if c.half then tol +. (u16 *. (r +. tol)) +. eta16 else tol in
        if Float.is_nan m then equal float_exact ~msg Float.nan y.(i)
        else if r +. m +. tol < largest && not (c.half && r +. tol >= 65504.)
        then equal (float tol) ~msg want.(i) y.(i))
      mag;
    y

  (* Generators *)

  let e2m1 =
    [ 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. ]
    @ [ -0.; -0.5; -1.; -1.5; -2.; -3.; -4.; -6. ]

  let finite = [ 0.; -0.; 0.5; -1.; 1.5; 3.; -3.; 1. ]

  let element dt =
    let open Gen in
    match dt with
    | F4 -> of_list e2m1
    | F32 | F16 | Bf16 | F64 ->
        frequency [ (12, of_list finite); (1, constant Float.nan) ]

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

  (* Scale values, 0.1 among them, which float16 and float32 round differently
     and float64 holds closer. *)
  let scale_value =
    let open Gen in
    frequency
      [
        (10, of_list [ 0.5; 2.; -1.; 0.25; 3.; 0.; 0.1 ]);
        (1, constant Float.nan);
      ]

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

  (* The block counts an axis of [e] elements takes: [e] itself, one block, or
     blocks of 2, 3 or 32; none for an empty axis. *)
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
        let* dt = of_list [ F32; F16; F64 ] in
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

  (* An id of [g] groups: in [\[-1, g\]] mostly, the one past the groups and the
     one below them included, else far out of range. *)
  let id g =
    let open Gen in
    frequency
      [
        (12, map Int64.of_int (int_range (-1) g));
        (1, of_list [ Int64.min_int; Int64.max_int; -2L; Int64.of_int (g + 1) ]);
      ]

  (* The ids of each batch index over an axis of [len]: unsorted mostly, sorted,
     or all of one group. *)
  let ids ~batch ~g len =
    let open Gen in
    let row =
      let* mode =
        frequency
          [ (4, constant `Any); (1, constant `Sorted); (1, constant `One) ]
      in
      match mode with
      | `One ->
          let+ e = int_range 0 (max 0 (g - 1)) in
          List.init len (fun _ -> Int64.of_int e)
      | `Sorted | `Any ->
          let+ xs = array ~size:(constant len) (id g) in
          if mode = `Sorted then Array.sort Int64.compare xs;
          Array.to_list xs
    in
    let+ rows = list ~size:(constant (max 1 batch)) row in
    let rows = if batch = 0 then [] else rows in
    Array.of_list (List.concat rows)

  (* [c] with a NaN in [a] at index 0 of the contracted axis, which
     [Contracting]'s groups leave out: group 0 from 1, the others empty. *)
  let planted c =
    match c.segs with
    | Some (Contracting os) when c.k > 1 && c.g > 0 ->
        let os = Array.mapi (fun i _ -> if i mod c.k = 0 then -1L else 0L) os in
        (* [k]'s index is the last of the view, before the last stored. *)
        let inner = if c.a.swap then c.m else 1 in
        let xs =
          Array.mapi
            (fun i x -> if i / inner mod c.k = 0 then Float.nan else x)
            c.a.xs
        in
        { c with a = { c.a with xs }; segs = Some (Contracting os) }
    | _ -> c

  (* A case of one of [kinds], of [groups] groups. *)
  let drawn ~groups kinds =
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
    let* kind = of_list kinds in
    let* g = groups in
    let lead = Option.to_list batch in
    let groups = if kind = `Rows then [ g ] else [] in
    let av = Array.of_list (lead @ [ m; k ]) in
    let bv = Array.of_list (lead @ groups @ [ k; n ]) in
    let* adt = of_list [ F32; F16; Bf16 ] in
    let* a = side adt av in
    let* b =
      if mx then mxfp4 bv
      else
        let* bdt = of_list [ F32; F16; Bf16; F4 ] in
        side bdt bv
    in
    let* f64 = bool in
    let nb = Option.value ~default:1 batch in
    let* segs =
      match kind with
      | `None -> constant None
      | `Rows -> map (fun o -> Some (Rows o)) (ids ~batch:nb ~g m)
      | `Contracting -> map (fun o -> Some (Contracting o)) (ids ~batch:nb ~g k)
    in
    let* plant = frequency [ (1, constant false); (2, constant true) ] in
    let* half = bool in
    let c = { batch; m; n; k; g; a; b; f64; half; init = None; segs } in
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

  let case =
    Gen.with_pp pp_case
      (drawn ~groups:(Gen.int_range 0 3) [ `None; `Rows; `Contracting ])

  (* The element of [s] at the view's index [idx]. *)
  let value_at s idx =
    let idx = stored s idx and shape = stored s s.view in
    let flat = ref 0 in
    Array.iteri (fun i x -> flat := (!flat * shape.(i)) + x) idx;
    s.xs.(!flat)

  let ids_at c os b =
    let w = segmented c in
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
               let o = ids_at c os (Option.value ~default:0 bt) in
               List.exists (fun r -> group_of c.g o r = None) (range c.m))
             (batches c)
    | _ -> false

  let nan_at s bt idx = Float.is_nan (value_at s (with_batch bt idx))

  (* Whether some batch index's ids satisfy [p], given [g]. *)
  let some_ids c p =
    match c.segs with
    | Some (Rows os | Contracting os) ->
        List.exists
          (fun bt -> p (ids_at c os (Option.value ~default:0 bt)))
          (batches c)
    | None -> false

  let unsorted c =
    some_ids c (fun o ->
        List.exists
          (fun r -> Int64.compare o.(r) o.(r + 1) > 0)
          (range (max 0 (Array.length o - 1))))

  let out_of_range c =
    some_ids c (fun o ->
        List.exists (fun r -> group_of c.g o r = None) (range (Array.length o)))

  let extreme c =
    some_ids c (Array.exists (fun e -> e = Int64.min_int || e = Int64.max_int))

  let empty_group c =
    some_ids c (fun o ->
        List.exists
          (fun e ->
            not
              (List.exists
                 (fun r -> group_of c.g o r = Some e)
                 (range (Array.length o))))
          (range c.g))

  (* Whether a NaN of [a] or [b] lies at an index of the contracted axis in no
     group. *)
  let nan_alone c =
    match c.segs with
    | Some (Contracting os) ->
        List.exists
          (fun bt ->
            let o = ids_at c os (Option.value ~default:0 bt) in
            List.exists
              (fun r ->
                group_of c.g o r = None
                && (List.exists (fun i -> nan_at c.a bt [ i; r ]) (range c.m)
                   || List.exists (fun j -> nan_at c.b bt [ r; j ]) (range c.n)
                   ))
              (range c.k))
          (batches c)
    | _ -> false

  (* Whether [s] has a float64 scale whose value float32 does not hold. *)
  let wide_scale s =
    match s.scale with
    | Some (_, Values (F64, vs)) -> Array.exists (fun v -> f32 v <> v) vs
    | _ -> false

  let law c =
    let codes s = match s.scale with Some (_, Codes _) -> true | _ -> false in
    let values s =
      match s.scale with Some (_, Values _) -> true | _ -> false
    in
    let cut s =
      match s.scale with
      | Some (sv, _) -> Array.exists2 (fun e d -> e > 1 && e < d) sv s.view
      | None -> false
    in
    cover "a scale of codes" (codes c.a || codes c.b);
    cover "a scale of values" (values c.a || values c.b);
    cover "a scale cutting an axis into blocks" (cut c.a || cut c.b);
    cover "a float64 scale the float32 accumulator rounds"
      ((not c.f64) && (wide_scale c.a || wide_scale c.b));
    cover "a bfloat16 operand" (c.a.dt = Bf16 || c.b.dt = Bf16);
    cover "an MXFP4 operand" (c.b.dt = F4 && codes c.b && c.k >= 32);
    let rows = match c.segs with Some (Rows _) -> true | _ -> false in
    let scaled = c.a.scale <> None || c.b.scale <> None in
    cover "rows by groups" rows;
    cover "groups of the contracted axis"
      (match c.segs with Some (Contracting _) -> c.g > 0 | _ -> false);
    cover "scaled and segmented" (scaled && c.segs <> None);
    cover "unsorted ids" (unsorted c);
    cover "an id out of range" (out_of_range c);
    cover "an id at int64's extremes" (extreme c);
    cover "a group of no index" (empty_group c);
    cover "a row in no group" (row_alone c);
    cover "a result rounded into float16" c.half;
    cover "an init cast into the accumulator" (c.half && c.init <> None);
    cover "a row in no group rounded into float16" (c.half && row_alone c);
    let y = if c.f64 then check Nx.float64 c else check Nx.float32 c in
    cover "a NaN output" (Array.exists Float.is_nan y);
    cover "a NaN in no group kept out"
      (nan_alone c && Array.exists Float.is_finite y)

  (* Placements *)

  module S2 = (val Nx.devices (List.init 2 Nx_support.memory))
  module S3 = (val Nx.devices (List.init 3 Nx_support.memory))
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
        {
          dt = F4;
          view = bv;
          swap = true;
          xs = some (numel bv);
          scale = b_scale;
        };
      f64 = false;
      half = false;
      init = None;
      segs;
    }

  (* [c] by [p] and on the host, in [acc] into [out], asked for its sum where
     [sum]: the result's placement, after its values, a sum's closed as they are
     read, equal one device's bit for bit. *)
  let same_bits (type r o e) ?sum (p : e put) (acc : (float, r) Nx.dtype)
      (out : (float, o) Nx.dtype) c =
    let y = P.eval ~by:"t" (op ?sum p acc out c) in
    let one = P.eval ~by:"t" (op on_host acc out c) in
    equal (array float_exact) (read one) (read y);
    Nx.placement y

  (* [c] in its accumulator into its result's dtype, against one device's bits:
     [placed] holds where [c]'s sums are exact in every order, or where its
     terms are each on one device. *)
  let placed ?sum p c =
    match (c.f64, c.half) with
    | false, false -> same_bits ?sum p Nx.float32 Nx.float32 c
    | false, true -> same_bits ?sum p Nx.float32 Nx.float16 c
    | true, false -> same_bits ?sum p Nx.float64 Nx.float64 c
    | true, true -> same_bits ?sum p Nx.float64 Nx.float16 c

  (* Rows by groups, row 1 and 3 in no group, alone and rounded into float16
     from [init]; a NaN outside every group of the contracted axis; a float64
     scale per row of a, 0.1 among its values. *)
  let examples =
    let rows = Rows [| 2L; Int64.min_int; 0L; Int64.max_int |] in
    let groups =
      Contracting (Array.init 32 (fun i -> if i < 4 then 0L else -1L))
    in
    let wide =
      let c = mx ~m:2 ~k:32 ~n:2 ~per:1 () in
      let scale = Some ([| 2; 1 |], Values (F64, [| 0.1; 3. |])) in
      { c with a = { c.a with scale } }
    in
    [
      mx ~segs:rows ~g:3 ~m:4 ~k:32 ~n:2 ~per:1 ();
      {
        (mx ~segs:rows ~g:3 ~m:4 ~k:32 ~n:2 ~per:1 ()) with
        half = true;
        init = Some (some 8);
      };
      planted
        {
          (mx ~segs:groups ~g:2 ~m:2 ~k:32 ~n:2 ~per:2 ()) with
          init = Some (some 8);
        };
      wide;
    ]

  let within_bound =
    prop ~examples
      "is within its bound of the exact sum of its elements times their \
       scales, each group's alone"
      case law

  (* The sums of [mx] and its rows' cases below are exact in every order: every
     term is a multiple of 2^-4 and their magnitudes sum below 2^20, float32's
     exact range at that step. *)

  (* Rows by three groups from an init of 2^25, row 4 in none. *)
  let from_init =
    let segs = Rows [| 2L; 0L; 1L; 2L; -1L; 2L; 1L; 0L |] in
    let c = mx ~segs ~g:3 ~m:8 ~k:32 ~n:8 ~per:2 () in
    let xs = Array.init (8 * 32) (fun i -> 0.1 +. (Float.of_int i *. 1e-3)) in
    { c with a = { c.a with xs }; init = Some (Array.make 64 0x1p25) }

  let placements =
    let free_cuts =
      [
        ("b's columns", [ B; B_scale ], 1, mx ~m:3 ~k:64 ~n:8 ~per:2 ());
        ("a's rows", [ A; A_scale ], 0, mx ~rows:true ~m:4 ~k:32 ~n:2 ~per:1 ());
        (* A row's ids are cut with it: each device computes its rows. *)
        ( "rows by groups, the ids cut with a's rows",
          [ A; Ids ],
          0,
          let segs = Rows [| 2L; 0L; -1L; 2L; 1L; 7L |] in
          mx ~segs ~g:3 ~m:6 ~k:32 ~n:4 ~per:2 () );
      ]
    in
    let b_groups = split_by [ B; B_scale ] (S2.split ~axis:0) S2.on in
    group "placements"
      [
        cases
          "a cut of a free axis at its scale's blocks cuts the result alike, \
           with one device's values"
          ~id:(fun (name, _, _, _) -> name)
          free_cuts
          (fun (_, roles, axis, c) ->
            let y = placed (split_by roles (S2.split ~axis:0) S2.on) c in
            equal (placement ()) (S2.split ~axis) y);
        (* Two columns per device cut b's scale, one value per four columns,
           inside its blocks: b's columns are read whole. *)
        test "a cut of b's columns inside its scale's blocks raises" (fun () ->
            let c = mx ~m:3 ~k:64 ~n:8 ~per:4 () in
            let p = split_by [ B; B_scale ] (S4.split ~axis:0) S4.on in
            raises
              (Invalid_argument
                 "t: operand 1 lies at split ~axis:1 [m0; m1; m2; m3] and is \
                  read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) y")
              (fun () -> placed p c));
        test "a product cut along no summed axis is the same asked for its sum"
          (fun () ->
            let c = mx ~m:3 ~k:64 ~n:8 ~per:2 () in
            let p = split_by [ B; B_scale ] (S2.split ~axis:0) S2.on in
            equal (placement ()) (S2.split ~axis:1) (placed ~sum:true p c));
        (* Rows of group 2 have their products on the second device: from an
           init of 2^25, whose spacing is 4, a chain of products near 0.1 rounds
           each away, which init plus their sum would not. *)
        test
          "rows by groups from init, b cut along its groups, give one device's \
           bits" (fun () -> ignore (placed b_groups from_init));
        (* Asked for its sum, a device's term starts from init where it holds
           the row's group, the first device's also for a row in no group, and
           from the identity elsewhere. *)
        test
          "rows by groups from init, b cut along its groups, asked for their \
           sum, close to one device's bits" (fun () ->
            let q = placed ~sum:true b_groups from_init in
            not_equal ~msg:"the result is a sum" (placement ())
              (Nx.Placement.closed q) q);
      ]

  (* An output of no products is [init] as the accumulator holds it, rounded
     into the result, bit for bit: at [k = 0], in a row of no group of [Rows],
     and in [Contracting]'s slice of a group its ids leave empty, by batch
     index, unsorted or out of range. Each output's init differs from its
     neighbours': zeros of both signs, NaNs, infinities and numbers. Outputs
     with products are not compared. *)
  let no_products =
    let inits =
      [|
        -0.;
        Float.nan;
        Float.infinity;
        0.;
        Float.neg_infinity;
        1.5;
        Float.neg Float.nan;
        -2.25;
      |]
    in
    let plain view =
      { dt = F32; view; swap = false; xs = some (numel view); scale = None }
    in
    let c ?batch ?segs ?(g = 0) ?(half = false) ~m ~k () =
      let n = 3 and bs = Option.to_list batch in
      let gs = match segs with Some (Rows _) -> [ g ] | _ -> [] in
      let a = plain (Array.of_list (bs @ [ m; k ])) in
      let b = plain (Array.of_list (bs @ gs @ [ k; n ])) in
      let c =
        { batch; m; n; k; g; a; b; f64 = false; half; init = None; segs }
      in
      let all = numel (result_shape c) in
      let init = Array.init all (fun i -> inits.(i mod Array.length inits)) in
      { c with init = Some init }
    in
    (* Whether output [i], in C order of the result, has no products. *)
    let alone c i =
      let mn = c.m * c.n in
      let batches = Option.value c.batch ~default:1 in
      c.k = 0
      ||
      match c.segs with
      | None -> false
      | Some (Rows os) ->
          group_of c.g (ids_at c os (i / mn)) (i / c.n mod c.m) = None
      | Some (Contracting os) ->
          let e = i / (batches * mn) and bi = i / mn mod batches in
          let o = ids_at c os bi in
          let mine r = group_of c.g o r = Some e in
          not (List.exists mine (List.init c.k Fun.id))
    in
    (* [init] and the result at the outputs of no products, widened to float64,
       which keeps every float32 and float16 apart, NaNs' signs and payloads
       too. *)
    let outputs (type o e) (p : e put) (out : (float, o) Nx.dtype) c =
      let xs = Option.get c.init in
      let init = Nx.cast out (Nx.create Nx.float64 (result_shape c) xs) in
      let y = P.eval ~by:"t" (op p Nx.float32 out c) in
      let pick y =
        let y = Nx.to_array (Nx.cast Nx.float64 y) in
        Array.of_list
          (List.filter_map
             (fun i -> if alone c i then Some y.(i) else None)
             (List.init (Array.length y) Fun.id))
      in
      (pick init, pick y)
    in
    let in_its_dtype p c =
      if c.half then outputs p Nx.float16 c else outputs p Nx.float32 c
    in
    let bits = Array.map Int64.bits_of_float in
    let rows =
      [
        ("k = 0", c ~m:2 ~k:0 ());
        ("k = 0, into float16", c ~half:true ~m:2 ~k:0 ());
        ("k = 0, batched", c ~batch:2 ~m:2 ~k:0 ());
        ("k = 0, rows by groups", c ~segs:(Rows [| 0L; 1L |]) ~g:2 ~m:2 ~k:0 ());
        ("k = 0, groups of k", c ~segs:(Contracting [||]) ~g:2 ~m:2 ~k:0 ());
        ( "k = 0, groups of k into float16",
          c ~half:true ~segs:(Contracting [||]) ~g:2 ~m:2 ~k:0 () );
        ("rows in no group", c ~segs:(Rows [| -1L; 0L; 5L |]) ~g:2 ~m:3 ~k:4 ());
        ( "rows in no group by batch index",
          c ~batch:2 ~segs:(Rows [| 0L; -1L; 7L; 1L |]) ~g:2 ~m:2 ~k:4 () );
        ( "rows in no group into float16",
          c ~half:true ~segs:(Rows [| -1L; 0L; 2L |]) ~g:2 ~m:3 ~k:4 () );
        ( "an empty group of k",
          c ~segs:(Contracting [| 1L; 1L; 1L; 1L |]) ~g:2 ~m:2 ~k:4 () );
        ( "an empty group of k by batch index",
          c ~batch:2
            ~segs:(Contracting [| 1L; 1L; 1L; 1L; 0L; 0L; 0L; 0L |])
            ~g:2 ~m:2 ~k:4 () );
        ( "an empty group among unsorted ids",
          c ~segs:(Contracting [| 2L; 0L; 2L; 0L |]) ~g:3 ~m:2 ~k:4 () );
        ( "an empty group among ids out of range",
          c
            ~segs:(Contracting [| -1L; 5L; Int64.max_int; 0L |])
            ~g:2 ~m:2 ~k:4 () );
        ( "an empty group of k into float16",
          c ~half:true
            ~segs:(Contracting [| 2L; 0L; 2L; 0L |])
            ~g:3 ~m:2 ~k:4 () );
        ( "rows in no group among unsorted ids",
          c ~segs:(Rows [| 1L; Int64.min_int; 0L |]) ~g:2 ~m:3 ~k:4 () );
      ]
    in
    let by_rows (_, c) =
      match c.segs with Some (Rows _) -> true | _ -> false
    in
    group "no products"
      [
        cases "an output of no products is its init's bits"
          ~id:(fun (n, _) -> n)
          rows
          (fun (_, c) ->
            let want, got = in_its_dtype on_host c in
            equal (array int64) (bits want) (bits got));
        (* [b] split along its groups over two devices, the rest whole on each:
           the closing adds [-0] to an init, which keeps a number's bits and a
           NaN's NaN, whose sign and payload the NaN rule leaves free. *)
        cases
          "an output of no products keeps its init with b split along its \
           groups"
          ~id:(fun (n, _) -> n)
          (List.filter by_rows rows)
          (fun (_, c) ->
            let nb = List.length (lead c) in
            let p = split_by [ B ] (S2.split ~axis:nb) S2.on in
            let want, got = in_its_dtype p c in
            equal (array float_exact) want got);
      ]

  (* Contractions cut along [k]: [a] [[m; k]] and [b], stored [[…; n; k]], along
     their last stored axis, the ids and [init] on every device, and the scales
     too but where a row cuts [b]'s with [b]. *)
  let partials =
    let k_cut (module S : Nx.Devices) roles c =
      let b_k = match c.segs with Some (Rows _) -> 2 | _ -> 1 in
      let at role =
        if not (List.mem role roles) then S.on
        else
          match role with
          | B | B_scale -> S.split ~axis:b_k
          | _ -> S.split ~axis:1
      in
      { put = (fun role x -> Nx.place (at role) x) }
    in
    (* [a] scaled in two blocks of [k / 2] per row. *)
    let halves c =
      let scale = Some ([| c.m; 2 |], Values (F32, some (2 * c.m))) in
      { c with a = { c.a with scale } }
    in
    (* [a] scaled by a code per 32 along [k], from [init]. *)
    let coded c =
      let sv = [| c.m; c.k / 32 |] in
      let scale = Some (sv, Codes (codes (numel sv))) in
      { c with a = { c.a with scale }; init = Some (some (c.m * c.n)) }
    in
    let rows = Rows [| 0L; 1L; 1L; 1L |] in
    let two = (module S2 : Nx.Devices) and four = (module S4 : Nx.Devices) in
    let cuts =
      [
        ( "b MXFP4 cut at its scale's blocks",
          two,
          [ A; B ],
          mx ~m:2 ~k:128 ~n:4 ~per:2 () );
        ( "over four devices, inside the scale's blocks",
          four,
          [ A; B ],
          mx ~m:2 ~k:96 ~n:2 ~per:1 () );
        ( "a's scale whole beside a's cut",
          two,
          [ A; B ],
          halves (mx ~m:2 ~k:128 ~n:2 ~per:1 ()) );
        ( "both operands scaled, from init",
          two,
          [ A; B ],
          coded (mx ~m:2 ~k:128 ~n:2 ~per:1 ()) );
        ( "b's scale cut with b",
          two,
          [ A; B; B_scale ],
          mx ~m:2 ~k:128 ~n:2 ~per:1 () );
        ( "rows by groups",
          two,
          [ A; B ],
          mx ~segs:rows ~g:2 ~m:4 ~k:64 ~n:2 ~per:2 () );
        ( "rows by groups over four devices, inside the scale's blocks",
          four,
          [ A; B ],
          mx ~segs:rows ~g:2 ~m:4 ~k:96 ~n:2 ~per:2 () );
      ]
    in
    (* Products near 0.1, whose sums round. *)
    let inexact c =
      let xs =
        Array.init (c.m * c.k) (fun i -> 0.1 +. (Float.of_int i *. 1e-3))
      in
      { c with a = { c.a with xs } }
    in
    group "partials"
      [
        cases
          "a contraction cut along its contracted axis has one device's bits"
          ~id:(fun (name, _, _, _) -> name)
          cuts
          (fun (_, devices, roles, c) ->
            let (module S : Nx.Devices) = devices in
            ignore (placed (k_cut (module S) roles c) c));
        test
          "asked for its sum without init, a contraction cut along its \
           contracted axis closes to the bits of the closed one" (fun () ->
            let c = inexact (mx ~m:2 ~k:128 ~n:2 ~per:1 ()) in
            let p = k_cut (module S2) [ A; B ] c in
            let closed = P.eval ~by:"t" (op p Nx.float32 Nx.float32 c) in
            let q = P.eval ~by:"t" (op ~sum:true p Nx.float32 Nx.float32 c) in
            not_equal ~msg:"the result is a sum" (placement ())
              (Nx.Placement.closed (Nx.placement q))
              (Nx.placement q);
            equal (array float_exact) (read closed) (read q));
        (* [Contracting] reads its contracting pairs whole. *)
        test "groups of a cut contracted axis raise" (fun () ->
            let segs =
              Contracting (Array.init 96 (fun i -> if i < 40 then 0L else 1L))
            in
            let c = mx ~segs ~g:2 ~m:2 ~k:96 ~n:2 ~per:1 () in
            raises
              (Invalid_argument
                 "t: operand 0 lies at split ~axis:1 [m0; m1] and is read at \
                  on [m0; m1]; Nx.place (Placement.on s) x") (fun () ->
                placed (k_cut (module S2) [ A; B ] c) c));
      ]

  (* Rows by groups cut along b's groups *)

  (* Rows by up to seven groups. *)
  let grouped =
    Gen.with_pp pp_case (drawn ~groups:(Gen.int_range 0 7) [ `Rows ])

  (* Whether [b]'s scale, beside [b] split along its groups over [t] devices, is
     cut alike: one scale per group, or blocks of groups in tiles that divide
     both extents. *)
  let follows c t =
    match c.b.scale with
    | None -> false
    | Some (sv, _) ->
        let e = sv.(List.length (lead c)) in
        e = c.g || (e > 1 && c.g mod t = 0 && e mod t = 0)

  (* [b] split along its groups over two devices and over three, its scale too
     where it follows the cut and whole on each device otherwise, [a], the ids
     and [init] whole on each, the results whole on each device. A row's
     products lie on the device holding its group and every other device's
     partial is [-0], so the closing adds exact zeros and [init] then joins once
     (nx.mli, Contraction); nx.cpu's order does not depend on the count of
     groups (nx_cpu.mli, Order): one device's bits. *)
  let grouped_law c =
    let nb = List.length (lead c) in
    let negative_zero x = Int64.bits_of_float x = Int64.min_int in
    cover "a row in no group" (row_alone c);
    cover "unsorted ids" (unsorted c);
    cover "fewer groups than devices" (c.g > 0 && c.g < 3);
    cover "more groups than devices" (c.g > 3);
    cover "a NaN in a" (Array.exists Float.is_nan c.a.xs);
    cover "a -0 in a" (Array.exists negative_zero c.a.xs);
    cover "an MXFP4 b" (c.b.dt = F4 && c.k >= 32);
    cover "from init" (c.init <> None);
    cover "a row in no group, from init" (c.init <> None && row_alone c);
    cover "rounded into float16 from init" (c.half && c.init <> None);
    List.iter
      (fun ((module S : Nx.Devices), t) ->
        let alike = follows c t in
        cover "b's scale cut alike" alike;
        cover "b's scale whole beside b's cut" (c.b.scale <> None && not alike);
        cover "b's scale in blocks of groups, whole beside b's cut"
          (match c.b.scale with
          | Some (sv, _) -> (not alike) && sv.(nb) > 1 && sv.(nb) < c.g
          | None -> false);
        let roles = if alike then [ B; B_scale ] else [ B ] in
        let y = placed (split_by roles (S.split ~axis:nb) S.on) c in
        equal ~msg:"the results whole on each device" (placement ()) S.on y)
      [ ((module S2 : Nx.Devices), 2); ((module S3 : Nx.Devices), 3) ]

  (* Four groups, b scaled per two of them: alike over two devices, whole beside
     the cut over three. *)
  let blocks_of_groups =
    let segs = Rows [| 0L; 3L; 1L; 2L |] in
    let c = mx ~segs ~g:4 ~m:4 ~k:32 ~n:2 ~per:1 () in
    let b =
      {
        dt = F32;
        view = [| 4; 32; 2 |];
        swap = false;
        xs = some (4 * 32 * 2);
        scale = Some ([| 2; 1; 1 |], Values (F32, [| 0.5; 3. |]));
      }
    in
    { c with b }

  let by_groups =
    prop ~count:300 ~examples:[ blocks_of_groups ]
      "rows by groups, b cut along its groups over two or three devices, have \
       one device's bits"
      grouped grouped_law

  let group =
    group "scales and segments"
      [ within_bound; no_products; placements; partials; by_groups ]
end

(* Convolution: Nx.conv against its sum over the taps computed in OCaml, on
   small values whose sums each dtype's carrier holds exactly, within its bound
   on floats; conv_transpose as conv's adjoint and as conv over its input
   spread; ceil_padding by its formula and conv's windows. *)
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
    let ints a =
      String.concat "; " (Array.to_list (Array.map string_of_int a))
    in
    let floats a =
      String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%.17g") a))
    in
    Format.fprintf ppf
      "n %d spatial [%s] groups %d cg %d og %d taps [%s] stride [%s] dilation \
       [%s] padding [%s], x [%s], w [%s]"
      c.n (ints c.spatial) c.groups c.cg c.og (ints c.taps) (ints c.stride)
      (ints c.dilation)
      (String.concat "; "
         (Array.to_list
            (Array.map (fun (a, b) -> Printf.sprintf "(%d, %d)" a b) c.padding)))
      (floats c.xs) (floats c.ws)

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

  (* The rows of conv's contraction, an output position each. *)
  let rows c = c.n * numel (Array.mapi (fun i _ -> windows c i) c.spatial)

  (* A contraction of more than four rows and columns, at least 64 outputs and a
     depth below 128 times them takes chain order (nx_cpu.mli, "Order"). conv's
     rows are its output positions, its columns its output channels a group and
     its depth its taps times its input channels a group; conv_transpose's rows
     are its result's positions, [n] times those of [d], its columns its
     result's channels a group and its depth its taps times [x]'s channels a
     group. The covers that read it steer the cases to both orders. *)
  let chain_order ~rows ~columns ~depth =
    rows > 4 && columns > 4
    && rows * columns >= 64
    && depth < 128 * rows * columns

  let chained c =
    c.cg > 1
    && chain_order ~rows:(rows c) ~columns:c.og ~depth:(numel c.taps * c.cg)

  let chained_transpose c d =
    c.og > 1
    && chain_order
         ~rows:(c.n * numel d)
         ~columns:c.cg
         ~depth:(numel c.taps * c.og)

  (* Cases whose [m] values [values] draws. *)
  let case_of values =
    Gen.with_pp pp_case
      (let open Gen in
       let* k = int_range 1 2 in
       let ints n lo hi = array ~size:(constant n) (int_range lo hi) in
       let* n = int_range 0 2 in
       let* spatial = ints k 0 6 in
       let* groups = int_range 1 2 in
       let* cg = of_list [ 0; 1; 2; 9 ] in
       (* Nine output channels a group take chain order over rows past four, one
          or two lane order, none sum nothing; nine input channels do so in
          conv_transpose, whose outputs they are. *)
       let* og = of_list [ 0; 1; 2; 9 ] in
       let* taps = ints k 0 3 in
       let* stride = ints k 1 3 in
       let* dilation = ints k 1 2 in
       let* los = ints k (-1) 2 in
       let* his = ints k (-1) 2 in
       (* One case in ten makes axis 0 an input below the formula's extent: one
          window of one tap, stride 2, over an input of 1 padded by 1 on each
          side, which an input of 0 has too. Drawn freely it is about one case
          in a hundred, which some seeds never reach. *)
       let* below = int_range 0 9 in
       let at0 v a =
         if below = 0 then Array.mapi (fun i e -> if i = 0 then v else e) a
         else a
       in
       let spatial = at0 1 spatial and taps = at0 1 taps in
       let stride = at0 2 stride and dilation = at0 1 dilation in
       let los = at0 1 los and his = at0 1 his in
       (* One case in ten makes axis 0 a padding that cuts past the axis: an
          input of 1 cropped by 1 on each side. A case of no padding or shaped
          below undoes it, so it is 1/10 x 7/8 x 4/10 = 0.035 of the cases, and
          drawn freely about 0.025 more. Such a case has no output, which most
          of the laws' other cases need, so the law that covers it draws 400
          cases instead of taking a larger share. *)
       let* past = int_range 0 9 in
       let cut v a =
         if past = 0 then Array.mapi (fun i e -> if i = 0 then v else e) a
         else a
       in
       let spatial = cut 1 spatial in
       let los = cut (-1) los and his = cut (-1) his in
       (* One case in eight has no padding, about one in thirty drawn freely. *)
       let* unpadded = int_range 0 7 in
       let none a = if unpadded = 0 then Array.map (fun _ -> 0) a else a in
       let los = none los and his = none his in
       (* One case in ten each of one input and one output channel a group,
          depthwise, and of two input channels a group, over two groups, and of
          a 1x1 kernel of two channels a group, each with an output: drawn
          freely each is about one case in fifteen or fewer, which some seeds
          never reach. *)
       let* shaped = int_range 0 9 in
       let groups, cg =
         match shaped with 0 -> (2, 1) | 1 | 2 -> (2, 2) | _ -> (groups, cg)
       in
       let taps = if shaped = 2 then Array.map (fun _ -> 1) taps else taps in
       (* One case in ten each of nine input and output channels a group, of one
          input and nine output channels, and of one input and two output
          channels, a tap or more along each axis and eight windows or more
          along axis 0, under an output, which conv and conv_transpose take in
          chain order, or lane order for two: drawn freely each is rare enough
          that some seeds never reach it. *)
       let cg, og =
         match shaped with
         | 0 -> (1, 1)
         | 3 -> (9, 9)
         | 4 -> (1, 9)
         | 5 -> (1, 2)
         | _ -> (cg, og)
       in
       let chain = shaped >= 3 && shaped <= 5 in
       let taps = if chain then Array.map (Int.max 1) taps else taps in
       let n = if shaped <= 5 then Int.max 1 n else n in
       let spatial =
         if shaped > 5 then spatial
         else
           Array.mapi
             (fun i d ->
               let least = (dilation.(i) * (taps.(i) - 1)) + 3 in
               let windows = if chain && i = 0 then 8 else 1 in
               Int.max d (least + ((windows - 1) * stride.(i))))
             spatial
       in
       let c0 =
         {
           n;
           spatial;
           groups;
           cg;
           og;
           taps;
           stride;
           dilation;
           padding = Array.map2 (fun a b -> (a, b)) los his;
           xs = [||];
           ws = [||];
         }
       in
       let+ xs = values (numel (x_shape c0))
       and+ ws = values (numel (w_shape c0)) in
       { c0 with xs; ws })

  let ints lo hi m =
    Gen.(array ~size:(constant m) (map Float.of_int (int_range lo hi)))

  (* Small integers, whose sums every dtype holds exactly. *)
  let case = case_of (ints (-3) 3)

  (* Values of 53 significant bits in (-2, 2), each rounded to a case's dtype
     when it runs. *)
  let float_case =
    case_of (fun m -> Gen.(array ~size:(constant m) (float_range (-2.) 2.)))

  (* A case of one batch element and one group along one axis, [x]'s element [i]
     [x i] and [w]'s [w i]. *)
  let one_axis ?(og = 2) ?(stride = 1) ?(dilation = 1) ?(padding = (0, 0)) ~cg
      ~spatial ~taps ~x ~w () =
    let c =
      {
        n = 1;
        spatial = [| spatial |];
        groups = 1;
        cg;
        og;
        taps = [| taps |];
        stride = [| stride |];
        dilation = [| dilation |];
        padding = [| padding |];
        xs = [||];
        ws = [||];
      }
    in
    {
      c with
      xs = Array.init (numel (x_shape c)) x;
      ws = Array.init (numel (w_shape c)) w;
    }

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

  (* The sum over the taps and the group's channels, at each output index, of
     [plus] of each product's operands from [zero], [x]'s values [xs] and [w]'s
     [ws]. *)
  let fold_terms ~zero ~plus xs ws c =
    let k = Array.length c.spatial in
    let t = Array.init k (windows c) in
    let o = c.groups * c.og in
    let out = Array.concat [ [| c.n |]; t; [| o |] ] in
    let xs_shape = x_shape c and ws_shape = w_shape c in
    Array.init (numel out) (fun flat ->
        let idx = unravel out flat in
        let b = idx.(0) and j = idx.(k + 1) in
        let g = j / c.og in
        let sum = ref zero in
        for tap = 0 to numel c.taps - 1 do
          let q = unravel c.taps tap in
          let pos =
            Array.init k (fun i ->
                (idx.(1 + i) * c.stride.(i))
                + (q.(i) * c.dilation.(i))
                - fst c.padding.(i))
          in
          if Array.for_all2 (fun p d -> p >= 0 && p < d) pos c.spatial then
            for i = 0 to c.cg - 1 do
              let xv =
                xs.(ravel xs_shape
                      (Array.concat [ [| b |]; pos; [| (g * c.cg) + i |] ]))
              in
              let wv =
                ws.(ravel ws_shape (Array.concat [ [| j |]; q; [| i |] ]))
              in
              sum := plus !sum xv wv
            done
        done;
        !sum)

  let reference c =
    fold_terms ~zero:0. ~plus:(fun s x w -> s +. (x *. w)) c.xs c.ws c

  (* [xs] as complex numbers, each value with the next as its imaginary part. *)
  let complex xs =
    let m = Array.length xs in
    Array.mapi (fun i re -> { Complex.re; im = xs.((i + 1) mod m) }) xs

  let complex_reference c =
    fold_terms ~zero:Complex.zero
      ~plus:(fun s x w -> Complex.add s (Complex.mul x w))
      (complex c.xs) (complex c.ws) c

  let parts zs =
    Array.concat
      (List.map (fun (z : Complex.t) -> [| z.re; z.im |]) (Array.to_list zs))

  (* [s + x w] in two floats, each sum and product split exactly (Ogita, Rump
     and Oishi, "Accurate Sum and Dot Product", 2005): within a few units of
     2^-106 of the exact value for the sums here, with the sum of the terms'
     magnitudes. *)
  let exact_terms c =
    let two_sum a b =
      let s = a +. b in
      let bb = s -. a in
      (s, a -. (s -. bb) +. (b -. bb))
    in
    fold_terms ~zero:(0., 0., 0.)
      ~plus:(fun (hi, lo, mag) x w ->
        let p = x *. w in
        let e = Float.fma x w (-.p) in
        let s, t = two_sum hi p in
        (s, lo +. t +. e, mag +. Float.abs p))
      c.xs c.ws c

  (* [γ(n, u)] and [β(n)] as nx_kernel.mli's "Bounds" defines them. *)
  let gamma n u = Float.of_int n *. u /. (1. -. (Float.of_int n *. u))
  let beta n = (((5 * Int.min n 256) + 3) / 4) + ((n + 127) / 128)

  (* The first eight outputs of [got] farther from their exact sums [terms] than
     [bound want mag] allows, [want] the sum and [mag] its terms' magnitudes. *)
  let past bound got terms =
    List.filteri
      (fun i _ -> i < 8)
      (List.concat
         (List.mapi
            (fun i (hi, lo, mag) ->
              let want = hi +. lo in
              let b = bound want mag and y = got.(i) in
              if Float.abs (y -. want) <= b then []
              else [ Printf.sprintf "%d: %h, %h past %h" i y want b ])
            (Array.to_list terms)))

  let conv_of ?init c x w =
    Nx.conv ?init ~stride:c.stride ~padding:c.padding ~dilation:c.dilation
      ~groups:c.groups x w

  (* The shape of [c]'s result. *)
  let out_shape c =
    let t = Array.init (Array.length c.spatial) (windows c) in
    Array.concat [ [| c.n |]; t; [| c.groups * c.og |] ]

  (* [vs] rounded once from float64 to [dt], read back. *)
  let rounded_to (type v s) (dt : (v, s) D.t) vs =
    let n = Array.length vs in
    Nx.to_array
      (Nx.cast Nx.float64 (Nx.cast dt (Nx.create Nx.float64 [| n |] vs)))

  (* [c]'s values as [dt] holds them. *)
  let rounded (type v s) (dt : (v, s) D.t) c =
    { c with xs = rounded_to dt c.xs; ws = rounded_to dt c.ws }

  (* [s], an integer, wrapped into the integer dtype [dt]. The sums here stay
     far inside 32 bits. *)
  let wrap (type v s) (dt : (v, s) D.t) s =
    let bits = D.bits dt in
    if bits >= 32 then s
    else
      let m = 1 lsl bits in
      let r = ((Float.to_int s mod m) + m) mod m in
      Float.of_int (if D.is D.Signed dt && r >= m / 2 then r - m else r)

  let is_integer (type v s) (dt : (v, s) D.t) =
    D.is D.Signed dt || D.is D.Unsigned dt

  (* Exact sums [sums] as [dt] holds conv's result: wrapped for an integer, else
     rounded once. *)
  let into (type v s) (dt : (v, s) D.t) sums =
    if is_integer dt then Array.map (wrap dt) sums else rounded_to dt sums

  let run (type v s) (dt : (v, s) D.t) c =
    let x = Nx.cast dt (Nx.create Nx.float64 (x_shape c) c.xs) in
    let w = Nx.cast dt (Nx.create Nx.float64 (w_shape c) c.ws) in
    Nx.to_array (Nx.cast Nx.float64 (conv_of c x w))

  (* [c]'s convolution in the complex [dt], its values as [complex] makes
     them. *)
  let run_complex (type s) (dt : (Complex.t, s) D.t) c =
    let x = Nx.create dt (x_shape c) (complex c.xs) in
    let w = Nx.create dt (w_shape c) (complex c.ws) in
    parts (Nx.to_array (conv_of c x w))

  (* conv: every dtype *)

  let checked =
    [ D.Any D.Float64; D.Any D.Float32; D.Any D.Float16; D.Any D.Int32 ]

  (* The dtypes but booleans that each case does not check: one of them is drawn
     per case. *)
  let others =
    List.filter
      (fun (D.Any dt) ->
        (not (D.is D.Boolean dt))
        && (not (D.equal dt D.Complex64))
        && not (List.exists (fun (D.Any c) -> D.equal c dt) checked))
      D.all

  (* Values of [dt] whose sums, of at most 81 products, its carrier holds
     exactly, and most of which [dt] holds: halves of [-0.5, 0.5] for
     float4_e2m1fn, whose greatest value is 6; integers of [-2, 2] for
     float8_e4m3fn, whose greatest is 448 and which no sum of them passes; [0,
     3] for the unsigned integers; [-3, 3] otherwise. *)
  let values_for (type v s) (dt : (v, s) D.t) m =
    match dt with
    | D.Float4_e2m1fn ->
        Gen.(
          array ~size:(constant m)
            (map (fun i -> Float.of_int i /. 2.) (int_range (-1) 1)))
    | D.Float8_e4m3fn -> ints (-2) 2 m
    | _ when D.is D.Unsigned dt -> ints 0 3 m
    | _ -> ints (-3) 3 m

  let drawn_case =
    Gen.(
      with_pp
        (fun ppf (D.Any dt, c) ->
          Format.fprintf ppf "%s, %a" (D.name dt) pp_case c)
        (bind (of_list others) (fun (D.Any dt) ->
             map (fun c -> (D.Any dt, c)) (case_of (values_for dt)))))

  (* One input and no output channel, a cover drawn about one case in forty; a
     kernel of no tap over an input the padding cuts past, whose windows are
     counted before the padding is clamped; one product of -0, which a sum from
     +0 makes +0; and sums of 54, which int4 wraps, where a drawn integer wraps
     about one case in twenty-five. *)
  let conv_examples =
    let v i = Float.of_int ((i mod 7) - 3) in
    [
      (D.Any D.Int8, one_axis ~og:0 ~cg:1 ~spatial:3 ~taps:2 ~x:v ~w:v ());
      ( D.Any D.Bfloat16,
        one_axis ~cg:4 ~spatial:1 ~taps:0 ~dilation:3 ~padding:(-2, 0)
          ~x:(fun _ -> 1.)
          ~w:(fun _ -> 1.)
          () );
      ( D.Any D.Int16,
        one_axis ~og:1 ~cg:1 ~spatial:1 ~taps:1
          ~x:(fun _ -> 0.)
          ~w:(fun _ -> -1.)
          () );
      ( D.Any D.Int4,
        one_axis ~og:1 ~cg:2 ~spatial:3 ~taps:3
          ~x:(fun _ -> 3.)
          ~w:(fun _ -> 3.)
          () );
    ]

  (* Each dtype's carrier holds these sums exactly, so each output is the exact
     sum rounded once into the dtype, or wrapped. *)
  let law_conv =
    prop ~count:400 ~examples:conv_examples
      "is the sum over its taps of x times w, rounded once into the dtype, \
       integers wrapped"
      drawn_case (fun (D.Any drawn, c) ->
        let want = reference c in
        let axes f =
          Array.exists Fun.id (Array.init (Array.length c.spatial) f)
        in
        cover "an output" (Array.length want > 0);
        cover "output channels past 4, over rows past 4" (chained c);
        cover "two groups" (c.groups = 2);
        cover "no input channel" (c.cg = 0);
        cover "one input channel and no output channel" (c.cg = 1 && c.og = 0);
        cover "a negative padding"
          (axes (fun i -> fst c.padding.(i) < 0 || snd c.padding.(i) < 0));
        cover "a padding that cuts past the axis"
          (axes (fun i ->
               c.spatial.(i) + fst c.padding.(i) + snd c.padding.(i) < 0));
        cover "a stride and a dilation above 1"
          (axes (fun i -> c.stride.(i) > 1 && c.dilation.(i) > 1));
        cover "a kernel axis of no element, under an output"
          (Array.mem 0 c.taps && Array.length want > 0);
        List.iter
          (fun (D.Any dt) -> cover ("drawn " ^ D.name dt) (D.equal dt drawn))
          others;
        cover "an integer output wrapped"
          (is_integer drawn
          && Array.exists
               (fun s -> wrap drawn s <> s)
               (reference (rounded drawn c)));
        let check (type v s) (dt : (v, s) D.t) =
          let r = rounded dt c in
          equal ~msg:(D.name dt) (array float_exact)
            (into dt (reference r))
            (run dt r)
        in
        List.iter (fun (D.Any dt) -> check dt) checked;
        equal ~msg:"complex64" (array float_exact)
          (parts (complex_reference c))
          (run_complex D.Complex64 c);
        match drawn with
        | D.Complex128 ->
            equal ~msg:"complex128" (array float_exact)
              (parts (complex_reference c))
              (run_complex D.Complex128 c)
        | _ -> check drawn)

  (* conv: sums of no term *)

  (* An output that sums no term, along a kernel extent of 0 or in a group of no
     input channel, is [init]'s element: no addition touches it, so a [-0] and a
     NaN keep their bits. *)
  let test_no_terms () =
    let f32 s v = Nx.create Nx.float32 s (Array.make (numel s) v) in
    let bits y =
      Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 y))
    in
    let kept msg init y =
      let s = Nx.shape y in
      equal ~msg (array int64)
        (bits (Nx.copy (Nx.broadcast_to s init)))
        (bits y)
    in
    let init = Nx.create Nx.float32 [| 2 |] [| -0.; Float.nan |] in
    kept "no tap" init
      (Nx.conv ~init (f32 [| 1; 5; 4 |] 1.) (f32 [| 2; 0; 4 |] 1.));
    kept "no channel" init
      (Nx.conv ~init (f32 [| 1; 5; 0 |] 1.) (f32 [| 2; 3; 0 |] 1.))

  (* conv: order *)

  (* Unless a group has one input and one output channel, conv is the
     contraction of its windows, stored, with [w]: [[n; t; g; q · cg + i]]
     against [[g; o; q · cg + i]], bit for bit. With one of each it is the sum
     over the taps of its windows times [w], stored: [[n; t; g; q]] times [[g;
     q]], summed along [q]. Complex64, where each order's roundings show. *)
  let law_order =
    prop ~count:300
      "sums as one contract of its windows, or as sum over its taps with one \
       channel each way a group"
      float_case (fun c ->
        let k = Array.length c.spatial and cg = c.cg in
        cover "output channels past 4, over rows past 4" (chained c);
        cover "lane order over rows past 4"
          (cg > 1 && (not (chained c)) && rows c > 4);
        cover "a dilation above 1"
          (cg > 1 && Array.exists (fun d -> d > 1) c.dilation);
        cover "one input channel, output channels past 4, over rows past 4"
          (cg = 1
          && chain_order ~rows:(rows c) ~columns:c.og ~depth:(numel c.taps));
        cover "one input and two output channels, lane order over rows past 4"
          (cg = 1 && c.og = 2 && rows c > 4);
        cover "one input and one output channel, under an output"
          (cg = 1 && c.og = 1 && rows c > 0);
        let xs = complex c.xs and ws = complex c.ws in
        let t = Array.init k (windows c) and q = numel c.taps in
        let shape = [| c.n; numel t; c.groups; q * cg |] in
        let window flat =
          let idx = unravel shape flat in
          let tw = unravel t idx.(1) and tap = unravel c.taps (idx.(3) / cg) in
          let pos =
            Array.init k (fun a ->
                (tw.(a) * c.stride.(a))
                + (tap.(a) * c.dilation.(a))
                - fst c.padding.(a))
          in
          if not (Array.for_all2 (fun p d -> p >= 0 && p < d) pos c.spatial)
          then Complex.zero
          else
            let ch = (idx.(2) * cg) + (idx.(3) mod cg) in
            xs.(ravel (x_shape c)
                  (Array.concat [ [| idx.(0) |]; pos; [| ch |] ]))
        in
        let bits y = Array.map Int64.bits_of_float (parts (Nx.to_array y)) in
        let a =
          Nx.create Nx.complex64 shape (Array.init (numel shape) window)
        in
        let x = Nx.create Nx.complex64 (x_shape c) xs in
        let w = Nx.create Nx.complex64 (w_shape c) ws in
        let want =
          if cg <> 1 || c.og <> 1 then
            let b = Nx.create Nx.complex64 [| c.groups; c.og; q * cg |] ws in
            Nx.contract Nx.complex64
              (Nx.Pattern.v "n t g k, g o k -> n t g o | k")
              a b
          else
            let b = Nx.create Nx.complex64 [| 1; 1; c.groups; q |] ws in
            Nx.sum ~axes:[ 3 ] (Nx.mul a b)
        in
        equal (array int64) (bits want) (bits (conv_of c x w)))

  (* Depthwise cases: [groups] channels last, one input and one output channel
     each, [x] laid out as [view] says and [w] as [layout]: [w]'s channels lie
     [numel taps] elements apart, none, or [m] times as far, a slice of an array
     of [m] channels a group. *)
  type view = Contiguous | Transposed | Broadcast
  type layout = Packed | Shared | Strided of int

  let view_name = function
    | Contiguous -> "contiguous"
    | Transposed -> "transposed from planes"
    | Broadcast -> "broadcast across channels"

  let layout_name = function
    | Packed -> "packed"
    | Shared -> "broadcast across channels"
    | Strided m -> Printf.sprintf "a slice of %d channels a group" m

  (* [w]'s step from one channel to the next. *)
  let channel_step c = function
    | Packed -> numel c.taps
    | Shared -> 0
    | Strided m -> m * numel c.taps

  let pp_depthwise ppf (c, v, l) =
    Format.fprintf ppf "%a, x %s, w %s" pp_case c (view_name v) (layout_name l)

  let depthwise_case =
    let open Gen in
    (* Floats of (-2, 2) and the values that break a sum: signed zeros,
       infinities and NaN. *)
    let value =
      frequency
        [
          (8, float_range (-2.) 2.);
          (1, of_list [ -0.; 0.; Float.infinity; Float.neg_infinity; Float.nan ]);
        ]
    in
    with_pp pp_depthwise
      ((* Three cases in ten of 64, 65, 128 or 129 channels, about a float64 row
          and a float32 row of 512 bytes, over a short image of one batch
          element, at stride and dilation one with outputs along each axis. *)
       let* many = int_range 0 9 in
       let many = many < 3 in
       (* One case in eight of the others of more taps than a lane holds, 4x5
          over two axes long enough for outputs: drawn freely, about one in a
          hundred, which some seeds never reach. *)
       let* wide = int_range 0 7 in
       let wide = wide = 0 && not many in
       let* k = if wide then constant 2 else int_range 1 2 in
       let ints n lo hi = array ~size:(constant n) (int_range lo hi) in
       let* groups =
         if many then of_list [ 64; 65; 128; 129 ]
         else of_list [ 1; 2; 2; 3; 3; 5; 8; 16 ]
       in
       let* n =
         if many then constant 1
         else frequency [ (1, constant 0); (6, int_range 1 2) ]
       in
       (* The last axis long enough, one case in three, for more than 128
          outputs of its positions and channels. *)
       let* long = int_range 0 2 in
       let* last = if long = 0 then int_range 44 90 else int_range 0 12 in
       let* first = ints (k - 1) 0 8 in
       let spatial = Array.append first [| last |] in
       let* taps = ints k 1 5 in
       let taps = if wide then [| 4; 5 |] else taps in
       let spatial = if wide then Array.map (Int.max 6) spatial else spatial in
       let* untapped = int_range 0 19 in
       let taps =
         if untapped = 0 && not many then Array.map (fun _ -> 0) taps else taps
       in
       (* Many channels: three to five positions along the last axis, one or two
          along the first. *)
       let* more = ints k 0 2 in
       let spatial =
         if not many then spatial
         else
           Array.mapi
             (fun i q -> q + if i = k - 1 then 2 + more.(i) else more.(i) / 2)
             taps
       in
       let* stride = ints k 1 2 in
       let* dilation = ints k 1 2 in
       let dilation =
         if wide || many then Array.map (fun _ -> 1) dilation else dilation
       in
       (* A negative padding one side in eight: more leave most cases with no
          output. *)
       let pads =
         array ~size:(constant k)
           (if many then constant 0
            else frequency [ (1, constant (-1)); (7, int_range 0 2) ])
       in
       let* los = pads in
       let* his = pads in
       (* Stride one along the last axis, three cases in four, where an output's
          neighbours along it and across channels lie one after the other in the
          result and in [x]. *)
       let* stepped = int_range 0 3 in
       let stride =
         if stepped = 0 && not many then stride
         else Array.mapi (fun i s -> if i = k - 1 then 1 else s) stride
       in
       let* v =
         if many then
           frequency [ (3, constant Contiguous); (1, constant Transposed) ]
         else
           frequency
             [
               (4, constant Contiguous);
               (1, constant Transposed);
               (1, constant Broadcast);
             ]
       in
       (* A slice wide enough that [w]'s channels lie further apart than
          [groups] elements: four cases in five of many channels, one in five of
          the others. *)
       let strided = Strided ((groups / Int.max 1 (numel taps)) + 2) in
       let* l =
         if many then
           frequency
             [
               (8, constant strided); (1, constant Packed); (1, constant Shared);
             ]
         else
           frequency
             [
               (3, constant Packed); (1, constant Shared); (1, constant strided);
             ]
       in
       let c0 =
         {
           n;
           spatial;
           groups;
           cg = 1;
           og = 1;
           taps;
           stride;
           dilation;
           padding = Array.map2 (fun a b -> (a, b)) los his;
           xs = [||];
           ws = [||];
         }
       in
       let xn =
         if v = Broadcast then numel (x_shape c0) / groups
         else numel (x_shape c0)
       in
       let wn =
         if l = Shared then numel (w_shape c0) / groups else numel (w_shape c0)
       in
       let+ xs = array ~size:(constant xn) value
       and+ ws = array ~size:(constant wn) value in
       ({ c0 with xs; ws }, v, l))

  (* A depthwise case of [groups] channels over one batch element, [x]
     contiguous. *)
  let planes_example ?(stride = [| 1 |]) ~groups ~spatial ~taps layout =
    let c =
      {
        n = 1;
        spatial;
        groups;
        cg = 1;
        og = 1;
        taps;
        stride;
        dilation = Array.map (fun _ -> 1) spatial;
        padding = Array.map (fun _ -> (0, 0)) spatial;
        xs = [||];
        ws = [||];
      }
    in
    let v i = Float.of_int ((i * 7 mod 13) - 6) /. 4. in
    ( {
        c with
        xs = Array.init (numel (x_shape c)) v;
        ws = Array.init (numel (w_shape c)) v;
      },
      Contiguous,
      layout )

  (* A depthwise convolution over channels last has, at every float, the bits of
     each channel's plane convolved alone: an output sums its taps' products in
     their order, whatever the layout. NaNs compare as a class. Cases of many
     channels check float32 and float64 alone. The covers of rows steer the
     cases to nx.cpu's sums of rows of positions and channels, up to 64 outputs
     a row at float64 and 128 at float32. Drawn, a kernel axis of no tap, a
     stride above one along the last axis, more channels than a float32 row and
     rows over [w] packed are each about one case in twenty: each has an
     example. *)
  let law_planes =
    prop ~count:300
      ~examples:
        [
          planes_example ~groups:2 ~spatial:[| 5 |] ~taps:[| 0 |] Packed;
          planes_example ~stride:[| 2 |] ~groups:3 ~spatial:[| 7 |]
            ~taps:[| 2 |] Packed;
          planes_example ~groups:129 ~spatial:[| 5 |] ~taps:[| 3 |] (Strided 45);
          planes_example ~groups:3 ~spatial:[| 8 |] ~taps:[| 4 |] Packed;
        ]
      "with one channel each way a group has each plane's bits, whatever the \
       layout, at every float"
      depthwise_case
      (fun (c, v, l) ->
        let k = Array.length c.spatial and g = c.groups in
        let t = Array.init k (windows c) in
        let outputs = c.n * numel t > 0 in
        let products = numel c.taps <= 16 in
        (* Where nx.cpu sums rows of positions and channels together, up to
           [row] outputs a row: sums of 16 products or fewer, stride one along
           the last axis over [x] laid out channels last, and [x]'s step from
           one position to the next, [groups], below its step from one channel
           to the next plus [w]'s. *)
        let rows row =
          g > 1 && products && outputs && v <> Broadcast
          && c.stride.(k - 1) = 1
          && t.(k - 1) > 1
          && g < 1 + channel_step c l
          && g <= row
        in
        cover "rows of positions and channels at every float, past 128 outputs"
          (rows 64 && g * t.(k - 1) > 128);
        cover "rows of positions and channels at float32, not float64"
          (rows 128 && g > 64);
        cover "positions and channels past a float32 row"
          (rows max_int && g > 128);
        cover "rows of positions and channels, w packed" (rows 64 && l = Packed);
        cover "more taps than a lane holds, under outputs"
          ((not products) && outputs);
        cover "a stride above one along the last axis, under outputs"
          (g > 1 && c.stride.(k - 1) > 1 && outputs);
        cover "x transposed from planes, under outputs"
          (v = Transposed && outputs);
        cover "x broadcast across channels, under outputs"
          (v = Broadcast && outputs);
        cover "w broadcast across channels, under outputs"
          (l = Shared && outputs);
        cover "a kernel axis of no tap" (Array.mem 0 c.taps);
        cover "no output" (not outputs);
        let bits y =
          Array.map
            (fun f ->
              if Float.is_nan f then 0x7FF8000000000000L
              else Int64.bits_of_float f)
            (Nx.to_array (Nx.cast Nx.float64 y))
        in
        let check (type v s) (dt : (v, s) D.t) =
          let at s xs = Nx.cast dt (Nx.create Nx.float64 s xs) in
          let x =
            match v with
            | Contiguous -> at (x_shape c) c.xs
            | Broadcast ->
                Nx.broadcast_to (x_shape c)
                  (at
                     (Array.append (Array.sub (x_shape c) 0 (k + 1)) [| 1 |])
                     c.xs)
            | Transposed ->
                let planes =
                  Array.append [| g |] (Array.sub (x_shape c) 0 (k + 1))
                in
                Nx.transpose
                  ~axes:(List.init (k + 1) (fun i -> i + 1) @ [ 0 ])
                  (at planes c.xs)
          in
          let w =
            match l with
            | Packed -> at (w_shape c) c.ws
            | Shared ->
                Nx.broadcast_to (w_shape c)
                  (at
                     (Array.append [| 1 |] (Array.sub (w_shape c) 1 (k + 1)))
                     c.ws)
            | Strided m ->
                (* [c.ws] at channel 0 of [m], zeros elsewhere. *)
                let s =
                  Array.append (Array.sub (w_shape c) 0 (k + 1)) [| m |]
                in
                let wide =
                  Array.init (numel s) (fun i ->
                      if i mod m = 0 then c.ws.(i / m) else 0.)
                in
                Nx.slice
                  (List.init (k + 2) (fun i ->
                       if i = k + 1 then Nx.R (0, 1) else Nx.A))
                  (at s wide)
          in
          let plane j =
            let along a =
              List.init (k + 2) (fun i ->
                  if i = a then Nx.R (j, j + 1) else Nx.A)
            in
            Nx.conv ~stride:c.stride ~padding:c.padding ~dilation:c.dilation
              (Nx.slice (along (k + 1)) x)
              (Nx.slice (along 0) w)
          in
          let want = Nx.concatenate ~axis:(k + 1) (List.init g plane) in
          equal ~msg:(D.name dt) (array int64) (bits want)
            (bits (conv_of c x w))
        in
        check D.Float32;
        check D.Float64;
        if g < 64 then begin
          check D.Bfloat16;
          check D.Float16;
          check D.Float8_e4m3fn;
          check D.Float8_e5m2;
          check D.Float4_e2m1fn
        end)

  (* conv: bounds *)

  (* Before rounding, each output is within [(1 + u) γ(β(K), u) Σ|x||w|] of the
     exact sum of its [K] products, [u] the carrier's unit roundoff (nx.mli,
     conv). The carrier of float64 and float32 is the dtype itself; a narrower
     float then rounds once, to within [u_out] of the sum's magnitude, or [eta],
     half its least subnormal's step, below its normals. The values stay below 2
     and the sums below 324, so float8_e4m3fn's 448 bounds no sum. *)
  let law_bound =
    let dtypes =
      [
        (D.Any D.Float64, 0x1p-53, 0., 0.);
        (D.Any D.Float32, 0x1p-24, 0., 0.);
        (D.Any D.Bfloat16, 0x1p-24, 0x1p-8, 0x1p-134);
        (D.Any D.Float16, 0x1p-24, 0x1p-11, 0x1p-25);
        (D.Any D.Float8_e4m3fn, 0x1p-24, 0x1p-4, 0x1p-10);
        (D.Any D.Float8_e5m2, 0x1p-24, 0x1p-3, 0x1p-17);
      ]
    in
    (* No input channel, and a kernel of three taps over an input of two padded
       by one each side, each under an output: drawn, each is a few cases in a
       hundred, which some seeds never reach. *)
    let v i = Float.of_int ((i mod 7) - 3) /. 3. in
    let examples =
      [
        one_axis ~cg:0 ~spatial:3 ~taps:2 ~x:v ~w:v ();
        one_axis ~cg:2 ~spatial:2 ~taps:3 ~padding:(1, 1) ~x:v ~w:v ();
      ]
    in
    prop ~count:200 ~examples
      "is within its bound of the exact sum, at every float" float_case
      (fun c ->
        let rank = Array.length c.spatial in
        let axes f = Array.exists Fun.id (Array.init rank f) in
        let outputs = Array.length (reference c) > 0 in
        cover "no batch element" (c.n = 0);
        cover "output channels past 4, over rows past 4" (chained c);
        cover "no input channel, under an output" (c.cg = 0 && outputs);
        cover "one input channel a group, two groups, under an output"
          (c.cg = 1 && c.groups = 2 && outputs);
        cover "two input channels a group, two groups, under an output"
          (c.cg = 2 && c.groups = 2 && outputs);
        cover "no padding" (Array.for_all (( = ) (0, 0)) c.padding);
        cover "a padding asymmetric"
          (axes (fun i -> fst c.padding.(i) <> snd c.padding.(i)));
        cover "a negative padding"
          (axes (fun i -> fst c.padding.(i) < 0 || snd c.padding.(i) < 0));
        cover "a stride above 1" (axes (fun i -> c.stride.(i) > 1));
        cover "a dilation above 1" (axes (fun i -> c.dilation.(i) > 1));
        cover "a 1×1 kernel, two channels a group"
          (Array.for_all (( = ) 1) c.taps && c.cg = 2);
        cover "a kernel wider than its input, under an output"
          (axes (fun i -> dilated c i > c.spatial.(i)) && outputs);
        let k = c.cg * numel c.taps in
        List.iter
          (fun (D.Any dt, u, u_out, eta) ->
            let r = rounded dt c in
            let g = (1. +. u) *. gamma (beta k) u in
            let bound want mag =
              let within = g *. mag in
              within +. (u_out *. (Float.abs want +. within)) +. eta
            in
            equal ~msg:(D.name dt) (list string) []
              (past bound (run dt r) (exact_terms r)))
          dtypes)

  (* conv: init *)

  (* Integers up to 40, held by every float dtype: a sum of up to 81 of their
     products is exact in float32 and may lie between two float16 or bfloat16
     values. *)
  let wide_case = case_of (ints (-40) 40)

  (* One window of three taps, every product [39 · 41], over one input and one
     or two output channels a group, two input and two output channels, and a
     kernel axis of no tap: sums of 4797 and 9594, which neither float16 nor
     bfloat16 holds. Drawn, each is a few cases in a hundred, which some seeds
     never reach. *)
  let init_examples =
    let one ~cg ~og ~taps =
      one_axis ~og ~cg ~spatial:3 ~taps ~x:(fun _ -> 39.) ~w:(fun _ -> 41.) ()
    in
    [
      one ~cg:1 ~og:1 ~taps:3;
      one ~cg:1 ~og:2 ~taps:3;
      one ~cg:2 ~og:2 ~taps:3;
      one ~cg:2 ~og:2 ~taps:0;
    ]

  (* With an [init] each output is [round (init + Σ x w)], rounded once from the
     sum in float32, which holds these sums exactly. The [init] cancels each sum
     as the dtype holds it, off by a few units, so that a sum rounded before
     [init] joins would give the offset alone. *)
  let law_init =
    prop ~examples:init_examples
      "from an init rounds once, at float16, bfloat16 and float32" wide_case
      (fun c ->
        let sums = reference c in
        let outputs = Array.length sums > 0 in
        cover "one input and one output channel a group, under an output"
          (c.cg = 1 && c.og = 1 && outputs);
        cover "other than one of each a group, under an output"
          ((c.cg <> 1 || c.og <> 1) && outputs);
        cover "a kernel axis of no tap, under an output"
          (Array.mem 0 c.taps && outputs);
        List.iter
          (fun (D.Any dt) ->
            let round = rounded_to dt in
            let offsets =
              Array.mapi (fun i _ -> Float.of_int ((i mod 7) - 3)) sums
            in
            let init = round (Array.map2 ( -. ) offsets (round sums)) in
            let want = round (Array.map2 ( +. ) init sums) in
            let twice = round (Array.map2 ( +. ) init (round sums)) in
            if not (D.equal dt D.Float32) then
              cover
                (D.name dt ^ ": an output two roundings miss")
                (want <> twice);
            let at vs s = Nx.cast dt (Nx.create Nx.float64 s vs) in
            let init = at init (out_shape c) in
            let y =
              conv_of ~init c (at c.xs (x_shape c)) (at c.ws (w_shape c))
            in
            equal ~msg:(D.name dt) (array float_exact) want
              (Nx.to_array (Nx.cast Nx.float64 y)))
          [ D.Any D.Float16; D.Any D.Bfloat16; D.Any D.Float32 ])

  (* At float32 and float64 the sum's dtype is the accumulator's, so rounding it
     before [init] joins changes nothing: conv from an [init] has the bits of
     conv followed by {!Nx.add}, for an [init] of the result's shape and for one
     broadcast to it. An output without products is [init] itself, which no
     addition touches. *)
  let law_add =
    prop ~count:200 "from an init is conv then add, at float32 and float64"
      float_case (fun c ->
        let outputs = numel (out_shape c) > 0 in
        let products = c.cg > 0 && not (Array.mem 0 c.taps) in
        cover "one input and one output channel a group, under an output"
          (c.cg = 1 && c.og = 1 && outputs);
        cover "output channels past 4, over rows past 4" (chained c);
        cover "no product, under an output" ((not products) && outputs);
        let m = Array.length c.xs in
        let draw n =
          Array.init n (fun i -> if m = 0 then 0.5 else 0.75 *. c.xs.(i mod m))
        in
        let bits y =
          Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 y))
        in
        List.iter
          (fun (D.Any dt) ->
            let name = D.name dt in
            let at vs s = Nx.cast dt (Nx.create Nx.float64 s vs) in
            let x = at c.xs (x_shape c) and w = at c.ws (w_shape c) in
            let o = c.groups * c.og in
            let whole = at (draw (numel (out_shape c))) (out_shape c) in
            let bias = at (draw o) [| o |] in
            let y = conv_of c x w in
            let plus i =
              if products then Nx.add y i
              else Nx.copy (Nx.broadcast_to (out_shape c) i)
            in
            equal
              ~msg:(name ^ ", an init of the result's shape")
              (array int64)
              (bits (plus whole))
              (bits (conv_of ~init:whole c x w));
            equal
              ~msg:(name ^ ", an init broadcast")
              (array int64)
              (bits (plus bias))
              (bits (conv_of ~init:bias c x w)))
          [ D.Any D.Float32; D.Any D.Float64 ])

  (* Three products reach one output: 2048 then 1 and 1 sum to 2048 at float16,
     but to 2050 at float32, which float16 holds. *)
  let test_narrow () =
    let f16 s vs = Nx.create Nx.float16 s vs in
    equal (array float_exact) [| 2050. |]
      (Nx.to_array
         (Nx.conv
            (f16 [| 1; 3; 1 |] [| 2048.; 1.; 1. |])
            (f16 [| 1; 3; 1 |] [| 1.; 1.; 1. |])))

  (* conv: placement and memory *)

  (* A kernel of no tap reads every operand where it lies, as a kernel of taps
     does: the result takes [x]'s cut of the batch, and an operand alone on
     another device raises, [init] or not. *)
  let test_no_taps_routing () =
    let f32 s v = Nx.create Nx.float32 s (Array.make (numel s) v) in
    let placement = Nx_support.placement () in
    let x = f32 [| 2; 5; 4 |] 1. and w = f32 [| 2; 0; 4 |] 1. in
    let init = Nx.create Nx.float32 [| 2 |] [| -0.; 3. |] in
    let split = S2.split ~axis:0 in
    let y =
      Nx.conv ~init:(Nx.place S2.on init) (Nx.place split x) (Nx.place S2.on w)
    in
    equal ~msg:"x's cut of the batch" placement split (Nx.placement y);
    equal ~msg:"init's elements" (array int64)
      (Array.map Int64.bits_of_float
         (Array.concat (List.init 12 (fun _ -> [| -0.; 3. |]))))
      (Array.map Int64.bits_of_float (Nx.to_array (Nx.place Nx.Host.on y)));
    (* Channels cut with as many output channels: the cut closes, as with taps,
       though the result has an axis of that extent there. *)
    let xc = f32 [| 2; 5; 2 |] 1. and none = f32 [| 2; 0; 2 |] 1. in
    let y = Nx.conv (Nx.place (S2.split ~axis:2) xc) (Nx.place S2.on none) in
    equal ~msg:"x's channel cut closes" placement S2.on (Nx.placement y);
    let taps = f32 [| 2; 1; 2 |] 1. in
    equal ~msg:"as it does with taps" placement S2.on
      (Nx.placement
         (Nx.conv (Nx.place (S2.split ~axis:2) xc) (Nx.place S2.on taps)));
    let at k v = Nx.place (Nx.Placement.device S2.v (Nx_support.memory k)) v in
    let refused msg f =
      raises_match ~msg (Exn.invalid_arg ~sub:"Nx.conv: ") (fun () ->
          ignore (f ()))
    in
    refused "w on another device" (fun () -> Nx.conv (at 0 x) (at 1 w));
    refused "init on another device" (fun () ->
        Nx.conv ~init:(at 1 init) (at 0 x) (at 0 w))

  (* The bytes the host holds after four rounds of collection, each of which
     returns to it what it collected. *)
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create Rig.host 0);
      Rig.free_cache Rig.host
    done;
    Rig.allocated Rig.host

  (* The bytes the host holds more after [f ()], its cache included, which count
     every buffer it held at once, below a quarter more than [bytes]: one more
     copy of a 1 MB input passes that. One uncounted run comes first, and the
     count before is read settled, so that no memory collected late serves the
     counted run. *)
  let held msg ~bytes f =
    ignore (f ());
    let before = settled () in
    let y = f () in
    let grew = Rig.allocated Rig.host - before in
    less ~msg int ~than:(5 * bytes / 4) grew;
    Nx.sync y

  (* An input of [[4; 32; 32; 64]] and a [[oc; 3; 3; ic]] kernel of [dt], the
     input's values [x], ready on the host. *)
  let operands (type v s) (dt : (v, s) Nx.dtype) ~oc ~ic =
    let x =
      Nx.init Nx.float32 [| 4; 32; 32; 64 |] (fun i ->
          Float.of_int (i.(1) mod 3))
    and w =
      Nx.init Nx.float32 [| oc; 3; 3; ic |] (fun i ->
          Float.of_int (i.(3) mod 2))
    in
    let x = Nx.cast dt x and w = Nx.cast dt w in
    Nx.sync x;
    Nx.sync w;
    (x, w)

  (* Convolutions over a 1 MB input padded by one, of 64 output channels, which
     take chain order, of 4, which take lane order, in complex64, and depthwise:
     each holds the input padded and its result, never a copy per output channel
     or per tap, nor its windows. *)
  let test_memory () =
    let pad = [| (1, 1); (1, 1) |] in
    let conv (type v s) (dt : (v, s) Nx.dtype) ?(groups = 1) ~oc ~ic name =
      let x, w = operands dt ~oc ~ic in
      let e = D.bytes dt 1 in
      let bytes = (4 * 34 * 34 * 64 * e) + (4 * 32 * 32 * oc * e) in
      held name ~bytes (fun () ->
          let y = Nx.conv ~groups ~padding:pad x w in
          Nx.sync y;
          y)
    in
    conv Nx.float32 ~oc:64 ~ic:64 "float32, 64 output channels";
    conv Nx.float32 ~oc:4 ~ic:64 "float32, 4 output channels";
    conv Nx.complex64 ~oc:4 ~ic:64 "complex64, 4 output channels";
    conv Nx.float32 ~groups:64 ~oc:64 ~ic:1 "float32, depthwise"

  let invalid f = raises_match (Exn.invalid_arg ~sub:"Nx.conv: ") f

  let refusals =
    let z s = Nx.zeros Nx.float32 s in
    let x = z [| 1; 5; 4 |] and w = z [| 2; 3; 4 |] in
    group "raises Invalid_argument naming it"
      [
        test "if x has fewer than three axes" (fun () ->
            invalid (fun () -> Nx.conv (z [| 5; 4 |]) w));
        test "if w has another rank" (fun () ->
            invalid (fun () -> Nx.conv x (z [| 2; 4 |]));
            invalid (fun () -> Nx.conv x (z [| 2; 3; 1; 4 |])));
        test "if stride does not have one entry per spatial axis" (fun () ->
            invalid (fun () -> Nx.conv ~stride:[| 1; 1 |] x w));
        test "if padding does not have one entry per spatial axis" (fun () ->
            invalid (fun () -> Nx.conv ~padding:[| (0, 0); (0, 0) |] x w));
        test "if dilation does not have one entry per spatial axis" (fun () ->
            invalid (fun () -> Nx.conv ~dilation:[| 1; 1 |] x w));
        test "if a stride is below 1" (fun () ->
            invalid (fun () -> Nx.conv ~stride:[| 0 |] x w));
        test "if a dilation is below 1" (fun () ->
            invalid (fun () -> Nx.conv ~dilation:[| 0 |] x w));
        test "if groups is below 1" (fun () ->
            invalid (fun () -> Nx.conv ~groups:0 x w));
        test "if groups does not divide the output channels" (fun () ->
            invalid (fun () ->
                Nx.conv ~groups:3 (z [| 1; 5; 6 |]) (z [| 4; 3; 2 |])));
        test "if x's channels are not groups times w's last extent" (fun () ->
            invalid (fun () -> Nx.conv ~groups:2 x w));
        test "for booleans" (fun () ->
            invalid (fun () ->
                Nx.conv
                  (Nx.zeros Nx.bool [| 1; 5; 4 |])
                  (Nx.zeros Nx.bool [| 2; 3; 4 |])));
        test "if init's shape does not broadcast to the result's" (fun () ->
            invalid (fun () -> Nx.conv ~init:(z [| 3 |]) x w));
      ]

  (* conv_transpose *)

  (* The least input extent along axis [i] whose convolution has [t] windows,
     searched from the count, or [None] where no extent has. The count grows by
     at most one per element, so the search stops past [t]. *)
  let least c i t =
    let rec from d =
      let n = count c i d in
      if n = t then Some d else if n > t then None else from (d + 1)
    in
    from 0

  (* [x]'s spatial extents are [c.spatial], each a count [t]. A count of one
     window under a padding of one each side, which an input of 0 already
     exceeds; that count at stride 2 and 3, which an input of 0 has, below the
     formula's extent, the second the input whose transpose raised before the
     least input was searched for; and nine channels each way over eight
     windows: drawn, each is a few cases in a hundred, which some seeds never
     reach. *)
  let law_transpose =
    let examples =
      let one ~cg ~og ~spatial ~stride ~padding =
        let v i = Float.of_int ((i mod 7) - 3) in
        one_axis ~og ~cg ~spatial ~taps:1 ~stride ~padding ~x:v ~w:v ()
      in
      [
        one ~cg:1 ~og:1 ~spatial:1 ~stride:1 ~padding:(1, 1);
        one ~cg:1 ~og:1 ~spatial:1 ~stride:2 ~padding:(1, 1);
        one ~cg:1 ~og:1 ~spatial:1 ~stride:3 ~padding:(1, 1);
        one ~cg:9 ~og:9 ~spatial:8 ~stride:1 ~padding:(0, 0);
      ]
    in
    prop ~examples
      "is conv's adjoint over the least input, or raises where no input has \
       x's shape"
      case (fun c ->
        let t = c.spatial in
        let least = Array.mapi (least c) t in
        cover "no input has the count" (Array.exists Option.is_none least);
        cover "input channels past 4, over rows past 4"
          (chained_transpose c (Array.map (Option.value ~default:0) least));
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
                  t.(i) > 0
                  && d = Some 0
                  && ((t.(i) - 1) * c.stride.(i)) + dilated c i - lo - hi < 0)
                least));
        let o = c.groups * c.og in
        let gs = Array.concat [ [| c.n |]; t; [| o |] ] in
        let g =
          Nx.init Nx.float64 gs (fun i ->
              Float.of_int ((Array.fold_left ( + ) 0 i mod 5) - 2))
        in
        let w = Nx.create Nx.float64 (w_shape c) c.ws in
        let stride = c.stride
        and padding = c.padding
        and dilation = c.dilation in
        let groups = c.groups in
        if Array.exists Option.is_none least then
          raises_match (Exn.invalid_arg ~sub:"Nx.conv_transpose: ") (fun () ->
              Nx.conv_transpose ~stride ~padding ~dilation ~groups g w)
        else
          let d = Array.map Option.get least in
          let z_shape = Array.concat [ [| c.n |]; d; [| c.groups * c.cg |] ] in
          let z =
            Nx.init Nx.float64 z_shape (fun i ->
                Float.of_int
                  ((Array.fold_left ( * ) 1 (Array.map succ i) mod 7) - 3))
          in
          let y = Nx.conv_transpose ~stride ~padding ~dilation ~groups g w in
          equal ~msg:"shape" (array int) z_shape (Nx.shape y);
          (* Small integers: float64 holds both sums exactly. *)
          let conv = Nx.conv ~stride ~padding ~dilation ~groups z w in
          let lhs = Nx.item [] (Nx.sum (Nx.mul conv g)) in
          let rhs = Nx.item [] (Nx.sum (Nx.mul z y)) in
          equal ~msg:"adjoint" float_exact lhs rhs)

  (* conv_transpose has the bits of conv at stride one over its input spread,
     [stride - 1] zeros between its elements, by [w] reversed along its taps
     with its channels exchanged, padded so that the result has the least
     input's extents: [span - lo] before and [d + span - spread - (span - lo)]
     after, [span] the dilated kernel's extent less one; from an [init] too,
     which each adds as conv does. In complex64, where each path's order shows,
     and at float32, float16 and int32, whose carriers differ. *)
  let law_transpose_order =
    prop ~count:250
      "sums as conv at stride one over x spread, by w reversed, from the same \
       init"
      float_case (fun c ->
        let k = Array.length c.spatial and t = c.spatial in
        let least = Array.mapi (least c) t in
        let whole =
          Array.for_all Option.is_some least && not (Array.mem 0 c.taps)
        in
        let outputs = whole && c.n > 0 && Array.for_all (fun e -> e > 0) t in
        cover "one input and one output channel a group, under an output"
          (outputs && c.cg = 1 && c.og = 1);
        cover "other than one of each, under an output"
          (outputs && (c.cg <> 1 || c.og <> 1));
        cover "a stride above 1, under an output"
          (outputs && Array.exists (fun e -> e > 1) c.stride);
        if whole then begin
          let d = Array.map Option.get least in
          let o = c.groups * c.og and m = Array.length c.xs in
          let value j = if m = 0 then 0. else c.xs.(j mod m) in
          let spread i =
            if t.(i) = 0 then 0 else ((t.(i) - 1) * c.stride.(i)) + 1
          in
          let x_shape' = Array.concat [ [| c.n |]; t; [| o |] ] in
          let spread_shape =
            Array.concat [ [| c.n |]; Array.init k spread; [| o |] ]
          in
          let z_shape = Array.concat [ [| c.n |]; d; [| c.groups * c.cg |] ] in
          (* The element of [x] at each element of [x] spread, none at the zeros
             between. *)
          let spread_at j =
            let idx = unravel spread_shape j in
            let on a = idx.(1 + a) mod c.stride.(a) = 0 in
            if not (Array.for_all on (Array.init k Fun.id)) then None
            else
              Some
                (ravel x_shape'
                   (Array.mapi
                      (fun a e ->
                        if a >= 1 && a <= k then e / c.stride.(a - 1) else e)
                      idx))
          in
          let span i = c.dilation.(i) * (c.taps.(i) - 1) in
          let lo = Array.init k (fun i -> span i - fst c.padding.(i)) in
          let hi = Array.init k (fun i -> d.(i) + span i - spread i - lo.(i)) in
          let w_shape' =
            Array.concat [ [| c.groups * c.cg |]; c.taps; [| c.og |] ]
          in
          (* The element of [w] at each element of [w] reversed. *)
          let reversed_at j =
            let idx = unravel w_shape' j in
            let g = idx.(0) / c.cg and i = idx.(0) mod c.cg in
            let q = Array.init k (fun a -> c.taps.(a) - 1 - idx.(1 + a)) in
            let j = idx.(k + 1) in
            ravel (w_shape c)
              (Array.concat [ [| (g * c.og) + j |]; q; [| i |] ])
          in
          let spread_of zero xs =
            Array.init (numel spread_shape) (fun j ->
                match spread_at j with Some i -> xs.(i) | None -> zero)
          in
          let reversed_of ws =
            Array.init (numel w_shape') (fun j -> ws.(reversed_at j))
          in
          let xv = Array.init (numel x_shape') value in
          let iv =
            Array.init (numel z_shape) (fun j -> 0.75 *. value (j + 1))
          in
          let check name bits x_spread w_reversed x w init =
            let want ?init () =
              Nx.conv ?init
                ~padding:(Array.map2 (fun a b -> (a, b)) lo hi)
                ~dilation:c.dilation ~groups:c.groups x_spread w_reversed
            in
            let got ?init () =
              Nx.conv_transpose ?init ~stride:c.stride ~padding:c.padding
                ~dilation:c.dilation ~groups:c.groups x w
            in
            equal ~msg:(name ^ ", shape") (array int)
              (Nx.shape (want ()))
              (Nx.shape (got ()));
            equal ~msg:name (array int64) (bits (want ())) (bits (got ()));
            equal ~msg:(name ^ ", from an init") (array int64)
              (bits (want ~init ()))
              (bits (got ~init ()))
          in
          let z s v = Nx.create Nx.complex64 s v in
          let xz = complex xv and wz = complex c.ws in
          check "complex64"
            (fun y -> Array.map Int64.bits_of_float (parts (Nx.to_array y)))
            (z spread_shape (spread_of Complex.zero xz))
            (z w_shape' (reversed_of wz))
            (z x_shape' xz)
            (z (w_shape c) wz)
            (z z_shape (complex iv));
          let real (type v s) (dt : (v, s) D.t) =
            let at s v = Nx.cast dt (Nx.create Nx.float64 s v) in
            check (D.name dt)
              (fun y ->
                Array.map Int64.bits_of_float
                  (Nx.to_array (Nx.cast Nx.float64 y)))
              (at spread_shape (spread_of 0. xv))
              (at w_shape' (reversed_of c.ws))
              (at x_shape' xv)
              (at (w_shape c) c.ws)
              (at z_shape iv)
          in
          real D.Float32;
          real D.Float16;
          real D.Int32
        end)

  (* An output that sums no term, along a kernel extent of 0, is [init]'s
     element, bit for bit. *)
  let test_transpose_no_terms () =
    let f32 s v = Nx.create Nx.float32 s (Array.make (numel s) v) in
    let init = Nx.create Nx.float32 [| 4 |] [| -0.; Float.nan; -0.; 1. |] in
    let y =
      Nx.conv_transpose ~init (f32 [| 1; 6; 2 |] 1.) (f32 [| 2; 0; 4 |] 1.)
    in
    let bits y =
      Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 y))
    in
    equal (array int64)
      (bits (Nx.copy (Nx.broadcast_to (Nx.shape y) init)))
      (bits y)

  (* Three channels reach one output: as for conv, 2048 then 1 and 1 sum to 2050
     at float32, which float16 holds. *)
  let test_transpose_narrow () =
    let f16 s vs = Nx.create Nx.float16 s vs in
    equal (array float_exact) [| 2050. |]
      (Nx.to_array
         (Nx.conv_transpose
            (f16 [| 1; 1; 3 |] [| 2048.; 1.; 1. |])
            (f16 [| 3; 1; 1 |] [| 1.; 1.; 1. |])))

  (* A sum of 2048 and 1 is exact at float32 and a tie at float16, where it
     rounds to 2048, and a bias of -2048 then leaves 1, which two roundings
     would make 0; at bfloat16 the same of 256 and 1. Over two channels and over
     two taps of one channel. *)
  let test_transpose_init () =
    List.iter
      (fun (D.Any dt, big) ->
        let v s vs = Nx.cast dt (Nx.create Nx.float64 s vs) in
        let bias = v [| 1 |] [| -.big |] in
        let ones s = v s (Array.make (numel s) 1.) in
        let check msg want y =
          equal
            ~msg:(D.name dt ^ ", " ^ msg)
            (array float_exact) want
            (Nx.to_array (Nx.cast Nx.float64 y))
        in
        check "two channels" [| 1. |]
          (Nx.conv_transpose ~init:bias
             (v [| 1; 1; 2 |] [| big; 1. |])
             (ones [| 2; 1; 1 |]));
        check "two taps"
          [| 0.; 1.; 1. -. big |]
          (Nx.conv_transpose ~init:bias
             (v [| 1; 2; 1 |] [| big; 1. |])
             (ones [| 1; 2; 1 |])))
      [ (D.Any D.Float16, 2048.); (D.Any D.Bfloat16, 256.) ]

  (* 100 + 100 is 200, which int8 wraps to -56. *)
  let test_transpose_wrap () =
    let i8 s vs = Nx.create Nx.int8 s vs in
    equal (array int) [| -56 |]
      (Nx.to_array
         (Nx.conv_transpose
            (i8 [| 1; 1; 2 |] [| 100; 100 |])
            (i8 [| 2; 1; 1 |] [| 1; 1 |])))

  (* A kernel of no tap reads every operand where it lies: one alone on another
     device raises. *)
  let test_transpose_routing () =
    let f32 s v = Nx.create Nx.float32 s (Array.make (numel s) v) in
    let at k v = Nx.place (Nx.Placement.device S2.v (Nx_support.memory k)) v in
    raises_match (Exn.invalid_arg ~sub:"Nx.conv_transpose: ") (fun () ->
        Nx.conv_transpose
          (at 0 (f32 [| 1; 6; 2 |] 1.))
          (at 1 (f32 [| 2; 0; 4 |] 1.)))

  (* A transpose at stride one over a 1 MB input, which holds it padded once and
     its result. *)
  let test_transpose_memory () =
    let x, w = operands Nx.float32 ~oc:64 ~ic:64 in
    let bytes = (4 * 34 * 34 * 64 * 4) + (4 * 32 * 32 * 64 * 4) in
    held "float32, 64 channels" ~bytes (fun () ->
        let y = Nx.conv_transpose ~padding:[| (1, 1); (1, 1) |] x w in
        Nx.sync y;
        y)

  let transpose_refusals =
    let z s = Nx.zeros Nx.float32 s in
    let w = z [| 2; 3; 4 |] in
    let invalid f =
      raises_match (Exn.invalid_arg ~sub:"Nx.conv_transpose: ") f
    in
    group "raises Invalid_argument naming it"
      [
        test "if x has fewer than three axes" (fun () ->
            invalid (fun () -> Nx.conv_transpose (z [| 3; 2 |]) w));
        test "if a stride is below 1" (fun () ->
            invalid (fun () ->
                Nx.conv_transpose ~stride:[| 0 |] (z [| 1; 3; 2 |]) w));
        test "if groups does not divide x's channels" (fun () ->
            invalid (fun () ->
                Nx.conv_transpose ~groups:4 (z [| 1; 3; 6 |]) (z [| 6; 3; 2 |])));
        test "for booleans" (fun () ->
            invalid (fun () ->
                Nx.conv_transpose
                  (Nx.zeros Nx.bool [| 1; 3; 2 |])
                  (Nx.zeros Nx.bool [| 2; 3; 4 |])));
        test "if no input has a conv of x's shape" (fun () ->
            invalid (fun () -> Nx.conv_transpose (z [| 1; 5; 4 |]) w);
            invalid (fun () ->
                Nx.conv_transpose ~padding:[| (3, 3) |] (z [| 1; 0; 2 |]) w));
        test "if init's shape does not broadcast to the result's" (fun () ->
            invalid (fun () ->
                Nx.conv_transpose ~init:(z [| 2 |]) (z [| 1; 3; 2 |]) w));
      ]

  (* ceil_padding *)

  (* Every axis of an extent up to 20, below and past every dilated kernel here
     and of each residue modulo each stride, with each kernel extent up to 7,
     stride up to 4 and dilation up to 3: 1764 values, each checked. *)
  let axis_case =
    Gen.quad (Gen.int_range 0 20) (Gen.int_range 1 7) (Gen.int_range 1 4)
      (Gen.int_range 1 3)

  let law_windows =
    prop ~count:1764 "gives conv ⌈n / stride⌉ windows along an axis of extent n"
      axis_case (fun (n, k, s, d) ->
        let padding =
          Nx.ceil_padding ~stride:[| s |] ~dilation:[| d |] ~kernel_size:[| k |]
            [| n |]
        in
        let y =
          Nx.conv ~stride:[| s |] ~dilation:[| d |] ~padding
            (Nx.zeros Nx.float32 [| 1; n; 1 |])
            (Nx.zeros Nx.float32 [| 1; k; 1 |])
        in
        equal int ((n + s - 1) / s) (Nx.dim 1 y))

  (* nx.mli: "it pads [max 0 ((⌈n / stride⌉ - 1) stride + e - n)] in all, [e]
     the dilated kernel's extent [dilation (k - 1) + 1], half of it (rounded
     down) before and the rest after." *)
  let pads n k s d =
    let e = (d * (k - 1)) + 1 in
    let total = Int.max 0 (((((n + s - 1) / s) - 1) * s) + e - n) in
    (total / 2, total - (total / 2))

  let law_pads =
    prop ~count:1764
      "pads max 0 ((⌈n / stride⌉ - 1) stride + e - n), half of it before, \
       rounded down"
      axis_case (fun (n, k, s, d) ->
        equal
          (array (pair int int))
          [| pads n k s d |]
          (Nx.ceil_padding ~stride:[| s |] ~dilation:[| d |]
             ~kernel_size:[| k |] [| n |]))

  let ceil_refusals =
    let invalid f = raises_match (Exn.invalid_arg ~sub:"Nx.ceil_padding: ") f in
    group "raises Invalid_argument naming it"
      [
        test "if stride does not have one entry per kernel extent" (fun () ->
            invalid (fun () ->
                Nx.ceil_padding ~stride:[| 2 |] ~kernel_size:[| 3; 3 |]
                  [| 4; 4 |]));
        test "if dilation does not have one entry per kernel extent" (fun () ->
            invalid (fun () ->
                Nx.ceil_padding ~dilation:[| 2 |] ~kernel_size:[| 3; 3 |]
                  [| 4; 4 |]));
        test "if s does not have one entry per kernel extent" (fun () ->
            invalid (fun () -> Nx.ceil_padding ~kernel_size:[| 3; 3 |] [| 4 |]));
        test "if an extent of s is negative" (fun () ->
            invalid (fun () -> Nx.ceil_padding ~kernel_size:[| 3 |] [| -1 |]));
        test "if a kernel extent is below 1" (fun () ->
            invalid (fun () -> Nx.ceil_padding ~kernel_size:[| 0 |] [| 4 |]));
        test "if a stride is below 1" (fun () ->
            invalid (fun () ->
                Nx.ceil_padding ~stride:[| 0 |] ~kernel_size:[| 3 |] [| 4 |]));
        test "if a dilation is below 1" (fun () ->
            invalid (fun () ->
                Nx.ceil_padding ~dilation:[| 0 |] ~kernel_size:[| 3 |] [| 4 |]));
      ]

  let group =
    group "convolution"
      [
        group "conv"
          [
            law_conv;
            test
              "sums no term along a kernel extent of 0 or in a group of no \
               channel, giving init's elements bit for bit"
              test_no_terms;
            law_order;
            law_planes;
            law_bound;
            law_init;
            law_add;
            test "sums narrow floats at float32" test_narrow;
            test "reads its operands where they lie, with a kernel extent of 0"
              test_no_taps_routing;
            test "holds x padded once, never its windows" test_memory;
            refusals;
          ];
        group "conv_transpose"
          [
            law_transpose;
            law_transpose_order;
            test
              "sums no term along a kernel extent of 0, giving init's elements \
               bit for bit"
              test_transpose_no_terms;
            test "sums narrow floats at float32" test_transpose_narrow;
            test "from an init rounds once" test_transpose_init;
            test "wraps integers" test_transpose_wrap;
            test "reads its operands where they lie, with a kernel extent of 0"
              test_transpose_routing;
            test "holds x padded once" test_transpose_memory;
            transpose_refusals;
          ];
        group "ceil_padding"
          [
            law_windows;
            law_pads;
            test "pads each axis alone" (fun () ->
                equal
                  (array (pair int int))
                  [| pads 56 3 2 1; pads 224 7 2 1 |]
                  (Nx.ceil_padding ~stride:[| 2; 2 |] ~kernel_size:[| 3; 7 |]
                     [| 56; 224 |]));
            ceil_refusals;
          ];
      ]
end

let () =
  exit
    (run "nx contractions"
       [ contraction; contract; einsum; matmul; dot; Scaled.group; Conv.group ])
