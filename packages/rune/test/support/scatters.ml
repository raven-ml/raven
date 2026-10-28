(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Operand
open Witness

type positions = Full | Trailing | Leading | Transposed

type t = {
  dtype : Nx.Dtype.any;
  combine : Nx.combine;
  into_shape : int array;
  updates_shape : int array;
  positions : positions;
  pos : int64 array;
  into : Operand.t;
  updates : Operand.t;
  donated : bool;
}

let combine_name : Nx.combine -> string = function
  | Set -> "set"
  | Add -> "add"
  | Max -> "max"
  | Min -> "min"

let positions_name = function
  | Full -> "full"
  | Trailing -> "trailing"
  | Leading -> "leading"
  | Transposed -> "transposed"

let pp ppf c =
  let (Nx.Dtype.Any dt) = c.dtype in
  let values ppf (o : Operand.t) =
    Format.pp_print_array
      ~pp_sep:(fun ppf () -> Format.pp_print_char ppf ' ')
      pp_value ppf o.values
  in
  Format.fprintf ppf "%s %a%s %a into %a [%a] updates %a [%a] pos %s [%s]"
    (combine_name c.combine) Nx.pp_dtype dt
    (if c.donated then " donated" else "")
    pp_shape c.into_shape pp_view c.into.view values c.into pp_view
    c.updates.view values c.updates
    (positions_name c.positions)
    (String.concat ";" (Array.to_list (Array.map Int64.to_string c.pos)))

(* Dtypes *)

let dtypes =
  Nx.Dtype.
    [
      Any Nx.float32;
      Any Nx.float64;
      Any Nx.float16;
      Any Nx.bfloat16;
      Any Nx.complex64;
      Any Nx.int8;
      Any Nx.int32;
      Any Nx.uint64;
      Any Nx.bool;
    ]

let is_bool (Nx.Dtype.Any dt) =
  match Nx.Dtype.kind dt with Boolean -> true | _ -> false

let is_floating (Nx.Dtype.Any dt) =
  match Nx.Dtype.kind dt with Float | Complex -> true | _ -> false

let is_complex (Nx.Dtype.Any dt) =
  match Nx.Dtype.kind dt with Complex -> true | _ -> false

let is_narrow (Nx.Dtype.Any dt as d) =
  is_floating d && (not (is_complex d)) && Nx.Dtype.bits dt < 32

let combines d : Nx.combine list =
  if is_bool d then [ Set; Max; Min ] else [ Set; Add; Max; Min ]

(* [exactly d c] is whether the case's values are halves, which an Add of
   floats sums exactly in any order. *)
let exactly d (c : Nx.combine) = c = Add && is_floating d

(* Positions *)

(* The shape the positions of [t] = [[p; k; q]] are drawn at. *)
let drawn kind t =
  let p = t.(0) and k = t.(1) and q = t.(2) in
  match kind with
  | Full -> [| p; k; q |]
  | Trailing -> [| p; k; 1 |]
  | Leading -> [| 1; k; q |]
  | Transposed -> [| 1; k; p |]

(* The drawn positions [i] at [t], inside the traced function. *)
let shaped kind t i =
  match kind with
  | Full -> i
  | Trailing | Leading -> Nx.broadcast_to t i
  | Transposed ->
      let wide = Nx.broadcast_to [| t.(2); t.(1); t.(0) |] i in
      Nx.transpose ~axes:[ 2; 1; 0 ] wide

(* Generated cases *)

let half =
  Gen.frequency
    [
      (8, Gen.map (fun h -> Float.of_int h /. 2.) (Gen.int_range (-16) 16));
      (1, Gen.constant (-0.));
    ]

let halved d (o : Operand.t) =
  let open Gen in
  let value =
    if is_complex d then
      let* re = half in
      let+ im = half in
      C (re, im)
    else map (fun x -> F x) half
  in
  let+ values = array ~size:(constant (Array.length o.values)) value in
  { o with values }

let gen =
  let open Gen in
  let* d = of_list dtypes in
  let* combine = of_list (combines d) in
  let* p = int_range 1 2 in
  let* q = int_range 1 3 in
  let* n = int_range 33 40 in
  let* k = int_range 33 48 in
  let s = [| p; n; q |] and t = [| p; k; q |] in
  let* positions = of_list [ Full; Trailing; Leading; Transposed ] in
  let at =
    frequency
      [
        (8, map Int64.of_int (int_range 0 (n - 1)));
        (1, of_list [ -1L; Int64.of_int n; Int64.max_int; Int64.min_int ]);
      ]
  in
  let* pos = array ~size:(constant (numel (drawn positions t))) at in
  let* into = draw s in
  let* updates = draw t in
  let exact = exactly d combine in
  let* into = if exact then halved d into else constant into in
  let* updates = if exact then halved d updates else constant updates in
  let+ donated = bool in
  {
    dtype = d;
    combine;
    into_shape = s;
    updates_shape = t;
    positions;
    pos;
    into;
    updates;
    donated;
  }

(* Examples *)

let example i (d, (combine : Nx.combine), positions, offset, donated) =
  let s = [| 2; 33; 3 |] and t = [| 2; 40; 3 |] in
  let ps = [| 0L; 0L; 2L; -1L; 33L; 1L; Int64.max_int; 32L; Int64.min_int |] in
  let operand view m : Operand.t =
    let value j =
      if not (exactly d combine) then pool.((i + j) mod Array.length pool)
      else
        let x = Float.of_int ((j mod 9) - 4) /. 2. in
        if is_complex d then C (x, -.x) else F x
    in
    { view; values = Array.init m value }
  in
  let updates =
    if offset then operand (Offset 2) (numel t + 2) else operand Plain (numel t)
  in
  let m = numel (drawn positions t) in
  {
    dtype = d;
    combine;
    into_shape = s;
    updates_shape = t;
    positions;
    pos = Array.init m (fun j -> ps.(j mod Array.length ps));
    into = operand Plain (numel s);
    updates;
    donated;
  }

let examples =
  List.mapi example
    Nx.Dtype.
      [
        (Any Nx.float32, Set, Full, false, true);
        (Any Nx.float64, Add, Trailing, false, false);
        (Any Nx.float16, Add, Leading, false, true);
        (Any Nx.bfloat16, Add, Transposed, true, false);
        (Any Nx.complex64, Add, Full, true, true);
        (Any Nx.complex64, Max, Trailing, false, false);
        (Any Nx.complex64, Min, Leading, false, true);
        (Any Nx.int8, Set, Transposed, false, false);
        (Any Nx.int32, Max, Full, true, false);
        (Any Nx.uint64, Min, Trailing, false, true);
        (Any Nx.bool, Set, Leading, false, false);
        (Any Nx.bool, Max, Transposed, false, true);
        (Any Nx.float32, Min, Trailing, true, true);
      ]

(* The law *)

let law (module S : Nx.Devices) c =
  let (Nx.Dtype.Any dt) = c.dtype in
  List.iter
    (fun k -> cover (combine_name k) (c.combine = k))
    [ Set; Add; Max; Min ];
  cover "a float" (is_floating c.dtype && not (is_complex c.dtype));
  cover "a complex number" (is_complex c.dtype);
  cover "a boolean" (is_bool c.dtype);
  cover "an integer" (not (is_floating c.dtype || is_bool c.dtype));
  cover "a narrow float's add" (is_narrow c.dtype && c.combine = Add);
  List.iter
    (fun k ->
      let complex = is_complex c.dtype && c.combine = k in
      cover ("a complex " ^ combine_name k) complex)
    [ Add; Max; Min ];
  cover "a donated target" c.donated;
  cover "an offset view of the updates"
    (match c.updates.view with Offset _ -> true | _ -> false);
  List.iter
    (fun k -> cover (positions_name k ^ " positions") (c.positions = k))
    [ Full; Trailing; Leading; Transposed ];
  let q = c.into_shape.(2) in
  let trailing = c.positions = Trailing || c.positions = Transposed in
  cover "positions broadcast along the trailing axes" (trailing && q > 1);
  cover "positions not broadcast along the trailing axes"
    ((not trailing) && q > 1);
  let n = c.into_shape.(1) in
  cover "a position outside the axis"
    (Array.exists (fun p -> p < 0L || p >= Int64.of_int n) c.pos);
  let t = c.updates_shape in
  let x = make dt c.into_shape c.into and u = make dt t c.updates in
  let i = Nx.create Nx.int64 (drawn c.positions t) c.pos in
  let f i u x =
    Nx.scatter ~combine:c.combine ~axis:1 (shaped c.positions t i) u x
  in
  let eager = outcome (fun () -> codes (f i u x)) in
  let g =
    Rune.jit Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor) f
  in
  let put y = Nx.place S.on y in
  let x = if c.donated then Nx.donate (put (Nx.copy x)) else put x in
  let before = Rune.evaluations () in
  let call () = codes (Nx.place Nx.Host.on (g (put i) (put u) x)) in
  let y = outcome call in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal (outcome_t codes_t) eager y
