(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Operand
open Witness

type positions = Trailing | Transposed | Written

type t = {
  dtype : Nx.Dtype.any;
  into_shape : int array;
  updates_shape : int array;
  positions : positions;
  pos : int64 array;
  seed : int;
  donated : bool;
}

let positions_name = function
  | Trailing -> "trailing"
  | Transposed -> "transposed"
  | Written -> "written"

let pp ppf c =
  let (Nx.Dtype.Any dt) = c.dtype in
  Format.fprintf ppf "%a%s %a updates %a pos %s [%s] seed %d" Nx.pp_dtype dt
    (if c.donated then " donated" else "")
    pp_shape c.into_shape pp_shape c.updates_shape
    (positions_name c.positions)
    (String.concat ";" (Array.to_list (Array.map Int64.to_string c.pos)))
    c.seed

let dtypes =
  Nx.Dtype.
    [
      Any Nx.bfloat16;
      Any Nx.float16;
      Any Nx.float32;
      Any Nx.float64;
      Any Nx.complex64;
      Any Nx.int8;
      Any Nx.int64;
    ]

(* The shape the positions of [t] = [[p; k; q]] are drawn at. *)
let drawn kind t =
  match kind with
  | Trailing -> [| t.(0); t.(1); 1 |]
  | Transposed -> [| 1; t.(1); t.(0) |]
  | Written -> [| t.(1) |]

(* The scatter of [u] into [x] at the drawn positions [i], traced. *)
let scatter c i u x =
  let t = c.updates_shape in
  match c.positions with
  | Trailing -> Nx.scatter ~axis:1 (Nx.broadcast_to t i) u x
  | Transposed ->
      let wide = Nx.broadcast_to [| t.(2); t.(1); t.(0) |] i in
      Nx.scatter ~axis:1 (Nx.transpose ~axes:[ 2; 1; 0 ] wide) u x
  | Written ->
      let ps = Array.to_list (Array.map Int64.to_int c.pos) in
      Nx.set [ A; L ps; A ] u x

(* The regime a GPU's scatter of [c] takes. *)
type regime = Walked | Elected | Parted | Distinct

let regime ren c =
  let p = c.updates_shape.(0) and k = c.updates_shape.(1) in
  let n = c.into_shape.(1) and q = c.into_shape.(2) in
  match (c.positions, Rune.Lower.scatter_election ren ~p ~n ~q ~k) with
  | Written, _ -> Distinct
  | _, None -> Walked
  | _, Some 1 -> Elected
  | _, Some _ -> Parted

let regime_name = function
  | Walked -> "a walk outside the election's bounds"
  | Elected -> "an election in one part"
  | Parted -> "an election in parts"
  | Distinct -> "distinct written targets"

(* Generated cases *)

let gen =
  let open Gen in
  let* d = of_list dtypes in
  let* p = int_range 1 3 in
  let* q = int_range 1 4 in
  let* positions =
    frequency
      [
        (2, constant Trailing);
        (2, constant Transposed);
        (1, constant Written);
      ]
  in
  let* k =
    frequency
      [ (1, int_range 1 15); (3, int_range 16 96); (1, int_range 700 1100) ]
  in
  (* Few targets repeat positions; written ones need [k] of them. *)
  let* n =
    if positions = Written then int_range k (k + 8)
    else frequency [ (1, int_range 1 8); (1, int_range (k / 2) (k + 8)) ]
  in
  let n = Int.max 1 n in
  let t = [| p; k; q |] in
  let* pos =
    if positions = Written then
      let+ seed = int in
      let st = Random.State.make [| seed |] in
      let all = Array.init n Int64.of_int in
      for i = n - 1 downto 1 do
        let j = Random.State.int st (i + 1) in
        let v = all.(i) in
        all.(i) <- all.(j);
        all.(j) <- v
      done;
      Array.sub all 0 k
    else
      let at =
        frequency
          [
            (8, map Int64.of_int (int_range 0 (n - 1)));
            (1, of_list [ -1L; Int64.of_int n; Int64.max_int; Int64.min_int ]);
          ]
      in
      array ~size:(constant (numel (drawn positions t))) at
  in
  let* seed = int in
  let+ donated = bool in
  {
    dtype = d;
    into_shape = [| p; n; q |];
    updates_shape = t;
    positions;
    pos;
    seed;
    donated;
  }

(* Examples *)

let example i (d, positions, p, n, k, q) =
  let t = [| p; k; q |] in
  let pos =
    match positions with
    | Written -> Array.init k (fun j -> Int64.of_int (n - 1 - j))
    | Trailing | Transposed ->
        let m = numel (drawn positions t) in
        Array.init m (fun j ->
            if j = m / 2 then Int64.of_int n else Int64.of_int (j * 7 mod n))
  in
  {
    dtype = d;
    into_shape = [| p; n; q |];
    updates_shape = t;
    positions;
    pos;
    seed = i;
    donated = i mod 2 = 0;
  }

let examples =
  List.mapi example
    Nx.Dtype.
      [
        (Any Nx.bfloat16, Trailing, 2, 5, 3, 4);
        (Any Nx.float32, Transposed, 3, 13, 40, 2);
        (Any Nx.int8, Trailing, 1, 40, 16, 1);
        (Any Nx.complex64, Trailing, 2, 1100, 1000, 8);
        (Any Nx.bfloat16, Written, 2, 40, 33, 4);
        (Any Nx.complex64, Written, 1, 20, 17, 3);
      ]

(* The law *)

(* The tangent of [scatter c i u x] along [u], whose scatters add under
   [Written]'s promise. *)
let tangent c i u x = snd (Rune.jvp' (fun u -> scatter c i u x) u u)

let law ren (module S : Nx.Devices) c =
  let (Nx.Dtype.Any dt) = c.dtype in
  let r = regime ren c in
  List.iter
    (fun g -> cover (regime_name g) (r = g))
    [ Walked; Elected; Parted; Distinct ];
  let t = c.updates_shape and n = c.into_shape.(1) in
  let drawn_pos = c.positions <> Written in
  cover "positions broadcast along the trailing axes"
    (drawn_pos && t.(2) > 1);
  cover "more than one index before the axis" (t.(0) > 1);
  let inside = List.filter (fun p -> p >= 0L && p < Int64.of_int n) in
  let seen = inside (Array.to_list c.pos) in
  cover "a repeated position"
    (drawn_pos && List.length (List.sort_uniq compare seen) < List.length seen);
  cover "a position outside the axis"
    (List.length seen < Array.length c.pos);
  let float, complex =
    match Nx.Dtype.kind dt with
    | Float -> (true, false)
    | Complex -> (false, true)
    | _ -> (false, false)
  in
  cover "a float" float;
  cover "a complex number" complex;
  cover "an integer" (not (float || complex));
  let added = c.positions = Written && (float || complex) in
  cover "a tangent added at distinct targets" added;
  cover "a complex tangent added at distinct targets" (added && complex);
  cover "a donated target" c.donated;
  let st = Random.State.make [| c.seed |] in
  let x = random_bits dt c.into_shape st in
  let u = random_bits dt t st in
  let i = Nx.create Nx.int64 (drawn c.positions t) c.pos in
  let eager = raw (scatter c i u x) in
  let g =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
      (scatter c)
  in
  let put y = Nx.place S.on y in
  let host_x = x in
  let x = if c.donated then Nx.donate (put (Nx.copy x)) else put x in
  let before = Rune.evaluations () in
  let y = raw (Nx.place Nx.Host.on (g (put i) (put u) x)) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal codes_t eager y;
  if added then begin
    let x = host_x in
    let eager = codes (tangent c i u x) in
    let g =
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
        (tangent c)
    in
    let y = codes (Nx.place Nx.Host.on (g (put i) (put u) (put x))) in
    equal ~msg:"tangent" codes_t eager y
  end
