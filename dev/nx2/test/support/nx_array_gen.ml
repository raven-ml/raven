(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Layouts and movements drawn for nx2's suites, and their positions by brute
   force. *)

open Windtrap
module L = Nx_array.Layout
module M = Nx_array.Move

(* Indices and positions, by brute force *)

(* Every index of shape [s], in C order. *)
let indices s =
  let r = Array.length s in
  let rec from i =
    if i = r then [ [] ]
    else
      let rest = from (i + 1) in
      List.concat_map
        (fun j -> List.map (fun idx -> j :: idx) rest)
        (List.init s.(i) Fun.id)
  in
  List.map Array.of_list (from 0)

let position l idx =
  let p = ref (L.offset l) in
  Array.iteri (fun i j -> p := !p + (j * L.stride l i)) idx;
  !p

let positions l = List.map (position l) (indices (L.shape l))
(* Generators *)

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_move ppf = function
  | M.Reshape s -> Format.fprintf ppf "Reshape %a" pp_ints s
  | M.Broadcast s -> Format.fprintf ppf "Broadcast %a" pp_ints s
  | M.Permute p -> Format.fprintf ppf "Permute %a" pp_ints p
  | M.Slice rs ->
      Format.fprintf ppf "Slice [%s]"
        (String.concat "; "
           (Array.to_list
              (Array.map
                 (fun (x : M.range) ->
                   Printf.sprintf "%d,%d,%d" x.start x.count x.step)
                 rs)))
  | M.Window ws ->
      Format.fprintf ppf "Window [%s]"
        (String.concat "; "
           (Array.to_list
              (Array.map
                 (fun (w : M.window) ->
                   Printf.sprintf "axis %d size %d step %d dilation %d" w.axis
                     w.size w.step w.dilation)
                 ws)))

let const n = Gen.constant ~pp:Format.pp_print_int n
let ints_of l = Gen.of_list ~pp:Format.pp_print_int l
let extent = Gen.frequency [ (1, Gen.int_range 0 0); (6, Gen.int_range 1 4) ]
let shape = Gen.array ~size:(Gen.int_range 0 4) extent

(* Layouts over arbitrary strides and offsets. *)
(* [o] raised by the reach of [s]'s negative strides, so that [o] is the least
   position: positions are non-negative. *)
let lift s strides o =
  let o = ref o in
  Array.iteri
    (fun i d ->
      if d > 1 && strides.(i) < 0 then o := !o - ((d - 1) * strides.(i)))
    s;
  !o

(* Layouts over arbitrary strides, their least position at an arbitrary
   offset. *)
let strided_of s =
  let open Gen in
  let+ strides = array ~size:(const (Array.length s)) (int_range (-5) 5)
  and+ least = int_range 0 40 in
  L.v ~offset:(lift s strides least) ~strides s

let strided = Gen.bind shape strided_of

let range d =
  let open Gen in
  if d = 0 then
    let+ step = ints_of [ -2; -1; 1; 2 ] in
    { M.start = 0; count = 0; step }
  else
    let* step = ints_of [ -3; -2; -1; 1; 2; 3 ] in
    let* start = int_range 0 (d - 1) in
    let room = if step > 0 then (d - 1 - start) / step else start / -step in
    let+ count = int_range 0 (room + 1) in
    { M.start; count; step }

(* Windows on some axes of [s]; [apart] keeps windows from overlapping. *)
let windows ~apart s =
  let open Gen in
  let r = Array.length s in
  let axes = List.filter (fun i -> s.(i) >= 1) (List.init r Fun.id) in
  let* axes = subsequence ~pp:Format.pp_print_int axes in
  (* Each window appends an axis: keep the result within the most axes. *)
  let axes = List.filteri (fun i _ -> r + i < L.max_rank) axes in
  let window axis =
    let d = s.(axis) in
    let* size = int_range 1 d in
    let* dilation =
      if size = 1 then int_range 1 3 else int_range 1 ((d - 1) / (size - 1))
    in
    let reach = (dilation * (size - 1)) + 1 in
    let+ step = if apart then int_range reach (reach + 2) else int_range 1 3 in
    { M.axis; size; step; dilation }
  in
  let rec all = function
    | [] -> constant []
    | a :: rest ->
        let+ w = window a and+ ws = all rest in
        w :: ws
  in
  let+ ws = all axes in
  M.Window (Array.of_list ws)

(* Shapes of [n] elements, of rank up to 4, with extents of 1 among them. *)
let reshape_target s =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  if n = 0 then (
    let+ s' = array ~size:(int_range 1 4) (int_range 0 3) in
    let s' = Array.copy s' in
    s'.(0) <- 0;
    M.Reshape s')
  else
    (* Deal [n]'s prime factors to four axes, then drop or keep ones. *)
    let rec primes n p =
      if n = 1 then []
      else if n mod p = 0 then p :: primes (n / p) p
      else primes n (p + 1)
    in
    let ps = primes n 2 in
    let+ slots = list ~size:(const (List.length ps)) (int_range 0 3)
    and+ keep = array ~size:(const 4) bool in
    let s' = Array.make 4 1 in
    List.iter2 (fun p k -> s'.(k) <- s'.(k) * p) ps slots;
    M.Reshape
      (Array.of_list
         (List.filteri (fun i d -> d > 1 || keep.(i)) (Array.to_list s')))

let movement ~apart s =
  let open Gen in
  let r = Array.length s in
  let broadcast =
    let* extra = array ~size:(int_range 0 2) (int_range 0 3) in
    let+ fill = array ~size:(const r) (int_range 0 3) in
    M.Broadcast
      (Array.append extra
         (Array.mapi (fun i d -> if d = 1 then fill.(i) else d) s))
  in
  let permute =
    let+ p = permutation ~pp:Format.pp_print_int (List.init r Fun.id) in
    M.Permute (Array.of_list p)
  in
  let slice =
    let rec all i =
      if i = r then constant []
      else
        let+ x = range s.(i) and+ xs = all (i + 1) in
        x :: xs
    in
    let+ rs = all 0 in
    M.Slice (Array.of_list rs)
  in
  let views = [ permute; slice; windows ~apart s ] in
  if apart then one_of views else one_of (reshape_target s :: broadcast :: views)

(* A layout [contiguous] reaches by up to three movements, views that keep
   positions apart if [apart]. *)
let reached ~apart =
  let open Gen in
  let* s = shape in
  let rec go l k =
    if k = 0 then constant l
    else
      let* m = movement ~apart (L.shape l) in
      match L.move m l with Some l' -> go l' (k - 1) | None -> go l (k - 1)
  in
  let* k = int_range 0 3 in
  go (L.contiguous s) k

let any_layout = Gen.with_pp L.pp (Gen.one_of [ strided; reached ~apart:false ])

let layout_and_move =
  let open Gen in
  with_pp
    (fun ppf (l, m) -> Format.fprintf ppf "%a, %a" L.pp l pp_move m)
    (let* l = one_of [ strided; reached ~apart:false ] in
     let+ m = movement ~apart:false (L.shape l) in
     (l, m))
