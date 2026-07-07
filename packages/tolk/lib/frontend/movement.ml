(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module T = Tensor

let prod = List.fold_left ( * ) 1
let take n l = List.filteri (fun idx _ -> idx < n) l
let drop n l = List.filteri (fun idx _ -> idx >= n) l
let sub_range lo hi l = List.filteri (fun idx _ -> idx >= lo && idx < hi) l

let reshape t dims =
  let cur = T.shape t in
  let n_infer = List.length (List.filter (fun s -> s = -1) dims) in
  if n_infer > 1 then
    invalid_arg "Movement.reshape: only one dimension can be inferred";
  let dims =
    if n_infer = 1 then (
      let known =
        List.fold_left (fun a s -> if s = -1 then a else a * s) 1 dims
      in
      let total = prod cur in
      if known = 0 || total mod known <> 0 then
        invalid_arg "Movement.reshape: cannot infer -1 dimension";
      List.map (fun s -> if s = -1 then total / known else s) dims)
    else dims
  in
  if prod cur <> prod dims then invalid_arg "Movement.reshape: size mismatch";
  if dims = cur then t
  else T.of_uop (U.reshape ~src:(T.uop t) ~shape:(T.shape_uop dims))

let broadcast_to t new_shape =
  let cur = T.shape t in
  if cur = new_shape then t
  else begin
    if T.ndim t > List.length new_shape then
      invalid_arg "Movement.broadcast_to: cannot broadcast to fewer dimensions";
    let aligned =
      List.init (List.length new_shape - List.length cur) (fun _ -> 1) @ cur
    in
    List.iter2
      (fun s ns ->
        if not (s = ns || s = 1) then
          invalid_arg "Movement.broadcast_to: incompatible shapes")
      aligned new_shape;
    let reshaped = reshape t aligned in
    T.of_uop (U.expand ~src:(T.uop reshaped) ~shape:(T.shape_uop new_shape))
  end

let expand t dims =
  let cur = T.shape t in
  let aligned =
    List.init (max 0 (List.length dims - List.length cur)) (fun _ -> 1) @ cur
  in
  if List.length aligned <> List.length dims then
    invalid_arg "Movement.expand: too few target dimensions";
  let new_shape =
    List.map2 (fun from_ to_ -> if to_ = -1 then from_ else to_) aligned dims
  in
  broadcast_to t new_shape

let permute t order =
  let order = List.map (T.resolve_dim t) order in
  if List.sort compare order <> List.init (T.ndim t) Fun.id then
    invalid_arg "Movement.permute: not a valid permutation";
  T.of_uop (U.permute ~src:(T.uop t) ~order)

let flip t axes =
  let axes = List.map (T.resolve_dim t) axes in
  if List.length axes <> List.length (List.sort_uniq compare axes) then
    invalid_arg "Movement.flip: axis appears more than once";
  let dims = List.init (T.ndim t) (fun idx -> List.mem idx axes) in
  T.of_uop (U.flip ~src:(T.uop t) ~dims)

let pad t padding =
  if T.ndim t <> List.length padding then
    invalid_arg "Movement.pad: padding length must match ndim";
  let cur = T.shape t in
  let offset = List.map fst padding in
  let size = List.map2 (fun (before, after) s -> s + before + after) padding cur in
  T.of_uop
    (U.pad ~src:(T.uop t) ~offset:(T.shape_uop offset) ~size:(T.shape_uop size))

let shrink t bounds =
  if T.ndim t <> List.length bounds then
    invalid_arg "Movement.shrink: bounds length must match ndim";
  let offset = List.map fst bounds in
  let size = List.map (fun (start, stop) -> stop - start) bounds in
  T.of_uop
    (U.shrink ~src:(T.uop t) ~offset:(T.shape_uop offset) ~size:(T.shape_uop size))

let squeeze ?dim t =
  match dim with
  | None -> reshape t (List.filter (fun d -> d <> 1) (T.shape t))
  | Some dim ->
      let dim = T.resolve_dim t dim in
      let sh = T.shape t in
      if T.ndim t = 0 || List.nth sh dim <> 1 then t
      else reshape t (List.filteri (fun idx _ -> idx <> dim) sh)

let unsqueeze t dim =
  let dim = T.resolve_dim ~extra:true t dim in
  let sh = T.shape t in
  reshape t (take dim sh @ [ 1 ] @ drop dim sh)

let transpose ?(dim0 = 1) ?(dim1 = 0) t =
  let d0 = T.resolve_dim t dim0 and d1 = T.resolve_dim t dim1 in
  let order = Array.init (T.ndim t) Fun.id in
  let tmp = order.(d0) in
  order.(d0) <- order.(d1);
  order.(d1) <- tmp;
  permute t (Array.to_list order)

let flatten ?(start_dim = 0) ?(end_dim = -1) t =
  let s = T.resolve_dim t start_dim and e = T.resolve_dim t end_dim in
  let sh = T.shape t in
  reshape t (take s sh @ [ prod (sub_range s (e + 1) sh) ] @ drop (e + 1) sh)

let unflatten t dim sizes =
  let dim = T.resolve_dim t dim in
  let sh = T.shape t in
  reshape t (take dim sh @ sizes @ drop (dim + 1) sh)

let reshape_opt t dims =
  let cur = T.shape t in
  reshape t
    (List.mapi
       (fun idx d -> match d with Some n -> n | None -> List.nth cur idx)
       dims)

let shrink_to t dims =
  let cur = T.shape t in
  shrink t
    (List.mapi
       (fun idx d -> match d with None -> (0, List.nth cur idx) | Some n -> (0, n))
       dims)

let pad_to t dims =
  let cur = T.shape t in
  pad t
    (List.map2 (fun s ns -> match ns with None -> (0, 0) | Some n -> (0, n - s)) cur dims)

let repeat t repeats =
  let cur = T.shape t in
  let base =
    List.init (max 0 (List.length repeats - List.length cur)) (fun _ -> 1) @ cur
  in
  let pairs = List.combine repeats base in
  let unsqueezed =
    List.concat_map (fun (r, s) -> if r = 1 then [ s ] else [ 1; s ]) pairs
  in
  let expanded =
    List.concat_map (fun (r, s) -> if r = 1 then [ s ] else [ r; s ]) pairs
  in
  let final = List.map (fun (r, s) -> r * s) pairs in
  reshape (expand (reshape t unsqueezed) expanded) final

let ceildiv a b = (a + b - 1) / b

let pool t ~k ?stride ?dilation () =
  let n = List.length k in
  let expand_arg = function
    | None -> List.init n (fun _ -> 1)
    | Some l ->
        if List.length l = 1 && n > 1 then List.init n (fun _ -> List.hd l)
        else l
  in
  let stride = expand_arg stride and dilation = expand_arg dilation in
  if List.length stride <> n || List.length dilation <> n then
    invalid_arg "Movement.pool: stride/dilation length must match kernel";
  let ndim = T.ndim t in
  if ndim < n then invalid_arg "Movement.pool: input rank smaller than kernel";
  let noop_len = ndim - n in
  let noop = List.init noop_len (fun _ -> None) in
  let ka = Array.of_list k
  and sa = Array.of_list stride
  and da = Array.of_list dilation in
  let ia = Array.of_list (drop noop_len (T.shape t)) in
  Array.iteri
    (fun j kj ->
      if (da.(j) * (kj - 1)) + 1 > ia.(j) then
        invalid_arg "Movement.pool: kernel size exceeds input size")
    ka;
  let o = Array.init n (fun j -> ceildiv (ia.(j) - (da.(j) * (ka.(j) - 1))) sa.(j)) in
  let fa = Array.init n (fun j -> max 1 (ceildiv ((o.(j) * sa.(j)) - da.(j)) ia.(j))) in
  let span j = (ia.(j) * fa.(j)) + da.(j) in
  let flat f = noop @ List.concat (List.init n f) in
  let x =
    repeat t
      (List.init noop_len (fun _ -> 1)
      @ List.init n (fun j -> ceildiv (ka.(j) * span j) ia.(j)))
  in
  let x = shrink_to x (noop @ List.init n (fun j -> Some (ka.(j) * span j))) in
  let x = reshape_opt x (flat (fun j -> [ Some ka.(j); Some (span j) ])) in
  let x =
    reshape_opt
      (shrink_to x (flat (fun j -> [ Some ka.(j); Some (o.(j) * sa.(j)) ])))
      (flat (fun j -> [ Some ka.(j); Some o.(j); Some sa.(j) ]))
  in
  let x =
    reshape_opt
      (shrink_to x (flat (fun j -> [ Some ka.(j); Some o.(j); Some 1 ])))
      (flat (fun j -> [ Some ka.(j); Some o.(j) ]))
  in
  permute x
    (List.init noop_len Fun.id
    @ List.init n (fun j -> noop_len + (j * 2) + 1)
    @ List.init n (fun j -> noop_len + (j * 2)))

let argsort l =
  List.map snd (List.stable_sort compare (List.mapi (fun idx x -> (x, idx)) l))

let unfold t dim ~size ~step =
  if size < 0 then invalid_arg "Movement.unfold: size must be non-negative";
  if step <= 0 then invalid_arg "Movement.unfold: step must be positive";
  let dim = T.resolve_dim t dim in
  if size > List.nth (T.shape t) dim then
    invalid_arg "Movement.unfold: size exceeds the unfolded axis";
  let nd = T.ndim t in
  let perm_to_last = List.filter (fun i -> i <> dim) (List.init nd Fun.id) @ [ dim ] in
  let pooled = pool (permute t perm_to_last) ~k:[ size ] ~stride:[ step ] () in
  permute pooled (argsort perm_to_last @ [ nd ])

let split ?(dim = 0) t size =
  if size <= 0 then invalid_arg "Movement.split: chunk size must be positive";
  let dim = T.resolve_dim t dim in
  let sh = T.shape t in
  let dim_sz = List.nth sh dim in
  let rec starts i acc = if i >= max 1 dim_sz then List.rev acc else starts (i + size) (i :: acc) in
  List.map
    (fun start ->
      let stop = min (start + size) dim_sz in
      shrink t (List.mapi (fun i s -> if i = dim then (start, stop) else (0, s)) sh))
    (starts 0 [])

(* Indexing *)

type index =
  | I of int
  | R of int option * int option * int option
  | All
  | New
  | Ellipsis
  | T of Tensor.t

type resolved = Newaxis | View | Advanced of Tensor.t

type parsed = {
  size : int;
  boundary : int * int;
  stride : int;
  collapse_dim : bool;
  resolved : resolved;
}

let round_up n amt = (n + amt - 1) / amt * amt

(* Python-style [slice.indices]: resolve possibly-omitted, possibly-negative
   bounds against a concrete [size], returning ascending sentinels the direction
   of [step] is read from. *)
let slice_indices size start stop step =
  let step = match step with None -> 1 | Some s -> s in
  if step = 0 then invalid_arg "Movement.getitem: slice step cannot be 0";
  let lo, hi = if step < 0 then (-1, size - 1) else (0, size) in
  let clamp = function
    | None -> assert false
    | Some s -> if s < 0 then max (s + size) lo else min s hi
  in
  let start = match start with None -> if step < 0 then hi else lo | s -> clamp s in
  let stop = match stop with None -> if step < 0 then lo else hi | s -> clamp s in
  (start, stop, step)

let parse_view_index index size =
  match index with
  | New -> { size = 1; boundary = (0, 1); stride = 1; collapse_dim = false; resolved = Newaxis }
  | I n ->
      if n >= size || n < -size then
        invalid_arg "Movement.getitem: index out of bounds";
      let b = if n >= 0 then n else n + size in
      { size; boundary = (b, b + 1); stride = 1; collapse_dim = true; resolved = View }
  | All | R _ ->
      let start, stop, step =
        match index with R (a, b, s) -> (a, b, s) | _ -> (None, None, None)
      in
      let s, e, st = slice_indices size start stop step in
      let lo, hi =
        if st * (e - s) < 0 then (0, 0)
        else if st < 0 then (e + 1, s + 1)
        else (s, e)
      in
      {
        size = ceildiv (hi - lo) (abs st);
        boundary = (lo, hi);
        stride = st;
        collapse_dim = false;
        resolved = View;
      }
  | Ellipsis | T _ -> invalid_arg "Movement.getitem: index must be resolved first"

let normalize_indices t indices =
  let is_ellipsis = function Ellipsis -> true | _ -> false in
  let is_new = function New -> true | _ -> false in
  if List.length (List.filter is_ellipsis indices) > 1 then
    invalid_arg "Movement.getitem: only one ellipsis is allowed";
  let num_real =
    List.length (List.filter (fun i -> not (is_ellipsis i || is_new i)) indices)
  in
  if num_real > T.ndim t then
    invalid_arg "Movement.getitem: too many indices for tensor rank";
  let fill = List.init (T.ndim t - num_real) (fun _ -> All) in
  let rec at i = function
    | [] -> indices @ fill
    | x :: rest -> if is_ellipsis x then take i indices @ fill @ rest else at (i + 1) rest
  in
  at 0 indices

let apply_view_ops t mops =
  let x = shrink t (List.map (fun m -> m.boundary) mops) in
  let flip_axes =
    List.filter_map Fun.id
      (List.mapi (fun i m -> if m.stride < 0 then Some i else None) mops)
  in
  let x = if flip_axes = [] then x else flip x flip_axes in
  let strides = List.map (fun m -> abs m.stride) mops in
  if List.for_all (fun st -> st = 1) strides then x
  else
    let padded = pad_to x (List.map2 (fun s st -> Some (round_up s st)) (T.shape x) strides) in
    let split =
      reshape padded
        (List.concat (List.map2 (fun s st -> [ s / st; st ]) (T.shape padded) strides))
    in
    let evens = List.filteri (fun i _ -> i mod 2 = 0) (T.shape split) in
    let kept = shrink_to split (List.concat_map (fun s -> [ Some s; Some 1 ]) evens) in
    reshape kept evens
