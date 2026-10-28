(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

let float_leaf x = Nx.Dtype.is Float (Nx.dtype x)

let element v =
  Format.asprintf "%a" (Nx.Dtype.pp_value (Nx.dtype v)) (Nx.item [] v)

(* Arithmetic over chains

   Positions lead with the chain axis. A row operation acts on each chain: [dot]
   is each chain's inner product over every float tensor, and a per-chain scalar
   broadcasts against a tensor's trailing axes. *)

let column (type f) (h : (float, f, 'd) Nx.t) x =
  let c = (Nx.shape h).(0) in
  Nx.reshape
    (Array.append [| c |] (Array.make (Nx.ndim x - 1) 1))
    (Nx.cast (Nx.dtype x) h)

let dot (type f) u (like : (float, f, 'd) Nx.t) a b : (float, f, 'd) Nx.t =
  let c = (Nx.shape like).(0) in
  P.fold u
    (fun _ t acc ->
      if not (float_leaf t) then acc
      else
        Nx.add acc
          (Nx.cast (Nx.dtype like)
             (Nx.sum ~axes:[ 1 ] (Nx.reshape [| c; -1 |] t))))
    (P.map2 u (fun _ x y -> Nx.mul x y) a b)
    (Nx.zeros_like like)

let axpy u h x y =
  P.map2 u
    (fun _ x y -> if float_leaf x then Nx.add y (Nx.mul (column h x) x) else y)
    x y

let choose u mask a b =
  let c = (Nx.shape mask).(0) in
  P.map2 u
    (fun _ x y ->
      let m =
        Nx.reshape (Array.append [| c |] (Array.make (Nx.ndim x - 1) 1)) mask
      in
      Nx.where m x y)
    a b

(* [beside context ~what u x] is the placement whole on the devices of [x]'s
   first tensor. *)
let beside context ~what u x =
  let at =
    P.fold u
      (fun _ t at -> match at with None -> Some (Nx.beside t) | at -> at)
      x None
  in
  match at with
  | Some at -> at
  | None -> invalid_argf "%s: %s holds no tensor" context what

let whole_chains context ~reads u x =
  P.fold u
    (fun p t () ->
      if not (Nx.Placement.equal (Nx.placement t) (Nx.beside t)) then
        invalid_argf
          "%s: %s is cut across devices; %s, so each device must hold \
           whole chains: Nx.place (Nx.beside x) x"
          context
          (match P.Path.to_string p with "" -> "the position" | s -> s)
          reads)
    x ()

let finite u x =
  let c = P.fold u (fun _ t _ -> (Nx.shape t).(0)) x 0 in
  P.fold u
    (fun _ t acc ->
      if not (float_leaf t) then acc
      else
        Nx.logical_and acc
          (Nx.all ~axes:[ 1 ] (Nx.reshape [| c; -1 |] (Nx.isfinite t))))
    x
    (Nx.ones_at (beside "Norn" ~what:"the position" u x) Nx.bool [| c |])

let add u a b = P.map2 u (fun _ x y -> Nx.add x y) a b
let zeros u x = P.map u (fun _ t -> Nx.zeros_like t) x

(* [elements u c x] is the number of float elements of one chain of [x]. *)
let elements u c x =
  P.fold u (fun _ t n -> if float_leaf t then n + (Nx.numel t / c) else n) x 0

(* Flat rows

   [ravel u like x] is the float elements of each chain of [x], in walk order,
   as the rows of a [[c; d]] matrix at [like]'s dtype; [unravel u x m] is its
   inverse, with the tensors of [x] that are not floats kept. Arithmetic on one
   matrix costs a kernel where arithmetic on a structure costs one per
   tensor. *)

let ravel (type f) u (like : (float, f, 'd) Nx.t) x : (float, f, 'd) Nx.t =
  let c = (Nx.shape like).(0) in
  let rows =
    P.fold u
      (fun _ t acc ->
        if float_leaf t then
          Nx.cast (Nx.dtype like) (Nx.reshape [| c; -1 |] t) :: acc
        else acc)
      x []
  in
  match List.rev rows with
  | [] -> Nx.zeros_at (Nx.beside like) (Nx.dtype like) [| c; 0 |]
  | [ r ] -> r
  | rows -> Nx.concatenate ~axis:1 rows

let unravel u x m =
  let c = (Nx.shape m).(0) in
  let at = ref 0 in
  P.map u
    (fun _ t ->
      if not (float_leaf t) then t
      else
        let n = Nx.numel t / c in
        let r = Nx.slice [ Nx.A; Nx.R (!at, !at + n) ] m in
        at := !at + n;
        Nx.cast (Nx.dtype t) (Nx.reshape (Nx.shape t) r))
    x

(* Densities *)

let count context u x =
  let c =
    P.fold u
      (fun _ t c -> match c with None -> Some (Nx.shape t).(0) | c -> c)
      x None
  in
  match c with
  | Some c -> c
  | None -> invalid_argf "%s: the position has no tensor" context

(* [evaluate u lp x] is the density at [x] and its gradient, chain by chain: the
   gradient of the summed density is each chain's, rows being independent. *)
let evaluate u lp x =
  let _, g, l =
    Rune.value_and_grad_aux u P.tensor
      (fun x ->
        let l = lp x in
        (Nx.sum l, l))
      x
  in
  (l, g)

(* The density's value is finite or -inf at a finite position. *)
let check_range context l =
  let c = (Nx.shape l).(0) in
  let chain = Nx.arange_at (Nx.beside l) Nx.int32 0 c 1 in
  let ok =
    Nx.logical_not
      (Nx.logical_or (Nx.isnan l)
         (Nx.equal l (Nx.scalar_like l Float.infinity)))
  in
  Nx.check
    P.(pair tensor tensor)
    ok (l, chain)
    (fun _ (v, chain) ->
      Invalid_argument
        (Printf.sprintf "%s: the density is %s at chain %ld, a finite position"
           context (element v) (Nx.item [] chain)))

(* [check_rows context u lp x l] refuses a density whose rows read each other:
   evaluated on the chains reversed, its result is not reversed. *)
let check_rows context u lp x l =
  let c = (Nx.shape l).(0) in
  if c > 1 then begin
    (* The flips read the chain axis whole, on every device chains lie on. *)
    let whole t = Nx.place (Nx.beside t) t in
    let flip = P.map u (fun _ t -> Nx.flip ~axes:[ 0 ] (whole t)) x in
    let r = Nx.flip ~axes:[ 0 ] (whole (lp flip)) in
    let same =
      Nx.logical_or (Nx.equal l r)
        (Nx.less_equal
           (Nx.abs (Nx.sub l r))
           (Nx.mul_s (Nx.add_s (Nx.abs l) 1.) 1e-5))
    in
    Nx.check P.tensor same (Nx.arange_at (Nx.beside l) Nx.int32 0 c 1) (fun _ row ->
        Invalid_argument
          (Printf.sprintf
             "%s: the density's row %ld changes when the chains are reversed; \
              a chain's log density reads only its own row"
             context (Nx.item [] row)))
  end

(* [check_density context u lp l x] refuses a density [lp] whose value [l] at
   the position [x] of [c] chains is not one log density per chain, finite or
   [-inf], each read from its own row. *)
let check_density context u lp x l =
  let c = count context u x in
  if Nx.shape l <> [| c |] then
    invalid_argf
      "%s: the density returned shape [%s] for a position of %d chains; a \
       density returns one log density per chain, shape [%d]"
      context
      (shape_string (Nx.shape l))
      c c;
  check_range context l;
  check_rows context u lp x l
