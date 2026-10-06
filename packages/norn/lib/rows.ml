(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

let float_leaf x = Nx_dtype.is Float (Nx.dtype x)

(* Arithmetic over chains

   Positions lead with the chain axis. A row operation acts on each chain: [dot]
   is each chain's inner product over every float tensor, and a per-chain scalar
   broadcasts against a tensor's trailing axes. *)

let column (type f) (h : (float, f) Nx.t) x =
  let c = (Nx.shape h).(0) in
  Nx.reshape
    (Array.append [| c |] (Array.make (Nx.ndim x - 1) 1))
    (Nx.cast (Nx.dtype x) h)

let dot (type f) u (like : (float, f) Nx.t) a b : (float, f) Nx.t =
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

let finite u x =
  let c = P.fold u (fun _ t _ -> (Nx.shape t).(0)) x 0 in
  P.fold u
    (fun _ t acc ->
      if not (float_leaf t) then acc
      else
        Nx.logical_and acc
          (Nx.all ~axes:[ 1 ] (Nx.reshape [| c; -1 |] (Nx.isfinite t))))
    x (Nx.ones Nx.bool [| c |])

let add u a b = P.map2 u (fun _ x y -> Nx.add x y) a b
let zeros u x = P.map u (fun _ t -> Nx.zeros_like t) x

(* [elements u c x] is the number of float elements of one chain of [x]. *)
let elements u c x =
  P.fold u (fun _ t n -> if float_leaf t then n + (Nx.numel t / c) else n) x 0

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
  let chain = Nx.arange Nx.int32 0 c 1 in
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
           context (Nx.to_string v) (Nx.item [] chain)))

(* [check_rows context u lp x l] refuses a density whose rows read each other:
   evaluated on the chains reversed, its result is not reversed. *)
let check_rows context u lp x l =
  let c = (Nx.shape l).(0) in
  if c > 1 then begin
    let flip = P.map u (fun _ t -> Nx.flip ~axes:[ 0 ] t) x in
    let r = Nx.flip ~axes:[ 0 ] (lp flip) in
    let same =
      Nx.logical_or (Nx.equal l r)
        (Nx.less_equal
           (Nx.abs (Nx.sub l r))
           (Nx.mul_s (Nx.add_s (Nx.abs l) 1.) 1e-5))
    in
    Nx.check P.tensor same (Nx.arange Nx.int32 0 c 1) (fun _ row ->
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
