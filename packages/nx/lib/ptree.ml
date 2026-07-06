(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module type S = sig
  type t

  val map : ('a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) -> t -> t

  val map2 :
    ('a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) ->
    t ->
    t ->
    t

  val iter : ('a 'b. ('a, 'b) Nx_effect.t -> unit) -> t -> unit
end

type tensor = P : ('a, 'b) Nx_effect.t -> tensor
type t = Tensor of tensor | List of t list | Dict of (string * t) list

let tensor x = Tensor (P x)
let list ts = List ts
let dict kvs = Dict kvs

let rec map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) (t : t) :
    t =
  match t with
  | Tensor (P x) -> Tensor (P (f x))
  | List ts -> List (List.map (map f) ts)
  | Dict kvs -> Dict (List.map (fun (k, v) -> (k, map f v)) kvs)

let rec map2
    (f :
      'a 'b.
      ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
    (a : t) (b : t) : t =
  match (a, b) with
  | Tensor (P x), Tensor (P y) -> (
      match
        Nx_core.Dtype.equal_witness (Nx_effect.dtype x) (Nx_effect.dtype y)
      with
      | Some Type.Equal -> Tensor (P (f x y))
      | None -> invalid_arg "Ptree.map2: leaf dtype mismatch")
  | List xs, List ys ->
      if List.length xs <> List.length ys then
        invalid_arg "Ptree.map2: list length mismatch"
      else List (List.map2 (map2 f) xs ys)
  | Dict xs, Dict ys ->
      if List.length xs <> List.length ys then
        invalid_arg "Ptree.map2: dict size mismatch"
      else
        Dict
          (List.map2
             (fun (k1, v1) (k2, v2) ->
               if not (String.equal k1 k2) then
                 invalid_arg "Ptree.map2: dict key mismatch"
               else (k1, map2 f v1 v2))
             xs ys)
  | _ -> invalid_arg "Ptree.map2: structure mismatch"

let rec iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) (t : t) : unit =
  match t with
  | Tensor (P x) -> f x
  | List ts -> List.iter (iter f) ts
  | Dict kvs -> List.iter (fun (_, v) -> iter f v) kvs
