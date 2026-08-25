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

module type Uniform = sig
  type 'a t

  val map : ('a -> 'b) -> 'a t -> 'b t
  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  val iter : ('a -> unit) -> 'a t -> unit
  val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc

  val fold2 :
    (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc

  val names : 'a t -> string t
end

module Make (U : Uniform) = struct
  type t = tensor U.t

  let map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) (t : t) : t
      =
    U.map (fun (P x) -> P (f x)) t

  let map2
      (f :
        'a 'b.
        ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
      (a : t) (b : t) : t =
    U.map2
      (fun (P x) (P y) ->
        match
          Nx_core.Dtype.equal_witness (Nx_effect.dtype x) (Nx_effect.dtype y)
        with
        | Some Type.Equal -> P (f x y)
        | None -> invalid_arg "Ptree.Make.map2: leaf dtype mismatch")
      a b

  let iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) (t : t) : unit =
    U.iter (fun (P x) -> f x) t
end

let typed (type a b) (module U : Uniform) :
    (module S with type t = (a, b) Nx_effect.t U.t) =
  (module struct
    type t = (a, b) Nx_effect.t U.t

    let map (f : 'p 'q. ('p, 'q) Nx_effect.t -> ('p, 'q) Nx_effect.t) (t : t) :
        t =
      U.map (fun x -> f x) t

    let map2
        (f :
          'p 'q.
          ('p, 'q) Nx_effect.t -> ('p, 'q) Nx_effect.t -> ('p, 'q) Nx_effect.t)
        (a : t) (b : t) : t =
      U.map2 (fun x y -> f x y) a b

    let iter (f : 'p 'q. ('p, 'q) Nx_effect.t -> unit) (t : t) : unit =
      U.iter (fun x -> f x) t
  end)

let unpack ?(at = "") (type a b) (dt : (a, b) Nx_core.Dtype.t) (p : tensor) :
    (a, b) Nx_effect.t =
  match p with
  | P x -> (
      match Nx_core.Dtype.equal_witness (Nx_effect.dtype x) dt with
      | Some Type.Equal -> x
      | None ->
          let msg =
            if at = "" then "Ptree.unpack: dtype mismatch"
            else "Ptree.unpack: dtype mismatch at " ^ at
          in
          invalid_arg msg)

module Tree = struct
  type 'a t = Leaf of 'a | List of 'a t list | Dict of (string * 'a t) list

  let rec map f = function
    | Leaf x -> Leaf (f x)
    | List ts -> List (List.map (map f) ts)
    | Dict kvs -> Dict (List.map (fun (k, v) -> (k, map f v)) kvs)

  let rec map2 f a b =
    match (a, b) with
    | Leaf x, Leaf y -> Leaf (f x y)
    | List xs, List ys ->
        if List.length xs <> List.length ys then
          invalid_arg "Ptree.Tree.map2: list length mismatch"
        else List (List.map2 (map2 f) xs ys)
    | Dict xs, Dict ys ->
        if List.length xs <> List.length ys then
          invalid_arg "Ptree.Tree.map2: dict size mismatch"
        else
          Dict
            (List.map2
               (fun (k1, v1) (k2, v2) ->
                 if not (String.equal k1 k2) then
                   invalid_arg "Ptree.Tree.map2: dict key mismatch"
                 else (k1, map2 f v1 v2))
               xs ys)
    | _ -> invalid_arg "Ptree.Tree.map2: structure mismatch"

  let rec iter f = function
    | Leaf x -> f x
    | List ts -> List.iter (iter f) ts
    | Dict kvs -> List.iter (fun (_, v) -> iter f v) kvs

  let join prefix seg = if prefix = "" then seg else prefix ^ "." ^ seg

  let fold f acc t =
    let rec go prefix acc = function
      | Leaf x -> f prefix acc x
      | List ts ->
          snd
            (List.fold_left
               (fun (i, acc) v ->
                 (i + 1, go (join prefix (string_of_int i)) acc v))
               (0, acc) ts)
      | Dict kvs ->
          List.fold_left (fun acc (k, v) -> go (join prefix k) acc v) acc kvs
    in
    go "" acc t

  let fold2 f acc a b =
    let rec go prefix acc a b =
      match (a, b) with
      | Leaf x, Leaf y -> f prefix acc x y
      | List xs, List ys ->
          if List.length xs <> List.length ys then
            invalid_arg "Ptree.Tree.fold2: list length mismatch"
          else
            snd
              (List.fold_left2
                 (fun (i, acc) x y ->
                   (i + 1, go (join prefix (string_of_int i)) acc x y))
                 (0, acc) xs ys)
      | Dict xs, Dict ys ->
          if List.length xs <> List.length ys then
            invalid_arg "Ptree.Tree.fold2: dict size mismatch"
          else
            List.fold_left2
              (fun acc (k1, v1) (k2, v2) ->
                if not (String.equal k1 k2) then
                  invalid_arg "Ptree.Tree.fold2: dict key mismatch"
                else go (join prefix k1) acc v1 v2)
              acc xs ys
      | _ -> invalid_arg "Ptree.Tree.fold2: structure mismatch"
    in
    go "" acc a b

  let names t =
    let rec go prefix = function
      | Leaf _ -> Leaf prefix
      | List ts ->
          List (List.mapi (fun i v -> go (join prefix (string_of_int i)) v) ts)
      | Dict kvs ->
          Dict (List.map (fun (k, v) -> (k, go (join prefix k) v)) kvs)
    in
    go "" t
end

type t = tensor Tree.t

let tensor x = Tree.Leaf (P x)
let list ts = Tree.List ts
let dict kvs = Tree.Dict kvs

include (Make (Tree) : S with type t := t)
