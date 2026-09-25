(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Structures as the transformations use them: fresh aliases of argument leaves,
   errors for two values that do not share a structure, and signatures uncurried
   into one argument. *)

module Ptree = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Aliases

   A transformation replaces each argument leaf by a new value over the same
   storage before it tracks, marks or seeds it, so a capture that is the same
   value as an argument stays a constant of the transformation. The alias is a
   reshape to the leaf's own shape: an operation that copies nothing and that
   enclosing transformations observe, so an outer [grad] or [vmap] sees the
   argument flow into it. *)

let alias x = Nx_effect.reshape x (Nx.shape x)
let aliases s v = Ptree.map s (fun _ x -> alias x) v

(* Mismatches *)

let describe path =
  match Ptree.Path.segments path with
  | [] -> "the root"
  | _ -> Ptree.Path.to_string path

let shape_string s =
  String.concat "," (Array.to_list (Array.map string_of_int s))

(* [check fn ~this k ~that k'] raises, naming [fn], if skeletons [k] and [k']
   differ. [this] and [that] name the two values. *)
let check fn ~this k ~that k' =
  match Ptree.Skeleton.diff ~this:("in " ^ this) k ~that:("in " ^ that) k' with
  | None -> ()
  | Some m -> invalid_arg (fn ^ ": " ^ m)

(* [map2 fn s ~this ~that f x y] is [x] with each tensor [t] at path [p]
   replaced by [f p t u], where [u] is [y]'s tensor at [p]. It raises, naming
   [fn], [this] for [x] and [that] for [y], before applying [f] when the two
   values differ in their skeletons, and at [p] when [t] and [u] differ in
   dtype. *)
let map2 fn s ~this ~that
    (f : 'a 'b. Ptree.Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
    x y =
  let _, kx = Ptree.flatten s x in
  let ys, ky = Ptree.flatten s y in
  check fn ~this kx ~that ky;
  let rest = ref ys in
  Ptree.map s
    (fun (type a b) path (t : (a, b) Nx.t) : (a, b) Nx.t ->
      match !rest with
      | [] ->
          invalid_arg (fn ^ ": the structure's walk visited one value two ways")
      | Nx.P u :: tail -> (
          rest := tail;
          match Nx_core.Dtype.equal_witness (Nx.dtype t) (Nx.dtype u) with
          | Some Type.Equal -> f path t u
          | None ->
              invalid_argf "%s: %s: %s in %s, %s in %s" fn (describe path)
                (Nx_core.Dtype.to_string (Nx.dtype t))
                this
                (Nx_core.Dtype.to_string (Nx.dtype u))
                that))
    x

(* Signatures

   A transformation of a curried function of any number of arguments sees it as
   a function of one value: the arguments nested in pairs, the last one alone.
   That value is walked as a sequence, argument [k] at [Index k], so every
   leaf's path starts with its argument's position from 0: the window of the
   second argument is at [1.window], and a first argument that is one tensor is
   at [0]. [roles] gives each argument's role, from the first. *)

type 's walker = { walk : 'a 'b. ('a, 'b) Ptree.Walk.cursor -> 's -> 's }

(* The structure that walks as [w] does. *)
let of_walker (type s) (w : s walker) : s Ptree.t =
  let module M = struct
    type _ t = s

    let walk c x = w.walk c x
  end in
  Ptree.instantiate (module M)

(* A list of tensors of any dtypes, each at its index. *)
let packed_list : Nx.packed list Ptree.t =
  of_walker
    {
      walk =
        (fun c l ->
          Ptree.Walk.list (fun c (Nx.P x) -> Nx.P (Ptree.Walk.tensor c x)) c l);
    }

type 'f spine =
  | Spine : {
      walk : 'a walker;
      result : 'r Ptree.t;
      arity : int;
      roles : Ptree.role list;
      apply : 'f -> 'a -> 'r;
      curry : ('a -> 'r) -> 'f;
    }
      -> 'f spine

let rec spine_from : type f. string -> int -> f Ptree.fn -> f spine =
 fun fn k -> function
  | Ptree.Arg (role, a, Returns r) ->
      Spine
        {
          walk =
            {
              walk =
                (fun c x -> Ptree.Walk.index c k (Ptree.Walk.structure a) x);
            };
          result = r;
          arity = 1;
          roles = [ role ];
          apply = (fun f x -> f x);
          curry = (fun g -> g);
        }
  | Arg (role, a, rest) ->
      let (Spine u) = spine_from fn (k + 1) rest in
      Spine
        {
          walk =
            {
              walk =
                (fun c (x, xs) ->
                  let x = Ptree.Walk.index c k (Ptree.Walk.structure a) x in
                  let xs = u.walk.walk c xs in
                  (x, xs));
            };
          result = u.result;
          arity = u.arity + 1;
          roles = role :: u.roles;
          apply = (fun f (x, xs) -> u.apply (f x) xs);
          curry = (fun g x -> u.curry (fun xs -> g (x, xs)));
        }
  | Returns _ -> invalid_arg (fn ^ ": the signature has no argument")

type 'f uncurried =
  | Uncurried : {
      args : 'a Ptree.t;
      result : 'r Ptree.t;
      arity : int;
      roles : Ptree.role array;
      apply : 'f -> 'a -> 'r;
      curry : ('a -> 'r) -> 'f;
    }
      -> 'f uncurried

(* [signature fn s] is [s] as a function of one value. It raises, naming [fn],
   if [s] has no argument. *)
let signature fn (s : ('a -> 'b) Ptree.fn) =
  let (Spine u) = spine_from fn 0 s in
  Uncurried
    {
      args = of_walker u.walk;
      result = u.result;
      arity = u.arity;
      roles = Array.of_list u.roles;
      apply = u.apply;
      curry = u.curry;
    }

(* [uncurry fn s] is [signature fn s], and raises, naming [fn], if [s] consumes
   an argument. *)
let uncurry fn s =
  let (Uncurried u as sg) = signature fn s in
  Array.iteri
    (fun i -> function
      | Ptree.Consumed ->
          invalid_argf
            "%s: the argument at %d is consumed; only a compiled call consumes \
             its arguments"
            fn i
      | Read -> ())
    u.roles;
  sg

(* [argument_of path] is the argument, from 0, that the leaf at [path] of a
   signature's arguments belongs to. *)
let argument_of path =
  match Ptree.Path.segments path with
  | Ptree.Path.Index k :: _ -> k
  | _ -> invalid_arg "Rune: a signature's leaf outside its arguments"
