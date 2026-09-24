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

   [vmap] and [remat] transform a curried function of any number of arguments.
   Its signature turns it into a function of one value, the arguments nested in
   pairs (the last one alone), and back. *)

type 'f uncurried =
  | Uncurried : {
      args : 'a Ptree.t;
      result : 'r Ptree.t;
      arity : int;
      apply : 'f -> 'a -> 'r;
      curry : ('a -> 'r) -> 'f;
    }
      -> 'f uncurried

let rec uncurry_from : type f. string -> int -> f Ptree.fn -> f uncurried =
 fun fn i -> function
  | Ptree.Arg (Consumed, _, _) ->
      invalid_argf
        "%s: argument %d is consumed; only a compiled call consumes its \
         arguments"
        fn (i + 1)
  | Arg (Read, a, Returns r) ->
      Uncurried
        {
          args = a;
          result = r;
          arity = 1;
          apply = (fun f x -> f x);
          curry = (fun g -> g);
        }
  | Arg (Read, a, rest) ->
      let (Uncurried u) = uncurry_from fn (i + 1) rest in
      Uncurried
        {
          args = Ptree.pair a u.args;
          result = u.result;
          arity = u.arity + 1;
          apply = (fun f (x, xs) -> u.apply (f x) xs);
          curry = (fun g x -> u.curry (fun xs -> g (x, xs)));
        }
  | Returns _ -> invalid_arg (fn ^ ": the signature has no argument")

(* [uncurry fn s] is [s] as a function of one value. It raises, naming [fn], if
   [s] consumes an argument. *)
let uncurry fn (s : ('a -> 'b) Ptree.fn) = uncurry_from fn 0 s

(* [argument arity path] names the leaf at [path] of the nested arguments of a
   signature of [arity] arguments: the argument, from 1, and the leaf's path in
   it. *)
let argument arity path =
  let rec go k = function
    | segs when k = arity - 1 -> (k, segs)
    | Ptree.Path.Index 0 :: segs -> (k, segs)
    | Index 1 :: segs -> go (k + 1) segs
    | segs -> (k, segs)
  in
  let k, segs = go 0 (Ptree.Path.segments path) in
  let leaf =
    match segs with
    | [] -> ""
    | _ ->
        ", leaf "
        ^ String.concat "."
            (List.map
               (function
                 | Ptree.Path.Field name -> name | Index i -> Int.to_string i)
               segs)
  in
  Printf.sprintf "argument %d%s" (k + 1) leaf
