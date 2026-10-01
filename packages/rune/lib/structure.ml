(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ptree = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let describe path =
  match Ptree.Path.segments path with
  | [] -> "the root"
  | _ -> Ptree.Path.to_string path

(* [partner fn s ~this x ~that y] checks that [x] and [y] have equal skeletons
   and is the function that returns [y]'s tensors in walk order, one per call,
   each checked against [x]'s tensor at its path for its dtype. *)
type partner = { next : 'a 'b. Ptree.Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let partner fn s ~this x ~that y =
  let _, kx = Ptree.flatten s x and ys, ky = Ptree.flatten s y in
  (match
     Ptree.Skeleton.diff ~this:("in " ^ this) kx ~that:("in " ^ that) ky
   with
  | None -> ()
  | Some m -> invalid_arg (fn ^ ": " ^ m));
  let rest = ref ys in
  let next (type a b) path (t : (a, b) Nx.t) : (a, b) Nx.t =
    match !rest with
    | [] ->
        invalid_arg (fn ^ ": the structure's walk visited one value two ways")
    | Nx.P u :: tail -> (
        rest := tail;
        match Nx_dtype.equal_witness (Nx.dtype t) (Nx.dtype u) with
        | Some Type.Equal -> u
        | None ->
            invalid_argf "%s: %s: %s in %s, %s in %s" fn (describe path)
              (Nx_dtype.to_string (Nx.dtype t))
              this
              (Nx_dtype.to_string (Nx.dtype u))
              that)
  in
  { next }

let check fn s ~this x ~that y =
  let { next } = partner fn s ~this x ~that y in
  Ptree.fold s (fun path t () -> ignore (next path t)) x ()

let map2 fn s ~this ~that
    (f : 'a 'b. Ptree.Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
    x y =
  let { next } = partner fn s ~this x ~that y in
  Ptree.map s
    (fun path t ->
      let u = next path t in
      if Nx.shape t <> Nx.shape u then
        invalid_arg
          (Format.asprintf "%s: %s: shape %a in %s, %a in %s" fn (describe path)
             Nx.pp_shape (Nx.shape t) this Nx.pp_shape (Nx.shape u) that);
      f path t u)
    x

(* Signatures *)

type 'f signature =
  | Signature : {
      args : 'a Ptree.t;
      result : 'r Ptree.t;
      roles : Ptree.role list;
      apply : 'f -> 'a -> 'r;
      curry : ('a -> 'r) -> 'f;
    }
      -> 'f signature

type 's walker = { walk : 'a 'b. ('a, 'b) Ptree.Walk.cursor -> 's -> 's }

let of_walker (type s) (w : s walker) : s Ptree.t =
  let module M = struct
    type _ t = s

    let walk c x = w.walk c x
  end in
  Ptree.instantiate (module M)

(* A signature from argument [k] on: the walk of its arguments from [k] on,
   nested in pairs, and their roles. *)
type 'f spine =
  | Spine : {
      walk : 'a walker;
      result : 'r Ptree.t;
      roles : Ptree.role list;
      apply : 'f -> 'a -> 'r;
      curry : ('a -> 'r) -> 'f;
    }
      -> 'f spine

let rec spine : type f. string -> int -> f Ptree.fn -> f spine =
 fun fn k -> function
  | Ptree.Arg (role, a, Returns result) ->
      let walk c x = Ptree.Walk.index c k (Ptree.Walk.structure a) x in
      Spine
        {
          walk = { walk };
          result;
          roles = [ role ];
          apply = (fun f x -> f x);
          curry = Fun.id;
        }
  | Arg (role, a, rest) ->
      let (Spine u) = spine fn (k + 1) rest in
      let walk c (x, xs) =
        let x = Ptree.Walk.index c k (Ptree.Walk.structure a) x in
        let xs = u.walk.walk c xs in
        (x, xs)
      in
      Spine
        {
          walk = { walk };
          result = u.result;
          roles = role :: u.roles;
          apply = (fun f (x, xs) -> u.apply (f x) xs);
          curry = (fun g x -> u.curry (fun xs -> g (x, xs)));
        }
  | Returns _ -> invalid_arg (fn ^ ": the signature has no argument")

let signature fn s =
  let (Spine u) = spine fn 0 s in
  Signature
    {
      args = of_walker u.walk;
      result = u.result;
      roles = u.roles;
      apply = u.apply;
      curry = u.curry;
    }

let uncurry fn s =
  let (Signature u as signature) = signature fn s in
  List.iteri
    (fun i -> function
      | Ptree.Consumed ->
          invalid_argf
            "%s: the argument at %d is consumed; only a compiled call consumes \
             its arguments"
            fn i
      | Read -> ())
    u.roles;
  signature
