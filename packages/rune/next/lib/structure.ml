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
