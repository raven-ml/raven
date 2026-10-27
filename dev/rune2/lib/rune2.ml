(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module Ptree = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Structures as values *)

let values s x =
  Array.of_list (List.map (fun (Nx.P y) -> V y) (fst (Ptree.flatten s x)))

let rebuild s ~like vs =
  Ptree.rebuild s ~like (List.map packed (Array.to_list vs))

(* Outputs [i] to [j - 1]. *)
let slice a i j = Array.sub a i (j - i)

(* Compilation *)

(* A signature as one structure of its arguments, numbered from [0]. *)
type ('s, 'd) walker = {
  walk : 'a 'b. ('a, 'b, 'd, 'd) Ptree.Walk.cursor -> 's -> 's;
}

type ('f, 'd) spine =
  | Spine : {
      walk : ('a, 'd) walker;
      result : ('r, 'd) Ptree.t;
      apply : 'f -> 'a -> 'r;
      curry : ('a -> 'r) -> 'f;
    }
      -> ('f, 'd) spine

let rec spine : type f d. string -> int -> (f, d) Ptree.fn -> (f, d) spine =
 fun name k -> function
  | Arg (a, Returns result) ->
      let walk c x = Ptree.Walk.index c k (Ptree.Walk.structure a) x in
      Spine
        { walk = { walk }; result; apply = (fun f x -> f x); curry = Fun.id }
  | Arg (a, rest) ->
      let (Spine u) = spine name (k + 1) rest in
      let walk c (x, xs) =
        let x = Ptree.Walk.index c k (Ptree.Walk.structure a) x in
        let xs = u.walk.walk c xs in
        (x, xs)
      in
      Spine
        {
          walk = { walk };
          result = u.result;
          apply = (fun f (x, xs) -> u.apply (f x) xs);
          curry = (fun g x -> u.curry (fun xs -> g (x, xs)));
        }
  | Returns _ -> invalid_argf "%s: the signature has no argument" name

let call name a r f args =
  let args_v = values a args in
  let like = ref None in
  let body, _ =
    Trace.trace Jit ~name (Array.map form args_v) (fun vs ->
        let y = f (rebuild a ~like:args vs) in
        like := Some y;
        values r y)
  in
  let y = Option.get !like in
  let out = run (Call { body; args = args_v }) in
  rebuild r ~like:y (slice out 0 (Array.length (values r y)))

let jit_named name s =
  let (Spine u) = spine name 0 s in
  let a = Ptree.v u.walk.walk in
  fun f -> u.curry (fun args -> call name a u.result (u.apply f) args)

let jit s = jit_named "Rune.jit" s

(* Functions of one array *)

let jit' f = jit_named "Rune.jit'" Ptree.(tensor @-> returns tensor) f
