(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Paths *)

module Path = struct
  type seg = Field of string | Index of int

  (* The steps from the part back to the root, so that [add] shares. *)
  type t = seg list

  let root = []
  let v segs = List.rev segs
  let add seg p = seg :: p
  let segments p = List.rev p

  let seg_equal a b =
    match (a, b) with
    | Field a, Field b -> String.equal a b
    | Index a, Index b -> Int.equal a b
    | Field _, Index _ | Index _, Field _ -> false

  let equal p q = List.equal seg_equal p q
  let seg_string = function Field s -> s | Index i -> string_of_int i
  let to_string p = String.concat "." (List.rev_map seg_string p)
  let pp ppf p = Format.pp_print_string ppf (to_string p)

  (* [p] as its steps, as [[a; b]], for paths that print alike. *)
  let pp_steps ppf p =
    Format.fprintf ppf "[%s]" (String.concat "; " (List.rev_map seg_string p))
end

type report = Int of int | Case of string | Present of bool | Length of int
type visit = Leaf of Path.t | Report of Path.t * report

type ('d, 'e) visitor = {
  tensor : 'v 's. Path.t -> ('v, 's, 'd) Value.t -> ('v, 's, 'e) Value.t;
  report : Path.t -> report -> unit;
}

(* A description is its walk: [w v p x] walks [x] at [p] with [v]. *)
type ('s, 'd) t = ('d, 'd) visitor -> Path.t -> 's -> 's

let keep = { tensor = (fun _ x -> x); report = (fun _ _ -> ()) }

(* Walks *)

module Walk = struct
  type ('a, 'b, 'd, 'e) cursor = {
    path : Path.t;
    leaf : Path.t -> 'a -> 'b;
    visitor : ('d, 'e) visitor;
  }

  let at c seg = { c with path = Path.add seg c.path }
  let field c name walk x = walk (at c (Field name)) x
  let index c i walk x = walk (at c (Index i)) x
  let leaf c x = c.leaf c.path x
  let tensor c x = c.visitor.tensor c.path x

  let int c n =
    c.visitor.report c.path (Int n);
    n

  let case c tag = c.visitor.report c.path (Case tag)

  let option walk c = function
    | None ->
        c.visitor.report c.path (Present false);
        None
    | Some x ->
        c.visitor.report c.path (Present true);
        Some (walk c x)

  let list walk c l =
    c.visitor.report c.path (Length (List.length l));
    let rec go i = function
      | [] -> []
      | x :: rest ->
          let y = walk (at c (Index i)) x in
          y :: go (i + 1) rest
    in
    go 0 l

  (* [outer]'s positions are walked by [inner] at their paths with [c]'s
     visitor, as [nest] walks them. *)
  let under outer inner c x =
    let leaf path y = inner { c with path } y in
    outer { path = c.path; leaf; visitor = c.visitor } x

  let structure s c x = s c.visitor c.path x
end

module type S = sig
  type ('a, 'd) t

  val walk : ('a, 'b, 'd, 'e) Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
end

let walk (type a b d e) (module U : S) leaf visitor (x : (a, d) U.t) :
    (b, e) U.t =
  U.walk { Walk.path = Path.root; leaf = (fun _ a -> leaf a); visitor } x

(* Structures *)

let nest (type s d) (module U : S) (s : (s, d) t) : ((s, d) U.t, d) t =
 fun visitor path x -> U.walk { Walk.path; leaf = s visitor; visitor } x

let tensor v p x = v.tensor p x
let unit _ _ () = ()
let instantiate (module U : S) = nest (module U) tensor

let v (walk : 'a 'b. ('a, 'b, 'd, 'd) Walk.cursor -> 's -> 's) visitor path x =
  walk { Walk.path; leaf = (fun _ x -> x); visitor } x

(* Combinators are walks, so that each kind of part is walked in one place:
   [Walk]. *)
let pair a b =
  v (fun c (x, y) ->
      let x = Walk.index c 0 (Walk.structure a) x in
      let y = Walk.index c 1 (Walk.structure b) y in
      (x, y))

let option a = v (fun c x -> Walk.option (Walk.structure a) c x)
let list a = v (fun c l -> Walk.list (Walk.structure a) c l)

(* Visits and skeletons *)

let pp_report ppf = function
  | Int n -> Format.fprintf ppf "int %d" n
  | Case tag -> Format.fprintf ppf "case %S" tag
  | Present true -> Format.pp_print_string ppf "Some"
  | Present false -> Format.pp_print_string ppf "None"
  | Length n -> Format.fprintf ppf "length %d" n

let pp_what ppf = function
  | Leaf _ -> Format.pp_print_string ppf "a leaf"
  | Report (_, r) -> pp_report ppf r

let path_of = function Leaf p | Report (p, _) -> p

let pp_path ppf = function
  | [] -> Format.pp_print_string ppf "the root"
  | p -> Path.pp ppf p

let pp_visit ppf v = Format.fprintf ppf "%a: %a" pp_path (path_of v) pp_what v

let visit_equal a b =
  match (a, b) with
  | Leaf p, Leaf q -> Path.equal p q
  | Report (p, r), Report (q, r') -> Path.equal p q && r = r'
  | Leaf _, Report _ | Report _, Leaf _ -> false

(* The first visit at which [a] and [b] differ, as [p: x this, y that]: a side
   with no more visits holds [nothing], and a visit at another path names it.
   Paths that print alike print as their steps. *)
let mismatch ~this a ~that b =
  let rec first i =
    if i = Array.length a && i = Array.length b then None
    else if i < Array.length a && i < Array.length b && visit_equal a.(i) b.(i)
    then first (i + 1)
    else Some i
  in
  let at v i = if i < Array.length v then Some v.(i) else None in
  let message x y =
    let px = Option.map path_of x and py = Option.map path_of y in
    let alike =
      match (px, py) with
      | Some p, Some q ->
          (not (Path.equal p q)) && Path.to_string p = Path.to_string q
      | _ -> false
    in
    let pp_p ppf p = if alike then Path.pp_steps ppf p else pp_path ppf p in
    let pp_side ppf = function
      | None -> Format.pp_print_string ppf "nothing"
      | Some v -> pp_what ppf v
    in
    let pp_elsewhere ppf () =
      match (px, py) with
      | Some p, Some q when not (Path.equal p q) ->
          Format.fprintf ppf " at %a" pp_p q
      | _ -> ()
    in
    Format.asprintf "%a: %a %s, %a%a %s" pp_p
      (Option.get (if Option.is_some px then px else py))
      pp_side x this pp_side y pp_elsewhere () that
  in
  Option.map (fun i -> message (at a i) (at b i)) (first 0)

module Skeleton = struct
  type t = visit array

  let equal a b =
    Array.length a = Array.length b && Array.for_all2 visit_equal a b

  let hash_seg = function Path.Field s -> Hashtbl.hash s | Index i -> i

  let hash_visit = function
    | Leaf p -> List.fold_left (fun h s -> (h * 31) + hash_seg s) 1 p
    | Report (p, r) ->
        List.fold_left (fun h s -> (h * 31) + hash_seg s) (Hashtbl.hash r) p

  let hash k = Array.fold_left (fun h v -> (h * 65599) + hash_visit v) 0 k
  let diff ~this a ~that b = mismatch ~this a ~that b
end

(* Derived *)

type 'd packed = P : ('v, 's, 'd) Value.t -> 'd packed

let recorder visits =
  {
    tensor =
      (fun p x ->
        visits := Leaf p :: !visits;
        x);
    report = (fun p r -> visits := Report (p, r) :: !visits);
  }

let visits s x =
  let vs = ref [] in
  ignore (s (recorder vs) Path.root x);
  List.rev !vs

let flatten s x =
  let leaves = ref [] and vs = ref [] in
  let v =
    {
      tensor =
        (fun p x ->
          leaves := P x :: !leaves;
          vs := Leaf p :: !vs;
          x);
      report = (fun p r -> vs := Report (p, r) :: !vs);
    }
  in
  ignore (s v Path.root x);
  (List.rev !leaves, Array.of_list (List.rev !vs))

let skeleton s x = Array.of_list (visits s x)

let map s (f : 'v 'r. Path.t -> ('v, 'r, 'd) Value.t -> ('v, 'r, 'd) Value.t) x
    =
  s { keep with tensor = (fun p x -> f p x) } Path.root x

let fold s (f : 'v 'r. Path.t -> ('v, 'r, 'd) Value.t -> 'acc -> 'acc) x acc =
  let acc = ref acc in
  let tensor p x =
    acc := f p x !acc;
    x
  in
  ignore (s { keep with tensor } Path.root x);
  !acc

let twice ~by = invalid_argf "%s: the structure walks one value two ways" by

let differ ~by a b =
  match mismatch ~this:"in the first value" a ~that:"in the second" b with
  | Some m -> invalid_argf "%s: %s" by m
  | None -> ()

let zip (type d) ~by (s : (_, d) t)
    (f :
      'v 'r.
      Path.t ->
      ('v, 'r, d) Value.t ->
      ('v, 'r, d) Value.t ->
      ('v, 'r, d) Value.t) x y =
  let ys, ky = flatten s y in
  differ ~by (skeleton s x) ky;
  let ys = ref ys in
  let tensor : type v r. Path.t -> (v, r, d) Value.t -> (v, r, d) Value.t =
   fun p a ->
    match !ys with
    | [] -> twice ~by
    | P b :: rest -> (
        ys := rest;
        match D.equal_witness (Prim.dtype a) (Prim.dtype b) with
        | Some Equal -> f p a b
        | None ->
            invalid_argf "%s: %a: %s in the first value, %s in the second" by
              pp_path p
              (D.name (Prim.dtype a))
              (D.name (Prim.dtype b)))
  in
  let z = s { keep with tensor } Path.root x in
  if not (List.is_empty !ys) then twice ~by;
  z

let map2 s
    (f :
      'v 'r.
      Path.t ->
      ('v, 'r, 'd) Value.t ->
      ('v, 'r, 'd) Value.t ->
      ('v, 'r, 'd) Value.t) x y =
  zip ~by:"Nx.Ptree.map2" s f x y

let rebuild (type d) (s : (_, d) t) ~like xs =
  let by = "Nx.Ptree.rebuild" in
  let given = List.length xs in
  let rest = ref xs and used = ref 0 in
  let tensor : type v r. Path.t -> (v, r, d) Value.t -> (v, r, d) Value.t =
   fun p a ->
    match !rest with
    | [] -> invalid_argf "%s: %d arrays, none for %a" by given pp_path p
    | P b :: tl -> (
        rest := tl;
        incr used;
        match D.equal_witness (Prim.dtype a) (Prim.dtype b) with
        | Some Equal -> b
        | None ->
            invalid_argf "%s: %a: %s in the value, %s given" by pp_path p
              (D.name (Prim.dtype a))
              (D.name (Prim.dtype b)))
  in
  let x = s { keep with tensor } Path.root like in
  if not (List.is_empty !rest) then
    invalid_argf "%s: %d arrays for a value of %d" by given !used;
  x

(* Payloads *)

module Payload = struct
  let map (type a b d) (module U : S) (f : Path.t -> a -> b) (x : (a, d) U.t) :
      (b, d) U.t =
    U.walk { Walk.path = Path.root; leaf = f; visitor = keep } x

  let fold (type a d acc) (module U : S) (f : Path.t -> a -> acc -> acc)
      (x : (a, d) U.t) (acc : acc) =
    let acc = ref acc in
    let leaf p a =
      acc := f p a !acc;
      a
    in
    ignore (U.walk { Walk.path = Path.root; leaf; visitor = keep } x);
    !acc

  (* A value's payloads and its visits without its fixed arrays. *)
  let record (type a d) (module U : S) (x : (a, d) U.t) =
    let payloads = ref [] and vs = ref [] in
    let leaf p a =
      payloads := a :: !payloads;
      vs := Leaf p :: !vs;
      a
    in
    let visitor =
      { keep with report = (fun p r -> vs := Report (p, r) :: !vs) }
    in
    ignore (U.walk { Walk.path = Path.root; leaf; visitor } x);
    (List.rev !payloads, Array.of_list (List.rev !vs))

  let map2 (type a b c d) (module U : S) (f : Path.t -> a -> b -> c)
      (x : (a, d) U.t) (y : (b, d) U.t) : (c, d) U.t =
    let by = "Nx.Ptree.Payload.map2" in
    let ys, ky = record (module U) y in
    differ ~by (snd (record (module U) x)) ky;
    let ys = ref ys in
    let leaf p a =
      match !ys with
      | [] -> twice ~by
      | b :: rest ->
          ys := rest;
          f p a b
    in
    let z = U.walk { Walk.path = Path.root; leaf; visitor = keep } x in
    if not (List.is_empty !ys) then twice ~by;
    z
end

(* Signatures *)

type (_, 'd) fn =
  | Returns : ('a, 'd) t -> ('a, 'd) fn
  | Arg : ('a, 'd) t * ('b, 'd) fn -> ('a -> 'b, 'd) fn

let ( @-> ) a f = Arg (a, f)
let returns a = Returns a
