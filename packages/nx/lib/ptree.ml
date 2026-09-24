(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, 'b) tensor = ('a, 'b) Nx_effect.t

module Path = struct
  type seg = Field of string | Index of int
  type t = Root | Dot of t * string | At of t * int

  let segments p =
    let rec go acc = function
      | Root -> acc
      | Dot (p, name) -> go (Field name :: acc) p
      | At (p, i) -> go (Index i :: acc) p
    in
    go [] p

  let rec equal a b =
    a == b
    ||
    match (a, b) with
    | Dot (p, x), Dot (q, y) -> (x == y || String.equal x y) && equal p q
    | At (p, i), At (q, j) -> Int.equal i j && equal p q
    | _ -> false

  let seg_to_string = function Field name -> name | Index i -> Int.to_string i
  let to_string p = String.concat "." (List.map seg_to_string (segments p))
  let pp ppf p = Format.pp_print_string ppf (to_string p)
  let describe = function Root -> "the root" | p -> to_string p

  let describe_segments p =
    let seg = function
      | Field name -> Printf.sprintf "%S" name
      | Index i -> Int.to_string i
    in
    "[" ^ String.concat "; " (List.map seg (segments p)) ^ "]"
end

type report = Int of int | Case of string | Present of bool | Length of int

type ops = {
  tensor : 'a 'b. Path.t -> ('a, 'b) tensor -> ('a, 'b) tensor;
  report : Path.t -> report -> unit;
}

type ('a, 'b) env = { ops : ops; leaf : ops -> Path.t -> 'a -> 'b }
type ('a, 'b) cursor = { env : ('a, 'b) env; path : Path.t }

let present = Present true
let absent = Present false
let ignore_report _ _ = ()
let keep = { tensor = (fun _ x -> x); report = ignore_report }

type 's t = { walk : ops -> Path.t -> 's -> 's } [@@unboxed]

module Walk = struct
  type nonrec ('a, 'b) cursor = ('a, 'b) cursor

  let field c name f x = f { env = c.env; path = Path.Dot (c.path, name) } x
  let index c i f x = f { env = c.env; path = Path.At (c.path, i) } x
  let leaf c x = c.env.leaf c.env.ops c.path x
  let tensor c x = c.env.ops.tensor c.path x

  let int c n =
    c.env.ops.report c.path (Int n);
    n

  let case c tag = c.env.ops.report c.path (Case tag)

  let option f c = function
    | None ->
        c.env.ops.report c.path absent;
        None
    | Some x ->
        c.env.ops.report c.path present;
        Some (f c x)

  let list f c l =
    c.env.ops.report c.path (Length (List.length l));
    let rec go i = function
      | [] -> []
      | x :: rest ->
          let y = f { env = c.env; path = Path.At (c.path, i) } x in
          y :: go (i + 1) rest
    in
    go 0 l

  let structure s c x = s.walk c.env.ops c.path x
end

module type S = sig
  type 'a t

  val walk : ('a, 'b) Walk.cursor -> 'a t -> 'b t
end

let nest (module U : S) (s : 's t) : 's U.t t =
  { walk = (fun ops path x -> U.walk { env = { ops; leaf = s.walk }; path } x) }

let tensor = { walk = (fun ops path x -> ops.tensor path x) }
let instantiate (module U : S) : ('a, 'b) tensor U.t t = nest (module U) tensor
let unit = { walk = (fun _ _ () -> ()) }

let pair a b =
  {
    walk =
      (fun ops path (x, y) ->
        let x = a.walk ops (Path.At (path, 0)) x in
        let y = b.walk ops (Path.At (path, 1)) y in
        (x, y));
  }

let option a =
  {
    walk =
      (fun ops path -> function
        | None ->
            ops.report path absent;
            None
        | Some x ->
            ops.report path present;
            Some (a.walk ops path x));
  }

let list a =
  {
    walk =
      (fun ops path l ->
        ops.report path (Length (List.length l));
        let rec go i = function
          | [] -> []
          | x :: rest ->
              let y = a.walk ops (Path.At (path, i)) x in
              y :: go (i + 1) rest
        in
        go 0 l);
  }

let iso f g a = { walk = (fun ops path x -> f (a.walk ops path (g x))) }

(* Signatures *)

type role = Read | Consumed

type _ fn =
  | Returns : 'a t -> 'a fn
  | Arg : role * 'a t * 'b fn -> ('a -> 'b) fn

let ( @-> ) a f = Arg (Read, a, f)
let consumes a f = Arg (Consumed, a, f)
let returns a = Returns a

(* Visits and skeletons *)

type visit = Leaf of Path.t | Report of Path.t * report

let equal_report a b =
  match (a, b) with
  | Int m, Int n | Length m, Length n -> Int.equal m n
  | Case x, Case y -> String.equal x y
  | Present x, Present y -> Bool.equal x y
  | _ -> false

let equal_visit a b =
  match (a, b) with
  | Leaf p, Leaf q -> Path.equal p q
  | Report (p, x), Report (q, y) -> equal_report x y && Path.equal p q
  | _ -> false

let visit_path = function Leaf p | Report (p, _) -> p

let describe_visit = function
  | Leaf _ -> "a leaf"
  | Report (_, Int n) -> Printf.sprintf "int %d" n
  | Report (_, Case tag) -> Printf.sprintf "case %S" tag
  | Report (_, Present true) -> "Some"
  | Report (_, Present false) -> "None"
  | Report (_, Length n) -> Printf.sprintf "length %d" n

let pp_visit ppf v =
  Format.fprintf ppf "%s: %s" (Path.describe (visit_path v)) (describe_visit v)

module Skeleton = struct
  type t = { rev_visits : visit list; length : int; hash : int }

  let hash_visit = function
    | Leaf _ -> 1
    | Report (_, Int n) -> 3 + (7 * n)
    | Report (_, Length n) -> 4 + (7 * n)
    | Report (_, Present b) -> if b then 5 else 6
    | Report (_, Case tag) -> Hashtbl.hash tag

  let of_rev rev_visits =
    let rec go length hash = function
      | [] -> { rev_visits; length; hash = hash land max_int }
      | v :: rest -> go (length + 1) ((31 * hash) + hash_visit v) rest
    in
    go 0 0 rev_visits

  let hash k = k.hash
  let visits k = List.rev k.rev_visits

  let equal a b =
    a == b
    || a.hash = b.hash && a.length = b.length
       && List.for_all2 equal_visit a.rev_visits b.rev_visits

  let diff ~this a ~that b =
    let where p q =
      if String.equal (Path.to_string p) (Path.to_string q) then
        (Path.describe_segments p, Path.describe_segments q)
      else (Path.describe p, Path.describe q)
    in
    let rec go = function
      | [], [] -> None
      | x :: _, [] ->
          Some
            (Printf.sprintf "%s: %s %s, nothing %s"
               (Path.describe (visit_path x))
               (describe_visit x) this that)
      | [], y :: _ ->
          Some
            (Printf.sprintf "%s: nothing %s, %s %s"
               (Path.describe (visit_path y))
               this (describe_visit y) that)
      | x :: xs, y :: ys ->
          let px = visit_path x and py = visit_path y in
          if equal_visit x y then go (xs, ys)
          else if Path.equal px py then
            Some
              (Printf.sprintf "%s: %s %s, %s %s" (Path.describe px)
                 (describe_visit x) this (describe_visit y) that)
          else
            let px, py = where px py in
            Some
              (Printf.sprintf "%s: %s %s, %s at %s %s" px (describe_visit x)
                 this (describe_visit y) py that)
    in
    go (visits a, visits b)
end

let check_same fn ~this ~that a b =
  match Skeleton.diff ~this a ~that b with
  | None -> ()
  | Some msg -> invalid_arg (fn ^ ": " ^ msg)

let walked_two_ways fn =
  invalid_arg (fn ^ ": the structure's walk visited one value two ways")

(* Walks at the structure's one type *)

let flatten (type s) (s : s t) (x : s) : Nx_effect.packed list * Skeleton.t =
  let leaves = ref [] and visits = ref [] in
  let tensor : type a b. Path.t -> (a, b) tensor -> (a, b) tensor =
   fun path x ->
    leaves := Nx_effect.P x :: !leaves;
    visits := Leaf path :: !visits;
    x
  in
  let report path r = visits := Report (path, r) :: !visits in
  ignore (s.walk { tensor; report } Path.Root x);
  (List.rev !leaves, Skeleton.of_rev !visits)

let skeleton s x = snd (flatten s x)
let visits s x = Skeleton.visits (skeleton s x)

exception Mismatch

let rebuild (type s) (s : s t) ~(like : s) (leaves : Nx_effect.packed list) : s
    =
  let rest = ref leaves and taken = ref 0 in
  let tensor : type a b. Path.t -> (a, b) tensor -> (a, b) tensor =
   fun path x ->
    match !rest with
    | [] ->
        invalid_arg
          (Printf.sprintf
             "Nx.Ptree.rebuild: %s: a leaf in the template, none left of the \
              %d given"
             (Path.describe path) !taken)
    | Nx_effect.P y :: tail -> (
        rest := tail;
        incr taken;
        match
          Nx_core.Dtype.equal_witness (Nx_effect.dtype x) (Nx_effect.dtype y)
        with
        | Some Type.Equal -> y
        | None ->
            invalid_arg
              (Printf.sprintf
                 "Nx.Ptree.rebuild: %s: %s in the template, %s given"
                 (Path.describe path)
                 (Nx_core.Dtype.to_string (Nx_effect.dtype x))
                 (Nx_core.Dtype.to_string (Nx_effect.dtype y))))
  in
  let v = s.walk { tensor; report = ignore_report } Path.Root like in
  match !rest with
  | [] -> v
  | extra ->
      invalid_arg
        (Printf.sprintf "Nx.Ptree.rebuild: the template has %d leaves, %d given"
           !taken
           (!taken + List.length extra))

let map (type s) (s : s t)
    (f : 'a 'b. Path.t -> ('a, 'b) tensor -> ('a, 'b) tensor) (x : s) : s =
  s.walk { tensor = f; report = ignore_report } Path.Root x

type recorded = Recorded_leaf of Path.t * Nx_effect.packed | Recorded of visit

let map2 (type s) (s : s t)
    (f : 'a 'b. Path.t -> ('a, 'b) tensor -> ('a, 'b) tensor -> ('a, 'b) tensor)
    (x : s) (y : s) : s =
  let fn = "Nx.Ptree.map2" in
  let recorded = ref [] in
  let record : type a b. Path.t -> (a, b) tensor -> (a, b) tensor =
   fun path y ->
    recorded := Recorded_leaf (path, Nx_effect.P y) :: !recorded;
    y
  in
  let record_report path r =
    recorded := Recorded (Report (path, r)) :: !recorded
  in
  ignore (s.walk { tensor = record; report = record_report } Path.Root y);
  let expected = ref (List.rev !recorded) in
  let tensor : type a b. Path.t -> (a, b) tensor -> (a, b) tensor =
   fun path x ->
    match !expected with
    | Recorded_leaf (p, Nx_effect.P y) :: rest when Path.equal p path -> (
        expected := rest;
        match
          Nx_core.Dtype.equal_witness (Nx_effect.dtype x) (Nx_effect.dtype y)
        with
        | Some Type.Equal -> f path x y
        | None ->
            invalid_arg
              (Printf.sprintf "%s: %s: %s in the first value, %s in the second"
                 fn (Path.describe path)
                 (Nx_core.Dtype.to_string (Nx_effect.dtype x))
                 (Nx_core.Dtype.to_string (Nx_effect.dtype y))))
    | _ -> raise_notrace Mismatch
  in
  let report path r =
    match !expected with
    | Recorded (Report (p, r')) :: rest
      when equal_report r r' && Path.equal p path ->
        expected := rest
    | _ -> raise_notrace Mismatch
  in
  let mismatch () =
    check_same fn ~this:"in the first value" ~that:"in the second"
      (skeleton s x) (skeleton s y);
    walked_two_ways fn
  in
  match s.walk { tensor; report } Path.Root x with
  | z -> ( match !expected with [] -> z | _ -> mismatch ())
  | exception Mismatch -> mismatch ()

let fold (type s) (s : s t)
    (f : 'a 'b. Path.t -> ('a, 'b) tensor -> 'acc -> 'acc) (x : s) (acc : 'acc)
    : 'acc =
  let acc = ref acc in
  let tensor : type a b. Path.t -> (a, b) tensor -> (a, b) tensor =
   fun path x ->
    acc := f path x !acc;
    x
  in
  ignore (s.walk { tensor; report = ignore_report } Path.Root x);
  !acc

(* Walks at any payload *)

let cast_tensor (type a b c d) (dt : (c, d) Nx_core.Dtype.t) (x : (a, b) tensor)
    : (c, d) tensor =
  match Nx_core.Dtype.equal_witness (Nx_effect.dtype x) dt with
  | Some Type.Equal -> x
  | None -> Nx_effect.cast ~dtype:dt x

module Payload = struct
  let map (module U : S) (f : Path.t -> 'a -> 'b) (x : 'a U.t) : 'b U.t =
    U.walk
      {
        env = { ops = keep; leaf = (fun _ path x -> f path x) };
        path = Path.Root;
      }
      x

  let fold (module U : S) (f : Path.t -> 'a -> 'acc -> 'acc) (x : 'a U.t)
      (acc : 'acc) : 'acc =
    let acc = ref acc in
    let leaf _ path x =
      acc := f path x !acc;
      x
    in
    ignore (U.walk { env = { ops = keep; leaf }; path = Path.Root } x);
    !acc

  let record (module U : S) (x : 'a U.t) : 'a list * Skeleton.t =
    let leaves = ref [] and visits = ref [] in
    let ops =
      {
        tensor = (fun _ x -> x);
        report = (fun path r -> visits := Report (path, r) :: !visits);
      }
    in
    let leaf _ path x =
      leaves := x :: !leaves;
      visits := Leaf path :: !visits;
      x
    in
    ignore (U.walk { env = { ops; leaf }; path = Path.Root } x);
    (List.rev !leaves, Skeleton.of_rev !visits)

  let map2 (module U : S) (f : Path.t -> 'a -> 'b -> 'c) (x : 'a U.t)
      (y : 'b U.t) : 'c U.t =
    let fn = "Nx.Ptree.Payload.map2" in
    let ys, ky = record (module U) y in
    check_same fn ~this:"in the first value" ~that:"in the second"
      (snd (record (module U) x))
      ky;
    let rest = ref ys in
    let leaf _ path x =
      match !rest with
      | [] -> walked_two_ways fn
      | y :: tail ->
          rest := tail;
          f path x y
    in
    U.walk { env = { ops = keep; leaf }; path = Path.Root } x
end

let cast (module U : S) (dt : ('c, 'd) Nx_core.Dtype.t)
    (x : ('a, 'b) tensor U.t) : ('c, 'd) tensor U.t =
  Payload.map (module U) (fun _ x -> cast_tensor dt x) x
