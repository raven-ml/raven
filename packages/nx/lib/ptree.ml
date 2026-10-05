(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, 'b) tensor = ('a, 'b) Value.t
type placement = Placement.t
type packed = Value.packed

module Path = struct
  type seg = Field of string | Index of int
  type t = Root | Dot of t * string | At of t * int

  let root = Root

  let add seg p =
    match seg with Field name -> Dot (p, name) | Index i -> At (p, i)

  let v segs = List.fold_left (fun p seg -> add seg p) Root segs

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

  (* [under name p] is [p] below [Field name]. *)
  let rec under name = function
    | Root -> Dot (Root, name)
    | Dot (p, n) -> Dot (under name p, n)
    | At (p, i) -> At (under name p, i)

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

(* [prefix] is the path every part is walked under, below the path the walk
   starts at: [field] extends it, [iso] and [option] keep it. *)
type 's t = { walk : ops -> Path.t -> 's -> 's; prefix : Path.t }

let make walk = { walk; prefix = Path.Root }

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
  make (fun ops path x -> U.walk { env = { ops; leaf = s.walk }; path } x)

let tensor =
  { walk = (fun ops path x -> ops.tensor path x); prefix = Path.Root }

let instantiate (module U : S) : ('a, 'b) tensor U.t t = nest (module U) tensor
let unit = make (fun _ _ () -> ())

let pair a b =
  make (fun ops path (x, y) ->
      let x = a.walk ops (Path.At (path, 0)) x in
      let y = b.walk ops (Path.At (path, 1)) y in
      (x, y))

let option a =
  let walk ops path = function
    | None ->
        ops.report path absent;
        None
    | Some x ->
        ops.report path present;
        Some (a.walk ops path x)
  in
  { walk; prefix = a.prefix }

let list a =
  make (fun ops path l ->
      ops.report path (Length (List.length l));
      let rec go i = function
        | [] -> []
        | x :: rest ->
            let y = a.walk ops (Path.At (path, i)) x in
            y :: go (i + 1) rest
      in
      go 0 l)

let iso f g a =
  { walk = (fun ops path x -> f (a.walk ops path (g x))); prefix = a.prefix }

let field name s =
  {
    walk = (fun ops path x -> s.walk ops (Path.Dot (path, name)) x);
    prefix = Path.under name s.prefix;
  }

let prefix s = s.prefix

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

let flatten (type s) (s : s t) (x : s) : Value.packed list * Skeleton.t =
  let leaves = ref [] and visits = ref [] in
  let tensor : type a b. Path.t -> (a, b) tensor -> (a, b) tensor =
   fun path x ->
    leaves := Value.P x :: !leaves;
    visits := Leaf path :: !visits;
    x
  in
  let report path r = visits := Report (path, r) :: !visits in
  ignore (s.walk { tensor; report } Path.Root x);
  (List.rev !leaves, Skeleton.of_rev !visits)

let skeleton s x = snd (flatten s x)
let visits s x = Skeleton.visits (skeleton s x)

exception Mismatch

let rebuild (type s) (s : s t) ~(like : s) (leaves : Value.packed list) : s =
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
    | Value.P y :: tail -> (
        rest := tail;
        incr taken;
        match Nx_dtype.equal_witness (Value.dtype x) (Value.dtype y) with
        | Some Type.Equal -> y
        | None ->
            invalid_arg
              (Printf.sprintf
                 "Nx.Ptree.rebuild: %s: %s in the template, %s given"
                 (Path.describe path)
                 (Nx_dtype.to_string (Value.dtype x))
                 (Nx_dtype.to_string (Value.dtype y))))
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

let place s p x = map s (fun _ t -> Entry.place p t) x

type recorded = Recorded_leaf of Path.t * Value.packed | Recorded of visit

(* [zip fn s f x y] is [map2 s f x y], with [fn] naming the function in its
   errors. *)
let zip (type s) fn (s : s t)
    (f : 'a 'b. Path.t -> ('a, 'b) tensor -> ('a, 'b) tensor -> ('a, 'b) tensor)
    (x : s) (y : s) : s =
  let recorded = ref [] in
  let record : type a b. Path.t -> (a, b) tensor -> (a, b) tensor =
   fun path y ->
    recorded := Recorded_leaf (path, Value.P y) :: !recorded;
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
    | Recorded_leaf (p, Value.P y) :: rest when Path.equal p path -> (
        expected := rest;
        match Nx_dtype.equal_witness (Value.dtype x) (Value.dtype y) with
        | Some Type.Equal -> f path x y
        | None ->
            invalid_arg
              (Printf.sprintf "%s: %s: %s in the first value, %s in the second"
                 fn (Path.describe path)
                 (Nx_dtype.to_string (Value.dtype x))
                 (Nx_dtype.to_string (Value.dtype y))))
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

let map2 s
    (f : 'a 'b. Path.t -> ('a, 'b) tensor -> ('a, 'b) tensor -> ('a, 'b) tensor)
    x y =
  zip "Nx.Ptree.map2" s f x y

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

let cast_tensor (type a b c d) (dt : (c, d) Nx_dtype.t) (x : (a, b) tensor) :
    (c, d) tensor =
  match Nx_dtype.equal_witness (Value.dtype x) dt with
  | Some Type.Equal -> x
  | None -> Entry.cast dt x

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

let cast (module U : S) (dt : ('c, 'd) Nx_dtype.t) (x : ('a, 'b) tensor U.t) :
    ('c, 'd) tensor U.t =
  Payload.map (module U) (fun _ x -> cast_tensor dt x) x

(* Arithmetic

   The float leaves of a value are one vector, every other leaf is carried. *)

let is_float x = Nx_dtype.is_float (Value.dtype x)
let shape x = Nx_array.View.shape (Value.view x)

let check_shapes fn path x y =
  let sx = shape x and sy = shape y in
  if not (Nx_array.Shape.equal sx sy) then
    invalid_arg
      (Printf.sprintf "%s: %s: shape %s in the first value, %s in the second" fn
         (Path.describe path)
         (Nx_array.Shape.to_string sx)
         (Nx_array.Shape.to_string sy))

let check_scalar fn a =
  let s = shape a in
  if Array.length s <> 0 then
    invalid_arg
      (Printf.sprintf "%s: the factor has shape %s, expected a scalar" fn
         (Nx_array.Shape.to_string s))

(* [a * x], [a] cast to [x]'s dtype and expanded to its shape. *)
let times a x =
  Entry.binary Mul (Entry.broadcast (cast_tensor (Value.dtype x) a) (shape x)) x

(* [vdot x y] is [Nx.vdot x y]: the row of [x]'s elements times the column of
   [y]'s. *)
let vdot x y =
  let n = Nx_array.Shape.numel (shape x) in
  let r =
    Entry.matmul (Entry.reshape x [| 1; n |]) (Entry.reshape y [| n; 1 |])
  in
  Entry.reshape r [||]

let dot (type c d) s (dt : (c, d) Nx_dtype.t) x y : (c, d) tensor =
  let fn = "Nx.Ptree.dot" in
  let acc = ref None in
  let leaf : type a b. Path.t -> (a, b) tensor -> (a, b) tensor -> (a, b) tensor
      =
   fun path x y ->
    check_shapes fn path x y;
    (if is_float x then
       let d = vdot (cast_tensor dt x) (cast_tensor dt y) in
       acc :=
         Some (match !acc with None -> d | Some acc -> Entry.binary Add acc d));
    x
  in
  ignore (zip fn s leaf x y);
  match !acc with
  | Some d -> d
  | None -> Entry.full Placement.host dt [||] (Nx_dtype.zero dt)

let norm s dt x = Entry.unary Sqrt (dot s dt x x)

let scale s a x =
  check_scalar "Nx.Ptree.scale" a;
  map s (fun _ x -> if is_float x then times a x else x) x

let axpy s a x y =
  let fn = "Nx.Ptree.axpy" in
  check_scalar fn a;
  zip fn s
    (fun path x y ->
      check_shapes fn path x y;
      if is_float x then Entry.binary Add (times a x) y else y)
    x y
