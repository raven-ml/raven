(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Structures of tensors: the operations derived from one walk agree with each
   other and with the walk over drawn structures of every kind of part; the
   walks of chosen structures visit what ptree.mli words; the documented errors
   name what differs. *)

open Windtrap
module P = Nx.Ptree

let vec xs = Nx.create Nx.float32 [| Array.length xs |] xs
let int32s xs = Nx.create Nx.int32 [| Array.length xs |] xs

let paths s x =
  List.rev (P.fold s (fun p _ acc -> P.Path.to_string p :: acc) x [])

let visit_lines s x = List.map (Format.asprintf "%a" P.pp_visit) (P.visits s x)
let skeleton s x = snd (P.flatten s x)
let tensors s x = fst (P.flatten s x)

let same_tensors =
  List.for_all2 (fun (Nx.P a) (Nx.P b) ->
      match Nx_dtype.equal_witness (Nx.dtype a) (Nx.dtype b) with
      | Some Type.Equal -> a == b
      | None -> false)

(* Drawn structures *)

(* A structure with every kind of part a walk reports: a parameter's position, a
   fixed tensor, an integer, an option, a list, a variant's case and named
   fields. *)
type 'a tree =
  | Leaf of 'a
  | Fixed of Nx.int32_t
  | Int of int * 'a tree
  | Opt of 'a tree option
  | List of 'a tree list
  | Case of string * 'a tree
  | Fields of (string * 'a tree) list

module Tree = struct
  type 'a t = 'a tree

  let rec walk c =
    let open P.Walk in
    function
    | Leaf x -> Leaf (leaf c x)
    | Fixed t -> Fixed (tensor c t)
    | Int (n, t) ->
        let n = int c n in
        Int (n, walk c t)
    | Opt o -> Opt (option walk c o)
    | List l -> List (list walk c l)
    | Case (tag, t) ->
        case c tag;
        Case (tag, walk c t)
    | Fields l ->
        let rec fields = function
          | [] -> []
          | (name, t) :: rest ->
              let t = field c name walk t in
              (name, t) :: fields rest
        in
        Fields (fields l)
end

let tree : Nx.float32_t tree P.t = P.instantiate (module Tree)

let rec tree_gen depth =
  let open Gen in
  let leaf =
    map
      (fun xs -> Leaf (vec xs))
      (array ~size:(int_range 0 3) (map float_of_int (int_range (-9) 9)))
  in
  let fixed =
    map (fun n -> Fixed (int32s [| Int32.of_int n |])) (int_range 0 9)
  in
  if depth = 0 then frequency [ (3, leaf); (1, fixed) ]
  else
    let sub = tree_gen (depth - 1) in
    frequency
      [
        (3, leaf);
        (1, fixed);
        (1, map (fun (n, t) -> Int (n, t)) (pair (int_range 0 9) sub));
        (1, map (fun o -> Opt o) (option sub));
        (1, map (fun l -> List l) (list ~size:(int_range 0 3) sub));
        ( 1,
          map (fun (tag, t) -> Case (tag, t)) (pair (of_list [ "a"; "b" ]) sub)
        );
        ( 2,
          let* names = subsequence [ "w"; "b"; "x" ] in
          let+ children = list ~size:(constant (List.length names)) sub in
          Fields (List.combine names children) );
      ]

(* A value prints as its visits and tensors. *)
let value =
  Testable.contramap
    (fun x ->
      ( visit_lines tree x,
        List.map
          (fun (Nx.P t) ->
            (Nx_dtype.to_string (Nx.dtype t), Nx.to_array (Nx.cast Nx.float64 t)))
          (tensors tree x) ))
    (pair (list string) (list (pair string (array float_exact))))

let trees = Gen.with_pp (Testable.pp value) (tree_gen 3)
let twice _ t = Nx.add t t

(* A function of the path: negates the tensors at an even depth. *)
let negate_even p t =
  if List.length (P.Path.segments p) mod 2 = 0 then Nx.neg t else t

let rec fixed = function
  | Leaf _ | Opt None -> []
  | Fixed t -> [ t ]
  | Int (_, t) | Case (_, t) | Opt (Some t) -> fixed t
  | List l -> List.concat_map fixed l
  | Fields l -> List.concat_map (fun (_, t) -> fixed t) l

let laws =
  group "derived operations"
    [
      prop
        "map with the identity, rebuild of flatten, and a cast to float64 and \
         back give the value; the cast keeps fixed tensors"
        trees (fun x ->
          is_true ~msg:"the identity keeps the tensors"
            (same_tensors (tensors tree x)
               (tensors tree (P.map tree (fun _ t -> t) x)));
          Law.round_trip value pass (P.flatten tree)
            (fun (ts, _) -> P.rebuild tree ~like:x ts)
            x;
          let cast dt = P.cast (module Tree) dt in
          Law.round_trip value pass (cast Nx.float64) (cast Nx.float32) x;
          is_true ~msg:"fixed tensors kept"
            (List.for_all2 ( == ) (fixed x) (fixed (cast Nx.float64 x))));
      prop
        "map of a map is the map of the composition, keeps paths and \
         skeletons, and map2 is map over the zipped tensors"
        trees (fun x ->
          let y = P.map tree negate_even x in
          equal value
            (P.map tree (fun p t -> twice p (negate_even p t)) x)
            (P.map tree twice y);
          equal ~msg:"paths" (list string) (paths tree x) (paths tree y);
          is_true ~msg:"skeletons"
            (P.Skeleton.equal (skeleton tree x) (skeleton tree y));
          let add (Nx.P a) (Nx.P b) =
            match Nx_dtype.equal_witness (Nx.dtype a) (Nx.dtype b) with
            | Some Type.Equal -> Nx.P (Nx.add a b)
            | None -> fail "tensors of different dtypes at one position"
          in
          equal value
            (P.rebuild tree ~like:x
               (List.map2 add (tensors tree x) (tensors tree y)))
            (P.map2 tree (fun _ a b -> Nx.add a b) x y));
      prop
        "fold gives the tensors of flatten at the paths of visits, and \
         Payload.fold the parameter's positions among them"
        trees (fun x ->
          let folded =
            List.rev
              (P.fold tree
                 (fun p t acc -> (P.Path.to_string p, Nx.P t) :: acc)
                 x [])
          in
          equal (list string)
            (List.filter_map
               (function
                 | P.Leaf p -> Some (P.Path.to_string p) | P.Report _ -> None)
               (P.visits tree x))
            (List.map fst folded);
          is_true ~msg:"tensors"
            (same_tensors (tensors tree x) (List.map snd folded));
          let named =
            P.Payload.map (module Tree) (fun p _ -> P.Path.to_string p) x
          in
          equal (list string)
            (List.filter_map
               (fun (p, Nx.P t) ->
                 if Nx_dtype.equal (Nx.dtype t) Nx.float32 then Some p else None)
               folded)
            (List.rev
               (P.Payload.fold
                  (module Tree)
                  (fun _ name acc -> name :: acc)
                  named [])));
      prop
        "skeletons are equal exactly when the visits are, and then hash alike; \
         map2 raises exactly when the visits or a dtype differ"
        (Gen.pair trees trees) (fun (x, y) ->
          let same =
            List.equal
              (fun a b ->
                match (a, b) with
                | P.Leaf p, P.Leaf q -> P.Path.equal p q
                | P.Report (p, r), P.Report (q, s) -> P.Path.equal p q && r = s
                | _ -> false)
              (P.visits tree x) (P.visits tree y)
          in
          cover "equal visits" same;
          cover "different visits" (not same);
          let kx = skeleton tree x and ky = skeleton tree y in
          equal (pair bool bool) (same, same)
            ( P.Skeleton.equal kx ky,
              P.Skeleton.diff ~this:"here" kx ~that:"there" ky = None );
          if same then
            equal ~msg:"hashes" int (P.Skeleton.hash kx) (P.Skeleton.hash ky);
          let dtypes x =
            List.map
              (fun (Nx.P t) -> Nx_dtype.to_string (Nx.dtype t))
              (tensors tree x)
          in
          let zips = same && dtypes x = dtypes y in
          match P.map2 tree (fun _ a _ -> a) x y with
          | exception Invalid_argument m ->
              is_false ~msg:"map2 refused values that zip" zips;
              starts_with ~affix:"Nx.Ptree.map2: " m
          | _ -> is_true ~msg:"map2 zipped values that differ" zips);
    ]

(* Chosen structures *)

module Linear = struct
  type 'a t = { w : 'a; b : 'a option }

  let walk c { w; b } =
    let open P.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" (option leaf) b in
    { w; b }
end

module Mlp = struct
  type 'a t = {
    l1 : 'a Linear.t;
    l2 : 'a Linear.t;
    steps : Nx.int32_t;
    window : int option;
  }

  let walk c m =
    let open P.Walk in
    let l1 = field c "l1" Linear.walk m.l1 in
    let l2 = field c "l2" Linear.walk m.l2 in
    let steps = field c "steps" tensor m.steps in
    let window = field c "window" (option int) m.window in
    { l1; l2; steps; window }
end

type 'a weight =
  | Float of 'a
  | Mxfp4 of { blocks : Nx.uint8_t; scales : Nx.uint8_t }

module Weight = struct
  type 'a t = 'a weight

  let walk c =
    let open P.Walk in
    function
    | Float w ->
        case c "float";
        Float (leaf c w)
    | Mxfp4 { blocks; scales } ->
        case c "mxfp4";
        let blocks = field c "blocks" tensor blocks in
        let scales = field c "scales" tensor scales in
        Mxfp4 { blocks; scales }
end

module Cache = struct
  type 'a t = { keys : 'a; values : 'a }

  let walk c { keys; values } =
    let open P.Walk in
    let keys = field c "keys" leaf keys in
    let values = field c "values" leaf values in
    { keys; values }
end

module Adam = struct
  type 'p t = { mu : 'p; nu : 'p; step : Nx.int32_t }

  let walk c s =
    let open P.Walk in
    let mu = field c "mu" leaf s.mu in
    let nu = field c "nu" leaf s.nu in
    let step = field c "step" tensor s.step in
    { mu; nu; step }
end

module Dotted = struct
  type 'a t = 'a

  let walk c x = P.Walk.field c "a.b" P.Walk.leaf x
end

module Nested = struct
  type 'a t = 'a

  let walk c x = P.Walk.(field c "a" (fun c x -> field c "b" leaf x) x)
end

let mlp : Nx.float32_t Mlp.t P.t = P.instantiate (module Mlp)
let cache : Nx.float32_t Cache.t P.t = P.instantiate (module Cache)
let dotted : Nx.float32_t Dotted.t P.t = P.instantiate (module Dotted)
let nested : Nx.float32_t Nested.t P.t = P.instantiate (module Nested)

let params ?(window = Some 4) ?(b2 = None) () =
  {
    Mlp.l1 = { w = vec [| 1.; 2. |]; b = Some (vec [| 3. |]) };
    l2 = { w = vec [| 4. |]; b = b2 };
    steps = int32s [| 7l |];
    window;
  }

type out = { loss : Nx.float32_t; model : Nx.float32_t Mlp.t }

let out =
  P.iso
    (fun (loss, model) -> { loss; model })
    (fun o -> (o.loss, o.model))
    P.(pair tensor mlp)

(* A structure whose parts have structures and no module, walked with
   [structure]. *)
module Parts = struct
  type 'a t = {
    w : 'a;
    both : Nx.float32_t Mlp.t * Nx.float32_t Cache.t;
    out : out;
    caches : Nx.float32_t Cache.t option list;
  }

  let walk c p =
    let open P.Walk in
    let w = field c "w" leaf p.w in
    let both = field c "both" (structure P.(pair mlp cache)) p.both in
    let out = field c "out" (structure out) p.out in
    let caches =
      field c "caches" (structure P.(list (option cache))) p.caches
    in
    { w; both; out; caches }
end

let kv () = { Cache.keys = vec [| 1. |]; values = vec [| 2. |] }

let parts =
  {
    Parts.w = vec [| 5. |];
    both = (params (), kv ());
    out = { loss = vec [| 0.5 |]; model = params ~b2:(Some (vec [| 6. |])) () };
    caches = [ Some (kv ()); None ];
  }

let chosen =
  group "chosen structures"
    [
      test "visit what they walk, at paths extended from each field" (fun () ->
          equal (list string)
            [
              "w: a leaf";
              "both.0.l1.w: a leaf";
              "both.0.l1.b: Some";
              "both.0.l1.b: a leaf";
              "both.0.l2.w: a leaf";
              "both.0.l2.b: None";
              "both.0.steps: a leaf";
              "both.0.window: Some";
              "both.0.window: int 4";
              "both.1.keys: a leaf";
              "both.1.values: a leaf";
              "out.0: a leaf";
              "out.1.l1.w: a leaf";
              "out.1.l1.b: Some";
              "out.1.l1.b: a leaf";
              "out.1.l2.w: a leaf";
              "out.1.l2.b: Some";
              "out.1.l2.b: a leaf";
              "out.1.steps: a leaf";
              "out.1.window: Some";
              "out.1.window: int 4";
              "caches: length 2";
              "caches.0: Some";
              "caches.0.keys: a leaf";
              "caches.0.values: a leaf";
              "caches.1: None";
            ]
            (visit_lines (P.instantiate (module Parts)) parts);
          let u8 = Nx.zeros Nx.uint8 [| 2 |] in
          equal ~msg:"case tags" (list string)
            [
              "the root: length 2";
              "0: case \"float\"";
              "0: a leaf";
              "1: case \"mxfp4\"";
              "1.blocks: a leaf";
              "1.scales: a leaf";
            ]
            (visit_lines
               (P.list (P.instantiate (module Weight)))
               [ Float (vec [| 1. |]); Mxfp4 { blocks = u8; scales = u8 } ]);
          equal ~msg:"nest" (list string)
            [
              "mu.l1.w";
              "mu.l1.b";
              "mu.l2.w";
              "mu.steps";
              "nu.l1.w";
              "nu.l1.b";
              "nu.l2.w";
              "nu.steps";
              "step";
            ]
            (paths
               (P.nest (module Adam) mlp)
               { Adam.mu = params (); nu = params (); step = int32s [| 0l |] });
          equal ~msg:"parts without tensors still count" (list bool)
            [ false; false ]
            [
              P.Skeleton.equal
                (skeleton P.(option unit) (Some ()))
                (skeleton P.(option unit) None);
              P.Skeleton.equal
                (skeleton P.(list unit) [ (); () ])
                (skeleton P.(list unit) [ () ]);
            ]);
      test
        "cast and Payload.map keep fixed tensors and the tensors of parts \
         walked with structure" (fun () ->
          let y = P.cast (module Parts) Nx.bfloat16 parts in
          let z =
            P.Payload.map (module Parts) (fun p _ -> P.Path.to_string p) parts
          in
          equal (pair string string) ("bfloat16", "w")
            (Nx_dtype.to_string (Nx.dtype y.w), z.w);
          is_true
            ((fst y.both).l1.w == (fst parts.both).l1.w
            && y.out.loss == parts.out.loss
            && (snd z.both).keys == (snd parts.both).keys
            && z.out.model.l2.w == parts.out.model.l2.w);
          let x = params () in
          is_true ~msg:"Payload.map2 keeps the first value's"
            ((P.Payload.map2 (module Mlp) (fun _ a _ -> a) x (params ())).steps
           == x.steps));
      test "Payload maps and folds hold any type" (fun () ->
          let dims = { Linear.w = [ 16; 128 ]; b = Some [ 128 ] } in
          let sizes =
            P.Payload.map
              (module Linear)
              (fun _ ds -> List.fold_left ( * ) 1 ds)
              dims
          in
          equal int 2176
            (P.Payload.fold (module Linear) (fun _ n acc -> n + acc) sizes 0);
          let scale =
            P.Payload.map
              (module Linear)
              (fun _ n x -> Nx.mul_s x (float_of_int n))
              sizes
          in
          equal (array float_exact) [| 2048. |]
            (Nx.to_array (scale.w (vec [| 1. |])));
          let zipped =
            P.Payload.map2
              (module Linear)
              (fun p d n -> (P.Path.to_string p, List.length d, n))
              dims sizes
          in
          equal (option (triple string int int)) (Some ("b", 1, 128)) zipped.b);
      test "paths compare by segments, not by how they print" (fun () ->
          let path s =
            Option.get (P.fold s (fun p _ _ -> Some p) (vec [| 1. |]) None)
          in
          equal (pair string bool)
            (P.Path.to_string (path dotted), false)
            ( P.Path.to_string (path nested),
              P.Path.equal (path dotted) (path nested) );
          equal ~msg:"the root prints empty" string ""
            (P.Path.to_string (path P.tensor));
          equal ~msg:"pair numbers its sides" (list string) [ "0"; "1" ]
            (paths P.(pair tensor tensor) (vec [| 1. |], vec [| 2. |]));
          equal (option string)
            (Some "[\"a.b\"]: a leaf here, a leaf at [\"a\"; \"b\"] there")
            (P.Skeleton.diff ~this:"here"
               (skeleton dotted (vec [| 1. |]))
               ~that:"there"
               (skeleton nested (vec [| 1. |]))));
      test "signatures record each argument's role in order" (fun () ->
          let rec roles : type f. f P.fn -> P.role list = function
            | P.Returns _ -> []
            | P.Arg (role, _, rest) -> role :: roles rest
          in
          is_true
            (roles
               P.(
                 tensor
                 @-> consumes (list cache)
                 @@ returns (pair tensor (list cache)))
            = [ P.Read; P.Consumed ]));
    ]

(* Constructed paths *)

let pp_seg ppf = function
  | P.Path.Field name -> Format.fprintf ppf "Field %S" name
  | P.Path.Index i -> Format.fprintf ppf "Index %d" i

let seg = Testable.make ~pp:pp_seg ~equal:( = )

(* A path prints as its segments, which tell apart paths that print alike. *)
let path =
  Testable.make ~equal:P.Path.equal ~pp:(fun ppf p ->
      Testable.pp (list seg) ppf (P.Path.segments p))

(* An equal segment whose name is held in another string. *)
let fresh = function
  | P.Path.Field name -> P.Path.Field (Bytes.to_string (Bytes.of_string name))
  | Index _ as seg -> seg

let seg_gen =
  let open Gen in
  with_pp pp_seg
    (frequency
       [
         ( 2,
           map
             (fun name -> P.Path.Field name)
             (one_of
                [
                  of_list
                    [ ""; "."; "a.b"; "a"; "b"; "\xc3\xa9t\xc3\xa9"; "\000" ];
                  string;
                ]) );
         ( 2,
           map
             (fun i -> P.Path.Index i)
             (one_of
                [
                  int_range (-1) 3;
                  of_list [ min_int; min_int + 1; max_int - 1; max_int ];
                ]) );
       ])

let segs_gen = Gen.list ~size:(Gen.int_range 0 6) seg_gen

(* Segment lists one edit away from [segs]: one segment replaced, the last
   dropped, one more appended, or an index turned into the field that prints
   like it. *)
let neighbour_gen segs =
  let open Gen in
  let n = List.length segs in
  let replace i seg = List.mapi (fun j s -> if j = i then seg else s) segs in
  let indices =
    List.concat
      (List.mapi
         (fun j -> function P.Path.Index i -> [ (j, i) ] | Field _ -> [])
         segs)
  in
  let appended = map (fun seg -> segs @ [ seg ]) seg_gen in
  let edits =
    if n = 0 then []
    else
      [
        bind (int_range 0 (n - 1)) (fun i -> map (replace i) seg_gen);
        constant (List.filteri (fun j _ -> j < n - 1) segs);
      ]
  in
  let renamed =
    if indices = [] then []
    else
      [
        map
          (fun (j, i) -> replace j (P.Path.Field (Int.to_string i)))
          (of_list indices);
      ]
  in
  one_of ((appended :: edits) @ renamed)

let rec is_prefix a b =
  match (a, b) with
  | [], _ -> true
  | x :: a, y :: b -> x = y && is_prefix a b
  | _ :: _, [] -> false

let walked s x = List.rev (P.fold s (fun p _ acc -> p :: acc) x [])

let constructed =
  group "constructed paths"
    [
      prop "segments of v is the list v was given" segs_gen (fun segs ->
          Law.round_trip (list seg) path P.Path.v P.Path.segments segs);
      prop "add appends one segment and agrees with v"
        (Gen.pair segs_gen seg_gen) (fun (segs, s) ->
          let p = P.Path.add s (P.Path.v segs) in
          equal (list seg) (segs @ [ s ]) (P.Path.segments p);
          equal path (P.Path.v (segs @ [ s ])) p);
      prop "v of a walked path's segments is that path" trees (fun x ->
          let ps = walked tree x in
          cover "a path of two segments or more"
            (List.exists (fun p -> List.length (P.Path.segments p) >= 2) ps);
          List.iter (Law.round_trip path (list seg) P.Path.segments P.Path.v) ps);
      prop "constructed paths are equal exactly when their segments are"
        ~count:300
        ~examples:
          P.Path.
            [
              ([ Field "a"; Field "w" ], [ Field "b"; Field "w" ]);
              ([ Index 0; Index 1 ], [ Index 2; Index 1 ]);
              ([ Index 0; Index 1 ], [ Index 0 ]);
              ([ Index 1 ], [ Field "1" ]);
            ]
        (Gen.with_pp
           (Testable.pp (pair (list seg) (list seg)))
           (Gen.frequency
              [
                (1, Gen.map (fun a -> (a, a)) segs_gen);
                ( 3,
                  Gen.bind segs_gen (fun a ->
                      Gen.map (fun b -> (a, b)) (neighbour_gen a)) );
                (1, Gen.pair segs_gen segs_gen);
              ]))
        (fun (a, b) ->
          let last l = List.nth_opt (List.rev l) 0 in
          cover "equal" (a = b);
          cover "differ before the last segment"
            (a <> b
            && List.length a = List.length b
            && a <> []
            && last a = last b);
          cover "one is a prefix of the other"
            (a <> b && (is_prefix a b || is_prefix b a));
          cover "differ and print alike"
            (a <> b
            && String.equal
                 (P.Path.to_string (P.Path.v a))
                 (P.Path.to_string (P.Path.v b)));
          let p = P.Path.v a and q = P.Path.v (List.map fresh b) in
          Law.equivalence path (p, q);
          equal bool (a = b) (P.Path.equal p q));
      test "the root has no segments and is v of no segments" (fun () ->
          equal (list seg) [] (P.Path.segments P.Path.root);
          equal path P.Path.root (P.Path.v []));
      test "constructed paths equal the paths a walk gives" (fun () ->
          let x = vec [| 1. |] in
          let field name = fresh (Field name) in
          equal (list path) [ P.Path.root ] (walked P.tensor x);
          equal (list path) [ P.Path.v [ field "a.b" ] ] (walked dotted x);
          equal (list path)
            [ P.Path.(add (field "b") (add (field "a") root)) ]
            (walked nested x);
          equal (list path)
            P.Path.[ v [ Index 0; Index 1 ]; v [ Index 1; Index 1 ] ]
            (walked
               P.(list (pair (option tensor) tensor))
               [ (None, x); (None, x) ]));
    ]

(* Errors *)

module Packed_list = struct
  type _ t = Nx.packed list

  let walk c l = P.Walk.list (fun c (Nx.P x) -> Nx.P (P.Walk.tensor c x)) c l
end

module Flaky = struct
  type 'a t = 'a

  let calls = ref 0

  let walk c x =
    incr calls;
    P.Walk.field c (if !calls = 2 then "a" else "b") P.Walk.leaf x
end

module Either_path = struct
  type 'a t = Dotted of 'a | Nested of 'a

  let walk c = function
    | Dotted x -> Dotted (Dotted.walk c x)
    | Nested x -> Nested (Nested.walk c x)
end

let errors =
  let map2 s x y () = ignore (P.map2 s (fun _ a _ -> a) x y) in
  let u8 = Nx.zeros Nx.uint8 [| 2 |] and x = params () in
  let leaves = tensors mlp x in
  let rebuild ts () = ignore (P.rebuild mlp ~like:x ts) in
  cases "raise, naming the first visit that differs" ~name:fst
    [
      ( "Nx.Ptree.map2: l2.b: Some in the first value, None in the second",
        map2 mlp (params ~b2:(Some (vec [| 1. |])) ()) x );
      ( "Nx.Ptree.map2: window: int 4 in the first value, int 8 in the second",
        map2 mlp x (params ~window:(Some 8) ()) );
      ( "Nx.Ptree.map2: 1: case \"mxfp4\" in the first value, case \"float\" \
         in the second",
        map2
          (P.list (P.instantiate (module Weight)))
          [ Float (vec [| 1. |]); Mxfp4 { blocks = u8; scales = u8 } ]
          [ Float (vec [| 1. |]); Float (vec [| 2. |]) ] );
      ( "Nx.Ptree.map2: the root: length 2 in the first value, length 1 in the \
         second",
        map2 P.(list tensor) [ vec [| 1. |]; vec [| 2. |] ] [ vec [| 1. |] ] );
      ( "Nx.Ptree.map2: 0: float32 in the first value, int32 in the second",
        map2
          (P.instantiate (module Packed_list))
          [ Nx.P (vec [| 1. |]) ]
          [ Nx.P (int32s [| 1l |]) ] );
      ( "Nx.Ptree.map2: [\"a.b\"]: a leaf in the first value, a leaf at \
         [\"a\"; \"b\"] in the second",
        map2
          (P.instantiate (module Either_path))
          (Either_path.Dotted (vec [| 1. |]))
          (Either_path.Nested (vec [| 2. |])) );
      ( "Nx.Ptree.map2: the structure's walk visited one value two ways",
        map2 (P.instantiate (module Flaky)) (vec [| 1. |]) (vec [| 2. |]) );
      ( "Nx.Ptree.Payload.map2: b: Some in the first value, None in the second",
        fun () ->
          ignore
            (P.Payload.map2
               (module Linear)
               (fun _ a _ -> a)
               { Linear.w = 1; b = Some 2 }
               { Linear.w = 1; b = None }) );
      ( "Nx.Ptree.Payload.map2: the root: length 1 in the first value, length \
         2 in the second",
        fun () ->
          ignore
            (P.Payload.map2
               (module Packed_list)
               (fun _ a _ -> a)
               [ Nx.P (vec [| 1. |]) ]
               [ Nx.P (vec [| 1. |]); Nx.P (vec [| 2. |]) ]) );
      ( "Nx.Ptree.rebuild: steps: a leaf in the template, none left of the 3 \
         given",
        rebuild (List.filteri (fun i _ -> i < 3) leaves) );
      ( "Nx.Ptree.rebuild: the template has 4 leaves, 5 given",
        rebuild (leaves @ [ List.hd leaves ]) );
      ( "Nx.Ptree.rebuild: steps: int32 in the template, float32 given",
        rebuild
          (List.mapi (fun i l -> if i = 3 then List.hd leaves else l) leaves) );
      ( "unpack: expected dtype float64, got float32",
        fun () -> ignore (Nx.unpack Nx.float64 (Nx.P (vec [| 1. |]))) );
    ]
    (fun (m, f) -> raises (Invalid_argument m) f)

(* Fields *)

let fields =
  let under names p =
    P.Path.v (List.map (fun n -> P.Path.Field n) names @ P.Path.segments p)
  in
  let v names = P.Path.v (List.map (fun n -> P.Path.Field n) names) in
  let f = P.field in
  group "fields"
    [
      prop "field walks as its structure with every path under the field" trees
        (fun x ->
          let s = f "model" (f "enc" tree) in
          cover "a leaf" (walked tree x <> []);
          equal (list path)
            (List.map (under [ "model"; "enc" ]) (walked tree x))
            (walked s x);
          equal (list string)
            (List.map
               (fun v -> Format.asprintf "%a" P.pp_visit v)
               (List.map
                  (function
                    | P.Leaf p -> P.Leaf (under [ "model"; "enc" ] p)
                    | P.Report (p, r) -> P.Report (under [ "model"; "enc" ] p, r))
                  (P.visits tree x)))
            (visit_lines s x);
          equal value x (P.map s (fun _ t -> t) x));
      cases ~name:fst "prefix is the path field states"
        [
          ("tensor", (P.prefix P.tensor, P.Path.root));
          ("mlp", (P.prefix mlp, P.Path.root));
          ("field", (P.prefix (f "a" P.tensor), v [ "a" ]));
          ("nested fields", (P.prefix (f "a" (f "b" P.tensor)), v [ "a"; "b" ]));
          ( "a field of a list",
            (P.prefix (f "a" (P.list (f "b" P.tensor))), v [ "a" ]) );
          ( "iso keeps it",
            (P.prefix (P.iso Fun.id Fun.id (f "a" P.tensor)), v [ "a" ]) );
          ("option keeps it", (P.prefix (P.option (f "a" P.tensor)), v [ "a" ]));
          ( "a list's is the root",
            (P.prefix (P.list (f "a" P.tensor)), P.Path.root) );
          ( "a pair's is the root",
            (P.prefix (P.pair (f "a" P.tensor) (f "a" P.tensor)), P.Path.root)
          );
          ( "a nest's is the root",
            (P.prefix (P.nest (module Tree) (f "a" P.tensor)), P.Path.root) );
        ]
        (fun (_, (actual, expected)) -> equal path expected actual);
    ]

(* Arithmetic. A value's float tensors are one vector. The values here are lists
   of tensors of every float dtype, beside an int32 tensor the vector skips,
   with scalars, empty tensors and strided views, drawn several at a time with
   one skeleton: at each position one dtype, shape and layout. *)

let packed = P.instantiate (module Packed_list)
let f64 = Nx.float64
let floats_of t = Nx.to_array (Nx.cast f64 t)
let pp_hex ppf x = Format.fprintf ppf "%h" x
let exactly = Nx_test.close ~rel:0. ()

type dtype = D : ('a, 'b) Nx.dtype * (float -> 'a) -> dtype

let dtypes =
  [
    D (Nx.float16, Fun.id);
    D (Nx.bfloat16, Fun.id);
    D (Nx.float32, Fun.id);
    D (Nx.float64, Fun.id);
    D (Nx.int32, Int32.of_float);
  ]

type position = { layout : Nx_test.layout list; at : Nx.packed list }

(* [k] tensors at one position, their elements drawn from [value]. *)
let position k value =
  let open Gen in
  let* (D (dt, of_float)) = of_list dtypes in
  let* s = array ~size:(int_range 0 3) (int_range 0 3) in
  let* layout = Nx_test.layout in
  let one =
    let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) value in
    Nx.P (Nx_test.lay_out layout (Nx.create dt s (Array.map of_float xs)))
  in
  let+ at = list ~size:(constant k) one in
  { layout; at }

let pp_tensor ppf (Nx.P t) =
  Format.fprintf ppf "%a %a" Nx_dtype.pp (Nx.dtype t) (Nx_test.Ref.pp pp_hex)
    (Nx_test.Ref.of_nx (Nx.cast f64 t))

let pp_value ppf x =
  Format.fprintf ppf "@[<v>[%a]@]" (Format.pp_print_list pp_tensor) x

(* [k] values of one skeleton, with the positions they were drawn at. *)
let values k value =
  Gen.with_pp
    (fun ppf (_, xs) -> Format.pp_print_list pp_value ppf xs)
    (Gen.map
       (fun ps ->
         (ps, List.init k (fun i -> List.map (fun p -> List.nth p.at i) ps)))
       (Gen.list ~size:(Gen.int_range 0 4) (position k value)))

let is_float (Nx.P t) = Nx_dtype.is_float (Nx.dtype t)
let numel (Nx.P t) = Nx.numel t
let dtype_name (Nx.P t) = Nx_dtype.to_string (Nx.dtype t)

(* The cases a law over values must see. *)
let cover_values ps =
  let firsts = List.map (fun p -> List.hd p.at) ps in
  let floats = List.filter is_float firsts in
  let names = List.sort_uniq String.compare (List.map dtype_name floats) in
  cover "an empty value" (ps = []);
  cover "one tensor" (List.length ps = 1);
  cover "several float dtypes" (List.length names > 1);
  cover "an int32 tensor beside a float one"
    (floats <> [] && List.length floats < List.length firsts);
  cover "an empty tensor" (List.exists (fun t -> numel t = 0) firsts);
  cover "a strided view" (List.exists (fun p -> p.layout <> []) ps)

(* Small integers, whose products and sums the laws form are exact at float64
   and at every tensor's dtype. *)
let small = Gen.map float_of_int (Gen.int_range (-6) 6)

let any =
  Gen.frequency
    [
      (6, Gen.float_range (-1e3) 1e3);
      ( 1,
        Gen.of_list ~pp:pp_hex
          [ Float.nan; infinity; neg_infinity; -0.; 1e300; 0x1p-1060 ] );
    ]

let factor =
  Gen.map (Nx.scalar f64)
    (Gen.frequency
       [
         (3, small); (1, Gen.of_list ~pp:pp_hex [ -0.; 0.5; Float.nan; 1e300 ]);
       ])

let scalar = Testable.contramap (fun t -> Nx.item [] (Nx.cast f64 t)) exactly

let value_w =
  Testable.contramap
    (List.map (fun (Nx.P t as p) -> (dtype_name p, Nx.shape t, floats_of t)))
    (list (triple string (array int) (array exactly)))

type floating = F : (float, 'b) Nx.dtype * float * float -> floating

let pp_floating ppf (F (dt, _, _)) = Nx_dtype.pp ppf dt

(* The dtypes a dot accumulates at, with their unit roundoff and their smallest
   subnormal. *)
let accumulations =
  Gen.of_list ~pp:pp_floating
    [ F (Nx.float32, 0x1p-24, 0x1p-149); F (Nx.float64, 0x1p-53, 0x1p-1074) ]

(* The float tensors of [x], cast to [dt] and flattened end to end. *)
let vector dt x =
  Nx.concatenate ~axis:0
    (Nx.zeros dt [| 0 |]
    :: List.filter_map
         (fun (Nx.P t as p) ->
           if is_float p then Some (Nx.flatten (Nx.cast dt t)) else None)
         x)

(* [f t u] at each float tensor of [x] and [y], [y]'s tensor elsewhere. *)
type leafwise = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let leafwise { f } x y =
  List.map2
    (fun (Nx.P t) (Nx.P u) ->
      match Nx_dtype.equal_witness (Nx.dtype t) (Nx.dtype u) with
      | Some Type.Equal -> if is_float (Nx.P t) then Nx.P (f t u) else Nx.P u
      | None -> assert false)
    x y

let two = function [ x; y ] -> (x, y) | _ -> assert false
let three = function [ x; y; z ] -> (x, y, z) | _ -> assert false
let times a t = Nx.mul (Nx.cast (Nx.dtype t) a) t
let dot dt x y = P.dot packed dt x y

let arithmetic =
  group "arithmetic"
    [
      prop
        "dot is the inner product of the float tensors cast and flattened end \
         to end, within the rounding of two orders of summation"
        (Gen.pair accumulations (values 2 any))
        (fun (F (dt, eps, tiny), (ps, xy)) ->
          cover_values ps;
          let x, y = two xy in
          let u = vector dt x and v = vector dt y in
          let terms =
            Array.map2
              (fun a b -> Float.abs (a *. b))
              (floats_of u) (floats_of v)
          in
          let n = float_of_int (Array.length terms) in
          let bound =
            (2. *. n *. eps *. Array.fold_left ( +. ) 0. terms) +. (n *. tiny)
          in
          equal
            (Testable.contramap
               (fun t -> Nx.item [] (Nx.cast f64 t))
               (Nx_test.close ~abs:bound ~rel:0. ()))
            (Nx.vdot u v) (dot dt x y));
      prop "dot is symmetric" (values 2 any) (fun (_, xy) ->
          let x, y = two xy in
          equal scalar (dot f64 x y) (dot f64 y x));
      prop "dot is additive in its first argument, under axpy with factor 1"
        (values 3 small) (fun (ps, xyz) ->
          cover_values ps;
          let x, y, z = three xyz in
          Law.homomorphic value_w scalar
            (fun x -> dot f64 x z)
            (P.axpy packed (Nx.scalar f64 1.))
            Nx.add (x, y));
      prop "dot is homogeneous in its first argument, under scale"
        (Gen.pair small (values 2 small))
        (fun (a, (_, xy)) ->
          let x, y = two xy in
          let a = Nx.scalar f64 a in
          equal scalar (Nx.mul a (dot f64 x y)) (dot f64 (P.scale packed a x) y));
      prop "norm is the square root of dot of a value with itself"
        (Gen.pair accumulations (values 1 any))
        (fun (F (dt, _, _), (_, x)) ->
          let x = List.hd x in
          let item t = Nx.item [] (Nx.cast f64 t) in
          equal exactly
            (item (Nx.sqrt (dot dt x x)))
            (item (P.norm packed dt x)));
      prop
        "axpy is a * t + u at each float tensor, rounded at its dtype, and the \
         second value's tensor elsewhere"
        (Gen.pair factor (values 2 any))
        (fun (a, (ps, xy)) ->
          cover_values ps;
          let x, y = two xy in
          equal value_w
            (leafwise { f = (fun t u -> Nx.add (times a t) u) } x y)
            (P.axpy packed a x y));
      prop "scale is a * t at each float tensor, and the tensor elsewhere"
        (Gen.pair factor (values 1 any))
        (fun (a, (_, x)) ->
          let x = List.hd x in
          equal value_w
            (leafwise { f = (fun t _ -> times a t) } x x)
            (P.scale packed a x));
      test "dot of a value without a float tensor is a zero of its dtype"
        (fun () ->
          let ints = [ Nx.P (int32s [| 3l; 4l |]) ] in
          List.iter
            (fun x ->
              let d = dot Nx.float32 x x in
              equal (array int) [||] (Nx.shape d);
              equal float_exact 0. (Nx.item [] d))
            [ []; ints ]);
      test
        "dot casts a bfloat16 tensor to float32 before it multiplies and sums"
        (fun () ->
          (* (1 + u)^2 + u/2 is 1 + 2^-6 + 2^-8 + 2^-14, exact at float32 and
             five bits past bfloat16. *)
          let u = 0x1p-7 in
          let x =
            [ Nx.P (Nx.create Nx.bfloat16 [| 2 |] [| 1. +. u; u /. 2. |]) ]
          and y = [ Nx.P (Nx.create Nx.bfloat16 [| 2 |] [| 1. +. u; 1. |]) ] in
          equal float_exact
            (1. +. 0x1p-6 +. 0x1p-8 +. 0x1p-14)
            (Nx.item [] (dot Nx.float32 x y)));
      cases "raise, naming what differs" ~name:fst
        [
          ( "Nx.Ptree.dot: w: shape [2,3] in the first value, [3,2] in the \
             second",
            fun () ->
              let w = P.field "w" P.tensor in
              ignore
                (P.dot w f64
                   (Nx.zeros f64 [| 2; 3 |])
                   (Nx.zeros f64 [| 3; 2 |])) );
          ( "Nx.Ptree.axpy: 0: shape [1] in the first value, [] in the second",
            fun () ->
              ignore
                (P.axpy packed (Nx.scalar f64 1.)
                   [ Nx.P (vec [| 1. |]) ]
                   [ Nx.P (Nx.scalar Nx.float32 1.) ]) );
          ( "Nx.Ptree.dot: the root: length 1 in the first value, length 2 in \
             the second",
            fun () ->
              ignore
                (dot f64
                   [ Nx.P (vec [| 1. |]) ]
                   [ Nx.P (vec [| 1. |]); Nx.P (vec [| 2. |]) ]) );
          ( "Nx.Ptree.axpy: 0: float32 in the first value, int32 in the second",
            fun () ->
              ignore
                (P.axpy packed (Nx.scalar f64 1.)
                   [ Nx.P (vec [| 1. |]) ]
                   [ Nx.P (int32s [| 1l |]) ]) );
          ( "Nx.Ptree.scale: the factor has shape [1], expected a scalar",
            fun () ->
              ignore (P.scale packed (vec [| 2. |]) [ Nx.P (vec [| 1. |]) ]) );
          ( "Nx.Ptree.axpy: the factor has shape [2], expected a scalar",
            fun () ->
              ignore
                (P.axpy packed
                   (vec [| 2.; 3. |])
                   [ Nx.P (vec [| 1. |]) ]
                   [ Nx.P (vec [| 1. |]) ]) );
        ]
        (fun (m, f) -> raises (Invalid_argument m) f);
    ]

let () =
  exit
    (run "nx ptree" [ laws; chosen; constructed; fields; errors; arithmetic ])
