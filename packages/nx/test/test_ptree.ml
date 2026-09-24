(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module P = Nx.Ptree

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let int32s xs = Nx.create Nx.int32 [| Array.length xs |] xs
let values (x : Nx.float32_t) = Nx.to_array x

(* Structures *)

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

module Numbered = struct
  type 'a t = { pair : 'a * 'a; arr : 'a array }

  let walk c m =
    let open P.Walk in
    let pair =
      field c "pair"
        (fun c (a, b) ->
          let a = index c 0 leaf a in
          let b = index c 1 leaf b in
          (a, b))
        m.pair
    in
    let arr =
      field c "arr"
        (fun c a ->
          ignore (int c (Array.length a));
          Array.mapi (fun i x -> index c i leaf x) a)
        m.arr
    in
    { pair; arr }
end

type affine = { scale : Nx.float32_t; shift : Nx.float32_t }

module Affine = struct
  type _ t = affine

  let walk c { scale; shift } =
    let open P.Walk in
    let scale = field c "scale" tensor scale in
    let shift = field c "shift" tensor shift in
    { scale; shift }
end

module Packed_list = struct
  type _ t = Nx.packed list

  let walk c l = P.Walk.list (fun c (Nx.P x) -> Nx.P (P.Walk.tensor c x)) c l
end

let mlp : Nx.float32_t Mlp.t P.t = P.instantiate (module Mlp)
let cache : Nx.float32_t Cache.t P.t = P.instantiate (module Cache)

let params ?(window = Some 4) ?(b2 = None) () =
  {
    Mlp.l1 = { w = vec [| 1.; 2. |]; b = Some (vec [| 3. |]) };
    l2 = { w = vec [| 4. |]; b = b2 };
    steps = int32s [| 7l |];
    window;
  }

(* Helpers *)

let paths s x =
  List.rev (P.fold s (fun p _ acc -> P.Path.to_string p :: acc) x [])

let key_lines s x = List.map (Format.asprintf "%a" P.pp_visit) (P.visits s x)

let same_tensor (Nx.P a) (Nx.P b) =
  match Nx_core.Dtype.equal_witness (Nx.dtype a) (Nx.dtype b) with
  | Some Type.Equal -> a == b
  | None -> false

let skeleton s x = snd (P.flatten s x)
let invalid msg f = raises (Invalid_argument msg) f
let same_key s x y = P.Skeleton.equal (skeleton s x) (skeleton s y)

(* Paths *)

let test_paths_follow_the_walk () =
  equal ~msg:"leaf paths" (list string)
    [ "l1.w"; "l1.b"; "l2.w"; "steps" ]
    (paths mlp (params ()));
  let segs = ref [] in
  ignore
    (P.map mlp
       (fun p x ->
         segs := P.Path.segments p :: !segs;
         x)
       (params ()));
  is_true ~msg:"typed segments"
    (List.rev !segs
    = P.Path.
        [
          [ Field "l1"; Field "w" ];
          [ Field "l1"; Field "b" ];
          [ Field "l2"; Field "w" ];
          [ Field "steps" ];
        ])

module Dotted = struct
  type 'a t = 'a

  let walk c x = P.Walk.field c "a.b" P.Walk.leaf x
end

module Nested = struct
  type 'a t = 'a

  let walk c x = P.Walk.(field c "a" (fun c x -> field c "b" leaf x) x)
end

let test_paths_compare_by_segments () =
  let path s =
    Option.get (P.fold s (fun p _ _ -> Some p) (vec [| 1. |]) None)
  in
  let dotted = path (P.instantiate (module Dotted)) in
  let nested = path (P.instantiate (module Nested)) in
  equal ~msg:"print alike" string (P.Path.to_string dotted)
    (P.Path.to_string nested);
  is_false ~msg:"differ" (P.Path.equal dotted nested);
  is_true ~msg:"equal to itself"
    (P.Path.equal nested (path (P.instantiate (module Nested))));
  equal ~msg:"root prints empty" string ""
    (P.fold P.tensor (fun p _ _ -> P.Path.to_string p) (vec [| 1. |]) "?")

let test_index_numbers_parts () =
  let numbered = P.instantiate (module Numbered) in
  let x =
    {
      Numbered.pair = (vec [| 1. |], vec [| 2. |]);
      arr = [| vec [| 3. |]; vec [| 4. |] |];
    }
  in
  equal ~msg:"paths" (list string)
    [ "pair.0"; "pair.1"; "arr.0"; "arr.1" ]
    (paths numbered x);
  equal ~msg:"array length reported" (list string)
    [
      "pair.0: a leaf";
      "pair.1: a leaf";
      "arr: int 2";
      "arr.0: a leaf";
      "arr.1: a leaf";
    ]
    (key_lines numbered x);
  equal ~msg:"pair numbers its sides alike" (list string) [ "0"; "1" ]
    (paths P.(pair tensor tensor) (vec [| 1. |], vec [| 2. |]))

let test_mask_by_path () =
  let frozen p =
    match P.Path.segments p with P.Path.Field "l1" :: _ -> true | _ -> false
  in
  let g =
    P.map mlp (fun p g -> if frozen p then Nx.zeros_like g else g) (params ())
  in
  equal ~msg:"l1.w masked" (array float_exact) [| 0.; 0. |] (values g.l1.w);
  equal ~msg:"l2.w kept" (array float_exact) [| 4. |] (values g.l2.w)

(* Walk order and round trip *)

let test_every_walk_agrees () =
  let x = params () in
  let tensors =
    [ Nx.P x.l1.w; Nx.P (Option.get x.l1.b); Nx.P x.l2.w; Nx.P x.steps ]
  in
  let seen = ref [] in
  ignore
    (P.map mlp
       (fun p t ->
         seen := (P.Path.to_string p, Nx.P t) :: !seen;
         t)
       x);
  let map_order = List.rev !seen in
  let fold_order =
    List.rev
      (P.fold mlp (fun p t acc -> (P.Path.to_string p, Nx.P t) :: acc) x [])
  in
  let leaves, _ = P.flatten mlp x in
  let key_paths =
    List.filter_map
      (function P.Leaf p -> Some (P.Path.to_string p) | P.Report _ -> None)
      (P.visits mlp x)
  in
  let same = List.for_all2 same_tensor in
  is_true ~msg:"map visits the tensors" (same (List.map snd map_order) tensors);
  is_true ~msg:"fold visits the tensors"
    (same (List.map snd fold_order) tensors);
  is_true ~msg:"flatten lists the tensors" (same leaves tensors);
  equal ~msg:"paths agree" (list string) (List.map fst map_order)
    (List.map fst fold_order);
  equal ~msg:"key paths agree" (list string) (List.map fst map_order) key_paths;
  let payload_order =
    List.rev
      (P.Payload.fold
         (module Mlp)
         (fun p _ acc -> P.Path.to_string p :: acc)
         x [])
  in
  equal ~msg:"payload fold visits the parameter's positions" (list string)
    [ "l1.w"; "l1.b"; "l2.w" ] payload_order

let test_round_trip () =
  let x = params () in
  let leaves, _ = P.flatten mlp x in
  let y = P.rebuild mlp ~like:x leaves in
  is_true ~msg:"rebuild keeps each leaf"
    (y.l1.w == x.l1.w
    && Option.get y.l1.b == Option.get x.l1.b
    && y.l2.w == x.l2.w && y.steps == x.steps);
  equal ~msg:"rebuild keeps the data" (option int) x.window y.window;
  let z = P.map mlp (fun _ t -> t) x in
  is_true ~msg:"identity map"
    (z.l1.w == x.l1.w && z.steps == x.steps && z.window = x.window);
  let fresh = List.map (fun (Nx.P t) -> Nx.P (Nx.copy t)) leaves in
  let r = P.rebuild mlp ~like:x fresh in
  equal ~msg:"rebuilt from fresh leaves" (array float_exact) [| 4. |]
    (values r.l2.w);
  is_false ~msg:"fresh leaf" (r.l2.w == x.l2.w)

(* Keys *)

let test_leafless_parts_are_keyed () =
  let c = { Cache.keys = vec [| 1. |]; values = vec [| 2. |] } in
  let caches = P.list (P.option cache) in
  let a = [ Some c; None ] and b = [ Some c ] in
  equal ~msg:"same leaves" (list string) (paths caches a) (paths caches b);
  equal ~msg:"[Some c; None]" (list string)
    [
      "the root: length 2";
      "0: Some";
      "0.keys: a leaf";
      "0.values: a leaf";
      "1: None";
    ]
    (key_lines caches a);
  equal ~msg:"[Some c]" (list string)
    [ "the root: length 1"; "0: Some"; "0.keys: a leaf"; "0.values: a leaf" ]
    (key_lines caches b);
  is_false ~msg:"keys differ" (same_key caches a b);
  is_false ~msg:"option unit" (same_key P.(option unit) (Some ()) None);
  is_false ~msg:"list unit" (same_key P.(list unit) [ (); () ] [ () ])

let test_reports_are_keyed () =
  let weights = P.list (P.instantiate (module Weight)) in
  let u8 () = Nx.zeros Nx.uint8 [| 2 |] in
  equal ~msg:"case tags" (list string)
    [
      "the root: length 2";
      "0: case \"float\"";
      "0: a leaf";
      "1: case \"mxfp4\"";
      "1.blocks: a leaf";
      "1.scales: a leaf";
    ]
    (key_lines weights
       [ Float (vec [| 1. |]); Mxfp4 { blocks = u8 (); scales = u8 () } ]);
  equal ~msg:"int" (list string)
    [
      "l1.w: a leaf";
      "l1.b: Some";
      "l1.b: a leaf";
      "l2.w: a leaf";
      "l2.b: None";
      "steps: a leaf";
      "window: Some";
      "window: int 4";
    ]
    (key_lines mlp (params ()));
  is_false ~msg:"window" (same_key mlp (params ()) (params ~window:(Some 8) ()));
  is_true ~msg:"equal data" (same_key mlp (params ()) (params ()));
  equal ~msg:"equal hashes" int
    (P.Skeleton.hash (skeleton mlp (params ())))
    (P.Skeleton.hash (skeleton mlp (params ())))

let test_skeletons_compare_paths () =
  let x = vec [| 1. |] in
  let dotted : Nx.float32_t Dotted.t P.t = P.instantiate (module Dotted) in
  let nested : Nx.float32_t Nested.t P.t = P.instantiate (module Nested) in
  let kd = skeleton dotted x and kn = skeleton nested x in
  equal ~msg:"equal hashes" int (P.Skeleton.hash kd) (P.Skeleton.hash kn);
  is_false ~msg:"skeletons differ" (P.Skeleton.equal kd kn);
  let module Either_path = struct
    type 'a t = Dotted of 'a | Nested of 'a

    let walk c = function
      | Dotted x -> Dotted (P.Walk.field c "a.b" P.Walk.leaf x)
      | Nested x -> Nested (Nested.walk c x)
  end in
  let either = P.instantiate (module Either_path) in
  invalid
    "Nx.Ptree.map2: [\"a.b\"]: a leaf in the first value, a leaf at [\"a\"; \
     \"b\"] in the second" (fun () ->
      ignore (P.map2 either (fun _ a _ -> a) (Dotted x) (Nested (vec [| 2. |]))))

let test_skeleton_diff () =
  let diff x y =
    P.Skeleton.diff ~this:"here" (skeleton mlp x) ~that:"in the previous key"
      (skeleton mlp y)
  in
  equal ~msg:"equal" (option string) None (diff (params ()) (params ()));
  equal ~msg:"window" (option string)
    (Some "window: int 8 here, int 4 in the previous key")
    (diff (params ~window:(Some 8) ()) (params ()));
  equal ~msg:"presence" (option string)
    (Some "window: None here, Some in the previous key")
    (diff (params ~window:None ()) (params ()));
  let x = vec [| 1. |] in
  let dotted : Nx.float32_t Dotted.t P.t = P.instantiate (module Dotted) in
  let nested : Nx.float32_t Nested.t P.t = P.instantiate (module Nested) in
  equal ~msg:"paths that print alike" (option string)
    (Some "[\"a.b\"]: a leaf here, a leaf at [\"a\"; \"b\"] there")
    (P.Skeleton.diff ~this:"here" (skeleton dotted x) ~that:"there"
       (skeleton nested x))

(* Structures of structures *)

let test_nest () =
  let adam = P.nest (module Adam) mlp in
  let state = { Adam.mu = params (); nu = params (); step = int32s [| 0l |] } in
  equal ~msg:"paths" (list string)
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
    (paths adam state);
  let sum = P.map2 adam (fun _ a b -> Nx.add a b) state state in
  equal ~msg:"map2" (array float_exact) [| 2.; 4. |] (values sum.mu.l1.w);
  equal ~msg:"map2 step" (array int32) [| 0l |] (Nx.to_array sum.step);
  equal ~msg:"state of a pair" (list string)
    [ "mu.0"; "mu.1"; "nu.0"; "nu.1"; "step" ]
    (paths
       (P.nest (module Adam) P.(pair tensor tensor))
       {
         Adam.mu = (vec [| 1. |], vec [| 2. |]);
         nu = (vec [| 1. |], vec [| 2. |]);
         step = int32s [| 0l |];
       })

let test_iso () =
  let module Out = struct
    type t = { loss : Nx.float32_t; params : Nx.float32_t Mlp.t }
  end in
  let out =
    P.iso
      (fun (loss, params) -> { Out.loss; params })
      (fun o -> (o.Out.loss, o.Out.params))
      P.(pair tensor mlp)
  in
  let o = { Out.loss = vec [| 0.5 |]; params = params () } in
  equal ~msg:"keeps the pair's paths" (list string)
    [ "0"; "1.l1.w"; "1.l1.b"; "1.l2.w"; "1.steps" ]
    (paths out o);
  let doubled = P.map out (fun _ t -> Nx.add t t) o in
  equal ~msg:"rebuilds the record" (array float_exact) [| 1. |]
    (values doubled.loss);
  let leaves, _ = P.flatten out o in
  let r = P.rebuild out ~like:o leaves in
  is_true ~msg:"round trip" (r.loss == o.loss && r.params.l2.w == o.params.l2.w)

type out = { loss : Nx.float32_t; model : Nx.float32_t Mlp.t }

let out =
  P.iso
    (fun (loss, model) -> { loss; model })
    (fun o -> (o.loss, o.model))
    P.(pair tensor mlp)

module Parts = struct
  type 'a t = {
    w : 'a;
    both : Nx.float32_t Mlp.t * Nx.float32_t Cache.t;
    out : out;
    caches : Nx.float32_t Cache.t list;
  }

  let walk c p =
    let open P.Walk in
    let w = field c "w" leaf p.w in
    let both = field c "both" (structure P.(pair mlp cache)) p.both in
    let out = field c "out" (structure out) p.out in
    let caches = field c "caches" (structure P.(list cache)) p.caches in
    { w; both; out; caches }
end

let test_structure () =
  let kv () = { Cache.keys = vec [| 1. |]; values = vec [| 2. |] } in
  let x =
    {
      Parts.w = vec [| 5. |];
      both = (params (), kv ());
      out =
        { loss = vec [| 0.5 |]; model = params ~b2:(Some (vec [| 6. |])) () };
      caches = [ kv (); kv () ];
    }
  in
  let parts = P.instantiate (module Parts) in
  equal ~msg:"visits" (list string)
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
      "caches.0.keys: a leaf";
      "caches.0.values: a leaf";
      "caches.1.keys: a leaf";
      "caches.1.values: a leaf";
    ]
    (key_lines parts x);
  let leaves, _ = P.flatten parts x in
  is_true ~msg:"round trip"
    (List.for_all2 same_tensor leaves
       (fst (P.flatten parts (P.rebuild parts ~like:x leaves))));
  is_true ~msg:"map with the identity"
    (List.for_all2 same_tensor leaves
       (fst (P.flatten parts (P.map parts (fun _ t -> t) x))));
  let y = P.cast (module Parts) Nx.bfloat16 x in
  is_true ~msg:"cast casts the parameter" (Nx.dtype y.w = Nx.bfloat16);
  is_true ~msg:"cast keeps the structures' tensors"
    ((fst y.both).l1.w == (fst x.both).l1.w
    && y.out.loss == x.out.loss
    && (List.nth y.caches 1).values == (List.nth x.caches 1).values);
  let z = P.Payload.map (module Parts) (fun p _ -> P.Path.to_string p) x in
  equal ~msg:"Payload.map changes the payload" string "w" z.w;
  is_true ~msg:"Payload.map keeps the structures' tensors"
    ((snd z.both).keys == (snd x.both).keys
    && z.out.model.l2.w == x.out.model.l2.w
    && (List.hd z.caches).keys == (List.hd x.caches).keys)

let test_monomorphic () =
  let affine = P.instantiate (module Affine) in
  let x = { scale = vec [| 2. |]; shift = vec [| 1. |] } in
  equal ~msg:"paths" (list string) [ "scale"; "shift" ] (paths affine x);
  let y = P.map affine (fun _ t -> Nx.add t t) x in
  equal ~msg:"map" (array float_exact) [| 4. |] (values y.scale)

(* Payloads *)

let test_cast_keeps_fixed_tensors () =
  let x = params () in
  let y = P.cast (module Mlp) Nx.bfloat16 x in
  is_true ~msg:"leaves cast" (Nx.dtype y.l1.w = Nx.bfloat16);
  is_true ~msg:"steps kept" (y.steps == x.steps);
  equal ~msg:"steps int32" (array int32) [| 7l |] (Nx.to_array y.steps);
  equal ~msg:"window kept" (option int) (Some 4) y.window

module Symo = struct
  type 'a t = { b : 'a; w : 'a; c : 'a; bias : 'a }

  let walk cur { b; w; c; bias } =
    let open P.Walk in
    let b = field cur "b" leaf b in
    let w = field cur "w" leaf w in
    let c = field cur "c" leaf c in
    let bias = field cur "bias" leaf bias in
    { b; w; c; bias }
end

type spec = Id | Perm of int

let test_payloads () =
  let dims : int list Symo.t =
    { b = [ 16; 128 ]; w = [ 128; 128 ]; c = [ 128; 32 ]; bias = [ 128 ] }
  in
  let symmetries : spec list Symo.t =
    {
      b = [ Id; Perm 0 ];
      w = [ Perm 0; Perm 0 ];
      c = [ Perm 0; Id ];
      bias = [ Perm 0 ];
    }
  in
  let permuted : int list Symo.t =
    P.Payload.map2
      (module Symo)
      (fun _ specs dims ->
        List.filteri (fun i _ -> List.nth specs i <> Id) dims)
      symmetries dims
  in
  equal ~msg:"zip" (list int) [ 128 ] permuted.b;
  equal ~msg:"zip w" (list int) [ 128; 128 ] permuted.w;
  let sizes : int Symo.t =
    P.Payload.map (module Symo) (fun _ ds -> List.fold_left ( * ) 1 ds) dims
  in
  equal ~msg:"sizes" int 2048 sizes.b;
  equal ~msg:"total" int
    (2048 + 16384 + 4096 + 128)
    (P.Payload.fold (module Symo) (fun _ n acc -> n + acc) sizes 0);
  let names =
    P.Payload.map (module Symo) (fun p _ -> P.Path.to_string p) sizes
  in
  equal ~msg:"names" string "bias" names.bias;
  let closures =
    P.Payload.map (module Symo) (fun _ n x -> Nx.mul_s x (Float.of_int n)) sizes
  in
  equal ~msg:"closures" (array float_exact) [| 128. |]
    (values (closures.bias (vec [| 1. |])));
  let factors : Nx.float32_t list Symo.t =
    P.Payload.map
      (module Symo)
      (fun _ ds -> List.map (fun d -> Nx.zeros f32 [| d |]) ds)
      dims
  in
  equal ~msg:"lists of tensors" (list int) [ 128; 32 ]
    (List.map (fun t -> (Nx.shape t).(0)) factors.c)

let test_payload_map2_keeps_first_fixed () =
  let x = params () and y = params () in
  let z = P.Payload.map2 (module Mlp) (fun _ a _ -> a) x y in
  is_true ~msg:"first value's fixed tensor" (z.steps == x.steps)

(* Errors *)

let test_map2_errors () =
  invalid "Nx.Ptree.map2: l2.b: Some in the first value, None in the second"
    (fun () ->
      ignore
        (P.map2 mlp
           (fun _ a _ -> a)
           (params ~b2:(Some (vec [| 1. |])) ())
           (params ())));
  invalid "Nx.Ptree.map2: window: int 4 in the first value, int 8 in the second"
    (fun () ->
      ignore
        (P.map2 mlp (fun _ a _ -> a) (params ()) (params ~window:(Some 8) ())));
  let weights = P.list (P.instantiate (module Weight)) in
  let u8 () = Nx.zeros Nx.uint8 [| 2 |] in
  invalid
    "Nx.Ptree.map2: 1: case \"mxfp4\" in the first value, case \"float\" in \
     the second" (fun () ->
      ignore
        (P.map2 weights
           (fun _ a _ -> a)
           [ Float (vec [| 1. |]); Mxfp4 { blocks = u8 (); scales = u8 () } ]
           [ Float (vec [| 1. |]); Float (vec [| 2. |]) ]));
  invalid
    "Nx.Ptree.map2: the root: length 2 in the first value, length 1 in the \
     second" (fun () ->
      ignore
        (P.map2
           P.(list tensor)
           (fun _ a _ -> a)
           [ vec [| 1. |]; vec [| 2. |] ]
           [ vec [| 1. |] ]));
  let packed = P.instantiate (module Packed_list) in
  invalid "Nx.Ptree.map2: 0: float32 in the first value, int32 in the second"
    (fun () ->
      ignore
        (P.map2 packed
           (fun _ a _ -> a)
           [ Nx.P (vec [| 1. |]) ]
           [ Nx.P (int32s [| 1l |]) ]))

let test_map2_walked_two_ways () =
  let module Flaky = struct
    type 'a t = 'a

    let calls = ref 0

    let walk c x =
      incr calls;
      P.Walk.field c (if !calls = 2 then "a" else "b") P.Walk.leaf x
  end in
  let flaky = P.instantiate (module Flaky) in
  invalid "Nx.Ptree.map2: the structure's walk visited one value two ways"
    (fun () ->
      ignore (P.map2 flaky (fun _ a _ -> a) (vec [| 1. |]) (vec [| 2. |])))

let test_payload_map2_errors () =
  invalid
    "Nx.Ptree.Payload.map2: b: Some in the first value, None in the second"
    (fun () ->
      ignore
        (P.Payload.map2
           (module Linear)
           (fun _ a _ -> a)
           { Linear.w = 1; b = Some 2 }
           { Linear.w = 1; b = None }));
  invalid
    "Nx.Ptree.Payload.map2: the root: length 1 in the first value, length 2 in \
     the second" (fun () ->
      ignore
        (P.Payload.map2
           (module Packed_list)
           (fun _ a _ -> a)
           [ Nx.P (vec [| 1. |]) ]
           [ Nx.P (vec [| 1. |]); Nx.P (vec [| 2. |]) ]))

let test_rebuild_errors () =
  let x = params () in
  let leaves, _ = P.flatten mlp x in
  invalid
    "Nx.Ptree.rebuild: steps: a leaf in the template, none left of the 3 given"
    (fun () ->
      ignore (P.rebuild mlp ~like:x (List.filteri (fun i _ -> i < 3) leaves)));
  invalid "Nx.Ptree.rebuild: the template has 4 leaves, 5 given" (fun () ->
      ignore (P.rebuild mlp ~like:x (leaves @ [ List.hd leaves ])));
  let swapped =
    List.mapi (fun i l -> if i = 3 then List.hd leaves else l) leaves
  in
  invalid "Nx.Ptree.rebuild: steps: int32 in the template, float32 given"
    (fun () -> ignore (P.rebuild mlp ~like:x swapped))

let test_unpack () =
  let p = Nx.P (vec [| 1.; 2. |]) in
  equal ~msg:"unpack" (array float_exact) [| 1.; 2. |]
    (values (Nx.unpack f32 p));
  invalid "unpack: expected dtype float64, got float32" (fun () ->
      ignore (Nx.unpack Nx.float64 p))

(* Signatures *)

let test_signatures () =
  let caches = P.list cache in
  let signature :
      (Nx.int32_t ->
      Nx.float32_t Cache.t list ->
      Nx.int32_t * Nx.float32_t Cache.t list)
      P.fn =
    P.(tensor @-> consumes caches @@ returns (pair tensor caches))
  in
  let rec roles : type f. f P.fn -> P.role list = function
    | P.Returns _ -> []
    | P.Arg (role, _, rest) -> role :: roles rest
  in
  is_true ~msg:"roles" (roles signature = [ P.Read; P.Consumed ])

let tests =
  [
    group "paths"
      [
        test "follow the walk" test_paths_follow_the_walk;
        test "compare by segments" test_paths_compare_by_segments;
        test "index numbers parts" test_index_numbers_parts;
        test "a mask is a pattern match" test_mask_by_path;
      ];
    group "walks"
      [
        test "every walk agrees on the order" test_every_walk_agrees;
        test "round trip" test_round_trip;
      ];
    group "skeletons"
      [
        test "leafless parts are keyed" test_leafless_parts_are_keyed;
        test "cases, lengths and ints are keyed" test_reports_are_keyed;
        test "skeletons compare paths" test_skeletons_compare_paths;
        test "skeleton diff" test_skeleton_diff;
      ];
    group "structures"
      [
        test "nest" test_nest;
        test "iso" test_iso;
        test "monomorphic" test_monomorphic;
        test "a part walked with structure" test_structure;
      ];
    group "payloads"
      [
        test "cast keeps fixed tensors" test_cast_keeps_fixed_tensors;
        test "maps at any payload" test_payloads;
        test "map2 keeps the first value's fixed tensors"
          test_payload_map2_keeps_first_fixed;
      ];
    group "errors"
      [
        test "map2" test_map2_errors;
        test "map2 on a walk that visits two ways" test_map2_walked_two_ways;
        test "Payload.map2" test_payload_map2_errors;
        test "rebuild" test_rebuild_errors;
        test "unpack" test_unpack;
      ];
    group "signatures" [ test "roles" test_signatures ];
  ]

let () = run "Nx.Ptree" tests
