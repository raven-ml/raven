(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Archives. A value read back through its structure is the value written, bit
   for bit, over drawn structures of records, lists and options, and through a
   file. Reading back is strict within the names the value owns, and names what
   differs; only [float] converts. *)

open Windtrap
open Io_support
module Archive = Nx_io.Archive
module P = Nx.Ptree

let vec dt xs = Nx.create dt [| Array.length xs |] xs

(* Placements compare as Nx.Placement.equal does. *)
let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
let fails_with message f = raises (Failure message) (fun () -> ignore (f ()))

let refuses message f =
  raises (Invalid_argument message) (fun () -> ignore (f ()))

(* Drawn structures *)

(* A structure and a generator of its values. *)
type desc = D : ('s, Nx.host) P.t * 's Gen.t -> desc

(* A drawn value and its structure. *)
type sample = S : ('s, Nx.host) P.t * 's -> sample

let record (type a b d) (ka, (sa : (a, d) P.t)) (kb, (sb : (b, d) P.t)) :
    (a * b, d) P.t =
  P.v (fun c (x, y) ->
      let open P.Walk in
      let x = field c ka (structure sa) x in
      let y = field c kb (structure sb) y in
      (x, y))

let field k s = P.v (fun c x -> P.Walk.field c k (P.Walk.structure s) x)
let names = [ "w"; "b"; "layer"; "é" ]

let two_names =
  Gen.map
    (function a :: b :: _ -> (a, b) | _ -> assert false)
    (Gen.permutation names)

(* Arrays of every dtype SafeTensors holds, of no element, one or several. *)
let leaf =
  let open Gen in
  let+ (D.Any dt) = of_list safetensors_dtypes in
  D
    ( P.tensor,
      let* kind = of_list [ Empty; Single; Many ] in
      let* s = shape ~rank:3 kind in
      let+ vs =
        array ~size:(constant (Array.fold_left ( * ) 1 s)) (element dt)
      in
      Nx.create dt s vs )

(* Structures at most [depth] deep. A record's names are distinct and hold no
   ["."], so no two arrays share a name. *)
let rec desc depth =
  let open Gen in
  if depth = 0 then leaf
  else
    let sub = desc (depth - 1) in
    frequency
      [
        (2, leaf);
        ( 2,
          let+ ka, kb = two_names
          and+ (D (sa, ga)) = sub
          and+ (D (sb, gb)) = sub in
          D (record (ka, sa) (kb, sb), pair ga gb) );
        ( 1,
          let+ (D (s, g)) = sub in
          D (P.list s, list ~size:(int_range 0 3) g) );
        ( 1,
          let+ (D (s, g)) = sub in
          D (P.option s, option g) );
      ]

(* A value's arrays all have names: its structure is a field or a record. *)
let top =
  let open Gen in
  let sub = desc 2 in
  one_of
    [
      (let+ k = of_list names and+ (D (s, g)) = sub in
       D (field k s, g));
      (let+ ka, kb = two_names
       and+ (D (sa, ga)) = sub
       and+ (D (sb, gb)) = sub in
       D (record (ka, sa) (kb, sb), pair ga gb));
    ]

(* A value compares as its visits and its arrays, bit for bit. *)
let same_visit a b =
  match (a, b) with
  | P.Leaf p, P.Leaf q -> P.Path.equal p q
  | P.Report (p, r), P.Report (q, r') -> P.Path.equal p q && r = r'
  | _ -> false

let visit = Testable.make ~pp:P.pp_visit ~equal:same_visit

let value_of s =
  Testable.contramap
    (fun x -> (P.visits s x, fst (P.flatten s x)))
    (pair (list visit) (list packed))

let samples =
  Gen.with_pp
    (fun ppf (S (s, x)) -> Testable.pp (value_of s) ppf x)
    (Gen.bind top (fun (D (s, g)) -> Gen.map (fun x -> S (s, x)) g))

(* [like s x] has [x]'s visits, dtypes and shapes, and none of its arrays. *)
let like s x = P.map s (fun _ t -> Nx.zeros_like t) x
let arrays s x = fst (P.flatten s x)

(* [covers s x] states the cases a round trip must reach: each dtype, an array
   of no element, an empty list and an absent option. *)
let covers s x =
  let arrays = arrays s x in
  List.iter
    (fun (D.Any dt) ->
      cover (D.name dt)
        (List.exists (fun (Nx.P a) -> D.equal (Nx.dtype a) dt) arrays))
    safetensors_dtypes;
  cover "an array of no element"
    (List.exists (fun (Nx.P a) -> Nx.numel a = 0) arrays);
  let reported r =
    List.exists (function P.Report (_, r') -> r = r' | P.Leaf _ -> false)
  in
  cover "an empty list" (reported (P.Length 0) (P.visits s x));
  cover "an absent option" (reported (P.Present false) (P.visits s x))

(* A model of [n] blocks on the host, for strictness. *)
let blocks n =
  let s = field "blocks" (P.list (record ("w", P.tensor) ("b", P.tensor))) in
  let block i =
    let i = Float.of_int i in
    ( Nx.create Nx.float32 [| 2 |] [| i; i |],
      Nx.create Nx.float32 [| 1 |] [| 0. |] )
  in
  (s, List.init n block)

let values =
  group "values"
    [
      prop "a value read back from its archive is the value" samples
        (fun (S (s, x)) ->
          covers s x;
          equal (value_of s) x
            (Archive.to_value s ~like:(like s x) (Archive.of_value s x)));
      prop "a value read back from a file of its archive is the value" ~count:40
        samples (fun (S (s, x)) ->
          let path = temp_file () in
          Nx_io.save_safetensors path (Archive.of_value s x);
          equal (value_of s) x
            (Archive.to_value s ~like:(like s x) (Nx_io.load_safetensors path)));
      test "an array is named by its path" (fun () ->
          let s, x = blocks 2 in
          equal (list string)
            [ "blocks.0.b"; "blocks.0.w"; "blocks.1.b"; "blocks.1.w" ]
            (Archive.names (Archive.of_value s x)));
      test "a one-block model refuses a file of 24 blocks" (fun () ->
          let s, x = blocks 1 and _, y = blocks 24 in
          fails_with
            "Nx_io.Archive.to_value: blocks.1.b: an entry in the archive, no \
             array in the value" (fun () ->
              Archive.to_value s ~like:x (Archive.of_value s y)));
      test "values under two prefixes share a file, each owning its own"
        (fun () ->
          let s, x = blocks 1 and _, y = blocks 2 in
          let path = temp_file () in
          Nx_io.save_safetensors path
            (Archive.union
               [
                 Archive.of_value ~prefix:"model" s x;
                 Archive.of_value ~prefix:"ema" s y;
               ]);
          let a = Nx_io.load_safetensors path in
          equal (list string)
            [
              "ema.blocks.0.b";
              "ema.blocks.0.w";
              "ema.blocks.1.b";
              "ema.blocks.1.w";
              "model.blocks.0.b";
              "model.blocks.0.w";
            ]
            (Archive.names a);
          equal (value_of s) x (Archive.to_value ~prefix:"model" s ~like:x a);
          equal (value_of s) y (Archive.to_value ~prefix:"ema" s ~like:y a);
          fails_with
            "Nx_io.Archive.to_value: model.blocks.0.b: an entry in the \
             archive, no array in the value" (fun () ->
              Archive.to_value ~prefix:"model" s ~like:(snd (blocks 0)) a));
      test "an array at the root is named by its prefix" (fun () ->
          let x = Nx.create Nx.int8 [| 1 |] [| 3 |] in
          let a = Archive.of_value ~prefix:"w" P.tensor x in
          equal (list string) [ "w" ] (Archive.names a);
          equal packed (Nx.P x)
            (Nx.P (Archive.to_value ~prefix:"w" P.tensor ~like:x a)));
      test "a missing entry, another shape or another dtype is named" (fun () ->
          let s, x = blocks 1 in
          let without =
            Archive.of_list
              [ ("blocks.0.w", Nx.P (Nx.zeros Nx.float32 [| 2 |])) ]
          in
          fails_with
            "Nx_io.Archive.to_value: blocks.0.b: no entry in the archive, an \
             array in the value" (fun () -> Archive.to_value s ~like:x without);
          let shaped =
            Archive.of_list
              [
                ("blocks.0.w", Nx.P (Nx.zeros Nx.float32 [| 3 |]));
                ("blocks.0.b", Nx.P (Nx.zeros Nx.float32 [| 1 |]));
              ]
          in
          fails_with
            "Nx_io.Archive.to_value: blocks.0.w: shape [3] in the archive, [2] \
             in the value" (fun () -> Archive.to_value s ~like:x shaped);
          let typed =
            Archive.of_list
              [
                ("blocks.0.w", Nx.P (Nx.zeros Nx.float64 [| 2 |]));
                ("blocks.0.b", Nx.P (Nx.zeros Nx.float32 [| 1 |]));
              ]
          in
          fails_with
            "Nx_io.Archive.to_value: blocks.0.w: float64 in the archive, \
             float32 in the value" (fun () -> Archive.to_value s ~like:x typed));
      test "two arrays of one name, or one at the root, are refused" (fun () ->
          let one = Nx.zeros Nx.float32 [| 1 |] in
          let s = record ("a.b", P.tensor) ("a", field "b" P.tensor) in
          refuses "Nx_io.Archive.of_value: a.b: two arrays have this name"
            (fun () -> Archive.of_value s (one, one));
          raises_match
            (Exn.invalid_arg
               ~substring:"Nx_io.Archive.of_value: an array at the root")
            (fun () -> Archive.of_value P.tensor one));
    ]

let entries =
  let a =
    Archive.of_list
      [
        ("w", Nx.P (vec Nx.float16 [| 1.5; -2. |]));
        ("q", Nx.P (vec Nx.float8_e4m3fn [| 1. |]));
      ]
  in
  group "entries"
    [
      test "names are sorted, and an entry is found by its name" (fun () ->
          equal (list string) [ "q"; "w" ] (Archive.names a);
          is_none ~pp:pp_packed (Archive.find "absent" a));
      test "a name given twice or empty, or in two archives, is refused"
        (fun () ->
          let x = Nx.P (Nx.zeros Nx.int8 [| 1 |]) in
          refuses "Nx_io.Archive.of_list: \"w\" is named twice" (fun () ->
              Archive.of_list [ ("w", x); ("w", x) ]);
          refuses "Nx_io.Archive.of_list: an empty name" (fun () ->
              Archive.of_list [ ("", x) ]);
          refuses "Nx_io.Archive.union: \"w\" is in two archives" (fun () ->
              Archive.union
                [ Archive.of_list [ ("w", x) ]; Archive.of_list [ ("w", x) ] ]));
      test "tensor is an entry as stored, at its shape and dtype" (fun () ->
          equal packed
            (Option.get (Archive.find "w" a))
            (Nx.P (Archive.tensor ~shape:[| 2 |] Nx.float16 "w" a));
          fails_with
            "Nx_io.Archive.tensor: w: shape [2] in the archive, [3] asked for"
            (fun () -> Archive.tensor ~shape:[| 3 |] Nx.float16 "w" a);
          fails_with
            "Nx_io.Archive.tensor: w: float16 in the archive, float32 asked for"
            (fun () -> Archive.tensor ~shape:[| 2 |] Nx.float32 "w" a);
          fails_with "Nx_io.Archive.tensor: x: no entry in the archive"
            (fun () -> Archive.tensor ~shape:[| 2 |] Nx.float16 "x" a));
      test "float casts between the four float dtypes, and no other" (fun () ->
          equal packed
            (Nx.P (vec Nx.float64 [| 1.5; -2. |]))
            (Nx.P (Archive.float ~shape:[| 2 |] Nx.float64 "w" a));
          equal packed
            (Option.get (Archive.find "w" a))
            (Nx.P (Archive.float ~shape:[| 2 |] Nx.float16 "w" a));
          fails_with
            "Nx_io.Archive.float: q: float8_e4m3fn in the archive, not \
             float16, bfloat16, float32 or float64" (fun () ->
              Archive.float ~shape:[| 1 |] Nx.float32 "q" a);
          raises_match (Exn.invalid_arg ~substring:"Nx_io.Archive.float")
            (fun () -> Archive.float ~shape:[| 1 |] Nx.float8_e4m3fn "q" a));
      test
        "to_value places each entry where the value's array lies, and refuses \
         an array of every set" (fun () ->
          let module M = (val Nx.devices [ Nx_support.memory 1 ]) in
          let s = record ("w", P.tensor) ("b", P.tensor) in
          let w = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]
          and b = Nx.create Nx.float32 [| 1 |] [| 3. |] in
          let host = Archive.of_value s (w, b) in
          let like =
            (Nx.place M.on (Nx.zeros_like w), Nx.place M.on (Nx.zeros_like b))
          in
          let w', b' =
            Archive.to_value (record ("w", P.tensor) ("b", P.tensor)) ~like host
          in
          equal
            (list (option (placement ())))
            [ Some M.on; Some M.on ]
            [ Nx.placement w'; Nx.placement b' ];
          equal (array float_exact) [| 1.; 2. |] (Nx.to_array w');
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "Nx_io.Archive.to_value: b: the array in the value is of \
                  every set") (fun () ->
              Archive.to_value s
                ~like:(Nx.zeros_like w, Nx.zeros Nx.float32 [| 1 |])
                host));
    ]

let () = exit (run "nx.io archives" [ values; entries ])
