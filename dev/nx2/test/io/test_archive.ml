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

(* Arrays of [dtypes], of no element, one or several. *)
let leaf dtypes =
  let open Gen in
  let+ (D.Any dt) = of_list dtypes in
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
let rec desc dtypes depth =
  let open Gen in
  if depth = 0 then leaf dtypes
  else
    let sub = desc dtypes (depth - 1) in
    frequency
      [
        (2, leaf dtypes);
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
let top dtypes =
  let open Gen in
  let sub = desc dtypes 2 in
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

(* Values of drawn structures whose arrays are of [dtypes]. *)
let samples dtypes =
  Gen.with_pp
    (fun ppf (S (s, x)) -> Testable.pp (value_of s) ppf x)
    (Gen.bind (top dtypes) (fun (D (s, g)) -> Gen.map (fun x -> S (s, x)) g))

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

(* A record of two arrays whose names print alike: the field "a.b", and the
   field "b" of the field "a". *)
let two = record ("a.b", P.tensor) ("a", field "b" P.tensor)
let one = Nx.create Nx.float32 [| 1 |] [| 0. |]

(* A model of two blocks and a scale, read under the prefix "model". *)
let model_on () = record ("blocks", P.list P.tensor) ("scale", P.tensor)
let model = model_on ()

let model_value =
  ( [ vec Nx.float32 [| 0.; -0. |]; vec Nx.float32 [| 1.; -0. |] ],
    Nx.create Nx.float64 [||] [| 0.5 |] )

(* [model_with extra] is the model's entries under "model", then [extra]. *)
let model_with extra =
  Archive.union
    [
      Archive.of_value ~prefix:"model" model model_value; Archive.of_list extra;
    ]

let ownership =
  let msg = ( ^ ) "Nx_io.Archive.to_value: " in
  let w = Nx.P (vec Nx.float32 [| 1. |]) in
  group "ownership"
    [
      cases ~name:fst
        "an entry the value owns and none of its arrays names is refused, \
         naming it"
        [
          ( "an entry at the prefix itself",
            ( "model",
              msg "model: an entry in the archive, no array in the value" ) );
          ( "an entry below an array",
            ( "model.scale.x",
              msg
                "model.scale.x: an entry in the archive, no array in the value"
            ) );
        ]
        (fun (_, (name, message)) ->
          fails_with message (fun () ->
              Archive.to_value ~prefix:"model" model ~like:model_value
                (model_with [ (name, w) ])));
      test "the empty prefix owns every name" (fun () ->
          let a =
            Archive.union
              [
                Archive.of_value model model_value;
                Archive.of_list [ ("optim.step", w) ];
              ]
          in
          fails_with
            (msg "optim.step: an entry in the archive, no array in the value")
            (fun () -> Archive.to_value model ~like:model_value a));
      test "entries outside the prefix are ignored, however close their names"
        (fun () ->
          let a =
            model_with
              [
                ("optim.step", w);
                ("models.w", w);
                ("mode", w);
                ("model_2.blocks.0", w);
              ]
          in
          equal (value_of model) model_value
            (Archive.to_value ~prefix:"model" model ~like:model_value a));
      test "to_value fails before it places an entry" (fun () ->
          let device = Nx_support.memory 1 in
          let module M = (val Nx.devices [ device ]) in
          let blocks, scale = model_value in
          let like = (List.map (Nx.place M.on) blocks, Nx.place M.on scale) in
          (* Donated host entries, which a placement consumes: the scale, read
             last, has another shape. *)
          let block () = Nx.full Nx.float32 [| 2 |] 1. in
          let entries b0 b1 scale =
            Archive.of_list
              [
                ("blocks.0", Nx.P (Nx.donate b0));
                ("blocks.1", Nx.P (Nx.donate b1));
                ("scale", Nx.P scale);
              ]
          in
          let b0 = block () and b1 = block () in
          let failing () =
            Archive.to_value (model_on ()) ~like
              (entries b0 b1 (Nx.full Nx.float64 [| 2 |] 0.5))
          in
          (* One uncounted run first: what the device allocates on its first use
             is not the read's. *)
          (try ignore (failing ()) with Failure _ -> ());
          let before = settled device in
          fails_with
            (msg "scale: shape [2] in the archive, [] in the value")
            failing;
          equal ~msg:"bytes placed by the failed read" int 0
            (Rig.allocated device - before);
          equal ~msg:"entries placed by the failed read" (array float_exact)
            [| 1.; 1.; 1.; 1. |]
            (Array.append (Nx.to_array b0) (Nx.to_array b1));
          let x =
            Archive.to_value (model_on ()) ~like
              (entries b0 b1 (Nx.full Nx.float64 [||] 0.5))
          in
          raises_match
            (Exn.invalid_arg ~substring:"was donated to")
            (fun () -> Nx.to_array b0);
          equal (array float_exact) [| 1.; 1. |]
            (Nx.to_array (List.nth (fst x) 1)));
    ]

let values =
  group "values"
    [
      prop "a value read back from its archive is the value"
        (samples safetensors_dtypes) (fun (S (s, x)) ->
          covers s x;
          Law.round_trip (value_of s) archive (Archive.of_value s)
            (Archive.to_value s ~like:(like s x))
            x);
      prop
        "a value read back from a SafeTensors file of its archive is the value"
        ~count:40 (samples safetensors_dtypes) (fun (S (s, x)) ->
          Law.round_trip (value_of s) string
            (fun x ->
              saved ".safetensors" Nx_io.save_safetensors (Archive.of_value s x))
            (fun path ->
              Archive.to_value s ~like:(like s x) (Nx_io.load_safetensors path))
            x);
      prop "a value read back from an npz file of its archive is the value"
        ~count:40 (samples npy_dtypes) (fun (S (s, x)) ->
          Law.round_trip (value_of s) string
            (fun x -> saved ".npz" Nx_io.save_npz (Archive.of_value s x))
            (fun path ->
              Archive.to_value s ~like:(like s x) (Nx_io.load_npz path))
            x);
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
      cases ~name:fst
        "two arrays of one name, or one at the root without a prefix, are \
         refused"
        [
          ( "of_value, two arrays of one name",
            ( "Nx_io.Archive.of_value: a.b: two arrays have this name",
              fun () -> ignore (Archive.of_value two (one, one)) ) );
          ( "to_value, two arrays of one name",
            ( "Nx_io.Archive.to_value: a.b: two arrays have this name",
              fun () ->
                ignore
                  (Archive.to_value two ~like:(one, one)
                     (Archive.of_list [ ("a.b", Nx.P one) ])) ) );
          ( "of_value, an array at the root",
            ( "Nx_io.Archive.of_value: an array at the root has no name; give \
               it a ~prefix",
              fun () -> ignore (Archive.of_value P.tensor one) ) );
          ( "to_value, an array at the root",
            ( "Nx_io.Archive.to_value: an array at the root has no name; give \
               it a ~prefix",
              fun () ->
                ignore
                  (Archive.to_value P.tensor ~like:one (Archive.of_list [])) )
          );
        ]
        (fun (_, (message, f)) -> refuses message f);
    ]

(* Float casts *)

type float_dtype = F : (float, 's) D.t -> float_dtype

let float_dtypes = [ F D.Float16; F D.Bfloat16; F D.Float32; F D.Float64 ]

(* One law per pair of the four float dtypes, over drawn elements: NaN, -0., the
   infinities and, from a wider dtype, finite values past the narrower's
   range. *)
let float_casts =
  List.concat_map
    (fun (F src) ->
      List.map
        (fun (F dst) ->
          let narrows =
            (D.float_format src).max_finite > (D.float_format dst).max_finite
          in
          let elements =
            Gen.with_pp
              (fun ppf vs ->
                Format.pp_print_seq
                  ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
                  (D.pp_value src) ppf (Array.to_seq vs))
              (Gen.array ~size:(Gen.int_range 1 8) (element src))
          in
          prop
            (D.name src ^ " to " ^ D.name dst)
            elements
            (fun vs ->
              let n = Array.length vs in
              let x = Nx.create src [| n |] vs in
              let cast = Nx.cast dst x in
              cover "NaN" (Array.exists Float.is_nan vs);
              cover "-0."
                (Array.exists (fun v -> v = 0. && Float.sign_bit v) vs);
              cover "an infinity"
                (Array.exists (fun v -> Float.abs v = Float.infinity) vs);
              if narrows then
                cover "a finite value past the range"
                  (Array.exists2
                     (fun v w -> Float.is_finite v && not (Float.is_finite w))
                     vs (Nx.to_array cast));
              let a = Archive.of_list [ ("x", Nx.P x) ] in
              equal packed (Nx.P cast)
                (Nx.P (Archive.float ~shape:[| n |] dst "x" a))))
        float_dtypes)
    float_dtypes

let entries =
  let a =
    Archive.of_list
      [
        ("w", Nx.P (vec Nx.float16 [| 1.5; -2. |]));
        ("q", Nx.P (vec Nx.float8_e4m3fn [| 1. |]));
        ("count", Nx.P (vec Nx.int32 [| 7l |]));
      ]
  in
  group "entries"
    [
      test "names are sorted, and an entry is found by its name" (fun () ->
          equal (list string) [ "count"; "q"; "w" ] (Archive.names a);
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
      cases ~name:fst "float refuses what it does not convert, naming it"
        [
          ( "an integer entry",
            ( Failure
                "Nx_io.Archive.float: count: int32 in the archive, not \
                 float16, bfloat16, float32 or float64",
              fun () -> Archive.float ~shape:[| 1 |] Nx.float32 "count" a ) );
          ( "an 8-bit float entry",
            ( Failure
                "Nx_io.Archive.float: q: float8_e4m3fn in the archive, not \
                 float16, bfloat16, float32 or float64",
              fun () -> Archive.float ~shape:[| 1 |] Nx.float32 "q" a ) );
          ( "a missing entry",
            ( Failure "Nx_io.Archive.float: absent: no entry in the archive",
              fun () -> Archive.float ~shape:[| 1 |] Nx.float32 "absent" a ) );
          ( "another shape",
            ( Failure
                "Nx_io.Archive.float: w: shape [2] in the archive, [3] asked \
                 for",
              fun () -> Archive.float ~shape:[| 3 |] Nx.float32 "w" a ) );
        ]
        (fun (_, (e, f)) -> raises e (fun () -> ignore (f ())));
      test "float to a dtype outside the four is refused" (fun () ->
          refuses
            "Nx_io.Archive.float: float8_e4m3fn is not float16, bfloat16, \
             float32 or float64" (fun () ->
              Archive.float ~shape:[| 1 |] Nx.float8_e4m3fn "q" a));
      test
        "float at the entry's dtype is the entry as stored, over the file's \
         pages" (fun () ->
          let a =
            Nx_io.load_safetensors
              (saved "" Nx_io.save_safetensors
                 (Archive.of_list
                    [ ("w", Nx.P (vec Nx.float32 [| 1.; -2. |])) ]))
          in
          let w = Nx.P (Archive.float ~shape:[| 2 |] Nx.float32 "w" a) in
          equal packed (find a "w") w;
          equal memory Over_file (memory_of w));
      group "float agrees with Nx.cast" float_casts;
      test "to_value places each entry where the value's array lies"
        (fun () ->
          let module M = (val Nx.devices [ Nx_support.memory 1 ]) in
          let s = record ("w", P.tensor) ("b", P.tensor) in
          let w = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]
          and b = Nx.create Nx.float32 [| 1 |] [| 3. |] in
          let host = Archive.of_value s (w, b) in
          let like =
            ( Nx.place M.on (Nx.zeros_like w),
              Nx.zeros_at M.on Nx.float32 [| 1 |] )
          in
          let w', b' =
            Archive.to_value (record ("w", P.tensor) ("b", P.tensor)) ~like host
          in
          equal
            (list (placement ()))
            [ M.on; M.on ]
            [ Nx.placement w'; Nx.placement b' ];
          equal (array float_exact) [| 1.; 2. |] (Nx.to_array w');
          equal (array float_exact) [| 3. |] (Nx.to_array b'));
    ]

let () = exit (run "nx.io archives" [ values; ownership; entries ])
