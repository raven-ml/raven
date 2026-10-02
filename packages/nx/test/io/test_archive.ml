(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Archives. A value read back through its structure is the value written, bit
   for bit, over drawn structures of records, lists, options and fields, every
   dtype a format stores and tensors with no elements. Reading back is strict
   within the structure's prefix and names what differs before any byte is read;
   only [float] converts. *)

open Windtrap
open Nx_test
open Stored
module Archive = Nx_io.Archive
module P = Nx.Ptree

let f32 = Nx.float32
let f64 = Nx.float64
let vec dtype xs = Nx.create dtype [| Array.length xs |] xs

(* An archive compares as its named tensors, in name order. *)
let listed a =
  List.map (fun n -> (n, Option.get (Archive.find n a))) (Archive.names a)

let archive = Testable.contramap listed (list (pair string packed))

let saved save a =
  let path = temp_file () in
  save path a;
  path

(* Bytes the host has read from files since the disk device started. *)
let disk_reads () = Nx_device.Stats.bytes_out (Nx_device.stats Nx_device.disk)
let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
let fails message f = raises (Failure message) (fun () -> ignore (f ()))

let refuses message f =
  raises (Invalid_argument message) (fun () -> ignore (f ()))

(* Drawn structures *)

(* A structure and a generator of its values. *)
type desc = D : 's P.t * 's Gen.t -> desc

(* A drawn value and its structure. *)
type sample = S : 's P.t * 's -> sample

(* [record (ka, sa) (kb, sb)] is the structure of a record of two fields named
   [ka] and [kb], walked by [sa] and [sb]. *)
let record (type a b) (ka, (sa : a P.t)) (kb, (sb : b P.t)) : (a * b) P.t =
  let module R = struct
    type _ t = a * b

    let walk c (x, y) =
      let open P.Walk in
      let x = field c ka (structure sa) x in
      let y = field c kb (structure sb) y in
      (x, y)
  end in
  P.nest (module R) P.unit

let names = [ "w"; "b"; "layer"; "é" ]

let two_names =
  Gen.map
    (function a :: b :: _ -> (a, b) | _ -> assert false)
    (Gen.permutation names)

(* Structures at most [depth] deep whose tensors are of [cases]' dtypes. A
   record's names are distinct and contain no ["."], so no two tensors share a
   name. *)
let rec desc cases depth =
  let open Gen in
  let leaf = map (fun (Case c) -> D (P.tensor, c.tensors)) (of_list cases) in
  if depth = 0 then leaf
  else
    let sub = desc cases (depth - 1) in
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
        ( 1,
          let+ k = of_list names and+ (D (s, g)) = sub in
          D (P.field k s, g) );
      ]

(* A value's tensors all have names: its structure is a section or a record. *)
let top cases =
  let open Gen in
  let sub = desc cases 2 in
  one_of
    [
      (let+ k = of_list names and+ (D (s, g)) = sub in
       D (P.field k s, g));
      (let+ ka, kb = two_names
       and+ (D (sa, ga)) = sub
       and+ (D (sb, gb)) = sub in
       D (record (ka, sa) (kb, sb), pair ga gb));
    ]

(* A value compares as its visits and its tensors, bit for bit. *)
let same_visit a b =
  match (a, b) with
  | P.Leaf p, P.Leaf q -> P.Path.equal p q
  | P.Report (p, r), P.Report (q, r') -> P.Path.equal p q && r = r'
  | _ -> false

let visit = Testable.make ~pp:P.pp_visit ~equal:same_visit

let value s =
  Testable.contramap
    (fun x -> (P.visits s x, fst (P.flatten s x)))
    (pair (list visit) (list packed))

let pp_sample ppf (S (s, x)) = Testable.pp (value s) ppf x

let samples cases =
  Gen.with_pp pp_sample
    (Gen.bind (top cases) (fun (D (s, g)) -> Gen.map (fun x -> S (s, x)) g))

(* [like s x] has [x]'s visits, dtypes and shapes, and none of its tensors. *)
let like s x = P.map s (fun _ t -> Nx.zeros_like t) x

let covers cases s x =
  let tensors = fst (P.flatten s x) in
  let dtypes = List.map (fun (Nx.P t) -> Nx_dtype.to_string (Nx.dtype t)) in
  List.iter
    (fun (Case c) ->
      let name = Nx_dtype.to_string c.dtype in
      cover name (List.mem name (dtypes tensors)))
    cases;
  cover "a tensor with no elements"
    (List.exists (fun (Nx.P t) -> Nx.numel t = 0) tensors);
  let reported r =
    List.exists (function P.Report (_, r') -> r = r' | _ -> false)
  in
  cover "an empty list" (reported (P.Length 0) (P.visits s x));
  cover "an absent option" (reported (P.Present false) (P.visits s x))

let safetensors_cases =
  (bool :: ints)
  @ [ float16; bfloat16; float32; float64; float8_e4m3; float8_e5m2 ]

let npz_cases =
  (bool :: ints) @ [ float16; float32; float64; complex64; complex128 ]

let all_cases =
  (bool :: ints)
  @ [
      float16;
      bfloat16;
      float32;
      float64;
      float8_e4m3;
      float8_e5m2;
      complex64;
      complex128;
    ]

let round_trips =
  let through name cases save load =
    prop ~count:200
      ("a value read back from " ^ name ^ " is the value written")
      (samples cases)
      (fun (S (s, x)) ->
        covers cases s x;
        Law.round_trip (value s) string
          (fun x -> saved save (Archive.of_value s x))
          (fun path -> Archive.to_value s ~like:(like s x) (load path))
          x)
  in
  group "round trip"
    [
      prop ~count:200 "a value read back from its archive is the value"
        (samples all_cases) (fun (S (s, x)) ->
          covers all_cases s x;
          Law.round_trip (value s) archive (Archive.of_value s)
            (Archive.to_value s ~like:(like s x))
            x);
      through "SafeTensors" safetensors_cases Nx_io.save_safetensors
        Nx_io.load_safetensors;
      through "NPZ" npz_cases Nx_io.save_npz Nx_io.load_npz;
      prop "of_value names each tensor by its path" (samples all_cases)
        (fun (S (s, x)) ->
          let paths =
            P.fold s (fun p _ acc -> P.Path.to_string p :: acc) x []
          in
          equal (list string) (List.sort compare paths)
            (Archive.names (Archive.of_value s x));
          equal ~msg:"under a field" (list string)
            (List.sort compare (List.map (fun p -> "s." ^ p) paths))
            (Archive.names (Archive.of_value (P.field "s" s) x)));
      test "a key saved under a field resumes its stream" (fun () ->
          let key = Nx.Rng.fold_in (Nx.Rng.key 5) 3 in
          let rng = P.field "rng" Nx.Rng.ptree in
          let a =
            Nx_io.load_safetensors
              (saved Nx_io.save_safetensors (Archive.of_value rng key))
          in
          let words (k : Nx.Rng.t) = Nx.to_array (k :> Nx.int32_t) in
          equal (array int32) (words key)
            (words (Archive.to_value rng ~like:(Nx.Rng.key 0) a)));
    ]

(* Strictness *)

(* A model of blocks, whose tensors are named [blocks.0], [blocks.1], ... and
   [scale]. *)
type model = { blocks : Nx.float32_t list; scale : Nx.float64_t }

module Model = struct
  type _ t = model

  let walk c m =
    let open P.Walk in
    let blocks = field c "blocks" (list tensor) m.blocks in
    let scale = field c "scale" tensor m.scale in
    { blocks; scale }
end

let bare : model P.t = P.instantiate (module Model)
let model = P.field "model" bare

let blocks n =
  {
    blocks = List.init n (fun i -> vec f32 [| float_of_int i; -0. |]);
    scale = Nx.scalar f64 0.5;
  }

let model_value = value model
let entry name t = (name, Nx.P t)

(* The entries of a model of two blocks, then [extra]. *)
let two_blocks extra = listed (Archive.of_value model (blocks 2)) @ extra
let without name l = List.filter (fun (n, _) -> n <> name) l

let replaced name t l =
  List.map (fun (n, x) -> if n = name then entry name t else (n, x)) l

let to_value entries =
  Archive.to_value model ~like:(blocks 2) (Archive.of_list entries)

let strictness =
  let msg = ( ^ ) "Nx_io.Archive.to_value: " in
  group "strictness"
    [
      cases ~name:fst
        "to_value refuses an archive that differs, naming the entry"
        [
          ( "a missing entry",
            ( without "model.scale" (two_blocks []),
              msg "model.scale: no entry in the archive, a leaf in the value" )
          );
          ( "another shape",
            ( replaced "model.blocks.1"
                (vec f32 [| 1.; 2.; 3. |])
                (two_blocks []),
              msg "model.blocks.1: shape [3] in the archive, [2] in the value"
            ) );
          ( "a scalar for a vector",
            ( replaced "model.blocks.0" (Nx.scalar f32 1.) (two_blocks []),
              msg "model.blocks.0: shape [] in the archive, [2] in the value" )
          );
          ( "another dtype",
            ( replaced "model.scale" (Nx.scalar Nx.int32 1l) (two_blocks []),
              msg "model.scale: int32 in the archive, float64 in the value" ) );
          ( "a narrower float",
            ( replaced "model.scale" (Nx.scalar f32 0.5) (two_blocks []),
              msg "model.scale: float32 in the archive, float64 in the value" )
          );
          ( "a block more",
            ( listed (Archive.of_value model (blocks 3)),
              msg
                "model.blocks.2: an entry in the archive, no leaf in the value"
            ) );
          ( "an entry at the prefix itself",
            ( two_blocks [ entry "model" (vec f32 [| 1. |]) ],
              msg "model: an entry in the archive, no leaf in the value" ) );
          ( "an entry below a leaf",
            ( two_blocks [ entry "model.scale.x" (vec f32 [| 1. |]) ],
              msg "model.scale.x: an entry in the archive, no leaf in the value"
            ) );
        ]
        (fun (_, (entries, message)) ->
          fails message (fun () -> to_value entries));
      test "the root's prefix holds every name" (fun () ->
          let a =
            Archive.of_list
              (listed (Archive.of_value bare (blocks 2))
              @ [ entry "optim.step" (Nx.scalar Nx.int32 3l) ])
          in
          fails
            (msg "optim.step: an entry in the archive, no leaf in the value")
            (fun () -> Archive.to_value bare ~like:(blocks 2) a));
      test "to_value ignores the entries outside the prefix" (fun () ->
          let x = blocks 2 in
          let others =
            [
              entry "optim.step" (Nx.scalar Nx.int32 3l);
              entry "models.w" (vec f32 [| 1. |]);
              entry "mode" (vec Nx.uint8 [| 1 |]);
              entry "model_2.blocks.0" (vec f32 [| 1.; 2.; 3. |]);
            ]
          in
          let a =
            Archive.of_list (listed (Archive.of_value model x) @ others)
          in
          equal model_value x (Archive.to_value model ~like:(blocks 2) a));
      test "a section reads back beside another" (fun () ->
          let x = blocks 2 and y = blocks 3 in
          let other = P.field "other" bare in
          let a =
            Nx_io.load_safetensors
              (saved Nx_io.save_safetensors
                 (Archive.union
                    [ Archive.of_value model x; Archive.of_value other y ]))
          in
          equal ~msg:"model" model_value x
            (Archive.to_value model ~like:(blocks 2) a);
          equal ~msg:"other" (value other) y
            (Archive.to_value other ~like:(blocks 3) a));
      test "to_value fails before it reads a byte, and reads none to succeed"
        (fun () ->
          let a =
            Nx_io.load_safetensors
              (saved Nx_io.save_safetensors (Archive.of_value model (blocks 3)))
          in
          let before = disk_reads () in
          fails
            (msg "model.blocks.2: an entry in the archive, no leaf in the value")
            (fun () -> Archive.to_value model ~like:(blocks 2) a);
          let x = Archive.to_value model ~like:(blocks 3) a in
          equal ~msg:"bytes read" int 0 (disk_reads () - before);
          equal model_value (blocks 3) x);
      test "a tensor at the root, which has no name, is refused" (fun () ->
          let x = vec f32 [| 1. |] in
          let root fn =
            "Nx_io.Archive." ^ fn
            ^ ": a leaf at the root has no name; put it under Nx.Ptree.field"
          in
          refuses (root "of_value") (fun () -> Archive.of_value P.tensor x);
          refuses (root "to_value") (fun () ->
              Archive.to_value P.tensor ~like:x (Archive.of_list [])));
      test "two tensors with one name are refused" (fun () ->
          (* The field "a.b" and the field "b" of "a" print alike. *)
          let s = record ("a.b", P.tensor) ("a", P.field "b" P.tensor) in
          let x = (vec f32 [| 1. |], vec f32 [| 2. |]) in
          refuses "Nx_io.Archive.of_value: a.b: two leaves have this name"
            (fun () -> Archive.of_value s x);
          refuses "Nx_io.Archive.to_value: a.b: two leaves have this name"
            (fun () ->
              Archive.to_value s ~like:x
                (Archive.of_list [ entry "a.b" (vec f32 [| 3. |]) ])));
    ]

(* Entries by name *)

(* A float dtype that [float] converts to and from. *)
type float_dtype = F : (float, 'b) Nx.dtype -> float_dtype

let entries =
  Archive.of_list
    [
      entry "w" (Nx.create f32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]);
      entry "half" (Nx.cast Nx.bfloat16 (vec f32 [| 0.5; -2. |]));
      entry "blocks" (vec Nx.uint8 [| 1; 2; 255 |]);
      entry "tiny" (Nx.cast Nx.float8_e4m3 (vec f32 [| 1. |]));
      entry "count" (vec Nx.int32 [| 7l |]);
    ]

let by_name =
  let tensor ~shape dtype name = Archive.tensor ~shape dtype name entries in
  let float ~shape dtype name = Archive.float ~shape dtype name entries in
  let floats =
    vec f64 [| 0.5; -0.; Float.nan; Float.infinity; 65504.; 1e-8 |]
  in
  let dtypes = [ F Nx.float16; F Nx.bfloat16; F f32; F f64 ] in
  let pairs =
    List.concat_map (fun src -> List.map (fun dst -> (src, dst)) dtypes) dtypes
  in
  let name (F src, F dst) =
    Nx_dtype.to_string src ^ " to " ^ Nx_dtype.to_string dst
  in
  group "entries by name"
    [
      test "find is an entry or none" (fun () ->
          equal (option packed)
            (Some (Nx.P (vec Nx.int32 [| 7l |])))
            (Archive.find "count" entries);
          equal (option packed) None (Archive.find "absent" entries));
      test "tensor is the entry as stored" (fun () ->
          equal packed
            (Nx.P (vec Nx.uint8 [| 1; 2; 255 |]))
            (Nx.P (tensor ~shape:[| 3 |] Nx.uint8 "blocks"));
          equal packed
            (Nx.P (Nx.cast Nx.float8_e4m3 (vec f32 [| 1. |])))
            (Nx.P (tensor ~shape:[| 1 |] Nx.float8_e4m3 "tiny")));
      cases ~name:fst "tensor refuses what the entry is not, naming it"
        [
          ( "a missing entry",
            ( "Nx_io.Archive.tensor: absent: no entry in the archive",
              fun () -> ignore (tensor ~shape:[| 1 |] f32 "absent") ) );
          ( "another shape",
            ( "Nx_io.Archive.tensor: w: shape [2; 2] in the archive, [4] asked \
               for",
              fun () -> ignore (tensor ~shape:[| 4 |] f32 "w") ) );
          ( "another dtype",
            ( "Nx_io.Archive.tensor: half: bfloat16 in the archive, float32 \
               asked for",
              fun () -> ignore (tensor ~shape:[| 2 |] f32 "half") ) );
        ]
        (fun (_, (message, f)) -> fails message f);
      cases ~name "float casts between the four float dtypes" pairs
        (fun (F src, F dst) ->
          let x = Nx.cast src floats in
          let a = Archive.of_list [ entry "x" x ] in
          equal packed
            (Nx.P (Nx.cast dst x))
            (Nx.P (Archive.float ~shape:(Nx.shape x) dst "x" a)));
      test
        "float at the entry's dtype is the entry as stored, read from the disk"
        (fun () ->
          let a =
            Nx_io.load_safetensors
              (saved Nx_io.save_safetensors
                 (Archive.of_list [ entry "w" (vec f32 [| 1.; -2. |]) ]))
          in
          let before = disk_reads () in
          let w = Archive.float ~shape:[| 2 |] f32 "w" a in
          equal ~msg:"bytes read" int 0 (disk_reads () - before);
          equal ~msg:"on the disk" placement
            (Nx.Placement.on (Nx.Device.of_memory Nx_device.disk))
            (Nx.placement w));
      cases ~name:fst "float refuses what it does not convert"
        [
          ( "an integer entry",
            ( Failure
                "Nx_io.Archive.float: count: int32 in the archive, not \
                 float16, bfloat16, float32 or float64",
              fun () -> ignore (float ~shape:[| 1 |] f32 "count") ) );
          ( "an 8-bit float entry",
            ( Failure
                "Nx_io.Archive.float: tiny: float8_e4m3 in the archive, not \
                 float16, bfloat16, float32 or float64",
              fun () -> ignore (float ~shape:[| 1 |] f32 "tiny") ) );
          ( "an 8-bit float dtype",
            ( Invalid_argument
                "Nx_io.Archive.float: float8_e5m2 is not float16, bfloat16, \
                 float32 or float64",
              fun () -> ignore (float ~shape:[| 2 |] Nx.float8_e5m2 "half") ) );
          ( "a missing entry",
            ( Failure "Nx_io.Archive.float: absent: no entry in the archive",
              fun () -> ignore (float ~shape:[| 1 |] f32 "absent") ) );
          ( "another shape",
            ( Failure
                "Nx_io.Archive.float: half: shape [2] in the archive, [3] \
                 asked for",
              fun () -> ignore (float ~shape:[| 3 |] f32 "half") ) );
        ]
        (fun (_, (e, f)) -> raises e f);
    ]

(* Names *)

let names_group =
  let one = Nx.P (vec Nx.int8 [| 1 |]) in
  let distinct =
    Gen.with_pp
      (Testable.pp (list string))
      (Gen.bind
         (Gen.subsequence [ "w"; "b"; "model.w"; "é"; "0"; "a b" ])
         Gen.permutation)
  in
  group "names"
    [
      prop "names are the archive's names, sorted" distinct (fun ns ->
          equal (list string) (List.sort compare ns)
            (Archive.names (Archive.of_list (List.map (fun n -> (n, one)) ns))));
      prop "union holds the entries of every archive"
        (Gen.pair distinct distinct) (fun (a, b) ->
          let b = List.filter (fun n -> not (List.mem n a)) b in
          let of_names ns = Archive.of_list (List.map (fun n -> (n, one)) ns) in
          equal (list string)
            (List.sort compare (a @ b))
            (Archive.names (Archive.union [ of_names a; of_names b ])));
      cases ~name:fst "a name that names no entry or two is refused"
        [
          ( "an empty name",
            ( "Nx_io.Archive.of_list: an empty name",
              fun () -> Archive.of_list [ ("", one) ] ) );
          ( "a name twice",
            ( "Nx_io.Archive.of_list: \"w\" is named twice",
              fun () -> Archive.of_list [ ("w", one); ("b", one); ("w", one) ]
            ) );
          ( "a name in two archives",
            ( "Nx_io.Archive.union: \"w\" is in two archives",
              fun () ->
                Archive.union
                  [
                    Archive.of_list [ ("w", one) ];
                    Archive.of_list [ ("b", one) ];
                    Archive.of_list [ ("w", one) ];
                  ] ) );
        ]
        (fun (_, (message, f)) -> refuses message f);
    ]

let () =
  exit (run "nx.io archive" [ round_trips; strictness; by_name; names_group ])
