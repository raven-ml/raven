(* HDF5 through Nx_io.load_hdf5, Nx_io.decode_hdf5 and Nx_io.Hdf5. Files
   come from libhdf5 2.0 through h5py 3.16 (gen/generate.py), each beside
   an npz of every dataset as h5py reads it, an element of a datatype nx
   lacks as its bytes: the expected values. A window is held to a slice of
   the dataset's values, and a partial file to the whole file's reading,
   as nx_io.mli's Partial files section states it. *)

open Windtrap
open Io_support
module H = Nx_io.Hdf5
module P = Nx_io.Partial

let load name = snd (Nx_io.load_hdf5 (fixture name))
let groups name = fst (Nx_io.load_hdf5 (fixture name))

let find ds path =
  match List.find_opt (fun d -> H.path d = path) ds with
  | Some d -> d
  | None -> failf "no dataset %s" path

let dataset file path = find (load file) path

(* The fixtures with an npz of their datasets. *)
let readable =
  [
    "hdf5_layouts";
    "hdf5_filters";
    "hdf5_types";
    "hdf5_links";
    "hdf5_small_k";
    "hdf5_strings";
  ]

(* Paths in the order nx_io.mli states, name by name. *)
let ordered paths =
  List.sort
    (fun a b ->
      compare (String.split_on_char '/' a) (String.split_on_char '/' b))
    paths

(* [path] without its first [/], as the npz names it. *)
let entry path = String.sub path 1 (String.length path - 1)

(* Readings *)

let readings =
  cases "every dataset reads as h5py reads it" ~id:Fun.id readable
    (fun file ->
      let ds = load (file ^ ".h5") in
      let a = Nx_io.load_npz (fixture (file ^ ".npz")) in
      let paths = List.map H.path ds in
      equal (list string) ~msg:"in path order" (ordered paths) paths;
      equal (list string) ~msg:"the datasets" (Nx_io.Archive.names a)
        (List.map entry paths);
      List.iter
        (fun d ->
          equal packed ~msg:(H.path d)
            (Option.get (Nx_io.Archive.find (entry (H.path d)) a))
            (H.values d))
        ds)

let layouts =
  group "layouts"
    [
      test "a contiguous dataset in the host's order at an aligned offset is \
            mapped" (fun () ->
          equal memory Over_file
            (memory_of (H.values (dataset "hdf5_layouts.h5" "/aligned"))));
      test "float64 elements at an offset of no multiple of 8 are copied"
        (fun () ->
          let d = dataset "hdf5_layouts.h5" "/f64" in
          equal memory Own_memory (memory_of (H.values d)));
      test "big-endian elements are swapped into memory of their own"
        (fun () ->
          let d = dataset "hdf5_layouts.h5" "/i16be" in
          equal memory Own_memory (memory_of (H.values d));
          equal packed
            (H.values (dataset "hdf5_layouts.h5" "/i16le"))
            (H.values d));
      cases "chunk shapes"
        ~id:(fun (p, _) -> p)
        [
          ("/chunked/edges", Some [| 3; 4 |]);
          ("/chunked/rank3", Some [| 2; 3; 4 |]);
          ("/f32", None);
          ("/compact", None);
          ("/scalar", None);
        ]
        (fun (path, c) ->
          equal (option (array int)) c
            (H.chunk (dataset "hdf5_layouts.h5" path)));
      cases "shapes"
        ~id:(fun (p, _) -> p)
        [
          ("/scalar", [||]);
          ("/empty", [| 0; 3 |]);
          ("/chunked/rank3", [| 5; 6; 7 |]);
          ("/many/d07", [||]);
        ]
        (fun (path, s) ->
          equal (array int) s (H.shape (dataset "hdf5_layouts.h5" path)));
      test "a dataset never written reads as its fill value" (fun () ->
          equal packed
            (Nx.P (Nx.full Nx.float32 [| 5 |] (-1.5)))
            (H.values (dataset "hdf5_layouts.h5" "/unwritten")));
      test "chunks never written read as the fill value, or zeros" (fun () ->
          equal packed
            (Nx.P
               (Nx.create Nx.float32 [| 9 |]
                  [| 0.; 1.; 2.; 3.; 2.5; 2.5; 2.5; 2.5; 2.5 |]))
            (H.values (dataset "hdf5_layouts.h5" "/chunked/fill"));
          equal packed
            (Nx.P (Nx.create Nx.int16 [| 9 |] [| 0; 0; 0; 0; 1; 2; 3; 4; 0 |]))
            (H.values (dataset "hdf5_layouts.h5" "/chunked/zeros")));
      test "an object reached by several links, a cycle's included, is read \
            once at its least path" (fun () ->
          let paths = List.map H.path (load "hdf5_layouts.h5") in
          equal (list string)
            [ "/g/h/deep" ]
            (List.filter
               (fun p -> List.mem p [ "/g/h/deep"; "/link" ]
                         || String.starts_with ~prefix:"/soft" p
                         || String.starts_with ~prefix:"/g/h/back" p)
               paths));
    ]

(* Links and groups *)

let links =
  let file = "hdf5_links.h5" in
  group "links"
    [
      test "soft links are followed, and each object read at its least path, \
            name by name" (fun () ->
          (* /a/x is also /a-, /s, /a/r and /ordered/up; /c/e is also /0
             and /t/e; /ordered/sub/w is also /ordered/rel. *)
          equal (list string)
            [ "/0"; "/a/r"; "/ordered/rel"; "/ordered/y"; "/ordered/z" ]
            (List.map H.path (load file));
          equal (list string)
            [ "/"; "/a"; "/c"; "/ordered"; "/ordered/sub" ]
            (List.map H.Group.path (groups file)));
      test "the attributes of groups, the root's included" (fun () ->
          let g path =
            List.find (fun g -> H.Group.path g = path) (groups file)
          in
          equal (list string) [ "title" ] (H.Group.attributes (g "/"));
          equal packed
            (Nx.P (Nx.create Nx.uint8 [| 5 |] [| 114; 97; 118; 101; 110 |]))
            (H.Group.attribute (g "/") "title");
          equal packed (Nx.P (Nx.scalar Nx.int32 3l))
            (H.Group.attribute (g "/a") "n");
          (* A group of a version-2 header, with link messages. *)
          equal packed (Nx.P (Nx.scalar Nx.int16 (-4)))
            (H.Group.attribute (g "/ordered") "k"));
      cases "lookup3, the checksum of version-2 headers, gives hashlittle's \
             values"
        ~id:(fun (k, i, _) -> Printf.sprintf "%S from %d" k i)
        [
          (* lookup3.c, driver5: "c = hashlittle("Four score and seven
             years ago", 30, 0); ... 17770551", and "/* cd628161 */" for an
             initial value of 1; the empty key's hash is its initial value
             plus 0xdeadbeef. *)
          ("", 0, 0xdeadbeef);
          ("Four score and seven years ago", 0, 0x17770551);
          ("Four score and seven years ago", 1, 0xcd628161);
        ]
        (fun (key, init, hash) ->
          equal int hash (H.lookup3 ~init key));
      test "a version-2 header whose byte is flipped fails its checksum"
        (fun () ->
          let whole = read (fixture file) in
          let at =
            let rec find i =
              if String.sub whole i 4 = "OHDR" then i else find (i + 1)
            in
            find 0
          in
          let path = Io_support.file (flip_byte whole (at + 12)) in
          fails ~path ~naming:"fails its checksum" (fun () ->
              Nx_io.load_hdf5 path));
    ]

(* Attributes *)

let attributes =
  let d () = dataset "hdf5_types.h5" "/f8" in
  group "attributes"
    [
      test "names, in header order" (fun () ->
          equal (list string)
            [ "scale"; "bands"; "unit"; "name"; "λ" ]
            (H.attributes (d ())));
      test "values" (fun () ->
          equal packed (Nx.P (Nx.scalar Nx.float64 0.25))
            (H.attribute (d ()) "scale");
          equal packed
            (Nx.P (Nx.create Nx.int32 [| 4 |] [| 3l; 1l; 4l; 1l |]))
            (H.attribute (d ()) "bands");
          equal packed
            (Nx.P (Nx.create Nx.uint8 [| 3 |] [| 110; 74; 121 |]))
            (H.attribute (d ()) "unit");
          equal packed (Nx.P (Nx.scalar Nx.float32 1.5))
            (H.attribute (d ()) "λ"));
      test "a UTF-8 name, in an attribute message of version 3" (fun () ->
          let i1 = dataset "hdf5_types.h5" "/i1" in
          equal (list string) [ "ν" ] (H.attributes i1);
          equal packed (Nx.P (Nx.scalar Nx.int32 7l)) (H.attribute i1 "ν"));
      test "a variable-length string is refused at its call" (fun () ->
          fails ~path:(fixture "hdf5_types.h5")
            ~naming:"/f8: name: datatype: variable-length, a string"
            (fun () -> H.attribute (d ()) "name"));
      test "a missing name" (fun () ->
          fails ~path:(fixture "hdf5_types.h5")
            ~naming:"/f8: no attribute named \"nope\"" (fun () ->
              H.attribute (d ()) "nope"));
    ]

(* Refusals *)

let at_load =
  cases "files refused at load, naming the structure"
    ~id:(fun (f, _) -> f)
    [
      ("hdf5_latest.h5", "superblock version 3; read are 0 and 1");
      ("hdf5_extlink.h5", "/g: /g/e: an external link");
      ("hdf5_userblock.h5", "no HDF5 signature at byte 0");
    ]
    (fun (file, naming) ->
      fails ~path:(fixture file) ~naming (fun () -> load file))

let at_read =
  let file = "hdf5_refused.h5" in
  group "datasets refused when read"
    [
      test "the file loads, and its other datasets read" (fun () ->
          equal (list string)
            [
              "/committed"; "/external"; "/null"; "/ok"; "/reference";
              "/scaleoffset"; "/two_compressors"; "/vlen_ints"; "/vlen_string";
            ]
            (List.map H.path (load file));
          equal packed
            (Nx.P (Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 2. |]))
            (H.values (dataset file "/ok")));
      cases "each names its path and the feature"
        ~id:(fun (p, _) -> p)
        [
          ("/vlen_string", "/vlen_string: datatype: variable-length, a string");
          ("/vlen_ints", "/vlen_ints: datatype: variable-length, a sequence");
          ("/reference", "/reference: datatype: a reference");
          ("/committed", "/committed: a committed datatype");
          ("/null", "/null: dataspace: null");
          ("/two_compressors", "/two_compressors: two compressing filters");
          ("/scaleoffset", "/scaleoffset: filter 6 (scaleoffset)");
          ("/external", "/external: storage in external files");
        ]
        (fun (path, naming) ->
          let d = dataset file path in
          fails ~path:(fixture file) ~naming (fun () -> H.values d);
          fails ~path:(fixture file) ~naming (fun () ->
              H.window d ~start:[| 0 |] ~shape:[| 1 |]));
    ]

let checksums =
  let file = "hdf5_fletcher_flipped.h5" in
  group "Fletcher-32"
    [
      test "a chunk whose byte is flipped raises, naming it" (fun () ->
          fails ~path:(fixture file)
            ~naming:"/fletcher: chunk at [8], byte "
            (fun () -> H.values (dataset file "/fletcher"));
          fails ~path:(fixture file)
            ~naming:"filter 3 (Fletcher-32): the checksum does not match the \
                     data" (fun () -> H.values (dataset file "/fletcher")));
      test "checksums in libhdf5's byte order before 1.6.3 read" (fun () ->
          let a = Nx_io.load_npz (fixture "hdf5_filters.npz") in
          equal packed
            (Option.get (Nx_io.Archive.find "fletcher" a))
            (H.values (dataset "hdf5_fletcher_reversed.h5" "/fletcher")));
      test "a window beside it reads" (fun () ->
          let d = dataset file "/fletcher" in
          match H.window d ~start:[| 0 |] ~shape:[| 8 |] with
          | Ok x ->
              let first = Array.init 8 (fun i -> i - 15) in
              equal packed (Nx.P (Nx.create Nx.int16 [| 8 |] first)) x
          | Error _ -> failf "a whole file lacked bytes");
    ]

(* [small d] is whether [d]'s values take at most 1 MiB, so that a damaged
   file's dimensions cannot make a test allocate past it. *)
let small d =
  match H.shape d with
  | dims -> Array.fold_left ( * ) 1 dims <= 1 lsl 17
  | exception Failure _ -> true

let corrupt =
  group "damaged files"
    [
      test "a shuffle filter of elements of no byte" (fun () ->
          fails ~path:(fixture "hdf5_damaged.h5")
            ~naming:"/s: filter 2 (shuffle) of elements of 0 bytes" (fun () ->
              H.values (dataset "hdf5_damaged.h5" "/s")));
      test "chunks whose bytes pass any integer" (fun () ->
          let d = dataset "hdf5_damaged.h5" "/h" in
          fails ~path:(fixture "hdf5_damaged.h5") ~naming:"/h: " (fun () ->
              H.values d);
          fails ~path:(fixture "hdf5_damaged.h5") ~naming:"/h: " (fun () ->
              H.chunk d));
      test "every proper prefix of a file fails to load, naming it" (fun () ->
          prefixes_fail Nx_io.load_hdf5 (read (fixture "hdf5_refused.h5")));
      test "a byte of the first 3 KiB flipped loads, reads or fails" (fun () ->
          let whole = read (fixture "hdf5_filters.h5") in
          flips_load_or_fail ~upto:3072
            (fun path ->
              List.iter
                (fun d ->
                  if small d then ignore (H.values d);
                  ignore (H.attributes d))
                (snd (Nx_io.load_hdf5 path)))
            whole);
    ]

(* Windows *)

(* A box of [dims]: whole, of no element, or any, from two draws an axis. *)
let box dims (whole, draws) =
  let r = Array.length dims in
  if whole then (Array.make r 0, Array.copy dims)
  else
    let start =
      Array.init r (fun a -> fst (List.nth draws a) mod (dims.(a) + 1))
    in
    let shape =
      Array.init r (fun a ->
          snd (List.nth draws a) mod (dims.(a) - start.(a) + 1))
    in
    (start, shape)

let draws =
  Gen.pair
    (Gen.frequency [ (1, Gen.constant true); (4, Gen.constant false) ])
    (Gen.list ~size:(Gen.constant 4)
       (Gen.pair (Gen.int_range 0 1000) (Gen.int_range 0 1000)))

let slice (Nx.P x) ~start ~shape =
  let r = Array.length start in
  let ranges =
    List.init r (fun a -> Nx.R (start.(a), start.(a) + shape.(a)))
  in
  Nx.P (Nx.copy (Nx.slice ranges x))

let windows =
  prop ~count:300 "a window is a slice of the values"
    (Gen.triple (Gen.of_list readable) (Gen.int_range 0 1000) draws)
    (fun (file, k, b) ->
      let ds = load (file ^ ".h5") in
      let d = List.nth ds (k mod List.length ds) in
      let dims = H.shape d in
      let start, shape = box dims b in
      let chunked = H.chunk d <> None in
      cover "chunked" chunked;
      cover "an edge chunk"
        (match H.chunk d with
        | Some c ->
            Array.exists Fun.id
              (Array.mapi
                 (fun a n ->
                   n > 0 && (start.(a) + n) mod c.(a) <> 0
                   && start.(a) + n = dims.(a)
                   && dims.(a) mod c.(a) <> 0)
                 shape)
        | None -> false);
      cover "no element" (Array.exists (( = ) 0) shape);
      match H.window d ~start ~shape with
      | Ok x -> equal packed (slice (H.values d) ~start ~shape) x
      | Error _ -> failf "a whole file lacked bytes")

let window_refusals =
  group "boxes outside a dataset"
    [
      test "past an axis's end, or of another rank" (fun () ->
          let d = dataset "hdf5_layouts.h5" "/chunked/edges" in
          invalid ~by:"Nx_io.Hdf5.window" (fun () ->
              H.window d ~start:[| 5; 0 |] ~shape:[| 3; 1 |]);
          invalid ~by:"Nx_io.Hdf5.window" (fun () ->
              H.window d ~start:[| 0 |] ~shape:[| 1 |]);
          invalid ~by:"Nx_io.Hdf5.window" (fun () ->
              H.window d ~start:[| -1; 0 |] ~shape:[| 1; 1 |]));
    ]

(* Partial files *)

(* The partial file "part" of [whole] that holds the bytes [held] marks,
   added in pieces of at most 61 bytes. *)
let partial whole held =
  let n = String.length whole in
  let rec from i p =
    if i = n then p
    else if not held.(i) then from (i + 1) p
    else begin
      let j = ref i in
      while !j < n && held.(!j) && !j - i < 61 do
        incr j
      done;
      from !j (P.add p ~at:i (String.sub whole i (!j - i)))
    end
  in
  from 0 (P.v ~name:"part" ~length:n)

let fill whole p rs =
  List.fold_left (fun p (at, n) -> P.add p ~at (String.sub whole at n)) p rs

(* What a call did, its message past its file's name. *)
type 'a outcome =
  | Done of 'a
  | Lacks of (int * int) list
  | Raised of string
  | Refused of string

let past name m =
  let prefix = name ^ ": " in
  let k = String.length prefix in
  if String.starts_with ~prefix m then String.sub m k (String.length m - k)
  else m

let outcome name f =
  match f () with
  | Ok x -> Done x
  | Error rs -> Lacks rs
  | exception Failure m -> Raised (past name m)
  | exception Invalid_argument m -> Refused m

let outcomes t =
  let pp ppf = function
    | Done x -> Format.fprintf ppf "Done %a" (Testable.pp t) x
    | Lacks rs ->
        Format.fprintf ppf "Lacks [%s]"
          (String.concat "; "
             (List.map (fun (a, n) -> Printf.sprintf "(%d, %d)" a n) rs))
    | Raised m -> Format.fprintf ppf "Raised %S" m
    | Refused m -> Format.fprintf ppf "Refused %S" m
  in
  let equal a b =
    match (a, b) with
    | Done x, Done y -> Testable.equal t x y
    | Done _, _ | _, Done _ -> false
    | _ -> a = b
  in
  Testable.make ~pp ~equal

let paths = outcomes (pair (list string) (list string))
let named (gs, ds) = (List.map H.Group.path gs, List.map H.path ds)
let windowed = outcomes packed

let map_done f = function
  | Done x -> Done (f x)
  | Lacks rs -> Lacks rs
  | Raised m -> Raised m
  | Refused m -> Refused m

(* The window of the box [b] draws of [d], as an outcome. *)
let window_of name d b =
  outcome name (fun () ->
      let start, shape = box (H.shape d) b in
      H.window d ~start ~shape)

(* A drawn partial file: a fixture, maybe with a byte flipped; the bytes
   held, none or all but a few holes; a dataset and a box of it. *)
type drawn = {
  file : string;
  whole : string;
  held : bool array;
  more : (int * int) list;
  k : int;
  b : bool * (int * int) list;
}

let pp_drawn ppf d =
  let holes = Array.fold_left (fun n h -> if h then n else n + 1) 0 d.held in
  Format.fprintf ppf "%s, %d of %d bytes lacking, dataset %d" d.file holes
    (String.length d.whole) d.k

let drawn =
  let open Gen in
  let* file = of_list readable in
  let built = read (fixture (file ^ ".h5")) in
  let n = String.length built in
  let* flip =
    frequency
      [
        (3, constant None);
        (1, map Option.some (int_range 0 (n - 1)));
        (1, map Option.some (int_range 0 (Int.min (n - 1) 3000)));
      ]
  in
  let whole = match flip with None -> built | Some i -> flip_byte built i in
  let range =
    let* at = int_range 0 (n - 1) in
    let+ len = int_range 1 (Int.max 1 (n / 8)) in
    (at, Int.min len (n - at))
  in
  let* held =
    frequency
      [
        (1, constant (Array.make n false));
        ( 4,
          let+ holes = list ~size:(int_range 0 4) range in
          let held = Array.make n true in
          List.iter (fun (at, len) -> Array.fill held at len false) holes;
          held );
      ]
  in
  let* more = list ~size:(int_range 0 3) range in
  let* k = int_range 0 1000 in
  let+ b = draws in
  { file; whole; held; more; k; b }

let drawn = Gen.with_pp pp_drawn drawn

(* The whole file's datasets' paths, and the window of dataset [k]. *)
let whole_of d =
  let path = Io_support.file d.whole in
  let ds = outcome path (fun () -> Ok (Nx_io.load_hdf5 path)) in
  let w =
    match ds with
    | Done (_, (_ :: _ as ds)) ->
        let x = List.nth ds (d.k mod List.length ds) in
        if small x then Some (window_of path x d.b) else None
    | _ -> None
  in
  (map_done named ds, w)

let check_ranges ~held n rs =
  if rs = [] then failf "no range";
  ignore
    (List.fold_left
       (fun prev (at, len) ->
         if len <= 0 || at < prev || at + len > n then
           failf "range (%d, %d) after byte %d, in %d bytes" at len prev n;
         for i = at to at + len - 1 do
           if held.(i) then failf "range (%d, %d) holds byte %d" at len i
         done;
         at + len + 1)
       0 rs
      : int)

let hold held rs = List.iter (fun (at, n) -> Array.fill held at n true) rs
let decode_of p = outcome "part" (fun () -> Nx_io.decode_hdf5 p)
let raised = function Raised _ -> true | _ -> false

let agreement =
  prop ~count:200
    "a partial file reads as its whole file, once it holds the ranges it \
     names"
    drawn (fun d ->
      let n = String.length d.whole and held = Array.copy d.held in
      let loaded, window = whole_of d in
      let rounds = ref 0 in
      let rec loop p call =
        incr rounds;
        if !rounds > n + 2 then failf "%d rounds for %d bytes" !rounds n;
        match call p with
        | Lacks rs ->
            check_ranges ~held n rs;
            hold held rs;
            loop (fill d.whole p rs) call
        | o -> (o, p)
      in
      let o, p = loop (partial d.whole d.held) decode_of in
      cover "nothing held at first" (Array.for_all not d.held);
      cover "raised" (raised o);
      equal paths loaded (map_done named o);
      let first = !rounds in
      let window_at p =
        match decode_of p with
        | Done (_, ds) ->
            window_of "part" (List.nth ds (d.k mod List.length ds)) d.b
        | o ->
            failf "decoded once, then %a" (Testable.pp paths) (map_done named o)
      in
      match (o, window) with
      | Done (_, _ :: _), Some window ->
          let w, _ = loop p window_at in
          equal windowed window w;
          cover "a window that lacked bytes" (!rounds > first + 1)
      | _ -> ())

let monotone =
  prop ~count:200 "adding bytes never turns a result into Error" drawn
    (fun d ->
      let p = partial d.whole d.held in
      let q = fill d.whole p d.more in
      match decode_of p with
      | Done (_, (_ :: _ as ds)) ->
          let ds' =
            match decode_of q with
            | Done (_, ds') -> ds'
            | o ->
                failf "decoded, then %a" (Testable.pp paths) (map_done named o)
          in
          equal (list string) (List.map H.path ds) (List.map H.path ds');
          let k = d.k mod List.length ds in
          let old = List.nth ds k and fresh = List.nth ds' k in
          if small old then begin
            match window_of "part" old d.b with
            | Done x ->
                equal windowed (Done x) (window_of "part" fresh d.b);
                cover "a window, then more bytes" (d.more <> [])
            | Lacks rs ->
                (* The dataset decoded before the ranges were added never
                   reads them. *)
                let q = fill d.whole q rs in
                equal windowed (Lacks rs) (window_of "part" old d.b);
                let fresh =
                  match decode_of q with
                  | Done (_, ds) -> List.nth ds k
                  | _ -> failf "decoded, then not"
                in
                if window_of "part" fresh d.b = Lacks rs then
                  failf "the ranges added, the same ranges again";
                cover "a dataset decoded before its ranges were added" true
            | Raised _ | Refused _ -> ()
          end
      | _ -> ())

(* A fixture's chunks, as the generator lists them: each dataset's path,
   offset in elements, byte and size. *)
let chunks_of file =
  read (fixture (file ^ ".chunks"))
  |> String.split_on_char '\n'
  |> List.filter (( <> ) "")
  |> List.map (fun l ->
      match String.split_on_char '\t' l with
      | [ p; o; at; n ] ->
          let o = List.map int_of_string (String.split_on_char ',' o) in
          (p, Array.of_list o, int_of_string at, int_of_string n)
      | _ -> failf "a line %S" l)

let only_covered =
  prop ~count:200 "a window reads only the chunks its box covers"
    (Gen.triple
       (Gen.of_list
          [ "hdf5_layouts"; "hdf5_filters"; "hdf5_small_k"; "hdf5_strings" ])
       (Gen.int_range 0 1000) draws)
    (fun (file, k, b) ->
      let whole = read (fixture (file ^ ".h5")) in
      let ds = List.filter (fun d -> H.chunk d <> None) (load (file ^ ".h5")) in
      let d = List.nth ds (k mod List.length ds) in
      let c = Option.get (H.chunk d) in
      let start, shape = box (H.shape d) b in
      let covers o =
        Array.for_all Fun.id
          (Array.mapi
             (fun a v -> v < start.(a) + shape.(a) && v + c.(a) > start.(a))
             o)
      in
      (* Every byte but those of the chunks the box does not cover. *)
      let held = Array.make (String.length whole) true in
      let dropped = ref 0 in
      List.iter
        (fun (path, o, at, n) ->
          if path = H.path d && not (covers o) then begin
            incr dropped;
            Array.fill held at n false
          end)
        (chunks_of file);
      cover "chunks left out" (!dropped > 0);
      match Nx_io.decode_hdf5 (partial whole held) with
      | Error _ -> failf "the partial file lacked structure"
      | Ok (_, ds') -> (
          match H.window (find ds' (H.path d)) ~start ~shape with
          | Ok x -> equal packed (slice (H.values d) ~start ~shape) x
          | Error rs -> failf "%d ranges lacking" (List.length rs)))

let partial_cases =
  group "partial files"
    [
      test "a remote read: decode, add the ranges, and read a window"
        (fun () ->
          let whole = read (fixture "hdf5_filters.h5") in
          let get at n = String.sub whole at n in
          let add p rs =
            List.fold_left (fun p (at, n) -> P.add p ~at (get at n)) p rs
          in
          let rec decoded p =
            match Nx_io.decode_hdf5 p with
            | Error rs -> decoded (add p rs)
            | Ok (_, ds) -> (p, find ds "/lzf")
          in
          let p, d = decoded (P.v ~name:"part" ~length:(String.length whole)) in
          let rec window p d =
            match H.window d ~start:[| 16; 0 |] ~shape:[| 9; 6 |] with
            | Ok x -> x
            | Error rs ->
                let p, d = decoded (add p rs) in
                window p d
          in
          equal packed
            (slice (H.values (dataset "hdf5_filters.h5" "/lzf"))
               ~start:[| 16; 0 |] ~shape:[| 9; 6 |])
            (window p d));
    ]

(* Rows *)

(* [joined d ws] is the windows [ws], each a first row and a count, of [d]'s
   rows by [H.window], joined along axis 0. *)
let joined d ws =
  let dims = H.shape d in
  let window (row, rows) =
    let start = Array.mapi (fun a _ -> if a = 0 then row else 0) dims in
    let shape = Array.mapi (fun a n -> if a = 0 then rows else n) dims in
    match H.window d ~start ~shape with
    | Ok x -> x
    | Error _ -> failf "a whole file lacked bytes"
  in
  match List.map window ((0, 0) :: ws) with
  | Nx.P first :: rest ->
      let dt = Nx.dtype first in
      Nx.P (Nx.concatenate ~axis:0 (first :: List.map (Nx.unpack dt) rest))
  | [] -> assert false

let rows_of d ws =
  H.rows d
    ~row:(Array.of_list (List.map fst ws))
    ~rows:(Array.of_list (List.map snd ws))

(* The datasets of the fixtures that have rows, each with its fixture. *)
let with_rows =
  lazy
    (List.concat_map
       (fun file ->
         List.filter_map
           (fun d -> if H.shape d = [||] then None else Some (file, d))
           (load (file ^ ".h5")))
       readable)

let is_compact (_, d) = H.path d = "/compact"
let is_chunked (_, d) = H.chunk d <> None
let is_array (_, d) = H.path d = "/array" || H.path d = "/array_be"
let axes_past (_, d) = (match H.values d with Nx.P x -> Nx.ndim x) - 1

(* A dataset with rows, drawn so that each layout, an array datatype and
   rows of three axes past the first come often. *)
let rows_dataset =
  let pp ppf (file, d) = Format.fprintf ppf "%s.h5 %s" file (H.path d) in
  let among f =
    let ds = List.filter f (Lazy.force with_rows) in
    Gen.of_list ~pp ds
  in
  let contiguous x = not (is_compact x || is_chunked x) in
  Gen.frequency
    [
      (1, among is_compact);
      (2, among contiguous);
      (3, among is_chunked);
      (1, among is_array);
      (1, among (fun x -> axes_past x = 3));
    ]

(* A window of [n] rows: at the first row, at the last, of no row, or any. *)
let rows_window n (kind, a, b) =
  match kind with
  | (`First | `Last) when n = 0 -> (0, 0)
  | `First -> (0, 1 + (a mod n))
  | `Last ->
      let rows = 1 + (a mod n) in
      (n - rows, rows)
  | `Empty -> (a mod (n + 1), 0)
  | `Any ->
      let row = a mod (n + 1) in
      (row, b mod (n - row + 1))

let window_draws =
  let kind =
    Gen.frequency
      [
        (1, Gen.constant `First);
        (1, Gen.constant `Last);
        (1, Gen.constant `Empty);
        (3, Gen.constant `Any);
      ]
  in
  let pp ppf (kind, a, b) =
    let name =
      match kind with
      | `First -> "first"
      | `Last -> "last"
      | `Empty -> "empty"
      | `Any -> "any"
    in
    Format.fprintf ppf "%s %d %d" name a b
  in
  Gen.list ~size:(Gen.int_range 0 4)
    (Gen.with_pp pp
       (Gen.triple kind (Gen.int_range 0 1000) (Gen.int_range 0 1000)))

let rows_cover n ws =
  let full = List.filter (fun (_, m) -> m > 0) ws in
  let pairs =
    List.concat_map
      (fun (i, a) ->
        List.filter_map
          (fun (j, b) -> if i < j then Some (a, b) else None)
          (List.mapi (fun j w -> (j, w)) full))
      (List.mapi (fun i w -> (i, w)) full)
  in
  cover "no window" (ws = []);
  cover "an empty window" (List.exists (fun (_, m) -> m = 0) ws);
  cover "a window at the first row" (List.exists (fun (r, _) -> r = 0) full);
  cover "a window at the last row"
    (List.exists (fun (r, m) -> r + m = n) full);
  cover "windows overlapping"
    (List.exists (fun ((r, m), (s, k)) -> r < s + k && s < r + m) pairs);
  cover "windows out of order"
    (List.exists (fun ((r, _), (s, _)) -> r > s) pairs)

let rows_agree =
  prop ~count:300 "rows are the windows of each row range, joined"
    (Gen.pair rows_dataset window_draws) (fun ((file, d), draws) ->
      let n = (H.shape d).(0) in
      let ws = List.map (rows_window n) draws in
      cover "compact" (is_compact (file, d));
      cover "contiguous" (not (is_compact (file, d) || is_chunked (file, d)));
      cover "chunked" (is_chunked (file, d));
      cover "an array datatype" (is_array (file, d));
      cover "rows of three axes past the first" (axes_past (file, d) = 3);
      rows_cover n ws;
      match rows_of d ws with
      | Ok x -> equal packed (joined d ws) x
      | Error _ -> failf "a whole file lacked bytes")

(* The chunked datasets of the fixtures that list their chunks. *)
let chunked_rows =
  let pp ppf (file, d) = Format.fprintf ppf "%s.h5 %s" file (H.path d) in
  let listed =
    [ "hdf5_layouts"; "hdf5_filters"; "hdf5_small_k"; "hdf5_strings" ]
  in
  Gen.of_list ~pp
    (List.filter
       (fun (file, d) -> List.mem file listed && is_chunked (file, d))
       (Lazy.force with_rows))

let rows_economy =
  prop ~count:200
    "rows read only the chunks that hold a window's row, and name each \
     chunk missing"
    (Gen.pair chunked_rows window_draws) (fun ((file, d), draws) ->
      let whole = read (fixture (file ^ ".h5")) in
      let dims = H.shape d and c = Option.get (H.chunk d) in
      let ws = List.map (rows_window dims.(0)) draws in
      let holds o =
        List.exists
          (fun (r, m) -> m > 0 && o.(0) < r + m && o.(0) + c.(0) > r)
          ws
      in
      let mine =
        List.filter (fun (p, _, _, _) -> p = H.path d) (chunks_of file)
      in
      let needed, others = List.partition (fun (_, o, _, _) -> holds o) mine in
      (* Every byte but those of the chunks no window holds a row of. *)
      let held = Array.make (String.length whole) true in
      List.iter (fun (_, _, at, n) -> Array.fill held at n false) others;
      cover "chunks left out" (others <> []);
      cover "a chunk needed" (needed <> []);
      let rows_in held =
        match Nx_io.decode_hdf5 (partial whole held) with
        | Error _ -> failf "the partial file lacked structure"
        | Ok (_, ds) -> rows_of (find ds (H.path d)) ws
      in
      (match rows_in held with
      | Ok x -> equal packed (joined d ws) x
      | Error rs -> failf "%d ranges lacking" (List.length rs));
      match needed with
      | [] -> ()
      | (_, _, at, n) :: _ ->
          Array.fill held at n false;
          let lacks = function Ok _ -> [] | Error rs -> rs in
          equal (list (pair int int)) [ (at, n) ] (lacks (rows_in held)))

let rows_cases =
  let d path = dataset "hdf5_layouts.h5" path in
  let by = "Nx_io.Hdf5.rows" in
  let refused ~naming f =
    raises_match (Exn.invalid_arg ~sub:(by ^ ": " ^ naming)) (fun () ->
        ignore (f ()))
  in
  group "rows"
    [
      test "no window is of no row" (fun () ->
          match H.rows (d "/chunked/rank3") ~row:[||] ~rows:[||] with
          | Ok x ->
              equal packed (Nx.P (Nx.zeros Nx.uint16 [| 0; 6; 7 |])) x
          | Error _ -> failf "a whole file lacked bytes");
      test "a dataset of no row takes windows of no row alone" (fun () ->
          (match H.rows (d "/empty") ~row:[| 0 |] ~rows:[| 0 |] with
          | Ok x -> equal packed (Nx.P (Nx.zeros Nx.float32 [| 0; 3 |])) x
          | Error _ -> failf "a whole file lacked bytes");
          refused ~naming:"window 0" (fun () ->
              H.rows (d "/empty") ~row:[| 0 |] ~rows:[| 1 |]));
      test "a scalar dataset has no rows" (fun () ->
          invalid ~by (fun () ->
              H.rows (d "/scalar") ~row:[| 0 |] ~rows:[| 1 |]));
      test "rows and counts of other lengths" (fun () ->
          invalid ~by (fun () ->
              H.rows (d "/chunked/edges") ~row:[| 0; 1 |] ~rows:[| 1 |]));
      test "windows of more than 2^53 elements are refused" (fun () ->
          let d = dataset "hdf5_huge.h5" "/huge" in
          let half = 1 lsl 27 in
          refused ~naming:"the windows hold more than 2^53 elements"
            (fun () -> H.rows d ~row:[| 0; 0 |] ~rows:[| half; half |]));
      cases "a window outside the rows, naming it" ~id:fst
        [
          ("a negative row", ([| 0; -1 |], [| 1; 1 |]));
          ("a negative count", ([| 0; 1 |], [| 1; -1 |]));
          ("past the last row", ([| 0; 5 |], [| 1; 3 |]));
        ]
        (fun (_, (row, rows)) ->
          refused ~naming:"window 1" (fun () ->
              H.rows (d "/chunked/edges") ~row ~rows));
    ]

(* Strings *)

let strings_cases =
  let file = "hdf5_strings.h5" in
  group "strings"
    [
      cases "each cut as its datatype's padding states" ~id:fst
        [
          ("/nullpad", [| "ab"; "abcdefgh"; ""; "a\000b"; " x" |]);
          ("/nullterm", [| "ab"; "abcdefgh"; ""; "a"; " x" |]);
          ("/spacepad", [| "ab"; "abcdefgh"; ""; "a\000b"; " x" |]);
          ("/utf8", [| "λ"; "aνb"; "" |]);
          ("/array", [| "a"; "bc"; "def"; "ghij"; ""; "k" |]);
        ]
        (fun (path, expected) ->
          let d = dataset file path in
          equal (array string) expected (H.strings d (H.values d)));
      test "the strings of rows" (fun () ->
          let d = dataset file "/chunked" in
          let row i =
            List.concat_map
              (fun j -> List.init 3 (fun k -> Printf.sprintf "r%dc%d%d" i j k))
              [ 0; 1 ]
          in
          match H.rows d ~row:[| 3; 0 |] ~rows:[| 1; 1 |] with
          | Ok x ->
              equal (array string) (Array.of_list (row 3 @ row 0))
                (H.strings d x)
          | Error _ -> failf "a whole file lacked bytes");
      cases "a dataset of other elements is refused" ~id:snd
        [
          ("hdf5_layouts.h5", "/f32");
          ("hdf5_types.h5", "/opaque");
          ("hdf5_types.h5", "/compound");
        ]
        (fun (file, path) ->
          let d = dataset file path in
          invalid ~by:"Nx_io.Hdf5.strings" (fun () ->
              H.strings d (H.values d)));
      cases "a padding the format reserves is refused by strings alone"
        ~id:string_of_int [ 3; 9; 15 ] (fun pad ->
          let whole = read (fixture file) in
          (* /nullpad's datatype message (FMT3, IV.A.2.d): version 1 and
             class 3, null padding in ASCII, then a size of 8. *)
          let message = "\x13\x01\x00\x00\x08\x00\x00\x00" in
          let at =
            List.filter
              (fun i -> String.sub whole i 8 = message)
              (List.init (String.length whole - 7) Fun.id)
          in
          let at =
            match at with
            | [ at ] -> at
            | ats -> failf "%d such messages" (List.length ats)
          in
          let b = Bytes.of_string whole in
          Bytes.set b (at + 1) (Char.chr pad);
          let path = Io_support.file (Bytes.to_string b) in
          let d = find (snd (Nx_io.load_hdf5 path)) "/nullpad" in
          let x = H.values d in
          equal packed (H.values (dataset file "/nullpad")) x;
          fails ~path
            ~naming:(Printf.sprintf "/nullpad: string padding %d" pad)
            (fun () -> H.strings d x));
      test "bytes of another dtype or last extent are refused" (fun () ->
          let d = dataset file "/nullpad" in
          invalid ~by:"Nx_io.Hdf5.strings" (fun () ->
              H.strings d (Nx.P (Nx.zeros Nx.uint8 [| 5; 4 |])));
          invalid ~by:"Nx_io.Hdf5.strings" (fun () ->
              H.strings d (Nx.P (Nx.zeros Nx.int8 [| 5; 8 |])));
          invalid ~by:"Nx_io.Hdf5.strings" (fun () ->
              H.strings d (Nx.P (Nx.scalar Nx.uint8 0))));
    ]

let () =
  exit
    (run "nx.io HDF5"
       [
         readings; layouts; attributes; at_load; at_read; checksums; corrupt;
         links; windows; window_refusals; agreement; monotone; only_covered;
         partial_cases; rows_agree; rows_economy; rows_cases; strings_cases;
       ])
