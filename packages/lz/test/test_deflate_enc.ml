(* The deflate, zlib and gzip encoders: their streams decode to their input at
   every level, fit [bound], and do not depend on how the input was cut into
   steps, which write encode's bytes exactly. The wrappers' fields are read as
   RFC 1950 section 2.2 and RFC 1952 section 2.3 lay them out. *)

open Windtrap

let kib = 1024
let levels = List.init 9 (fun i -> i + 1)

(* lz.mli: an encoder's quota for deflate, zlib and gzip. *)
let quota = 65_536

(* Deflate, zlib and gzip share one signature. *)
module type Family = module type of Lz.Deflate

let formats : (string * (module Family)) list =
  [
    ("deflate", (module Lz.Deflate));
    ("zlib", (module Lz.Zlib));
    ("gzip", (module Lz.Gzip));
  ]

(* Streams print as their length and digest. *)
let stream =
  Testable.make ~equal:String.equal ~pp:(fun ppf s ->
      Format.fprintf ppf "<%d bytes, md5 %s>" (String.length s)
        (Digest.to_hex (Digest.string s)))

(* Buffers *)

let guard = 64

(* A destination of [n] bytes at an odd offset, followed by [guard] bytes that
   the encoder must not write. *)
let guarded n =
  let whole = Lz_support.create (3 + n + guard) in
  (whole, Bigarray.Array1.sub whole 3 n)

let untouched (whole, view) =
  let n = Bigarray.Array1.dim view in
  equal string ~msg:"the bytes past the destination" (String.make guard '\xa5')
    (String.init guard (fun i ->
         Char.chr (Bigarray.Array1.get whole (3 + n + i))))

let encode (module F : Family) ~level s =
  let dst = Lz_support.create (F.bound (String.length s)) in
  let k =
    require_some ~msg:"fits its bound"
      (F.encode ~level (Lz_support.of_string s) dst)
  in
  Lz_support.to_string ~length:k dst

(* [decode f z n] decodes [z] into exactly [n] bytes, checking that it consumed
   all of [z]. *)
let decode (module F : Family) z n =
  let dst = Lz_support.create n in
  match F.decode (Lz_support.of_string z) dst with
  | Error e -> Error e
  | Ok p ->
      equal Lz_support.progress ~msg:"consumed and produced"
        { Lz.consumed = String.length z; produced = n }
        p;
      Ok (Lz_support.to_string ~length:p.produced dst)

(* [steps e ~cuts s] steps the encoder [e] over [s] as a writer of a stream
   does: step [k] is given at most the [k]th of [cuts] bytes from the first byte
   not yet taken, and all that is left once [cuts] runs out, with [~last:true]
   once the slice reaches [s]'s end. It is the stream and the slices given, as
   [(length, last)]. It fails the test unless the walk ends at a step with
   [~last:true] that takes all of its slice and returns [Done], the only step
   that may (lz.mli). *)
let steps e ~cuts s =
  let n = String.length s in
  let all = Lz_support.of_string s in
  let out = Buffer.create (n + 64) in
  let add run = Buffer.add_string out (Lz_support.to_string run) in
  let rec go at cuts k given =
    if k > n + 100 then fail "the steps make no progress";
    let len, rest =
      match cuts with [] -> (n - at, []) | c :: r -> (min c (n - at), r)
    in
    let last = at + len = n in
    let given = (len, last) :: given in
    match Lz.Encoder.step e ~last (Bigarray.Array1.sub all at len) add with
    | Lz.Done taken when last && taken = len -> List.rev given
    | Done taken -> failf "Done %d on %d bytes with ~last:%b" taken len last
    | More taken when last && taken = len ->
        failf "More %d on the last %d bytes" taken len
    | More taken -> go (at + taken) rest (k + 1) given
  in
  let given = go 0 cuts 0 [] in
  (Buffer.contents out, given)

(* Drawn inputs *)

let format =
  Gen.of_list ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name) formats

let level = Gen.int_range 1 9

let cover_input r s (name, _) level =
  cover "empty" (s = "");
  cover "more than a block of 64 KiB" (String.length s > 64 * kib);
  cover "a match" (match r with Lz_support.Match _ -> true | _ -> false);
  cover "noise" (match r with Lz_support.Noise _ -> true | _ -> false);
  cover "a run" (match r with Lz_support.Run _ -> true | _ -> false);
  cover "deflate" (name = "deflate");
  cover "zlib" (name = "zlib");
  cover "gzip" (name = "gzip");
  cover "the hash table, level 1" (level = 1);
  cover "greedy, levels 2 to 4" (level >= 2 && level <= 4);
  cover "lazy, levels 5 to 7" (level >= 5 && level <= 7);
  cover "lazy2, levels 8 and 9" (level >= 8)

let input = Gen.triple Lz_support.recipe format level
let placed = Gen.pair input (Gen.int_range 0 7)

(* One input of each kind [cover_input] covers, the [i]th in format [i mod 3] at
   level [(i + 5) mod 9 + 1], so that each kind, each format and each class of
   levels holds on every run, and level 1 meets the input past two windows. *)
let kinds =
  List.mapi
    (fun i r -> (r, List.nth formats (i mod 3), ((i + 5) mod 9) + 1))
    Lz_support.every_kind

(* An encoder keeps at most about 300 KiB of input and takes a quota of 64 KiB a
   step (lz.mli), so past 300 KiB and two quotas its window must move: 448 KiB
   of text, at levels 1, 6 and 9. *)
let past_window =
  List.map2
    (fun level f -> (Lz_support.Corpus (Text, 448 * kib), f, level))
    [ 1; 6; 9 ] formats

(* Step sizes: no bytes, short, and around the lookahead and the quota. *)
let cuts =
  Gen.list ~size:(Gen.int_range 0 8)
    (Gen.frequency
       [
         (1, Gen.int_range 0 0);
         (2, Gen.int_range 1 300);
         (2, Gen.int_range 1000 70_000);
         (2, Gen.int_range 100_000 300_000);
       ])

(* encode *)

let round_trip =
  prop "writes one stream holding all of src"
    ~examples:(List.map (fun k -> (k, 0)) (kinds @ past_window))
    placed
    (fun ((r, ((_, (module F : Family)) as f), level), offset) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      let n = String.length s in
      let src = Lz_support.placed ~offset s in
      let dst = Lz_support.room ~offset (F.bound n) in
      let k = require_some ~msg:"fits its bound" (F.encode ~level src dst) in
      equal Lz_support.decoded (Ok s)
        (decode (snd f) (Lz_support.to_string ~length:k dst) n))

let short =
  prop "returns None exactly when dst is shorter than the stream"
    ~examples:
      (List.concat_map
         (fun e -> List.map (fun k -> ((k, 0), e)) kinds)
         Lz_support.short_edges)
    (Gen.pair placed Lz_support.short)
    (fun (((r, ((_, (module F : Family)) as f), level), offset), short) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      let k = String.length (encode (snd f) ~level s) in
      let c = Lz_support.capacity k short in
      cover "one byte short" (c = k - 1);
      cover "an empty destination" (c = 0);
      let src = Lz_support.placed ~offset s in
      let g = guarded c in
      equal (option int)
        ~msg:(Printf.sprintf "%d bytes of %d" c k)
        None
        (F.encode ~level src (snd g));
      untouched g;
      equal (option int) ~msg:"exactly its length" (Some k)
        (F.encode ~level src (Lz_support.create k)))

(* bound *)

let bound =
  prop "a destination of bound n holds the stream"
    ~examples:(List.map (fun k -> (k, 0)) kinds)
    placed
    (fun ((r, ((_, (module F : Family)) as f), level), offset) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      let g = guarded (F.bound (String.length s)) in
      ignore
        (require_some (F.encode ~level (Lz_support.placed ~offset s) (snd g)));
      untouched g)

(* Incompressible input: stored blocks, the bound's worst case, written one byte
   at a time to the destination's end. 4999 to 5001 bytes straddle the shortest
   block the encoder ends early. *)
let noise_bound =
  cases "noise fits its bound at every level"
    ~id:(fun (n, l) -> Printf.sprintf "%d bytes, level %d" n l)
    (List.concat_map
       (fun n -> List.map (fun l -> (n, l)) levels)
       [ 0; 1; 4999; 5000; 5001; 65535; 65536; 70_000; 300_001 ])
    (fun (n, level) ->
      let s = Lz_support.build (Noise (n, n)) in
      List.iter
        (fun (name, (module F : Family)) ->
          let g = guarded (F.bound n) in
          ignore
            (require_some ~msg:name
               (F.encode ~level (Lz_support.of_string s) (snd g)));
          untouched g)
        formats)

let negative =
  cases "a negative size has no bound"
    ~id:(fun ((name, _), size) -> Printf.sprintf "%s, %d" name size)
    (List.concat_map (fun f -> [ (f, -1); (f, min_int) ]) formats)
    (fun ((_, (module F : Family)), size) ->
      raises_match (Exn.invalid_arg ~sub:"negative") (fun () -> F.bound size))

let largest =
  cases "a bound past max_int raises" ~id:fst formats
    (fun (_, (module F : Family)) -> Lz_support.largest_bound F.bound)

(* encode_string *)

let strings =
  prop "writes encode's stream" ~examples:kinds input
    (fun (r, ((_, (module F : Family)) as f), level) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      equal stream (encode (snd f) ~level s) (F.encode_string ~level s))

(* Encoder *)

(* Each row's input is given to a step all that is left, with [~last] as the row
   says, at a level. Seven quotas, 448 KiB, pass the input an encoder keeps,
   about 300 KiB, and two quotas (lz.mli), so its window moves before the last
   steps, at level 1 as at the others. *)
let quota_rows =
  let more = List.init 6 (fun _ -> Lz.More quota) in
  Lz.
    [
      (6, quota, true, [ Done quota ]);
      (6, quota, false, [ More quota ]);
      (6, quota + 1, true, [ More quota; Done 1 ]);
      (6, quota + 1, false, [ More quota; More 1 ]);
      (6, (2 * quota) + 1, true, [ More quota; More quota; Done 1 ]);
      (6, (2 * quota) + 1, false, [ More quota; More quota; More 1 ]);
      (1, 7 * quota, true, more @ [ Done quota ]);
      (6, 7 * quota, true, more @ [ Done quota ]);
    ]

let quota_steps =
  cases "a step takes all of src up to the quota and the quota of a longer src"
    ~id:(fun ((name, _), (level, n, last, _)) ->
      Printf.sprintf "%s, level %d, %d bytes, ~last:%b" name level n last)
    (List.concat_map (fun f -> List.map (fun r -> (f, r)) quota_rows) formats)
    (fun ((_, (module F : Family)), (level, n, last, expected)) ->
      let e = F.encoder ~level () in
      let all = Lz_support.of_string (Lz_support.corpus Text n) in
      let rec go at =
        if at = n && not last then []
        else
          let src = Bigarray.Array1.sub all at (n - at) in
          match Lz.Encoder.step e ~last src ignore with
          | Lz.Done k -> [ Lz.Done k ]
          | More 0 -> [ More 0 ] (* A step that takes nothing ends the walk. *)
          | More k -> More k :: go (at + k)
      in
      equal (list Lz_support.step) expected (go 0))

(* Two streams for one encoder and the level at which the law holds them. Half
   the time, a stream of 64 bytes to 1 KiB after a prefix of 2 to 8 KiB of the
   same corpus. The short stream empties only the heads it hashes. The prefix,
   too long for that, empties every head, then leaves a position in each head
   the short stream reads and in heads past its reach. Otherwise two drawn
   recipes. *)
let stream_pair =
  let kind = Gen.of_list Lz_support.[ Text; Columns; Random ] in
  let short =
    Gen.map
      (fun ((k, na), nb) ->
        (Lz_support.Corpus (k, na), Lz_support.Corpus (k, nb)))
      (Gen.pair
         (Gen.pair kind (Gen.int_range (2 * kib) (8 * kib)))
         (Gen.int_range 64 kib))
  in
  let drawn = Gen.pair Lz_support.recipe Lz_support.recipe in
  Gen.with_pp
    (fun ppf ((a, b), level) ->
      Format.fprintf ppf "%a then %a at level %d" Lz_support.pp_recipe a
        Lz_support.pp_recipe b level)
    (Gen.pair (Gen.frequency [ (1, short); (1, drawn) ]) level)

(* A short stream after a prefix at every level, the formats in turn, so that
   every run covers each level, which drawn cases reach one in 18. *)
let after_prefix =
  List.map
    (fun level ->
      ( ( (Lz_support.Corpus (Text, 4 * kib), Lz_support.Corpus (Text, 512)),
          level ),
        List.nth formats (level mod 3) ))
    levels

let streams =
  prop "after Done, the next step begins a new stream" ~examples:after_prefix
    (Gen.pair stream_pair format)
    (fun (((ra, rb), level), (_, (module F : Family))) ->
      let a = Lz_support.build ra and b = Lz_support.build rb in
      let nb = String.length b in
      let short = nb >= 64 && nb <= kib && String.length a > nb in
      List.iter
        (fun l ->
          cover
            (Printf.sprintf
               "a stream of 64 bytes to 1 KiB after a longer one, at level %d" l)
            (short && level = l))
        levels;
      let e = F.encoder ~level () in
      ignore (steps e ~cuts:[] a);
      equal stream ~msg:"as a fresh encoder writes it"
        (fst (steps (F.encoder ~level ()) ~cuts:[] b))
        (fst (steps e ~cuts:[] b)))

let empty =
  cases "a fresh encoder stepped on no input writes the empty string's stream"
    ~id:(fun ((name, _), level) -> Printf.sprintf "%s, level %d" name level)
    (List.concat_map (fun f -> List.map (fun l -> (f, l)) levels) formats)
    (fun ((_, (module F : Family)), level) ->
      equal stream
        (F.encode_string ~level "")
        (fst (steps (F.encoder ~level ()) ~cuts:[] "")))

(* Formats *)

(* After an empty step, 777 bytes, then the rest in one slice, past the quota on
   the inputs past 64 KiB; on the inputs past the window, 70,000 bytes without
   [~last] first. *)
let cut_examples =
  List.map (fun (r, f, l) -> (r, f, l, [ 0; 777 ])) kinds
  @ List.map (fun (r, f, l) -> (r, f, l, [ 0; 777; 70_000 ])) past_window

let cut_free =
  prop "any cut into steps writes encode's bytes" ~examples:cut_examples
    (Gen.quad Lz_support.recipe format level cuts)
    (fun (r, ((_, (module F : Family)) as f), level, cuts) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      let z, given = steps (F.encoder ~level ()) ~cuts s in
      let given_past_quota ~last =
        List.exists (fun (len, l) -> len > quota && l = last) given
      in
      cover "an empty step before the last"
        (List.exists (fun (len, last) -> len = 0 && not last) given);
      cover "a step without ~last given more than the quota"
        (given_past_quota ~last:false);
      cover "a step with ~last:true given more than the quota"
        (given_past_quota ~last:true);
      equal stream (encode (snd f) ~level s) z)

(* Deflate *)

let levels_outside =
  cases "a level outside 1 to 9 raises"
    ~id:(fun ((name, _), level) -> Printf.sprintf "%s, %d" name level)
    (List.concat_map
       (fun f -> List.map (fun l -> (f, l)) [ min_int; -1; 0; 10; max_int ])
       formats)
    (fun ((_, (module F : Family)), level) ->
      let src = Lz_support.of_string "abc" and dst = Lz_support.create 64 in
      let invalid what g =
        raises_match ~msg:what (Exn.invalid_arg ~sub:"level") g
      in
      invalid "encode" (fun () -> F.encode ~level src dst);
      invalid "encode_string" (fun () -> F.encode_string ~level "abc");
      invalid "encoder" (fun () -> F.encoder ~level ()))

let default_level =
  cases "the default level is 6" ~id:fst formats
    (fun ((_, (module F : Family)) as f) ->
      let s = Lz_support.corpus Text (64 * kib) in
      let at level = encode (snd f) ~level s in
      (* An input on which the levels around 6 write other bytes. *)
      not_equal stream ~msg:"the input tells levels 5 and 6 apart" (at 6) (at 5);
      not_equal stream ~msg:"the input tells levels 7 and 6 apart" (at 6) (at 7);
      let dst = Lz_support.create (F.bound (String.length s)) in
      let k = require_some (F.encode (Lz_support.of_string s) dst) in
      equal stream ~msg:"encode" (at 6) (Lz_support.to_string ~length:k dst);
      equal stream ~msg:"encode_string" (at 6) (F.encode_string s);
      equal stream ~msg:"encoder" (at 6) (fst (steps (F.encoder ()) ~cuts:[] s)))

(* Zlib and gzip *)

let byte z i = Char.code z.[i]
let le32 z i = Int32.to_int (String.get_int32_le z i) land 0xffffffff
let be32 z i = Int32.to_int (String.get_int32_be z i) land 0xffffffff

let zlib_adler =
  prop "a stream ends with the Adler-32 of its input"
    ~examples:(List.map (fun (r, _, l) -> (r, l)) kinds)
    (Gen.pair Lz_support.recipe level)
    (fun (r, level) ->
      let s = Lz_support.build r in
      let z = Lz.Zlib.encode_string ~level s in
      let adler = Int32.to_int (Lz.Adler32.string s) land 0xffffffff in
      equal int ~msg:"ADLER32" adler (be32 z (String.length z - 4)))

(* RFC 1952, 2.3.1: FLG's bits FEXTRA, FNAME and FCOMMENT. *)
let fextra_fname_fcomment = 0x04 lor 0x08 lor 0x10

let gzip_header =
  prop "a member has no name, comment or extra field and MTIME 0"
    ~examples:(List.map (fun (r, _, l) -> (r, l)) kinds)
    (Gen.pair Lz_support.recipe level)
    (fun (r, level) ->
      let z = Lz.Gzip.encode_string ~level (Lz_support.build r) in
      equal int ~msg:"FEXTRA, FNAME and FCOMMENT" 0
        (byte z 3 land fextra_fname_fcomment);
      equal int ~msg:"MTIME" 0 (le32 z 4))

let gzip_trailer =
  prop "a member ends with the CRC-32 and length of its input"
    ~examples:(List.map (fun (r, _, l) -> (r, l)) kinds)
    (Gen.pair Lz_support.recipe level)
    (fun (r, level) ->
      let s = Lz_support.build r in
      let z = Lz.Gzip.encode_string ~level s in
      let t = String.length z - 8 in
      equal int ~msg:"CRC32" (Lz.Crc32.string s) (le32 z t);
      equal int ~msg:"ISIZE" (String.length s land 0xffffffff) (le32 z (t + 4)))

(* Paths *)

(* The match finders' positions rebase every 32 KiB of input, so an input past
   64 KiB rebases with a full window behind it. *)
let paths =
  prop "every path writes the bytes of Lz.Deflate.encode"
    ~examples:
      (List.map (fun (r, _, l) -> (r, l)) kinds
      @ [ (Lz_support.Corpus (Columns, (128 * kib) + 1), 6) ])
    (Gen.pair Lz_support.recipe level)
    (fun (r, level) ->
      let s = Lz_support.build r in
      let past = String.length s > 2 * 32 * kib in
      cover "a rebase at level 1, the hash table" (past && level = 1);
      cover "a rebase at levels 2 to 9, hash chains" (past && level >= 2);
      let expected = encode (module Lz.Deflate) ~level s in
      List.iter
        (fun (label, encode) ->
          let dst = Lz_support.create (Lz.Deflate.bound (String.length s)) in
          match encode ~level (Lz_support.of_string s) dst with
          | Some k ->
              equal stream ~msg:label expected
                (Lz_support.to_string ~length:k dst)
          | None -> failf "%s: the stream exceeds its bound" label)
        Lz.Paths.deflate_encode)

let () =
  exit
    (run "Deflate_enc"
       [
         group "encode" [ round_trip; short ];
         group "bound" [ bound; noise_bound; negative; largest ];
         group "encode_string" [ strings ];
         group "Encoder" [ quota_steps; streams; empty ];
         group "formats" [ cut_free ];
         group "Deflate" [ levels_outside; default_level ];
         group "Zlib" [ zlib_adler ];
         group "Gzip" [ gzip_header; gzip_trailer ];
         group "Paths" [ paths ];
       ])
