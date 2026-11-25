(* The Zstandard encoder: its frames decode to their input at every level,
   fit [bound], and do not depend on how the input was cut into steps, the
   whole-buffer frame of an input of at least the level's window differing
   only in its header. Header fields are read as RFC 8878 section 3.1.1.1
   lays them out. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib
let levels = List.init 12 (fun i -> i + 1)

(* Buffers *)

let guard = 64

(* A destination of [n] bytes at an odd offset, followed by [guard] bytes
   that the encoder must not write. *)
let guarded n =
  let whole = Lz_support.create (3 + n + guard) in
  (whole, Bigarray.Array1.sub whole 3 n)

let untouched (whole, view) =
  let n = Bigarray.Array1.dim view in
  equal string ~msg:"the bytes past the destination"
    (String.make guard '\xa5')
    (String.init guard (fun i ->
         Char.chr (Bigarray.Array1.get whole (3 + n + i))))

let encode ~level s =
  let n = String.length s in
  let dst = Lz_support.create (Lz.Zstd.bound n) in
  let k =
    require_some ~msg:"fits its bound"
      (Lz.Zstd.encode ~level (Lz_support.of_string s) dst)
  in
  Lz_support.to_string ~length:k dst

(* [decode z n] decodes [z] into exactly [n] bytes, checking that it
   consumed all of [z]. *)
let decode z n =
  let dst = Lz_support.create n in
  match Lz.Zstd.decode (Lz_support.of_string z) dst with
  | Error e -> Error e
  | Ok p ->
      equal Lz_support.progress ~msg:"consumed and produced"
        { Lz.consumed = String.length z; produced = n }
        p;
      Ok (Lz_support.to_string ~length:p.produced dst)

(* A decoder with the default window, over slices of 1 byte to 70 KB. *)
let decode_stepped z =
  Result.map fst
    (Lz_support.stepped
       ~cut:(fun k -> 1 + (k * 7919 mod 70_000))
       (Lz.Zstd.decoder ()) z)

(* [steps ~level ~cuts s] is the frame an encoder writes over [s]: step k
   is given at most the k-th of [cuts] bytes from the first byte not yet
   taken, all that is left after [cuts] runs out, with [~last:true] once
   the slice reaches the end. *)
let steps ?e ~level ~cuts s =
  let e = match e with Some e -> e | None -> Lz.Zstd.encoder ~level () in
  let out = Buffer.create (String.length s + 64) in
  let add run = Buffer.add_string out (Lz_support.to_string run) in
  let n = String.length s in
  let rec go at cuts k =
    if k > n + 100 then fail "the steps make no progress";
    let len, rest =
      match cuts with [] -> (n - at, []) | c :: r -> (min c (n - at), r)
    in
    let last = at + len = n in
    let src = Lz_support.of_string (String.sub s at len) in
    match Lz.Encoder.step e ~last src add with
    | Lz.Done taken -> equal int ~msg:"Done takes the whole slice" len taken
    | More taken ->
        if last && taken = len then fail "More after taking the last byte";
        go (at + taken) rest (k + 1)
  in
  go 0 cuts 0;
  Buffer.contents out

(* Frame headers (RFC 8878 section 3.1.1.1) *)

let descriptor z = Char.code z.[4]
let single z = descriptor z land 0x20 <> 0
let checksummed z = descriptor z land 0x04 <> 0

let fcs_length z =
  match descriptor z lsr 6 with
  | 0 -> if single z then 1 else 0
  | 1 -> 2
  | 2 -> 4
  | _ -> 8

let header_length z = 5 + (if single z then 0 else 1) + fcs_length z

let content_size z =
  let at = 5 + if single z then 0 else 1 in
  let le k =
    let v = ref 0 in
    for i = k - 1 downto 0 do
      v := (!v lsl 8) lor Char.code z.[at + i]
    done;
    !v
  in
  match fcs_length z with
  | 0 -> None
  | 2 -> Some (le 2 + 256)
  | k -> Some (le k)

let body z =
  let h = header_length z in
  String.sub z h (String.length z - h)

(* The window size a frame's Window_Descriptor declares, [None] in a
   frame without one: an exponent in its high 5 bits and a mantissa in
   its low 3, the size 2^(10 + exponent) plus mantissa eighths of it. *)
let window_size z =
  if single z then None
  else
    let d = Char.code z.[5] in
    let base = 1 lsl (10 + (d lsr 3)) in
    Some (base + (base / 8 * (d land 7)))

(* A level's window size, as lz.mli states it. *)
let window level =
  if level = 1 then 512 * kib
  else if level = 2 then mib
  else if level <= 8 then 2 * mib
  else 4 * mib

(* Drawn inputs *)

let cover_input r s level =
  cover "empty" (s = "");
  cover "more than a block" (String.length s > 128 * kib);
  cover "a match" (match r with Lz_support.Match _ -> true | _ -> false);
  cover "noise" (match r with Lz_support.Noise _ -> true | _ -> false);
  cover "a run" (match r with Lz_support.Run _ -> true | _ -> false);
  cover "fast, levels 1 and 2" (level <= 2);
  cover "double fast, levels 3 and 4" (level = 3 || level = 4);
  cover "greedy and lazy, levels 5 to 7" (level >= 5 && level <= 7);
  cover "lazy2, levels 8 to 12" (level >= 8)

let level = Gen.int_range 1 12
let placed = Gen.triple Lz_support.recipe level (Gen.int_range 0 7)

(* Lengths at a block's edges, each with its neighbours: 128 KiB, the largest
   block, and multiples of 8 KiB, the shortest block a cut leaves but the last.
   Noise makes raw blocks, the bound's worst case. The bound grows at each
   multiple of 8 KiB alike, so multiples to 136 KiB, past one block, reach every
   edge: longer ones cost their length in every encode and reach no other. *)
let block_edge =
  let open Gen in
  let length =
    frequency
      [
        (1, of_list [ (128 * kib) - 1; 128 * kib; (128 * kib) + 1 ]);
        ( 1,
          map
            (fun (k, d) -> (k * 8 * kib) + d)
            (pair (int_range 1 17) (int_range (-1) 1)) );
      ]
  in
  let fill = of_list [ `Noise; `Text; `Columns ] in
  with_pp Lz_support.pp_recipe
    (map
       (fun ((fill, n), seed) ->
         match fill with
         | `Noise -> Lz_support.Noise (seed, n)
         | `Text -> Corpus (Text, n)
         | `Columns -> Corpus (Columns, n))
       (pair (pair fill length) small_int))

let cover_edge s =
  let n = String.length s in
  cover "128 KiB or a neighbour" (abs (n - (128 * kib)) <= 1);
  cover "a multiple of 8 KiB or a neighbour"
    (n > 1 && (n + 1) mod (8 * kib) <= 2)

(* The drawn inputs and, one case in four, a length at a block's edge. *)
let edged =
  Gen.triple
    (Gen.frequency [ (3, Lz_support.recipe); (1, block_edge) ])
    level (Gen.int_range 0 7)

(* The large inputs, so that each run reaches every level past its window
   (lib/zstd_encode.c): 128 KiB + 1, a block and a byte, and 4 MiB + 1, past
   every window and by one byte past the 4 MiB of levels 9 to 12, at every
   level; the longest, 6 MiB with a match 1 MiB back, at levels 9 to 12, whose
   windows it passes by most. Every input at every level cost 30 % more under
   the sanitizers. A law over them is a test a level, so that no test runs them
   all. *)
let large_at level =
  match Lz_support.large with
  | [ block; frame; longest ] ->
      if level >= 9 then [ block; frame; longest ] else [ block; frame ]
  | _ -> invalid_arg "test_zstd_enc: Lz_support.large"

let at_levels name f =
  group name
    (List.map
       (fun level ->
         test (Printf.sprintf "level %d" level) (fun () ->
             List.iter
               (fun r -> f ~level ~offset:1 (Lz_support.build r))
               (large_at level)))
       levels)

(* One input of each kind [cover_input] covers, the [i]th at level [i + 1], so
   that each kind and the levels below 8, whose classes are drawn least, hold on
   every run. The corpus is past a block, 128 KiB + 1, so the edges [cover_edge]
   covers hold too. *)
let kinds = List.mapi (fun i r -> (r, i + 1)) Lz_support.every_kind

(* Step sizes: below a byte's worth, short, and around the block and the
   quota. *)
let cuts =
  Gen.list ~size:(Gen.int_range 0 8)
    (Gen.frequency
       [
         (1, Gen.int_range 0 0);
         (2, Gen.int_range 1 300);
         (2, Gen.int_range 100_000 300_000);
       ])

(* Levels *)

let levels_outside =
  cases "a level outside 1 to 12 raises" ~id:string_of_int [ -1; 0; 13 ]
    (fun level ->
      let src = Lz_support.of_string "abc" and dst = Lz_support.create 64 in
      raises_match ~msg:"encode" (Exn.invalid_arg ~sub:"level") (fun () ->
          Lz.Zstd.encode ~level src dst);
      raises_match ~msg:"encode_string" (Exn.invalid_arg ~sub:"level")
        (fun () -> Lz.Zstd.encode_string ~level "abc");
      raises_match ~msg:"encoder" (Exn.invalid_arg ~sub:"level") (fun () ->
          Lz.Zstd.encoder ~level ()))

(* Round trips *)

let round_trips ~level ~offset s =
  let n = String.length s in
  let src = Lz_support.placed ~offset s in
  let dst = Lz_support.room ~offset (Lz.Zstd.bound n) in
  let k = require_some ~msg:"fits its bound" (Lz.Zstd.encode ~level src dst) in
  let z = Lz_support.to_string ~length:k dst in
  equal Lz_support.decoded (Ok s) (decode z n);
  equal Lz_support.decoded (Ok s) (decode_stepped z)

let round_trip =
  prop
    ~examples:(List.map (fun (r, l) -> (r, l, 0)) kinds)
    "decode (encode x) = x, whole and stepped" placed
    (fun (r, level, offset) ->
      let s = Lz_support.build r in
      cover_input r s level;
      round_trips ~level ~offset s)

let large_round_trip =
  at_levels "decode (encode x) = x, whole and stepped, past the window"
    round_trips

let strings =
  prop ~examples:kinds "decode (encode_string x) = x"
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      cover_input r s level;
      equal Lz_support.decoded (Ok s)
        (decode (Lz.Zstd.encode_string ~level s) (String.length s)))

let stepped_round_trip =
  prop "a stepped encoder's frame decodes, whole and stepped"
    ~examples:(List.map (fun (r, l) -> (r, l, [])) kinds)
    (Gen.triple Lz_support.recipe level cuts)
    (fun (r, level, cuts) ->
      let s = Lz_support.build r in
      cover_input r s level;
      let z = steps ~level ~cuts s in
      equal Lz_support.decoded (Ok s) (decode z (String.length s));
      equal Lz_support.decoded (Ok s) (decode_stepped z))

(* The bound *)

let fits ~level ~offset s =
  let g = guarded (Lz.Zstd.bound (String.length s)) in
  ignore
    (require_some (Lz.Zstd.encode ~level (Lz_support.placed ~offset s) (snd g)));
  untouched g

let bound =
  prop
    ~examples:(List.map (fun (r, l) -> (r, l, 0)) kinds)
    "a destination of bound n holds the frame" edged
    (fun (r, level, offset) ->
      let s = Lz_support.build r in
      cover_input r s level;
      cover_edge s;
      fits ~level ~offset s)

let large_bound =
  at_levels "a destination of bound n holds the frame past the window" fits

let short =
  prop "a destination short of the frame is refused"
    ~examples:
      (List.concat_map
         (fun e -> List.map (fun (r, l) -> ((r, l, 0), e)) kinds)
         Lz_support.short_edges)
    (Gen.pair edged Lz_support.short)
    (fun ((r, level, offset), short) ->
      let s = Lz_support.build r in
      cover_input r s level;
      cover_edge s;
      let k = String.length (encode ~level s) in
      let c = Lz_support.capacity k short in
      cover "one byte short" (c = k - 1);
      cover "an empty destination" (c = 0);
      let src = Lz_support.placed ~offset s in
      let g = guarded c in
      is_none ~msg:(Printf.sprintf "%d bytes of %d" c k)
        (Lz.Zstd.encode ~level src (snd g));
      untouched g;
      equal (option int) ~msg:"exactly its length" (Some k)
        (Lz.Zstd.encode ~level src (Lz_support.create k)))

let negative =
  test "a negative size has no bound" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"negative") (fun () ->
          Lz.Zstd.bound (-1)))

(* Cuts into steps *)

let cut_free =
  prop "any cut into steps writes one frame"
    ~examples:(List.map (fun (r, l) -> ((r, l, []), [])) kinds)
    (Gen.pair (Gen.triple Lz_support.recipe level cuts) cuts)
    (fun ((r, level, a), b) ->
      let s = Lz_support.build r in
      cover_input r s level;
      cover "a step of no bytes" (List.mem 0 a);
      cover "a step past the quota" (List.exists (fun c -> c > 128 * kib) a);
      equal string ~msg:"two cuts" (steps ~level ~cuts:a s)
        (steps ~level ~cuts:b s))

(* An input of 5 MiB, 8 blocks past the largest window, 4 MiB, and
   encode's frame and a stream's of it at every level, each written once
   for [wide] and [rebased]. Its columns and text recur 3 MiB back, so
   matches reach across the window. *)
let wide_input =
  lazy
    (Lz_support.build
       (Concat
          [
            Corpus (Columns, mib);
            Corpus (Text, 2 * mib);
            Corpus (Columns, mib);
            Corpus (Text, mib);
          ]))

let wide_frames =
  Array.init 12 (fun i ->
      lazy
        (let s = Lazy.force wide_input in
         (encode ~level:(i + 1) s, steps ~level:(i + 1) ~cuts:[] s)))

let at_level name f =
  group name
    (List.map
       (fun level -> test (Printf.sprintf "level %d" level) (fun () -> f level))
       levels)

let wide =
  at_level "past the window, encode's frame is a stream's but for its header"
    (fun level ->
      let z, zs = Lazy.force wide_frames.(level - 1) in
      equal string ~msg:"the blocks and the checksum" (body z) (body zs))

(* Inputs of the level's window size and a byte either side, at a level of
   each strategy and of each window size, and of a block and a byte, the
   least whose stream declares its level's window, at every level. Each
   is a prefix of the wide input. *)
let window_edges =
  List.concat_map
    (fun level ->
      let w = window level in
      [ (level, w - 1); (level, w); (level, w + 1) ])
    [ 1; 2; 3; 5; 6; 8; 9; 12 ]
  @ List.map (fun level -> (level, (128 * kib) + 1)) levels

let edges =
  group "from the window on, encode's frame is a stream's but for its header"
    (List.map
       (fun (level, n) ->
         test (Printf.sprintf "level %d, %d bytes" level n) (fun () ->
             let s = String.sub (Lazy.force wide_input) 0 n in
             let z = encode ~level s and zs = steps ~level ~cuts:[] s in
             equal Lz_support.decoded ~msg:"encode's frame" (Ok s) (decode z n);
             equal Lz_support.decoded ~msg:"the stream's" (Ok s) (decode zs n);
             equal (option int) ~msg:"the stream's Window_Descriptor"
               (Some (window level))
               (window_size zs);
             if n >= window level then
               equal string ~msg:"the blocks and the checksum" (body z)
                 (body zs)))
       window_edges)

(* What the writer writes *)

let headers =
  prop "encode states the content size, a stream never, both the checksum"
    ~examples:kinds (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      let n = String.length s in
      let z = encode ~level s and zs = steps ~level ~cuts:[] s in
      equal (option int) ~msg:"encode's Frame_Content_Size" (Some n)
        (content_size z);
      equal (option int) ~msg:"the stream's" None (content_size zs);
      equal bool ~msg:"the stream's Single_Segment_Flag" false (single zs);
      cover "more than a block" (n > 128 * kib);
      if n > 128 * kib then
        equal (option int) ~msg:"the stream's Window_Descriptor"
          (Some (window level)) (window_size zs);
      let digest = Int64.logand (Lz.Xxh64.string s) 0xFFFFFFFFL in
      List.iter
        (fun z ->
          equal bool ~msg:"Content_Checksum_Flag" true (checksummed z);
          let tail = String.get_int32_le z (String.length z - 4) in
          equal int64 ~msg:"the low 32 bits of XXH64, little-endian" digest
            (Int64.logand (Int64.of_int32 tail) 0xFFFFFFFFL))
        [ z; zs ])

let empty =
  [
    test "encode_string \"\" is a frame of nothing" (fun () ->
        let z = Lz.Zstd.encode_string "" in
        equal Lz_support.decoded (Ok "") (decode z 0);
        equal (option int) (Some 0) (content_size z));
    test "a fresh encoder stepped with ~last:true and nothing writes a frame"
      (fun () ->
        let z = steps ~level:3 ~cuts:[] "" in
        equal Lz_support.decoded (Ok "") (decode z 0));
  ]

(* [quota_steps e s] steps [e] over [s] with [~last:true], each step given
   all that is left: the [k] of every [More k], then [Done]'s. *)
let quota_steps e s =
  let n = String.length s in
  let all = Lz_support.of_string s in
  let rec go at ks =
    let rest = Bigarray.Array1.sub all at (n - at) in
    match Lz.Encoder.step e ~last:true rest ignore with
    | Lz.Done k -> (List.rev ks, k)
    | More 0 -> fail "a step took nothing"
    | More k -> go (at + k) (k :: ks)
  in
  go 0 []

let encode_strings =
  test "encode_string writes encode's frame" (fun () ->
      List.iter
        (fun (r, level) ->
          let s = Lz_support.build r in
          equal string ~msg:(string_of_int level) (encode ~level s)
            (Lz.Zstd.encode_string ~level s))
        kinds)

let default_level =
  test "the default level is 3" (fun () ->
      let s = Lz_support.corpus Text (64 * kib) in
      let z = Lz.Zstd.encode_string s in
      equal string (Lz.Zstd.encode_string ~level:3 s) z;
      List.iter
        (fun level ->
          if level <> 3 then
            not_equal string
              ~msg:(Printf.sprintf "level %d" level)
              (Lz.Zstd.encode_string ~level s)
              z)
        levels)

(* A frame's window size is its Window_Descriptor's, or its content size in a
   single segment. A run of 4 MiB + 1, past every level's window. *)
let declared_windows =
  cases "every frame declares a window of at most 4 MiB"
    ~id:(fun level -> Printf.sprintf "level %d" level)
    levels
    (fun level ->
      let s = String.make ((4 * mib) + 1) 'r' in
      List.iter
        (fun (name, z) ->
          let declared = if single z then content_size z else window_size z in
          match declared with
          | None -> failf "%s declares no window" name
          | Some w -> at_most int ~msg:name ~than:(4 * mib) w)
        [
          ("encode's frame", encode ~level s);
          ("a stream's", steps ~level ~cuts:[] s);
        ])

let quota =
  test "every step takes the quota until the rest fits in one" (fun () ->
      let s = Lz_support.corpus Text mib in
      match quota_steps (Lz.Zstd.encoder ()) s with
      | [], _ -> fail "Done before the last byte was taken"
      | (q :: _ as ks), last ->
          equal int ~msg:"the quota" 131_072 q;
          List.iteri
            (fun i k -> equal int ~msg:(Printf.sprintf "step %d" i) q k)
            ks;
          equal int ~msg:"Done takes the rest" (mib - (q * List.length ks)) last;
          at_most int ~msg:"the rest is at most the quota" ~than:q last)

let frames =
  prop "after Done, the next step begins a new frame"
    (Gen.triple Lz_support.recipe Lz_support.recipe level)
    (fun (ra, rb, level) ->
      let a = Lz_support.build ra and b = Lz_support.build rb in
      let e = Lz.Zstd.encoder ~level () in
      let za = steps ~e ~level ~cuts:[] a in
      let zb = steps ~e ~level ~cuts:[] b in
      equal string ~msg:"as a fresh encoder writes it"
        (steps ~level ~cuts:[] b) zb;
      equal Lz_support.decoded (Ok (a ^ b))
        (decode (za ^ zb) (String.length a + String.length b)))

(* The bytes do not depend on the instruction set. *)

let paths =
  prop "every path writes the bytes of Lz.Zstd.encode" ~examples:kinds
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      cover_input r s level;
      let expected = encode ~level s in
      List.iter
        (fun (label, encode) ->
          let dst = Lz_support.create (Lz.Zstd.bound (String.length s)) in
          match encode ~level (Lz_support.of_string s) dst with
          | Some k ->
              equal string ~msg:label expected
                (Lz_support.to_string ~length:k dst)
          | None -> failf "%s: the frame exceeds its bound" label)
        Lz.Paths.zstd_encode)

(* [limited_steps e ~step src dst] steps [e] over [src], at most [step] bytes a
   step, and writes the runs to [dst]: the stream's length, or [None] when it
   does not fit. *)
let limited_steps e ~step src dst =
  let n = Bigarray.Array1.dim src in
  let out = ref 0 in
  let add run =
    let k = Bigarray.Array1.dim run in
    if !out + k <= Bigarray.Array1.dim dst then
      Bigarray.Array1.blit run (Bigarray.Array1.sub dst !out k);
    out := !out + k
  in
  let rec go at =
    let len = min step (n - at) in
    let last = at + len = n in
    match Lz.Encoder.step e ~last (Bigarray.Array1.sub src at len) add with
    | Lz.Done _ -> ()
    | More taken -> go (at + taken)
  in
  go 0;
  if !out <= Bigarray.Array1.dim dst then Some !out else None

(* [encode_limit ~limit ~level src dst] is [Lz.Zstd.encode ~level src dst] by
   match finders that rebase past index [limit]; with [~step], the frame an
   encoder writes given at most [step] bytes a step. *)
let encode_limit ~limit ?step ~level src dst =
  match step with
  | None -> Lz.zstd_encode_limit ~limit ~level src dst
  | Some s when s > 0 ->
      limited_steps (Lz.zstd_encoder_limit ~limit ~level) ~step:s src dst
  | Some _ -> invalid_arg "encode_limit: step"

(* The bytes do not depend on where the match finders rebase their
   indices, which the default limit reaches only 3 GiB into a frame. The
   wide input's recurring columns and text make matches reach across
   rebases. A limit of 0 rebases every block past the window; 1.5 MiB
   rebases every few blocks at levels 1 and 2, whose windows of 512 KiB
   and 1 MiB are below it, and at the other levels acts as 0. *)
let rebased =
  at_level "the index limit changes no byte" (fun level ->
      let s = Lazy.force wide_input in
      let z, zs = Lazy.force wide_frames.(level - 1) in
      let src = Lz_support.of_string s in
      let limited ~limit ?step () =
        let dst = Lz_support.create (Lz.Zstd.bound (String.length s)) in
        let k =
          require_some ~msg:"fits its bound"
            (encode_limit ~limit ?step ~level src dst)
        in
        Lz_support.to_string ~length:k dst
      in
      let limits = if level <= 2 then [ 0; 3 * mib / 2 ] else [ 0 ] in
      List.iter
        (fun limit ->
          let msg what = Printf.sprintf "limit %d, %s" limit what in
          equal string ~msg:(msg "encode's") z (limited ~limit ());
          equal string ~msg:(msg "a stream's") zs
            (limited ~limit ~step:100_000 ()))
        limits)

let () =
  exit
    (run "Zstd_enc"
       [
         group "levels" [ levels_outside; default_level ];
         group "round trip"
           ([ round_trip; large_round_trip; stepped_round_trip ] @ empty);
         group "encode_string" [ strings; encode_strings ];
         group "bound"
           [
             bound;
             large_bound;
             short;
             negative;
             test "a bound past max_int raises" (fun () ->
                 Lz_support.largest_bound Lz.Zstd.bound);
           ];
         group "steps" [ cut_free; wide; edges; quota; frames ];
         group "frames" [ headers; declared_windows ];
         group "paths" [ paths; rebased ];
       ])
