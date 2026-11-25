(* The Zstandard encoder: its levels, the frames it writes and their header
   fields, which do not depend on how the input was cut into steps, the
   whole-buffer frame of an input of at least the level's window differing only
   in its header; frames that fit [bound] and decode to their input at every
   level; the encoder's steps. Header fields are read as RFC 8878 section
   3.1.1.1 lays them out. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib
let levels = List.init 12 (fun i -> i + 1)

(* An encoder's quota, lz.mli's. *)
let quota = 131_072

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

let encode ~level s =
  let n = String.length s in
  let dst = Lz_support.create (Lz.Zstd.bound n) in
  let k =
    require_some ~msg:"fits its bound"
      (Lz.Zstd.encode ~level (Lz_support.of_string s) dst)
  in
  Lz_support.to_string ~length:k dst

(* [decode z n] decodes [z] into exactly [n] bytes, checking that it consumed
   all of [z]. *)
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

(* [steps e ~cuts s] is the frame [e] writes over [s]: step k is given at most
   the k-th of [cuts] bytes from the first byte not yet taken, all that is left
   after [cuts] runs out, with [~last:true] once the slice reaches the end. *)
let steps e ~cuts s =
  let out = Buffer.create (String.length s + 64) in
  let add run = Buffer.add_string out (Lz_support.to_string run) in
  let n = String.length s in
  let rec go at cuts k =
    if k > n + 100 then fail "the steps make no progress";
    let len, rest =
      match cuts with [] -> (n - at, []) | c :: r -> (min c (n - at), r)
    in
    let src = Lz_support.of_string (String.sub s at len) in
    match Lz.Encoder.step e ~last:(at + len = n) src add with
    | Lz.Done _ -> ()
    | More taken -> go (at + taken) rest (k + 1)
  in
  go 0 cuts 0;
  Buffer.contents out

(* The frame a fresh encoder at [level] writes over [s] in one slice. *)
let stream ~level s = steps (Lz.Zstd.encoder ~level ()) ~cuts:[] s

(* Frame headers (RFC 8878 section 3.1.1.1) *)

let descriptor z = Char.code z.[4]
let single z = descriptor z land 0x20 <> 0

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

(* The window size a frame's Window_Descriptor declares, [None] in a frame
   without one: an exponent in its high 5 bits and a mantissa in its low 3, the
   size 2^(10 + exponent) plus mantissa eighths of it. *)
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

(* The kinds of input, and the levels in the four classes of match finder the
   levels use. *)
let cover_input r s level =
  cover "empty" (s = "");
  cover "more than a block" (String.length s > 128 * kib);
  cover "a match" (match r with Lz_support.Match _ -> true | _ -> false);
  cover "noise" (match r with Lz_support.Noise _ -> true | _ -> false);
  cover "a run" (match r with Lz_support.Run _ -> true | _ -> false);
  cover "levels 1 and 2" (level <= 2);
  cover "levels 3 and 4" (level = 3 || level = 4);
  cover "levels 5 to 7" (level >= 5 && level <= 7);
  cover "levels 8 to 12" (level >= 8)

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
   the sanitizers. *)
let large_at level =
  match Lz_support.large with
  | [ block; frame; longest ] ->
      if level >= 9 then [ block; frame; longest ] else [ block; frame ]
  | _ -> invalid_arg "test_zstd_enc: Lz_support.large"

(* A test a level, so that no test runs the large inputs at every level under
   one timeout. *)
let at_level name f =
  group name
    (List.map
       (fun level -> test (Printf.sprintf "level %d" level) (fun () -> f level))
       levels)

(* One input of each kind [cover_input] covers, the [i]th at level [i + 1], so
   that each kind and the levels below 8, whose classes are drawn least, hold on
   every run. The corpus is past a block, 128 KiB + 1, so the edges [cover_edge]
   covers hold too, and past the quota. *)
let kinds = List.mapi (fun i r -> (r, i + 1)) Lz_support.every_kind

(* Step sizes: below a byte's worth, short, and around the block and the
   quota. *)
let cut =
  Gen.frequency
    [
      (1, Gen.int_range 0 0);
      (2, Gen.int_range 1 300);
      (2, Gen.int_range 100_000 300_000);
    ]

let cuts = Gen.list ~size:(Gen.int_range 0 8) cut

(* Levels *)

let levels_outside =
  cases "a level outside 1 to 12 raises" ~id:string_of_int [ -1; 0; 13 ]
    (fun level ->
      let src = Lz_support.of_string "abc" and dst = Lz_support.create 64 in
      raises_match ~msg:"encode" (Exn.invalid_arg ?sub:None) (fun () ->
          Lz.Zstd.encode ~level src dst);
      raises_match ~msg:"encode_string" (Exn.invalid_arg ?sub:None) (fun () ->
          Lz.Zstd.encode_string ~level "abc");
      raises_match ~msg:"encoder" (Exn.invalid_arg ?sub:None) (fun () ->
          Lz.Zstd.encoder ~level ()))

let default_level =
  test "the default level is 3" (fun () ->
      let s = Lz_support.corpus Text (64 * kib) in
      let z = Lz.Zstd.encode_string s in
      equal string ~msg:"encode_string" (Lz.Zstd.encode_string ~level:3 s) z;
      equal string ~msg:"encode" (encode ~level:3 s)
        (let dst = Lz_support.create (Lz.Zstd.bound (String.length s)) in
         let k =
           require_some ~msg:"fits its bound"
             (Lz.Zstd.encode (Lz_support.of_string s) dst)
         in
         Lz_support.to_string ~length:k dst);
      equal string ~msg:"encoder" (stream ~level:3 s)
        (steps (Lz.Zstd.encoder ()) ~cuts:[] s);
      List.iter
        (fun level ->
          if level <> 3 then
            not_equal string
              ~msg:
                (Printf.sprintf
                   "level %d's frame, which must differ for the input to tell \
                    level 3 apart"
                   level)
              (Lz.Zstd.encode_string ~level s)
              z)
        levels)

(* Frames *)

(* A stream's frame of more than 128 KiB also declares its level's window, which
   [declared_window] states at every level. *)
let content_sizes =
  prop "encode's frame states its content size, a stream's its window"
    ~examples:kinds (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      let n = String.length s in
      cover_input r s level;
      equal (option int) ~msg:"encode's Frame_Content_Size" (Some n)
        (content_size (encode ~level s));
      let zs = stream ~level s in
      equal (option int) ~msg:"a stream's" None (content_size zs);
      if n > 128 * kib then
        equal (option int) ~msg:"a stream's Window_Descriptor"
          (Some (window level))
          (window_size zs))

let checksums =
  prop "every frame ends in Content_Checksum, XXH64's low 32 bits"
    ~examples:kinds (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      cover_input r s level;
      let digest = Int64.logand (Lz.Xxh64.string s) 0xFFFFFFFFL in
      List.iter
        (fun (name, z) ->
          equal int
            ~msg:(name ^ "'s Content_Checksum_Flag")
            0x04
            (descriptor z land 0x04);
          let tail = String.get_int32_le z (String.length z - 4) in
          equal int64
            ~msg:(name ^ "'s last 4 bytes, little-endian")
            digest
            (Int64.logand (Int64.of_int32 tail) 0xFFFFFFFFL))
        [ ("encode's frame", encode ~level s); ("a stream's", stream ~level s) ])

let declared_window =
  cases "a stream's frame of more than 128 KiB declares its level's window"
    ~id:(fun level -> Printf.sprintf "level %d" level)
    levels
    (fun level ->
      let s = Lz_support.corpus Columns ((128 * kib) + 1) in
      equal (option int) ~msg:"its Window_Descriptor"
        (Some (window level))
        (window_size (stream ~level s)))

(* An input of 5 MiB, 8 blocks past the largest window, 4 MiB, and encode's
   frame and a stream's of it at every level, each written once for [wide] and
   [rebased]. Its columns and text recur 3 MiB back, so matches reach across the
   window. *)
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
         (encode ~level:(i + 1) s, stream ~level:(i + 1) s)))

let wide =
  at_level "past the window, encode's frame is a stream's but for its header"
    (fun level ->
      let z, zs = Lazy.force wide_frames.(level - 1) in
      equal string ~msg:"the blocks and the checksum" (body z) (body zs))

(* Inputs of the level's window size and a byte more, at a level of each match
   finder and of each window size. Each is a prefix of the wide input. *)
let window_edges =
  List.concat_map
    (fun level ->
      let w = window level in
      [ (level, w); (level, w + 1) ])
    [ 1; 2; 3; 5; 6; 8; 9; 12 ]

let edges =
  group "from the window on, encode's frame is a stream's but for its header"
    (List.map
       (fun (level, n) ->
         test (Printf.sprintf "level %d, %d bytes" level n) (fun () ->
             let s = String.sub (Lazy.force wide_input) 0 n in
             equal string ~msg:"the blocks and the checksum"
               (body (encode ~level s))
               (body (stream ~level s))))
       window_edges)

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
        [ ("encode's frame", encode ~level s); ("a stream's", stream ~level s) ])

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
  at_level "a destination of bound n holds the frame past the window"
    (fun level ->
      List.iter
        (fun r -> fits ~level ~offset:1 (Lz_support.build r))
        (large_at level))

let exact_fit =
  prop "encode fits a destination exactly when it holds the frame"
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
      equal (option int)
        ~msg:(Printf.sprintf "%d bytes of %d" c k)
        None
        (Lz.Zstd.encode ~level src (snd g));
      untouched g;
      equal (option int) ~msg:"exactly its length" (Some k)
        (Lz.Zstd.encode ~level src (Lz_support.create k)))

let negative =
  test "a negative size has no bound" (fun () ->
      raises_match (Exn.invalid_arg ?sub:None) (fun () -> Lz.Zstd.bound (-1)))

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
  at_level "decode (encode x) = x, whole and stepped, past the window"
    (fun level ->
      List.iter
        (fun r -> round_trips ~level ~offset:1 (Lz_support.build r))
        (large_at level))

(* encode_string *)

let encode_string =
  prop "encode_string writes encode's frame" ~examples:kinds
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      cover_input r s level;
      equal string (encode ~level s) (Lz.Zstd.encode_string ~level s))

(* Encoders *)

let stepped_round_trip =
  prop "a stepped encoder's frame decodes, whole and stepped"
    ~examples:(List.map (fun (r, l) -> (r, l, [])) kinds)
    (Gen.triple Lz_support.recipe level cuts)
    (fun (r, level, cuts) ->
      let s = Lz_support.build r in
      cover_input r s level;
      let z = steps (Lz.Zstd.encoder ~level ()) ~cuts s in
      equal Lz_support.decoded (Ok s) (decode z (String.length s));
      equal Lz_support.decoded (Ok s) (decode_stepped z))

(* [quota_steps e s] steps [e] over [s] with [~last:true], each step given all
   that is left: the [k] of every [More k], then [Done]'s. *)
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

let quotas =
  test "every step takes the quota until the rest fits in one" (fun () ->
      let s = Lz_support.corpus Text mib in
      match quota_steps (Lz.Zstd.encoder ()) s with
      | [], _ -> fail "Done before the last byte was taken"
      | ks, last ->
          equal (list int) ~msg:"the steps before the last"
            (List.map (fun _ -> quota) ks)
            ks;
          equal int ~msg:"Done takes the rest"
            (mib - (quota * List.length ks))
            last;
          at_most int ~msg:"the rest is at most the quota" ~than:quota last)

(* [step_rules e plan s] steps [e] over [s], step k given at most the k-th of
   [plan]'s sizes with its [~last], then the rest with [~last:true] until Done,
   and checks that each step takes all of its [src] up to the quota and returns
   Done exactly when it had [~last:true] and took all of its [src]. *)
let step_rules e plan s =
  let n = String.length s in
  let rec go at plan k =
    if k > n + 100 then fail "the steps make no progress";
    let len, last, rest =
      match plan with
      | [] -> (n - at, true, [])
      | (c, last) :: rest -> (min c (n - at), last, rest)
    in
    let src = Lz_support.of_string (String.sub s at len) in
    let got = Lz.Encoder.step e ~last src ignore in
    let taken = match got with Lz.Done j | More j -> j in
    equal int ~msg:"takes all of src up to the quota" (min len quota) taken;
    cover "a ~last:true step that takes only the quota" (last && taken < len);
    cover "a ~last:true step on no input" (last && len = 0);
    cover "a ~last:false step that takes all of src" ((not last) && taken = len);
    cover "a ~last:false step past the quota" ((not last) && len > quota);
    cover "Done before the input's end" (last && taken = len && at + len < n);
    equal Lz_support.step
      ~msg:(Printf.sprintf "step %d, %d bytes, ~last:%b" k len last)
      (if last && taken = len then Lz.Done len else More taken)
      got;
    match (got, plan) with Done _, [] -> () | _ -> go (at + taken) rest (k + 1)
  in
  go 0 plan 0

let only_done =
  prop
    "a step takes src up to the quota, Done only with ~last:true on all of it"
    ~examples:
      [
        (* The quota of 128 KiB + 1 with ~last:false, then a byte. *)
        (Lz_support.Corpus (Columns, (128 * kib) + 1), 5, [ (200_000, false) ]);
        (* The quota of 128 KiB + 1 with ~last:true, then a byte. *)
        (Lz_support.Corpus (Columns, (128 * kib) + 1), 3, [ (200_000, true) ]);
        (* An empty stream before the input's one byte, then the byte with
           ~last:false. *)
        (Byte 'a', 1, [ (0, false); (0, true); (1, false) ]);
      ]
    (Gen.triple Lz_support.recipe level
       (Gen.list ~size:(Gen.int_range 0 8) (Gen.pair cut Gen.bool)))
    (fun (r, level, plan) ->
      step_rules (Lz.Zstd.encoder ~level ()) plan (Lz_support.build r))

let after_done =
  let past = Lz_support.Corpus (Columns, (128 * kib) + 1) in
  prop "after Done, the next step begins a new frame"
    ~examples:[ (past, Lz_support.Byte 'a', 9); (Byte 'a', past, 12) ]
    (Gen.triple Lz_support.recipe Lz_support.recipe level)
    (fun (ra, rb, level) ->
      let a = Lz_support.build ra and b = Lz_support.build rb in
      let block = 128 * kib in
      cover "a stream past a block, then a smaller one"
        (String.length a > block && String.length b <= block);
      cover "a stream of at most a block, then a larger one"
        (String.length a <= block && String.length b > block);
      let e = Lz.Zstd.encoder ~level () in
      let za = steps e ~cuts:[] a in
      let zb = steps e ~cuts:[] b in
      equal string ~msg:"as a fresh encoder writes it" (stream ~level b) zb;
      equal Lz_support.decoded
        (Ok (a ^ b))
        (decode (za ^ zb) (String.length a + String.length b)))

let cut_free =
  prop "a stream's bytes do not depend on how its input is cut"
    ~examples:(List.map (fun (r, l) -> ((r, l, [ 1; 0; 131_073 ]), [])) kinds)
    (Gen.pair (Gen.triple Lz_support.recipe level cuts) cuts)
    (fun ((r, level, a), b) ->
      let s = Lz_support.build r in
      cover_input r s level;
      cover "a step of no bytes" (List.mem 0 a);
      cover "an input past the quota cut two ways"
        (String.length s > quota && a <> b);
      equal string ~msg:"two cuts"
        (steps (Lz.Zstd.encoder ~level ()) ~cuts:a s)
        (steps (Lz.Zstd.encoder ~level ()) ~cuts:b s))

(* Paths *)

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

(* The bytes do not depend on where the match finders rebase their indices,
   which the default limit reaches only 3 GiB into a frame. The wide input's
   recurring columns and text make matches reach across rebases. A limit of 0
   rebases every block past the window; 1.5 MiB rebases every few blocks at
   levels 1 and 2, whose windows of 512 KiB and 1 MiB are below it, and at the
   other levels acts as 0. A stream is stepped 100,000 bytes at a time. *)
let rebased =
  at_level "the index limit changes no byte" (fun level ->
      let s = Lazy.force wide_input in
      let z, zs = Lazy.force wide_frames.(level - 1) in
      let limits = if level <= 2 then [ 0; 3 * mib / 2 ] else [ 0 ] in
      List.iter
        (fun limit ->
          let msg what = Printf.sprintf "limit %d, %s" limit what in
          let dst = Lz_support.create (Lz.Zstd.bound (String.length s)) in
          let k =
            require_some ~msg:(msg "fits its bound")
              (Lz.zstd_encode_limit ~limit ~level (Lz_support.of_string s) dst)
          in
          equal string ~msg:(msg "encode's") z
            (Lz_support.to_string ~length:k dst);
          equal string ~msg:(msg "a stream's") zs
            (steps
               (Lz.zstd_encoder_limit ~limit ~level)
               ~cuts:(List.init 60 (fun _ -> 100_000))
               s))
        limits)

let () =
  exit
    (run "Zstd_enc"
       [
         group "levels" [ levels_outside; default_level ];
         group "frames"
           [
             content_sizes;
             checksums;
             declared_window;
             wide;
             edges;
             declared_windows;
           ];
         group "bound"
           [
             bound;
             large_bound;
             exact_fit;
             negative;
             test "a bound past max_int raises" (fun () ->
                 Lz_support.largest_bound Lz.Zstd.bound);
           ];
         group "encode" [ round_trip; large_round_trip ];
         group "encode_string" [ encode_string ];
         group "encoder"
           [ stepped_round_trip; quotas; only_done; after_done; cut_free ];
         group "paths"
           [
             test "the generic path is first" (fun () ->
                 equal (option string) (Some "generic")
                   (match Lz.Paths.zstd_encode with
                   | (label, _) :: _ -> Some label
                   | [] -> None));
             paths;
             rebased;
           ];
       ])
