(* The Zstandard encoder: its frames decode to their input at every level
   (RFC 0039 law 3), fit [bound] (law 4), and do not depend on how the
   input was cut into steps, the whole-buffer frame differing only in its
   header (law 6). Header fields are read as RFC 8878 section 3.1.1.1
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

(* Lengths at a block's edges, each with its neighbours: 128 KiB, the
   largest block, and multiples of 8 KiB, the shortest block a cut leaves
   but the last. Noise makes raw blocks, the bound's worst case. *)
let block_edge =
  let open Gen in
  let length =
    frequency
      [
        (1, of_list [ (128 * kib) - 1; 128 * kib; (128 * kib) + 1 ]);
        ( 1,
          map
            (fun (k, d) -> (k * 8 * kib) + d)
            (pair (int_range 1 40) (int_range (-1) 1)) );
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

(* Every level over every large input, so that each run reaches all. *)
let large =
  List.concat_map
    (fun r -> List.map (fun l -> (r, l, 1)) levels)
    Lz_support.large

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

(* Law 3 *)

let round_trip =
  prop ~examples:large "decode (encode x) = x, whole and stepped" placed
    (fun (r, level, offset) ->
      let s = Lz_support.build r in
      cover_input r s level;
      let n = String.length s in
      let src = Lz_support.placed ~offset s in
      let dst = Lz_support.room ~offset (Lz.Zstd.bound n) in
      let k =
        require_some ~msg:"fits its bound" (Lz.Zstd.encode ~level src dst)
      in
      let z = Lz_support.to_string ~length:k dst in
      equal Lz_support.decoded (Ok s) (decode z n);
      equal Lz_support.decoded (Ok s) (decode_stepped z))

let strings =
  prop "decode (encode_string x) = x" (Gen.pair Lz_support.recipe level)
    (fun (r, level) ->
      let s = Lz_support.build r in
      cover_input r s level;
      equal Lz_support.decoded (Ok s)
        (decode (Lz.Zstd.encode_string ~level s) (String.length s)))

let stepped_round_trip =
  prop "a stepped encoder's frame decodes, whole and stepped"
    (Gen.triple Lz_support.recipe level cuts) (fun (r, level, cuts) ->
      let s = Lz_support.build r in
      cover_input r s level;
      let z = steps ~level ~cuts s in
      equal Lz_support.decoded (Ok s) (decode z (String.length s));
      equal Lz_support.decoded (Ok s) (decode_stepped z))

(* Law 4 *)

let bound =
  prop ~examples:large "a destination of bound n holds the frame" edged
    (fun (r, level, offset) ->
      let s = Lz_support.build r in
      cover_input r s level;
      cover_edge s;
      let g = guarded (Lz.Zstd.bound (String.length s)) in
      ignore
        (require_some
           (Lz.Zstd.encode ~level (Lz_support.placed ~offset s) (snd g)));
      untouched g)

let short =
  prop "a destination short of the frame is refused"
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

(* Law 6 *)

let cut_free =
  prop "any cut into steps writes one frame, encode's but for its header"
    (Gen.pair (Gen.triple Lz_support.recipe level cuts) cuts)
    (fun ((r, level, a), b) ->
      let s = Lz_support.build r in
      cover_input r s level;
      cover "a step of no bytes" (List.mem 0 a);
      cover "a step past the quota" (List.exists (fun c -> c > 128 * kib) a);
      let za = steps ~level ~cuts:a s in
      equal string ~msg:"two cuts" za (steps ~level ~cuts:b s);
      equal string ~msg:"the blocks and the checksum"
        (body (encode ~level s))
        (body za))

(* What the writer writes *)

let headers =
  prop "encode states the content size, a stream never, both the checksum"
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      let n = String.length s in
      let z = encode ~level s and zs = steps ~level ~cuts:[] s in
      equal (option int) ~msg:"encode's Frame_Content_Size" (Some n)
        (content_size z);
      equal (option int) ~msg:"the stream's" None (content_size zs);
      equal bool ~msg:"the stream's Single_Segment_Flag" false (single zs);
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
        equal Lz_support.decoded (Ok "") (decode z 0);
        equal string ~msg:"encode's but for its header"
          (body (encode ~level:3 ""))
          (body z));
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

let quota =
  test "every step takes the quota until the rest fits in one" (fun () ->
      let s = Lz_support.corpus Text mib in
      match quota_steps (Lz.Zstd.encoder ()) s with
      | [], _ -> fail "Done before the last byte was taken"
      | (q :: _ as ks), last ->
          less int ~msg:"short of the slice" ~than:mib q;
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

(* Law 6: the bytes do not depend on the instruction set. *)

let paths =
  prop "every path writes the bytes of Lz.Zstd.encode"
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      cover_input r s level;
      let expected = encode ~level s in
      List.iter
        (fun (p : Lz_support.zstd_enc_path) ->
          let dst = Lz_support.create (Lz.Zstd.bound (String.length s)) in
          match p.encode ~level (Lz_support.of_string s) dst with
          | Some k ->
              equal string ~msg:p.label expected
                (Lz_support.to_string ~length:k dst)
          | None -> failf "%s: the frame exceeds its bound" p.label)
        Lz_support.zstd_enc_paths)

(* Law 6: the bytes do not depend on where the match finders rebase their
   indices, which the default limit reaches only 3 GiB into a frame. The
   input runs 8 blocks past the largest window, 4 MiB, and its columns
   and text recur 3 MiB back, so matches reach across rebases. A limit of
   0 rebases every block past the window; 1.5 MiB rebases every few blocks
   at levels 1 and 2, whose windows of 512 KiB and 1 MiB are below it, and
   at the other levels acts as 0. *)
let rebased =
  let s =
    lazy
      (Lz_support.build
         (Concat
            [
              Corpus (Columns, mib);
              Corpus (Text, 2 * mib);
              Corpus (Columns, mib);
              Corpus (Text, mib);
            ]))
  in
  cases
    ~name:(fun (level, limit) -> Printf.sprintf "level %d, limit %d" level limit)
    "the index limit changes no byte"
    (List.map (fun level -> (level, 0)) levels
    @ [ (1, 3 * mib / 2); (2, 3 * mib / 2) ])
    (fun (level, limit) ->
      let s = Lazy.force s in
      let src = Lz_support.of_string s in
      let limited ?step () =
        let dst = Lz_support.create (Lz.Zstd.bound (String.length s)) in
        let k =
          require_some ~msg:"fits its bound"
            (Lz_support.zstd_encode_limit ~limit ?step ~level src dst)
        in
        Lz_support.to_string ~length:k dst
      in
      equal string ~msg:"encode's" (encode ~level s) (limited ());
      equal string ~msg:"a stream's"
        (steps ~level ~cuts:[] s)
        (limited ~step:100_000 ()))

let () =
  exit
    (run "Zstd_enc"
       [
         cases ~name:string_of_int "a level outside 1 to 12 raises" [ 0; 13 ]
           (fun level ->
             raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
                 Lz.Zstd.encode_string ~level "abc"));
         group "round trip" [ round_trip; strings; stepped_round_trip ];
         group "bound" [ bound; short ];
         group "steps" [ cut_free; quota; frames ];
         group "paths" [ paths; rebased ];
         group "frames" (headers :: empty);
       ])
