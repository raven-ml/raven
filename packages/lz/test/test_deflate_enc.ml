(* The deflate, zlib and gzip encoders: their streams decode to their input
   at every level (RFC 0039 law 3), fit [bound] (law 4), and do not depend
   on how the input was cut into steps, which write encode's bytes exactly
   (law 6). The wrappers' fields are read as RFC 1950 section 2.2 and RFC
   1952 section 2.3 lay them out. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib
let levels = List.init 9 (fun i -> i + 1)

(* Formats *)

type format = {
  name : string;
  encode : ?level:int -> Lz.bigbytes -> Lz.bigbytes -> int option;
  encode_string : ?level:int -> string -> string;
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
  bound : int -> int;
  encoder : ?level:int -> unit -> Lz.Encoder.t;
  decoder : unit -> Lz.Decoder.t;
}

let deflate =
  Lz.Deflate.
    { name = "deflate"; encode; encode_string; decode; bound; encoder; decoder }

let zlib =
  Lz.Zlib.
    { name = "zlib"; encode; encode_string; decode; bound; encoder; decoder }

let gzip =
  Lz.Gzip.
    { name = "gzip"; encode; encode_string; decode; bound; encoder; decoder }

let formats = [ deflate; zlib; gzip ]

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

let encode f ~level s =
  let dst = Lz_support.create (f.bound (String.length s)) in
  let k =
    require_some ~msg:"fits its bound"
      (f.encode ~level (Lz_support.of_string s) dst)
  in
  Lz_support.to_string ~length:k dst

(* [decode f z n] decodes [z] into exactly [n] bytes, checking that it
   consumed all of [z]. *)
let decode f z n =
  let dst = Lz_support.create n in
  match f.decode (Lz_support.of_string z) dst with
  | Error e -> Error e
  | Ok p ->
      equal Lz_support.progress ~msg:"consumed and produced"
        { Lz.consumed = String.length z; produced = n }
        p;
      Ok (Lz_support.to_string ~length:p.produced dst)

(* A decoder over slices of 1 byte to 70 KB. *)
let decode_stepped f z =
  Result.map fst
    (Lz_support.stepped
       ~cut:(fun k -> 1 + (k * 7919 mod 70_000))
       (f.decoder ()) z)

(* [steps f ~level ~cuts s] is the stream an encoder writes over [s]: step
   k is given at most the k-th of [cuts] bytes from the first byte not yet
   taken, all that is left after [cuts] runs out, with [~last:true] once
   the slice reaches the end. *)
let steps ?e f ~level ~cuts s =
  let e = match e with Some e -> e | None -> f.encoder ~level () in
  let out = Buffer.create (String.length s + 64) in
  let add run = Buffer.add_string out (Lz_support.to_string run) in
  let n = String.length s in
  let all = Lz_support.of_string s in
  let rec go at cuts k =
    if k > n + 100 then fail "the steps make no progress";
    let len, rest =
      match cuts with [] -> (n - at, []) | c :: r -> (min c (n - at), r)
    in
    let last = at + len = n in
    match Lz.Encoder.step e ~last (Bigarray.Array1.sub all at len) add with
    | Lz.Done taken -> equal int ~msg:"Done takes the whole slice" len taken
    | More taken ->
        if last && taken = len then fail "More after taking the last byte";
        go (at + taken) rest (k + 1)
  in
  go 0 cuts 0;
  Buffer.contents out

(* Drawn inputs *)

let format =
  Gen.of_list ~pp:(fun ppf f -> Format.pp_print_string ppf f.name) formats
let level = Gen.int_range 1 9

let cover_input r s f level =
  cover "empty" (s = "");
  cover "more than a block of 64 KiB" (String.length s > 64 * kib);
  cover "a match" (match r with Lz_support.Match _ -> true | _ -> false);
  cover "noise" (match r with Lz_support.Noise _ -> true | _ -> false);
  cover "a run" (match r with Lz_support.Run _ -> true | _ -> false);
  cover "deflate" (f.name = "deflate");
  cover "zlib" (f.name = "zlib");
  cover "gzip" (f.name = "gzip");
  cover "the hash table, level 1" (level = 1);
  cover "greedy, levels 2 to 4" (level >= 2 && level <= 4);
  cover "lazy, levels 5 to 7" (level >= 5 && level <= 7);
  cover "lazy2, levels 8 and 9" (level >= 8)

let placed =
  Gen.pair (Gen.triple Lz_support.recipe format level) (Gen.int_range 0 7)

(* The large inputs at levels 1, 6 and 9, one format each. *)
let large =
  List.concat
    (List.mapi
       (fun i r ->
         List.map
           (fun l -> ((r, List.nth formats ((i + l) mod 3), l), 1))
           [ 1; 6; 9 ])
       Lz_support.large)

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

(* Law 3 *)

let round_trip =
  prop ~examples:large "decode (encode x) = x, whole and stepped" placed
    (fun ((r, f, level), offset) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      let n = String.length s in
      let src = Lz_support.placed ~offset s in
      let dst = Lz_support.room ~offset (f.bound n) in
      let k = require_some ~msg:"fits its bound" (f.encode ~level src dst) in
      let z = Lz_support.to_string ~length:k dst in
      equal Lz_support.decoded (Ok s) (decode f z n);
      equal Lz_support.decoded (Ok s) (decode_stepped f z))

let strings =
  prop "decode (encode_string x) = x"
    (Gen.triple Lz_support.recipe format level) (fun (r, f, level) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      equal Lz_support.decoded (Ok s)
        (decode f (f.encode_string ~level s) (String.length s)))

(* Law 4 *)

let bound =
  prop ~examples:large "a destination of bound n holds the stream" placed
    (fun ((r, f, level), offset) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      let g = guarded (f.bound (String.length s)) in
      ignore
        (require_some (f.encode ~level (Lz_support.placed ~offset s) (snd g)));
      untouched g)

let short =
  prop "a destination short of the stream is refused"
    (Gen.pair placed Lz_support.short)
    (fun (((r, f, level), offset), short) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      let k = String.length (encode f ~level s) in
      let c = Lz_support.capacity k short in
      cover "one byte short" (c = k - 1);
      cover "an empty destination" (c = 0);
      let src = Lz_support.placed ~offset s in
      let g = guarded c in
      is_none ~msg:(Printf.sprintf "%d bytes of %d" c k)
        (f.encode ~level src (snd g));
      untouched g;
      equal (option int) ~msg:"exactly its length" (Some k)
        (f.encode ~level src (Lz_support.create k)))

(* Bound on incompressible input: stored blocks, the bound's worst case,
   written one byte at a time to the destination's end. *)
let noise_bound =
  cases "noise fits its bound at every level"
    ~id:(fun (n, l) -> Printf.sprintf "%d bytes, level %d" n l)
    (List.concat_map
       (fun n -> List.map (fun l -> (n, l)) levels)
       [ 0; 1; 4999; 5000; 5001; 65535; 65536; 70_000; 300_001 ])
    (fun (n, level) ->
      let s = Lz_support.build (Noise (n, n)) in
      List.iter
        (fun f ->
          let g = guarded (f.bound n) in
          let k =
            require_some ~msg:f.name
              (f.encode ~level (Lz_support.of_string s) (snd g))
          in
          untouched g;
          equal Lz_support.decoded ~msg:f.name (Ok s)
            (decode f (Lz_support.to_string ~length:k (snd g)) n))
        formats)

(* Law 6 *)

let cut_free =
  prop
    ~examples:(List.map (fun ((r, f, l), _) -> ((r, f, l, [ 777 ]), [])) large)
    "any cut into steps writes encode's bytes"
    (Gen.pair (Gen.quad Lz_support.recipe format level cuts) cuts)
    (fun ((r, f, level, a), b) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      cover "a step of no bytes" (List.mem 0 a);
      cover "a step past the quota" (List.exists (fun c -> c > 64 * kib) a);
      let za = steps f ~level ~cuts:a s in
      equal string ~msg:"two cuts" za (steps f ~level ~cuts:b s);
      equal string ~msg:"encode's" (encode f ~level s) za)

let stepped_round_trip =
  prop "a stepped encoder's stream decodes"
    (Gen.quad Lz_support.recipe format level cuts) (fun (r, f, level, cuts) ->
      let s = Lz_support.build r in
      cover_input r s f level;
      equal Lz_support.decoded (Ok s)
        (decode f (steps f ~level ~cuts s) (String.length s)))

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
      List.iter
        (fun f ->
          match quota_steps (f.encoder ()) s with
          | [], _ -> failf "%s: Done before the last byte was taken" f.name
          | (q :: _ as ks), last ->
              less int ~msg:(f.name ^ ": short of the slice") ~than:mib q;
              List.iteri
                (fun i k -> equal int ~msg:(Printf.sprintf "step %d" i) q k)
                ks;
              equal int ~msg:"Done takes the rest" (mib - (q * List.length ks))
                last;
              at_most int ~msg:"the rest is at most the quota" ~than:q last)
        formats)

let streams =
  prop "after Done, the next step begins a new stream"
    (Gen.quad Lz_support.recipe Lz_support.recipe format level)
    (fun (ra, rb, f, level) ->
      let a = Lz_support.build ra and b = Lz_support.build rb in
      let e = f.encoder ~level () in
      ignore (steps ~e f ~level ~cuts:[] a);
      equal string ~msg:"as a fresh encoder writes it"
        (steps f ~level ~cuts:[] b)
        (steps ~e f ~level ~cuts:[] b))

(* Law 6: the bytes do not depend on the instruction set. *)

let paths =
  prop "every path writes the bytes of Lz.Deflate.encode"
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      cover "the hash table, level 1" (level = 1);
      cover "hash chains, levels 2 to 9" (level >= 2);
      cover "past twice the window, where positions rebase"
        (String.length s > 2 * 32 * kib);
      let expected = encode deflate ~level s in
      List.iter
        (fun (p : Lz_support.deflate_enc_path) ->
          let dst = Lz_support.create (deflate.bound (String.length s)) in
          match p.encode ~level (Lz_support.of_string s) dst with
          | Some k ->
              equal string ~msg:p.label expected
                (Lz_support.to_string ~length:k dst)
          | None -> failf "%s: the stream exceeds its bound" p.label)
        Lz_support.deflate_enc_paths)

(* What the writers write *)

let byte z i = Char.code z.[i]
let le32 z i = Int32.to_int (String.get_int32_le z i) land 0xffffffff
let be32 z i = Int32.to_int (String.get_int32_be z i) land 0xffffffff

(* Both writers' streams, encode's and a stepped one. *)
let both f ~level s = [ encode f ~level s; steps f ~level ~cuts:[ 1000 ] s ]

let gzip_member =
  prop "gzip: one member, no name, comment or extra field, MTIME 0"
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      let n = String.length s in
      List.iter
        (fun z ->
          equal int ~msg:"ID1" 0x1f (byte z 0);
          equal int ~msg:"ID2" 0x8b (byte z 1);
          equal int ~msg:"CM" 8 (byte z 2);
          equal int ~msg:"FLG" 0 (byte z 3);
          equal int ~msg:"MTIME" 0 (le32 z 4);
          let t = String.length z - 8 in
          equal int ~msg:"CRC32" (Lz.Crc32.string s) (le32 z t);
          equal int ~msg:"ISIZE" (n land 0xffffffff) (le32 z (t + 4)))
        (both gzip ~level s))

let zlib_stream =
  prop "zlib: CM 8, no dictionary, FCHECK, and the Adler-32"
    (Gen.pair Lz_support.recipe level) (fun (r, level) ->
      let s = Lz_support.build r in
      List.iter
        (fun z ->
          let cmf = byte z 0 and flg = byte z 1 in
          equal int ~msg:"CM" 8 (cmf land 15);
          at_most int ~msg:"a window of at most 32 KiB" ~than:7 (cmf lsr 4);
          equal int ~msg:"FDICT" 0 (flg land 0x20);
          equal int ~msg:"FCHECK" 0 (((cmf lsl 8) lor flg) mod 31);
          let adler = Int32.to_int (Lz.Adler32.string s) land 0xffffffff in
          equal int ~msg:"ADLER32" adler (be32 z (String.length z - 4)))
        (both zlib ~level s))

let empty =
  cases "the empty input"
    ~id:(fun f -> f.name)
    formats
    (fun f ->
      List.iter
        (fun level ->
          let z = f.encode_string ~level "" in
          equal Lz_support.decoded (Ok "") (decode f z 0);
          equal string ~msg:"a fresh encoder stepped with ~last:true" z
            (steps f ~level ~cuts:[] ""))
        levels)

let () =
  exit
    (run "Deflate_enc"
       [
         cases "a level outside 1 to 9 raises" ~id:string_of_int [ 0; 10 ]
           (fun level ->
             List.iter
               (fun f ->
                 raises_match (Exn.invalid_arg ~sub:"level") (fun () ->
                     f.encode_string ~level "abc"))
               formats);
         group "round trip" [ round_trip; strings; stepped_round_trip ];
         group "bound" [ bound; short; noise_bound ];
         group "steps" [ cut_free; quota; streams ];
         group "paths" [ paths ];
         group "wrappers" [ gzip_member; zlib_stream; empty ];
       ])
