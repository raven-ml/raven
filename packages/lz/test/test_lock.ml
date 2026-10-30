(* The domain lock. A call releases it at the first report of its work
   that reaches its kernel's threshold, and holds it otherwise (lib/lz.h,
   lib/stubs.c): a decoder reports the bytes it has taken and written
   before each block it begins and after each 32 KiB of output, an
   encoder the input it is about to compress, a checksum its input.
   [Lz.releases] counts the calling thread's releases, so that each law
   is exact: at a threshold's neighbours, on calls that do little work
   whatever their input, and on calls that do much from a short input. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib

(* Each kernel's threshold, in bytes of work (stubs.c, lz.ml). *)
let deflate_decode = 8 * kib
let deflate_encode = 4 * kib
let zstd_decode = 8 * kib
let zstd_encode = 8 * kib
let lz4_decode = 16 * kib
let lz4_encode = 8 * kib
let snappy_decode = 16 * kib
let snappy_encode = 8 * kib
let checksum = mib

(* [releases f] is [f ()] and the times it released the lock. *)
let releases f =
  let before = Lz.releases () in
  let r = f () in
  (r, Lz.releases () - before)

let count f = snd (releases f)

(* [neighbours name t work] states that the call [work w] prepares, whose
   work is [w], holds the lock at [t - 1] and releases it once at [t]. *)
let neighbours name t work =
  cases
    ~name:(fun (w, _) -> string_of_int w)
    name
    [ (t - 1, 0); (t, 1) ]
    (fun (w, n) ->
      let call = work w in
      equal int ~msg:"releases" n (count call))

(* Bytes *)

let byte v = String.make 1 (Char.chr v)
let le n v = String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xff))
let text n = Lz_support.corpus Text n

let ok_produced n = function
  | Ok { Lz.produced; _ } -> equal int ~msg:"produced" n produced
  | Error e -> failf "error at %d" e.Lz.at

(* Deflate (RFC 1951, 3.2.4): a stored block, its header byte then LEN and
   NLEN, 5 bytes before its data. *)
let stored ~final s =
  let n = String.length s in
  byte (if final then 1 else 0) ^ le 2 n ^ le 2 (n lxor 0xffff) ^ s

(* A raw deflate stream whose blocks before the last take and write [w]
   bytes, with what it decodes to: one stored block of [(w - 5) / 2] bytes
   when [w] is odd, two when it is even, then an empty final block. *)
let deflate_work w =
  let a = String.make ((w - 5) / 2) 'a' in
  let n = (w - 10) / 2 in
  let b = String.make (n / 2) 'b' and c = String.make (n - (n / 2)) 'c' in
  let data = if w mod 2 = 1 then a else b ^ c in
  let body =
    if w mod 2 = 1 then stored ~final:false a
    else stored ~final:false b ^ stored ~final:false c
  in
  (body ^ stored ~final:true "", data)

(* Zstandard (RFC 8878, 3.1.1): a header with a 128 KiB window and no
   content size or checksum, 6 bytes, then blocks. *)
let windowed = "\x28\xB5\x2F\xFD\x00" ^ byte (7 lsl 3)

let rle ~last n =
  le 3 ((n lsl 3) lor (1 lsl 1) lor if last then 1 else 0) ^ "z"

(* A frame whose bytes before its last block's header, 10 of header and
   RLE block, and output, [w - 10] bytes, make [w]. *)
let zstd_work w = (windowed ^ rle ~last:false (w - 10) ^ rle ~last:true 1, w - 9)

(* LZ4 (lz4_Block_format.md): a sequence of one literal and a match of
   [len] (4 to 18) at offset 1, then [n] last literals. Its input and
   output make [6 + len + 2n] and the length bytes of [n]. *)
let lz4_block ~len n =
  let ext k =
    if k < 15 then ""
    else String.make ((k - 15) / 255) '\xff' ^ byte ((k - 15) mod 255)
  in
  let token lit ml = byte ((min lit 15 lsl 4) lor ml) in
  let s =
    token 1 (len - 4) ^ "a" ^ le 2 1
    ^ token n 0 ^ ext n ^ String.make n 'b'
  in
  (s, 1 + len + n)

(* A block whose input and output make [w], all of them copied by the
   exact loop's last literals, under LZ_WORK_SPAN. *)
let lz4_work w =
  let rec find n =
    let s, out = lz4_block ~len:4 n in
    let len = w - String.length s - out + 4 in
    if len >= 4 && len <= 18 then lz4_block ~len n else find (n - 1)
  in
  find (w / 2)

(* Snappy (format_description.txt): the preamble, a literal 'a', a copy
   of [len] (4 to 11) at offset 1, then a literal of [n] bytes, above 60,
   with its length less one in the [k] bytes after its tag. *)
let snappy_block ~len n =
  let out = 1 + len + n in
  let rec varint v =
    if v < 0x80 then byte v else byte (v land 0x7f lor 0x80) ^ varint (v lsr 7)
  in
  let k = if n - 1 < 0x100 then 1 else if n - 1 < 0x10000 then 2 else 3 in
  let s =
    varint out ^ "\x00a"
    ^ byte (((len - 4) lsl 2) lor 1)
    ^ "\x01"
    ^ byte ((59 + k) lsl 2)
    ^ le k (n - 1)
    ^ String.make n 'b'
  in
  (s, out)

let snappy_work w =
  let rec find n =
    let s, out = snappy_block ~len:4 n in
    let len = w - String.length s - out + 4 in
    if len >= 4 && len <= 11 then snappy_block ~len n else find (n - 1)
  in
  find (w / 2)

(* An LZ4 frame (lz4_Frame_format.md) of independent blocks of at most
   64 KiB, with no checksum or content size: 7 bytes of header. *)
let lz4_frame blocks =
  let desc = "\x60\x40" in
  let hc = Int32.to_int (Int32.shift_right_logical (Lz.Xxh32.string desc) 8) in
  "\x04\x22\x4D\x18" ^ desc ^ byte (hc land 0xff) ^ String.concat "" blocks
  ^ le 4 0

let lz4_stored s = le 4 (String.length s lor 0x8000_0000) ^ s

(* A compressed block of five literals: 4 + 6 bytes of input, 5 of
   output. *)
let lz4_five = le 4 6 ^ "\x50xxxxx"

(* A frame whose blocks before its last, with the header and the last
   block's size, take and write [w] bytes: a stored block of [n] bytes
   counts [4 + 2n], the five literals 15, the header 7 and the last
   block's size 4. *)
let lz4_frame_work w =
  let blocks, out =
    if w mod 2 = 1 then
      let n = (w - 15) / 2 in
      ([ lz4_stored (String.make n 'a') ], n)
    else
      let n = (w - 30) / 2 in
      ([ lz4_five; lz4_stored (String.make n 'a') ], 5 + n)
  in
  (lz4_frame (blocks @ [ lz4_stored "z" ]), out + 1)

(* Whole-buffer calls *)

let decode_case name t work decode =
  neighbours name t (fun w ->
      let s, n = work w in
      let src = Lz_support.of_string s and dst = Lz_support.create n in
      fun () -> ok_produced n (decode src dst))

let encode_case name t bound encode =
  neighbours name t (fun w ->
      let src = Lz_support.of_string (text w) in
      let dst = Lz_support.create (bound w) in
      fun () -> is_some (encode src dst))

let whole =
  group "whole-buffer calls"
    [
      decode_case "deflate decode" deflate_decode
        (fun w -> let s, d = deflate_work w in (s, String.length d))
        Lz.Deflate.decode;
      decode_case "Zstandard decode" zstd_decode zstd_work Lz.Zstd.decode;
      decode_case "LZ4 frame decode" lz4_decode lz4_frame_work
        Lz.Lz4.Frame.decode;
      encode_case "deflate encode" deflate_encode Lz.Deflate.bound
        (Lz.Deflate.encode ~level:6);
      encode_case "gzip encode" deflate_encode Lz.Gzip.bound
        (Lz.Gzip.encode ~level:1);
      encode_case "Zstandard encode" zstd_encode Lz.Zstd.bound
        (Lz.Zstd.encode ~level:3);
      encode_case "LZ4 block encode" lz4_encode Lz.Lz4.Block.bound
        Lz.Lz4.Block.encode;
      encode_case "LZ4 frame encode" lz4_encode Lz.Lz4.Frame.bound
        Lz.Lz4.Frame.encode;
      encode_case "Snappy encode" snappy_encode Lz.Snappy.bound
        Lz.Snappy.encode;
      neighbours "CRC-32 of a buffer" checksum (fun w ->
          let b = Lz_support.create w in
          fun () -> ignore (Lz.Crc32.bigbytes b));
    ]

(* A block of LZ4 or Snappy has no block inside to report before: its
   decode reports after each LZ_WORK_SPAN (32 KiB) of output and before a
   copy of as much. A block whose work passes the threshold only in its
   last unit holds the lock to its end; one longer releases once. *)

let one_block name t work decode =
  let run (s, n) =
    let src = Lz_support.of_string s and dst = Lz_support.create n in
    count (fun () -> ok_produced n (decode src dst))
  in
  cases
    ~name:(fun (label, _, _) -> label)
    name
    [
      ("work under the threshold", work (t - 1), 0);
      ("work at the threshold, in one unit", work t, 0);
      ("a literal run of 64 KiB", work (128 * kib), 1);
    ]
    (fun (_, input, n) -> equal int ~msg:"releases" n (run input))

let blocks =
  group "single blocks"
    [
      one_block "LZ4 block decode" lz4_decode lz4_work Lz.Lz4.Block.decode;
      one_block "Snappy decode" snappy_decode snappy_work Lz.Snappy.decode;
      test "an LZ4 block decoding 1 MiB of text releases" (fun () ->
          let s = text mib in
          let src = Lz_support.of_string (Lz.Lz4.Block.encode_string s) in
          let dst = Lz_support.create mib in
          equal int ~msg:"releases" 1
            (count (fun () -> ok_produced mib (Lz.Lz4.Block.decode src dst))));
      test "a Snappy block decoding 1 MiB of text releases" (fun () ->
          let s = text mib in
          let src = Lz_support.of_string (Lz.Snappy.encode_string s) in
          let dst = Lz_support.create mib in
          equal int ~msg:"releases" 1
            (count (fun () -> ok_produced mib (Lz.Snappy.decode src dst))));
    ]

(* Work from a short input *)

(* 512 RLE blocks of 128 KiB, 2 KiB of input, decode to 64 MiB. *)
let rle_frame =
  let blocks = 512 and size = 128 * kib in
  let body =
    String.concat "" (List.init (blocks - 1) (fun _ -> rle ~last:false size))
  in
  (windowed ^ body ^ rle ~last:true size, blocks * size)

let short_input =
  group "a short input decoding long"
    [
      test "a Zstandard frame of 2 KiB decoding to 64 MiB releases" (fun () ->
          let s, n = rle_frame in
          less int ~msg:"input bytes" ~than:zstd_decode (String.length s);
          let src = Lz_support.of_string s and dst = Lz_support.create n in
          let r, k = releases (fun () -> Lz.Zstd.decode src dst) in
          ok_produced n r;
          equal int ~msg:"releases" 1 k);
      test "deflate of a few KiB decoding to 4 MiB releases" (fun () ->
          let raw = String.make (4 * mib) 'a' in
          let s = Lz.Deflate.encode_string ~level:9 raw in
          less int ~msg:"input bytes" ~than:deflate_decode (String.length s);
          let src = Lz_support.of_string s in
          let dst = Lz_support.create (String.length raw) in
          let r, k = releases (fun () -> Lz.Deflate.decode src dst) in
          ok_produced (String.length raw) r;
          equal int ~msg:"releases" 1 k);
    ]

(* Steps *)

(* [decode_step d s] is one step of [d] over [s] with [~last:true], which
   must take all of it and end the stream. *)
let decode_step d s =
  let src = Lz_support.of_string s in
  fun () ->
    match Lz.Decoder.step d ~last:true src ignore with
    | Ok (Done k) -> equal int ~msg:"taken" (String.length s) k
    | Ok (More _) -> fail "the step did not end the stream"
    | Error e -> failf "error at %d" e.Lz.at

(* [kept e k] steps [e] over [k] bytes of text without [~last], and
   returns the releases of that step, which must take them all. *)
let kept e k =
  let src = Lz_support.of_string (text k) in
  count (fun () ->
      equal Lz_support.step ~msg:"taken" (More k)
        (Lz.Encoder.step e ~last:false src ignore))

let flush e () =
  equal Lz_support.step ~msg:"the flush" (Done 0)
    (Lz.Encoder.step e ~last:true (Lz_support.create 0) ignore)

(* [tiny_steps e s] steps [e] over [s] in steps of 100 bytes without
   [~last], and is the releases they made. *)
let tiny_steps e s =
  let src = Lz_support.of_string s in
  let n = String.length s in
  let rec go at acc =
    if at = n then acc
    else
      let len = min 100 (n - at) in
      let sub = Bigarray.Array1.sub src at len in
      let r, k = releases (fun () -> Lz.Encoder.step e ~last:false sub ignore) in
      equal Lz_support.step ~msg:"taken" (More len) r;
      go (at + len) (acc + k)
  in
  go 0 0

let steps =
  group "steps"
    [
      neighbours "a deflate decoder step" deflate_decode (fun w ->
          decode_step (Lz.Deflate.decoder ()) (fst (deflate_work w)));
      neighbours "a Zstandard decoder step" zstd_decode (fun w ->
          decode_step (Lz.Zstd.decoder ()) (fst (zstd_work w)));
      neighbours "a deflate flush writing out kept input" deflate_encode
        (fun w ->
          (* Under its 8 KiB lookahead, the encoder parses nothing until
             the flush. *)
          let e = Lz.Deflate.encoder ~level:6 () in
          equal int ~msg:"releases taking the input" 0 (kept e w);
          flush e);
      neighbours "a Zstandard flush writing out kept input" zstd_encode
        (fun w ->
          let e = Lz.Zstd.encoder ~level:3 () in
          equal int ~msg:"releases taking the input" 0 (kept e w);
          flush e);
      test "a gzip encoder in 100-byte steps holds the lock" (fun () ->
          let e = Lz.Gzip.encoder () in
          equal int ~msg:"releases" 0 (tiny_steps e (text (256 * kib))));
      test "a Zstandard encoder in 100-byte steps holds the lock until a \
            block ends" (fun () ->
          let e = Lz.Zstd.encoder () in
          equal int ~msg:"releases" 0 (tiny_steps e (text (128 * kib)));
          (* 128 KiB and 100 bytes kept: the step compresses a block. *)
          equal int ~msg:"the step that ends a block" 1
            (tiny_steps e (text 100)));
      test "a gzip decoder in 100-byte steps holds the lock" (fun () ->
          let s = text mib in
          let z = Lz_support.of_string (Lz.Gzip.encode_string s) in
          let d = Lz.Gzip.decoder () in
          let n = Bigarray.Array1.dim z in
          (* Every step takes 100 bytes and writes a few hundred: its work
             stays far under the threshold. *)
          let most = ref 0 in
          let rec go at acc =
            let len = min 100 (n - at) in
            let last = at + len = n in
            let run = ref 0 in
            let r, k =
              releases (fun () ->
                  Lz.Decoder.step d ~last (Bigarray.Array1.sub z at len)
                    (fun b -> run := Bigarray.Array1.dim b))
            in
            most := max !most (len + !run);
            match r with
            | Ok (Done _) -> acc + k
            | Ok (More t) -> go (at + t) (acc + k)
            | Error e -> failf "error at %d" e.Lz.at
          in
          let total = go 0 0 in
          less int ~msg:"the largest step's input and output"
            ~than:(deflate_decode / 2) !most;
          equal int ~msg:"releases" 0 total);
    ]

let () = exit (run "Lock" [ whole; blocks; short_input; steps ])
