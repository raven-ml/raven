(* Zstandard decoding (RFC 8878): the vendor's goldens, the specification's
   refusals on frames built byte by byte, the stream against the
   whole-buffer decode, and the decoder's memory and lock. Round trips and
   the bound belong to the encoder.

   An error's offset is the first byte of what is wrong: of a field the
   format refuses or the library declines, of a section or table
   description that runs past its block, of a bitstream not read exactly
   to its start; the byte holding a sequence's first bit when the
   sequence breaks a rule; a frame's first byte when no frame begins
   there; the input's length when it ends inside a frame. A block that
   asks for more output than the destination holds is Too_long at its
   header. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib

(* Bytes *)

let byte v = String.make 1 (Char.chr v)
let le n v = String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xFF))

(* Section 3.1.1: the magic number. *)
let magic = "\x28\xB5\x2F\xFD"

(* Section 3.1.1.1: a header with no content size whose window is
   2^(10 + exponent) bytes: the descriptor, then the Window_Descriptor. It
   ends at offset 6. *)
let windowed exponent = magic ^ "\x00" ^ byte (exponent lsl 3)

(* A single-segment header whose window is its content size [n] < 256,
   Frame_Content_Size in 1 byte. It ends at offset 6. *)
let single n = magic ^ "\x20" ^ byte n

(* Section 3.1.1.2: a block header, then the block. *)
let block_header ~last typ size =
  le 3 ((size lsl 3) lor (typ lsl 1) lor if last then 1 else 0)

let raw ?(last = true) s = block_header ~last 0 (String.length s) ^ s
let rle ?(last = true) c n = block_header ~last 1 n ^ String.make 1 c
let compressed ?(last = true) s = block_header ~last 2 (String.length s) ^ s

(* Section 3.1.1.3.1.1: raw literals, fewer than 32, in a 1-byte header. *)
let raw_literals s = byte (String.length s lsl 3) ^ s

(* Section 3.1.1.3.2.1: one sequence whose three codes are in RLE_Mode, so
   that its bitstream [bits] holds only their extra bits. *)
let one_sequence ~ll ~of_ ~ml bits =
  "\x01\x54" ^ byte ll ^ byte of_ ^ byte ml ^ bits

(* Section 4.1.1, as an encoder: the accuracy log, then each probability
   in the fewest bits that hold any value up to the points left plus one,
   low values one bit shorter (Table 20); a zero is followed by a 2-bit
   count of further zeros, here 0. *)
let ncount ~log probs =
  let out = Buffer.create 16 and acc = ref 0 and n = ref 0 in
  let add v k =
    acc := !acc lor (v lsl !n);
    n := !n + k;
    while !n >= 8 do
      Buffer.add_char out (Char.chr (!acc land 0xFF));
      acc := !acc lsr 8;
      n := !n - 8
    done
  in
  let rec high x k = if x <= 1 then k else high (x lsr 1) (k + 1) in
  add (log - 5) 4;
  let left = ref ((1 lsl log) + 1) in
  List.iter
    (fun p ->
      let value = p + 1 in
      let bits = high !left 0 + 1 in
      let threshold = 1 lsl (bits - 1) in
      let low = (2 * threshold) - 1 - !left in
      if value < low then add value (bits - 1)
      else if value < threshold then add value bits
      else add (value + low) bits;
      (left := !left - if p < 0 then 1 else p);
      if p = 0 then add 0 2)
    probs;
  if !n > 0 then Buffer.add_char out (Char.chr (!acc land 0xFF));
  Buffer.contents out

(* Results *)

(* A result as one line: what it read and wrote, its output when short, or
   the error's kind and offset. *)
let seen ?dst = function
  | Ok { Lz.consumed; produced } -> (
      match dst with
      | Some d when produced <= 64 ->
          Printf.sprintf "consumed %d, %S" consumed
            (Lz_support.to_string ~length:produced d)
      | _ -> Printf.sprintf "consumed %d, produced %d" consumed produced)
  | Error { Lz.at; why } ->
      let kind =
        match why with
        | Lz.Truncated -> "truncated"
        | Malformed _ -> "malformed"
        | Too_long -> "too long"
        | Unsupported _ -> "unsupported"
      in
      Printf.sprintf "%s at %d" kind at

let decode ?(cap = 4096) s =
  let dst = Lz_support.create cap in
  seen ~dst (Lz.Zstd.decode (Lz_support.of_string s) dst)

let golden name =
  List.find (fun g -> g.Lz_support.file = name) (Lz_support.goldens_of "zstd")

let data name = Lazy.force (golden name).Lz_support.data
let raw_of name = Lazy.force (golden name).Lz_support.raw

(* A destination with guard bytes after it. *)
let guard = 64

let guarded n =
  let whole = Lz_support.create (n + guard) in
  (whole, Bigarray.Array1.sub whole 0 n)

let untouched (whole, view) =
  let n = Bigarray.Array1.dim view in
  equal string ~msg:"the bytes past the destination" (String.make guard '\xa5')
    (String.init guard (fun i -> Char.chr (Bigarray.Array1.get whole (n + i))))

(* Frames *)

let frames =
  group "frames"
    [
      test "skippable frames produce nothing" (fun () ->
          let s = data "skippable.zst" in
          equal string
            (Printf.sprintf "consumed %d, %S" (String.length s) "")
            (decode s));
      test "frames and skippable frames decode in turn" (fun () ->
          let a = data "text_100.zst3" and b = data "columns_100.zst1" in
          let s = a ^ "\x5A\x2A\x4D\x18\x03\x00\x00\x00abc" ^ b in
          equal string
            (Printf.sprintf "consumed %d, produced %d" (String.length s) 200)
            (decode s);
          let dst = Lz_support.create 200 in
          is_ok (Lz.Zstd.decode (Lz_support.of_string s) dst);
          equal string
            (raw_of "text_100.zst3" ^ raw_of "columns_100.zst1")
            (Lz_support.to_string dst));
      test "bytes that begin no frame are malformed where they begin"
        (fun () ->
          let a = data "text_100.zst3" in
          equal string
            (Printf.sprintf "malformed at %d" (String.length a))
            (decode (a ^ "\x28\xB5\x00")));
      test "a cut inside the next magic is truncated" (fun () ->
          let a = data "text_100.zst3" in
          equal string
            (Printf.sprintf "truncated at %d" (String.length a + 2))
            (decode (a ^ "\x28\xB5")));
      test "an empty input is truncated at 0" (fun () ->
          equal string "truncated at 0" (decode ""));
      test "a magic no frame begins with is malformed at its first byte"
        (fun () ->
          equal string "malformed at 0" (decode "\x00\xB5\x2F\xFD");
          equal string "malformed at 0" (decode "\x28\xB5\x2E\xFD");
          equal string "malformed at 0" (decode "\x5F\x2B"));
      test "frames end in Done only on the step with ~last:true" (fun () ->
          let d = Lz.Zstd.decoder () in
          let got = Buffer.create 65636 in
          let add run = Buffer.add_string got (Lz_support.to_string run) in
          let step ~last s =
            Lz.Decoder.step d ~last (Lz_support.of_string s) add
          in
          let rec feed s =
            let n = String.length s in
            match step ~last:false s with
            | Ok (More k) when k < n -> feed (String.sub s k (n - k))
            | r -> equal (result Lz_support.step Lz_support.error) (Ok (More n)) r
          in
          feed (data "frames.zst");
          equal (result Lz_support.step Lz_support.error) (Ok (Done 0))
            (step ~last:true "");
          equal string (raw_of "frames.zst") (Buffer.contents got));
    ]

(* Headers, blocks and checksums *)

let header =
  let fcs n = magic ^ "\x40\x00" ^ le 2 (n - 256) in
  let xs = String.make 300 'x' in
  group "header"
    [
      test "a reserved descriptor bit is malformed" (fun () ->
          equal string "malformed at 4" (decode (magic ^ "\x08\x00" ^ raw "abc")));
      test "the unused descriptor bit is ignored" (fun () ->
          equal string "consumed 12, \"abc\""
            (decode (magic ^ "\x10\x00" ^ raw "abc")));
      test "a dictionary identifier is unsupported at its first byte"
        (fun () ->
          equal string "unsupported at 6"
            (decode (magic ^ "\x01\x00\x07" ^ raw "abc"));
          equal string "unsupported at 6"
            (decode (magic ^ "\x03\x00\x00\x00\x00\x80" ^ raw "abc")));
      test "a dictionary identifier of 0 names no dictionary" (fun () ->
          equal string "consumed 13, \"abc\""
            (decode (magic ^ "\x01\x00\x00" ^ raw "abc")));
      test "the reserved block type is malformed" (fun () ->
          equal string "malformed at 6"
            (decode (windowed 0 ^ block_header ~last:true 3 1 ^ "a")));
      test "a block above 128 KiB is malformed at its header" (fun () ->
          let n = (128 * kib) + 1 in
          equal string "malformed at 6"
            (decode ~cap:(2 * n) (windowed 10 ^ raw (String.make n 'a')));
          equal string "malformed at 6"
            (decode ~cap:(2 * n) (windowed 10 ^ rle 'a' n));
          equal string
            (Printf.sprintf "consumed 10, produced %d" (n - 1))
            (decode ~cap:(2 * n) (windowed 10 ^ rle 'a' (n - 1))));
      test "a block above the window is malformed at its header" (fun () ->
          equal string "malformed at 6"
            (decode (single 100 ^ raw (String.make 101 'a')));
          equal string "malformed at 6"
            (decode (windowed 0 ^ raw (String.make 1025 'a'))));
      (* Frame_Content_Size, at 6, is what does not match, however the
         blocks show it: a raw block's header, the last block, or a
         compressed block's output. *)
      test "a content size larger than the content is malformed at it"
        (fun () ->
          equal string "malformed at 6" (decode (fcs 301 ^ raw xs));
          equal string "malformed at 6"
            (decode
               (fcs 301
               ^ raw ~last:false (String.sub xs 0 200)
               ^ raw (String.sub xs 0 100))));
      test "a content size smaller than the content is malformed at it"
        (fun () ->
          equal string "malformed at 6" (decode (fcs 299 ^ raw xs));
          equal string "malformed at 6"
            (decode
               (fcs 299
               ^ raw ~last:false (String.sub xs 0 200)
               ^ raw (String.sub xs 0 100)));
          equal string "malformed at 6"
            (decode (fcs 257 ^ compressed (raw_literals "abcd" ^ "\x00")));
          equal string "consumed 311, produced 300" (decode (fcs 300 ^ raw xs)));
      test "a wrong checksum is malformed at the checksum" (fun () ->
          let s = data "text_100.zst3" in
          let n = String.length s in
          let b = Bytes.of_string s in
          Bytes.set b (n - 1) (Char.chr (Char.code s.[n - 1] lxor 1));
          equal string
            (Printf.sprintf "malformed at %d" (n - 4))
            (decode (Bytes.to_string b)));
      cases
        ~name:(fun (field, _) -> field)
        "a refusal's message names the field at its offset"
        [
          ("Magic_Number", "\x28\xB5\x2E\xFD");
          ("Frame_Header_Descriptor", magic ^ "\x08\x00" ^ raw "abc");
          ("Dictionary_ID", magic ^ "\x01\x00\x07" ^ raw "abc");
          ("Block_Type", windowed 0 ^ block_header ~last:true 3 1 ^ "a");
          ("Block_Size", single 100 ^ raw (String.make 101 'a'));
          ("Frame_Content_Size", fcs 301 ^ raw xs);
        ]
        (fun (field, s) ->
          let dst = Lz_support.create 4096 in
          contains ~sub:field
            (match Lz.Zstd.decode (Lz_support.of_string s) dst with
            | Error { Lz.why = Malformed m | Unsupported m; _ } -> m
            | _ -> "no refusal"));
      test "a zip bomb ends at the destination" (fun () ->
          (* Blocks of 128 KiB from 4 bytes each: 10 MiB of input state
             320 GiB. 8 blocks fill 1 MiB; the ninth's header is at 38. *)
          let block = rle ~last:false 'z' (128 * kib) in
          let s =
            windowed 7 ^ String.concat "" (List.init (10 * mib / 4) (fun _ -> block))
          in
          let g = guarded mib in
          equal string "too long at 38"
            (seen (Lz.Zstd.decode (Lz_support.of_string s) (snd g)));
          untouched g);
    ]

(* Destinations *)

let destinations =
  let nonempty =
    List.filter
      (fun g -> String.length (Lazy.force g.Lz_support.raw) > 0)
      (Lz_support.goldens_of "zstd")
  in
  let kind = function
    | Ok _ -> "ok"
    | Error { Lz.why; _ } -> (
        match why with
        | Lz.Truncated -> "truncated"
        | Malformed _ -> "malformed"
        | Too_long -> "too long"
        | Unsupported _ -> "unsupported")
  in
  cases ~name:(fun g -> g.Lz_support.file)
    "a destination of the decoded size holds it, one byte less is too long"
    nonempty (fun g ->
      let raw = Lazy.force g.Lz_support.raw in
      let src = Lz_support.of_string (Lazy.force g.data) in
      let n = String.length raw in
      let exact = guarded n in
      is_ok (Lz.Zstd.decode src (snd exact));
      equal string raw (Lz_support.to_string (snd exact));
      untouched exact;
      let short = guarded (n - 1) in
      equal string "too long" (kind (Lz.Zstd.decode src (snd short)));
      untouched short)

(* Windows *)

let window =
  let step d =
    match Lz_support.stepped d (data "window1m.zst") with
    | Ok (s, _) -> Ok s
    | Error e -> Error (seen (Error e))
  in
  let expected = lazy (raw_of "window1m.zst") in
  group "window"
    [
      test "the default decoder holds a 1 MiB window" (fun () ->
          equal (result string string) (Ok (Lazy.force expected))
            (step (Lz.Zstd.decoder ())));
      test "a window above the decoder's is unsupported at its descriptor"
        (fun () ->
          equal (result string string) (Error "unsupported at 5")
            (step (Lz.Zstd.decoder ~window:(512 * kib) ()));
          equal (result string string) (Error "unsupported at 5")
            (step (Lz.Zstd.decoder ~window:(mib - 1) ())));
      test "a window equal to the decoder's is decoded" (fun () ->
          equal (result string string) (Ok (Lazy.force expected))
            (step (Lz.Zstd.decoder ~window:mib ())));
      test "a single segment's window is its content size" (fun () ->
          let s = magic ^ "\x60" ^ le 2 (2000 - 256) ^ raw (String.make 2000 'w') in
          let go w =
            match Lz_support.stepped (Lz.Zstd.decoder ~window:w ()) s with
            | Ok (out, _) -> string_of_int (String.length out)
            | Error e -> seen (Error e)
          in
          (* Frame_Content_Size, bytes 5 and 6, is the window. *)
          equal string "unsupported at 5" (go 1999);
          equal string "2000" (go 2000));
      test "the whole-buffer decode takes any window" (fun () ->
          equal string "consumed 9, \"\"" (decode (magic ^ "\x00\xF8" ^ raw "")));
    ]

(* Literals *)

let literals =
  (* A frame of one compressed block, whose content starts at 9: literals
     [lits], then no sequences. *)
  let frame lits = windowed 0 ^ compressed (lits ^ "\x00") in
  (* Section 3.1.1.3.1.1, Size_Format 00 (one stream) or 01 (four):
     10-bit sizes in a 3-byte header. *)
  let huffman ?(typ = 2) ?(four = false) regen tree streams =
    let csize = String.length tree + String.length streams in
    le 3 (typ lor (if four then 4 else 0) lor (regen lsl 4) lor (csize lsl 14))
    ^ tree ^ streams
  in
  (* Section 4.2.1.1: weights as 4-bit fields. *)
  let direct ws =
    let n = List.length ws in
    let a = Array.of_list (ws @ [ 0 ]) in
    byte (127 + n)
    ^ String.init ((n + 1) / 2) (fun i ->
          Char.chr ((a.(2 * i) lsl 4) lor a.((2 * i) + 1)))
  in
  group "literals"
    [
      cases
        ~name:(fun (n, _) -> n)
        "huffman weights that build no prefix code are malformed at the \
         description's first byte"
        [
          ("weights that complete to no power of 2", [ 3; 1 ]);
          ("codes longer than 11 bits", [ 11; 11 ]);
        ]
        (fun (_, ws) ->
          equal string "malformed at 12"
            (decode (frame (huffman 4 (direct ws) "\x01"))));
      test "a weight above 11 is malformed at its byte" (fun () ->
          (* Weights 1 and 1 in the description's byte 13, 12 in 14. *)
          equal string "malformed at 14"
            (decode (frame (huffman 4 (direct [ 1; 1; 12 ]) "\x01"))));
      test "weights complete to a code without a weight of 1" (fun () ->
          (* Weights 2, 2, 2 complete with a fourth 2: four 2-bit codes,
             00 to 11 by symbol. The stream reads them from its last byte,
             after the end mark: 00 01 10 11. *)
          equal string "consumed 18, \"\\000\\001\\002\\003\""
            (decode (frame (huffman 4 (direct [ 2; 2; 2 ]) "\x1B\x01"))));
      test "treeless literals with no earlier tree are malformed" (fun () ->
          equal string "malformed at 9"
            (decode (frame (huffman ~typ:3 4 "" "\x01"))));
      (* Weights 1 and 1 complete with a 2: codes 00, 01 and 1 for 0, 1 and
         2. Section 3.1.1.3.1.6: each stream regenerates (n + 3) / 4
         literals, the last up to 3 fewer, none when n is 3. A stream
         holds its codes, then its end mark. *)
      cases ~name:string_of_int "four streams regenerate as few as 3 literals"
        [ 3; 4 ] (fun n ->
          let streams = [ "\x03"; "\x04"; "\x05"; (if n = 4 then "\x03" else "\x01") ] in
          let jump = le 2 1 ^ le 2 1 ^ le 2 1 in
          let lits =
            huffman ~four:true n (direct [ 1; 1 ]) (jump ^ String.concat "" streams)
          in
          equal string
            (Printf.sprintf "consumed 25, %S" (String.sub "\x02\x00\x01\x02" 0 n))
            (decode (frame lits)));
      test "four streams of 5 literals are malformed at Regenerated_Size"
        (fun () ->
          (* The first three streams take 2 each, the last none left. *)
          equal string "malformed at 9"
            (decode
               (frame
                  (huffman ~four:true 5 (direct [ 1; 1 ]) (String.make 10 '\x01')))));
      test "a jump table past its streams is malformed at its first byte"
        (fun () ->
          (* The jump table, at 14, states 9 bytes of streams of 4. *)
          equal string "malformed at 14"
            (decode
               (frame
                  (huffman ~four:true 8 (direct [ 1; 1 ])
                     (le 2 3 ^ le 2 3 ^ le 2 3 ^ String.make 4 '\x01')))));
      test "an FSE description cut by its section is malformed at its first \
            byte" (fun () ->
          (* Weights FSE-coded in 1 byte at 13, accuracy log 5: its first
             probability needs bits past the byte. *)
          equal string "malformed at 13"
            (decode (frame (huffman 4 "\x01\x00" "\x01"))));
      test "raw literals past the block are malformed" (fun () ->
          equal string "malformed at 9" (decode (windowed 0 ^ compressed "\x28abc")));
    ]

(* Sequences *)

let sequences =
  (* One compressed block: literals "abcd", then [seqs] from offset 14;
     one_sequence's bitstream is at 19. *)
  let frame ?(head = windowed 0) seqs =
    head ^ compressed (raw_literals "abcd" ^ seqs)
  in
  group "sequences"
    [
      test "a sequence copies literals then a match" (fun () ->
          (* Offset code 2 with extra bits 00: Offset_Value 4, offset 1. *)
          equal string "consumed 20, \"abcdddd\""
            (decode (frame (one_sequence ~ll:4 ~of_:2 ~ml:0 "\x04"))));
      test "an offset before the frame's start is malformed" (fun () ->
          (* Offset code 5: Offset_Value 32, offset 29, after 4 bytes. *)
          equal string "malformed at 19"
            (decode (frame (one_sequence ~ll:4 ~of_:5 ~ml:0 "\x20"))));
      test "a repeat offset of 0 is malformed" (fun () ->
          (* Literals length 0 and Offset_Value 3: Repeated_Offset1 - 1. *)
          equal string "malformed at 19"
            (decode (frame (one_sequence ~ll:0 ~of_:1 ~ml:0 "\x03"))));
      test "a literals length past the literals is malformed" (fun () ->
          equal string "malformed at 19"
            (decode (frame (one_sequence ~ll:5 ~of_:2 ~ml:0 "\x04"))));
      test "a match past the block's maximum is malformed" (fun () ->
          (* A window, so a block, of 20 bytes; 4 + 34 bytes. *)
          equal string "malformed at 19"
            (decode
               (frame ~head:(single 20) (one_sequence ~ll:4 ~of_:2 ~ml:31 "\x04"))));
      test "an error in a sequence is at the byte holding its first bit"
        (fun () ->
          (* Two sequences of 3 literals, offset code 2 and match length
             code 43, 2 and 7 extra bits each, in a 3-byte bitstream at 19
             whose end mark is bit 18: the first sequence's bits are 17 to
             9, from byte 21, the second's 8 to 0, from byte 20. With
             literals "abcd" the second has 1 literal left; with offset
             extra bits 11 the first reaches 4 back after 3 bytes. *)
          let two bits = "\x02\x54\x03\x02\x2B" ^ bits in
          equal string "malformed at 20" (decode (frame (two "\x00\x00\x04")));
          equal string "malformed at 21" (decode (frame (two "\x00\x00\x07"))));
      test "a compressed block's output past the destination is too long at \
            its header" (fun () ->
          (* The first sequence writes 102 bytes. *)
          equal string "too long at 6"
            (decode ~cap:50 (frame ("\x02\x54\x03\x02\x2B\x00\x00\x04"))));
      test "a bitstream not read to its start is malformed" (fun () ->
          equal string "malformed at 19"
            (decode (frame (one_sequence ~ll:4 ~of_:2 ~ml:0 "\x08"))));
      test "reserved compression mode bits are malformed" (fun () ->
          equal string "malformed at 15"
            (decode (frame "\x01\x55\x04\x02\x00\x04")));
      test "a repeated table with none before is malformed at its mode"
        (fun () ->
          equal string "malformed at 15" (decode (frame "\x01\xC0\x04")));
      test "an FSE table decodes" (fun () ->
          (* Offsets coded by a table of accuracy log 5 giving code 2 all 32
             points: a 5-bit state, then code 2's 2 extra bits. *)
          equal string "consumed 22, \"abcdddd\""
            (decode
               (frame ("\x01\x64\x04" ^ ncount ~log:5 [ 0; 0; 32 ] ^ "\x00\x80"))));
      test "an FSE description cut by the block is malformed at its first byte"
        (fun () ->
          (* Literals lengths, at 16, accuracy log 6: the first probability
             needs bits past the block. *)
          equal string "malformed at 16" (decode (frame "\x01\x80\x01")));
      cases
        ~name:(fun (n, _) -> n)
        "an accuracy log above the limit is malformed"
        [
          ("literals lengths, 10 above 9", "\x01\x80\x05");
          ("offsets, 9 above 8", "\x01\x20\x04");
          ("match lengths, 10 above 9", "\x01\x08\x05");
        ]
        (fun (_, s) -> equal string "malformed at 16" (decode (frame s)));
      test "a distribution short of its total is malformed past its alphabet"
        (fun () ->
          (* 36 literals length codes of 1 point each, 36 of 64: a 37th
             probability begins in the description's last byte. *)
          let d = ncount ~log:6 (List.init 36 (fun _ -> 1)) in
          equal string
            (Printf.sprintf "malformed at %d" (16 + String.length d - 1))
            (decode (frame ("\x01\x80" ^ d))));
      test "a distribution short of its total at the last offset code is \
            malformed past its alphabet" (fun () ->
          (* Offset codes 0 to 30 at 1 point and 31 at 0: 31 of 32, and a
             33rd probability begins in the description's last byte. *)
          let d = ncount ~log:5 (List.init 31 (fun _ -> 1) @ [ 0 ]) in
          equal string
            (Printf.sprintf "malformed at %d" (16 + String.length d - 1))
            (decode (frame ("\x01\x20" ^ d))));
    ]

(* The stream against the whole-buffer decode *)

(* The output of complete frames, an input that ends inside a frame, or an
   error. *)
type outcome = Output of string | Cut | Failed of Lz.error

let outcome =
  Testable.make ~equal:( = )
    ~pp:(fun ppf -> function
      | Output s ->
          Format.fprintf ppf "Output <%d bytes, md5 %s>" (String.length s)
            (Digest.to_hex (Digest.string s))
      | Cut -> Format.pp_print_string ppf "Cut"
      | Failed e -> Format.fprintf ppf "Failed %a" (Testable.pp Lz_support.error) e)

(* A slice or a destination: one byte, a few, or up to 256 KiB. *)
let size rng =
  match Random.State.int rng 4 with
  | 0 -> 1
  | 1 -> 1 + Random.State.int rng 16
  | 2 -> 1 + Random.State.int rng 4096
  | _ -> 1 + Random.State.int rng (256 * kib)

(* [stepped d data seed] steps [d] over [data] in slices drawn from
   [seed]. *)
let stepped d data seed =
  let rng = Random.State.make [| seed |] in
  match Lz_support.stepped d data ~cut:(fun _ -> size rng) with
  | Ok (s, _) -> Output s
  | Error { Lz.why = Truncated; _ } -> Cut
  | Error e -> Failed e

(* The whole-buffer decode, with room for what a mutation adds, and the
   input the stream is given: what the decode read. *)
let whole data room =
  let dst = Lz_support.create room in
  match Lz.Zstd.decode (Lz_support.of_string data) dst with
  | Ok { Lz.consumed; produced } ->
      Some
        ( String.sub data 0 consumed,
          Output (Lz_support.to_string ~length:produced dst) )
  | Error { Lz.why = Truncated; _ } -> Some (data, Cut)
  | Error { Lz.why = Too_long; _ } -> None
  | Error e -> Some (data, Failed e)

(* A golden, unchanged, with a byte replaced, or cut. *)
type change = Same | Replace of int * char | Cut_at of int

let goldens = lazy (Array.of_list (Lz_support.goldens_of "zstd"))

let change =
  Gen.(
    frequency
      [
        (1, constant Same);
        (2, map (fun (i, c) -> Replace (i, c)) (pair nat char));
        (1, map (fun i -> Cut_at i) nat);
      ])

let pp_change ppf (k, change) =
  Format.fprintf ppf "%s, %s" (Lazy.force goldens).(k).Lz_support.file
    (match change with
    | Same -> "unchanged"
    | Replace (i, c) -> Printf.sprintf "byte %d replaced by %C" i c
    | Cut_at i -> Printf.sprintf "cut at %d" i)

let changed k change =
  let data = Lazy.force (Lazy.force goldens).(k).Lz_support.data in
  let n = String.length data in
  match change with
  | Replace (i, c) ->
      let b = Bytes.of_string data in
      Bytes.set b (i mod n) c;
      Bytes.to_string b
  | Cut_at i -> String.sub data 0 (i mod n)
  | Same -> data

let golden_index = Gen.int_range 0 (List.length (Lz_support.goldens_of "zstd") - 1)

let steps =
  let pp ppf (k, change, seed) =
    Format.fprintf ppf "%a, seed %d" pp_change (k, change) seed
  in
  prop "whole equals steps"
    Gen.(with_pp pp (triple golden_index change nat))
    (fun (k, change, seed) ->
      let g = (Lazy.force goldens).(k) in
      let data = changed k change in
      match whole data (String.length (Lazy.force g.raw) + mib) with
      | None -> reject ()
      | Some (input, expected) ->
          let got = stepped (Lz.Zstd.decoder ()) input seed in
          (* A window above the stream's limit is the whole-buffer
             decode's to take. *)
          (match got with
          | Failed { Lz.why = Unsupported _; _ } when got <> expected -> reject ()
          | _ -> ());
          cover "frames that decode" (match expected with Output _ -> true | _ -> false);
          cover "an input cut short" (expected = Cut);
          cover "a malformed input"
            (match expected with
            | Failed { Lz.why = Malformed _; _ } -> true
            | _ -> false);
          equal outcome expected got)

(* A frame with a 1 KiB window whose 20,000 blocks are each a few raw
   literals and one match up to the window back, built with the
   specification's codes beside the output it states: about 500 KB, so
   that a stream's buffer wraps several times over it however the input
   is cut, and matches cross from one segment into the next. *)
let far_blocks = 20_000

let far_matches =
  let rng = Random.State.make [| 7 |] in
  let out = Buffer.create (1 lsl 19) and frame = Buffer.create (1 lsl 19) in
  Buffer.add_string frame (windowed 0);
  for b = 1 to far_blocks do
    let lits =
      String.init (1 + Random.State.int rng 15) (fun _ ->
          Char.chr (97 + Random.State.int rng 26))
    in
    Buffer.add_string out lits;
    let have = Buffer.length out in
    let off = 1 + Random.State.int rng (min have 1024) in
    let ml = 3 + Random.State.int rng 32 in
    for _ = 1 to ml do
      Buffer.add_char out (Buffer.nth out (Buffer.length out - off))
    done;
    (* Offset_Value off + 3 is code c and c extra bits; the end mark
       follows them. *)
    let v = off + 3 in
    let rec high x k = if x <= 1 then k else high (x lsr 1) (k + 1) in
    let c = high v 0 in
    let bits = le ((c / 8) + 1) ((v - (1 lsl c)) lor (1 lsl c)) in
    Buffer.add_string frame
      (compressed ~last:(b = far_blocks)
         (raw_literals lits
         ^ one_sequence ~ll:(String.length lits) ~of_:c ~ml:(ml - 3) bits))
  done;
  (Buffer.contents frame, Buffer.contents out)

(* A frame with a 1 MiB window of 1 MiB of text in compressed blocks of
   100 raw literals and no sequence (a 2-byte literals header, Size_Format
   1, then Number_of_Sequences 0), as a stream flushed every 100 bytes
   writes. *)
let flushed =
  lazy
    (let text = Lz_support.corpus Text mib in
     let b = Buffer.create (mib + (mib / 25)) in
     Buffer.add_string b (windowed 10);
     let n = String.length text in
     let rec blocks i =
       let k = min 100 (n - i) in
       let lits = String.sub text i k in
       let body = byte (4 lor ((k land 0xF) lsl 4)) ^ byte (k lsr 4) ^ lits ^ "\x00" in
       Buffer.add_string b (compressed ~last:(i + k = n) body);
       if i + k < n then blocks (i + k)
     in
     blocks 0;
     (Buffer.contents b, text))

(* Stepped in large slices, a step takes its whole slice or lends a run
   that holds at least a block's 128 KiB: the steps that broke it, as
   (step, taken, slice, run). *)
let short_steps data slice =
  let src = Lz_support.of_string data and n = String.length data in
  let d = Lz.Zstd.decoder () in
  let out = Buffer.create (2 * mib) and broke = ref [] in
  let rec go at k =
    let len = min slice (n - at) in
    let run = ref 0 in
    let f r =
      run := Bigarray.Array1.dim r;
      Buffer.add_string out (Lz_support.to_string r)
    in
    match Lz.Decoder.step d ~last:(at + len = n) (Bigarray.Array1.sub src at len) f with
    | Error e -> failf "step %d: %a" k Lz.pp_error e
    | Ok (Done _) -> ()
    | Ok (More j) ->
        if j < len && !run < 128 * kib then broke := (k, j, len, !run) :: !broke;
        go (at + j) (k + 1)
  in
  go 0 0;
  (Buffer.contents out, List.rev !broke)

let small_blocks =
  cases ~name:string_of_int "a step of small blocks fills its run"
    [ max_int; 1 lsl 20; 300_000 ]
    (fun slice ->
      let data, raw = Lazy.force flushed in
      let out, broke = short_steps data slice in
      equal string raw out;
      equal (list (quad int int int int)) ~msg:"steps short of their slice with a run below a block"
        [] (List.filteri (fun i _ -> i < 3) broke))

(* Three frames whose windows grow, then shrink: a single-segment frame
   of 4 KiB, then a 1 MiB window over 300,000 bytes in raw blocks, then
   [far_matches] with its 1 KiB window. A step that takes the first frame
   and the second's header holds a run when the decoder needs a larger
   buffer; the third frame then decodes in the buffer the second one
   asked for, its matches across its wraps. *)
let growing =
  lazy
    (let text = Lz_support.corpus Text 304_096 in
     let a = String.sub text 0 4096 and b = String.sub text 4096 300_000 in
     let raws s =
       let n = String.length s in
       let rec go i acc =
         let k = min (128 * kib) (n - i) in
         let acc = raw ~last:(i + k = n) (String.sub s i k) :: acc in
         if i + k < n then go (i + k) acc else String.concat "" (List.rev acc)
       in
       go 0 []
     in
     let frame_c, out_c = far_matches in
     ( (magic ^ "\x60" ^ le 2 (4096 - 256) ^ raw a) ^ (windowed 10 ^ raws b) ^ frame_c,
       a ^ b ^ out_c ))

let grow_then_shrink =
  cases ~name:string_of_int "frames whose windows grow then shrink"
    [ max_int; 1 lsl 20; 4100; 97 ]
    (fun slice ->
      let data, expected = Lazy.force growing in
      let dst = Lz_support.create (String.length expected) in
      is_ok (Lz.Zstd.decode (Lz_support.of_string data) dst);
      equal string ~msg:"decode" expected (Lz_support.to_string dst);
      equal (result string Lz_support.error) ~msg:"steps" (Ok expected)
        (Result.map fst
           (Lz_support.stepped (Lz.Zstd.decoder ()) data ~cut:(fun _ -> slice)));
      (* In one slice, the first step's run holds the first frame and the
         second's first block: the buffer grew with a run pending. *)
      if slice = max_int then begin
        let first = ref 0 in
        is_ok
          (Lz.Decoder.step (Lz.Zstd.decoder ()) ~last:true (Lz_support.of_string data)
             (fun r -> first := Bigarray.Array1.dim r));
        less int ~msg:"the first frame's 4096 bytes, under the first run" ~than:!first 4096
      end)

let stream =
  group "stream"
    [
      steps;
      small_blocks;
      grow_then_shrink;
      test "matches across the stream's window wraps decode" (fun () ->
          let frame, expected = far_matches in
          let dst = Lz_support.create (String.length expected) in
          is_ok (Lz.Zstd.decode (Lz_support.of_string frame) dst);
          equal string expected (Lz_support.to_string dst);
          List.iter
            (fun seed ->
              equal outcome (Output expected)
                (stepped (Lz.Zstd.decoder ()) frame seed))
            [ 1; 2; 3; 4; 5 ]);
      test "a decoder is dead after an error" (fun () ->
          let d = Lz.Zstd.decoder () in
          let src = Lz_support.of_string "\x00" in
          is_error (Lz.Decoder.step d ~last:false src ignore);
          raises_match (Exn.invalid_arg ~substring:"dead") (fun () ->
              Lz.Decoder.step d ~last:false src ignore));
    ]

(* Paths *)

(* Every build of the hot loops gives Lz.Zstd.decode's result and output,
   into a destination of the decoded size, where the literals share the
   output's last bytes, or with room past it. *)
let paths =
  let run decode data room =
    let dst = Lz_support.create room in
    match decode (Lz_support.of_string data) dst with
    | Ok { Lz.produced; _ } as r -> (r, Lz_support.to_string ~length:produced dst)
    | Error _ as r -> (r, "")
  in
  let outcome = pair (result Lz_support.progress Lz_support.error) string in
  let pp ppf (k, change, exact) =
    Format.fprintf ppf "%a, %s" pp_change (k, change)
      (if exact then "exact" else "roomy")
  in
  group "paths"
    [
      test "the generic path is first" (fun () ->
          equal (option string) (Some "generic")
            (match Lz_support.zstd_paths with
            | (p : Lz_support.zstd_path) :: _ -> Some p.label
            | [] -> None));
      prop "every path decodes as Lz.Zstd.decode"
        Gen.(with_pp pp (triple golden_index change bool))
        (fun (k, change, exact) ->
          let data = changed k change in
          let size = String.length (Lazy.force (Lazy.force goldens).(k).raw) in
          let room = if exact then size else size + (256 * kib) in
          let expected = run Lz.Zstd.decode data room in
          cover "frames that decode" (Result.is_ok (fst expected));
          cover "a malformed input"
            (match fst expected with
            | Error { Lz.why = Malformed _; _ } -> true
            | _ -> false);
          List.iter
            (fun (p : Lz_support.zstd_path) ->
              equal ~msg:p.Lz_support.label outcome expected
                (run p.Lz_support.decode data room))
            Lz_support.zstd_paths);
      test "every path decodes matches across the window" (fun () ->
          let frame, expected = far_matches in
          List.iter
            (fun (p : Lz_support.zstd_path) ->
              let dst = Lz_support.create (String.length expected) in
              is_ok ~msg:p.Lz_support.label
                (p.Lz_support.decode (Lz_support.of_string frame) dst);
              equal string ~msg:p.label expected (Lz_support.to_string dst))
            Lz_support.zstd_paths);
    ]

(* Memory and lock *)

let allocation =
  let words f =
    let before = Gc.minor_words () in
    ignore (Sys.opaque_identity (f ()));
    Gc.minor_words () -. before
  in
  cases ~name:Fun.id "decode allocates only its result"
    [ "text_100.zst3"; "text_200000.zst19"; "window1m.zst"; "frames.zst" ]
    (fun name ->
      let src = Lz_support.of_string (data name) in
      let dst = Lz_support.create (String.length (raw_of name)) in
      let baseline = words (fun () -> ()) in
      (* Ok { consumed; produced }: 2 words for Ok, 3 for the record. *)
      equal float_exact 5. (words (fun () -> Lz.Zstd.decode src dst) -. baseline))

(* A long decode, of [k] times 336 frames of 200000 bytes, 64 MiB, leaves
   this domain's minor collections running. *)
let lock =
  let small =
    lazy (Lz_support.of_string (data "text_100.zst3"), Lz_support.create 100)
  in
  let long k =
    let one = data "text_200000.zst3" and frames = 336 * k in
    let src =
      Lz_support.of_string (String.concat "" (List.init frames (fun _ -> one)))
    in
    let dst = Lz_support.create (frames * 200_000) in
    fun () -> Lz.Zstd.decode src dst
  in
  Lz_support.lock_released "a long decode lets another domain collect" long
    ~small:(fun () ->
      let src, room = Lazy.force small in
      ignore (Sys.opaque_identity (Lz.Zstd.decode src room)))
    ~check:is_ok

let () =
  exit
    (run "Zstd"
       [
         Lz_support.agreement "zstd";
         frames;
         header;
         destinations;
         window;
         literals;
         sequences;
         stream;
         paths;
         allocation;
         lock;
         group "arguments"
           [
             cases ~name:string_of_int "a level outside 1 to 12 raises"
               [ 0; 13; -1 ] (fun level ->
                 let src = Lz_support.of_string "abc" in
                 let dst = Lz_support.create 64 in
                 raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
                     Lz.Zstd.encode ~level src dst);
                 raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
                     Lz.Zstd.encoder ~level ()));
             cases ~name:string_of_int "a window below 1 KiB raises"
               [ -1; 0; 1023 ] (fun window ->
                 raises_match (Exn.invalid_arg ~substring:"window") (fun () ->
                     Lz.Zstd.decoder ~window ()));
             test "a negative size has no bound" (fun () ->
                 raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
                     Lz.Zstd.bound (-1)));
             Lz_support.largest_bound Lz.Zstd.bound;
           ];
       ])
