(* Zstandard decoding (RFC 8878): the decode's memory, the decoder's runs and
   buffers, the specification's refusals on frames built byte by byte, the
   vendor's goldens against the destination, and the decoder's windows. Round
   trips, levels and the bound belong to the encoder; Step states the decoder's
   laws over every format.

   An error's offset is the first byte of what is wrong: of a field the format
   refuses or the library declines, of a section or table description that runs
   past its block, of a bitstream not read exactly to its start; the byte
   holding a sequence's first bit when the sequence breaks a rule; a frame's
   first byte when no frame begins there; the input's length when it ends inside
   a frame. A block that asks for more output than the destination holds is
   Too_long at its header. A refusal's message names the RFC 8878 field at
   fault, which each test checks where the specification names one. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib

(* Bytes *)

let byte v = String.make 1 (Char.chr v)
let le n v = String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xFF))

(* Section 3.1.1: the magic number. *)
let magic = "\x28\xB5\x2F\xFD"

(* Section 3.1.1.1: a header with no content size whose window is 2^(10 +
   exponent) bytes: the descriptor, then the Window_Descriptor. It ends at
   offset 6. *)
let windowed exponent = magic ^ "\x00" ^ byte (exponent lsl 3)

(* A single-segment header whose window is its content size [n] < 256,
   Frame_Content_Size in 1 byte. It ends at offset 6. *)
let single n = magic ^ "\x20" ^ byte n

(* A header of a 1 KiB window and a content size [n] of 256 to 65791 bytes,
   Frame_Content_Size in 2 bytes at offset 6. It ends at offset 8. *)
let sized n = magic ^ "\x40\x00" ^ le 2 (n - 256)

(* Section 3.1.1.2: a block header, then the block. *)
let block_header ~last typ size =
  le 3 ((size lsl 3) lor (typ lsl 1) lor if last then 1 else 0)

let raw ?(last = true) s = block_header ~last 0 (String.length s) ^ s
let rle ?(last = true) c n = block_header ~last 1 n ^ String.make 1 c
let compressed ?(last = true) s = block_header ~last 2 (String.length s) ^ s

(* Section 3.1.1.3.1.1: raw literals, fewer than 32, in a 1-byte header. *)
let raw_literals s = byte (String.length s lsl 3) ^ s

(* Section 3.1.1.3.2.1: one sequence whose three codes are in RLE_Mode, so that
   its bitstream [bits] holds only their extra bits. *)
let one_sequence ~ll ~of_ ~ml bits =
  "\x01\x54" ^ byte ll ^ byte of_ ^ byte ml ^ bits

(* The highest set bit of [x], positive: an Offset_Value's code. *)
let rec high x = if x <= 1 then 0 else 1 + high (x lsr 1)

(* Section 4.1.1, as an encoder: the accuracy log, then each probability in the
   fewest bits that hold any value up to the points left plus one, low values
   one bit shorter (Table 20); a zero is followed by a 2-bit count of further
   zeros, here 0. *)
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
  add (log - 5) 4;
  let left = ref ((1 lsl log) + 1) in
  List.iter
    (fun p ->
      let value = p + 1 in
      let bits = high !left + 1 in
      let threshold = 1 lsl (bits - 1) in
      let low = (2 * threshold) - 1 - !left in
      if value < low then add value (bits - 1)
      else if value < threshold then add value bits
      else add (value + low) bits;
      left := !left - if p < 0 then 1 else p;
      if p = 0 then add 0 2)
    probs;
  if !n > 0 then Buffer.add_char out (Char.chr (!acc land 0xFF));
  Buffer.contents out

(* Results *)

let decode_result = result Lz_support.progress Lz_support.error
let pp_length ppf s = Format.fprintf ppf "%d decoded bytes" (String.length s)

(* [decodes s out] checks that decoding [s] into [cap] bytes reads all of it and
   writes [out]. *)
let decodes ?(cap = 4096) s out =
  let dst = Lz_support.create cap in
  equal decode_result
    (Ok { Lz.consumed = String.length s; produced = String.length out })
    (Lz.Zstd.decode (Lz_support.of_string s) dst);
  equal Lz_support.decoded (Ok out)
    (Ok (Lz_support.to_string ~length:(String.length out) dst))

(* [refused ?field why ~at e] checks that [e] has [why]'s kind and the offset
   [at] and, given [field], that its message names it: RFC 8878's name for what
   is wrong. *)
let refused ?field why ~at e =
  equal Lz_support.refusal { Lz.at; why } e;
  match (field, e.Lz.why) with
  | Some sub, (Malformed m | Unsupported m) ->
      contains ~msg:"the message names the field at fault" ~sub m
  | Some _, (Truncated | Too_long) | None, _ -> ()

(* The error decoding [s] into [cap] bytes returns. *)
let error_of ?(cap = 4096) s =
  let dst = Lz_support.create cap in
  require_error
    ~pp:(Testable.pp Lz_support.progress)
    (Lz.Zstd.decode (Lz_support.of_string s) dst)

let malformed ?cap ?field ~at s =
  refused ?field
    (Malformed (Option.value field ~default:""))
    ~at (error_of ?cap s)

let unsupported ~field ~at e = refused ~field (Unsupported field) ~at e
let truncated ~at s = refused Truncated ~at (error_of s)
let too_long ?cap ~at s = refused Too_long ~at (error_of ?cap s)

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

(* Decoding *)

(* The bytes [f] allocates on the OCaml heap, minor or major, with the bytes
   that counting them allocates. *)
let allocated f =
  let before = Gc.allocated_bytes () in
  ignore (Sys.opaque_identity (f ()));
  Gc.allocated_bytes () -. before

let allocation =
  cases "decode allocates only its result" ~id:Fun.id
    [ "text_100.zst3"; "text_200000.zst19"; "window1m.zst"; "frames.zst" ]
    (fun name ->
      let src = Lz_support.of_string (data name) in
      let dst = Lz_support.create (String.length (raw_of name)) in
      let baseline = allocated (fun () -> ()) in
      (* Ok { consumed; produced }: 2 words for Ok, 3 for the record. *)
      equal float_exact
        (Float.of_int (5 * (Sys.word_size / 8)))
        (allocated (fun () -> Lz.Zstd.decode src dst) -. baseline))

(* Decoders *)

(* A frame with a 1 MiB window of 1 MiB of text in compressed blocks of 100 raw
   literals and no sequence (a 2-byte literals header, Size_Format 1, then
   Number_of_Sequences 0), as a stream flushed every 100 bytes writes. Its
   Block_Maximum_Size is 128 KiB. *)
let flushed =
  lazy
    (let text = Lz_support.corpus Text mib in
     let b = Buffer.create (mib + (mib / 25)) in
     Buffer.add_string b (windowed 10);
     let n = String.length text in
     let rec blocks i =
       let k = min 100 (n - i) in
       let lits = String.sub text i k in
       let body =
         byte (4 lor ((k land 0xF) lsl 4)) ^ byte (k lsr 4) ^ lits ^ "\x00"
       in
       Buffer.add_string b (compressed ~last:(i + k = n) body);
       if i + k < n then blocks (i + k)
     in
     blocks 0;
     (Buffer.contents b, text))

(* [stepped_runs data slice] steps a decoder over [data] in slices of [slice]
   bytes: the runs concatenated, and the length of the run of each step that
   took less than its slice. *)
let stepped_runs data slice =
  let src = Lz_support.of_string data and n = String.length data in
  let d = Lz.Zstd.decoder () in
  let out = Buffer.create (2 * mib) and full = ref [] in
  let rec go at k =
    let len = min slice (n - at) in
    let run = ref 0 in
    let f r =
      run := Bigarray.Array1.dim r;
      Buffer.add_string out (Lz_support.to_string r)
    in
    let src = Bigarray.Array1.sub src at len in
    match Lz.Decoder.step d ~last:(at + len = n) src f with
    | Error e -> failf "step %d: %a" k Lz.pp_error e
    | Ok (Done _) -> ()
    | Ok (More j) ->
        if j < len then full := !run :: !full;
        go (at + j) (k + 1)
  in
  go 0 0;
  (Buffer.contents out, !full)

(* A run is full once it holds the run capacity, 256 KiB, or once its next block
   might pass that capacity or the end of the decoder's buffer. The frame's
   Block_Maximum_Size is 128 KiB, so a run of less than 256 - 128 KiB cannot
   pass the capacity by one block; and the buffer holds the 1 MiB window and the
   run capacity, whose end the frame's 1 MiB of output never reaches. So a step
   that stops short holds at least 128 KiB and at most 256 KiB. The output
   passes the capacity, so a step on the whole frame stops short. *)
let full_runs =
  cases "a step that stops short holds a full run" ~id:string_of_int
    [ max_int; 1 lsl 20; 300_000 ]
    (fun slice ->
      let data, raw = Lazy.force flushed in
      let out, full = stepped_runs data slice in
      equal Lz_support.decoded ~msg:"the runs" (Ok raw) (Ok out);
      if full = [] then fail "no step stopped short of its slice";
      equal (list int) ~msg:"runs of steps that stopped short, out of range" []
        (List.filter
           (fun run -> run < (256 - 128) * kib || run > 256 * kib)
           full))

(* Frames whose Block_Maximum_Size is each block they hold: a compressed block
   of that size, of raw literals; then compressed blocks that regenerate that
   many literals, RLE and Huffman. Their windows grow from 1 KiB to 256 KiB,
   past the 128 KiB a block may hold, then shrink to 1 KiB again. A decoder that
   steps them gathers each block, whole or in pieces, and puts its literals in
   buffers each frame sizes. *)
let largest =
  lazy
    (let text = Lz_support.corpus Text (128 * kib) in
     (* Section 3.1.1.3.1.1, Size_Format 11: a 20-bit size in 3 bytes. *)
     let lits typ n = le 3 (typ lor (3 lsl 2) lor (n lsl 4)) in
     (* Section 3.1.1.3.1.6: four streams of [n / 4] literals each. Weights 1
        and 1 complete with a 2, whose code is the 1 bit 1: every literal is
        symbol 2. Each stream ends with its end mark. *)
     let huffman n =
       let stream = String.make (n / 32) '\xFF' ^ "\x01" in
       let len = String.length stream in
       let tree = byte 129 ^ byte 0x11 in
       let csize = String.length tree + 6 + (4 * len) in
       le 5 (2 lor (3 lsl 2) lor (n lsl 4) lor (csize lsl 22))
       ^ tree ^ le 2 len ^ le 2 len ^ le 2 len
       ^ String.concat "" (List.init 4 (fun _ -> stream))
     in
     let frame exponent =
       let b = min (128 * kib) (1 lsl (10 + exponent)) in
       let r = String.sub text 0 (b - 4) in
       ( windowed exponent
         ^ compressed ~last:false (lits 0 (b - 4) ^ r ^ "\x00")
         ^ compressed ~last:false (lits 1 b ^ "a" ^ "\x00")
         ^ compressed (huffman b ^ "\x00"),
         r ^ String.make b 'a' ^ String.make b '\x02' )
     in
     let frames = List.map frame [ 0; 3; 6; 7; 8; 0 ] in
     ( String.concat "" (List.map fst frames),
       String.concat "" (List.map snd frames) ))

let at_block_max =
  cases "blocks of their frame's Block_Maximum_Size decode as windows grow"
    ~id:string_of_int [ max_int; 4099; 997 ] (fun slice ->
      let data, expected = Lazy.force largest in
      decodes ~cap:(String.length expected) data expected;
      equal Lz_support.decoded ~msg:"steps" (Ok expected)
        (Result.map fst
           (Lz_support.stepped (Lz.Zstd.decoder ()) data ~cut:(fun _ -> slice))))

(* A frame with a 1 KiB window whose 20,000 blocks are each a few raw literals
   and one match up to the window back, built with the specification's codes
   beside the output it states: about 500 KB, so that a stream's buffer wraps
   several times over it however the input is cut, and matches cross from one
   segment into the next. *)
let far_blocks = 20_000

let far_matches =
  lazy
    (let rng = Random.State.make [| 7 |] in
     let out = Buffer.create (1 lsl 19) and frame = Buffer.create (1 lsl 19) in
     Buffer.add_string frame (windowed 0);
     for b = 1 to far_blocks do
       let lits =
         String.init
           (1 + Random.State.int rng 15)
           (fun _ -> Char.chr (97 + Random.State.int rng 26))
       in
       Buffer.add_string out lits;
       let have = Buffer.length out in
       let off = 1 + Random.State.int rng (min have 1024) in
       let ml = 3 + Random.State.int rng 32 in
       for _ = 1 to ml do
         Buffer.add_char out (Buffer.nth out (Buffer.length out - off))
       done;
       (* Offset_Value off + 3 is code c and c extra bits; the end mark follows
          them. *)
       let v = off + 3 in
       let c = high v in
       let bits = le ((c / 8) + 1) ((v - (1 lsl c)) lor (1 lsl c)) in
       Buffer.add_string frame
         (compressed ~last:(b = far_blocks)
            (raw_literals lits
            ^ one_sequence ~ll:(String.length lits) ~of_:c ~ml:(ml - 3) bits))
     done;
     (Buffer.contents frame, Buffer.contents out))

(* A slice: one byte, a few, up to 4 KiB, or up to the run capacity. *)
let slice =
  Gen.frequency
    [
      (1, Gen.constant 1);
      (1, Gen.int_range 2 16);
      (1, Gen.int_range 17 4096);
      (1, Gen.int_range 4097 (256 * kib));
    ]

(* Each case steps the whole frame, up to 375,000 steps of one byte: 25 cases
   keep the test near a tenth of a second. *)
let wrapping =
  prop ~count:25
    "a frame whose matches cross its buffer's wraps decodes however cut"
    Gen.(
      with_pp
        Format.(
          pp_print_list
            ~pp_sep:(fun ppf () -> pp_print_string ppf "; ")
            pp_print_int)
        (list ~size:(int_range 1 4) slice))
    (fun slices ->
      let frame, expected = Lazy.force far_matches in
      let slices = Array.of_list slices in
      let cut k = slices.(k mod Array.length slices) in
      equal Lz_support.decoded (Ok expected)
        (Result.map fst (Lz_support.stepped (Lz.Zstd.decoder ()) frame ~cut)))

(* Frames *)

let frames =
  group "frames"
    [
      test "skippable frames produce nothing" (fun () ->
          decodes (data "skippable.zst") "");
      test "frames and skippable frames decode in turn" (fun () ->
          let a = "text_100.zst3" and b = "columns_100.zst1" in
          decodes ~cap:200
            (data a ^ "\x5A\x2A\x4D\x18\x03\x00\x00\x00abc" ^ data b)
            (raw_of a ^ raw_of b));
      test "bytes that begin no frame are Malformed where they begin" (fun () ->
          let a = data "text_100.zst3" in
          malformed ~field:"Magic_Number" ~at:(String.length a)
            (a ^ "\x28\xB5\x00"));
      test "a cut inside the next magic is Truncated" (fun () ->
          let a = data "text_100.zst3" in
          truncated ~at:(String.length a + 2) (a ^ "\x28\xB5"));
      test "an empty input is Truncated at 0" (fun () -> truncated ~at:0 "");
      cases "a magic no frame begins with is Malformed at its first byte"
        ~id:fst
        [
          ("its first byte", "\x00\xB5\x2F\xFD");
          ("its third byte", "\x28\xB5\x2E\xFD");
          ("a skippable magic's second byte, cut", "\x5F\x2B");
        ]
        (fun (_, s) -> malformed ~field:"Magic_Number" ~at:0 s);
    ]

(* Frame headers *)

let header =
  group "header"
    [
      test "a reserved descriptor bit is Malformed" (fun () ->
          malformed ~field:"Frame_Header_Descriptor" ~at:4
            (magic ^ "\x08\x00" ^ raw "abc"));
      test "the unused descriptor bit is ignored" (fun () ->
          decodes (magic ^ "\x10\x00" ^ raw "abc") "abc");
      cases "a dictionary identifier is Unsupported at its first byte" ~id:fst
        [ ("1 byte", "\x01\x00\x07"); ("4 bytes", "\x03\x00\x00\x00\x00\x80") ]
        (fun (_, h) ->
          unsupported ~field:"Dictionary_ID" ~at:6
            (error_of (magic ^ h ^ raw "abc")));
      test "a dictionary identifier of 0 names no dictionary" (fun () ->
          decodes (magic ^ "\x01\x00\x00" ^ raw "abc") "abc");
    ]

(* Blocks *)

let blocks =
  group "blocks"
    [
      test "the reserved block type is Malformed" (fun () ->
          malformed ~field:"Block_Type" ~at:6
            (windowed 0 ^ block_header ~last:true 3 1 ^ "a"));
      cases "a block above 128 KiB is Malformed at its header" ~id:fst
        [ ("raw", 0); ("RLE", 1); ("compressed", 2) ]
        (fun (_, typ) ->
          let n = (128 * kib) + 1 in
          malformed ~cap:(2 * n) ~field:"Block_Size" ~at:6
            (windowed 10 ^ block_header ~last:true typ n ^ String.make n 'a'));
      test "a block of 128 KiB decodes" (fun () ->
          let n = 128 * kib in
          decodes ~cap:(2 * n) (windowed 10 ^ rle 'a' n) (String.make n 'a'));
      cases "a block above the window is Malformed at its header" ~id:fst
        [
          ("a single segment of 100", single 100 ^ raw (String.make 101 'a'));
          ("a window of 1 KiB", windowed 0 ^ raw (String.make 1025 'a'));
        ]
        (fun (_, s) -> malformed ~field:"Block_Size" ~at:6 s);
    ]

(* Literals *)

let literals =
  (* A frame of one compressed block, whose content starts at 9: literals
     [lits], then no sequences. *)
  let frame lits = windowed 0 ^ compressed (lits ^ "\x00") in
  (* Section 3.1.1.3.1.1, Size_Format 00 (one stream) or 01 (four): 10-bit sizes
     in a 3-byte header. *)
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
    ^ String.init
        ((n + 1) / 2)
        (fun i -> Char.chr ((a.(2 * i) lsl 4) lor a.((2 * i) + 1)))
  in
  (* Section 4.2.1.2: weights FSE-coded by a table of accuracy log 5 with
     [probs], then their bitstream [bits]. The table's description takes 2 bytes
     here, so the bitstream starts at 15. *)
  let fse_weights ?(probs = [ 16; 16 ]) bits =
    let d = ncount ~log:5 probs ^ bits in
    byte (String.length d) ^ d
  in
  group "literals"
    [
      test "a Literals_Section_Header past the block is Malformed at it"
        (fun () ->
          (* Size_Format 01: a 2-byte header in a 1-byte block. *)
          malformed ~field:"Literals_Section_Header" ~at:9
            (windowed 0 ^ compressed "\x04"));
      test
        "an empty block's Literals_Section_Header is at the header's last byte"
        (fun () ->
          malformed ~field:"Literals_Section_Header" ~at:8
            (windowed 0 ^ compressed ""));
      cases "raw or RLE literals past the block are Malformed at their header"
        ~id:fst
        [
          ("Raw_Literals_Block", "\x28abc");
          ("RLE_Literals_Block", byte (1 lor (5 lsl 3)));
        ]
        (fun (field, lits) ->
          malformed ~field ~at:9 (windowed 0 ^ compressed lits));
      (* RLE literals, Size_Format 01: Regenerated_Size in 12 bits. The frame's
         window, so its Block_Maximum_Size, is 1 KiB. *)
      test "a Regenerated_Size above Block_Maximum_Size is Malformed at it"
        (fun () ->
          let rle n = frame (le 2 (1 lor 4 lor (n lsl 4)) ^ "x") in
          decodes (rle 1024) (String.make 1024 'x');
          malformed ~field:"Regenerated_Size" ~at:9 (rle 1025));
      (* Each of the first three streams decodes (n + 3) / 4 literals (RFC 8878,
         3.1.1.3.1.6), and the fourth the rest: 1, 2 and 5 literals leave it
         fewer than none. *)
      cases
        "four streams of 1, 2 or 5 literals are Malformed at Regenerated_Size"
        ~id:string_of_int [ 1; 2; 5 ] (fun n ->
          malformed ~field:"Regenerated_Size" ~at:9
            (frame
               (huffman ~four:true n (direct [ 1; 1 ]) (String.make 10 '\x01'))));
      (* Weights 1 and 1 complete with a 2: codes 00, 01 and 1 for 0, 1 and 2.
         Section 3.1.1.3.1.6: each stream regenerates (n + 3) / 4 literals, the
         last up to 3 fewer, none when n is 3. A stream holds its codes, then
         its end mark. *)
      cases "four streams regenerate as few as 3 literals" ~id:string_of_int
        [ 3; 4 ] (fun n ->
          let streams =
            [ "\x03"; "\x04"; "\x05"; (if n = 4 then "\x03" else "\x01") ]
          in
          let jump = le 2 1 ^ le 2 1 ^ le 2 1 in
          decodes
            (frame
               (huffman ~four:true n
                  (direct [ 1; 1 ])
                  (jump ^ String.concat "" streams)))
            (String.sub "\x02\x00\x01\x02" 0 n));
      (* Compressed_Size, from bit 14 of the header, in byte 10. *)
      test "a Compressed_Size past the block is Malformed at it" (fun () ->
          let lits = huffman 4 (direct [ 2; 2; 2 ]) "\x1B\x01" in
          malformed ~field:"Compressed_Size" ~at:10
            (windowed 0
            ^ compressed (String.sub lits 0 (String.length lits - 1))));
      (* 4 bytes, the tree's 2 and 2 of streams, short of the 6-byte jump
         table. *)
      test "a Compressed_Size too small for four streams is Malformed at it"
        (fun () ->
          malformed ~field:"Compressed_Size" ~at:10
            (frame (huffman ~four:true 4 (direct [ 1; 1 ]) "\x01\x01")));
      test "treeless literals with no earlier tree are Malformed" (fun () ->
          malformed ~field:"Treeless_Literals_Block" ~at:9
            (frame (huffman ~typ:3 4 "" "\x01")));
      test "a direct tree past its block is Malformed at its first byte"
        (fun () ->
          (* 128 weights in 64 bytes, of which the section holds none. *)
          malformed ~field:"Huffman_Tree_Description" ~at:12
            (frame (huffman 4 "\xFF" "")));
      test "a weight above 11 is Malformed at its byte" (fun () ->
          (* Weights 1 and 1 in the description's byte 13, 12 in 14. *)
          malformed ~field:"Weight" ~at:14
            (frame (huffman 4 (direct [ 1; 1; 12 ]) "\x01")));
      cases
        "Huffman weights that build no prefix code are Malformed at the \
         description's first byte"
        ~id:(fun (n, _) -> n)
        [
          ("weights that complete to no power of 2", [ 3; 1 ]);
          ("codes longer than 11 bits", [ 11; 11 ]);
        ]
        (fun (_, ws) -> malformed ~at:12 (frame (huffman 4 (direct ws) "\x01")));
      test "weights complete to a code without a weight of 1" (fun () ->
          (* Weights 2, 2, 2 complete with a fourth 2: four 2-bit codes, 00 to
             11 by symbol. The stream reads them from its last byte, after the
             end mark: 00 01 10 11. *)
          decodes
            (frame (huffman 4 (direct [ 2; 2; 2 ]) "\x1B\x01"))
            "\x00\x01\x02\x03");
      test "an FSE description past its section is Malformed at its first byte"
        (fun () ->
          (* Weights FSE-coded in 1 byte at 13, accuracy log 5: its first
             probability needs bits past the byte. *)
          malformed ~at:13 (frame (huffman 4 "\x01\x00" "\x01")));
      test "a weights' Accuracy_Log above 6 is Malformed at its description"
        (fun () ->
          (* The description at 13, its first 4 bits 2: accuracy log 7. *)
          malformed ~field:"Accuracy_Log" ~at:13
            (frame (huffman 4 "\x01\x02" "\x01")));
      test "a weights' bitstream whose last byte is 0 is Malformed at it"
        (fun () ->
          malformed ~at:15 (frame (huffman 4 (fse_weights "\x00") "\x01")));
      test "a weights' bitstream not read to its start is Malformed at it"
        (fun () ->
          (* Its end mark alone, with no bits for the two 5-bit states. *)
          malformed ~at:15 (frame (huffman 4 (fse_weights "\x01") "\x01")));
      test "a weights' bitstream of more than 255 weights is Malformed at it"
        (fun () ->
          (* One weight holds all 32 points: its states read no bits past the
             two 5-bit ones, which the 10 bits under the end mark hold. *)
          malformed ~at:15
            (frame (huffman 4 (fse_weights ~probs:[ 32 ] "\x00\x04") "\x01")));
      test "a jump table past its streams is Malformed at its first byte"
        (fun () ->
          (* The jump table, at 14, states 9 bytes of streams of 4. *)
          malformed ~field:"Jump_Table" ~at:14
            (frame
               (huffman ~four:true 8
                  (direct [ 1; 1 ])
                  (le 2 3 ^ le 2 3 ^ le 2 3 ^ String.make 4 '\x01'))));
      test "a Huffman stream whose last byte is 0 is Malformed at that byte"
        (fun () ->
          (* The stream "\x1B\x01" of the weights 2, 2, 2 above, its end mark
             cleared: the stream is at 15 and 16. *)
          malformed ~at:16 (frame (huffman 4 (direct [ 2; 2; 2 ]) "\x1B\x00")));
      test "a Huffman stream not read to its start is Malformed at it"
        (fun () ->
          (* The stream of four 2-bit codes above, for 3 literals: 2 bits are
             left. *)
          malformed ~at:15 (frame (huffman 3 (direct [ 2; 2; 2 ]) "\x1B\x01")));
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
      test "a Sequences_Section_Header past the block is Malformed at it"
        (fun () ->
          (* Number_of_Sequences 1, then no Symbol_Compression_Modes. *)
          malformed ~field:"Sequences_Section_Header" ~at:14 (frame "\x01"));
      test "a sequence copies literals then a match" (fun () ->
          (* Offset code 2 with extra bits 00: Offset_Value 4, offset 1. *)
          decodes (frame (one_sequence ~ll:4 ~of_:2 ~ml:0 "\x04")) "abcdddd");
      (* Offset code 5: Offset_Value 32, offset 29, after 4 bytes. A frame
         before, of 100 bytes, holds the bytes the offset reaches. *)
      cases "an offset before the frame's start is Malformed" ~id:fst
        [
          ("the input's first frame", "");
          ("after a frame", single 100 ^ raw (String.make 100 'a'));
        ]
        (fun (_, before) ->
          let n = String.length before in
          malformed ~at:(n + 19)
            (before ^ frame (one_sequence ~ll:4 ~of_:5 ~ml:0 "\x20")));
      test "a repeat offset of 0 is Malformed" (fun () ->
          (* Literals length 0 and Offset_Value 3: Repeated_Offset1 - 1. *)
          malformed ~at:19 (frame (one_sequence ~ll:0 ~of_:1 ~ml:0 "\x03")));
      test "a literals length past the literals is Malformed" (fun () ->
          malformed ~field:"Literals_Length" ~at:19
            (frame (one_sequence ~ll:5 ~of_:2 ~ml:0 "\x04")));
      test "a match past the block's maximum is Malformed" (fun () ->
          (* A window, so a block, of 20 bytes; 4 + 34 bytes. *)
          malformed ~field:"Block_Maximum_Size" ~at:19
            (frame ~head:(single 20) (one_sequence ~ll:4 ~of_:2 ~ml:31 "\x04")));
      (* Two sequences of 3 literals, offset code 2 and match length code 43, 2
         and 7 extra bits each, in a 3-byte bitstream at 19 whose end mark is
         bit 18: the first sequence's bits are 17 to 9, from byte 21, the
         second's 8 to 0, from byte 20. With literals "abcd" the second has 1
         literal left; with offset extra bits 11 the first reaches 4 back after
         3 bytes. *)
      cases "an error in a sequence is at the byte holding its first bit"
        ~id:fst
        [
          ("the second's literals", ("\x00\x00\x04", Some "Literals_Length", 20));
          ("the first's offset", ("\x00\x00\x07", None, 21));
        ]
        (fun (_, (bits, field, at)) ->
          malformed ?field ~at (frame ("\x02\x54\x03\x02\x2B" ^ bits)));
      test "a sequences bitstream whose last byte is 0 is Malformed at it"
        (fun () ->
          malformed ~at:19 (frame (one_sequence ~ll:4 ~of_:2 ~ml:0 "\x00")));
      test "a sequences bitstream not read to its start is Malformed" (fun () ->
          malformed ~at:19 (frame (one_sequence ~ll:4 ~of_:2 ~ml:0 "\x08")));
      cases "an RLE_Mode symbol above its code's maximum is Malformed at it"
        ~id:(fun (n, _, _) -> n)
        [
          ("literals length 36", one_sequence ~ll:36 ~of_:2 ~ml:0 "\x04", 16);
          ("offset 32", one_sequence ~ll:4 ~of_:32 ~ml:0 "\x04", 17);
          ("match length 53", one_sequence ~ll:4 ~of_:2 ~ml:53 "\x04", 18);
        ]
        (fun (_, seqs, at) -> malformed ~field:"RLE_Mode" ~at (frame seqs));
      test "an RLE_Mode symbol past the block is Malformed at its last byte"
        (fun () ->
          (* The literals length's symbol at 16 ends the block: the offsets'
             would begin past it. *)
          malformed ~field:"RLE_Mode" ~at:16 (frame "\x01\x54\x04"));
      test "a byte after a Number_of_Sequences of 0 is Malformed at it"
        (fun () ->
          malformed ~field:"Number_of_Sequences" ~at:15 (frame "\x00x"));
      (* A 1 KiB window after 2000 bytes of output, in RLE blocks of 1000 'a'
         then 1000 'b', the most a block holds being the window. The third
         block's one sequence has no literals and a match of 3; Offset_Value is
         the offset plus 3, its code the value's highest bit, its extra bits the
         rest, under an end mark in the 2-byte bitstream at 23 and 24. *)
      test "an offset above Window_Size is Malformed at the sequence" (fun () ->
          let reaching offset =
            let v = offset + 3 in
            let code = high v in
            let bits = le 2 ((1 lsl code) lor (v - (1 lsl code))) in
            windowed 0 ^ rle ~last:false 'a' 1000 ^ rle ~last:false 'b' 1000
            ^ compressed
                (raw_literals "" ^ one_sequence ~ll:0 ~of_:code ~ml:0 bits)
          in
          let before = String.make 1000 'a' ^ String.make 1000 'b' in
          decodes (reaching 1000) (before ^ "bbb");
          decodes (reaching 1024) (before ^ "aaa");
          malformed ~field:"Window_Size" ~at:24 (reaching 1025));
      test "reserved compression mode bits are Malformed" (fun () ->
          malformed ~field:"Symbol_Compression_Modes" ~at:15
            (frame "\x01\x55\x04\x02\x00\x04"));
      test "a repeated table with none before is Malformed at its mode"
        (fun () -> malformed ~field:"Repeat_Mode" ~at:15 (frame "\x01\xC0\x04"));
      test "an offset coded with an FSE table decodes" (fun () ->
          (* Offsets coded by a table of accuracy log 5 giving code 2 all 32
             points: a 5-bit state, then code 2's 2 extra bits. *)
          decodes
            (frame ("\x01\x64\x04" ^ ncount ~log:5 [ 0; 0; 32 ] ^ "\x00\x80"))
            "abcdddd");
      test "an FSE description cut by the block is Malformed at its first byte"
        (fun () ->
          (* Literals lengths, at 16, accuracy log 6: the first probability
             needs bits past the block. *)
          malformed ~at:16 (frame "\x01\x80\x01"));
      cases "an accuracy log above the limit is Malformed"
        ~id:(fun (n, _) -> n)
        [
          ("literals lengths, 10 above 9", "\x01\x80\x05");
          ("offsets, 9 above 8", "\x01\x20\x04");
          ("match lengths, 10 above 9", "\x01\x08\x05");
        ]
        (fun (_, s) -> malformed ~field:"Accuracy_Log" ~at:16 (frame s));
      (* 36 literals length codes of 1 point each, 36 of 64: a 37th probability
         begins in the description's last byte. Offset codes 0 to 30 at 1 point
         and 31 at 0: 31 of 32, and a 33rd probability begins in the
         description's last byte. *)
      cases "a distribution short of its total is Malformed past its alphabet"
        ~id:fst
        [
          ( "literals lengths",
            "\x01\x80" ^ ncount ~log:6 (List.init 36 (fun _ -> 1)) );
          ( "offsets, to the last code",
            "\x01\x20" ^ ncount ~log:5 (List.init 31 (fun _ -> 1) @ [ 0 ]) );
        ]
        (fun (_, s) -> malformed ~at:(14 + String.length s - 1) (frame s));
    ]

(* Frame content *)

let content =
  let xs = String.make 300 'x' in
  group "content"
    [
      (* Frame_Content_Size, at 6, is what does not match, however the blocks
         show it: a raw block's header, the last block, or a compressed block's
         output. *)
      cases "a Frame_Content_Size other than the output is Malformed at it"
        ~id:fst
        [
          ("301, a raw block of 300", sized 301 ^ raw xs);
          ( "301, raw blocks of 200 and 100",
            sized 301
            ^ raw ~last:false (String.sub xs 0 200)
            ^ raw (String.sub xs 0 100) );
          ("299, a raw block of 300", sized 299 ^ raw xs);
          ( "299, raw blocks of 200 and 100",
            sized 299
            ^ raw ~last:false (String.sub xs 0 200)
            ^ raw (String.sub xs 0 100) );
          ( "257, a compressed block of 4",
            sized 257 ^ compressed (raw_literals "abcd" ^ "\x00") );
          ( "256, a compressed block past it",
            sized 256
            ^ raw ~last:false (String.sub xs 0 254)
            ^ compressed (raw_literals "abcd" ^ "\x00") );
        ]
        (fun (_, s) -> malformed ~field:"Frame_Content_Size" ~at:6 s);
      test "a wrong checksum is Malformed at the checksum" (fun () ->
          let s = data "text_100.zst3" in
          let n = String.length s in
          let b = Bytes.of_string s in
          Bytes.set b (n - 1) (Char.chr (Char.code s.[n - 1] lxor 1));
          malformed ~field:"Content_Checksum" ~at:(n - 4) (Bytes.to_string b));
      test "an error before a wrong checksum is the one returned" (fun () ->
          (* Content_Checksum_Flag set, then the offset before the frame's start
             of the sequences above, at 19, and a checksum of 0. *)
          malformed ~at:19
            (magic ^ "\x04\x00"
            ^ compressed
                (raw_literals "abcd" ^ one_sequence ~ll:4 ~of_:5 ~ml:0 "\x20")
            ^ "\x00\x00\x00\x00"));
    ]

(* Destinations *)

let destinations =
  group "destinations"
    [
      cases "a destination of the decoded size holds the output"
        ~id:(fun g -> g.Lz_support.file)
        (Lz_support.goldens_of "zstd")
        (fun g ->
          let raw = Lazy.force g.Lz_support.raw in
          let data = Lazy.force g.data in
          let exact = guarded (String.length raw) in
          equal decode_result
            (Ok
               {
                 Lz.consumed = String.length data;
                 produced = String.length raw;
               })
            (Lz.Zstd.decode (Lz_support.of_string data) (snd exact));
          equal Lz_support.decoded (Ok raw)
            (Ok (Lz_support.to_string (snd exact)));
          untouched exact);
      (* The offset is the header of the block that writes the last byte, which
         the hand-built frames above state. *)
      cases "a destination one byte short is Too_long"
        ~id:(fun g -> g.Lz_support.file)
        (Lz_support.goldens_of "zstd")
        (fun g ->
          let n = String.length (Lazy.force g.Lz_support.raw) in
          if n = 0 then
            skip ~reason:"an empty output has no shorter destination" ();
          let src = Lz_support.of_string (Lazy.force g.data) in
          let short = guarded (n - 1) in
          satisfies ~claim:"Error Too_long" decode_result
            (function Error { Lz.why = Too_long; _ } -> true | _ -> false)
            (Lz.Zstd.decode src (snd short));
          untouched short);
      test "a block's output past the destination is Too_long at its header"
        (fun () ->
          (* Literals "abcd", then a sequence that writes 102 bytes. *)
          too_long ~cap:50 ~at:6
            (windowed 0
            ^ compressed
                (raw_literals "abcd" ^ "\x02\x54\x03\x02\x2B\x00\x00\x04")));
      (* Each input asks for output past its destination, at a block header or a
         sequence that breaks the format: the header with 10 bytes of room; the
         sequences of [sequences] above, with room for their literals and none
         for their match. *)
      cases "a code that breaks the format is Malformed though it passes dst"
        ~id:(fun (id, _, _, _, _) -> id)
        (let seq bits ~ll ~of_ =
           windowed 0
           ^ compressed (raw_literals "abcd" ^ one_sequence ~ll ~of_ ~ml:10 bits)
         in
         let far =
           (* A match of 13 at 1025 back, after 2000 bytes, into 2001. *)
           windowed 0 ^ rle ~last:false 'a' 1000 ^ rle ~last:false 'b' 1000
           ^ compressed
               (raw_literals "" ^ one_sequence ~ll:0 ~of_:10 ~ml:10 "\x04\x04")
         in
         [
           ( "a reserved Block_Type",
             windowed 0 ^ block_header ~last:true 3 100 ^ String.make 100 'a',
             10,
             Some "Block_Type",
             6 );
           ( "a Block_Size above Block_Maximum_Size",
             windowed 0 ^ raw (String.make 1025 'a'),
             10,
             Some "Block_Size",
             6 );
           ( "a raw block past Frame_Content_Size",
             sized 300 ^ raw (String.make 301 'a'),
             10,
             Some "Frame_Content_Size",
             6 );
           ("an offset of 0", seq "\x03" ~ll:0 ~of_:1, 6, None, 19);
           ( "an offset before the frame's start",
             seq "\x20" ~ll:4 ~of_:5,
             6,
             None,
             19 );
           ("an offset above Window_Size", far, 2001, Some "Window_Size", 24);
         ])
        (fun (_, s, cap, field, at) -> malformed ~cap ?field ~at s);
      test "a zip bomb is Too_long at the first block past the destination"
        (fun () ->
          (* 16 blocks of 128 KiB from 4 bytes each, 2 MiB from 70 bytes. 8
             blocks fill 1 MiB; the ninth's header is at 38. *)
          let block = rle ~last:false 'z' (128 * kib) in
          let s =
            windowed 7 ^ String.concat "" (List.init 16 (fun _ -> block))
          in
          let g = guarded mib in
          refused Too_long ~at:38
            (require_error
               ~pp:(Testable.pp Lz_support.progress)
               (Lz.Zstd.decode (Lz_support.of_string s) (snd g)));
          untouched g);
    ]

(* Windows *)

let window =
  let steps d s = Result.map fst (Lz_support.stepped d s) in
  let one_mib = lazy (data "window1m.zst", raw_of "window1m.zst") in
  group "window"
    [
      cases "a window below 1 KiB raises" ~id:string_of_int [ -1; 0; 1023 ]
        (fun window ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Lz.Zstd.decoder ~window ()));
      test "a window above the decoder's is Unsupported at its descriptor"
        (fun () ->
          let s, _ = Lazy.force one_mib in
          List.iter
            (fun window ->
              unsupported ~field:"Window_Size" ~at:5
                (require_error ~pp:pp_length
                   (steps (Lz.Zstd.decoder ~window ()) s)))
            [ 512 * kib; mib - 1 ]);
      cases "a window equal to the decoder's decodes" ~id:fst
        [
          ( "1 KiB",
            ( 1024,
              lazy
                (let s = String.make 1024 'w' in
                 (windowed 0 ^ raw s, s)) ) );
          ("1 MiB", (mib, one_mib));
        ]
        (fun (_, (window, frame)) ->
          let s, expected = Lazy.force frame in
          equal Lz_support.decoded (Ok expected)
            (steps (Lz.Zstd.decoder ~window ()) s));
      test "a single segment's window is its content size" (fun () ->
          let content = String.make 2000 'w' in
          let s = magic ^ "\x60" ^ le 2 (2000 - 256) ^ raw content in
          (* Frame_Content_Size, bytes 5 and 6, is the window. *)
          unsupported ~field:"Window_Size" ~at:5
            (require_error ~pp:pp_length
               (steps (Lz.Zstd.decoder ~window:1999 ()) s));
          equal Lz_support.decoded (Ok content)
            (steps (Lz.Zstd.decoder ~window:2000 ()) s));
      (* Window_Descriptor at 5: exponent 17, 2^27 bytes, then mantissa 1, 2^27
         + 2^24. Frame_Content_Size, at 6 and 7, of 300 bytes. *)
      test "the default decoder takes a window of 128 MiB and none above"
        (fun () ->
          let content = String.make 300 'w' in
          let frame wd =
            magic ^ "\x40" ^ byte wd ^ le 2 (300 - 256) ^ raw content
          in
          equal Lz_support.decoded ~msg:"128 MiB" (Ok content)
            (steps (Lz.Zstd.decoder ()) (frame 0x88));
          unsupported ~field:"Window_Size" ~at:5
            (require_error ~pp:pp_length ~msg:"144 MiB"
               (steps (Lz.Zstd.decoder ()) (frame 0x89))));
      test "the whole-buffer decode takes any window" (fun () ->
          (* Exponent 31: a window of 2 TiB. *)
          decodes (magic ^ "\x00\xF8" ^ raw "") "");
    ]

(* Paths *)

(* A golden, unchanged, with a byte replaced, or cut. *)
type change = Same | Replace of int * char | Cut_at of int

let goldens = lazy (Array.of_list (Lz_support.goldens_of "zstd"))

let change =
  Gen.(
    frequency
      [
        (1, constant Same);
        (2, map (fun (i, c) -> Replace (i, c)) (pair Lz_support.anywhere char));
        (1, map (fun i -> Cut_at i) Lz_support.anywhere);
      ])

let pp_change ppf (k, change) =
  let g = (Lazy.force goldens).(k) in
  let n = String.length (Lazy.force g.Lz_support.data) in
  Format.fprintf ppf "%s, %s" g.file
    (match change with
    | Same -> "unchanged"
    | Replace (i, c) ->
        Printf.sprintf "byte %d replaced by %C" (Lz_support.position i n) c
    | Cut_at i -> Printf.sprintf "cut at %d" (Lz_support.position i n))

let changed k change =
  let data = Lazy.force (Lazy.force goldens).(k).Lz_support.data in
  let n = String.length data in
  match change with
  | Replace (i, c) ->
      let b = Bytes.of_string data in
      Bytes.set b (Lz_support.position i n) c;
      Bytes.to_string b
  | Cut_at i -> String.sub data 0 (Lz_support.position i n)
  | Same -> data

let golden_index =
  Gen.int_range 0 (List.length (Lz_support.goldens_of "zstd") - 1)

(* Every build of the hot loops gives Lz.Zstd.decode's result and output, into a
   destination of the decoded size, where the literals share the output's last
   bytes, or with room past it. *)
let paths =
  let run decode data room =
    let dst = Lz_support.create room in
    match decode (Lz_support.of_string data) dst with
    | Ok { Lz.produced; _ } as r ->
        (r, Lz_support.to_string ~length:produced dst)
    | Error _ as r -> (r, "")
  in
  let outcome = pair decode_result string in
  let pp ppf (k, change, exact) =
    Format.fprintf ppf "%a, %s" pp_change (k, change)
      (if exact then "exact" else "roomy")
  in
  (* The first and last bytes of a frame of 100 bytes, replaced. *)
  let edges () =
    let gs = Lazy.force goldens in
    let k = ref 0 in
    Array.iteri
      (fun i g -> if g.Lz_support.file = "text_100.zst3" then k := i)
      gs;
    [ (!k, Replace (0, '\x07'), true); (!k, Replace (-1, '\x07'), true) ]
  in
  group "paths"
    [
      test "the generic path is first" (fun () ->
          equal (option string) (Some "generic")
            (match Lz.Paths.zstd_decode with
            | (label, _) :: _ -> Some label
            | [] -> None));
      prop "every path decodes as Lz.Zstd.decode" ~examples:(edges ())
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
          cover "a truncated input"
            (match fst expected with
            | Error { Lz.why = Truncated; _ } -> true
            | _ -> false);
          List.iter
            (fun (label, decode) ->
              equal ~msg:label outcome expected (run decode data room))
            Lz.Paths.zstd_decode);
      test "every path decodes matches across the window" (fun () ->
          let frame, expected = Lazy.force far_matches in
          List.iter
            (fun (label, decode) ->
              let dst = Lz_support.create (String.length expected) in
              equal decode_result ~msg:label
                (Ok
                   {
                     Lz.consumed = String.length frame;
                     produced = String.length expected;
                   })
                (decode (Lz_support.of_string frame) dst);
              equal Lz_support.decoded ~msg:label (Ok expected)
                (Ok (Lz_support.to_string dst)))
            Lz.Paths.zstd_decode);
    ]

let () =
  exit
    (run "Zstd"
       [
         group "decode" [ allocation ];
         group "decoder" [ full_runs; at_block_max; wrapping ];
         frames;
         header;
         blocks;
         literals;
         sequences;
         content;
         destinations;
         window;
         paths;
       ])
