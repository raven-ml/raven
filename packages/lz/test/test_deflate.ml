(* Deflate, zlib and gzip decoding. The hand-built streams follow RFC 1951
   (deflate), RFC 1950 (zlib) and RFC 1952 (gzip). A refused stream carries
   zlib's message (inflate.c) at the first byte of what is wrong: the byte
   holding the first bit of the Huffman code, code length item or bit field that
   breaks the format, the first byte of a byte-aligned field that does not hold
   (a stored length's complement, a header byte, a checksum, ISIZE), whichever
   of its bytes differ, and a gzip member's first byte when it does not begin
   with the magic. A code length set is wrong at the item that proves it so.
   Too_long is at the code that asks for the output. *)

open Windtrap
module Array1 = Bigarray.Array1

let kib = 1024
let mib = 1024 * kib

(* Decodes *)

(* [decode_into ~cap decode s] is [decode] of [s] into a destination of [cap]
   bytes (default: room for every output here): the bytes written and the bytes
   read, or the error. *)
let decode_into ?(cap = 64 * kib) decode s =
  let dst = Lz_support.create cap in
  match decode (Lz_support.of_string s) dst with
  | Ok { Lz.consumed; produced } ->
      Ok (Lz_support.to_string ~length:produced dst, consumed)
  | Error e -> Error e

let decodes ?cap decode s expected =
  match decode_into ?cap decode s with
  | Error e -> equal Lz_support.decoded (Ok expected) (Error e)
  | Ok (out, consumed) ->
      equal Lz_support.decoded (Ok expected) (Ok out);
      equal int ~msg:"consumed" (String.length s) consumed

let refuses ?cap decode s expected =
  equal Lz_support.decoded (Error expected)
    (Result.map fst (decode_into ?cap decode s))

let malformed at msg = { Lz.at; why = Malformed msg }

(* A destination of [n] bytes followed by [guard] bytes that a decode must not
   write. *)
let guard = 64

let guarded n =
  let whole = Lz_support.create (n + guard) in
  (whole, Array1.sub whole 0 n)

let untouched (whole, view) =
  let n = Array1.dim view in
  equal string ~msg:"the bytes past the destination" (String.make guard '\xa5')
    (String.init guard (fun i -> Char.chr (Array1.get whole (n + i))))

(* Bits *)

(* A bit writer: fields packed from each byte's least significant bit (RFC 1951,
   3.1.1), the first field lowest. *)
type writer = { buf : Buffer.t; mutable acc : int; mutable n : int }

let writer () = { buf = Buffer.create 64; acc = 0; n = 0 }

let put w (v, k) =
  w.acc <- w.acc lor ((v land ((1 lsl k) - 1)) lsl w.n);
  w.n <- w.n + k;
  while w.n >= 8 do
    Buffer.add_char w.buf (Char.chr (w.acc land 0xff));
    w.acc <- w.acc lsr 8;
    w.n <- w.n - 8
  done

(* The bytes written, the last one padded with zeros. *)
let contents w =
  if w.n > 0 then Buffer.contents w.buf ^ String.make 1 (Char.chr w.acc)
  else Buffer.contents w.buf

let pack fields =
  let w = writer () in
  List.iter (put w) fields;
  contents w

(* The byte holding the first bit after [fields]: where the element that follows
   them begins. *)
let after fields = List.fold_left (fun a (_, k) -> a + k) 0 fields / 8
let rec take n = function x :: l when n > 0 -> x :: take (n - 1) l | _ -> []

(* A Huffman code is packed from its most significant bit (RFC 1951, 3.1.1). *)
let huff (code, len) =
  let r = ref 0 in
  for i = 0 to len - 1 do
    if code land (1 lsl i) <> 0 then r := !r lor (1 lsl (len - 1 - i))
  done;
  (!r, len)

(* [canonical lens] is each symbol's code in the canonical code with these
   lengths (RFC 1951, 3.2.2), as [(code, length)]. *)
let canonical lens =
  let max = Array.fold_left max 0 lens in
  let count = Array.make (max + 1) 0 in
  Array.iter (fun l -> if l > 0 then count.(l) <- count.(l) + 1) lens;
  let next = Array.make (max + 1) 0 in
  let code = ref 0 in
  for bits = 1 to max do
    code := (!code + count.(bits - 1)) lsl 1;
    next.(bits) <- !code
  done;
  let codes = Array.make (Array.length lens) (0, 0) in
  for s = 0 to Array.length lens - 1 do
    let l = lens.(s) in
    if l > 0 then begin
      codes.(s) <- (next.(l), l);
      next.(l) <- next.(l) + 1
    end
  done;
  codes

(* A block's symbols: a literal/length symbol, a distance symbol, or raw bits
   (extra bits, or a code that the block's code leaves unused). *)
type sym = L of int | D of int | X of int * int

let symbols lit dist =
  List.map (function
    | L s -> huff lit.(s)
    | D s -> huff dist.(s)
    | X (v, n) -> (v, n))

let block_header ~final ty = [ ((if final then 1 else 0), 1); (ty, 2) ]

(* RFC 1951, 3.2.6. *)
let fixed_lit =
  canonical
    (Array.init 288 (fun s ->
         if s < 144 then 8 else if s < 256 then 9 else if s < 280 then 7 else 8))

let fixed_dist = canonical (Array.make 32 5)

let fixed ?(final = true) syms =
  block_header ~final 1 @ symbols fixed_lit fixed_dist syms

let fixed_syms syms = symbols fixed_lit fixed_dist syms

(* RFC 1951, 3.2.7: the order of the code length code's lengths. *)
let order =
  [| 16; 17; 18; 0; 8; 7; 9; 6; 10; 5; 11; 4; 12; 3; 13; 2; 14; 1; 15 |]

(* A dynamic block's header with [nlen] and [ndist] code lengths, the code
   length code [pre] (lengths by symbol), and the code length symbols [items]
   coded with it. *)
let dynamic_header ~nlen ~ndist ~pre items =
  let hclen = ref 4 in
  Array.iteri (fun i s -> if pre.(s) > 0 then hclen := max !hclen (i + 1)) order;
  block_header ~final:true 2
  @ [ (nlen - 257, 5); (ndist - 1, 5); (!hclen - 4, 4) ]
  @ List.init !hclen (fun i -> (pre.(order.(i)), 3))
  @ symbols (canonical pre) [||] items

(* A complete code length code over all 19 symbols. *)
let full_pre = Array.init 19 (fun s -> if s < 13 then 4 else 5)

(* The header of 257 literal/length and 1 distance code lengths written by
   [items] in [full_pre]'s code. *)
let lengths_header items = dynamic_header ~nlen:257 ~ndist:1 ~pre:full_pre items

(* [lengths ~single lens] codes code lengths in [full_pre]'s symbols, as groups
   [(first, syms)] that write the lengths from [first] on: each length as
   itself, runs of 3 to 138 zeros as 17 and 18, except that the lengths of
   [single] are each a group of one symbol. *)
let lengths ~single lens =
  let n = Array.length lens in
  let rec go i acc =
    if i >= n then List.rev acc
    else if lens.(i) <> 0 || List.mem i single then
      go (i + 1) ((i, [ L lens.(i) ]) :: acc)
    else begin
      let j = ref i in
      while
        !j < n && lens.(!j) = 0 && !j - i < 138 && not (List.mem !j single)
      do
        incr j
      done;
      let run = !j - i in
      if run >= 11 then go !j ((i, [ L 18; X (run - 11, 7) ]) :: acc)
      else if run >= 3 then go !j ((i, [ L 17; X (run - 3, 3) ]) :: acc)
      else go (i + 1) ((i, [ L 0 ]) :: acc)
    end
  in
  go 0 []

(* A dynamic block's header for these lengths, through the item that writes
   length [through] (default: all). The lengths a code's validity turns on,
   symbol 256's and each code's last, are items of their own, so that the item
   that proves a code wrong writes no other length: it begins where the header
   through the length before it ends. *)
let dynamic_codes ?through lit dist =
  let nlen = Array.length lit and ndist = Array.length dist in
  let groups =
    lengths ~single:[ 256; nlen - 1; nlen + ndist - 1 ] (Array.append lit dist)
  in
  let groups =
    match through with
    | None -> groups
    | Some i -> List.filter (fun (first, _) -> first <= i) groups
  in
  dynamic_header ~nlen ~ndist ~pre:full_pre (List.concat_map snd groups)

let dynamic lit dist syms =
  dynamic_codes lit dist @ symbols (canonical lit) (canonical dist) syms

(* Literal/length lengths: those of [set] given, the others 0, for at least the
   257 symbols a dynamic block lists. *)
let lit_lens set =
  let n = List.fold_left (fun n (s, _) -> max n (s + 1)) 257 set in
  let a = Array.make n 0 in
  List.iter (fun (s, l) -> a.(s) <- l) set;
  a

(* 'a' in 1 bit; end of block and length 3 in 2: a complete code. *)
let a_len3 = lit_lens [ (97, 1); (256, 2); (257, 2) ]

(* A stored block (RFC 1951, 3.2.4): LEN, NLEN, the bytes. *)
let stored ?(final = true) s =
  let n = String.length s and c = lnot (String.length s) in
  Printf.sprintf "%c%c%c%c%c%s"
    (if final then '\001' else '\000')
    (Char.chr (n land 0xff))
    (Char.chr (n lsr 8))
    (Char.chr (c land 0xff))
    (Char.chr ((c lsr 8) land 0xff))
    s

let literals s =
  List.map (fun c -> L (Char.code c)) (List.of_seq (String.to_seq s))

(* 'a', then a match of 3 from 2 back: too far back at [after too_far]. *)
let too_far = fixed [ L 97; L 257 ]
let too_far_stream = pack (too_far @ fixed_syms [ D 1; L 256 ])

(* "abcd" in a fixed block, which a destination of 3 bytes cannot hold: the 'd'
   is Too_long at [after abc]. *)
let abc = fixed [ L 97; L 98; L 99 ]
let abcd = pack (abc @ fixed_syms [ L 100; L 256 ])

(* RFC 1951, 3.2.5: the lengths that codes 257 to 285 begin at and their extra
   bits, a row of the table per count of extra bits. *)
let length_base =
  Array.concat
    [
      [| 3; 4; 5; 6; 7; 8; 9; 10 |];
      [| 11; 13; 15; 17 |];
      [| 19; 23; 27; 31 |];
      [| 35; 43; 51; 59 |];
      [| 67; 83; 99; 115 |];
      [| 131; 163; 195; 227 |];
      [| 258 |];
    ]

let length_extra =
  Array.concat
    [
      Array.make 8 0;
      Array.make 4 1;
      Array.make 4 2;
      Array.make 4 3;
      Array.make 4 4;
      Array.make 4 5;
      [| 0 |];
    ]

(* RFC 1951, 3.2.5: the distances that codes 0 to 29 begin at. Codes 0 to 3 have
   no extra bits, and from code 4 on each pair of codes has one more: rows of
   none, 1 to 4, 5 to 8, 9 to 11, and 12 and 13 extra bits. *)
let dist_base =
  Array.concat
    [
      [| 1; 2; 3; 4 |];
      [| 5; 7; 9; 13; 17; 25; 33; 49 |];
      [| 65; 97; 129; 193; 257; 385; 513; 769 |];
      [| 1025; 1537; 2049; 3073; 4097; 6145 |];
      [| 8193; 12289; 16385; 24577 |];
    ]

let dist_extra = Array.init 30 (fun i -> if i < 4 then 0 else (i / 2) - 1)

let coded base extra v =
  let i = ref (Array.length base - 1) in
  while base.(!i) > v do
    decr i
  done;
  (!i, X (v - base.(!i), extra.(!i)))

let match_syms (len, dist) =
  let l, lx = coded length_base length_extra len in
  let d, dx = coded dist_base dist_extra dist in
  [ L (257 + l); lx; D d; dx ]

(* 40,000 bytes in a stored block, then 70,000 bytes of matches at distances up
   to the window's 32 KiB: a stepped decode reads its window for the matches
   before its run. Where the output is within 258 bytes of a multiple of 32 KiB,
   a match of 258 at 32 KiB reads across the window's end. Then 190,000 bytes of
   matches at 32 KiB, of 258 and 100 bytes in turn, whose sources follow each
   other: wherever a stepped decode goes back to its buffer's start, a match's
   source spans the point, and the match reads the bytes before it and those
   after. *)
let window_stream =
  lazy
    (let start = Lz_support.corpus Random 40_000 in
     let lens = [| 3; 4; 10; 17; 31; 64; 130; 258; 257; 100 |] in
     let dists = [| 32768; 32767; 20000; 1; 7; 300; 16; 9; 31000 |] in
     let out = Buffer.create 300_000 in
     Buffer.add_string out start;
     let rec matches i acc =
       if Buffer.length out >= 300_000 then List.rev acc
       else begin
         let b = Buffer.length out in
         let len, dist =
           if b >= 110_000 then ((if i mod 2 = 0 then 258 else 100), 32768)
           else if 32768 - (b mod 32768) < 258 then (258, 32768)
           else
             (lens.(i mod Array.length lens), dists.(i mod Array.length dists))
         in
         for j = 0 to len - 1 do
           Buffer.add_char out (Buffer.nth out (b - dist + j))
         done;
         matches (i + 1) ((len, dist) :: acc)
       end
     in
     let m = matches 0 [] in
     ( stored ~final:false start
       ^ pack (fixed (List.concat_map match_syms m @ [ L 256 ])),
       Buffer.contents out ))

(* Errors *)

let pp_error =
  test "pp_error prints the kind, the offset and the message" (fun () ->
      let printed e = Format.asprintf "%a" Lz.pp_error e in
      (* lz.mli's example. *)
      equal string "malformed at byte 1024: invalid distance too far back"
        (printed (malformed 1024 "invalid distance too far back"));
      (* The other kinds, a baseline. *)
      expect
        (String.concat "\n"
           (List.map printed
              [
                { Lz.at = 7; why = Truncated };
                { Lz.at = 8; why = Too_long };
                { Lz.at = 1; why = Unsupported "preset dictionary" };
              ]))
      @@ __POS_OF__
           {|
        truncated at byte 7
        output too long at byte 8
        unsupported at byte 1: preset dictionary
        |})

(* Whole-buffer decodes *)

let goldens =
  cases "every golden decodes to its input, all of it consumed"
    ~id:(fun g -> g.Lz_support.file)
    (List.concat_map Lz_support.goldens_of [ "deflate"; "zlib"; "gzip" ])
    (fun g ->
      let raw = Lazy.force g.raw and data = Lazy.force g.data in
      let dst = Lz_support.create (String.length raw) in
      equal
        (result Lz_support.progress Lz_support.error)
        (Ok { Lz.consumed = String.length data; produced = String.length raw })
        (Lz_support.decoder g.codec (Lz_support.of_string data) dst);
      equal Lz_support.decoded (Ok raw) (Ok (Lz_support.to_string dst)))

(* The bytes [f] allocates on the OCaml heap, minor or major, with the bytes
   that counting them allocates. *)
let allocated f =
  let before = Gc.allocated_bytes () in
  let r = f () in
  let after = Gc.allocated_bytes () in
  (r, int_of_float (after -. before))

let allocation =
  cases "decode allocates only its result" ~id:Fun.id
    [
      "text_65536.zlib6";
      "columns_65536.gz9";
      "members.gz";
      "random_65536.ld6.deflate";
    ] (fun file ->
      let g =
        List.find (fun g -> g.Lz_support.file = file) (Lz_support.goldens ())
      in
      let decode = Lz_support.decoder g.codec in
      let src = Lz_support.of_string (Lazy.force g.data) in
      let dst = Lz_support.create (String.length (Lazy.force g.raw)) in
      let (), baseline = allocated (fun () -> ()) in
      let r, bytes = allocated (fun () -> decode src dst) in
      equal
        (result Lz_support.progress Lz_support.error)
        (Ok { Lz.consumed = Array1.dim src; produced = Array1.dim dst })
        r;
      let result_bytes =
        Obj.reachable_words (Obj.repr r) * (Sys.word_size / 8)
      in
      equal int ~msg:"bytes allocated" (baseline + result_bytes) bytes)

(* Decoders *)

(* Deflate and zlib goldens small enough to step a byte at a time. Each decodes
   to at most 200,000 bytes, less than a decoder's run capacity of 256 KiB, so a
   step takes its whole slice until the stream ends: [begins_at] counts steps
   from the cuts alone, and the cover of the stream's last byte relies on it. *)
let small_streams =
  List.filter
    (fun g -> g.Lz_support.size <= 64 * kib)
    (Lz_support.goldens_of "deflate" @ Lz_support.goldens_of "zlib")

let tail_case =
  let open Gen in
  let golden =
    of_list
      ~pp:(fun ppf g -> Format.pp_print_string ppf g.Lz_support.file)
      small_streams
  in
  let cut =
    frequency
      [ (2, constant 1); (1, int_range 2 64); (1, int_range 65 100_000) ]
  in
  triple golden
    (string_of ~size:(int_range 1 64) char)
    (list ~size:(int_range 1 3) cut)

(* Whether a step begins at byte [i] when step [k] takes the [k]th of [cuts]
   bytes, the list used in turn. *)
let begins_at i cuts =
  let cuts = Array.of_list cuts in
  let rec go p k =
    if p >= i then p = i else go (p + cuts.(k mod Array.length cuts)) (k + 1)
  in
  go 0 0

let exact_end =
  prop "a stream followed by other bytes is Done at its end whatever the cut"
    tail_case (fun (g, tail, cuts) ->
      let s = Lazy.force g.Lz_support.data in
      let n = String.length s in
      cover "deflate" (g.codec = "deflate");
      cover "zlib" (g.codec = "zlib");
      cover "a step begins at the stream's last byte" (begins_at (n - 1) cuts);
      let cuts = Array.of_list cuts in
      let d =
        (if g.codec = "zlib" then Lz.Zlib.decoder else Lz.Deflate.decoder) ()
      in
      let r =
        Lz_support.stepped d (s ^ tail) ~cut:(fun k ->
            cuts.(k mod Array.length cuts))
      in
      equal Lz_support.decoded (Ok (Lazy.force g.raw)) (Result.map fst r);
      equal
        (result int Lz_support.error)
        ~msg:"Done at" (Ok n) (Result.map snd r))

let window =
  cases "a step reads matches from its window whatever the cut"
    ~id:(fun l -> String.concat "," (List.map string_of_int l))
    [ [ 1 ]; [ 7; 300 ]; [ 4096 ]; [ 33; 1 ]; [ 1 lsl 20 ] ]
    (fun slices ->
      let data, raw = Lazy.force window_stream in
      let slices = Array.of_list slices in
      equal Lz_support.decoded (Ok raw)
        (Result.map fst
           (Lz_support.stepped (Lz.Deflate.decoder ()) data ~cut:(fun k ->
                slices.(k mod Array.length slices)))))

(* Deflate *)

let deflate = Lz.Deflate.decode

let accepted =
  [
    test "a stored block decodes to its bytes" (fun () ->
        decodes deflate (stored "hello") "hello");
    test "a fixed block decodes to its symbols" (fun () ->
        decodes deflate (pack (fixed [ L 97; L 257; D 0; L 256 ])) "aaaa");
    test "a dynamic block decodes to its symbols" (fun () ->
        decodes deflate
          (pack (dynamic a_len3 [| 1 |] [ L 97; L 257; D 0; L 256 ]))
          "aaaa");
    test "286 literal/length and 30 distance codes decode" (fun () ->
        (* HLIT and HDIST of 29, the largest RFC 1951 allows. *)
        let lit = lit_lens [ (97, 1); (256, 2); (285, 2) ] in
        let dist = Array.init 30 (fun i -> if i = 0 || i = 29 then 1 else 0) in
        decodes deflate
          (pack (dynamic lit dist [ L 97; L 285; D 0; L 256 ]))
          (String.make 259 'a'));
    test "a distance into a stored block's output decodes" (fun () ->
        decodes deflate
          (stored ~final:false "xy" ^ pack (fixed [ L 257; D 1; L 256 ]))
          "xyxyx");
    test "a distance to the stream's first byte decodes" (fun () ->
        (* Distance 5 after five bytes: code 4 and its extra bit 0. *)
        decodes deflate
          (pack (fixed (literals "abcde" @ [ L 257; D 4; X (0, 1); L 256 ])))
          "abcdeabc");
    test "matches up to 32 KiB back decode" (fun () ->
        let data, raw = Lazy.force window_stream in
        decodes ~cap:(String.length raw) deflate data raw);
    test "a literal/length code of one code of length 1 decodes" (fun () ->
        decodes deflate
          (pack (dynamic (lit_lens [ (256, 1) ]) [| 0 |] [ L 256 ]))
          "");
    test "a distance code whose lengths are all 0 decodes literals" (fun () ->
        decodes deflate (pack (dynamic a_len3 [| 0 |] [ L 97; L 256 ])) "a");
  ]

(* A table of inputs [decode] refuses, each with its error. *)
let refused decode name rows =
  cases name
    ~id:(fun (id, _, _) -> id)
    rows
    (fun (_, s, e) -> refuses decode s e)

let block_type =
  (* BTYPE is the header's second and third bits. Across a byte, it is bits 82
     to 84 after a block of nine literals, so its first bit is in byte 10. *)
  let first = block_header ~final:true 3 in
  let nine = fixed ~final:false (List.init 9 (fun _ -> L 97) @ [ L 256 ]) in
  refused deflate "block type 3 is Malformed at BTYPE"
    [
      ( "the first block",
        pack first,
        malformed (after (take 1 first)) "invalid block type" );
      ( "across a byte",
        pack (nine @ block_header ~final:true 3),
        malformed (after (nine @ [ (1, 1) ])) "invalid block type" );
    ]

let nlen =
  test "a stored NLEN that is not LEN's complement is Malformed at NLEN"
    (fun () ->
      refuses deflate "\001\005\000\000\000hello"
        (malformed 3 "invalid stored block lengths"))

let counts =
  (* HLIT begins at the header's fourth bit, HDIST after it. *)
  let row id (hlit, hdist) =
    let f = block_header ~final:true 2 @ [ (hlit, 5); (hdist, 5); (0, 4) ] in
    let before = take (if hlit > 29 then 2 else 3) f in
    (id, pack f, malformed (after before) "too many length or distance symbols")
  in
  refused deflate
    "HLIT above 286 or HDIST above 30 codes is Malformed at the field"
    [
      row "287 literal/length codes" (30, 0);
      row "288 literal/length codes" (31, 0);
      row "31 distance codes" (0, 30);
      row "32 distance codes" (0, 31);
    ]

let broken_codes =
  let code_lengths id pre ~proof =
    let f = dynamic_header ~nlen:257 ~ndist:1 ~pre [] in
    (id, pack f, malformed (after (take proof f)) "invalid code lengths set")
  in
  let codes id lit dist ~through msg =
    ( id,
      pack (dynamic_codes lit dist),
      malformed (after (dynamic_codes ~through lit dist)) msg )
  in
  let n = Array.length a_len3 in
  refused deflate
    "code lengths that break a code are Malformed at the length that proves it"
    [
      (* The third length 1, for symbol 18, over-subscribes it: after the
         header's 5 fields and the lengths of 16 and 17. *)
      code_lengths "an over-subscribed code length code"
        (Array.init 19 (fun s -> if s >= 16 then 1 else 0))
        ~proof:7;
      (* Incomplete from its last length, the fourth. *)
      code_lengths "a code length code of one symbol"
        (Array.init 19 (fun s -> if s = 18 then 1 else 0))
        ~proof:9;
      (* Symbol 256's length 1 is the third, over-subscribing it. *)
      codes "an over-subscribed literal/length code"
        (lit_lens [ (97, 1); (98, 1); (256, 1) ])
        [| 1 |] ~through:255 "invalid literal/lengths set";
      (* Proven by the last literal/length length, symbol 256's. *)
      codes "an incomplete literal/length code"
        (lit_lens [ (97, 2); (256, 2) ])
        [| 1 |] ~through:255 "invalid literal/lengths set";
      (* The third distance length over-subscribes the code. *)
      codes "an over-subscribed distance code" a_len3 [| 1; 1; 1 |]
        ~through:(n + 1) "invalid distances set";
      codes "an incomplete distance code" a_len3 [| 2; 2 |] ~through:n
        "invalid distances set";
    ]

let repeats =
  (* 257 literal/length lengths and 1 distance length; 18 writes 11 to 138
     zeros, 17 writes 3 to 10, 16 repeats the last length 3 to 6 times. *)
  let row id before repeat =
    ( id,
      pack (lengths_header (before @ repeat)),
      malformed (after (lengths_header before)) "invalid bit length repeat" )
  in
  let zeros_256 = [ L 18; X (127, 7); L 18; X (107, 7) ] in
  refused deflate
    "a repeat before any length or past the count is Malformed at it"
    [
      row "16 first" [] [ L 16; X (0, 2) ];
      row "16 where 1 length remains" (zeros_256 @ [ L 1 ]) [ L 16; X (0, 2) ];
      row "17 where 1 length remains" (zeros_256 @ [ L 1 ]) [ L 17; X (0, 3) ];
      row "18 where 120 lengths remain"
        [ L 18; X (127, 7) ]
        [ L 18; X (127, 7) ];
    ]

let no_end_of_block =
  test "no end-of-block code is Malformed at the length that leaves it none"
    (fun () ->
      let lit = lit_lens [ (97, 1); (98, 1) ] in
      refuses deflate
        (pack (dynamic_codes lit [| 1 |]))
        (malformed
           (after (dynamic_codes ~through:255 lit [| 1 |]))
           "invalid code -- missing end-of-block"))

let no_symbol =
  let unused_lit = lit_lens [ (256, 1) ] in
  let in_fixed id before code msg =
    let f = fixed before in
    (id, pack (f @ [ huff code ]), malformed (after f) msg)
  in
  let after_length id dist (v, k) =
    let f = dynamic a_len3 dist [ L 97; L 257 ] in
    ( id,
      pack (dynamic a_len3 dist [ L 97; L 257; X (v, k) ]),
      malformed (after f) "invalid distance code" )
  in
  refused deflate "a code that names no symbol is Malformed at the code"
    [
      ( "a one-symbol literal/length code's unused code",
        pack (dynamic unused_lit [| 0 |] [ X (1, 1) ]),
        malformed
          (after (dynamic unused_lit [| 0 |] []))
          "invalid literal/length code" );
      after_length "a one-symbol distance code's unused code" [| 1 |] (1, 1);
      after_length "a length with no distance code" [| 0 |] (0, 1);
      in_fixed "literal/length symbol 286" [ L 97 ] fixed_lit.(286)
        "invalid literal/length code";
      in_fixed "literal/length symbol 287" [ L 97 ] fixed_lit.(287)
        "invalid literal/length code";
      in_fixed "distance symbol 30" [ L 97; L 257 ] fixed_dist.(30)
        "invalid distance code";
      in_fixed "distance symbol 31" [ L 97; L 257 ] fixed_dist.(31)
        "invalid distance code";
    ]

let too_far_back =
  let row id before rest =
    let f = fixed before in
    ( id,
      pack (f @ fixed_syms rest),
      malformed (after f) "invalid distance too far back" )
  in
  refused deflate
    "a distance before the stream's first byte is Malformed at its code"
    [
      row "distance 1 at the start" [ L 257 ] [ D 0; L 256 ];
      ( "distance 2 after one byte",
        too_far_stream,
        malformed (after too_far) "invalid distance too far back" );
      (* Distance 6 after five bytes, wrong from its code's first bit. *)
      row "distance 6 after five bytes, by its extra bit"
        (literals "abcde" @ [ L 257 ])
        [ D 4; X (1, 1); L 256 ];
    ]

let empty_code_length_code =
  test "a code length code of all 0 reads every length as 0" (fun () ->
      let f = dynamic_header ~nlen:257 ~ndist:1 ~pre:(Array.make 19 0) [] in
      let zeros n = List.init n (fun _ -> (0, 1)) in
      (* Symbol 256's length is the 257th. *)
      refuses deflate
        (pack (f @ zeros 258))
        (malformed
           (after (f @ zeros 256))
           "invalid code -- missing end-of-block"))

(* A fixed block of a literal then [pairs] matches of 258 bytes at distance 1,
   each 13 bits: symbol 285 then distance symbol 0. *)
let bomb pairs =
  let w = writer () in
  List.iter (put w) (fixed [ L 97 ]);
  let m = huff fixed_lit.(285) and d = huff fixed_dist.(0) in
  for _ = 1 to pairs do
    put w m;
    put w d
  done;
  put w (huff fixed_lit.(256));
  contents w

let zip_bomb =
  test "a zip bomb is Too_long at the first match past dst's end" (fun () ->
      let cap = mib in
      (* The literal and [fit] matches leave less than a match of room. The
         matches after the error keep the input going past the point where the
         destination fills, as a stream that decodes to gigabytes would. *)
      let fit = (cap - 1) / 258 in
      let src = Lz_support.of_string (bomb (fit + 64)) in
      let ((_, dst) as g) = guarded cap in
      (* 3 header bits, the literal's 8 and 13 bits per match before it. *)
      equal
        (result Lz_support.progress Lz_support.error)
        (Error { Lz.at = (3 + 8 + (13 * fit)) / 8; why = Too_long })
        (Lz.Deflate.decode src dst);
      untouched g)

(* A destination of [cap] bytes, one short of the stream's output. *)
let past_dst =
  cases "output past dst's end is Too_long at the code or stored byte asking"
    ~id:(fun (id, _, _, _) -> id)
    [
      (let f = fixed [ L 97 ] in
       ("a match", 3, pack (f @ fixed_syms [ L 257; D 0; L 256 ]), after f));
      ("a literal", 3, abcd, after abc);
      ("a stored byte", 3, stored "abcd", 8);
      ("a stored byte into no room", 0, stored "a", 5);
    ]
    (fun (_, cap, s, at) ->
      let ((_, dst) as g) = guarded cap in
      equal
        (result Lz_support.progress Lz_support.error)
        (Error { Lz.at; why = Too_long })
        (Lz.Deflate.decode (Lz_support.of_string s) dst);
      untouched g)

let too_far_and_past =
  test "a distance too far back is Malformed even when its match passes dst"
    (fun () ->
      (* One byte of room for the match of 3 after the 'a'. *)
      refuses ~cap:2 deflate too_far_stream
        (malformed (after too_far) "invalid distance too far back"))

(* Zlib *)

let be32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * (3 - i))) land 0xff))
let le32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))
let adler s = Int32.to_int (Lz.Adler32.string s) land 0xffffffff
let crc s = Lz.Crc32.string s

(* CMF and FLG with FCHECK set (RFC 1950, 2.2). *)
let zlib_header cmf flg =
  let flg = flg land 0xe0 in
  let r = ((cmf * 256) + flg) mod 31 in
  let flg = if r = 0 then flg else flg + 31 - r in
  Printf.sprintf "%c%c" (Char.chr cmf) (Char.chr flg)

let zlib_member ?(header = zlib_header 0x78 0) body raw =
  header ^ body ^ be32 (adler raw)

let zlib =
  let zlib = Lz.Zlib.decode in
  let body = stored "hello" in
  group "Zlib"
    [
      test "a stream decodes, its Adler-32 consumed" (fun () ->
          decodes zlib (zlib_member body "hello") "hello");
      cases "every window size up to 32 KiB decodes" ~id:string_of_int
        [ 0; 1; 6; 7 ] (fun cinfo ->
          let header = zlib_header ((cinfo lsl 4) lor 8) 0 in
          decodes zlib (zlib_member ~header body "hello") "hello");
      refused zlib
        "a header that breaks the format is Malformed at its first byte"
        [
          ( "check bits that fail",
            "\x78\x9d" ^ body,
            malformed 0 "incorrect header check" );
          ( "method 7",
            zlib_member ~header:(zlib_header 0x77 0) body "hello",
            malformed 0 "unknown compression method" );
          ( "a 64 KiB window",
            zlib_member ~header:(zlib_header 0x88 0) body "hello",
            malformed 0 "invalid window size" );
        ];
      test "a preset dictionary is Unsupported at the header's second byte"
        (fun () ->
          refuses zlib
            (zlib_header 0x78 0x20 ^ "\000\000\000\001" ^ body)
            { Lz.at = 1; why = Unsupported "preset dictionary" });
      test "a wrong Adler-32 is Malformed at its first byte" (fun () ->
          refuses zlib (zlib_member body "hellp")
            (malformed (2 + String.length body) "incorrect data check"));
      test "an error in the deflate data comes before a wrong Adler-32"
        (fun () ->
          refuses zlib
            (zlib_header 0x78 0 ^ too_far_stream ^ be32 0)
            (malformed (2 + after too_far) "invalid distance too far back"));
      test "output past dst's end is Too_long at its code's offset in src"
        (fun () ->
          refuses ~cap:3 zlib (zlib_member abcd "abcd")
            { Lz.at = 2 + after abc; why = Too_long });
    ]

(* Gzip *)

let gzip_header ?(cm = 8) ?(flg = 0) () =
  Printf.sprintf "\x1f\x8b%c%c\000\000\000\000\000\003" (Char.chr cm)
    (Char.chr flg)

let gzip_member ?(header = gzip_header ()) ?(size = -1) body raw =
  let size = if size < 0 then String.length raw else size in
  header ^ body ^ le32 (crc raw) ^ le32 size

(* A header with every optional field (RFC 1952, 2.3.1): FEXTRA, FNAME, FCOMMENT
   and FHCRC, whose CRC16 is [hcrc] or the right one. *)
let full_header ?hcrc () =
  let h =
    gzip_header ~flg:0x1f () ^ "\006\000lz\002\000ab" ^ "name.txt\000"
    ^ "a comment\000"
  in
  let c = match hcrc with Some c -> c | None -> crc h land 0xffff in
  h ^ Printf.sprintf "%c%c" (Char.chr (c land 0xff)) (Char.chr (c lsr 8))

let gzip =
  let gzip = Lz.Gzip.decode in
  let body = stored "hello" in
  let member = gzip_member body "hello" in
  let n = String.length member in
  let rest = String.sub member 2 (n - 2) in
  let bad_magic = "incorrect header check" in
  group "Gzip"
    [
      test "a member decodes" (fun () -> decodes gzip member "hello");
      test "a header with every optional field decodes" (fun () ->
          decodes gzip
            (gzip_member ~header:(full_header ()) body "hello")
            "hello");
      test "members decode one after another" (fun () ->
          decodes gzip
            (member ^ gzip_member (stored "") "" ^ member)
            "hellohello");
      test "a gzip header inside a member is data" (fun () ->
          decodes gzip (gzip_member (stored member) member) member);
      refused gzip "a wrong ID1 or ID2 is Malformed at the member's first byte"
        [
          ("ID1 of the first member", "\x1e\x8b" ^ rest, malformed 0 bad_magic);
          ("ID2 of the first member", "\x1f\x8c" ^ rest, malformed 0 bad_magic);
          ( "ID1 of a later member",
            member ^ "\x1e\x8b" ^ rest,
            malformed n bad_magic );
          ( "ID2 of a later member",
            member ^ "\x1f\x8c" ^ rest,
            malformed n bad_magic );
        ];
      test "a method other than 8 is Malformed at CM" (fun () ->
          refuses gzip
            (gzip_member ~header:(gzip_header ~cm:7 ()) body "hello")
            (malformed 2 "unknown compression method"));
      refused gzip "a reserved flag bit is Malformed at FLG"
        [
          ( "0x20",
            gzip_member ~header:(gzip_header ~flg:0x20 ()) body "hello",
            malformed 3 "unknown header flags set" );
          ( "0x40",
            gzip_member ~header:(gzip_header ~flg:0x40 ()) body "hello",
            malformed 3 "unknown header flags set" );
          ( "0x80",
            gzip_member ~header:(gzip_header ~flg:0x80 ()) body "hello",
            malformed 3 "unknown header flags set" );
          (* The first error the decode meets, with nothing after it. *)
          ( "0x20 of a later member, the input ending at it",
            member ^ "\x1f\x8b\008\x20",
            malformed (n + 3) "unknown header flags set" );
        ];
      refused gzip "ID1, ID2 and CM are checked as each arrives"
        [
          ("ID1 of the first member", "\x1e", malformed 0 bad_magic);
          ("ID2 of the first member", "\x1f\x8c", malformed 0 bad_magic);
          ( "CM of the first member",
            "\x1f\x8b\007",
            malformed 2 "unknown compression method" );
          ("ID1 of a later member", member ^ "\x1e", malformed n bad_magic);
          ("ID2 of a later member", member ^ "\x1f\x8c", malformed n bad_magic);
          ( "CM of a later member",
            member ^ "\x1f\x8b\007",
            malformed (n + 2) "unknown compression method" );
        ];
      refused gzip
        "a cut inside a later member's header is Truncated at its length"
        [
          ("after ID1", member ^ "\x1f", { Lz.at = n + 1; why = Truncated });
          ( "after CM",
            member ^ "\x1f\x8b\008",
            { Lz.at = n + 3; why = Truncated } );
        ];
      test "a wrong header CRC is Malformed at its first byte" (fun () ->
          let header = full_header ~hcrc:0 () in
          refuses gzip
            (gzip_member ~header body "hello")
            (malformed (String.length header - 2) "header crc mismatch"));
      refused gzip "a wrong CRC-32 is Malformed at its first byte"
        [
          ( "the CRC-32 of other bytes",
            gzip_member body "hellp",
            malformed (n - 8) "incorrect data check" );
          ( "only its last byte wrong",
            String.mapi
              (fun i c ->
                if i = n - 5 then Char.chr (Char.code c lxor 1) else c)
              member,
            malformed (n - 8) "incorrect data check" );
        ];
      refused gzip "a wrong ISIZE is Malformed at its first byte"
        [
          ( "6 for 5 bytes, only its first byte wrong",
            gzip_member ~size:6 body "hello",
            malformed (n - 4) "incorrect length check" );
          ( "261 for 5 bytes, only its second byte wrong",
            gzip_member ~size:(5 + 256) body "hello",
            malformed (n - 4) "incorrect length check" );
        ];
      refused gzip
        "zero padding after the last member is Malformed at its first byte"
        [
          ("one zero byte", member ^ "\000", malformed n bad_magic);
          ( "512 zero bytes",
            member ^ String.make 512 '\000',
            malformed n bad_magic );
          ( "two zero bytes and ID1",
            member ^ "\000\000\x1f",
            malformed n bad_magic );
        ];
      test "the members before the padding decode given src up to the error"
        (fun () ->
          let src =
            Lz_support.of_string (member ^ member ^ String.make 512 '\000')
          in
          let at =
            (require_error (Lz.Gzip.decode src (Lz_support.create 64))).Lz.at
          in
          let dst = Lz_support.create 10 in
          equal
            (result Lz_support.progress Lz_support.error)
            (Ok { Lz.consumed = at; produced = 10 })
            (Lz.Gzip.decode (Array1.sub src 0 at) dst);
          equal string "hellohello" (Lz_support.to_string dst));
      test "a distance before a later member is Malformed at its code"
        (fun () ->
          let f = fixed [ L 257 ] in
          refuses gzip
            (member ^ gzip_header ()
            ^ pack (f @ fixed_syms [ D 0; L 256 ])
            ^ le32 0 ^ le32 0)
            (malformed (n + 10 + after f) "invalid distance too far back"));
      test "an error in the deflate data comes before a wrong CRC-32" (fun () ->
          refuses gzip
            (gzip_header () ^ too_far_stream ^ le32 0 ^ le32 1)
            (malformed (10 + after too_far) "invalid distance too far back"));
      test "output past dst's end is Too_long at its code's offset in src"
        (fun () ->
          refuses ~cap:3 gzip (gzip_member abcd "abcd")
            { Lz.at = 10 + after abc; why = Too_long });
    ]

let () =
  exit
    (run "Deflate"
       [
         pp_error;
         group "decode" [ goldens; allocation ];
         group "Decoder" [ exact_end; window ];
         group "Deflate"
           (accepted
           @ [
               block_type;
               nlen;
               counts;
               broken_codes;
               repeats;
               no_end_of_block;
               no_symbol;
               too_far_back;
               empty_code_length_code;
               zip_bomb;
               past_dst;
               too_far_and_past;
             ]);
         zlib;
         gzip;
       ])
