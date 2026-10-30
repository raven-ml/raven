(* Deflate, zlib and gzip. The hand-built streams follow RFC 1951 (deflate),
   RFC 1950 (zlib) and RFC 1952 (gzip). A refused stream carries zlib's
   message (inflate.c) at the first byte of what is wrong: the byte holding
   the first bit of the Huffman code, code length item or bit field that
   breaks the format, the first byte of a byte-aligned field that does not
   hold (a stored length's complement, a header byte, a checksum, ISIZE),
   whichever of its bytes differ, and a gzip member's first byte when it
   does not begin with the magic. A code length set is wrong at the item
   that proves it so. Too_long is at the code that asks for the output. *)

open Windtrap
module Array1 = Bigarray.Array1

let kib = 1024
let mib = 1024 * kib

(* Arguments *)

let src = Lz_support.of_string "abc"
let dst = Lz_support.create 64

let levels (name, encode, create, bound) =
  group name
    [
      cases ~name:string_of_int "a level outside 1 to 9 raises" [ 0; 10; -1 ]
        (fun level ->
          raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
              encode ~level src dst);
          raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
              create ~level ()));
      test "a negative size has no bound" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
              bound (-1)));
      Lz_support.largest_bound bound;
    ]

let arguments =
  group "arguments"
    (List.map levels
       [
         ( "Deflate",
           (fun ~level s d -> Lz.Deflate.encode ~level s d),
           (fun ~level () -> ignore (Lz.Deflate.encoder ~level ())),
           Lz.Deflate.bound );
         ( "Zlib",
           (fun ~level s d -> Lz.Zlib.encode ~level s d),
           (fun ~level () -> ignore (Lz.Zlib.encoder ~level ())),
           Lz.Zlib.bound );
         ( "Gzip",
           (fun ~level s d -> Lz.Gzip.encode ~level s d),
           (fun ~level () -> ignore (Lz.Gzip.encoder ~level ())),
           Lz.Gzip.bound );
       ])

(* Codecs *)

(* A codec's whole-buffer decode and its decoder. *)
let codec = function
  | "deflate" -> (Lz.Deflate.decode, Lz.Deflate.decoder)
  | "zlib" -> (Lz.Zlib.decode, Lz.Zlib.decoder)
  | "gzip" -> (Lz.Gzip.decode, Lz.Gzip.decoder)
  | c -> invalid_arg c

(* [decode c ?cap s] is the whole-buffer decode of [s] into a destination
   of [cap] bytes (default: room for any output here), with what it
   consumed. *)
let decode c ?(cap = 64 * kib) s =
  let decode, _ = codec c in
  let dst = Lz_support.create cap in
  match decode (Lz_support.of_string s) dst with
  | Ok { Lz.consumed; produced } ->
      (Ok (Lz_support.to_string ~length:produced dst), consumed)
  | Error e -> (Error e, 0)

let decodes c s expected =
  let r, consumed = decode c s in
  equal Lz_support.decoded (Ok expected) r;
  equal int ~msg:"consumed" (String.length s) consumed

let refuses c s expected = equal Lz_support.decoded (Error expected) (fst (decode c s))
let malformed at msg = { Lz.at; why = Malformed msg }

(* Bits *)

(* A bit writer: fields packed from each byte's least significant bit
   (RFC 1951, 3.1.1), the first field lowest. *)
type writer = { buf : Buffer.t; mutable acc : int; mutable n : int; mutable bits : int }

let writer () = { buf = Buffer.create 64; acc = 0; n = 0; bits = 0 }

let put w (v, k) =
  w.acc <- w.acc lor ((v land ((1 lsl k) - 1)) lsl w.n);
  w.n <- w.n + k;
  w.bits <- w.bits + k;
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

let width fields = List.fold_left (fun a (_, k) -> a + k) 0 fields

(* The byte holding the first bit after [fields]: where the element that
   follows them begins. *)
let after fields = width fields / 8

let rec take n = function x :: l when n > 0 -> x :: take (n - 1) l | _ -> []

(* A Huffman code is packed from its most significant bit (RFC 1951,
   3.1.1). *)
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

(* A block's symbols: a literal/length symbol, a distance symbol, or raw
   bits (extra bits, or a code that the block's code leaves unused). *)
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
let fixed ?(final = true) syms = block_header ~final 1 @ symbols fixed_lit fixed_dist syms

(* RFC 1951, 3.2.7: the order of the code length code's lengths. *)
let order = [| 16; 17; 18; 0; 8; 7; 9; 6; 10; 5; 11; 4; 12; 3; 13; 2; 14; 1; 15 |]

(* A dynamic block's header with [nlen] and [ndist] code lengths, the code
   length code [pre] (lengths by symbol), and the code length symbols
   [items] coded with it. *)
let dynamic_header ?(final = true) ~nlen ~ndist ~pre items =
  let hclen = ref 4 in
  Array.iteri (fun i s -> if pre.(s) > 0 then hclen := max !hclen (i + 1)) order;
  block_header ~final 2
  @ [ (nlen - 257, 5); (ndist - 1, 5); (!hclen - 4, 4) ]
  @ List.init !hclen (fun i -> (pre.(order.(i)), 3))
  @ symbols (canonical pre) [||] items

(* A complete code length code over all 19 symbols. *)
let full_pre = Array.init 19 (fun s -> if s < 13 then 4 else 5)

(* [lengths ~single lens] codes code lengths in [full_pre]'s symbols, as
   groups [(first, syms)] that write the lengths from [first] on: each
   length as itself, runs of 3 to 138 zeros as 17 and 18, except that the
   lengths of [single] are each a group of one symbol. *)
let lengths ~single lens =
  let n = Array.length lens in
  let rec go i acc =
    if i >= n then List.rev acc
    else if lens.(i) <> 0 || List.mem i single then go (i + 1) ((i, [ L lens.(i) ]) :: acc)
    else begin
      let j = ref i in
      while !j < n && lens.(!j) = 0 && !j - i < 138 && not (List.mem !j single) do
        incr j
      done;
      let run = !j - i in
      if run >= 11 then go !j ((i, [ L 18; X (run - 11, 7) ]) :: acc)
      else if run >= 3 then go !j ((i, [ L 17; X (run - 3, 3) ]) :: acc)
      else go (i + 1) ((i, [ L 0 ]) :: acc)
    end
  in
  go 0 []

(* A dynamic block's header for these lengths, through the item that
   writes length [through] (default: all). The lengths a code's validity
   turns on, symbol 256's and each code's last, are items of their own, so
   that the item that proves a code wrong writes no other length: it
   begins where the header through the length before it ends. *)
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

(* Literal/length lengths: those of [set] given, the others 0, for at
   least the 257 symbols a dynamic block lists. *)
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
    (Char.chr (n land 0xff)) (Char.chr (n lsr 8))
    (Char.chr (c land 0xff)) (Char.chr ((c lsr 8) land 0xff))
    s

(* Deflate *)

let blocks =
  let raw = decodes "deflate" and bad = refuses "deflate" in
  group "deflate"
    [
      test "a stored block" (fun () -> raw (stored "hello") "hello");
      test "a stored block whose NLEN is not LEN's complement" (fun () ->
          bad "\001\005\000\000\000hello" (malformed 3 "invalid stored block lengths"));
      test "block type 3" (fun () ->
          (* BTYPE is the header's second and third bits. *)
          let f = block_header ~final:true 3 in
          bad (pack f) (malformed (after (take 1 f)) "invalid block type");
          (* Here bits 82 to 84, so BTYPE's first bit is in byte 10. *)
          let f = fixed ~final:false (List.init 9 (fun _ -> L 97) @ [ L 256 ]) in
          bad (pack (f @ block_header ~final:true 3))
            (malformed (after (f @ [ (1, 1) ])) "invalid block type"));
      test "a fixed block" (fun () ->
          raw (pack (fixed [ L 97; L 257; D 0; L 256 ])) "aaaa");
      test "fixed literal/length symbols 286 and 287" (fun () ->
          List.iter
            (fun s ->
              let f = fixed [ L 97 ] in
              bad (pack (f @ [ huff fixed_lit.(s) ]))
                (malformed (after f) "invalid literal/length code"))
            [ 286; 287 ]);
      test "fixed distance symbols 30 and 31" (fun () ->
          List.iter
            (fun s ->
              let f = fixed [ L 97; L 257 ] in
              bad (pack (f @ [ huff fixed_dist.(s) ]))
                (malformed (after f) "invalid distance code"))
            [ 30; 31 ]);
      test "a distance before the start of the output" (fun () ->
          let f = fixed [ L 97; L 257 ] in
          bad (pack (f @ symbols fixed_lit fixed_dist [ D 1; L 256 ]))
            (malformed (after f) "invalid distance too far back");
          let f = fixed [ L 257 ] in
          bad (pack (f @ symbols fixed_lit fixed_dist [ D 0; L 256 ]))
            (malformed (after f) "invalid distance too far back"));
      test "a distance too far back by its extra bit" (fun () ->
          (* After five bytes, distance 5 is valid and 6 is not: the
             distance is wrong from its code's first bit. *)
          let lits = List.map (fun c -> L (Char.code c)) [ 'a'; 'b'; 'c'; 'd'; 'e' ] in
          raw (pack (fixed (lits @ [ L 257; D 4; X (0, 1); L 256 ]))) "abcdeabc";
          let f = fixed (lits @ [ L 257 ]) in
          bad (pack (f @ symbols fixed_lit fixed_dist [ D 4; X (1, 1); L 256 ]))
            (malformed (after f) "invalid distance too far back"));
      test "a distance before a stored block's output is valid" (fun () ->
          raw (stored ~final:false "xy" ^ pack (fixed [ L 257; D 1; L 256 ])) "xyxyx");
      test "a dynamic block" (fun () ->
          raw (pack (dynamic a_len3 [| 1 |] [ L 97; L 257; D 0; L 256 ])) "aaaa");
      test "more than 286 literal/length or 30 distance codes" (fun () ->
          (* At HLIT's first bit, or HDIST's after it. *)
          List.iter
            (fun (hlit, hdist) ->
              let f = block_header ~final:true 2 @ [ (hlit, 5); (hdist, 5); (0, 4) ] in
              let before = take (if hlit > 29 then 2 else 3) f in
              bad (pack f) (malformed (after before) "too many length or distance symbols"))
            [ (30, 0); (31, 0); (0, 30); (0, 31) ]);
      test "an over-subscribed code length code" (fun () ->
          let pre = Array.make 19 0 in
          pre.(16) <- 1;
          pre.(17) <- 1;
          pre.(18) <- 1;
          let f = dynamic_header ~nlen:257 ~ndist:1 ~pre [] in
          (* The third length 1, for symbol 18, over-subscribes it: the
             header's 5 fields, then the lengths of 16 and 17 before it. *)
          bad (pack f) (malformed (after (take 7 f)) "invalid code lengths set"));
      test "a code length code of one symbol" (fun () ->
          (* Incomplete from its last length, the fourth. *)
          let pre = Array.make 19 0 in
          pre.(18) <- 1;
          let f = dynamic_header ~nlen:257 ~ndist:1 ~pre [] in
          bad (pack f) (malformed (after (take 9 f)) "invalid code lengths set"));
      test "an empty code length code reads every length as 0" (fun () ->
          let f = dynamic_header ~nlen:257 ~ndist:1 ~pre:(Array.make 19 0) [] in
          let zeros n = List.init n (fun _ -> (0, 1)) in
          (* Symbol 256's length is the 257th. *)
          bad (pack (f @ zeros 258))
            (malformed (after (f @ zeros 256)) "invalid code -- missing end-of-block"));
      test "a repeat with no length before it" (fun () ->
          let f = dynamic_header ~nlen:257 ~ndist:1 ~pre:full_pre [] in
          let g = dynamic_header ~nlen:257 ~ndist:1 ~pre:full_pre [ L 16 ] in
          bad (pack (g @ [ (0, 2) ])) (malformed (after f) "invalid bit length repeat"));
      test "a repeat past the last code length" (fun () ->
          (* 138 zeros, then 138 more where 120 lengths remain. *)
          let f = dynamic_header ~nlen:257 ~ndist:1 ~pre:full_pre [ L 18; X (127, 7) ] in
          let g =
            dynamic_header ~nlen:257 ~ndist:1 ~pre:full_pre
              [ L 18; X (127, 7); L 18; X (127, 7) ]
          in
          bad (pack g) (malformed (after f) "invalid bit length repeat"));
      test "no end-of-block code" (fun () ->
          let lit = lit_lens [ (97, 1); (98, 1) ] in
          bad (pack (dynamic_codes lit [| 1 |]))
            (malformed (after (dynamic_codes ~through:255 lit [| 1 |]))
               "invalid code -- missing end-of-block"));
      test "an over-subscribed literal/length code" (fun () ->
          (* Symbol 256's length 1 is the third, over-subscribing it. *)
          let lit = lit_lens [ (97, 1); (98, 1); (256, 1) ] in
          bad (pack (dynamic_codes lit [| 1 |]))
            (malformed (after (dynamic_codes ~through:255 lit [| 1 |]))
               "invalid literal/lengths set"));
      test "an incomplete literal/length code" (fun () ->
          (* Proven by the last literal/length length, symbol 256's. *)
          let lit = lit_lens [ (97, 2); (256, 2) ] in
          bad (pack (dynamic_codes lit [| 1 |]))
            (malformed (after (dynamic_codes ~through:255 lit [| 1 |]))
               "invalid literal/lengths set"));
      test "a literal/length code of one symbol" (fun () ->
          let lit = lit_lens [ (256, 1) ] in
          raw (pack (dynamic lit [| 0 |] [ L 256 ])) "";
          bad (pack (dynamic lit [| 0 |] [ X (1, 1) ]))
            (malformed (after (dynamic lit [| 0 |] [])) "invalid literal/length code"));
      test "an over-subscribed distance code" (fun () ->
          (* The third distance length over-subscribes the code. *)
          let n = Array.length a_len3 in
          bad (pack (dynamic_codes a_len3 [| 1; 1; 1 |]))
            (malformed (after (dynamic_codes ~through:(n + 1) a_len3 [| 1; 1; 1 |]))
               "invalid distances set"));
      test "an incomplete distance code" (fun () ->
          let n = Array.length a_len3 in
          bad (pack (dynamic_codes a_len3 [| 2; 2 |]))
            (malformed (after (dynamic_codes ~through:n a_len3 [| 2; 2 |]))
               "invalid distances set"));
      test "a distance code of one symbol" (fun () ->
          let f = dynamic a_len3 [| 1 |] [ L 97; L 257 ] in
          bad (pack (dynamic a_len3 [| 1 |] [ L 97; L 257; X (1, 1) ]))
            (malformed (after f) "invalid distance code"));
      test "a length with no distance code" (fun () ->
          raw (pack (dynamic a_len3 [| 0 |] [ L 97; L 256 ])) "a";
          let f = dynamic a_len3 [| 0 |] [ L 97; L 257 ] in
          bad (pack (dynamic a_len3 [| 0 |] [ L 97; L 257; X (0, 1) ]))
            (malformed (after f) "invalid distance code"));
    ]

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

let zlib_member ?(header = zlib_header 0x78 0) body raw = header ^ body ^ be32 (adler raw)

let zlib =
  let ok = decodes "zlib" and bad = refuses "zlib" in
  let body = stored "hello" in
  group "zlib"
    [
      test "a stream" (fun () -> ok (zlib_member body "hello") "hello");
      cases ~name:string_of_int "every window size up to 32 KiB" [ 0; 1; 6; 7 ]
        (fun cinfo ->
          ok (zlib_member ~header:(zlib_header ((cinfo lsl 4) lor 8) 0) body "hello")
            "hello");
      test "FCHECK" (fun () ->
          bad ("\x78\x9d" ^ body) (malformed 0 "incorrect header check"));
      test "a method other than 8" (fun () ->
          bad (zlib_member ~header:(zlib_header 0x77 0) body "hello")
            (malformed 0 "unknown compression method"));
      test "a window above 32 KiB" (fun () ->
          bad (zlib_member ~header:(zlib_header 0x88 0) body "hello")
            (malformed 0 "invalid window size"));
      test "a preset dictionary is Unsupported" (fun () ->
          bad (zlib_header 0x78 0x20 ^ "\000\000\000\001" ^ body)
            { Lz.at = 1; why = Unsupported "preset dictionary" });
      test "a wrong Adler-32" (fun () ->
          bad (zlib_member body "hellp")
            (malformed (2 + String.length body) "incorrect data check"));
    ]

(* Gzip *)

let gzip_header ?(cm = 8) ?(flg = 0) () =
  Printf.sprintf "\x1f\x8b%c%c\000\000\000\000\000\003" (Char.chr cm) (Char.chr flg)

let gzip_member ?(header = gzip_header ()) ?(size = -1) body raw =
  let size = if size < 0 then String.length raw else size in
  header ^ body ^ le32 (crc raw) ^ le32 size

(* A header with every optional field (RFC 1952, 2.3.1): FEXTRA, FNAME,
   FCOMMENT and FHCRC, whose CRC16 is [hcrc] or the right one. *)
let full_header ?hcrc () =
  let h =
    gzip_header ~flg:0x1f () ^ "\006\000lz\002\000ab" ^ "name.txt\000" ^ "a comment\000"
  in
  let c = match hcrc with Some c -> c | None -> crc h land 0xffff in
  h ^ Printf.sprintf "%c%c" (Char.chr (c land 0xff)) (Char.chr (c lsr 8))

let gzip =
  let ok = decodes "gzip" and bad = refuses "gzip" in
  let body = stored "hello" in
  let member = gzip_member body "hello" in
  let n = String.length member in
  group "gzip"
    [
      test "a member" (fun () -> ok member "hello");
      test "every optional header field" (fun () ->
          ok (gzip_member ~header:(full_header ()) body "hello") "hello");
      test "a wrong header CRC16" (fun () ->
          let header = full_header ~hcrc:0 () in
          bad (gzip_member ~header body "hello")
            (malformed (String.length header - 2) "header crc mismatch"));
      test "a wrong magic is at the member's first byte" (fun () ->
          bad ("\x1f\x8c" ^ String.sub member 2 (n - 2))
            (malformed 0 "incorrect header check");
          bad ("\x1e\x8b" ^ String.sub member 2 (n - 2))
            (malformed 0 "incorrect header check"));
      test "a method other than 8" (fun () ->
          bad (gzip_member ~header:(gzip_header ~cm:7 ()) body "hello")
            (malformed 2 "unknown compression method"));
      cases ~name:string_of_int "a reserved flag" [ 0x20; 0x40; 0x80 ] (fun flg ->
          bad (gzip_member ~header:(gzip_header ~flg ()) body "hello")
            (malformed 3 "unknown header flags set"));
      test "a wrong CRC-32" (fun () ->
          bad (gzip_member body "hellp") (malformed (n - 8) "incorrect data check");
          (* Only the CRC's last byte wrong: still its first byte. *)
          let m = Bytes.of_string member in
          Bytes.set m (n - 5) (Char.chr (Char.code member.[n - 5] lxor 1));
          bad (Bytes.to_string m) (malformed (n - 8) "incorrect data check"));
      test "a wrong ISIZE" (fun () ->
          (* Only its second byte wrong: still its first byte. *)
          bad (gzip_member ~size:6 body "hello") (malformed (n - 4) "incorrect length check");
          bad (gzip_member ~size:(5 + 256) body "hello")
            (malformed (n - 4) "incorrect length check"));
      test "members follow each other" (fun () ->
          ok (member ^ gzip_member (stored "") "" ^ member) "hellohello");
      test "a gzip header inside a member is data" (fun () ->
          ok (gzip_member (stored member) member) member);
      test "a byte that begins no member is Malformed at its offset" (fun () ->
          bad (member ^ "\000\000\x1f") (malformed n "incorrect header check");
          bad (member ^ "\000") (malformed n "incorrect header check");
          bad (member ^ "\x1f\000") (malformed n "incorrect header check"));
      test "zero padding after the last member is Malformed at its first byte"
        (fun () ->
          let padded = member ^ String.make 512 '\000' in
          let e = malformed n "incorrect header check" in
          bad padded e;
          equal
            (result (pair string int) Lz_support.error)
            ~msg:"stepped" (Error e)
            (Lz_support.stepped (Lz.Gzip.decoder ()) padded));
      test "a later member's header is checked byte by byte" (fun () ->
          bad (member ^ "\x1f\x8b\007") (malformed (n + 2) "unknown compression method");
          bad (member ^ "\x1f\x8b\008\x20") (malformed (n + 3) "unknown header flags set");
          bad (member ^ "\x1f\x8b\008") { Lz.at = n + 3; why = Truncated });
      test "a distance before a later member's output" (fun () ->
          let f = pack (fixed [ L 257 ] @ symbols fixed_lit fixed_dist [ D 0; L 256 ]) in
          bad (member ^ gzip_header () ^ f ^ le32 0 ^ le32 0)
            (malformed (n + 10 + after (fixed [ L 257 ])) "invalid distance too far back"));
      test "a cut inside the next member's magic is Truncated" (fun () ->
          bad (member ^ "\x1f") { Lz.at = n + 1; why = Truncated });
    ]

let empty =
  cases ~name:Fun.id "an empty input is Truncated at 0" [ "deflate"; "zlib"; "gzip" ]
    (fun c -> refuses c "" { Lz.at = 0; why = Truncated })

(* Bounds *)

(* A destination of [n] bytes followed by [guard] bytes that a decode must
   not write. *)
let guard = 64

let guarded n =
  let whole = Lz_support.create (n + guard) in
  (whole, Array1.sub whole 0 n)

let untouched (whole, view) =
  let n = Array1.dim view in
  equal string ~msg:"the bytes past the destination" (String.make guard '\xa5')
    (String.init guard (fun i -> Char.chr (Array1.get whole (n + i))))

(* A fixed block of a literal then [pairs] matches of 258 bytes at
   distance 1, each 13 bits: symbol 285 then distance symbol 0. *)
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

let bounds =
  group "bounds"
    [
      test "a zip bomb ends in Too_long at the destination's end" (fun () ->
          let pairs = 10 * mib * 8 / 13 in
          let src = Lz_support.of_string (bomb pairs) in
          let cap = mib in
          let ((_, dst) as g) = guarded cap in
          (* At the first match that does not fit: 3 header bits, the
             literal's 8 and 13 bits per match before it. *)
          let fit = (cap - 1) / 258 in
          equal
            (result Lz_support.progress Lz_support.error)
            (Error { Lz.at = (3 + 8 + (13 * fit)) / 8; why = Too_long })
            (Lz.Deflate.decode src dst);
          untouched g);
      test "an output one byte too long" (fun () ->
          (* At the code asking for the byte: a match, a literal, a stored
             byte. *)
          let too_long ?(cap = 3) s at =
            let ((_, dst) as g) = guarded cap in
            equal
              (result Lz_support.progress Lz_support.error)
              (Error { Lz.at; why = Too_long })
              (Lz.Deflate.decode (Lz_support.of_string s) dst);
            untouched g
          in
          let f = fixed [ L 97 ] in
          too_long (pack (f @ symbols fixed_lit fixed_dist [ L 257; D 0; L 256 ])) (after f);
          let f = fixed [ L 97; L 98; L 99 ] in
          too_long (pack (f @ symbols fixed_lit fixed_dist [ L 100; L 256 ])) (after f);
          too_long (stored "abcd") 8;
          too_long ~cap:0 (stored "a") 5);
    ]

(* Goldens *)

let goldens =
  group "goldens"
    [
      group "deflate" [ Lz_support.agreement "deflate" ];
      group "zlib" [ Lz_support.agreement "zlib" ];
      group "gzip" [ Lz_support.agreement "gzip" ];
    ]

(* Whole equals steps *)

(* The stepped decode of [data] by a reader that passes its input in
   slices of [slices] bytes, the list used in turn. *)
let stepped c data ~slices =
  let _, decoder = codec c in
  let slices = Array.of_list slices in
  Lz_support.stepped (decoder ()) data ~cut:(fun k ->
      slices.(k mod Array.length slices))

type change = Same | Flip of int * int | Cut of int

type case = {
  golden : Lz_support.golden;
  change : change;
  slices : int list;
}

let pp_case ppf { golden; change; slices } =
  let ints = Format.(pp_print_list ~pp_sep:(fun p () -> pp_print_string p "; ") pp_print_int) in
  Format.fprintf ppf "{ golden = %s; change = %s; slices = [%a] }"
    golden.Lz_support.file
    (match change with
    | Same -> "Same"
    | Flip (i, b) -> Printf.sprintf "Flip (%d, %d)" i b
    | Cut i -> Printf.sprintf "Cut %d" i)
    ints slices

(* Every golden of 70 KiB or less of output: members.gz among them, whose
   second member has output. *)
let small_goldens =
  List.filter
    (fun g -> String.length (Lazy.force g.Lz_support.raw) <= 70 * kib)
    (Lz_support.goldens_of "deflate" @ Lz_support.goldens_of "zlib"
   @ Lz_support.goldens_of "gzip")

let several = List.filter (fun g -> g.Lz_support.members > 1) small_goldens

let case =
  let open Gen in
  let size =
    frequency
      [
        (1, constant 1);
        (1, int_range 2 64);
        (1, int_range 65 4096);
        (1, int_range 4097 100_000);
      ]
  in
  let sizes = list ~size:(int_range 1 3) size in
  let change =
    frequency
      [
        (2, constant Same);
        (2, map (fun (i, b) -> Flip (i, b)) (pair nat (int_range 0 7)));
        (1, map (fun i -> Cut i) nat);
      ]
  in
  let golden = frequency [ (5, of_list small_goldens); (1, of_list several) ] in
  with_pp pp_case
    (map
       (fun (golden, change, slices) -> { golden; change; slices })
       (triple golden change sizes))

let apply change data =
  let n = String.length data in
  match change with
  | Same -> data
  | _ when n = 0 -> data
  | Flip (i, b) ->
      let i = i mod n in
      String.mapi (fun j c -> if j = i then Char.chr (Char.code c lxor (1 lsl b)) else c) data
  | Cut i -> String.sub data 0 (i mod n)

let steps =
  prop "a stepped decode equals the whole-buffer decode" case
    (fun { golden; change; slices } ->
      let c = golden.Lz_support.codec in
      let data = apply change (Lazy.force golden.Lz_support.data) in
      let raw = Lazy.force golden.Lz_support.raw in
      let whole, consumed = decode c ~cap:(String.length raw + (64 * kib)) data in
      assume (match whole with Error { why = Too_long; _ } -> false | _ -> true);
      cover "a one-byte slice" (List.mem 1 slices);
      cover "several members" (golden.Lz_support.members > 1 && change = Same);
      cover "a decode"
        (match whole with Ok s -> String.length s > 0 | Error _ -> false);
      cover "a Truncated input"
        (match whole with Error { why = Truncated; _ } -> true | _ -> false);
      cover "a Malformed input"
        (match whole with Error { why = Malformed _; _ } -> true | _ -> false);
      match (whole, stepped c data ~slices) with
      | Ok s, Ok (s', read) ->
          equal Lz_support.decoded (Ok s) (Ok s');
          equal int ~msg:"read" consumed read
      | w, Ok (s', _) -> equal Lz_support.decoded w (Ok s')
      | w, Error e -> equal Lz_support.decoded w (Error e))

(* Exact ends *)

type tail_case = { stream : Lz_support.golden; tail : string; cuts : int list }

let pp_tail_case ppf { stream; tail; cuts } =
  let ints = Format.(pp_print_list ~pp_sep:(fun p () -> pp_print_string p "; ") pp_print_int) in
  Format.fprintf ppf "{ stream = %s; tail = %S; cuts = [%a] }" stream.Lz_support.file tail
    ints cuts

let tail_case =
  let open Gen in
  let ends = List.filter (fun g -> g.Lz_support.codec <> "gzip") small_goldens in
  let size =
    frequency [ (2, constant 1); (1, int_range 2 64); (1, int_range 65 100_000) ]
  in
  with_pp pp_tail_case
    (map
       (fun (stream, tail, cuts) -> { stream; tail; cuts })
       (triple (of_list ends)
          (string_of ~size:(int_range 1 64) char)
          (list ~size:(int_range 1 3) size)))

(* [cut_steps c data cuts] steps a decoder of [c] over [data] in slices of
   [cuts] bytes, the list used in turn, with [~last:true] on the slice
   that reaches [data]'s end. It is the runs and the offset of [Done], or
   the error, with the steps that took less than their slice and lent an
   empty run, and the offsets where steps began. *)
let cut_steps c data cuts =
  let _, decoder = codec c in
  let d = decoder () in
  let src = Lz_support.of_string data and n = String.length data in
  let cuts = Array.of_list cuts in
  let out = Buffer.create n in
  let rec go pos k idle short starts =
    if idle > 1 then fail "two steps in a row made no progress";
    let left = n - pos in
    let len = min cuts.(k mod Array.length cuts) left in
    let run = ref 0 in
    let lend r =
      run := Array1.dim r;
      Buffer.add_string out (Lz_support.to_string r)
    in
    let starts = pos :: starts in
    match Lz.Decoder.step d ~last:(len = left) (Array1.sub src pos len) lend with
    | Error e -> (Error e, short, starts)
    | Ok (Lz.Done j) -> (Ok (Buffer.contents out, pos + j), short, starts)
    | Ok (More j) ->
        let short = if j < len && !run = 0 then (pos, len, j) :: short else short in
        let idle = if j = 0 && !run = 0 then idle + 1 else 0 in
        go (pos + j) (k + 1) idle short starts
  in
  go 0 0 0 [] []

let exact_end =
  prop "a stream followed by other bytes is Done at its end whatever the cut"
    tail_case (fun { stream; tail; cuts } ->
      let c = stream.Lz_support.codec in
      let s = Lazy.force stream.Lz_support.data in
      let n = String.length s in
      let r, short, starts = cut_steps c (s ^ tail) cuts in
      cover "deflate" (c = "deflate");
      cover "zlib" (c = "zlib");
      cover "a step begins in the stream's last 8 bytes"
        (List.exists (fun p -> p > 0 && p >= n - 8 && p < n) starts);
      cover "a step begins at the stream's last byte"
        (List.exists (fun p -> p > 0 && p = n - 1) starts);
      equal Lz_support.decoded (Ok (Lazy.force stream.Lz_support.raw))
        (Result.map fst r);
      equal (result int Lz_support.error) ~msg:"Done at" (Ok n) (Result.map snd r);
      equal (list (triple int int int)) ~msg:"steps short of their slice with no run"
        [] short)

(* The streaming window *)

(* RFC 1951, 3.2.5: the symbol and extra bits that code a length or a
   distance. *)
let length_base =
  [| 3; 4; 5; 6; 7; 8; 9; 10; 11; 13; 15; 17; 19; 23; 27; 31; 35; 43; 51; 59; 67; 83;
     99; 115; 131; 163; 195; 227; 258 |]

let length_extra =
  [| 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 2; 2; 2; 2; 3; 3; 3; 3; 4; 4; 4; 4; 5; 5; 5; 5; 0 |]

let dist_base =
  [| 1; 2; 3; 4; 5; 7; 9; 13; 17; 25; 33; 49; 65; 97; 129; 193; 257; 385; 513; 769; 1025;
     1537; 2049; 3073; 4097; 6145; 8193; 12289; 16385; 24577 |]

let dist_extra = Array.init 30 (fun i -> if i < 4 then 0 else (i / 2) - 1)

let coded base extra v =
  let i = ref (Array.length base - 1) in
  while base.(!i) > v do decr i done;
  (!i, X (v - base.(!i), extra.(!i)))

let match_syms (len, dist) =
  let l, lx = coded length_base length_extra len in
  let d, dx = coded dist_base dist_extra dist in
  [ L (257 + l); lx; D d; dx ]

(* 40,000 bytes in a stored block, then 70,000 bytes of matches at
   distances up to the window's 32 KiB: a stepped decode reads its window
   for the matches before its run. Where the output is within 258 bytes
   of a multiple of 32 KiB, a match of 258 at 32 KiB reads across the
   window's end. Then 190,000 bytes of matches at 32 KiB, of 258 and 100
   bytes in turn, whose sources follow each other: wherever a stepped
   decode goes back to its buffer's start, a match's source spans the
   point, and the match reads the bytes before it and those after. *)
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
           else (lens.(i mod Array.length lens), dists.(i mod Array.length dists))
         in
         for j = 0 to len - 1 do
           Buffer.add_char out (Buffer.nth out (b - dist + j))
         done;
         matches (i + 1) ((len, dist) :: acc)
       end
     in
     let m = matches 0 [] in
     ( stored ~final:false start ^ pack (fixed (List.concat_map match_syms m @ [ L 256 ])),
       Buffer.contents out ))

let window =
  cases
    ~name:(fun l -> String.concat "," (List.map string_of_int l))
    "a stepped decode reads matches from its window"
    [ [ 1 ]; [ 7; 300 ]; [ 4096 ]; [ 33; 1 ]; [ 1 lsl 20 ] ]
    (fun slices ->
      let data, raw = Lazy.force window_stream in
      equal Lz_support.decoded (Ok raw) (fst (decode "deflate" ~cap:(String.length raw) data));
      equal Lz_support.decoded (Ok raw) (Result.map fst (stepped "deflate" data ~slices)))

(* Memory and domains *)

let allocated f =
  let before = Gc.minor_words () in
  let r = f () in
  let after = Gc.minor_words () in
  (r, int_of_float (after -. before))

let allocation =
  cases ~name:(fun g -> g.Lz_support.file)
    "a whole-buffer decode allocates only its result"
    (List.filter_map
       (fun f -> List.find_opt (fun g -> g.Lz_support.file = f) (Lz_support.goldens ()))
       [ "text_65536.zlib6"; "columns_65536.gz9"; "members.gz"; "random_65536.ld6.deflate" ])
    (fun g ->
      let decode, _ = codec g.Lz_support.codec in
      let src = Lz_support.of_string (Lazy.force g.Lz_support.data) in
      let dst = Lz_support.create (String.length (Lazy.force g.Lz_support.raw)) in
      let (), baseline = allocated (fun () -> ()) in
      let r, words = allocated (fun () -> decode src dst) in
      is_ok r;
      equal int ~msg:"words allocated" (baseline + Obj.reachable_words (Obj.repr r)) words)

(* [literals n] is a deflate stream of [n] copies of a fixed block of
   64 KiB of literals, then an empty final block, with what it decodes
   to. Each block takes a whole number of bytes: 3 header bits, 8 bits per
   literal below 144, 9 per literal above and 7 for the end of block, with
   six literals 0xff to make the total a multiple of 8. *)
let literals n =
  let data = Lz_support.corpus Text (64 * kib) ^ String.make 6 '\xff' in
  let w = writer () in
  List.iter (put w) (fixed ~final:false (List.init (String.length data) (fun i -> L (Char.code data.[i]))));
  put w (huff fixed_lit.(256));
  assert (w.n = 0);
  let unit = contents w in
  let last = pack (fixed [ L 256 ]) in
  (String.concat "" (List.init n (fun _ -> unit)) ^ last, n * String.length data)

(* A long decode, of [k] times 64 MiB, leaves this domain's minor
   collections running. *)
let domains =
  let small = Lz_support.of_string (gzip_member (stored "hello") "hello") in
  let small_dst = Lz_support.create 5 in
  let long k =
    let big, n = literals (1024 * k) in
    let src = Lz_support.of_string big and dst = Lz_support.create n in
    fun () -> (Lz.Deflate.decode src dst, String.length big, n)
  in
  Lz_support.lock_released "a long decode leaves another domain's calls running"
    long
    ~small:(fun () -> is_ok (Lz.Gzip.decode small small_dst))
    ~check:(fun (r, consumed, produced) ->
      equal
        (result Lz_support.progress Lz_support.error)
        (Ok { Lz.consumed; produced })
        r)

let () =
  exit
    (run "Deflate"
       [
         arguments;
         goldens;
         blocks;
         zlib;
         gzip;
         empty;
         bounds;
         group "streaming" [ steps; exact_end; window ];
         group "memory" [ allocation; domains ];
       ])
