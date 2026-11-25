(* Snappy blocks through Lz.Snappy. Expected values come from the format
   description (format_description.txt in google/snappy). An error's expected
   offset is the first byte of what is wrong, as lz.mli states it: the
   preamble's first byte, or the tag of the element. Errors are compared on
   their offset and kind: lz's messages are a baseline, which the pp_error test
   prints. *)

open Windtrap
open Lz_support

(* The format, written from its description *)

(* An element of a block: a literal, or a copy of [length] bytes from [offset]
   back in the form with [kind] (1, 2 or 4) bytes of offset. *)
type element =
  | Literal of string
  | Copy of { kind : int; offset : int; length : int }

(* Section 1: 7 bits a byte, low bits first, the high bit set on every byte but
   the last. *)
let varint b n =
  let rec go n =
    if n < 0x80 then Buffer.add_char b (Char.chr n)
    else begin
      Buffer.add_char b (Char.chr (n land 0x7f lor 0x80));
      go (n lsr 7)
    end
  in
  go n

(* Section 2.1: the length minus one in the tag up to 60 bytes, else in the 1 to
   4 bytes after a tag of 60 to 63. *)
let add_literal b s =
  let m = String.length s - 1 in
  if m < 60 then Buffer.add_char b (Char.chr (m lsl 2))
  else begin
    let rec count k = if m lsr (8 * k) = 0 then k else count (k + 1) in
    let k = count 1 in
    Buffer.add_char b (Char.chr ((59 + k) lsl 2));
    for i = 0 to k - 1 do
      Buffer.add_char b (Char.chr ((m lsr (8 * i)) land 0xff))
    done
  end;
  Buffer.add_string b s

(* Section 2.2. *)
let add_copy b ~kind ~offset ~length =
  match kind with
  | 1 ->
      Buffer.add_char b
        (Char.chr (1 lor ((length - 4) lsl 2) lor ((offset lsr 8) lsl 5)));
      Buffer.add_char b (Char.chr (offset land 0xff))
  | 2 ->
      Buffer.add_char b (Char.chr (2 lor ((length - 1) lsl 2)));
      Buffer.add_uint16_le b offset
  | _ ->
      Buffer.add_char b (Char.chr (3 lor ((length - 1) lsl 2)));
      Buffer.add_int32_le b (Int32.of_int offset)

let size = function Literal s -> String.length s | Copy c -> c.length

(* The bytes of [elements], after a block's preamble. *)
let body elements =
  let b = Buffer.create 64 in
  List.iter
    (function
      | Literal s -> add_literal b s
      | Copy { kind; offset; length } -> add_copy b ~kind ~offset ~length)
    elements;
  Buffer.contents b

(* The block of [elements], with a preamble of [n] (default: the length they
   decode to). *)
let block ?n elements =
  let total = List.fold_left (fun n e -> n + size e) 0 elements in
  let b = Buffer.create 16 in
  varint b (Option.value n ~default:total);
  Buffer.contents b ^ body elements

(* What [elements] decode to. A copy reads byte by byte, so one longer than its
   offset repeats the bytes it has just written. *)
let meaning elements =
  let b = Buffer.create 64 in
  List.iter
    (function
      | Literal s -> Buffer.add_string b s
      | Copy { offset; length; _ } ->
          for _ = 1 to length do
            Buffer.add_char b (Buffer.nth b (Buffer.length b - offset))
          done)
    elements;
  Buffer.contents b

let pp_element ppf = function
  | Literal s when String.length s <= 16 -> Format.fprintf ppf "Literal %S" s
  | Literal s -> Format.fprintf ppf "Literal <%d bytes>" (String.length s)
  | Copy { kind; offset; length } ->
      Format.fprintf ppf "Copy { kind = %d; offset = %d; length = %d }" kind
        offset length

let pp_elements = Format.pp_print_list ~pp_sep:Format.pp_print_space pp_element

(* Blocks of every element form: literals in the tag and after it, copies in
   each form, from under 16 bytes back or anywhere back. *)
let elements =
  let open Gen in
  let literal =
    let* n =
      frequency
        [
          (4, int_range 1 16);
          (3, int_range 17 60);
          (2, int_range 61 300);
          (1, int_range 257 5000);
        ]
    in
    (* Bytes from a drawn seed: a literal's bytes change no law, and drawing
       each byte as an element costs more than the decode. *)
    map (fun seed -> Literal (build (Noise (seed, n)))) small_int
  in
  let copy produced =
    let* kind = of_list [ 1; 2; 4 ] in
    let far = match kind with 1 -> 2047 | 2 -> 65535 | _ -> produced in
    let far = min far produced in
    let* offset = one_of [ int_range 1 (min 16 far); int_range 1 far ] in
    let+ length = if kind = 1 then int_range 4 11 else int_range 1 64 in
    Copy { kind; offset; length }
  in
  let rec go k produced acc =
    if k = 0 then constant (List.rev acc)
    else
      let* e =
        if produced = 0 then literal
        else frequency [ (1, literal); (2, copy produced) ]
      in
      go (k - 1) (produced + size e) (e :: acc)
  in
  with_pp pp_elements (bind (int_range 1 40) (fun k -> go k 0 []))

let noise n = build (Noise (n, n))

(* Calls *)

let guard = 64

(* [decode ~offset ~cap s] decodes [s], placed at [offset], into [cap] bytes at
   [offset], followed by [guard] bytes. A success consumes all of [s], and no
   decode writes past the destination. *)
let decode ?(offset = 0) ~cap s =
  let whole = create (offset + cap + guard) in
  let dst = Bigarray.Array1.sub whole offset cap in
  let r = Lz.Snappy.decode (placed ~offset s) dst in
  equal string ~msg:"bytes past the destination" (String.make guard '\xa5')
    (String.sub (to_string whole) (offset + cap) guard);
  match r with
  | Ok { consumed; produced } ->
      equal int ~msg:"consumed" (String.length s) consumed;
      Ok (to_string ~length:produced dst)
  | Error e -> Error e

(* [encode ~offset ~cap s] is [s]'s block, encoded from [offset] into [cap]
   bytes at [offset], or [None]. *)
let encode ?(offset = 0) ~cap s =
  let dst = room ~offset cap in
  Lz.Snappy.encode (placed ~offset s) dst
  |> Option.map (fun n -> to_string ~length:n dst)

let encode_bound s =
  require_some ~msg:"the bound fits"
    (encode ~cap:(Lz.Snappy.bound (String.length s)) s)

(* A decode's output, or its error compared on its offset and kind. *)
let outcome =
  Testable.make ~pp:(Testable.pp decoded) ~equal:(fun a b ->
      match (a, b) with
      | Ok a, Ok b -> String.equal a b
      | Error a, Error b -> Testable.equal refusal a b
      | Ok _, Error _ | Error _, Ok _ -> false)

(* [fails ~cap ~at why s] asserts that [s], decoded into [cap] bytes (default 64
   times its length), is refused at [at] for [why]. *)
let fails ?cap ~at why s =
  let cap = Option.value cap ~default:(64 * String.length s) in
  equal outcome (Error { Lz.at; why }) (decode ~cap s)

let malformed ~at s = fails ~at (Lz.Malformed "") s

(* Blocks *)

let xababab =
  (* Section 2.2's example: "xababab" is a literal "xab" and a copy of 4 bytes
     from 2 back. *)
  cases "section 2.2's example decodes to xababab"
    ~id:(fun k -> Printf.sprintf "copy with %d-byte offset" k)
    [ 1; 2; 4 ]
    (fun kind ->
      let s = block [ Literal "xab"; Copy { kind; offset = 2; length = 4 } ] in
      equal outcome (Ok "xababab") (decode ~cap:7 s))

let preambles =
  (* Section 1: 64 is 0x40, 2097150 is FE FF 7F. *)
  test "the preamble's examples, 64 and 2097150, decode" (fun () ->
      let s = String.make 64 'a' in
      equal outcome (Ok s) (decode ~cap:64 ("\x40" ^ body [ Literal s ]));
      let n = 2097150 in
      let run = Copy { kind = 2; offset = 1; length = 64 } in
      let rest = n - 1 in
      let es =
        (Literal "a" :: List.init (rest / 64) (fun _ -> run))
        @ [ Copy { kind = 2; offset = 1; length = rest mod 64 } ]
      in
      equal outcome
        (Ok (String.make n 'a'))
        (decode ~cap:n ("\xfe\xff\x7f" ^ body es)))

let literal_lengths =
  (* Section 2.1: lengths up to 60 in the tag, longer ones in 1 to 4 bytes after
     it, at the edge of each count. *)
  cases "a literal of every length class decodes" ~id:string_of_int
    [ 1; 59; 60; 61; 256; 257; 65536; 65537; 70000 ] (fun n ->
      let s = noise n in
      equal outcome (Ok s) (decode ~cap:n (block [ Literal s ])))

let tag_63 =
  (* Section 2.1: tag 63 holds the length minus one in 4 bytes after it,
     whatever the length. *)
  test "a literal under tag 63 decodes" (fun () ->
      equal outcome (Ok "hello") (decode ~cap:5 "\x05\xfc\x04\x00\x00\x00hello"))

let far_offsets =
  (* Section 2.2: a 1-byte offset reaches 2047 back, a 2-byte one 65535, a
     4-byte one any distance the output holds. *)
  cases "a copy reaches back as far as its offset's form holds"
    ~id:(fun (kind, offset) -> Printf.sprintf "%d-byte offset %d" kind offset)
    [ (1, 2047); (2, 2048); (2, 65535); (4, 65536); (4, 70000) ]
    (fun (kind, offset) ->
      let es = [ Literal (noise 70000); Copy { kind; offset; length = 11 } ] in
      let raw = meaning es in
      equal outcome (Ok raw) (decode ~cap:(String.length raw) (block es)))

let larger_destination =
  (* The preamble, not the destination, sets the output's length. *)
  test "a destination larger than the preamble gets its length" (fun () ->
      let s =
        block [ Literal "xab"; Copy { kind = 1; offset = 2; length = 4 } ]
      in
      let dst = create 64 in
      equal (result progress error)
        (Ok { Lz.consumed = String.length s; produced = 7 })
        (Lz.Snappy.decode (of_string s) dst);
      equal string "xababab" (to_string ~length:7 dst))

(* A copy over its own output from each distance under 16, which the drawn
   blocks reach one case in six or seven. *)
let overlapping =
  Literal (noise 70)
  :: List.init 15 (fun i -> Copy { kind = 2; offset = i + 1; length = i + 2 })

let law_blocks =
  prop "a block decodes to what its elements mean" ~examples:[ overlapping ]
    elements (fun es ->
      List.iter
        (function
          | Literal s ->
              cover "literal in the tag" (String.length s <= 60);
              cover "literal after the tag" (String.length s > 60)
          | Copy { kind; offset; length } ->
              cover "copy with 1-byte offset" (kind = 1);
              cover "copy with 2-byte offset" (kind = 2);
              cover "copy with 4-byte offset" (kind = 4);
              cover "copy over its own output" (offset < length);
              cover "offset under 16, length over 16"
                (offset < 16 && length > 16);
              (* Each offset under 16 repeats its pattern its own way. *)
              for d = 1 to 15 do
                cover
                  (Printf.sprintf "copy over its own output, %d back" d)
                  (offset = d && length > d)
              done)
        es;
      let raw = meaning es in
      let offset = String.length raw mod 16 in
      equal outcome (Ok raw)
        (decode ~offset ~cap:(String.length raw) (block es)))

let blocks =
  group "blocks"
    [
      xababab;
      preambles;
      (* Section 1: the empty block is the preamble of 0. *)
      test "a preamble of 0 is the empty block" (fun () ->
          equal outcome (Ok "") (decode ~cap:0 "\x00"));
      literal_lengths;
      tag_63;
      far_offsets;
      larger_destination;
      law_blocks;
    ]

(* Refusals *)

let preambles_malformed =
  (* Section 1: a length is at most 2^32 - 1, so its varint at most 5 bytes, the
     last holding 4 bits. *)
  cases "a preamble past 32 bits is Malformed at 0" ~id:String.escaped
    [
      "\x80\x80\x80\x80\x80\x00"; "\xff\xff\xff\xff\x10"; "\x80\x80\x80\x80\x80";
    ] (fun s -> malformed ~at:0 s)

let offsets =
  cases "a copy offset of 0 is Malformed at its tag"
    ~id:(fun kind -> Printf.sprintf "zero in the %d-byte form" kind)
    [ 1; 2; 4 ]
    (fun kind ->
      (* The copy's tag is at 6. *)
      malformed ~at:6
        (block [ Literal "abcd"; Copy { kind; offset = 0; length = 4 } ]))

let before_start =
  (* Section 2.2: an offset past the output is not allowed. The offset starts
     from a 1-byte-offset tag's bits and its bytes follow, low byte first, each
     only adding: once the bytes so far take it past the output, the copy is
     Malformed at its tag, even when the input ends inside its offset. Section
     2: a block cannot start with a copy. *)
  let far4 =
    block [ Literal "abcd"; Copy { kind = 4; offset = 0x7fffffff; length = 4 } ]
  in
  cases "a copy offset past the output is Malformed at its tag"
    ~id:(fun (name, _, _) -> name)
    [
      ( "in its 1-byte offset",
        block [ Literal "ab"; Copy { kind = 1; offset = 3; length = 4 } ],
        4 );
      ( "in its tag's bits",
        block [ Literal "abcd"; Copy { kind = 1; offset = 300; length = 4 } ],
        6 );
      ("in the first of 4 bytes", far4, 6);
      ("in the first of 4 bytes, cut", String.sub far4 0 8, 6);
      ( "in the second of 2 bytes",
        block [ Literal "abcd"; Copy { kind = 2; offset = 0x100; length = 4 } ],
        6 );
      ( "in the last of 4 bytes",
        block
          [ Literal "abcd"; Copy { kind = 4; offset = 0x1000000; length = 4 } ],
        6 );
      ( "as the first element",
        block [ Copy { kind = 2; offset = 1; length = 4 } ],
        1 );
    ]
    (fun (_, s, at) -> malformed ~at s)

let long_literals =
  (* A long literal's length follows its tag, low byte first, so each byte only
     adds to it: once the bytes so far take it past the preamble's room, the
     element is Malformed at its tag, at 1, even when the input ends inside its
     length. *)
  cases "a long literal past the preamble is Malformed at its tag"
    ~id:(fun (name, _) -> name)
    [
      ("its 1-byte length", "\x03\xf0\x63");
      ("the first of 2 length bytes", "\x03\xf4\x63\x00");
      ("the first of 2 length bytes, cut", "\x03\xf4\x63");
      ("the second of 2 length bytes", "\x03\xf4\x01\x01");
      ("the fourth of 4 length bytes", "\x03\xfc\x01\x00\x00\x01");
    ]
    (fun (_, s) -> malformed ~at:1 s)

let past_preamble =
  (* A literal and copies of 64 bytes from 1 back under a preamble of 1 MiB: the
     16384th copy, at 5 + 3 * 16383 after the preamble's 3 bytes and the
     literal's 2, passes it. *)
  test "a block decoding past its preamble is Malformed at the copy that does"
    (fun () ->
      let cap = 1 lsl 20 in
      let b = Buffer.create (3 * cap / 64) in
      varint b cap;
      add_literal b "a";
      for _ = 1 to (cap / 64) + 1 do
        add_copy b ~kind:2 ~offset:1 ~length:64
      done;
      equal outcome
        (Error { Lz.at = 5 + (3 * ((cap / 64) - 1)); why = Malformed "" })
        (decode ~cap (Buffer.contents b)))

let truncated =
  let copy4 =
    block [ Literal "abcd"; Copy { kind = 4; offset = 4; length = 4 } ]
  in
  cases "an input that ends inside an element is Truncated at its length"
    ~id:(fun (name, _) -> name)
    [
      ("empty", "");
      ("inside the preamble", "\x80");
      ("before any element", "\x05");
      ("inside a literal", String.sub (block [ Literal "abcde" ]) 0 4);
      ("inside a long literal's length", "\xff\x01\xf4\x10");
      ("inside a copy's offset", String.sub copy4 0 8);
    ]
    (fun (_, s) -> fails ~cap:1024 ~at:(String.length s) Truncated s)

(* A reference, written from the format and lz.mli's offsets *)

(* [reference ~cap s] is the block [s]'s bytes, byte by byte and with no fast
   path, with the error each input earns: [Too_long] when the preamble passes
   [cap], the input's length for a truncation, otherwise the tag of the first
   element that is wrong, the preamble's first byte for a preamble past 32 bits
   (Sections 1 and 2.1-2.2). *)
let reference ~cap s =
  let input = String.length s in
  let byte i = Char.code (String.get s i) in
  let exception Stop of Lz.error in
  let fail at why = raise (Stop { Lz.at; why }) in
  let truncated () = fail input Lz.Truncated in
  let malformed at msg = fail at (Lz.Malformed msg) in
  let rec preamble k acc i =
    if i = input then truncated ()
    else
      let b = byte i in
      if k = 4 && b > 0x0f then malformed 0 "invalid preamble"
      else
        let acc = acc lor ((b land 0x7f) lsl (7 * k)) in
        if b < 0x80 then (acc, i + 1) else preamble (k + 1) acc (i + 1)
  in
  let element out n at =
    let tag = byte at and produced = Buffer.length out in
    let room = n - produced and kind = tag land 3 in
    let is_copy = kind <> 0 in
    let len, hi, extra =
      match kind with
      | 0 ->
          let h = tag lsr 2 in
          if h < 60 then (h + 1, 0, 0) else (0, 0, h - 59)
      | 1 -> (4 + ((tag lsr 2) land 7), (tag lsr 5) lsl 8, 1)
      | 2 -> ((tag lsr 2) + 1, 0, 2)
      | _ -> ((tag lsr 2) + 1, 0, 4)
    in
    if room = 0 || len > room then
      malformed at "output longer than the preamble";
    if is_copy && (produced = 0 || hi > produced) then
      malformed at "copy offset too far back";
    let rec trailer k acc i =
      if k = extra then (acc, i)
      else if i = input then truncated ()
      else
        let acc = acc lor (byte i lsl (8 * k)) in
        if len = 0 && acc + 1 > room then
          malformed at "output longer than the preamble";
        if is_copy && hi + acc > produced then
          malformed at "copy offset too far back";
        trailer (k + 1) acc (i + 1)
    in
    let t, i = trailer 0 0 (at + 1) in
    if is_copy then begin
      let off = hi + t in
      if off = 0 then malformed at "invalid copy offset";
      for _ = 1 to len do
        Buffer.add_char out (Buffer.nth out (Buffer.length out - off))
      done;
      i
    end
    else begin
      let len = if len = 0 then t + 1 else len in
      if len > input - i then truncated ();
      Buffer.add_substring out s i len;
      i + len
    end
  in
  match
    let n, i = preamble 0 0 0 in
    if n > cap then fail 0 Lz.Too_long;
    let out = Buffer.create n in
    let i = ref i in
    while !i < input do
      i := element out n !i
    done;
    if Buffer.length out = n then Buffer.contents out else truncated ()
  with
  | v -> Ok v
  | exception Stop e -> Error e

(* [header e] is how many bytes [e] takes before any literal data: its tag and
   the length or offset bytes after it. *)
let header = function
  | Literal s ->
      let m = String.length s - 1 in
      let rec count k = if m lsr (8 * k) = 0 then k else count (k + 1) in
      if m < 60 then 1 else 1 + count 1
  | Copy { kind; _ } -> 1 + kind

let encoded e =
  header e + match e with Literal s -> String.length s | Copy _ -> 0

let set_byte s p v = String.mapi (fun i c -> if i = p then Char.chr v else c) s

(* The offsets in [block ~n es] of the first [header_bytes e] bytes of each
   element, as a list in order. *)
let offsets_of ?(header_bytes = header) s es =
  let body = List.fold_left (fun n e -> n + encoded e) 0 es in
  let first = String.length s - body in
  let at = ref first in
  List.concat_map
    (fun e ->
      let here = !at in
      at := here + encoded e;
      List.init (header_bytes e) (fun k -> here + k))
    es

type change = Same | Set of int * int | Cut of int

let pp_change ppf = function
  | Same -> Format.pp_print_string ppf "unchanged"
  | Set (p, v) -> Format.fprintf ppf "byte %d set to %d" p v
  | Cut p -> Format.fprintf ppf "cut to %d bytes" p

let apply change s =
  match change with
  | Same -> s
  | Set (p, v) -> set_byte s p v
  | Cut p -> String.sub s 0 p

(* Well-formed blocks, then the same with a tag, length or offset byte changed,
   a byte changed anywhere, or cut short, under a preamble that may lie; into a
   destination of the preamble's length or the elements', or one smaller than
   the preamble. *)
let changed =
  let open Gen in
  let* es = elements in
  let total = List.fold_left (fun n e -> n + size e) 0 es in
  let* n =
    frequency
      [
        (3, constant total);
        (1, map (fun d -> max 0 (total - d)) (int_range 1 8));
        (1, map (fun d -> total + d) (int_range 1 8));
      ]
  in
  let s = block ~n es in
  let* change =
    frequency
      [
        (1, constant Same);
        ( 4,
          let+ p = of_list (offsets_of s es) and+ v = int_range 0 255 in
          Set (p, v) );
        ( 1,
          let+ p = int_range 0 (String.length s - 1) and+ v = int_range 0 255 in
          Set (p, v) );
        (2, map (fun p -> Cut p) (int_range 0 (String.length s)));
      ]
  in
  let+ cap =
    frequency [ (3, constant (max n total)); (1, int_range 0 (max 0 (n - 1))) ]
  in
  (es, n, change, cap)

let pp_changed ppf (es, n, change, cap) =
  Format.fprintf ppf "[%a] under a preamble of %d, %a, into %d bytes"
    pp_elements es n pp_change change cap

let law_reference =
  (* A literal byte changed, which a drawn case is one time in seven. *)
  let literal_changed = ([ Literal "abcd" ], 4, Set (5, Char.code 'e'), 4) in
  prop "any bytes decode as the reference decodes them"
    ~examples:[ literal_changed ] (Gen.with_pp pp_changed changed)
    (fun (es, n, change, cap) ->
      let s = block ~n es in
      let header =
        match change with
        | Set (p, _) -> List.mem p (offsets_of s es)
        | Same | Cut _ -> false
      in
      let s = apply change s in
      let want = reference ~cap s in
      cover "a tag, length or offset byte changed" header;
      cover "a preamble above the destination"
        (match want with Error { Lz.why = Too_long; _ } -> true | _ -> false);
      cover "a block refused" (Result.is_error want);
      cover "a block decoded after a change"
        (change <> Same && Result.is_ok want);
      equal outcome want (decode ~cap s))

let every_tag name es =
  (* Every value of every tag of the tail of [es], past the first element,
     through the source and destination of exactly the block's size that the
     sanitize profile checks. *)
  let s = block es in
  let cap = String.length (meaning es) in
  let tags = offsets_of ~header_bytes:(fun _ -> 1) s es in
  List.iter
    (fun p ->
      for v = 0 to 255 do
        equal outcome
          ~msg:(Printf.sprintf "%s: byte %d = %d" name p v)
          (reference ~cap (set_byte s p v))
          (decode ~cap (set_byte s p v))
      done)
    (List.tl tags)

let boundaries =
  let lead = Literal (noise 200) in
  [
    (* Two input bytes for four of output: the input runs out first, each tag at
       its own distance from the end. The extra literal shifts the distances by
       3, an odd step over copies 2 bytes apart. *)
    test "a tag at each distance from the input's end decodes as the reference"
      (fun () ->
        let copies =
          List.init 60 (fun _ -> Copy { kind = 1; offset = 3; length = 4 })
        in
        every_tag "input" (lead :: copies);
        every_tag "input, shifted" ((lead :: copies) @ [ Literal "ab" ]));
    (* Two or three input bytes for one of output: the output runs out first,
       with the input's room to spare, a tag at each byte of room. *)
    test
      "a tag at each byte of room before the output's end decodes as the \
       reference" (fun () ->
        let tail =
          List.init 80 (fun i ->
              if i mod 2 = 0 then Literal "x"
              else Copy { kind = 2; offset = 1; length = 1 })
        in
        every_tag "output" (lead :: tail));
  ]

let refusals =
  group "refusals"
    ([
       test "a preamble above the destination is Too_long at the preamble"
         (fun () ->
           fails ~cap:199 ~at:0 Too_long
             (block [ Literal (String.make 200 'a') ]));
       test "a preamble of 2^32 - 1 is a length, Too_long at 0" (fun () ->
           fails ~cap:16 ~at:0 Too_long "\xff\xff\xff\xff\x0f\x00a");
       preambles_malformed;
       offsets;
       before_start;
       test "a preamble below the data's length is Malformed at the element"
         (fun () ->
           malformed ~at:4 (block ~n:3 [ Literal "ab"; Literal "cde" ]));
       test "a copy past the preamble is Malformed at the copy" (fun () ->
           malformed ~at:3
             (block ~n:5
                [ Literal "a"; Copy { kind = 1; offset = 1; length = 8 } ]));
       long_literals;
       past_preamble;
       test "an element after the preamble's length is Malformed" (fun () ->
           malformed ~at:3 (block ~n:1 [ Literal "a"; Literal "b" ]));
       (* A block has no end mark: bytes after it are read as more of it, so a
          literal after the preamble's length is refused at its tag. *)
       test "bytes after a block are read as more of it" (fun () ->
           let b = block [ Literal "hello" ] in
           malformed ~at:(String.length b) (b ^ "\x00x"));
       truncated;
       test "a preamble above the data's length is Truncated" (fun () ->
           let s = block ~n:10 [ Literal "abcde" ] in
           fails ~at:(String.length s) Truncated s);
       law_reference;
     ]
    @ boundaries)

let messages =
  group "pp_error"
    [
      test "prints each refusal's kind, offset and message" (fun () ->
          let copy offset =
            block [ Literal "abcd"; Copy { kind = 1; offset; length = 4 } ]
          in
          List.iter
            (fun (name, cap, s) ->
              match decode ~cap s with
              | Ok _ -> Format.printf "%s: decodes@." name
              | Error e -> Format.printf "%s: %a@." name Lz.pp_error e)
            [
              ("empty", 16, "");
              ("a preamble past 32 bits", 16, "\x80\x80\x80\x80\x80\x00");
              ("a preamble above the destination", 3, block [ Literal "abcd" ]);
              ("a copy offset of 0", 16, copy 0);
              ("a copy offset past the output", 16, copy 5);
              ( "an element past the preamble",
                16,
                block ~n:3 [ Literal "ab"; Literal "cde" ] );
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            empty: truncated at byte 0
            a preamble past 32 bits: malformed at byte 0: invalid preamble
            a preamble above the destination: output too long at byte 0
            a copy offset of 0: malformed at byte 6: invalid copy offset
            a copy offset past the output: malformed at byte 6: copy offset too far back
            an element past the preamble: malformed at byte 4: output longer than the preamble
          |});
    ]

(* Decode *)

let every_golden =
  cases "every golden decodes to its input, all of it consumed"
    ~id:(fun g -> g.file)
    (goldens_of "snappy")
    (fun g ->
      let raw = Lazy.force g.raw and data = Lazy.force g.data in
      let dst = create (String.length raw) in
      equal (result progress error)
        (Ok { Lz.consumed = String.length data; produced = String.length raw })
        (Lz.Snappy.decode (of_string data) dst);
      equal decoded (Ok raw) (Ok (to_string dst)))

(* The words [f ()] allocates on the OCaml heap, less what measuring an empty
   call allocates, and the words its result holds. *)
let allocated f =
  let measure g =
    let before = Gc.allocated_bytes () in
    let r = Sys.opaque_identity (g ()) in
    (Gc.allocated_bytes () -. before, r)
  in
  let empty, () = measure (fun () -> ()) in
  let bytes, r = measure f in
  ( int_of_float ((bytes -. empty) /. float_of_int (Sys.word_size / 8)),
    Obj.reachable_words (Obj.repr r) )

let text () =
  List.find (fun g -> g.file = "text_200000.g12.snappy") (goldens_of "snappy")

let decoding =
  group "decode"
    [
      every_golden;
      test "decode allocates no memory but its result" (fun () ->
          let g = text () in
          let src = of_string (Lazy.force g.data) in
          let dst = create (String.length (Lazy.force g.raw)) in
          let words, result = allocated (fun () -> Lz.Snappy.decode src dst) in
          equal int result words);
    ]

(* Bound *)

let bounding =
  group "bound"
    [
      prop "bound n is 32 + n + n / 6"
        ~examples:[ 0; 5; 6; 65_536; 0xffff_ffff ] (Gen.int_range 0 0xffff_ffff)
        (fun n -> equal int (32 + n + (n / 6)) (Lz.Snappy.bound n));
      test "a negative size has no bound" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Lz.Snappy.bound (-1)));
      test "a size above 2^32 - 1 has no bound" (fun () ->
          (* Section 1: the preamble states at most 2^32 - 1. *)
          raises_match Exn.invalid_arg (fun () -> Lz.Snappy.bound (1 lsl 32)));
    ]

(* Encode *)

let law_round_trip =
  prop "decode (encode x) = x" ~examples:(Lz_support.every_kind @ large) recipe
    (fun r ->
      let x = build r in
      let n = String.length x in
      cover "empty" (n = 0);
      cover "shorter than the scan's margin" (n > 0 && n < 15);
      cover "several fragments" (n > 65536);
      let offset = n mod 16 in
      let z =
        require_some ~msg:"the bound fits"
          (encode ~offset ~cap:(Lz.Snappy.bound n) x)
      in
      Law.round_trip string string
        (fun _ -> z)
        (fun z ->
          require_ok
            ~pp:(Testable.pp Lz_support.error)
            (decode ~offset ~cap:n z))
        x)

let law_exact =
  prop "a destination of the block's length holds it"
    ~examples:Lz_support.every_kind recipe (fun r ->
      let x = build r in
      let z = encode_bound x in
      equal (option string) (Some z) (encode ~cap:(String.length z) x))

let law_short =
  prop "a destination one byte short of the block is refused"
    ~examples:Lz_support.every_kind recipe (fun r ->
      let x = build r in
      let k = String.length (encode_bound x) in
      is_none (encode ~cap:(k - 1) x))

let law_within =
  prop "an encode writes nothing past its destination"
    ~examples:Lz_support.every_kind recipe (fun r ->
      let x = build r in
      let k = String.length (encode_bound x) in
      List.iter
        (fun cap ->
          let buf = create (cap + 64) in
          ignore
            (Lz.Snappy.encode (of_string x) (Bigarray.Array1.sub buf 0 cap));
          equal string ~msg:"bytes past the destination" (String.make 64 '\xa5')
            (String.sub (to_string buf) cap 64))
        [ Lz.Snappy.bound (String.length x); k - 1 ])

let encoding =
  group "encode"
    [
      law_round_trip;
      law_exact;
      law_short;
      law_within;
      test "encode allocates its tables outside the OCaml heap" (fun () ->
          let src = of_string (Lazy.force (text ()).raw) in
          let dst = create (Lz.Snappy.bound (Bigarray.Array1.dim src)) in
          let words, result = allocated (fun () -> Lz.Snappy.encode src dst) in
          equal int result words);
    ]

(* Encode_string *)

let encode_string =
  group "encode_string"
    [
      prop "encode_string writes encode's block" ~examples:Lz_support.every_kind
        recipe (fun r ->
          let x = build r in
          cover "empty" (x = "");
          equal string (encode_bound x) (Lz.Snappy.encode_string x));
      (* Section 1: the empty block is the preamble of 0. *)
      test "the empty string is the preamble of 0" (fun () ->
          equal string "\x00" (Lz.Snappy.encode_string ""));
    ]

let () =
  exit
    (run "Snappy"
       [
         blocks; refusals; messages; decoding; bounding; encoding; encode_string;
       ])
