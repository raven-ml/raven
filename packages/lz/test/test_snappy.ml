(* Snappy blocks through Lz.Snappy. Expected values come from the format
   description (format_description.txt in google/snappy). An error's
   expected offset is the first byte of what is wrong, as lz.mli states
   it: the preamble's first byte, or the tag of the element. *)

open Windtrap
open Lz_support

(* The format, written from its description *)

(* An element of a block: a literal, or a copy of [length] bytes from
   [offset] back in the form with [kind] (1, 2 or 4) bytes of offset. *)
type element =
  | Literal of string
  | Copy of { kind : int; offset : int; length : int }

(* Section 1: 7 bits a byte, low bits first, the high bit set on every byte
   but the last. *)
let varint b n =
  let rec go n =
    if n < 0x80 then Buffer.add_char b (Char.chr n)
    else begin
      Buffer.add_char b (Char.chr (n land 0x7f lor 0x80));
      go (n lsr 7)
    end
  in
  go n

(* Section 2.1: the length minus one in the tag up to 60 bytes, else in the
   1 to 4 bytes after a tag of 60 to 63. *)
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

(* The block of [elements], with a preamble of [n] (default: the length
   they decode to). *)
let block ?n elements =
  let body = Buffer.create 64 in
  List.iter
    (function
      | Literal s -> add_literal body s
      | Copy { kind; offset; length } -> add_copy body ~kind ~offset ~length)
    elements;
  let b = Buffer.create (Buffer.length body + 5) in
  let total = List.fold_left (fun n e -> n + size e) 0 elements in
  varint b (Option.value n ~default:total);
  Buffer.add_buffer b body;
  Buffer.contents b

(* What [elements] decode to. A copy reads byte by byte, so one longer than
   its offset repeats the bytes it has just written. *)
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
  | Literal s -> Format.fprintf ppf "Literal <%d bytes>" (String.length s)
  | Copy { kind; offset; length } ->
      Format.fprintf ppf "Copy { kind = %d; offset = %d; length = %d }" kind
        offset length

(* Blocks of every element form: literals in the tag and after it, copies
   in each form, from under 16 bytes back or anywhere back. *)
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
    (* Bytes from a drawn seed: a literal's bytes change no law, and
       drawing each byte as an element costs more than the decode. *)
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
  with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_element)
    (bind (int_range 1 40) (fun k -> go k 0 []))

let noise n = build (Noise (n, n))

(* Calls *)

(* [decode ~offset ~cap s] decodes [s], placed at [offset], into [cap] bytes
   at [offset]. A success consumes all of [s]. *)
let decode ?(offset = 0) ~cap s =
  let dst = room ~offset cap in
  match Lz.Snappy.decode (placed ~offset s) dst with
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

let fails ?cap ~at why s =
  let cap = Option.value cap ~default:(64 * String.length s) in
  equal decoded (Error { Lz.at; why }) (decode ~cap s)

let malformed ~at msg s = fails ~at (Lz.Malformed msg) s

(* Format *)

let xababab =
  (* Section 2.2's example: "xababab" is a literal "xab" and a copy of 4
     bytes from 2 back. *)
  cases
    ~name:(fun k -> Printf.sprintf "copy with %d-byte offset" k)
    "xababab" [ 1; 2; 4 ]
    (fun kind ->
      let s = block [ Literal "xab"; Copy { kind; offset = 2; length = 4 } ] in
      equal decoded (Ok "xababab") (decode ~cap:7 s))

let preambles =
  (* Section 1: 64 is 0x40, 2097150 is FE FF 7F. *)
  test "the preamble's examples" (fun () ->
      let s = block [ Literal (String.make 64 'a') ] in
      equal string "\x40" (String.sub s 0 1);
      equal decoded (Ok (String.make 64 'a')) (decode ~cap:64 s);
      let n = 2097150 in
      let run = Copy { kind = 2; offset = 1; length = 64 } in
      let rest = n - 1 in
      let es =
        (Literal "a" :: List.init (rest / 64) (fun _ -> run))
        @ [ Copy { kind = 2; offset = 1; length = rest mod 64 } ]
      in
      let s = block es in
      equal string "\xfe\xff\x7f" (String.sub s 0 3);
      equal decoded (Ok (String.make n 'a')) (decode ~cap:n s))

let literal_lengths =
  (* Section 2.1: lengths up to 60 in the tag, longer ones in 1 to 4 bytes
     after it, at the edge of each count. *)
  cases ~name:string_of_int "literal lengths"
    [ 1; 59; 60; 61; 256; 257; 65536; 65537; 70000 ]
    (fun n ->
      let s = noise n in
      equal decoded (Ok s) (decode ~cap:n (block [ Literal s ])))

let far_offsets =
  (* Section 2.2: a 1-byte offset reaches 2047 back, a 2-byte one 65535,
     a 4-byte one any distance the output holds. *)
  cases
    ~name:(fun (kind, offset) -> Printf.sprintf "%d-byte offset %d" kind offset)
    "far offsets"
    [ (1, 2047); (2, 2048); (2, 65535); (4, 65536); (4, 70000) ]
    (fun (kind, offset) ->
      let es = [ Literal (noise 70000); Copy { kind; offset; length = 11 } ] in
      let raw = meaning es in
      equal decoded (Ok raw) (decode ~cap:(String.length raw) (block es)))

let law_blocks =
  prop "a block decodes to what its elements mean" elements (fun es ->
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
                (offset < 16 && length > 16))
        es;
      let raw = meaning es in
      let offset = String.length raw mod 16 in
      equal decoded (Ok raw)
        (decode ~offset ~cap:(String.length raw) (block es)))

(* Round trips *)

let law_round_trip =
  prop "decode (encode x) = x" ~examples:large recipe (fun r ->
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
        (fun z -> require_ok ~pp:(Testable.pp Lz_support.error) (decode ~offset ~cap:n z))
        x)

let law_strings =
  prop "decode (encode_string x) = x" recipe (fun r ->
      let x = build r in
      cover "empty" (x = "");
      let z = Lz.Snappy.encode_string x in
      equal decoded (Ok x) (decode ~cap:(String.length x) z))

let empty =
  (* Section 1: the empty block is the preamble of 0. *)
  test "the empty string is the block of no elements" (fun () ->
      equal string "\x00" (Lz.Snappy.encode_string "");
      equal decoded (Ok "") (decode ~cap:0 "\x00"))

let law_exact =
  (* A destination of the block's length holds it; one a byte shorter does
     not, and nothing past it is written. *)
  prop "a destination of the block's length, and one short" recipe (fun r ->
      let x = build r in
      let z = encode_bound x in
      let k = String.length z in
      equal (option string) (Some z) (encode ~cap:k x);
      let buf = create (k + 64) in
      let dst = Bigarray.Array1.sub buf 0 (k - 1) in
      equal (option int) None (Lz.Snappy.encode (of_string x) dst);
      equal string ~msg:"bytes past the destination" (String.make 65 '\xa5')
        (String.sub (to_string buf) (k - 1) 65))

(* Refusals *)

let preamble_lies =
  [
    test "a preamble above the data's length is Truncated" (fun () ->
        let s = block ~n:10 [ Literal "abcde" ] in
        fails ~at:(String.length s) Truncated s);
    test "a preamble below the data's length is Malformed at the element"
      (fun () ->
        malformed ~at:4 "output longer than the preamble"
          (block ~n:3 [ Literal "ab"; Literal "cde" ]));
    test "a copy past the preamble is Malformed at the copy" (fun () ->
        malformed ~at:3 "output longer than the preamble"
          (block ~n:5
             [ Literal "a"; Copy { kind = 1; offset = 1; length = 8 } ]));
    test "an element after the preamble's length is Malformed" (fun () ->
        malformed ~at:3 "output longer than the preamble"
          (block ~n:1 [ Literal "a"; Literal "b" ]));
    test "a preamble above the destination is Too_long at the preamble"
      (fun () ->
        fails ~cap:199 ~at:0 Too_long (block [ Literal (String.make 200 'a') ]));
  ]

let long_literals =
  (* A long literal's length follows its tag, low byte first, so each byte
     only adds to it: once the bytes so far take it past the preamble's
     room, the element is Malformed at its tag, at 1, even when the input
     ends inside its length. *)
  cases
    ~name:(fun (name, _) -> name)
    "a long literal past the preamble"
    [
      ("its 1-byte length", "\x03\xf0\x63");
      ("the first of 2 length bytes", "\x03\xf4\x63\x00");
      ("the first of 2 length bytes, cut", "\x03\xf4\x63");
      ("the second of 2 length bytes", "\x03\xf4\x01\x01");
      ("the fourth of 4 length bytes", "\x03\xfc\x01\x00\x00\x01");
    ]
    (fun (_, s) -> malformed ~at:1 "output longer than the preamble" s)

let tag_63 =
  (* Section 2.1: tag 63 holds the length minus one in 4 bytes after it,
     whatever the length. *)
  test "a literal under tag 63" (fun () ->
      equal decoded (Ok "hello") (decode ~cap:5 "\x05\xfc\x04\x00\x00\x00hello"))

let larger_destination =
  (* The preamble, not the destination, sets the output's length. *)
  test "a destination larger than the preamble" (fun () ->
      let s = block [ Literal "xab"; Copy { kind = 1; offset = 2; length = 4 } ] in
      let dst = create 64 in
      equal (result progress error)
        (Ok { Lz.consumed = String.length s; produced = 7 })
        (Lz.Snappy.decode (of_string s) dst);
      equal string "xababab" (to_string ~length:7 dst))

let after_block =
  (* A block has no end mark: bytes after it are read as more of it, so a
     literal after the preamble's length is refused at its tag. *)
  test "bytes after a block are read as more of it" (fun () ->
      let b = block [ Literal "hello" ] in
      malformed ~at:(String.length b) "output longer than the preamble"
        (b ^ "\x00x"))

let preambles_malformed =
  (* Section 1: a length is at most 2^32 - 1, so its varint at most 5
     bytes, the last holding 4 bits. *)
  cases ~name:String.escaped "preambles past 32 bits"
    [ "\x80\x80\x80\x80\x80\x00"; "\xff\xff\xff\xff\x10"; "\x80\x80\x80\x80\x80" ]
    (fun s -> malformed ~at:0 "invalid preamble" s)

let preamble_largest =
  test "a preamble of 2^32 - 1 is a length" (fun () ->
      fails ~cap:16 ~at:0 Too_long "\xff\xff\xff\xff\x0f\x00a")

let truncated =
  let copy4 = block [ Literal "abcd"; Copy { kind = 4; offset = 4; length = 4 } ] in
  cases
    ~name:(fun (name, _) -> name)
    "the input ends inside an element"
    [
      ("empty", "");
      ("inside the preamble", "\x80");
      ("before any element", "\x05");
      ("inside a literal", String.sub (block [ Literal "abcde" ]) 0 4);
      ("inside a long literal's length", "\xff\x01\xf4\x10");
      ("inside a copy's offset", String.sub copy4 0 8);
    ]
    (fun (_, s) -> fails ~cap:1024 ~at:(String.length s) Truncated s)

let offsets =
  cases
    ~name:(fun kind -> Printf.sprintf "zero in the %d-byte form" kind)
    "a copy offset of zero" [ 1; 2; 4 ]
    (fun kind ->
      (* The copy's tag is at 6. *)
      malformed ~at:6 "invalid copy offset"
        (block [ Literal "abcd"; Copy { kind; offset = 0; length = 4 } ]))

let before_start =
  (* Section 2.2: an offset past the output is not allowed. The offset
     starts from a 1-byte-offset tag's bits and its bytes follow, low byte
     first, each only adding: once the bytes so far take it past the
     output, the copy is Malformed at its tag, even when the input ends
     inside its offset. Section 2: a block cannot start with a copy. *)
  let far4 =
    block [ Literal "abcd"; Copy { kind = 4; offset = 0x7fffffff; length = 4 } ]
  in
  cases
    ~name:(fun (name, _, _) -> name)
    "a copy offset past the output"
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
    (fun (_, s, at) -> malformed ~at "copy offset too far back" s)

(* A 10 MiB block of a literal and copies of 64 bytes from 1 back: 220 MB
   of output, with a preamble of [n]. *)
let bomb n =
  let b = Buffer.create (10 lsl 20) in
  varint b n;
  add_literal b "a";
  for _ = 1 to (10 lsl 20) / 3 do
    add_copy b ~kind:2 ~offset:1 ~length:64
  done;
  Buffer.contents b

let bombs =
  let cap = 1 lsl 20 in
  let decode_into s =
    let buf = create (cap + 64) in
    let r = Lz.Snappy.decode (of_string s) (Bigarray.Array1.sub buf 0 cap) in
    (r, String.sub (to_string buf) cap 64)
  in
  [
    test "a bomb whose preamble exceeds the destination is Too_long"
      (fun () ->
        let r, past = decode_into (bomb 0xffff_ffff) in
        equal (result progress error) (Error { Lz.at = 0; why = Too_long }) r;
        equal string ~msg:"bytes past the destination" (String.make 64 '\xa5')
          past);
    test "a bomb that outgrows its preamble is Malformed where it does"
      (fun () ->
        (* The preamble of [cap] takes 3 bytes and the literal 2. The
           output passes [cap] at the [cap / 64]-th copy. *)
        let r, past = decode_into (bomb cap) in
        let at = 5 + (3 * ((cap / 64) - 1)) in
        let why = Lz.Malformed "output longer than the preamble" in
        equal (result progress error) (Error { Lz.at; why }) r;
        equal string ~msg:"bytes past the destination" (String.make 64 '\xa5')
          past);
  ]

(* Memory *)

(* The minor words [f ()] allocates, and the words its result holds. *)
let words f =
  let before = Gc.minor_words () in
  let r = Sys.opaque_identity (f ()) in
  let after = Gc.minor_words () in
  (int_of_float (after -. before), Obj.reachable_words (Obj.repr r))

let allocation =
  let g () =
    List.find (fun g -> g.file = "text_200000.g12.snappy") (goldens_of "snappy")
  in
  [
    test "a decode allocates nothing but its result" (fun () ->
        let g = g () in
        let src = of_string (Lazy.force g.data) in
        let dst = create (String.length (Lazy.force g.raw)) in
        let allocated, result = words (fun () -> Lz.Snappy.decode src dst) in
        equal int result allocated);
    test "an encode allocates nothing but its result" (fun () ->
        let src = of_string (Lazy.force (g ()).raw) in
        let dst = create (Lz.Snappy.bound (Bigarray.Array1.dim src)) in
        let allocated, result = words (fun () -> Lz.Snappy.encode src dst) in
        equal int result allocated);
  ]

let () =
  exit
    (run "Snappy"
       [
         group "arguments"
           [
             test "a negative size has no bound" (fun () ->
                 raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
                     Lz.Snappy.bound (-1)));
             test "a block holds at most 2^32 - 1 bytes" (fun () ->
                 (* Section 1: the preamble states at most 2^32 - 1. *)
                 equal int (32 + 0xffff_ffff + (0xffff_ffff / 6))
                   (Lz.Snappy.bound 0xffff_ffff);
                 raises_match (Exn.invalid_arg ~substring:"exceed") (fun () ->
                     Lz.Snappy.bound (1 lsl 32)));
           ];
         group "format"
           [
             xababab; preambles; literal_lengths; tag_63; far_offsets;
             larger_destination; after_block; law_blocks;
           ];
         group "round trips" [ law_round_trip; law_strings; empty; law_exact ];
         group "vendor" [ agreement "snappy" ];
         group "refusals"
           (preamble_lies
           @ [
               long_literals; preambles_malformed; preamble_largest; truncated;
               offsets;
             ]
           @ [ before_start ] @ bombs);
         group "memory" allocation;
       ])
