open Windtrap
open Bigarray

(* References, written from the specifications' definitions, one byte or
   one lane at a time: RFC 1952 section 8's bitwise CRC-32, RFC 1950
   section 2.2's Adler-32 sums, the xxHash specification's steps. *)

let m32 = 0xFFFFFFFF

let ref_crc32 crc s =
  let r = ref (Int32.to_int crc land m32 lxor m32) in
  String.iter
    (fun c ->
      r := !r lxor Char.code c;
      for _ = 1 to 8 do
        r := if !r land 1 = 1 then (!r lsr 1) lxor 0xEDB88320 else !r lsr 1
      done)
    s;
  Int32.of_int (!r lxor m32)

let ref_adler32 adler s =
  let a = Int32.to_int adler land m32 in
  let s1 = ref (a land 0xFFFF) and s2 = ref (a lsr 16) in
  String.iter
    (fun c ->
      s1 := (!s1 + Char.code c) mod 65521;
      s2 := (!s2 + !s1) mod 65521)
    s;
  Int32.of_int ((!s2 lsl 16) lor !s1)

let rotl32 x r = ((x lsl r) lor (x lsr (32 - r))) land m32
let mul32 a b = a * b land m32
let le32 s i = Int32.to_int (String.get_int32_le s i) land m32

let ref_xxh32 seed s =
  let p1 = 0x9E3779B1 and p2 = 0x85EBCA77 and p3 = 0xC2B2AE3D in
  let p4 = 0x27D4EB2F and p5 = 0x165667B1 in
  let seed = Int32.to_int seed land m32 and n = String.length s in
  let round acc lane = mul32 (rotl32 ((acc + mul32 lane p2) land m32) 13) p1 in
  let i = ref 0 and h = ref ((seed + p5) land m32) in
  if n >= 16 then begin
    let a =
      [|
        (seed + p1 + p2) land m32;
        (seed + p2) land m32;
        seed;
        (seed - p1) land m32;
      |]
    in
    while n - !i >= 16 do
      for k = 0 to 3 do
        a.(k) <- round a.(k) (le32 s (!i + (4 * k)))
      done;
      i := !i + 16
    done;
    h :=
      (rotl32 a.(0) 1 + rotl32 a.(1) 7 + rotl32 a.(2) 12 + rotl32 a.(3) 18)
      land m32
  end;
  h := (!h + n) land m32;
  while n - !i >= 4 do
    h := mul32 (rotl32 ((!h + mul32 (le32 s !i) p3) land m32) 17) p4;
    i := !i + 4
  done;
  while n - !i >= 1 do
    h := mul32 (rotl32 ((!h + mul32 (Char.code s.[!i]) p5) land m32) 11) p1;
    incr i
  done;
  let h = !h in
  let h = mul32 (h lxor (h lsr 15)) p2 in
  let h = mul32 (h lxor (h lsr 13)) p3 in
  Int32.of_int (h lxor (h lsr 16))

let ref_xxh64 seed s =
  let open Int64 in
  let p1 = 0x9E3779B185EBCA87L and p2 = 0xC2B2AE3D27D4EB4FL in
  let p3 = 0x165667B19E3779F9L and p4 = 0x85EBCA77C2B2AE63L in
  let p5 = 0x27D4EB2F165667C5L in
  let rotl x r = logor (shift_left x r) (shift_right_logical x (64 - r)) in
  let round acc lane = mul (rotl (add acc (mul lane p2)) 31) p1 in
  let merge h acc = add (mul (logxor h (round 0L acc)) p1) p4 in
  let n = String.length s in
  let i = ref 0 and h = ref (add seed p5) in
  if n >= 32 then begin
    let a = [| add seed (add p1 p2); add seed p2; seed; sub seed p1 |] in
    while n - !i >= 32 do
      for k = 0 to 3 do
        a.(k) <- round a.(k) (String.get_int64_le s (!i + (8 * k)))
      done;
      i := !i + 32
    done;
    h :=
      add (add (rotl a.(0) 1) (rotl a.(1) 7)) (add (rotl a.(2) 12) (rotl a.(3) 18));
    Array.iter (fun acc -> h := merge !h acc) a
  end;
  h := add !h (of_int n);
  while n - !i >= 8 do
    let lane = String.get_int64_le s !i in
    h := add (mul (rotl (logxor !h (round 0L lane)) 27) p1) p4;
    i := !i + 8
  done;
  if n - !i >= 4 then begin
    let lane = of_int (le32 s !i) in
    h := add (mul (rotl (logxor !h (mul lane p1)) 23) p2) p3;
    i := !i + 4
  end;
  while n - !i >= 1 do
    h := mul (rotl (logxor !h (mul (of_int (Char.code s.[!i])) p5)) 11) p1;
    incr i
  done;
  let h = !h in
  let h = mul (logxor h (shift_right_logical h 33)) p2 in
  let h = mul (logxor h (shift_right_logical h 29)) p3 in
  logxor h (shift_right_logical h 32)

(* The four checksums, each with its running value or seed as an int64, so
   that one table drives every law. *)

type sum = {
  sum : string;
  big : int64 -> Lz.bigbytes -> int64;
  str : int64 -> ?first:int -> ?length:int -> string -> int64;
  reference : int64 -> string -> int64;
  valid : int64 -> int64;
      (** A running value or seed of the checksum's width drawn from any. *)
}

let u32 x = Int64.logand (Int64.of_int32 x) 0xFFFFFFFFL
let i32 = Int64.to_int32

let width32 c = Int64.logand c 0xFFFFFFFFL

(* An Adler-32 value's halves are each below 65521. *)
let adler_valid c =
  let half x = Int64.rem (Int64.logand x 0xFFFFL) 65521L in
  Int64.logor (Int64.shift_left (half (Int64.shift_right_logical c 16)) 16) (half c)

let sums =
  [
    {
      sum = "CRC-32";
      big =
        (fun c b -> Int64.of_int (Lz.Crc32.bigbytes ~crc:(Int64.to_int c) b));
      str =
        (fun c ?first ?length s ->
          Int64.of_int (Lz.Crc32.string ~crc:(Int64.to_int c) ?first ?length s));
      reference = (fun c s -> u32 (ref_crc32 (i32 c) s));
      valid = width32;
    };
    {
      sum = "Adler-32";
      big = (fun c b -> u32 (Lz.Adler32.bigbytes ~adler:(i32 c) b));
      str =
        (fun c ?first ?length s ->
          u32 (Lz.Adler32.string ~adler:(i32 c) ?first ?length s));
      reference = (fun c s -> u32 (ref_adler32 (i32 c) s));
      valid = adler_valid;
    };
    {
      sum = "XXH32";
      big = (fun c b -> u32 (Lz.Xxh32.bigbytes ~seed:(i32 c) b));
      str =
        (fun c ?first ?length s ->
          u32 (Lz.Xxh32.string ~seed:(i32 c) ?first ?length s));
      reference = (fun c s -> u32 (ref_xxh32 (i32 c) s));
      valid = width32;
    };
    {
      sum = "XXH64";
      big = (fun c b -> Lz.Xxh64.bigbytes ~seed:c b);
      str = (fun c ?first ?length s -> Lz.Xxh64.string ~seed:c ?first ?length s);
      reference = ref_xxh64;
      valid = Fun.id;
    };
  ]

let sum_named n = List.find (fun s -> s.sum = n) sums

(* What the agreement laws run: each checksum through Lz, and each path of
   CRC-32 and Adler-32 this host can run, which Lz alone cannot reach: it
   takes only the fastest. *)

type run = {
  label : string;
  run : int64 -> Lz.bigbytes -> int64;
  expect : int64 -> string -> int64;
  draw : int64 -> int64;
}

let of_sum s = { label = s.sum; run = s.big; expect = s.reference; draw = s.valid }

let of_path name (p : Lz_support.path) =
  let s = sum_named name in
  {
    label = name ^ " " ^ p.name;
    run = (fun c b -> u32 (p.run (i32 c) b));
    expect = s.reference;
    draw = s.valid;
  }

let runs =
  List.map of_sum sums
  @ List.map (of_path "CRC-32") Lz_support.crc32_paths
  @ List.map (of_path "Adler-32") Lz_support.adler32_paths

let path_names paths = List.map (fun (p : Lz_support.path) -> p.name) paths

let hex =
  Testable.make ~pp:(fun ppf x -> Format.fprintf ppf "0x%Lx" x) ~equal:Int64.equal

(* Check values *)

(* CRC-32's check value is the CRC catalogue's ("CRC-32/ISO-HDLC",
   check = 0xCBF43926); Adler-32's is the worked example of "Wikipedia".
   The xxHash specification (0.2.0) lists no test vectors: the XXH32 and
   XXH64 rows are the digests of xxHash 0.8.3, the reference library,
   through its Python binding. The inputs cross a stripe, and leave tails of
   every kind the specification's step 5 reads. *)
let checks =
  [
    ("CRC-32", 0L, "123456789", 0xCBF43926L);
    ("CRC-32", 0L, "", 0L);
    ("Adler-32", 1L, "Wikipedia", 0x11E60398L);
    ("Adler-32", 1L, "", 1L);
    ("XXH32", 0L, "", 0x02CC5D05L);
    ("XXH32", 0L, "a", 0x550D7456L);
    ("XXH32", 0L, "abc", 0x32D153FFL);
    ("XXH32", 0L, "123456789", 0x937BAD67L);
    ("XXH32", 0L, "Nobody inspects the spammish repetition", 0xE2293B2FL);
    ("XXH32", 0L, "0123456789abcdef0123456789abcdef0123456789", 0x386F91D2L);
    ("XXH64", 0L, "", 0xEF46DB3751D8E999L);
    ("XXH64", 0L, "a", 0xD24EC4F1A98C6E5BL);
    ("XXH64", 0L, "abc", 0x44BC2CF5AD770999L);
    ("XXH64", 0L, "123456789", 0x8CB841DB40E6AE83L);
    ("XXH64", 0L, "Nobody inspects the spammish repetition", 0xFBCEA83C8A378BF1L);
    ("XXH64", 0L, "0123456789abcdef0123456789abcdef0123456789", 0xA76190C3ACF08A1CL);
  ]

let check_name (sum, c, s, _) = Printf.sprintf "%s 0x%Lx %S" sum c s

let check (name, c, s, expected) =
  let sum = sum_named name in
  equal hex expected (sum.str c s);
  equal hex expected (sum.big c (Lz_support.of_string s));
  equal ~msg:"reference" hex expected (sum.reference c s)

(* Drawn inputs *)

let bytes_of_seed seed n =
  let st = Random.State.make [| seed |] in
  String.init n (fun _ -> Char.chr (Random.State.int st 256))

(* Every path agrees with the reference at every length to 300 and every
   alignment to 15: the lengths cross each block and each tail of the
   paths, whose widest step is 256 bytes. *)
let short_lengths = 300
let alignments = 16

(* Each view, and each prefix's reference value, is made once a case and
   read by every run: the paths of a checksum share its reference, found
   by its function. Made per run, they cost more than the runs. *)
let every_length (seed, c) =
  let s = bytes_of_seed seed short_lengths in
  let views =
    Array.init alignments (fun offset ->
        let v = Lz_support.placed ~offset s in
        Array.init (short_lengths + 1) (fun n -> Array1.sub v 0 n))
  in
  let made = ref [] in
  let references expect c =
    let same (e, c', _) = e == expect && Int64.equal c c' in
    match List.find_opt same !made with
    | Some (_, _, values) -> values
    | None ->
        let values =
          Array.init (short_lengths + 1) (fun n -> expect c (String.sub s 0 n))
        in
        made := (expect, c, values) :: !made;
        values
  in
  List.iter
    (fun r ->
      let c = r.draw c in
      let expected = references r.expect c in
      for n = 0 to short_lengths do
        Array.iteri
          (fun a v ->
            let got = r.run c v.(n) in
            if not (Int64.equal got expected.(n)) then
              failf "%s: length %d at alignment %d: 0x%Lx, expected 0x%Lx"
                r.label n a got expected.(n))
          views
      done)
    runs

(* Lengths to 70000 at any alignment: across Adler-32's block of 5552 bytes
   and many turns of every loop. Each band is drawn half the time, so that
   every run of 100 cases covers both. *)
let long_input =
  Gen.(
    quad
      (one_of [ int_range (short_lengths + 1) 5552; int_range 5553 70000 ])
      (int_range 0 (alignments - 1))
      int int64)

let long_agrees (n, offset, seed, c) =
  cover "over 5552 bytes" (n > 5552);
  cover "from 301 to 5552 bytes" (n > short_lengths && n <= 5552);
  let s = bytes_of_seed seed n in
  let b = Lz_support.placed ~offset s in
  List.iter
    (fun r ->
      let c = r.draw c in
      equal ~msg:r.label hex (r.expect c s) (r.run c b))
    runs

(* Bytes of 255 from the largest running value, where Adler-32's sums
   grow fastest: at the edges of its 5552-byte reduction and of the vector
   paths' runs of 5504 bytes, and over many of them. *)
let saturated = [ 5503; 5504; 5505; 5551; 5552; 5553; 11008; 11009; 70001 ]

let saturating n =
  let s = String.make n '\xff' in
  let b = Lz_support.of_string s in
  List.iter
    (fun r ->
      let c = r.draw 0xFFF0FFF0L in
      equal ~msg:r.label hex (r.expect c s) (r.run c b))
    runs

(* 8 MiB, past the size from which the bigarray entries release the
   runtime lock. *)
let eight_mib = 8 lsl 20

let large () =
  let s = Lz_support.corpus Random eight_mib in
  let b = Lz_support.of_string s in
  List.iter
    (fun r ->
      let c = r.draw 0x0123456789ABCDEFL in
      equal ~msg:r.label hex (r.expect c s) (r.run c b))
    runs

(* CRC-32 and Adler-32 continue a running value: the checksum of [a ^ b] is
   the checksum of [b] from the checksum of [a]. *)
let pair =
  let s = Gen.(string_of ~size:(int_range 0 600) char) in
  Gen.pair s s

let running (a, b) =
  List.iter
    (fun name ->
      let sum = sum_named name in
      let start = if name = "Adler-32" then 1L else 0L in
      let whole = sum.str start (a ^ b) in
      equal ~msg:name hex whole (sum.str (sum.str start a) b);
      let ba = Lz_support.of_string a and bb = Lz_support.of_string b in
      equal ~msg:name hex whole (sum.big (sum.big start ba) bb))
    [ "CRC-32"; "Adler-32" ]

(* [string ~first ~length] is [bigbytes] of the substring. *)
let range_input =
  Gen.(
    let* s = string_of ~size:(int_range 0 400) char in
    let n = String.length s in
    let* first = int_range 0 n in
    let* length = int_range 0 (n - first) in
    let+ c = int64 in
    (s, first, length, c))

let substring (s, first, length, c) =
  List.iter
    (fun sum ->
      let c = sum.valid c in
      let sub = sum.big c (Lz_support.of_string (String.sub s first length)) in
      equal ~msg:sum.sum hex sub (sum.str c ~first ~length s);
      if first + length = String.length s then
        equal ~msg:sum.sum hex sub (sum.str c ~first s))
    sums

(* A range outside the string, for each checksum's [string]. *)
let ranges = [ (-1, None); (0, Some 4); (4, Some 0); (2, Some (-1)) ]

let name (first, length) =
  Printf.sprintf "first %d, length %s" first
    (match length with Some l -> string_of_int l | None -> "none")

let () =
  exit
    (run "Checksum"
       [
         cases ~name:check_name "check values" checks check;
         test "the generic paths come first" (fun () ->
             equal string "generic"
               (List.hd (path_names Lz_support.crc32_paths));
             equal string "generic"
               (List.hd (path_names Lz_support.adler32_paths)));
         prop "every length to 300 at every alignment agrees with the reference"
           Gen.(pair int int64)
           every_length;
         prop "long inputs agree with the reference" long_input long_agrees;
         cases ~name:(Printf.sprintf "%d bytes of 255") "bytes of 255 agree with the reference"
           saturated saturating;
         test "8 MiB agrees with the reference" large;
         prop "a running value continues the checksum" pair running;
         prop "string ~first ~length is bigbytes of the substring" range_input
           substring;
         cases ~name "a range outside the string raises" ranges
           (fun (first, length) ->
             let s = "abc" in
             let bad f = raises_match (Exn.invalid_arg ~substring:"range") f in
             bad (fun () -> Lz.Crc32.string ~first ?length s);
             bad (fun () -> Lz.Adler32.string ~first ?length s);
             bad (fun () -> Lz.Xxh32.string ~first ?length s);
             bad (fun () -> Lz.Xxh64.string ~first ?length s));
         cases ~name:string_of_int "a CRC-32 outside [0, 2^32) raises"
           [ -1; 1 lsl 32; min_int; max_int ]
           (fun crc ->
             let bad f = raises_match (Exn.invalid_arg ~substring:"crc") f in
             bad (fun () -> Lz.Crc32.string ~crc "abc");
             bad (fun () -> Lz.Crc32.bigbytes ~crc (Lz_support.of_string "abc")));
       ])
