(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Marks *)

(* A float field's number carries one of these metas, told apart by physical
   equality as jsont tells [Meta.none]. [Sys.opaque_identity] keeps the
   compiler from building them as one shared constant. *)
let f64_meta = Jsont.Meta.make (Sys.opaque_identity Jsont.Textloc.none)
let f32_meta = Jsont.Meta.make (Sys.opaque_identity Jsont.Textloc.none)
let f64 x = Jsont.Json.number ~meta:f64_meta x
let single x = Int32.float_of_bits (Int32.bits_of_float x)
let f32 x = Jsont.Json.number ~meta:f32_meta (single x)

(* Numbers *)

(* serde_json's layout of the shortest digits, from its zmij writer, the
   same as ryu's but for the exponent's [+]. With [d] the [n] digits and
   [kk = n + e] the exponent of the decimal point:
   - [0 <= e] and [kk <= wide]: [d], [e] zeros, [.0] ([12340000000.0]);
   - [0 < kk <= wide]: [d] with a point after [kk] digits ([12.34]);
   - [low < kk <= 0]: [0.], [-kk] zeros, [d] ([0.001234]);
   - otherwise [d0], [.] and the rest of [d] if any, [e], the sign of
     [kk - 1] and its digits ([1.5e-7], [1e+16]).
   [wide] and [low] are 16 and -5 for [f64], 13 and -6 for [f32]. Zero is
   [("0", 0)], which the first case writes [0.0]. *)
let add_float b ~wide ~low x (d, e) =
  let n = String.length d in
  let kk = n + e in
  if Float.sign_bit x then Buffer.add_char b '-';
  if 0 <= e && kk <= wide then begin
    Buffer.add_string b d;
    for _ = 1 to e do
      Buffer.add_char b '0'
    done;
    Buffer.add_string b ".0"
  end
  else if 0 < kk && kk <= wide then begin
    Buffer.add_substring b d 0 kk;
    Buffer.add_char b '.';
    Buffer.add_substring b d kk (n - kk)
  end
  else if low < kk && kk <= 0 then begin
    Buffer.add_string b "0.";
    for _ = 1 to -kk do
      Buffer.add_char b '0'
    done;
    Buffer.add_string b d
  end
  else begin
    Buffer.add_char b d.[0];
    if n > 1 then begin
      Buffer.add_char b '.';
      Buffer.add_substring b d 1 (n - 1)
    end;
    Buffer.add_string b (if kk >= 1 then "e+" else "e-");
    Buffer.add_string b (Int.to_string (abs (kk - 1)))
  end

let exact_int = 0x1p53

let add_number b x meta =
  if not (Float.is_finite x) then Buffer.add_string b "null"
  else if meta == f32_meta then
    add_float b ~wide:13 ~low:(-6) x
      (Float_text.shortest Float_text.binary32 x)
  else if meta != f64_meta && Float.is_integer x && Float.abs x < exact_int
  then begin
    (* Below 2^53 an integer's interval is at most 1 wide and holds no other
       decimal of as few digits, so its shortest digits are its own without
       their trailing zeros, and [e >= 0]. *)
    let d, e = Float_text.shortest Float_text.binary64 x in
    if Float.sign_bit x then Buffer.add_char b '-';
    Buffer.add_string b d;
    for _ = 1 to e do
      Buffer.add_char b '0'
    done
  end
  else
    add_float b ~wide:16 ~low:(-5) x
      (Float_text.shortest Float_text.binary64 x)

(* Strings, escaped as jsont's encoder escapes them. *)
let add_string b s =
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | ('\x00' .. '\x1F' | '\x7F') as c ->
          Buffer.add_string b (Printf.sprintf "\\u%04X" (Char.code c))
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"'

let rec add b = function
  | Jsont.Null _ -> Buffer.add_string b "null"
  | Jsont.Bool (v, _) -> Buffer.add_string b (if v then "true" else "false")
  | Jsont.Number (x, meta) -> add_number b x meta
  | Jsont.String (s, _) -> add_string b s
  | Jsont.Array (l, _) ->
      Buffer.add_char b '[';
      List.iteri
        (fun i v ->
          if i > 0 then Buffer.add_char b ',';
          add b v)
        l;
      Buffer.add_char b ']'
  | Jsont.Object (mems, _) ->
      Buffer.add_char b '{';
      List.iteri
        (fun i ((name, _), v) ->
          if i > 0 then Buffer.add_char b ',';
          add_string b name;
          Buffer.add_char b ':';
          add b v)
        mems;
      Buffer.add_char b '}'

let to_string j =
  let b = Buffer.create 4096 in
  add b j;
  Buffer.contents b
