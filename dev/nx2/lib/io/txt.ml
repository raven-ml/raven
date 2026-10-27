(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype

(* Exact floats

   A C runtime only has to round [%e] correctly up to 17 significant digits;
   numpy's default of 19 needs the exact binary value, so the digits come from
   the significand and exponent here and the output is the same on every
   platform. Little-endian base-1e9 limbs hold the exact integer. *)

let limb_base = 1_000_000_000

let mul_small limbs n =
  let carry = ref 0 in
  let limbs =
    Array.map
      (fun l ->
        let v = (l * n) + !carry in
        carry := v / limb_base;
        v mod limb_base)
      limbs
  in
  if !carry > 0 then Array.append limbs [| !carry |] else limbs

let digits_of_limbs limbs =
  let n = Array.length limbs in
  let b = Buffer.create (9 * n) in
  Buffer.add_string b (string_of_int limbs.(n - 1));
  for i = n - 2 downto 0 do
    Buffer.add_string b (Printf.sprintf "%09d" limbs.(i))
  done;
  Buffer.contents b

(* [exact_decimal x] is the decimal digits [d] of the finite, nonzero [|x|] and
   the count [k] of those digits after the decimal point: [|x| = d * 10^-k]. [x
   = f * 2^e] for the 53-bit integer [f]; a negative [e] turns [2^e] into [5^-e
   / 10^-e]. *)
let exact_decimal x =
  let m, e = Float.frexp x in
  let f = Int64.of_float (Float.ldexp (Float.abs m) 53) in
  let e = e - 53 in
  let lo = Int64.to_int (Int64.rem f 1_000_000_000L) in
  let hi = Int64.to_int (Int64.div f 1_000_000_000L) in
  let limbs = ref (if hi = 0 then [| lo |] else [| lo; hi |]) in
  if e >= 0 then begin
    for _ = 1 to e do
      limbs := mul_small !limbs 2
    done;
    (digits_of_limbs !limbs, 0)
  end
  else begin
    for _ = 1 to -e do
      limbs := mul_small !limbs 5
    done;
    (digits_of_limbs !limbs, -e)
  end

(* [%.<precision>e] of [x], rounded to nearest with ties to even, with at least
   two exponent digits. NaN prints as [nan] whatever its sign. *)
let format_e ~precision x =
  let sign = if Float.sign_bit x then "-" else "" in
  if Float.is_nan x then "nan"
  else if Float.abs x = Float.infinity then sign ^ "inf"
  else if x = 0. then sign ^ "0." ^ String.make precision '0' ^ "e+00"
  else
    let digits, point = exact_decimal x in
    let len = String.length digits in
    let keep = precision + 1 in
    let kept, carried =
      if len <= keep then (digits ^ String.make (keep - len) '0', false)
      else
        let head = Bytes.of_string (String.sub digits 0 keep) in
        let next = digits.[keep] in
        let rest_nonzero = ref false in
        for i = keep + 1 to len - 1 do
          if digits.[i] <> '0' then rest_nonzero := true
        done;
        let last_odd =
          (Char.code (Bytes.get head (keep - 1)) - 48) land 1 = 1
        in
        if next < '5' || (next = '5' && not (!rest_nonzero || last_odd)) then
          (Bytes.to_string head, false)
        else
          let i = ref (keep - 1) in
          while !i >= 0 && Bytes.get head !i = '9' do
            Bytes.set head !i '0';
            decr i
          done;
          if !i < 0 then ("1" ^ Bytes.sub_string head 0 (keep - 1), true)
          else begin
            Bytes.set head !i (Char.chr (Char.code (Bytes.get head !i) + 1));
            (Bytes.to_string head, false)
          end
    in
    let exponent = len - point - 1 + if carried then 1 else 0 in
    Printf.sprintf "%s%c.%se%c%02d" sign kept.[0]
      (String.sub kept 1 precision)
      (if exponent < 0 then '-' else '+')
      (abs exponent)

(* Fields *)

(* [printer dt] writes a value of [dt] as a field, and [parser dt] reads one.
   Both are [None] for a complex [dt], which text does not hold. *)

let printer (type v s) (dt : (v, s) D.t) : (Buffer.t -> v -> unit) option =
  let int b v = Buffer.add_string b (string_of_int v) in
  let bool b v = Buffer.add_char b (if v then '1' else '0') in
  let float b v = Buffer.add_string b (format_e ~precision:18 v) in
  match dt with
  | D.Float64 -> Some float
  | D.Float32 -> Some float
  | D.Float16 -> Some float
  | D.Bfloat16 -> Some float
  | D.Float8_e4m3fn -> Some float
  | D.Float8_e5m2 -> Some float
  | D.Float4_e2m1fn -> Some float
  | D.Int64 -> Some (fun b v -> Buffer.add_string b (Int64.to_string v))
  | D.Uint64 -> Some (fun b v -> Printf.bprintf b "%Lu" v)
  | D.Int32 -> Some (fun b v -> Buffer.add_string b (Int32.to_string v))
  | D.Uint32 -> Some (fun b v -> Printf.bprintf b "%lu" v)
  | D.Int16 -> Some int
  | D.Uint16 -> Some int
  | D.Int8 -> Some int
  | D.Uint8 -> Some int
  | D.Int4 -> Some int
  | D.Uint4 -> Some int
  | D.Bool -> Some bool
  | D.Bit -> Some bool
  | D.Complex128 | D.Complex64 -> None

type 'v parsed = Value of 'v | Not_literal | Out_of_range

(* The grammar of fields, checked before OCaml's readers see them, which also
   take hexadecimal and underscores, and [nan(...)] through C's strtod. *)

let is_digit = function '0' .. '9' -> true | _ -> false

(* [digits s i] is the index past the digits of [s] from [i]. *)
let rec digits s i =
  if i < String.length s && is_digit s.[i] then digits s (i + 1) else i

let signed s = if s <> "" && (s.[0] = '-' || s.[0] = '+') then 1 else 0

(* An integer field: a sign, then decimal digits. One that does not fit an
   [int64] lies outside the range of every integer type. *)
let is_integer s =
  let i = signed s in
  let j = digits s i in
  j > i && j = String.length s

(* A float field: a sign, then decimal digits with a point and an exponent, or
   [nan], [inf] or [infinity] in any case. *)
let is_float s =
  let i = signed s in
  let word = String.lowercase_ascii (String.sub s i (String.length s - i)) in
  word = "nan" || word = "inf" || word = "infinity"
  ||
  let j = digits s i in
  let k = if j < String.length s && s.[j] = '.' then digits s (j + 1) else j in
  let mantissa = (k - i - if k > j then 1 else 0) > 0 in
  let e =
    if k < String.length s && (s.[k] = 'e' || s.[k] = 'E') then
      let m = k + 1 + signed (String.sub s (k + 1) (String.length s - k - 1)) in
      let n = digits s m in
      if n > m then n else -1
    else k
  in
  mantissa && e = String.length s

(* An integer field, read as an [int64] in [\[lo, hi\]]. *)
let int64 ~lo ~hi s =
  if not (is_integer s) then Not_literal
  else
    match Int64.of_string_opt s with
    | None -> Out_of_range
    | Some v when Int64.compare v lo < 0 || Int64.compare v hi > 0 ->
        Out_of_range
    | Some v -> Value v

let map f = function
  | Value v -> Value (f v)
  | (Not_literal | Out_of_range) as r -> r

(* A field of a dtype read as [int], in its range. *)
let small (type s) (dt : (int, s) D.t) s =
  let lo = Int64.of_int (D.min_value dt)
  and hi = Int64.of_int (D.max_value dt) in
  map Int64.to_int (int64 ~lo ~hi s)

let parser (type v s) (dt : (v, s) D.t) : (string -> v parsed) option =
  let float s = if is_float s then Value (float_of_string s) else Not_literal in
  let bool s : bool parsed =
    match String.lowercase_ascii s with
    | "true" -> Value true
    | "false" -> Value false
    | s -> map (fun f -> f <> 0.) (float s)
  in
  match dt with
  | D.Float64 -> Some float
  | D.Float32 -> Some float
  | D.Float16 -> Some float
  | D.Bfloat16 -> Some float
  | D.Float8_e4m3fn -> Some float
  | D.Float8_e5m2 -> Some float
  | D.Float4_e2m1fn -> Some float
  | D.Int64 -> Some (int64 ~lo:Int64.min_int ~hi:Int64.max_int)
  | D.Uint64 ->
      Some
        (fun s ->
          if not (is_integer s) then Not_literal
          else if s.[0] = '-' then Out_of_range
          else
            let d =
              if s.[0] = '+' then String.sub s 1 (String.length s - 1) else s
            in
            match Int64.of_string_opt ("0u" ^ d) with
            | Some v -> Value v
            | None -> Out_of_range)
  | D.Int32 ->
      let lo = Int64.of_int32 Int32.min_int
      and hi = Int64.of_int32 Int32.max_int in
      Some (fun s -> map Int64.to_int32 (int64 ~lo ~hi s))
  | D.Uint32 ->
      Some (fun s -> map Int64.to_int32 (int64 ~lo:0L ~hi:0xFFFF_FFFFL s))
  | D.Int16 -> Some (small dt)
  | D.Uint16 -> Some (small dt)
  | D.Int8 -> Some (small dt)
  | D.Uint8 -> Some (small dt)
  | D.Int4 -> Some (small dt)
  | D.Uint4 -> Some (small dt)
  | D.Bool -> Some bool
  | D.Bit -> Some bool
  | D.Complex128 | D.Complex64 -> None

let refuse ~by dt = Fail.invalid_argf "%s: text has no %s" by (D.name dt)

(* [fields sep line] is [line]'s fields: the non-empty pieces between [sep]s of
   the line without its surrounding blanks. *)
let fields sep line =
  let s = String.trim line in
  let n = String.length sep and len = String.length s in
  let at i =
    let rec go k = k = n || (s.[i + k] = sep.[k] && go (k + 1)) in
    go 0
  in
  let rec split acc start i =
    if i > len - n then String.sub s start (len - start) :: acc
    else if at i then
      split (String.sub s start (i - start) :: acc) (i + n) (i + n)
    else split acc start (i + 1)
  in
  if s = "" then [||]
  else if sep = "" then [| s |]
  else
    List.rev (split [] 0 0) |> List.filter (fun f -> f <> "") |> Array.of_list

(* Saving *)

(* The text is written to the file in chunks of about [chunk] bytes. *)
let chunk = 65536

let save ~sep path x =
  let by = "Nx_io.save_txt" in
  let dt = Nx.dtype x in
  let print = match printer dt with Some p -> p | None -> refuse ~by dt in
  let shape = Nx.shape x in
  if Array.length shape > 2 then
    Fail.invalid_argf "%s: %d axes, text holds at most 2" by
      (Array.length shape);
  let values = Fail.renamed ~by Nx.to_array x in
  let cols = match shape with [| _; c |] -> c | [| c |] -> c | _ -> 1 in
  Save.write path @@ fun fd ->
  let b = Buffer.create chunk in
  let drain () =
    Save.string fd (Buffer.contents b);
    Buffer.clear b
  in
  Array.iteri
    (fun i v ->
      if i mod cols <> 0 then Buffer.add_string b sep;
      print b v;
      if (i + 1) mod cols = 0 then Buffer.add_char b '\n';
      if Buffer.length b >= chunk then drain ())
    values;
  drain ()

(* Loading *)

let load ~sep ~comments ~skiprows dt path =
  let by = "Nx_io.load_txt" in
  let parse = match parser dt with Some p -> p | None -> refuse ~by dt in
  if skiprows < 0 then Fail.invalid_argf "%s: skiprows is %d" by skiprows;
  Fail.file path @@ fun () ->
  let comment = String.trim comments in
  let is_comment line =
    comment <> "" && String.starts_with ~prefix:comment (String.trim line)
  in
  let rows =
    In_channel.with_open_bin path @@ fun ic ->
    let rec read skip rows =
      match In_channel.input_line ic with
      | None -> rows
      | Some _ when skip > 0 -> read (skip - 1) rows
      | Some line when is_comment line -> read 0 rows
      | Some line ->
          let f = fields sep line in
          if Array.length f = 0 then read 0 rows else read 0 (f :: rows)
    in
    Array.of_list (List.rev (read skiprows []))
  in
  if Array.length rows = 0 then failwith "no row";
  let cols = Array.length rows.(0) in
  Array.iteri
    (fun i r ->
      if Array.length r <> cols then
        Fail.failf "row %d has %d fields, row 1 has %d" (i + 1) (Array.length r)
          cols)
    rows;
  let field i j =
    let s = rows.(i).(j) in
    match parse s with
    | Value v -> v
    | Not_literal -> Fail.failf "%S is no %s literal" s (D.name dt)
    | Out_of_range -> Fail.failf "%s lies outside %s's range" s (D.name dt)
  in
  let n = Array.length rows in
  let values = Array.init (n * cols) (fun k -> field (k / cols) (k mod cols)) in
  let shape =
    if n = 1 then [| cols |] else if cols = 1 then [| n |] else [| n; cols |]
  in
  Nx.create dt shape values
