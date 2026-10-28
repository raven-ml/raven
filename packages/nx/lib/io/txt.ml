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

(* Correct rounding

   A literal is read here rather than by [float_of_string], whose C strtod is
   not correctly rounded on every system. Its leading digits, scaled by exact
   powers of ten, make a candidate a few floats from the nearest, which [settle]
   moves a float at a time until the literal's exact value lies in the
   candidate's rounding interval: between the midpoints to its neighbours, each
   included when the candidate's significand is even. The exact values compare
   as integers in the limbs above. *)

(* A literal keeps this many significant digits, and a [1] after them in place
   of the others where they are not all zero. A midpoint's decimal expansion has
   at most 768 significant digits, so a value and its shortened form lie on the
   same side of each midpoint. *)
let max_digits = 800

(* An exponent is read up to this bound, which no string's length reaches, so
   that a larger one still makes a value that rounds to zero or infinity. *)
let max_exponent = 1 lsl 58

(* [decimal s] is the magnitude of the float literal [s] as [Some (d, e)], the
   value [d 10^e] for the digits [d] without leading zeros, or [None] for [nan]
   and the infinities. *)
let decimal s =
  let n = String.length s and i = signed s in
  if i < n && not (is_digit s.[i] || s.[i] = '.') then None
  else
    let d = Buffer.create 32 in
    let e = ref 0 and point = ref false and sticky = ref false and j = ref i in
    while !j < n && (is_digit s.[!j] || s.[!j] = '.') do
      (match s.[!j] with
      | '.' -> point := true
      | '0' when Buffer.length d = 0 -> if !point then decr e
      | c when Buffer.length d < max_digits ->
          Buffer.add_char d c;
          if !point then decr e
      | c ->
          if not !point then incr e;
          if c <> '0' then sticky := true);
      incr j
    done;
    if !sticky then (
      Buffer.add_char d '1';
      decr e);
    if !j < n then begin
      let k = !j + 1 + signed (String.sub s (!j + 1) (n - !j - 1)) in
      let x = ref 0 in
      for p = k to n - 1 do
        x := Int.min max_exponent ((10 * !x) + Char.code s.[p] - 48)
      done;
      e := if s.[!j + 1] = '-' then !e - !x else !e + !x
    end;
    Some (Buffer.contents d, !e)

let pow10 k = int_of_string ("1" ^ String.make k '0')

(* [shift10 a k] is [a 10^k] and [shift2 a k] is [a 2^k], for [k >= 0]. *)
let shift10 a k =
  mul_small (Array.append (Array.make (k / 9) 0) a) (pow10 (k mod 9))

let rec shift2 a k =
  if k > 20 then shift2 (mul_small a (1 lsl 20)) (k - 20)
  else mul_small a (1 lsl k)

(* The limbs of the decimal digits [d]. *)
let limbs d =
  let n = String.length d in
  Array.init
    ((n + 8) / 9)
    (fun i ->
      let stop = n - (9 * i) in
      let start = Int.max 0 (stop - 9) in
      int_of_string (String.sub d start (stop - start)))

let compare_limbs a b =
  let n = Array.length a in
  if n <> Array.length b then Int.compare n (Array.length b)
  else
    let rec go i =
      if i < 0 then 0
      else if a.(i) <> b.(i) then Int.compare a.(i) b.(i)
      else go (i - 1)
    in
    go (n - 1)

(* [compare_exact (d, e) (m, k)] compares [d 10^e], [d] as limbs, with [m
   2^k]. *)
let compare_exact (d, e) (m, k) =
  let lhs = shift2 (shift10 d (Int.max e 0)) (Int.max (-k) 0) in
  let rhs =
    shift10 (shift2 (limbs (string_of_int m)) (Int.max k 0)) (-Int.min e 0)
  in
  compare_limbs lhs rhs

(* The overflow threshold, the largest finite float64 and half its unit in the
   last place: [(2^54 - 1) 2^970], which rounds to infinity. *)
let threshold = ((1 lsl 54) - 1, 970)

(* [settle v c] is the float64 nearest the positive value [v], from the
   candidate [c >= 0]. The candidate [n 2^u], [n] its significand, lies between
   the midpoints [(2n - 1) 2^(u - 1)] and [(2n + 1) 2^(u - 1)], the lower one
   [(4n - 1) 2^(u - 2)] at the bottom of a binade above the subnormals, where
   the unit below is half. *)
let rec settle v c =
  if c = Float.infinity then
    if compare_exact v threshold < 0 then settle v Float.max_float else c
  else
    let u =
      if c = 0. then -1074 else Int.max (snd (Float.frexp c) - 53) (-1074)
    in
    let n = Float.to_int (Float.ldexp c (-u)) in
    let even = n land 1 = 0 in
    let low =
      if n = 1 lsl 52 && u > -1074 then ((4 * n) - 1, u - 2)
      else ((2 * n) - 1, u - 1)
    in
    let l = if n = 0 then 1 else compare_exact v low in
    if l < 0 || (l = 0 && not even) then settle v (Float.pred c)
    else
      let h = compare_exact v ((2 * n) + 1, u - 1) in
      if h > 0 || (h = 0 && not even) then settle v (Float.succ c) else c

(* The powers of ten a float64 holds exactly: [tens.(k)] is [10^k]. *)
let tens =
  let t = Array.make 23 1. in
  for k = 1 to 22 do
    t.(k) <- 10. *. t.(k - 1)
  done;
  t

(* [scale x k] is [x 10^k], rounded once for each power of ten it applies. *)
let rec scale x k =
  if k > 22 then scale (x *. tens.(22)) (k - 22)
  else if k < -22 then scale (x /. tens.(22)) (k + 22)
  else if k >= 0 then x *. tens.(k)
  else x /. tens.(-k)

(* [candidate d e] is a float64 near [d 10^e]: the first 18 digits of [d], which
   an [int] holds, scaled by powers of ten. Its 17 roundings at most leave it
   within a few floats of the nearest. *)
let candidate d e =
  let n = String.length d in
  let k = Int.min 18 n in
  scale (Float.of_int (int_of_string (String.sub d 0 k))) (e + n - k)

(* [nearest s] is the float64 nearest the value of the float literal [s], a tie
   to the even one. A value whose leading digit lies past [10^310] rounds to
   infinity, and one below [10^-330] to zero, which spares their exact
   comparison. *)
let nearest s =
  let x =
    match decimal s with
    | None ->
        if Char.lowercase_ascii s.[signed s] = 'n' then Float.nan
        else Float.infinity
    | Some ("", _) -> 0.
    | Some (d, e) ->
        let lead = String.length d - 1 + e in
        if lead < -330 then 0.
        else if lead > 310 then Float.infinity
        else settle (limbs d, e) (candidate d e)
  in
  if s.[0] = '-' then Float.neg x else x

let parser (type v s) (dt : (v, s) D.t) : (string -> v parsed) option =
  let float s = if is_float s then Value (nearest s) else Not_literal in
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
          let digits = String.sub s (signed s) (String.length s - signed s) in
          if not (is_integer s) then Not_literal
          else if s.[0] = '-' then
            if String.for_all (Char.equal '0') digits then Value 0L
            else Out_of_range
          else
            match Int64.of_string_opt ("0u" ^ digits) with
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

(* The separator of which a run separates two fields, so that columns aligned
   with spaces read as one field each. *)
let space = " "

(* [fields sep line] is [line]'s fields: the pieces between [sep]s, each trimmed
   of blanks, empty ones included unless [sep] is [space]. A line that is blank
   once trimmed has none. The line is split before any trimming, so that a blank
   [sep] such as a tab keeps an empty first or last field. *)
let fields sep line =
  let n = String.length sep and len = String.length line in
  let at i =
    let rec go k = k = n || (line.[i + k] = sep.[k] && go (k + 1)) in
    go 0
  in
  let rec split acc start i =
    if i > len - n then String.sub line start (len - start) :: acc
    else if at i then
      split (String.sub line start (i - start) :: acc) (i + n) (i + n)
    else split acc start (i + 1)
  in
  if String.trim line = "" then [||]
  else if sep = "" then [| String.trim line |]
  else
    let pieces = List.rev_map String.trim (split [] 0 0) in
    let pieces =
      if sep = space then List.filter (fun f -> f <> "") pieces else pieces
    in
    Array.of_list pieces

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
  (* rig.disk refuses a path that names no regular file, such as a FIFO, which
     an open for reading would wait on for a writer. *)
  let file = Storage.open_file path in
  let text = Storage.string file ~pos:0 (Rig.Buffer.length file) in
  (* Each row with the number of its line, counted from 1. *)
  let rec read line skip rows = function
    | [] -> rows
    | _ :: lines when skip > 0 -> read (line + 1) (skip - 1) rows lines
    | l :: lines when is_comment l -> read (line + 1) 0 rows lines
    | l :: lines ->
        let f = fields sep l in
        let rows = if Array.length f = 0 then rows else (line, f) :: rows in
        read (line + 1) 0 rows lines
  in
  let rows =
    Array.of_list
      (List.rev (read 1 skiprows [] (String.split_on_char '\n' text)))
  in
  if Array.length rows = 0 then failwith "no row";
  let first, row = rows.(0) in
  let cols = Array.length row in
  Array.iter
    (fun (line, r) ->
      if Array.length r <> cols then
        Fail.failf "line %d has %d fields, line %d has %d" line (Array.length r)
          first cols)
    rows;
  let field i j =
    let line, r = rows.(i) in
    let s = r.(j) in
    match parse s with
    | Value v -> v
    | Not_literal ->
        Fail.failf "line %d: %S is not a %s literal" line s (D.name dt)
    | Out_of_range ->
        Fail.failf "line %d: %s lies outside %s's range" line s (D.name dt)
  in
  let n = Array.length rows in
  let values = Array.init (n * cols) (fun k -> field (k / cols) (k mod cols)) in
  let shape =
    if n = 1 then [| cols |] else if cols = 1 then [| n |] else [| n; cols |]
  in
  Nx.create dt shape values
