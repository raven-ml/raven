(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* whisper/normalizers/english.py and basic.py at 86098128. Each step is one of
   upstream's regular expressions over a Python str, read as Python's [re] reads
   it: [\w] is a letter, a number or [_], [\d] a decimal digit of any script,
   [\s] Python's white space, and a substitution scans left to right, going on
   after each match. Text is handled as an array of code points. *)

(* Text *)

let code_points s =
  let n = String.length s in
  let rec go i acc =
    if i >= n then Array.of_list (List.rev acc)
    else
      let d = String.get_utf_8_uchar s i in
      go
        (i + Uchar.utf_decode_length d)
        (Uchar.to_int (Uchar.utf_decode_uchar d) :: acc)
  in
  go 0 []

let utf8 c =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int c);
  Buffer.contents b

(* [rewrite f s] is the text [f a b] writes to [b] for [s]'s code points [a]. *)
let rewrite f s =
  let b = Buffer.create (String.length s) in
  f (code_points s) b;
  Buffer.contents b

let add b c = Buffer.add_utf_8_uchar b (Uchar.of_int c)
let add_all b a = Array.iter (add b) a
let chars s = code_points s

(* Character classes. Beyond ASCII, a class is the set brot's regular
   expressions give it. *)

let in_class pattern =
  let n = Brot.Normalizer.replace_regex ~pattern ~replacement:"" in
  fun c -> Brot.Normalizer.apply n (utf8 c) = ""

let ascii_digit c = c >= 0x30 && c <= 0x39
let ascii_lower c = c >= 0x61 && c <= 0x7A
let unicode_digit = in_class {|\p{Nd}|}
let unicode_word = in_class {|[\p{L}\p{N}_]|}
let is_digit c = if c < 0x80 then ascii_digit c else unicode_digit c

let is_word c =
  if c < 0x80 then
    ascii_digit c || ascii_lower c || (c >= 0x41 && c <= 0x5A) || c = 0x5F
  else unicode_word c

let is_space c = Space.is_space (Uchar.of_int c)

(* A decimal digit's value: decimal digits come in runs of ten, from zero to
   nine, so it is its distance to the start of its run, modulo ten. *)
let digit_value c =
  let rec start c = if c > 0 && is_digit (c - 1) then start (c - 1) else c in
  (c - start c) mod 10

(* Python's [\b] at position [i] of [a]. *)
let boundary a i =
  let word j = j >= 0 && j < Array.length a && is_word a.(j) in
  word (i - 1) <> word i

let matches a i p =
  let n = Array.length p in
  i + n <= Array.length a
  &&
  let rec go k = k = n || (a.(i + k) = p.(k) && go (k + 1)) in
  go 0

(* [i]'s match of the first of [alternatives] that holds there, [\b] before them
   where [left] and after them where [right], as its end. *)
let match_words ~left ~right alternatives a i =
  if left && not (boundary a i) then None
  else
    List.find_map
      (fun p ->
        let j = i + Array.length p in
        if matches a i p && ((not right) || boundary a j) then Some j else None)
      alternatives

(* Python's [re.sub] of a pattern matching at [i] up to [Some j], replaced by
   [replace b a i j]. *)
let substitute matcher replace s =
  rewrite
    (fun a b ->
      let n = Array.length a in
      let rec go i =
        if i < n then
          match matcher a i with
          | Some j ->
              replace b a i j;
              go j
          | None ->
              add b a.(i);
              go (i + 1)
      in
      go 0)
    s

let replace_words ~left ~right alternatives by =
  let alternatives = List.map chars alternatives and by = chars by in
  substitute (match_words ~left ~right alternatives) (fun b _ _ _ ->
      add_all b by)

let index_from a i p =
  let rec go j =
    if j >= Array.length a then None else if p a.(j) then Some j else go (j + 1)
  in
  go i

let ( >>= ) = Option.bind
let cp c = Char.code c

(* Lowercasing. Python's [str.lower] lowers a capital sigma to a final sigma
   where it ends a word, Unicode's Final_Sigma (The Unicode Standard, §3.13):
   preceded by a cased letter and not followed by one, case-ignorable characters
   skipped. *)

let cased = in_class {|[\p{Lu}\p{Ll}\p{Lt}]|}
let ignorable_category = in_class {|[\p{Mn}\p{Me}\p{Cf}\p{Lm}\p{Sk}]|}

(* Word_Break MidLetter, MidNumLet and Single_Quote. *)
let ignorable_breaks =
  [
    0x27;
    0x2E;
    0x3A;
    0xB7;
    0x387;
    0x55F;
    0x5F4;
    0x2018;
    0x2019;
    0x2024;
    0x2027;
    0xFE13;
    0xFE52;
    0xFE55;
    0xFF07;
    0xFF0E;
    0xFF1A;
  ]

let ignorable c = List.mem c ignorable_breaks || ignorable_category c

let final_sigma a i =
  let rec cased_from j step =
    if j < 0 || j >= Array.length a then false
    else if ignorable a.(j) then cased_from (j + step) step
    else cased a.(j)
  in
  cased_from (i - 1) (-1) && not (cased_from (i + 1) 1)

let lowercase_engine = Brot.Normalizer.lowercase

let lower s =
  let s =
    rewrite
      (fun a b ->
        Array.iteri
          (fun i c -> add b (if c = 0x3A3 && final_sigma a i then 0x3C2 else c))
          a)
      s
  in
  Brot.Normalizer.apply lowercase_engine s

(* [<\[][^>\]]*[>\]] and \(([^)]+?)\) *)

let brackets =
  substitute
    (fun a i ->
      if a.(i) <> cp '<' && a.(i) <> cp '[' then None
      else
        index_from a (i + 1) (fun c -> c = cp '>' || c = cp ']')
        |> Option.map succ)
    (fun _ _ _ _ -> ())

let parentheses =
  substitute
    (fun a i ->
      if a.(i) <> cp '(' then None
      else
        index_from a (i + 1) (fun c -> c = cp ')') >>= fun j ->
        if j > i + 1 then Some (j + 1) else None)
    (fun _ _ _ _ -> ())

(* \b(hmm|mm|mhm|mmm|uh|um)\b *)
let fillers =
  replace_words ~left:true ~right:true
    [ "hmm"; "mm"; "mhm"; "mmm"; "uh"; "um" ]
    ""

let space_run a i =
  let rec go j =
    if j < Array.length a && is_space a.(j) then go (j + 1) else j
  in
  go i

(* \s+' *)
let apostrophes =
  substitute
    (fun a i ->
      let j = space_run a i in
      if j > i && j < Array.length a && a.(j) = cp '\'' then Some (j + 1)
      else None)
    (fun b _ _ _ -> add b (cp '\''))

(* EnglishTextNormalizer.replacers, in their order: the patterns of the first
   two groups are [\bX\b], those of the last two [X\b]. *)
let replacers =
  List.map
    (fun (p, by) -> replace_words ~left:true ~right:true [ p ] by)
    [
      ("won't", "will not");
      ("can't", "can not");
      ("let's", "let us");
      ("ain't", "aint");
      ("y'all", "you all");
      ("wanna", "want to");
      ("gotta", "got to");
      ("gonna", "going to");
      ("i'ma", "i am going to");
      ("imma", "i am going to");
      ("woulda", "would have");
      ("coulda", "could have");
      ("shoulda", "should have");
      ("ma'am", "madam");
      ("mr", "mister ");
      ("mrs", "missus ");
      ("st", "saint ");
      ("dr", "doctor ");
      ("prof", "professor ");
      ("capt", "captain ");
      ("gov", "governor ");
      ("ald", "alderman ");
      ("gen", "general ");
      ("sen", "senator ");
      ("rep", "representative ");
      ("pres", "president ");
      ("rev", "reverend ");
      ("hon", "honorable ");
      ("asst", "assistant ");
      ("assoc", "associate ");
      ("lt", "lieutenant ");
      ("col", "colonel ");
      ("jr", "junior ");
      ("sr", "senior ");
      ("esq", "esquire ");
    ]
  @ List.map
      (fun (p, by) -> replace_words ~left:false ~right:true [ p ] by)
      [
        ("'d been", " had been");
        ("'s been", " has been");
        ("'d gone", " had gone");
        ("'s gone", " has gone");
        ("'d done", " had done");
        ("'s got", " has got");
        ("n't", " not");
        ("'re", " are");
        ("'s", " is");
        ("'d", " would");
        ("'ll", " will");
        ("'t", " not");
        ("'ve", " have");
        ("'m", " am");
      ]

(* (\d),(\d) *)
let digit_commas =
  substitute
    (fun a i ->
      if
        i + 2 < Array.length a
        && is_digit a.(i)
        && a.(i + 1) = cp ','
        && is_digit a.(i + 2)
      then Some (i + 3)
      else None)
    (fun b a i _ ->
      add b a.(i);
      add b a.(i + 2))

(* \.([^0-9]|$) *)
let periods =
  substitute
    (fun a i ->
      if a.(i) <> cp '.' then None
      else if i + 1 = Array.length a then Some (i + 1)
      else if ascii_digit a.(i + 1) then None
      else Some (i + 2))
    (fun b a i j ->
      add b (cp ' ');
      if j > i + 1 then add b a.(i + 1))

(* remove_symbols_and_diacritics, keeping [. % $ ¢ € £]: NFKD, then per
   character: kept, a letter NFKD leaves whole written in ASCII, a nonspacing
   mark dropped, any other mark, symbol or punctuation a space. *)

let diacritics =
  [
    ("œ", "oe");
    ("Œ", "OE");
    ("ø", "o");
    ("Ø", "O");
    ("æ", "ae");
    ("Æ", "AE");
    ("ß", "ss");
    ("ẞ", "SS");
    ("đ", "d");
    ("Đ", "D");
    ("ð", "d");
    ("Ð", "D");
    ("þ", "th");
    ("Þ", "th");
    ("ł", "l");
    ("Ł", "L");
  ]

let symbols =
  Brot.Normalizer.sequence
    ([ Brot.Normalizer.nfkd ]
    @ List.map
        (fun (pattern, replacement) ->
          Brot.Normalizer.replace ~pattern ~replacement)
        diacritics
    @ [
        Brot.Normalizer.replace_regex ~pattern:{|\p{Mn}|} ~replacement:"";
        Brot.Normalizer.replace_regex
          ~pattern:{|[[\p{M}\p{S}\p{P}]&&[^.%$¢€£]]|} ~replacement:" ";
      ])

(* Python's [str.split ()]: the runs of characters between white space. *)
let split s =
  let a = code_points s in
  let n = Array.length a in
  let rec go i acc =
    let i = space_run a i in
    if i >= n then List.rev acc
    else
      let rec stop j =
        if j < n && not (is_space a.(j)) then stop (j + 1) else j
      in
      let j = stop i in
      let b = Buffer.create (j - i) in
      Array.iter (add b) (Array.sub a i (j - i));
      go j (Buffer.contents b :: acc)
  in
  go 0 []

(* Numbers

   EnglishNumberNormalizer. Upstream's values are Python integers, of any size
   ("twenty decillion" is 2 * 10{^34}), or strings: an integer is held here as
   its decimal digits. *)

type value = Int of string | Str of string

let to_string = function Int d | Str d -> d

(* Python's [value or ""]: [None], [0] and [""] are false. *)
let or_empty = function
  | None | Some (Int "0") | Some (Str "") -> ""
  | Some v -> to_string v

let strip_zeros d =
  let n = String.length d in
  let rec go i = if i < n - 1 && d.[i] = '0' then go (i + 1) else i in
  if n = 0 then "0" else String.sub d (go 0) (n - go 0)

(* The sum of two numbers of decimal digits. *)
let add_digits x y =
  let n = max (String.length x) (String.length y) in
  let digit s k =
    let i = String.length s - 1 - k in
    if i >= 0 then Char.code s.[i] - 48 else 0
  in
  let b = Bytes.create (n + 1) in
  let carry = ref 0 in
  for k = 0 to n do
    let s = digit x k + digit y k + !carry in
    Bytes.set b (n - k) (Char.chr (48 + (s mod 10)));
    carry := s / 10
  done;
  strip_zeros (Bytes.to_string b)

let low d k =
  let n = String.length d in
  int_of_string (String.sub d (max 0 (n - k)) (min n k))

let plus d k = add_digits d (string_of_int k)
let power p = "1" ^ String.make p '0'

(* [value // 1000 * 1000 + value % 1000 * 10{^p}] *)
let scale d p =
  let n = String.length d in
  let before = if n <= 3 then "0" else String.sub d 0 (n - 3) ^ "000" in
  add_digits before (strip_zeros (string_of_int (low d 3) ^ String.make p '0'))

(* Python's [Fraction s] of a decimal [s]: digits, a point and digits, at least
   one digit. [Some (whole, fraction)] in ASCII digits. *)
let decimal s =
  let a = chars s in
  let n = Array.length a in
  let rec digits i = if i < n && is_digit a.(i) then digits (i + 1) else i in
  let ascii i j =
    String.init (j - i) (fun k -> Char.chr (48 + digit_value a.(i + k)))
  in
  let w = digits 0 in
  if w = n then if n = 0 then None else Some (ascii 0 w, "")
  else if a.(w) <> cp '.' then None
  else
    let f = digits (w + 1) in
    if f <> n || (w = 0 && f = w + 1) then None
    else Some (ascii 0 w, ascii (w + 1) f)

(* [Fraction s * 10{^p}] if it is an integer. *)
let times s p =
  decimal s >>= fun (whole, fraction) ->
  let f = String.length fraction in
  let kept = String.sub fraction 0 (min f p) in
  if String.exists (( <> ) '0') (String.sub fraction (min f p) (f - min f p))
  then None
  else
    Some (strip_zeros (whole ^ kept ^ String.make (p - String.length kept) '0'))

(* ^\d+(\.\d+)?$ *)
let numeric s =
  match decimal s with
  | Some (w, f) -> w <> "" && (f <> "" || not (String.ends_with ~suffix:"." s))
  | None -> false

let ones =
  [
    "one";
    "two";
    "three";
    "four";
    "five";
    "six";
    "seven";
    "eight";
    "nine";
    "ten";
    "eleven";
    "twelve";
    "thirteen";
    "fourteen";
    "fifteen";
    "sixteen";
    "seventeen";
    "eighteen";
    "nineteen";
  ]
  |> List.mapi (fun i name -> (name, i + 1))

let ones_suffixed =
  List.map
    (fun (name, v) ->
      ((if name = "six" then "sixes" else name ^ "s"), (v, "s")))
    ones
  @ [
      ("zeroth", (0, "th"));
      ("first", (1, "st"));
      ("second", (2, "nd"));
      ("third", (3, "rd"));
      ("fifth", (5, "th"));
      ("twelfth", (12, "th"));
    ]
  @ List.filter_map
      (fun (name, v) ->
        if v > 3 && v <> 5 && v <> 12 then
          Some
            ( (name ^ if String.ends_with ~suffix:"t" name then "h" else "th"),
              (v, "th") )
        else None)
      ones

let tens =
  [
    ("twenty", 20);
    ("thirty", 30);
    ("forty", 40);
    ("fifty", 50);
    ("sixty", 60);
    ("seventy", 70);
    ("eighty", 80);
    ("ninety", 90);
  ]

let replace_y name by = String.concat by (String.split_on_char 'y' name)

let tens_suffixed =
  List.map (fun (name, v) -> (replace_y name "ies", (v, "s"))) tens
  @ List.map (fun (name, v) -> (replace_y name "ieth", (v, "th"))) tens

(* Multipliers by their power of ten. *)
let multipliers =
  [
    ("hundred", 2);
    ("thousand", 3);
    ("million", 6);
    ("billion", 9);
    ("trillion", 12);
    ("quadrillion", 15);
    ("quintillion", 18);
    ("sextillion", 21);
    ("septillion", 24);
    ("octillion", 27);
    ("nonillion", 30);
    ("decillion", 33);
  ]

let multipliers_suffixed =
  List.map (fun (name, p) -> (name ^ "s", (p, "s"))) multipliers
  @ List.map (fun (name, p) -> (name ^ "th", (p, "th"))) multipliers

let zeros = [ "o"; "oh"; "zero" ]

let preceding =
  [ ("minus", "-"); ("negative", "-"); ("plus", "+"); ("positive", "+") ]

let following =
  [
    ("pound", "£");
    ("pounds", "£");
    ("euro", "€");
    ("euros", "€");
    ("dollar", "$");
    ("dollars", "$");
    ("cent", "¢");
    ("cents", "¢");
  ]

let prefixes = List.map cp [ '-'; '+'; '$' ] @ [ 0xA3; 0x20AC; 0xA2 ]
let specials = [ "and"; "double"; "triple"; "point" ]

let words =
  let keys l = List.map fst l in
  zeros @ keys ones @ keys ones_suffixed @ keys tens @ keys tens_suffixed
  @ keys multipliers @ keys multipliers_suffixed @ keys preceding
  @ keys following @ [ "per"; "percent" ] @ specials

let is_in l w = match w with Some w -> List.mem_assoc w l | None -> false
let is_word_of l w = match w with Some w -> List.mem w l | None -> false

(* Upstream's branch for a value that is a string or follows a ones word: [k]'s
   digits follow the value's, replacing its last zero after a tens word. *)
let appended v ~prev k =
  let s = to_string v in
  if is_in tens prev && k < 10 then
    String.sub s 0 (String.length s - 1) ^ string_of_int k
  else s ^ string_of_int k

(* EnglishNumberNormalizer.process_words, a word at a time with the words before
   and after it. *)
let process words_ =
  let out = ref [] and prefix = ref None and value = ref None in
  let output r =
    out := (match !prefix with Some p -> p ^ r | None -> r) :: !out;
    value := None;
    prefix := None
  in
  let output_value () = Option.iter (fun v -> output (to_string v)) !value in
  let a = Array.of_list words_ in
  let n = Array.length a in
  let rec go i =
    if i < n then begin
      let prev = if i > 0 then Some a.(i - 1) else None
      and current = a.(i)
      and next = if i + 1 < n then Some a.(i + 1) else None in
      let next_numeric =
        match next with Some w -> numeric w | None -> false
      in
      let first = (chars current).(0) in
      let has_prefix = List.mem first prefixes in
      let bare =
        if has_prefix then
          String.sub current
            (String.length (utf8 first))
            (String.length current - String.length (utf8 first))
        else current
      in
      let skip =
        step ~prev ~current ~next ~next_numeric ~first ~has_prefix ~bare
      in
      go (if skip then i + 2 else i + 1)
    end
  and step ~prev ~current ~next ~next_numeric ~first ~has_prefix ~bare =
    if numeric bare then
      begin match !value with
      | Some (Str s) when String.ends_with ~suffix:"." s ->
          value := Some (Str (s ^ current));
          false
      | v ->
          if v <> None then output_value ();
          if has_prefix then prefix := Some (utf8 first);
          (value :=
             match times bare 0 with
             | Some d -> Some (Int d)
             | None -> Some (Str bare));
          false
      end
    else if not (List.mem current words) then begin
      output_value ();
      output current;
      false
    end
    else if List.mem current zeros then begin
      value := Some (Str (or_empty !value ^ "0"));
      false
    end
    else if List.mem_assoc current ones then begin
      let k = List.assoc current ones in
      (value :=
         match !value with
         | None -> Some (Int (string_of_int k))
         | Some (Int d) when not (is_in ones prev) ->
             if low d (if k < 10 then 1 else 2) = 0 then Some (Int (plus d k))
             else Some (Str (d ^ string_of_int k))
         | Some v -> Some (Str (appended v ~prev k)));
      false
    end
    else if List.mem_assoc current ones_suffixed then begin
      let k, suffix = List.assoc current ones_suffixed in
      (match !value with
      | None -> output (string_of_int k ^ suffix)
      | Some (Int d) when not (is_in ones prev) ->
          if low d (if k < 10 then 1 else 2) = 0 then output (plus d k ^ suffix)
          else output (d ^ string_of_int k ^ suffix)
      | Some v -> output (appended v ~prev k ^ suffix));
      value := None;
      false
    end
    else if List.mem_assoc current tens then begin
      let k = List.assoc current tens in
      (value :=
         match !value with
         | None -> Some (Int (string_of_int k))
         | Some (Str s) -> Some (Str (s ^ string_of_int k))
         | Some (Int d) ->
             if low d 2 = 0 then Some (Int (plus d k))
             else Some (Str (d ^ string_of_int k)));
      false
    end
    else if List.mem_assoc current tens_suffixed then begin
      let k, suffix = List.assoc current tens_suffixed in
      (match !value with
      | None -> output (string_of_int k ^ suffix)
      | Some (Str s) -> output (s ^ string_of_int k ^ suffix)
      | Some (Int d) ->
          if low d 2 = 0 then output (plus d k ^ suffix)
          else output (d ^ string_of_int k ^ suffix));
      false
    end
    else if List.mem_assoc current multipliers then begin
      let p = List.assoc current multipliers in
      (match !value with
      | None -> value := Some (Int (power p))
      | Some (Int "0") -> value := Some (Int "0")
      | Some (Str s) -> (
          match times s p with
          | Some d -> value := Some (Int d)
          | None ->
              output s;
              value := Some (Int (power p)))
      | Some (Int d) -> value := Some (Int (scale d p)));
      false
    end
    else if List.mem_assoc current multipliers_suffixed then begin
      let p, suffix = List.assoc current multipliers_suffixed in
      (match !value with
      | None -> output (power p ^ suffix)
      | Some (Str s) -> (
          match times s p with
          | Some d -> output (d ^ suffix)
          | None ->
              output s;
              output (power p ^ suffix))
      | Some (Int d) -> output (scale d p ^ suffix));
      value := None;
      false
    end
    else if List.mem_assoc current preceding then begin
      output_value ();
      if is_word_of words next || next_numeric then
        prefix := Some (List.assoc current preceding)
      else output current;
      false
    end
    else if List.mem_assoc current following then begin
      (match !value with
      | Some v ->
          prefix := Some (List.assoc current following);
          output (to_string v)
      | None -> output current);
      false
    end
    else if current = "per" || current = "percent" then
      begin match !value with
      | None ->
          output current;
          false
      | Some v when current = "percent" ->
          output (to_string v ^ "%");
          false
      | Some v ->
          if next = Some "cent" then begin
            output (to_string v ^ "%");
            true
          end
          else begin
            output (to_string v);
            output current;
            false
          end
      end
    else if not (is_word_of words next || next_numeric) then begin
      output_value ();
      output current;
      false
    end
    else if current = "and" then begin
      if not (is_in multipliers prev) then begin
        output_value ();
        output current
      end;
      false
    end
    else if current = "double" || current = "triple" then
      begin if is_in ones next || is_word_of zeros next then begin
        let repeats = if current = "double" then 2 else 3 in
        let k =
          match next with
          | Some w -> Option.value ~default:0 (List.assoc_opt w ones)
          | None -> 0
        in
        let digits =
          String.concat "" (List.init repeats (fun _ -> string_of_int k))
        in
        value := Some (Str (or_empty !value ^ digits));
        true
      end
      else begin
        output_value ();
        output current;
        false
      end
      end
    else begin
      (* "point" *)
      let decimals = zeros @ List.map fst ones @ List.map fst tens in
      if is_word_of decimals next || next_numeric then
        value := Some (Str (or_empty !value ^ "."));
      false
    end
  in
  go 0;
  output_value ();
  List.rev !out

(* EnglishNumberNormalizer.preprocess *)

(* \band\s+a\s+half\b *)
let and_a_half a i =
  if not (boundary a i && matches a i (chars "and")) then None
  else
    let j = space_run a (i + 3) in
    if j = i + 3 || not (matches a j (chars "a")) then None
    else
      let k = space_run a (j + 1) in
      if k = j + 1 || not (matches a k (chars "half")) then None
      else if boundary a (k + 4) then Some (k + 4)
      else None

let sub_string a i j =
  let b = Buffer.create (j - i) in
  for k = i to j - 1 do
    add b a.(k)
  done;
  Buffer.contents b

(* Python's [re.split]: the text between the matches. *)
let split_at matcher s =
  let a = code_points s in
  let n = Array.length a in
  let rec go start i acc =
    if i >= n then List.rev (sub_string a start n :: acc)
    else
      match matcher a i with
      | Some j -> go j j (sub_string a start i :: acc)
      | None -> go start (i + 1) acc
  in
  go 0 0 []

let halves s =
  let segments = split_at and_a_half s in
  let last = List.length segments - 1 in
  let decimals = zeros @ List.map fst ones @ List.map fst tens in
  List.concat
    (List.mapi
       (fun i segment ->
         if split segment = [] then []
         else if i = last then [ segment ]
         else
           let w = List.nth (split segment) (List.length (split segment) - 1) in
           [
             segment;
             (if List.mem w decimals || List.mem_assoc w multipliers then
                "point five"
              else "and a half");
           ])
       segments)
  |> String.concat " "

(* ([a-z])([0-9]) and ([0-9])([a-z]) *)
let split_pair first second =
  substitute
    (fun a i ->
      if i + 1 < Array.length a && first a.(i) && second a.(i + 1) then
        Some (i + 2)
      else None)
    (fun b a i _ ->
      add b a.(i);
      add b (cp ' ');
      add b a.(i + 1))

(* ([0-9])\s+(st|nd|rd|th|s)\b *)
let suffixes =
  let alternatives = List.map chars [ "st"; "nd"; "rd"; "th"; "s" ] in
  substitute
    (fun a i ->
      if not (ascii_digit a.(i)) then None
      else
        let j = space_run a (i + 1) in
        if j = i + 1 then None
        else match_words ~left:false ~right:true alternatives a j)
    (fun b a i j ->
      add b a.(i);
      let k = space_run a (i + 1) in
      for m = k to j - 1 do
        add b a.(m)
      done)

let preprocess s =
  let s = halves s in
  let s = split_pair ascii_lower ascii_digit s in
  let s = split_pair ascii_digit ascii_lower s in
  suffixes s

(* EnglishNumberNormalizer.postprocess *)

let currencies = [ 0x20AC; 0xA3; cp '$' ]
let cent = 0xA2

let digit_run a i =
  let rec go j =
    if j < Array.length a && ascii_digit a.(j) then go (j + 1) else j
  in
  go i

(* [Some j] for the 2 or 1 digits at [i] followed by [\b], as [{1,2}\b] tries
   them. *)
let cents a i =
  let n = digit_run a i - i in
  List.find_opt (fun k -> k <= n && boundary a (i + k)) [ 2; 1 ]
  |> Option.map (fun k -> i + k)

(* ([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b *)
let combine_cents =
  substitute
    (fun a i ->
      if not (List.mem a.(i) currencies) then None
      else
        let j = digit_run a (i + 1) in
        if j = i + 1 || not (matches a j [| cp ' ' |]) then None
        else
          List.find_map
            (fun k -> if matches a k [| cent |] then cents a (k + 1) else None)
            (if matches a (j + 1) (chars "and ") then [ j + 5; j + 1 ]
             else [ j + 1 ]))
    (fun b a i j ->
      let integer = digit_run a (i + 1) in
      let rec cents_start k =
        if ascii_digit a.(k - 1) then cents_start (k - 1) else k
      in
      let c = cents_start j in
      add b a.(i);
      for k = i + 1 to integer - 1 do
        add b a.(k)
      done;
      Buffer.add_string b
        (Printf.sprintf ".%02d" (int_of_string (sub_string a c j))))

(* [€£$]0.([0-9]{1,2})\b *)
let extract_cents =
  substitute
    (fun a i ->
      if
        List.mem a.(i) currencies
        && i + 2 < Array.length a
        && a.(i + 1) = cp '0'
        && a.(i + 2) <> cp '\n'
      then cents a (i + 3)
      else None)
    (fun b a i j ->
      add b cent;
      Buffer.add_string b
        (string_of_int (int_of_string (sub_string a (i + 3) j))))

(* (?<!\S)1(s?)(?!\S) *)
let ones_alone =
  substitute
    (fun a i ->
      let n = Array.length a in
      let space j = j >= n || is_space a.(j) in
      if a.(i) <> cp '1' || not (i = 0 || is_space a.(i - 1)) then None
      else if i + 1 < n && a.(i + 1) = cp 's' && space (i + 2) then Some (i + 2)
      else if space (i + 1) then Some (i + 1)
      else None)
    (fun b _ i j -> Buffer.add_string b (if j = i + 2 then "ones" else "one"))

let numbers s =
  let s = preprocess s in
  let s = String.concat " " (process (split s)) in
  ones_alone (extract_cents (combine_cents s))

(* EnglishSpellingNormalizer *)

let spellings =
  let t = Hashtbl.create 2048 in
  (* english.json holds one ["british": "american"] pair a line. *)
  List.iter
    (fun line ->
      match String.split_on_char '"' line with
      | [ _; british; _; american; _ ] -> Hashtbl.replace t british american
      | _ -> ())
    (String.split_on_char '\n' English_json.text);
  t

let spell s =
  String.concat " "
    (List.map
       (fun w -> Option.value ~default:w (Hashtbl.find_opt spellings w))
       (split s))

(* [.$¢€£]([^0-9]) and ([^0-9])% *)

let unattached = List.map cp [ '.'; '$' ] @ [ cent; 0x20AC; 0xA3 ]

let symbols_before =
  substitute
    (fun a i ->
      if
        List.mem a.(i) unattached
        && i + 1 < Array.length a
        && not (ascii_digit a.(i + 1))
      then Some (i + 2)
      else None)
    (fun b a i _ ->
      add b (cp ' ');
      add b a.(i + 1))

let percents =
  substitute
    (fun a i ->
      if
        (not (ascii_digit a.(i)))
        && i + 1 < Array.length a
        && a.(i + 1) = cp '%'
      then Some (i + 2)
      else None)
    (fun b a i _ ->
      add b a.(i);
      add b (cp ' '))

(* \s+ *)
let spaces =
  substitute
    (fun a i ->
      let j = space_run a i in
      if j > i then Some j else None)
    (fun b _ _ _ -> add b (cp ' '))

let normalize s =
  let s = lower s in
  let s = brackets s in
  let s = parentheses s in
  let s = fillers s in
  let s = apostrophes s in
  let s = List.fold_left (fun s replace -> replace s) s replacers in
  let s = digit_commas s in
  let s = periods s in
  let s = Brot.Normalizer.apply symbols s in
  let s = numbers s in
  let s = spell s in
  let s = symbols_before s in
  let s = percents s in
  spaces s
