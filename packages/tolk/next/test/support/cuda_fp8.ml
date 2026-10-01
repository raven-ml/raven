open Tolk_next

let helper = "template <class T, class F> __device__ __forceinline__ T tg_fp8("

let infinity_byte t infinity =
  match Dtype.bitcast t Uint8 (`Float infinity) with
  | `Int z -> Printf.sprintf "0x%02x" (Bigint.to_int z)
  | _ -> invalid_arg "a byte is an integer"

let fp8_of_name = function
  | "__nv_fp8_e4m3" -> Dtype.Fp8e4m3
  | "__nv_fp8_e5m2" -> Fp8e5m2
  | t -> invalid_arg ("no CUDA float8 type " ^ t)

(* [closing s i] is the index of the parenthesis that closes the one opened just
   before [i], and the index of the last ", " at depth zero on the way. *)
let closing s i =
  let rec go i depth comma =
    match s.[i] with
    | '(' -> go (i + 1) (depth + 1) comma
    | ')' when depth = 0 -> (i, comma)
    | ')' -> go (i + 1) (depth - 1) comma
    | ',' when depth = 0 -> go (i + 1) depth i
    | _ -> go (i + 1) depth comma
  in
  go i 0 (-1)

let starts_at s i prefix =
  i + String.length prefix <= String.length s
  && String.sub s i (String.length prefix) = prefix

let starts_with prefix s = String.starts_with ~prefix s

let rec tinygrad_of_d16 src =
  let b = Buffer.create (String.length src) in
  let rec scan i =
    if i >= String.length src then ()
    else if starts_at src i "tg_fp8<" then begin
      let t_end = String.index_from src i '>' in
      let t = String.sub src (i + 7) (t_end - i - 7) in
      let close, comma = closing src (t_end + 2) in
      let value = String.sub src (t_end + 2) (comma - t_end - 2) in
      Printf.bprintf b "((%s)(%s))" t (tinygrad_of_d16 value);
      scan (close + 1)
    end
    else if starts_at src i "(tg_bitcast<__nv_fp8_" then begin
      let t_end = String.index_from src i '>' in
      let t = String.sub src (i + 12) (t_end - i - 12) in
      let close, _ = closing src (t_end + 2) in
      let byte = String.sub src (close - 4) 4 in
      let dt = fp8_of_name t in
      let value =
        if byte = infinity_byte dt infinity then "INFINITY"
        else if byte = infinity_byte dt neg_infinity then "-INFINITY"
        else invalid_arg ("no infinity of " ^ t ^ " is " ^ byte)
      in
      Printf.bprintf b "((%s)(%s))" t value;
      scan (close + 2)
    end
    else begin
      Buffer.add_char b src.[i];
      scan (i + 1)
    end
  in
  scan 0;
  String.split_on_char '\n' (Buffer.contents b)
  |> List.filter (fun line -> not (starts_with helper line))
  |> String.concat "\n"
