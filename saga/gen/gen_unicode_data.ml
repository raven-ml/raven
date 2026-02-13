(* Generate saga/lib/tokenizers/unicode_data.{ml,mli} from uucp.
   Run: dune exec saga/gen/gen_unicode_data.exe *)

let max_cp = 0x10FFFF
let block_sz = 256
let pf = Format.fprintf
let gc_str gc = Format.asprintf "%a" Uucp.Gc.pp gc
let sc_str sc = Format.asprintf "%a" Uucp.Script.pp sc

(* ── Collect ── *)

let ranges pred =
  let rs = ref [] and s = ref (-1) in
  for cp = 0 to max_cp do
    if Uchar.is_valid cp && pred (Uchar.of_int cp)
    then (if !s < 0 then s := cp)
    else if !s >= 0 then (rs := (!s, cp - 1) :: !rs; s := -1)
  done;
  if !s >= 0 then rs := (!s, max_cp) :: !rs;
  List.rev !rs

let block_table default data =
  let nb = max_cp / block_sz + 1 in
  let unique = Hashtbl.create 256 and next = ref 0 in
  let index = Array.init nb (fun i ->
    let key = List.init block_sz (fun j ->
      let cp = i * block_sz + j in
      if cp <= max_cp then data.(cp) else default) in
    match Hashtbl.find_opt unique key with
    | Some id -> id
    | None -> let id = !next in incr next; Hashtbl.replace unique key id; id) in
  let ordered = Array.make !next [||] in
  Hashtbl.iter (fun k id -> ordered.(id) <- Array.of_list k) unique;
  (index, ordered)

(* ── Emit ── *)

let pp_iarray ppf a =
  pf ppf "@[<2>[|";
  Array.iteri (fun i v ->
    if i > 0 then pf ppf ";";
    if i mod 16 = 0 then pf ppf "@\n  ";
    pf ppf " %d" v) a;
  pf ppf "@\n|]@]"

let emit_array ppf name a = pf ppf "@[<2>let %s =@ %a@]@\n@\n" name pp_iarray a

let emit_block ppf prefix default names idx blks =
  let n = Array.length blks in
  let flat = Array.make (n * block_sz) 0 in
  Array.iteri (fun i b -> Array.iteri (fun j v -> flat.(i * block_sz + j) <- v) b) blks;
  emit_array ppf (prefix ^ "_bi") idx;
  emit_array ppf (prefix ^ "_bv") flat;
  pf ppf "let %s_variants : %s array =@\n  [|" prefix prefix;
  Array.iter (fun s -> pf ppf " `%s;" s) names;
  pf ppf " |]@\n@\n";
  pf ppf "let %s cp =@\n\
         \  if cp < 0 || cp > 0x%X then `%s@\n\
         \  else %s_variants.(%s_bv.(%s_bi.(cp lsr 8) * 256 + (cp land 0xFF)))@\n@\n"
    prefix max_cp default prefix prefix prefix

let emit_bool ppf name rs =
  let n = List.length rs in
  let lo = Array.make n 0 and hi = Array.make n 0 in
  List.iteri (fun i (l, h) -> lo.(i) <- l; hi.(i) <- h) rs;
  emit_array ppf (name ^ "_lo") lo;
  emit_array ppf (name ^ "_hi") hi;
  pf ppf "let %s cp =@\n\
         \  let lo = %s_lo and hi = %s_hi in@\n\
         \  let rec go a b =@\n\
         \    if a > b then false else@\n\
         \    let m = (a + b) / 2 in@\n\
         \    if cp < lo.(m) then go a (m - 1)@\n\
         \    else if cp > hi.(m) then go (m + 1) b@\n\
         \    else true in@\n\
         \  go 0 (Array.length lo - 1)@\n@\n" name name name

let emit_case_fold ppf entries =
  let n = List.length entries in
  let keys = Array.make n 0 and off = Array.make (n + 1) 0 in
  let tgts = ref [] and pos = ref 0 in
  List.iteri (fun i (cp, ts) ->
    keys.(i) <- cp; off.(i) <- !pos;
    List.iter (fun t -> tgts := t :: !tgts) ts;
    pos := !pos + List.length ts) entries;
  off.(n) <- !pos;
  let vals = Array.of_list (List.rev !tgts) in
  emit_array ppf "case_fold_keys" keys;
  emit_array ppf "case_fold_vals" vals;
  emit_array ppf "case_fold_off" off;
  pf ppf "let case_fold cp =@\n\
         \  let keys = case_fold_keys in@\n\
         \  let rec go a b =@\n\
         \    if a > b then [cp] else@\n\
         \    let m = (a + b) / 2 in@\n\
         \    if cp < keys.(m) then go a (m - 1)@\n\
         \    else if cp > keys.(m) then go (m + 1) b@\n\
         \    else@\n\
         \      let rec collect acc i =@\n\
         \        if i >= case_fold_off.(m + 1) then List.rev acc@\n\
         \        else collect (case_fold_vals.(i) :: acc) (i + 1) in@\n\
         \      collect [] case_fold_off.(m) in@\n\
         \  go 0 (Array.length keys - 1)@\n@\n"

let pp_type_variants ppf ~per_line names =
  Array.iteri (fun i s ->
    if i > 0 then
      (if i mod per_line = 0 then pf ppf "@\n  | " else pf ppf " | ");
    pf ppf "`%s" s) names

(* ── Main ── *)

let file = "saga/lib/tokenizers/unicode_data"

let with_file name f =
  let oc = open_out_gen [Open_trunc; Open_creat; Open_wronly] 0o664 name in
  let ppf = Format.formatter_of_out_channel oc in
  Format.pp_set_margin ppf 10000;
  f ppf; Format.pp_print_flush ppf (); close_out oc

let () =
  Format.printf "Generating Unicode v%s data to %s.{ml,mli}@." Uucp.unicode_version file;

  let gc_names = [| "Cc"; "Cf"; "Cn"; "Co"; "Cs"; "Ll"; "Lm"; "Lo"; "Lt"; "Lu";
    "Mc"; "Me"; "Mn"; "Nd"; "Nl"; "No"; "Pc"; "Pd"; "Pe"; "Pf";
    "Pi"; "Po"; "Ps"; "Sc"; "Sk"; "Sm"; "So"; "Zl"; "Zp"; "Zs" |] in
  let gc_idx = Hashtbl.create 30 in
  Array.iteri (fun i s -> Hashtbl.replace gc_idx s i) gc_names;
  let gc_default = Hashtbl.find gc_idx "Cn" in
  let gc_data = Array.init (max_cp + 1) (fun cp ->
    if Uchar.is_valid cp then
      Hashtbl.find gc_idx (gc_str (Uucp.Gc.general_category (Uchar.of_int cp)))
    else if cp >= 0xD800 && cp <= 0xDFFF then Hashtbl.find gc_idx "Cs"
    else gc_default) in
  let gc_bi, gc_bv = block_table gc_default gc_data in

  let sc_set = Hashtbl.create 200 in
  for cp = 0 to max_cp do
    if Uchar.is_valid cp then
      Hashtbl.replace sc_set (sc_str (Uucp.Script.script (Uchar.of_int cp))) true
  done;
  let sc_names = Hashtbl.fold (fun k _ acc -> k :: acc) sc_set []
    |> List.sort String.compare |> Array.of_list in
  let sc_idx = Hashtbl.create 200 in
  Array.iteri (fun i s -> Hashtbl.replace sc_idx s i) sc_names;
  let sc_default = Hashtbl.find sc_idx "Zyyy" in
  let sc_data = Array.init (max_cp + 1) (fun cp ->
    if Uchar.is_valid cp then
      Hashtbl.find sc_idx (sc_str (Uucp.Script.script (Uchar.of_int cp)))
    else sc_default) in
  let sc_bi, sc_bv = block_table sc_default sc_data in

  let ws = ranges Uucp.White.is_white_space in
  let alpha = ranges Uucp.Alpha.is_alphabetic in
  let numeric = ranges (fun u -> Uucp.Num.numeric_type u <> `None) in

  let folds = ref [] in
  for cp = 0 to max_cp do
    if Uchar.is_valid cp then match Uucp.Case.Fold.fold (Uchar.of_int cp) with
    | `Self -> () | `Uchars us -> folds := (cp, List.map Uchar.to_int us) :: !folds
  done;
  let folds = List.rev !folds in

  let pp_header ppf =
    pf ppf "(* Do not edit. Generated from uucp %s via saga/gen/gen_unicode_data.ml. *)@\n@\n"
      Uucp.unicode_version in

  let pp_types ppf =
    pf ppf "type general_category =@\n  [ %a ]@\n@\n" (pp_type_variants ~per_line:5) gc_names;
    pf ppf "type script =@\n  [ %a ]@\n@\n" (pp_type_variants ~per_line:8) sc_names in

  with_file (file ^ ".mli") (fun ppf ->
    pp_header ppf; pp_types ppf;
    pf ppf "val general_category : int -> general_category@\n";
    pf ppf "val script : int -> script@\n";
    pf ppf "val is_white_space : int -> bool@\n";
    pf ppf "val is_alphabetic : int -> bool@\n";
    pf ppf "val is_numeric : int -> bool@\n";
    pf ppf "val case_fold : int -> int list@\n");

  with_file (file ^ ".ml") (fun ppf ->
    pp_header ppf; pp_types ppf;
    emit_block ppf "general_category" "Cn" gc_names gc_bi gc_bv;
    emit_bool ppf "is_white_space" ws;
    emit_bool ppf "is_alphabetic" alpha;
    emit_bool ppf "is_numeric" numeric;
    emit_case_fold ppf folds;
    emit_block ppf "script" "Zyyy" sc_names sc_bi sc_bv)
