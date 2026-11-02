(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Munin

(* Fixtures *)

(* A fixed provenance keeps git out of every start. *)
let provenance =
  {
    Provenance.command = [ "test_munin" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    ocaml = Sys.ocaml_version;
    raven = "test";
    os = "test";
    arch = "test";
    cpu = None;
    git = None;
    seeds = [];
    env = [];
  }

(* Windows removes no read-only file, such as git's objects, until its
   attribute is cleared. *)
let rec rm_rf p =
  match Unix.lstat p with
  | exception Unix.Unix_error _ -> ()
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (fun e -> rm_rf (Filename.concat p e)) (Sys.readdir p);
      Unix.rmdir p
  | _ ->
      if Sys.win32 then Unix.chmod p 0o644;
      Sys.remove p

let with_store f =
  let root = Filename.temp_dir "munin" "" in
  Fun.protect ~finally:(fun () -> rm_rf root) @@ fun () ->
  f (Store.open_ ~root ())

let start ?(experiment = "e") ?params ?tags ?steps store =
  Session.start ~store ~provenance ?params ?tags ?steps ~experiment ()

let reread store s = require_some (Store.run store (Session.id s))
let history store s = History.read (reread store s)

let write_file path data =
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc data)

let read_file path = In_channel.with_open_bin path In_channel.input_all

let replace_first text ~sub ~by =
  let n = String.length sub in
  let rec find i =
    if i + n > String.length text then text
    else if String.sub text i n = sub then
      String.sub text 0 i ^ by
      ^ String.sub text (i + n) (String.length text - i - n)
    else find (i + 1)
  in
  find 0

let child_exe = Filename.concat (Sys.getcwd ()) "support/child.exe"

(* The name a deletion renames a run's log to (Munin.Store_format, resume). *)
let deletion_mark = ".deleted"

(* Witnesses *)

let value : Value.t testable =
  Testable.make ~pp:Value.pp ~equal:(fun a b ->
      match (a, b) with
      | `Float x, `Float y -> Testable.equal float_exact x y
      | a, b -> a = b)

let sample : Metric.sample testable =
  Testable.make
    ~pp:(fun ppf (s : Metric.sample) ->
      Format.fprintf ppf "{step=%d; time=%a; value=%a}" s.step
        (Testable.pp float_exact) s.time (Testable.pp float_exact) s.value)
    ~equal:(fun (a : Metric.sample) b ->
      a.step = b.step
      && Testable.equal float_exact a.time b.time
      && Testable.equal float_exact a.value b.value)

let samples_of s = List.init (Metric.length s) (Metric.sample s)

let status : Run.status testable =
  Testable.make
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | `Running -> "running"
        | `Silent -> "silent"
        | `Crashed -> "crashed"
        | `Finished -> "finished"
        | `Failed e -> "failed: " ^ e
        | `Killed -> "killed"))
    ~equal:( = )

(* Generators *)

let gen_float = Gen.any_float

let gen_value : Value.t Gen.t =
  Gen.one_of
    [
      Gen.map (fun b -> `Bool b) Gen.bool;
      Gen.map (fun n -> `Int n) Gen.int;
      Gen.map (fun n -> `Int n) (Gen.int_range (-(1 lsl 54)) (1 lsl 54));
      Gen.map (fun x -> `Float x) gen_float;
      Gen.map (fun x -> `Float (Float.of_int x)) Gen.small_int;
      Gen.map (fun s -> `String s) (Gen.string_of (Gen.char_range ' ' '~'));
      Gen.constant (`String "tab\tline\nfeed \xc3\xa9 \"q\" \\");
    ]

(* A log written from the format's text alone *)

(* The records a law test draws: samples of metrics, samples of readings and
   resumes. *)
type ev =
  | Sample of string * int * float
  | Reading of string * float
  | Resume of int

let pp_ev ppf = function
  | Sample (k, s, v) -> Format.fprintf ppf "sample %s@%d=%h" k s v
  | Reading (k, v) -> Format.fprintf ppf "reading %s=%h" k v
  | Resume s -> Format.fprintf ppf "resume %d" s

let json_float x =
  if Float.is_nan x then {|{"float":"NaN"}|}
  else if x = Float.infinity then {|{"float":"Infinity"}|}
  else if x = Float.neg_infinity then {|{"float":"-Infinity"}|}
  else Printf.sprintf "%.17g" x

let prov_json =
  {|{"command":["t"],"cwd":"/","pid":1,"ocaml":"5","raven":"t","os":"t","arch":"t","seeds":{},"env":{}}|}

let start_line =
  Printf.sprintf
    {|{"type":"start","time":1.0,"format":1,"experiment":"spec","params":{},"provenance":%s}|}
    prov_json

let ev_line i = function
  | Sample (k, s, v) ->
      Printf.sprintf
        {|{"type":"metrics","time":%d.0,"step":%d,"values":{"%s":%s}}|} (i + 2)
        s k (json_float v)
  | Reading (k, v) ->
      Printf.sprintf
        {|{"type":"reading","time":%d.0,"step":0,"values":{"%s":%s}}|} (i + 2) k
        (json_float v)
  | Resume s ->
      Printf.sprintf {|{"type":"resume","time":%d.0,"step":%d,"provenance":%s}|}
        (i + 2) s prov_json

let metric_line k =
  Printf.sprintf {|{"type":"metric","time":1.5,"key":"%s","goal":"minimize"}|} k

let log_text evs =
  String.concat ""
    (List.map
       (fun l -> l ^ "\n")
       (start_line :: metric_line "a" :: metric_line "b"
      :: List.mapi ev_line evs))

let mkdir_p d =
  try Unix.mkdir d 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()

let spec_run store text =
  let runs = Filename.concat (Store.root store) "runs" in
  mkdir_p (Store.root store);
  mkdir_p runs;
  let dir = Filename.concat runs "spec" in
  rm_rf dir;
  mkdir_p dir;
  write_file (Filename.concat dir "events.jsonl") text;
  require_some (Store.run store "spec")

(* The reference: the format's orphan rule read off the record list, and the
   statistics by their definitions. *)
let reference_samples evs k =
  let arr = Array.of_list evs in
  let n = Array.length arr in
  let orphan i s =
    let rec go j =
      j < n
      && ((match arr.(j) with Resume r -> r < s | _ -> false) || go (j + 1))
    in
    go (i + 1)
  in
  List.concat
    (List.mapi
       (fun i e ->
         match e with
         | Sample (k', s, v) when k' = k && not (orphan i s) ->
             [
               ({ step = s; time = Float.of_int (i + 2); value = v }
                 : Metric.sample);
             ]
         | Reading (k', v) when k' = k ->
             [ { step = 0; time = Float.of_int (i + 2); value = v } ]
         | _ -> [])
       evs)

let gen_ev =
  Gen.frequency
    [
      ( 8,
        Gen.map
          (fun ((k, s), v) -> Sample (k, s, v))
          Gen.(
            pair
              (pair (of_list [ "a"; "b" ]) (int_range 0 20))
              (frequency
                 [
                   (6, float_range (-5.) 5.);
                   (1, gen_float);
                   (1, of_list [ 1.; -0.; 0. ]);
                 ])) );
      (1, Gen.map (fun v -> Reading ("r", v)) (Gen.float_range 0. 100.));
      (1, Gen.map (fun s -> Resume s) (Gen.int_range 0 20));
    ]

let gen_evs = Gen.list ~size:(Gen.int_range 0 60) (Gen.with_pp pp_ev gen_ev)

let check_stats ~msg (expected : Metric.sample list) (got : Metric.stats option)
    =
  match (expected, got) with
  | [], None -> ()
  | [], Some _ -> fail (msg ^ ": statistics of no sample")
  | _, None -> fail (msg ^ ": no statistics")
  | l, Some st -> (
      equal ~msg:(msg ^ " count") int (List.length l) st.count;
      equal ~msg:(msg ^ " last") sample (List.nth l (List.length l - 1)) st.last;
      let finite =
        List.filter (fun (s : Metric.sample) -> Float.is_finite s.value) l
      in
      let non_finite =
        List.filter (fun (s : Metric.sample) -> not (Float.is_finite s.value)) l
      in
      (match (finite, st.finite) with
      | [], None -> ()
      | [], Some _ | _ :: _, None -> fail (msg ^ ": finite statistics disagree")
      | f, Some fs ->
          let pick better =
            List.fold_left
              (fun (b : Metric.sample) (s : Metric.sample) ->
                if better s.value b.value then s else b)
              (List.hd f) (List.tl f)
          in
          equal ~msg:(msg ^ " least") sample (pick ( < )) fs.least;
          equal ~msg:(msg ^ " greatest") sample (pick ( > )) fs.greatest;
          (* The mean by its definition, each term divided first so that no sum
             overflows; within the rounding of that naive sum. *)
          let n = Float.of_int (List.length f) in
          let mean =
            List.fold_left
              (fun a (s : Metric.sample) -> a +. (s.value /. n))
              0. f
          in
          let scale =
            List.fold_left
              (fun a (s : Metric.sample) -> Float.max a (Float.abs s.value))
              0. f
          in
          equal ~msg:(msg ^ " mean")
            (float_rel ~rel:1e-9 ~abs:(Float.max 1e-300 (1e-12 *. scale)))
            mean fs.mean);
      match (non_finite, st.non_finite) with
      | [], None -> ()
      | (s :: _ as l), Some (n, first) ->
          equal ~msg:(msg ^ " non-finite count") int (List.length l) n;
          equal ~msg:(msg ^ " first non-finite") sample s first
      | _ -> fail (msg ^ ": non-finite statistics disagree"))

let check_history ~msg evs h =
  List.iter
    (fun k ->
      let expected = reference_samples evs k in
      equal
        ~msg:(msg ^ " series " ^ k)
        (list sample) expected
        (samples_of (History.series h k));
      check_stats ~msg:(msg ^ " " ^ k) expected (History.stats h k))
    [ "a"; "b"; "r" ];
  let with_samples = List.filter (fun k -> reference_samples evs k <> []) in
  equal ~msg:(msg ^ " metrics") (list string)
    (with_samples [ "a"; "b" ])
    (History.metrics h);
  equal ~msg:(msg ^ " readings") (list string) (with_samples [ "r" ])
    (History.readings h)

(* Values *)

let values_tests =
  group "values"
    [
      prop "params read back equal, with their constructors"
        (Gen.list ~size:(Gen.int_range 0 8) gen_value)
        (fun vs ->
          List.iter
            (fun v ->
              cover "integral float"
                (match v with `Float x -> Float.is_integer x | _ -> false);
              cover "non-finite float"
                (match v with
                | `Float x -> not (Float.is_finite x)
                | _ -> false);
              cover "int beyond 2^53"
                (match v with `Int n -> abs n > 1 lsl 53 | _ -> false))
            vs;
          with_store @@ fun store ->
          let params = List.mapi (fun i v -> (Printf.sprintf "p%d" i, v)) vs in
          let s = start ~params store in
          Session.finish s;
          equal (list (pair string value)) params (Run.params (reread store s)));
      prop "samples read back equal"
        (Gen.list ~size:(Gen.int_range 1 20) (Gen.pair gen_float gen_float))
        (fun pairs ->
          cover "NaN" (List.exists (fun (_, v) -> Float.is_nan v) pairs);
          with_store @@ fun store ->
          let s = start store in
          let m = Session.metric s "m" in
          let expected =
            List.mapi
              (fun step (t, v) ->
                let time = if Float.is_finite t then t else 1. in
                Session.log m ~step ~time v;
                ({ step; time; value = v } : Metric.sample))
              pairs
          in
          Session.finish s;
          equal (list sample) expected
            (samples_of (History.series (history store s) "m")));
      test "Value.pp prints tab-separated text" (fun () ->
          equal string {|a\tb\nc|}
            (Format.asprintf "%a" Value.pp (`String "a\tb\nc"));
          equal string "1.0" (Format.asprintf "%a" Value.pp (`Float 1.));
          equal string "-0.0" (Format.asprintf "%a" Value.pp (`Float (-0.)));
          equal string "nan" (Format.asprintf "%a" Value.pp (`Float Float.nan)));
    ]

(* Reading floats *)

let gen_decimal =
  let open Gen in
  let digits = string_of ~size:(int_range 1 20) (char_range '0' '9') in
  map
    (fun ((neg, (int, frac)), e) ->
      (if neg then "-" else "")
      ^ (if int.[0] = '0' && String.length int > 1 then "1" ^ int else int)
      ^ (if frac = "" then "" else "." ^ frac)
      ^ match e with None -> "" | Some e -> "e" ^ string_of_int e)
    (pair
       (pair bool
          (pair digits (string_of ~size:(int_range 0 20) (char_range '0' '9'))))
       (option (int_range (-340) 330)))

(* A decimal of at most 18 significant digits, with an exponent that keeps most
   of them within 10^-22 to 10^22 of an integer. *)
let gen_short_decimal =
  let open Gen in
  map
    (fun ((neg, m), (e, point)) ->
      let d = string_of_int m in
      let point = Int.min point (String.length d) in
      let int = String.sub d 0 (String.length d - point)
      and frac = String.sub d (String.length d - point) point in
      (if neg then "-" else "")
      ^ (if int = "" then "0" else int)
      ^ (if frac = "" then "" else "." ^ frac)
      ^ if e = 0 then "" else "e" ^ string_of_int e)
    (pair
       (pair bool
          (one_of
             [
               int_range 1 (1 lsl 53);
               int_range (1 lsl 53) 999_999_999_999_999_999;
             ]))
       (pair (int_range (-30) 30) (int_range 0 20)))

(* A midpoint between two floats [M 2^k] and [(M + 1) 2^k], [M] from 2^52 to
   2^53, as [D 10^-j] with [D] of at most 18 digits, or [D] moved by one unit
   below or above it, and the float it reads as: the even one of the two at the
   midpoint, else the nearer. [padded] adds three zeros to [D], taking the text
   past 18 digits. *)
let gen_midpoint =
  let open Gen in
  map
    (fun (((m, k), off), padded) ->
      let mid = (2 * m) + 1 in
      let d, j =
        match k with
        | -1 -> (mid * 25, 2)
        | 0 -> (mid * 5, 1)
        | k -> (mid lsl (k - 1), 0)
      in
      let text =
        if padded then Printf.sprintf "%d000e-%d" (d + off) (j + 3)
        else Printf.sprintf "%de-%d" (d + off) j
      in
      let lo = Float.ldexp (Float.of_int m) k
      and hi = Float.ldexp (Float.of_int (m + 1)) k in
      let expected =
        if off < 0 then lo
        else if off > 0 then hi
        else if m mod 2 = 0 then lo
        else hi
      in
      (text, off, expected))
    (pair
       (pair
          (pair (int_range (1 lsl 52) ((1 lsl 53) - 1)) (int_range (-1) 6))
          (int_range (-1) 1))
       bool)

let float_text_tests =
  group "float text"
    [
      cases "decimals the C library reads wrongly somewhere" ~id:fst
        [ ("3.839223843572815e-239", Float.of_string "0x1.fffffffffffffp-793") ]
        (fun (text, expected) ->
          equal float_exact expected
            (require_some (Munin_float_text.Float_text.of_string text)));
      prop "a float's shortest text reads back as it" Gen.float (fun x ->
          equal float_exact x
            (require_some
               (Munin_float_text.Float_text.of_string
                  (Munin_float_text.Float_text.exact x))));
      prop "a decimal reads as the float nearest it" gen_decimal (fun text ->
          cover "beyond the fast path" (String.length text > 18);
          equal (option float_exact)
            (Munin_float_text.Float_text.read text)
            (Munin_float_text.Float_text.of_string text));
      prop "a decimal of at most 18 digits reads as the float nearest it"
        gen_short_decimal (fun text ->
          equal (option float_exact)
            (Munin_float_text.Float_text.read text)
            (Munin_float_text.Float_text.of_string text));
      prop "a midpoint reads as the even float, a unit off it as the nearer"
        gen_midpoint (fun (text, off, expected) ->
          cover "a midpoint" (off = 0);
          equal float_exact expected
            (require_some (Munin_float_text.Float_text.of_string text)));
      prop "a substring reads as its copy" gen_decimal (fun text ->
          let s = "[" ^ text ^ "]" in
          equal (option float_exact)
            (Munin_float_text.Float_text.of_string text)
            (Munin_float_text.Float_text.of_substring s 1 (String.length text)));
      test "a sample of a decimal strtod misreads reads back exactly" (fun () ->
          with_store @@ fun store ->
          let line =
            {|{"type":"metrics","time":2.0,"step":0,"values":{"a":3.839223843572815e-239}}|}
          in
          let run = spec_run store (start_line ^ "\n" ^ line ^ "\n") in
          equal float_exact
            (Float.of_string "0x1.fffffffffffffp-793")
            (Metric.value (History.series (History.read run) "a") 0));
    ]

(* The fold *)

(* Values records, in the layout munin writes and in another: members reordered,
   spaces between tokens, every key escaped. *)
let gen_layout =
  let open Gen in
  let key = Gen.of_list [ "a"; "b"; "c/d"; "q\"\\"; "caf\xc3\xa9" ] in
  list ~size:(int_range 0 12)
    (map
       (fun (((reading, time), step), values) -> (reading, time, step, values))
       (pair
          (pair (pair bool gen_float) (int_range 0 1000))
          (list ~size:(int_range 0 4) (pair key gen_float))))

let json_key k =
  let b = Buffer.create 16 in
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b {|\"|}
      | '\\' -> Buffer.add_string b {|\\|}
      | c -> Buffer.add_char b c)
    k;
  Buffer.add_char b '"';
  Buffer.contents b

(* Every byte escaped, as \u00XX for ASCII. *)
let escaped_key k =
  if String.exists (fun c -> Char.code c >= 0x80) k then json_key k
  else
    "\""
    ^ String.concat ""
        (List.map
           (fun c -> Printf.sprintf "\\u%04x" (Char.code c))
           (List.of_seq (String.to_seq k)))
    ^ "\""

let shortest x =
  if Float.is_finite x then Munin_float_text.Float_text.exact_float x
  else json_float x

let canonical_line (reading, time, step, values) =
  Printf.sprintf {|{"type":"%s","time":%s,"step":%d,"values":{%s}}|}
    (if reading then "reading" else "metrics")
    (shortest time) step
    (String.concat ","
       (List.map (fun (k, v) -> json_key k ^ ":" ^ shortest v) values))

let spaced_line (reading, time, step, values) =
  Printf.sprintf
    {|{ "values" : { %s } , "step" : %d , "time" : %s , "type" : "%s" }|}
    (String.concat " , "
       (List.map (fun (k, v) -> escaped_key k ^ " : " ^ shortest v) values))
    step (shortest time)
    (if reading then "reading" else "metrics")

let fold_tests =
  group "fold"
    [
      prop "a history follows the format's orphan rule and statistics" gen_evs
        (fun evs ->
          cover "a resume orphans a sample"
            (List.exists (function Resume _ -> true | _ -> false) evs
            && List.length (reference_samples evs "a")
               < List.length
                   (List.filter
                      (function Sample ("a", _, _) -> true | _ -> false)
                      evs));
          cover "a non-finite sample"
            (List.exists
               (function
                 | Sample (_, _, v) -> not (Float.is_finite v) | _ -> false)
               evs);
          cover "every sample of a key is an orphan"
            (List.exists
               (fun k ->
                 reference_samples evs k = []
                 && List.exists
                      (function Sample (k', _, _) -> k' = k | _ -> false)
                      evs)
               [ "a"; "b" ]);
          with_store @@ fun store ->
          let run = spec_run store (log_text evs) in
          check_history ~msg:"keep all" evs (History.read run);
          (* Statistics alone, a resume's re-read included. *)
          let h = History.read ~series:[] run in
          List.iter
            (fun k ->
              check_stats ~msg:("keep none " ^ k) (reference_samples evs k)
                (History.stats h k))
            [ "a"; "b"; "r" ];
          equal ~msg:"best is the least finite under minimize" (option sample)
            (match
               List.filter
                 (fun (s : Metric.sample) -> Float.is_finite s.value)
                 (reference_samples evs "a")
             with
            | [] -> None
            | x :: rest ->
                Some
                  (List.fold_left
                     (fun (b : Metric.sample) (s : Metric.sample) ->
                       if s.value < b.value then s else b)
                     x rest))
            (History.best h "a"));
      prop
        "following in pieces equals reading at once, and leaves snapshots as \
         they were"
        (Gen.pair gen_evs
           (Gen.list ~size:(Gen.int_range 1 6) (Gen.int_range 0 100000)))
        (fun (evs, cuts) ->
          with_store @@ fun store ->
          let text = log_text evs in
          let n = String.length text in
          let first = String.length start_line + 1 in
          let cuts =
            List.sort_uniq compare
              (List.map (fun c -> first + (c mod (n - first + 1))) cuts)
          in
          cover "a cut inside a line"
            (List.exists (fun c -> c < n && text.[c - 1] <> '\n') cuts);
          let run = spec_run store (String.sub text 0 first) in
          let path = Filename.concat (Run.dir run) "events.jsonl" in
          let h0 = History.read run in
          let snaps =
            List.fold_left
              (fun acc c ->
                let h = match acc with (h, _) :: _ -> h | [] -> h0 in
                write_file path (String.sub text 0 c);
                (* Rewriting keeps the inode; the prefix is unchanged. *)
                let h' = History.follow h in
                (h', samples_of (History.series h' "a")) :: acc)
              [] cuts
          in
          let last = match snaps with (h, _) :: _ -> h | [] -> h0 in
          write_file path text;
          let final = History.follow last in
          check_history ~msg:"followed" evs final;
          List.iter
            (fun (h, seen) ->
              equal ~msg:"a snapshot stays" (list sample) seen
                (samples_of (History.series h "a")))
            snaps;
          (* A second follow of one history copies. *)
          let again = History.follow last in
          check_history ~msg:"followed twice" evs again;
          check_history ~msg:"first follow kept" evs final);
      prop "a values record reads the same in any layout" gen_layout
        (fun records ->
          cover "a key with an escape"
            (List.exists
               (fun (_, _, _, vs) -> List.mem_assoc "q\"\\" vs)
               records);
          cover "a non-finite value"
            (List.exists
               (fun (_, _, _, vs) ->
                 List.exists (fun (_, v) -> not (Float.is_finite v)) vs)
               records);
          with_store @@ fun store ->
          let read layout =
            History.read
              (spec_run store
                 (String.concat ""
                    (List.map
                       (fun l -> l ^ "\n")
                       (start_line :: List.map layout records))))
          in
          let canonical = read canonical_line and spaced = read spaced_line in
          equal ~msg:"damaged" int (History.damaged spaced)
            (History.damaged canonical);
          equal ~msg:"metrics" (list string) (History.metrics spaced)
            (History.metrics canonical);
          equal ~msg:"readings" (list string) (History.readings spaced)
            (History.readings canonical);
          List.iter
            (fun k ->
              equal ~msg:k (list sample)
                (samples_of (History.series spaced k))
                (samples_of (History.series canonical k)))
            (History.metrics canonical @ History.readings canonical));
      prop "a damaged line is skipped and counted, and never raises"
        (Gen.triple gen_evs (Gen.int_range 0 1000)
           (Gen.of_list
              [
                ( "invalid UTF-8",
                  "{\"type\":\"metrics\",\"time\":1.0,\"step\":0,\"values\":{\"a\":\xff}}"
                );
                ( "a repeated member",
                  {|{"type":"metrics","type":"metrics","time":1.0,"step":0,"values":{}}|}
                );
                (* Any nesting past the format's depth is the same case. *)
                ("deep nesting", String.make 1_000 '[');
                ("an unknown type", {|{"type":"unknown","time":1.0}|});
                ( "a wrong type",
                  {|{"type":"metrics","time":"x","step":0,"values":{"a":1.0}}|}
                );
                ( "a negative step",
                  {|{"type":"metrics","time":1.0,"step":-1,"values":{"a":1.0}}|}
                );
                ( "a bad key",
                  {|{"type":"metrics","time":1.0,"step":0,"values":{"../a":1.0}}|}
                );
                ("not JSON", "not json at all");
                ( "a repeated key",
                  {|{"type":"metrics","time":1.0,"step":0,"values":{"a":1.0,"a":2.0}}|}
                );
                ( "a leading zero",
                  {|{"type":"metrics","time":1.0,"step":01,"values":{"a":1.0}}|}
                );
                ( "a number cut short",
                  {|{"type":"metrics","time":1.0,"step":0,"values":{"a":1.}}|}
                );
                ( "a control character in a key",
                  "{\"type\":\"metrics\",\"time\":1.0,\"step\":0,\"values\":{\"a\tb\":1.0}}"
                );
                ( "trailing bytes",
                  {|{"type":"metrics","time":1.0,"step":0,"values":{"a":1.0}}x|}
                );
              ]))
        (fun (evs, at, (kind, garbage)) ->
          List.iter
            (fun k -> cover k (k = kind))
            [
              "invalid UTF-8";
              "a repeated member";
              "deep nesting";
              "an unknown type";
              "a wrong type";
              "a negative step";
              "a bad key";
              "not JSON";
              "a repeated key";
              "a leading zero";
              "a number cut short";
              "a control character in a key";
              "trailing bytes";
            ];
          with_store @@ fun store ->
          let lines = List.mapi ev_line evs in
          let at = at mod (List.length lines + 1) in
          let damaged =
            List.filteri (fun i _ -> i < at) lines
            @ [ garbage ]
            @ List.filteri (fun i _ -> i >= at) lines
          in
          let text l =
            String.concat ""
              (List.map
                 (fun l -> l ^ "\n")
                 (start_line :: metric_line "a" :: metric_line "b" :: l))
          in
          let clean = History.read (spec_run store (text lines)) in
          let h = History.read (spec_run store (text damaged)) in
          equal ~msg:"counted" int 1 (History.damaged h);
          equal ~msg:"skipped" (list sample)
            (samples_of (History.series clean "a"))
            (samples_of (History.series h "a")));
      test "a replaced log is read again from its start" (fun () ->
          with_store @@ fun store ->
          let evs = [ Sample ("a", 1, 1.); Sample ("a", 2, 2.) ] in
          let run = spec_run store (log_text evs) in
          let h = History.read run in
          let path = Filename.concat (Run.dir run) "events.jsonl" in
          let tmp = path ^ ".new" in
          write_file tmp (log_text [ Sample ("a", 1, 5.) ]);
          Sys.rename tmp path;
          let h' = History.follow h in
          equal ~msg:"restarted" bool true (History.restarted h');
          equal (list sample)
            [ { step = 1; time = 2.; value = 5. } ]
            (samples_of (History.series h' "a"));
          equal ~msg:"an unchanged log" bool false
            (History.restarted (History.follow h')));
      test "a line with a NUL byte stops the reader until it is filled"
        (fun () ->
          with_store @@ fun store ->
          let good = log_text [ Sample ("a", 1, 1.) ] in
          let line = ev_line 1 (Sample ("a", 2, 2.)) ^ "\n" in
          let hole = String.make (String.length line - 1) '\000' ^ "\n" in
          let run =
            spec_run store (good ^ hole ^ ev_line 2 (Sample ("a", 3, 3.)) ^ "\n")
          in
          let h = History.read run in
          equal ~msg:"stops before the hole" int 1
            (Metric.length (History.series h "a"));
          write_file
            (Filename.concat (Run.dir run) "events.jsonl")
            (good ^ line ^ ev_line 2 (Sample ("a", 3, 3.)) ^ "\n");
          equal ~msg:"reads on once filled" int 3
            (Metric.length (History.series (History.follow h) "a")));
      test "a sample of a key of the other kind is damaged" (fun () ->
          with_store @@ fun store ->
          let run =
            spec_run store
              (log_text
                 [ Reading ("a", 1.); Reading ("r", 1.); Sample ("r", 1, 2.) ])
          in
          let h = History.read run in
          equal ~msg:"damaged" int 2 (History.damaged h);
          equal (list string) [] (History.readings h |> List.filter (( = ) "a"));
          equal (list string) [ "r" ] (History.readings h));
      test "a limit within the start reads no record" (fun () ->
          with_store @@ fun store ->
          let run =
            spec_run store
              (start_line ^ "\n" ^ ev_line 0 (Sample ("a", 0, 1.)) ^ "\n")
          in
          equal int 0
            (Metric.length (History.series (History.read ~limit:1 run) "a")));
      test "series raises for a key not kept" (fun () ->
          with_store @@ fun store ->
          let run = spec_run store (log_text [ Sample ("a", 1, 1.) ]) in
          let h = History.read ~series:[ "b" ] run in
          raises_match (Exn.invalid_arg ~sub:"\"a\"") (fun () ->
              History.series h "a"));
    ]

(* Sessions *)

let session_tests =
  group "session"
    [
      test "writes on a finished session raise" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let m = Session.metric s "m" in
          Session.finish s;
          Session.finish s;
          raises_match (Exn.invalid_arg ~sub:"finished") (fun () ->
              Session.log m ~step:0 1.));
      test "a handle of another session is refused" (fun () ->
          with_store @@ fun store ->
          let s1 = start store and s2 = start store in
          let m = Session.metric s1 "m" in
          raises_match (Exn.invalid_arg ~sub:"another session") (fun () ->
              Session.log_metrics s2 ~step:0 [ (m, 1.) ]);
          Session.finish s1;
          Session.finish s2);
      cases "names that are not keys are refused" ~id:Fun.id
        [ ""; "a//b"; ".x"; "a/.."; "a\nb"; "/a"; "a/" ] (fun key ->
          with_store @@ fun store ->
          let s = start store in
          raises_match
            (fun e -> Exn.invalid_arg e)
            (fun () -> Session.metric s key);
          Session.finish s);
      test "steps outside [0;2^53] are refused" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let m = Session.metric s "m" in
          raises_match (Exn.invalid_arg ~sub:"step") (fun () ->
              Session.log m ~step:(-1) 0.);
          raises_match (Exn.invalid_arg ~sub:"step") (fun () ->
              Session.log m ~step:((1 lsl 53) + 1) 0.);
          Session.finish s);
      test "within finishes as failed and raises again" (fun () ->
          with_store @@ fun store ->
          let id = ref "" in
          raises (Failure "boom") (fun () ->
              Session.within (start store) (fun s ->
                  id := Session.id s;
                  failwith "boom"));
          equal status (`Failed "Failure(\"boom\")")
            (Run.status (require_some (Store.run store !id))));
      test "provenance and failure text that are not UTF-8 are kept, escaped"
        (fun () ->
          with_store @@ fun store ->
          let bad =
            { provenance with cwd = "/data/caf\xe9"; command = [ "x\xff" ] }
          in
          let s = Session.start ~store ~provenance:bad ~experiment:"e" () in
          Session.finish ~status:(`Failed "caf\xe9") s;
          let run = require_some (Store.run store (Session.id s)) in
          equal string "/data/caf\xef\xbf\xbd" (Run.provenance run).cwd;
          equal (list string) [ "x\xef\xbf\xbd" ] (Run.provenance run).command;
          equal status (`Failed "caf\xef\xbf\xbd") (Run.status run));
      test "a summary keeps each key's latest value" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          Session.set_summary s [ ("acc", `Float 0.5); ("note", `String "x") ];
          Session.set_summary s [ ("acc", `Float 0.9) ];
          Session.finish s;
          equal
            (list (pair string value))
            [ ("acc", `Float 0.9); ("note", `String "x") ]
            (History.summary (history store s)));
      test "files are copied and named by step" (fun () ->
          with_store @@ fun store ->
          let src = Filename.temp_file "munin" ".png" in
          write_file src "png";
          let s = start store in
          Session.log_file s ~step:1 ~key:"viz/cm" src;
          Session.log_file s ~step:1 ~key:"viz/cm" src;
          Session.finish s;
          let files = History.files (history store s) in
          equal (list string) [ "1.png"; "1-1.png" ]
            (List.map
               (fun (f : History.file) -> Filename.basename f.path)
               files);
          equal string "png" (read_file (List.hd files).path);
          Sys.remove src);
      test "an equal declaration writes nothing, another raises" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let a = Session.metric s ~goal:`Minimize "a" in
          let a' = Session.metric s ~goal:`Minimize "a" in
          raises_match (Exn.invalid_arg ~sub:"declared otherwise") (fun () ->
              Session.metric s ~goal:`Maximize "a");
          Session.log a ~step:0 1.;
          Session.log a' ~step:1 2.;
          equal ~msg:"the session's store" string (Store.root store)
            (Store.root (Session.store s));
          Session.finish s;
          let log =
            read_file
              (Filename.concat (Run.dir (reread store s)) "events.jsonl")
          in
          let declarations =
            List.length
              (List.filter
                 (fun l ->
                   String.length l > 16
                   && String.sub l 0 16 = {|{"type":"metric"|})
                 (String.split_on_char '\n' log))
          in
          equal ~msg:"one metric record" int 1 declarations;
          equal int 2 (Metric.length (History.series (history store s) "a")));
      test "logging a file streams it" (fun () ->
          with_store @@ fun store ->
          let size = 32 * 1024 * 1024 in
          let src = Filename.temp_file "munin" ".bin" in
          write_file src (String.make size 'x');
          let s = start store in
          let words () = (Gc.quick_stat ()).Gc.major_words in
          let before = words () in
          Session.log_file s ~step:0 ~key:"big" src;
          let allocated =
            (words () -. before) *. Float.of_int (Sys.word_size / 8)
          in
          Session.finish s;
          Sys.remove src;
          less ~msg:"bytes allocated on the major heap" float_exact
            ~than:(Float.of_int (size / 8))
            allocated);
      test "a reading's key cannot become a metric" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          Session.sample s ~every:1000. [ (fun () -> [ ("k", 1.) ]) ];
          raises_match (Exn.invalid_arg ~sub:"reading") (fun () ->
              Session.metric s "k");
          Session.finish s);
    ]

(* Sampling *)

(* [counting f] is a reading that runs [f n] on its [n]th call, from 1, and then
   posts to a semaphore that the test waits on: the sampler's thread is held by
   its calls, never by time. *)
let counting f =
  let calls = Atomic.make 0 and posted = Semaphore.Counting.make 0 in
  let reading () =
    let n = Atomic.fetch_and_add calls 1 + 1 in
    Fun.protect
      ~finally:(fun () -> Semaphore.Counting.release posted)
      (fun () -> f n)
  in
  let await n =
    for _ = 1 to n do
      Semaphore.Counting.acquire posted
    done
  in
  (reading, await, calls)

let sampling_tests =
  group "sampling"
    [
      test "readings are logged as reading records, never orphans" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let m = Session.metric s "m" in
          Session.log m ~step:7 1.;
          let reading, await, _ =
            counting (fun n -> [ ("sys/x", Float.of_int n) ])
          in
          Session.sample s ~every:0.001 [ reading ];
          await 3;
          Session.finish s;
          let h = history store s in
          equal (list string) [ "sys/x" ] (History.readings h);
          let series = History.series h "sys/x" in
          greater ~msg:"several calls" int ~than:2 (Metric.length series);
          equal ~msg:"at the latest step" (list int)
            (List.init (Metric.length series) (fun _ -> 7))
            (List.init (Metric.length series) (Metric.step series)));
      test "a reading that raises Sys_error gives nothing that time" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let reading, await, calls =
            counting (fun n ->
                if n mod 2 = 0 then raise (Sys_error "gone") else [ ("k", 1.) ])
          in
          Session.sample s ~every:0.001 [ reading ];
          await 4;
          Session.finish s;
          let n = Atomic.get calls in
          equal ~msg:"one sample per call that gave one" int
            ((n + 1) / 2)
            (Metric.length (History.series (history store s) "k")));
      test
        "a reading that raises stops the sampler, and finish raises after the \
         end" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let reading, await, _ =
            counting (fun n -> if n > 1 then raise Exit else [])
          in
          Session.sample s ~every:0.001 [ reading ];
          await 2;
          raises Exit (fun () -> Session.finish s);
          equal status `Finished (Run.status (reread store s)));
      test "finish stops the sampler without waiting for its period" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          Session.sample s ~every:3600. [ (fun () -> []) ];
          let t = Unix.gettimeofday () in
          Session.finish s;
          less ~msg:"seconds to finish" float_exact ~than:1.
            (Unix.gettimeofday () -. t));
    ]

(* Statistics hint *)

let key_stats : History.key_stats testable =
  Testable.make
    ~pp:(fun ppf (k : History.key_stats) ->
      Format.fprintf ppf "{count=%d; last=%a}" k.stats.count
        (Testable.pp sample) k.stats.last)
    ~equal:(fun a b -> compare a b = 0)

let hint_path run = Filename.concat (Run.dir run) "stats.json"

let hint_tests =
  group "hint"
    [
      prop ~count:25 "an ended run's hint gives what its fold gives" gen_evs
        (fun evs ->
          with_store @@ fun store ->
          let s = start store in
          let a = Session.metric s ~goal:`Minimize "a" in
          List.iteri
            (fun i e ->
              match e with
              | Sample (_, st, v) ->
                  Session.log a ~step:st ~time:(Float.of_int i) v
              | _ -> ())
            evs;
          Session.finish s;
          let run = reread store s in
          let hinted = History.stats_of run in
          Sys.remove (hint_path run);
          equal (list (pair string key_stats)) (History.stats_of run) hinted);
      prop ~count:40 "a resumed run's hint gives what its fold gives"
        (Gen.triple gen_evs (Gen.int_range 0 20) gen_evs)
        (fun (before, step, after) ->
          cover "the resume orphans a sample"
            (List.exists
               (function Sample (_, s, _) -> s > step | _ -> false)
               before);
          cover "a resume in the log before"
            (List.exists (function Resume _ -> true | _ -> false) before);
          with_store @@ fun store ->
          (* The log a crashed writer left, then a session that resumes it. *)
          let run = spec_run store (log_text before) in
          let s = Session.resume ~provenance run ~step in
          let a = Session.metric s ~goal:`Minimize "a" in
          List.iteri
            (fun i e ->
              match e with
              | Sample (_, st, v) ->
                  Session.log a ~step:st ~time:(Float.of_int i) v
              | _ -> ())
            after;
          Session.finish s;
          let run = require_some (Store.run store "spec") in
          let hinted = History.stats_of run in
          Sys.remove (hint_path run);
          equal (list (pair string key_stats)) (History.stats_of run) hinted);
      test "a damaged hint is ignored and named by check" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          Session.log (Session.metric s "a") ~step:0 1.;
          Session.finish s;
          let run = reread store s in
          let good = History.stats_of run in
          write_file (hint_path run) "{\"format\":1,";
          equal (list (pair string key_stats)) good (History.stats_of run);
          equal (list string)
            [ hint_path run ]
            (List.map fst (Store.check store)));
      test "a hint that disagrees with its log is named by check" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          Session.log (Session.metric s "a") ~step:0 1.;
          Session.finish s;
          let run = reread store s in
          write_file (hint_path run)
            (replace_first
               (read_file (hint_path run))
               ~sub:{|"count":1|} ~by:{|"count":2|});
          equal (list string)
            [ hint_path run ]
            (List.map fst (Store.check store)));
      test "check reads a hint's floats bit for bit" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let a = Session.metric s "a" in
          Session.log a ~step:0 Float.nan;
          Session.log a ~step:1 (-0.);
          Session.finish s;
          let run = reread store s in
          equal ~msg:"a NaN and a -0. agree with their log" (list string) []
            (List.map fst (Store.check store));
          write_file (hint_path run)
            (replace_first
               (read_file (hint_path run))
               ~sub:{|"value":-0.0|} ~by:{|"value":0.0|});
          equal ~msg:"0. disagrees with -0." (list string)
            [ hint_path run ]
            (List.map fst (Store.check store)));
    ]

(* Processes *)

(* [spawn args] runs the support program with a pipe on its standard input, and
   is its pid, the pipe and its standard output. *)
let spawn args =
  let in_r, in_w = Unix.pipe ~cloexec:true () in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process child_exe
      (Array.of_list (child_exe :: args))
      in_r out_w Unix.stderr
  in
  Unix.close in_r;
  Unix.close out_w;
  (pid, in_w, Unix.in_channel_of_descr out_r)

let wait pid =
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED 0 -> ()
  | _ -> fail "the child failed"

(* How a run reads once its writer exited without an end: crashed where the
   writer records the boot it ran under, which a writer on Windows does not
   (Munin.Store_format, status). *)
let dead_status = if Sys.win32 then `Silent else `Crashed

let no_ulimit () =
  if Sys.win32 then skip ~reason:"Windows has no ulimit" ()

let process_tests =
  group "processes"
    [
      test
        "a live writer's run reads running and refuses a resume; a dead one \
         reads crashed under a known boot" (fun () ->
          with_store @@ fun store ->
          let pid, stdin, out = spawn [ "hold"; Store.root store ] in
          let id = input_line out in
          let run = require_some (Store.run store id) in
          equal ~msg:"while it lives" status `Running (Run.status run);
          raises_match (Exn.failure ~sub:"being written") (fun () ->
              Session.resume ~provenance run ~step:0);
          raises_match (Exn.failure ~sub:"being written") (fun () ->
              Store.delete_run store id);
          Unix.close stdin;
          wait pid;
          close_in out;
          let run = require_some (Store.run store id) in
          equal ~msg:"after it exits" status dead_status (Run.status run);
          let s = Session.resume ~provenance run ~step:0 in
          Session.finish s;
          equal ~msg:"resumed and finished" status `Finished
            (Run.status (reread store s)));
      test "a reader sees whole records while a run is written" (fun () ->
          with_store @@ fun store ->
          let pid, stdin, out = spawn [ "log"; Store.root store; "40" ] in
          Unix.close stdin;
          let id = input_line out in
          let run = require_some (Store.run store id) in
          let rec poll h k =
            let h = History.follow h in
            let got =
              List.map
                (fun (s : Metric.sample) -> s.value)
                (samples_of (History.series h "big"))
            in
            equal ~msg:"a prefix" (list float_exact)
              (List.init (List.length got) Float.of_int)
              got;
            if Run.status (History.run h) = `Running && k < 10_000 then
              poll h (k + 1)
            else h
          in
          ignore (poll (History.read run) 0);
          wait pid;
          close_in out;
          equal int 40
            (Metric.length
               (History.series
                  (History.read (require_some (Store.run store id)))
                  "big")));
      test "a file a resume abandoned is an orphan, and check names nothing"
        (fun () ->
          with_store @@ fun store ->
          let src = Filename.temp_file "munin" ".png" in
          write_file src "png";
          let pid, stdin, out = spawn [ "file"; Store.root store; src ] in
          Unix.close stdin;
          let id = input_line out in
          wait pid;
          close_in out;
          let r =
            Session.resume ~provenance
              (require_some (Store.run store id))
              ~step:3
          in
          Session.log_file r ~step:4 ~key:"viz" src;
          Session.finish r;
          Sys.remove src;
          let h = History.read (require_some (Store.run store id)) in
          equal (list int) [ 4 ]
            (List.map (fun (f : History.file) -> f.step) (History.files h));
          equal (list (pair string string)) [] (Store.check store));
      test "a file whose copy fails leaves no temporary" (fun () ->
          no_ulimit ();
          with_store @@ fun store ->
          let src = Filename.temp_file "munin" ".bin" in
          write_file src (String.make 65536 'x');
          let cmd =
            Printf.sprintf "ulimit -f 4; exec %s logfile %s %s"
              (Filename.quote child_exe)
              (Filename.quote (Store.root store))
              (Filename.quote src)
          in
          let ic = Unix.open_process_in cmd in
          let id = input_line ic in
          let line = input_line ic in
          ignore (Unix.close_process_in ic);
          Sys.remove src;
          equal string "Sys_error" line;
          let dir =
            Filename.concat
              (Run.dir (require_some (Store.run store id)))
              "files/big"
          in
          equal (list string) []
            (Array.to_list (try Sys.readdir dir with Sys_error _ -> [||])));
      test "a write the system refuses raises Sys_error" (fun () ->
          no_ulimit ();
          with_store @@ fun store ->
          let src = Filename.temp_file "munin" ".bin" in
          write_file src (String.make 65536 'x');
          (* A limit of 4 blocks of 512 bytes on each file the child writes. *)
          let cmd =
            Printf.sprintf "ulimit -f 4; exec %s fill %s %s"
              (Filename.quote child_exe)
              (Filename.quote (Store.root store))
              (Filename.quote src)
          in
          let ic = Unix.open_process_in cmd in
          let line = input_line ic in
          ignore (Unix.close_process_in ic);
          Sys.remove src;
          equal string "Sys_error" line);
      test "gc waits for a publication in progress" (fun () ->
          with_store @@ fun store ->
          let pid, stdin, out = spawn [ "publishing"; Store.root store ] in
          equal string "held" (input_line out);
          (* gc can only start once the child's lock is gone; the child looks at
             its entry before it lets the lock go. *)
          let gc = Thread.create (fun () -> ignore (Store.gc store)) () in
          Unix.close stdin;
          equal string "intact" (input_line out);
          wait pid;
          close_in out;
          Thread.join gc);
      test "two processes publishing one name claim distinct numbers" (fun () ->
          with_store @@ fun store ->
          let src = Filename.temp_file "munin" ".bin" in
          write_file src "x";
          let a = spawn [ "publish"; Store.root store; "ckpt"; "20"; src ] in
          let b = spawn [ "publish"; Store.root store; "ckpt"; "20"; src ] in
          List.iter
            (fun (pid, stdin, out) ->
              Unix.close stdin;
              wait pid;
              close_in out)
            [ a; b ];
          equal (list int) (List.init 40 succ)
            (List.map Artifact.version (Store.artifacts ~name:"ckpt" store));
          Sys.remove src);
    ]

(* Crashes *)

let crash_tests =
  group "crashes"
    [
      test
        "every byte prefix of a log reads as whole records, and a resume cuts \
         the rest" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          let m = Session.metric s "m" in
          for step = 0 to 9 do
            Session.log m ~step (Float.of_int step)
          done;
          Session.finish s;
          let text =
            read_file
              (Filename.concat (Run.dir (reread store s)) "events.jsonl")
          in
          let first = String.index text '\n' + 1 in
          for n = first to String.length text do
            let run = spec_run store (String.sub text 0 n) in
            let h = History.read run in
            let got =
              List.map
                (fun (x : Metric.sample) -> x.value)
                (samples_of (History.series h "m"))
            in
            equal
              ~msg:(Printf.sprintf "prefix of %d bytes" n)
              (list float_exact)
              (List.init (List.length got) Float.of_int)
              got;
            equal ~msg:"no damage" int 0 (History.damaged h)
          done;
          let run =
            spec_run store (String.sub text 0 (String.length text - 30))
          in
          let r = Session.resume ~provenance run ~step:3 in
          Session.finish r;
          let h = History.read (require_some (Store.run store "spec")) in
          equal ~msg:"orphans left out" (list float_exact) [ 0.; 1.; 2.; 3. ]
            (List.map
               (fun (x : Metric.sample) -> x.value)
               (samples_of (History.series h "m")));
          equal ~msg:"the cut tail is gone" int 0 (History.damaged h));
    ]

(* Stores *)

let store_tests =
  group "store"
    [
      test "opening and reading a missing root creates nothing" (fun () ->
          let root =
            Filename.concat (Filename.get_temp_dir_name ()) "munin-missing-root"
          in
          rm_rf root;
          let store = Store.open_ ~root () in
          equal (list string) [] (List.map Run.id (Store.runs store));
          equal (list string) []
            (List.map Artifact.name (Store.artifacts store));
          equal int 0 (Store.gc store);
          equal bool false (Sys.file_exists root));
      test "tags are equal by their bytes, and change after a run ends"
        (fun () ->
          with_store @@ fun store ->
          let s = start ~tags:[ "best" ] store in
          Session.finish s;
          let id = Session.id s in
          Store.add_tags store id [ "Best"; "caf\xc3\xa9"; "best" ];
          equal (list string)
            [ "Best"; "best"; "caf\xc3\xa9" ]
            (Run.tags (reread store s));
          Store.remove_tags store id [ "best" ];
          equal (list string) [ "Best"; "caf\xc3\xa9" ]
            (Run.tags (reread store s));
          raises_match
            (fun e -> Exn.invalid_arg e)
            (fun () -> Store.add_tags store id [ "a\nb" ]));
      test "notes are replaced whole, and a listing with prev sees the change"
        (fun () ->
          with_store @@ fun store ->
          let s = start store in
          Session.finish s;
          let listing = Store.runs store in
          Store.set_notes store (Session.id s) (Some "first");
          let listing = Store.runs ~prev:listing store in
          equal (option string) (Some "first") (Run.notes (List.hd listing));
          Store.set_notes store (Session.id s) (Some "second");
          Store.add_tags store (Session.id s) [ "t" ];
          let listing = Store.runs ~prev:listing store in
          equal (option string) (Some "second") (Run.notes (List.hd listing));
          equal (list string) [ "t" ] (Run.tags (List.hd listing));
          Store.set_notes store (Session.id s) None;
          equal (option string) None (Run.notes (reread store s)));
      test
        "a listing with prev reads again tags changed within the time \
         resolution" (fun () ->
          with_store @@ fun store ->
          let s = start ~tags:[ "a" ] store in
          Session.finish s;
          let tags = Filename.concat (Run.dir (reread store s)) "tags" in
          (* A file system that keeps whole seconds: the directory's time is
             this second, and another tag leaves its time, size and inode as
             they were. *)
          let second = Float.of_int (truncate (Unix.gettimeofday ())) in
          Unix.utimes tags second second;
          let before = Unix.stat tags in
          let listing = Store.runs store in
          Store.add_tags store (Session.id s) [ "b" ];
          Store.remove_tags store (Session.id s) [ "a" ];
          Unix.utimes tags second second;
          equal ~msg:"the stamp as it was" int before.Unix.st_size
            (Unix.stat tags).Unix.st_size;
          equal (list string) [ "b" ]
            (Run.tags (List.hd (Store.runs ~prev:listing store))));
      test "a listing allocates less than 64 KiB a run" (fun () ->
          (* A buffer of a read chunk for each of a run's small files cost 64
             KiB apiece. *)
          with_store @@ fun store ->
          let runs = 20 in
          for i = 1 to runs do
            let s = start ~tags:[ "a"; string_of_int i ] store in
            Session.log (Session.metric s "loss") ~step:0 1.;
            Session.finish s;
            Store.set_notes store (Session.id s) (Some "notes")
          done;
          ignore (Store.runs store);
          let words () =
            let minor, promoted, major = Gc.counters () in
            minor +. major -. promoted
          in
          let before = words () in
          let listing = Store.runs store in
          let per_run = (words () -. before) /. Float.of_int runs in
          equal ~msg:"every run" int runs (List.length listing);
          less ~msg:"bytes allocated a run" float_exact ~than:65536.
            (per_run *. Float.of_int (Sys.word_size / 8)));
      cases "notes, a start and a failure of any length read back whole"
        ~id:string_of_int
        [ 0; 1; 511; 512; 513; 4096; 65535; 65536; 65537; 200_000 ] (fun n ->
          with_store @@ fun store ->
          let text = String.make n 'x' in
          let s = start ~params:[ ("p", `String text) ] store in
          Session.finish ~status:(`Failed text) s;
          let notes = if n = 0 then None else Some text in
          Store.set_notes store (Session.id s) notes;
          let run = reread store s in
          equal ~msg:"notes" (option string) notes (Run.notes run);
          equal ~msg:"params"
            (list (pair string value))
            [ ("p", `String text) ]
            (Run.params run);
          equal ~msg:"failure" status (`Failed text) (Run.status run));
      test "a run whose log the user cannot write is deleted" (fun () ->
          with_store @@ fun store ->
          let s = start store in
          Session.finish s;
          let dir = Run.dir (reread store s) in
          Unix.chmod (Filename.concat dir "events.jsonl") 0o444;
          Store.delete_run store (Session.id s);
          equal (option string) None
            (Option.map Run.id (Store.run store (Session.id s))));
      test "a silent run is not deleted" (fun () ->
          with_store @@ fun store ->
          (* A run without its writer file, as one copied, reads silent. *)
          let run = spec_run store (log_text []) in
          equal status `Silent (Run.status run);
          raises_match (Exn.failure ~sub:"silent") (fun () ->
              Store.delete_run store "spec"));
      test "a run directory without a log is not a run" (fun () ->
          with_store @@ fun store ->
          (* A start claims its directory before it creates the log. *)
          let runs = Filename.concat (Store.root store) "runs" in
          mkdir_p runs;
          let dir = Filename.concat runs "claimed" in
          mkdir_p dir;
          equal (option string) None
            (Option.map Run.id (Store.run store "claimed"));
          equal (list string) [] (List.map Run.id (Store.runs store));
          equal
            (list (pair string string))
            [ (dir, "not a run: its log holds no whole first line") ]
            (Store.check store);
          equal ~msg:"gc leaves a claimed directory" int 0 (Store.gc store);
          is_true (Sys.file_exists dir));
      test "a run directory that cannot be searched is unreadable" (fun () ->
          if Sys.win32 then
            skip ~reason:"Windows keeps no search permission in a mode" ();
          if Unix.geteuid () = 0 then
            skip ~reason:"root searches any directory" ();
          with_store @@ fun store ->
          let s = start store in
          Session.finish s;
          let id = Session.id s in
          let dir = Run.dir (reread store s) in
          Unix.chmod dir 0o600;
          Fun.protect ~finally:(fun () -> Unix.chmod dir 0o755) @@ fun () ->
          let denied =
            Filename.concat dir "events.jsonl" ^ ": "
            ^ Unix.error_message Unix.EACCES
          in
          raises_match (Exn.failure ~sub:denied) (fun () -> Store.run store id);
          equal
            (list (pair string string))
            [ (dir, denied) ]
            (Store.check store));
      test "what an interrupted deletion leaves is no run, and gc removes it"
        (fun () ->
          with_store @@ fun store ->
          let runs = Filename.concat (Store.root store) "runs" in
          let ended () =
            let s = start ~tags:[ "t" ] store in
            Session.finish s;
            Store.set_notes store (Session.id s) (Some "n");
            (Session.id s, Run.dir (reread store s))
          in
          let kept, _ = ended () in
          (* Cut short under the lock or after it, before the directory moved:
             the log renamed to the mark (Store_format, resume). *)
          let renamed, renamed_dir = ended () in
          Unix.rename
            (Filename.concat renamed_dir "events.jsonl")
            (Filename.concat renamed_dir deletion_mark);
          (* Cut short while it removed the directory in place: the mark
             alone. *)
          let emptied, emptied_dir = ended () in
          Unix.rename
            (Filename.concat emptied_dir "events.jsonl")
            (Filename.concat emptied_dir deletion_mark);
          Array.iter
            (fun n ->
              if n <> deletion_mark then rm_rf (Filename.concat emptied_dir n))
            (Sys.readdir emptied_dir);
          (* Cut short while it removed the moved directory. *)
          let moved, moved_dir = ended () in
          let dot = Filename.concat runs ".moved" in
          Unix.rename
            (Filename.concat moved_dir "events.jsonl")
            (Filename.concat moved_dir deletion_mark);
          Unix.rename moved_dir dot;
          List.iter
            (fun id ->
              equal ~msg:id (option string) None
                (Option.map Run.id (Store.run store id)))
            [ renamed; emptied; moved ];
          equal (list string) [ kept ] (List.map Run.id (Store.runs store));
          equal
            (list (pair string string))
            (List.sort compare
               [
                 (renamed_dir, "left by a deletion");
                 (emptied_dir, "left by a deletion");
                 (dot, "left by a writer");
               ])
            (Store.check store);
          equal ~msg:"removed" int 3 (Store.gc store);
          equal ~msg:"left" (list string) [ kept ]
            (Array.to_list (Sys.readdir runs));
          equal (list (pair string string)) [] (Store.check store));
      test "runs are listed newest first, and a deleted run goes" (fun () ->
          with_store @@ fun store ->
          let a = start ~experiment:"x" store in
          Session.finish a;
          let b = start ~experiment:"y" store in
          Session.finish b;
          (* Newest first by start time, then by id. *)
          let key s =
            let r = reread store s in
            (Run.started_at r, Run.id r)
          in
          let newest =
            List.map snd (List.sort (fun x y -> compare y x) [ key a; key b ])
          in
          equal (list string) newest (List.map Run.id (Store.runs store));
          Store.delete_run store (Session.id a);
          equal (list string)
            [ Session.id b ]
            (List.map Run.id (Store.runs store)));
      test "a free lock reads crashed under the writer's boot, silent elsewhere"
        (fun () ->
          with_store @@ fun store ->
          (* This machine's [writer] file, as a session writes it. *)
          let s = start store in
          let mine =
            read_file (Filename.concat (Run.dir (reread store s)) "writer")
          in
          let s_status = Run.status (reread store s) in
          Session.finish s;
          equal ~msg:"a held lock" status `Running s_status;
          let field k =
            match String.split_on_char '"' mine with
            | l ->
                let rec find = function
                  | k' :: _ :: v :: _ when k' = k -> Some v
                  | _ :: rest -> find rest
                  | [] -> None
                in
                find l
          in
          let run = spec_run store (log_text [ Sample ("a", 0, 1.) ]) in
          let writer = Filename.concat (Run.dir run) "writer" in
          let log = Filename.concat (Run.dir run) "events.jsonl" in
          let status_now () =
            Run.status (require_some (Store.run store "spec"))
          in
          equal ~msg:"no writer file" status `Silent (status_now ());
          (match field "boot" with
          | None -> skip ~reason:"this system names no boot" ()
          | Some boot ->
              write_file writer (Printf.sprintf {|{"boot":"%s"}|} boot);
              equal ~msg:"same boot" status `Crashed (status_now ()));
          write_file writer {|{"machine":"elsewhere","boot":"another"}|};
          equal ~msg:"another machine" status `Silent (status_now ());
          match field "machine" with
          | None -> ()
          | Some machine ->
              write_file writer
                (Printf.sprintf {|{"machine":"%s","boot":"another"}|} machine);
              equal ~msg:"same machine, log written since this boot" status
                `Silent (status_now ());
              Unix.utimes log 1. 1.;
              equal ~msg:"same machine, log older than this boot" status
                `Crashed (status_now ()));
    ]

(* Artifacts *)

let artifact_tests =
  group "artifacts"
    [
      test "a version whose digest is not 64 hexadecimal digits is unreadable"
        (fun () ->
          with_store @@ fun store ->
          let dir = Filename.concat (Store.root store) "artifacts" in
          Unix.mkdir dir 0o755;
          Unix.mkdir (Filename.concat dir "x") 0o755;
          write_file
            (Filename.concat dir "x/1.json")
            {|{"format":1,"time":1.0,"run":"r","digest":"blake2b-256:../../../etc/passwd","size":1,"metadata":{}}|};
          equal (option string) None
            (Option.map Artifact.path (Store.artifact store "x" (`Version 1))));
      test "versions, aliases, latest and deletion" (fun () ->
          with_store @@ fun store ->
          let src = Filename.temp_file "munin" ".bin" in
          write_file src "weights";
          let s = start store in
          let v1 =
            Session.log_artifact s ~step:1
              ~metadata:[ ("f", `String "raw") ]
              ~name:"ckpt" src
          in
          let v2 = Session.log_artifact s ~step:2 ~name:"ckpt" src in
          Session.finish s;
          equal (list int) [ 1; 2 ] [ Artifact.version v1; Artifact.version v2 ];
          equal ~msg:"same bytes share a blob" string (Artifact.path v1)
            (Artifact.path v2);
          equal ~msg:"content" string "weights" (read_file (Artifact.path v1));
          Store.set_alias store v1 "best";
          equal (option int) (Some 1)
            (Option.map Artifact.version
               (Store.artifact store "ckpt" (`Alias "best")));
          equal (option int) (Some 2)
            (Option.map Artifact.version (Store.artifact store "ckpt" `Latest));
          Store.delete_artifact store v2;
          equal ~msg:"latest after a delete" (option int) (Some 1)
            (Option.map Artifact.version (Store.artifact store "ckpt" `Latest));
          let s = start store in
          let v3 = Session.log_artifact s ~name:"ckpt" src in
          Session.finish s;
          equal ~msg:"a deleted number is never reused" int 3
            (Artifact.version v3);
          equal (list (pair string string)) [] (Store.check store);
          Store.delete_artifact store v1;
          Store.delete_artifact store v3;
          equal ~msg:"gc removes the blob no version names" int 1
            (Store.gc store);
          Sys.remove src);
      test
        "a directory is stored with its listing's digest; a linked directory \
         is refused" (fun () ->
          with_store @@ fun store ->
          let dir = Filename.temp_dir "munin" "" in
          Unix.mkdir (Filename.concat dir "sub") 0o755;
          write_file (Filename.concat dir "a.txt") "a";
          write_file (Filename.concat dir "sub/b") "b";
          let s = start store in
          let v = Session.log_artifact s ~name:"data" dir in
          equal string "b"
            (read_file (Filename.concat (Artifact.path v) "sub/b"));
          equal int 2 (Artifact.size_bytes v);
          Unix.symlink (Filename.concat dir "sub") (Filename.concat dir "link");
          raises_match (Exn.invalid_arg ~sub:"links to a directory") (fun () ->
              Session.log_artifact s ~name:"data" dir);
          Session.finish s;
          write_file (Filename.concat (Artifact.path v) "a.txt") "changed";
          equal ~msg:"check sees a changed blob" (list string)
            [ Artifact.path v ]
            (List.map fst (Store.check store));
          rm_rf dir);
    ]

(* Provenance *)

let git args cwd =
  Unix.system
    (Filename.quote_command "git" ("-C" :: cwd :: args) ~stdout:Filename.null
       ~stderr:Filename.null)

let provenance_tests =
  group "provenance"
    [
      test "two environment names that escape alike are refused" (fun () ->
          Unix.putenv "MUNIN_TEST_\xff" "1";
          Unix.putenv "MUNIN_TEST_\xfe" "2";
          raises_match (Exn.invalid_arg ~sub:"capture_env") (fun () ->
              Provenance.detect ~cwd:"/"
                ~capture_env:[ "MUNIN_TEST_\xff"; "MUNIN_TEST_\xfe" ]
                ()));
      test "a clean tree, a dirty tree and no tree" (fun () ->
          let dir = Filename.temp_dir "munin-git" "" in
          Fun.protect ~finally:(fun () -> rm_rf dir) @@ fun () ->
          let outside = Provenance.detect ~cwd:dir () in
          equal ~msg:"outside a tree" bool true (outside.git = None);
          if git [ "init"; "-q" ] dir <> Unix.WEXITED 0 then
            skip ~reason:"no git" ();
          write_file (Filename.concat dir "f") "1";
          ignore (git [ "add"; "f" ] dir);
          ignore
            (git
               [
                 "-c";
                 "user.name=t";
                 "-c";
                 "user.email=t@t";
                 "commit";
                 "-qm";
                 "c";
               ]
               dir);
          let clean = Provenance.detect ~cwd:dir () in
          equal ~msg:"clean" (option bool) (Some false)
            (Option.map (fun (g : Provenance.git) -> g.dirty) clean.git);
          write_file (Filename.concat dir "f") "2";
          let dirty = Provenance.detect ~cwd:dir ~seeds:[ ("init", 7) ] () in
          equal ~msg:"dirty" (option bool) (Some true)
            (Option.map (fun (g : Provenance.git) -> g.dirty) dirty.git);
          with_store @@ fun store ->
          let s = Session.start ~store ~provenance:dirty ~experiment:"e" () in
          Session.finish s;
          let run = reread store s in
          let diff = read_file (require_some (Run.diff run)) in
          contains ~sub:"+2" diff;
          equal
            (list (pair string int))
            [ ("init", 7) ]
            (Run.provenance run).seeds);
    ]

(* A store against a model *)

(* The operations of a program on two runs, and the model of what a reader must
   then see: each run's samples, tags, notes, summary and end. *)
type op =
  | Log of int * int * float
  | Tag of int * string
  | Untag of int * string
  | Note of int * string option
  | Summary of int * string * float
  | Finish of int

let pp_op ppf = function
  | Log (r, s, v) -> Format.fprintf ppf "log r%d %d %g" r s v
  | Tag (r, t) -> Format.fprintf ppf "tag r%d %s" r t
  | Untag (r, t) -> Format.fprintf ppf "untag r%d %s" r t
  | Note (r, n) ->
      Format.fprintf ppf "note r%d %s" r (Option.value ~default:"-" n)
  | Summary (r, k, v) -> Format.fprintf ppf "summary r%d %s=%g" r k v
  | Finish r -> Format.fprintf ppf "finish r%d" r

let gen_op =
  let open Gen in
  let r = int_range 0 1 and tag = of_list [ "a"; "A"; "b" ] in
  frequency
    [
      ( 6,
        map
          (fun ((r, s), v) -> Log (r, s, v))
          (pair (pair r (int_range 0 9)) (float_range (-1.) 1.)) );
      (2, map (fun (r, t) -> Tag (r, t)) (pair r tag));
      (1, map (fun (r, t) -> Untag (r, t)) (pair r tag));
      ( 1,
        map
          (fun (r, n) -> Note (r, n))
          (pair r (option (of_list [ "x"; "y\nz" ]))) );
      ( 1,
        map
          (fun ((r, k), v) -> Summary (r, k, v))
          (pair (pair r (of_list [ "k"; "l" ])) (float_range 0. 1.)) );
      (1, map (fun r -> Finish r) r);
    ]

type model = {
  values : float list; (* reversed *)
  tags : string list;
  notes : string option;
  summary : (string * float) list;
  ended : bool;
}

let model_tests =
  group "model"
    [
      prop ~count:40 "a store's readers see what its writers did"
        (Gen.list ~size:(Gen.int_range 0 30) (Gen.with_pp pp_op gen_op))
        (fun ops ->
          with_store @@ fun store ->
          let sessions = Array.init 2 (fun _ -> start store) in
          let metrics = Array.map (fun s -> Session.metric s "m") sessions in
          let model =
            Array.make 2
              {
                values = [];
                tags = [];
                notes = None;
                summary = [];
                ended = false;
              }
          in
          let id r = Session.id sessions.(r) in
          List.iter
            (fun op ->
              let r =
                match op with
                | Log (r, _, _)
                | Tag (r, _)
                | Untag (r, _)
                | Note (r, _)
                | Summary (r, _, _)
                | Finish r ->
                    r
              in
              let m = model.(r) in
              match op with
              | Log (_, step, v) when not m.ended ->
                  Session.log metrics.(r) ~step v;
                  model.(r) <- { m with values = v :: m.values }
              | Summary (_, k, v) when not m.ended ->
                  Session.set_summary sessions.(r) [ (k, `Float v) ];
                  model.(r) <-
                    { m with summary = (k, v) :: List.remove_assoc k m.summary }
              | Finish _ when not m.ended ->
                  Session.finish sessions.(r);
                  model.(r) <- { m with ended = true }
              | Log _ | Summary _ | Finish _ -> ()
              | Tag (_, t) ->
                  Store.add_tags store (id r) [ t ];
                  model.(r) <-
                    { m with tags = List.sort_uniq compare (t :: m.tags) }
              | Untag (_, t) ->
                  Store.remove_tags store (id r) [ t ];
                  model.(r) <- { m with tags = List.filter (( <> ) t) m.tags }
              | Note (_, n) ->
                  Store.set_notes store (id r) n;
                  model.(r) <- { m with notes = n })
            ops;
          let listing = Store.runs store in
          Array.iteri
            (fun r m ->
              let run = List.find (fun x -> Run.id x = id r) listing in
              equal ~msg:"tags" (list string) m.tags (Run.tags run);
              equal ~msg:"notes" (option string) m.notes (Run.notes run);
              equal ~msg:"status" status
                (if m.ended then `Finished else `Running)
                (Run.status run);
              let h = History.read run in
              equal ~msg:"samples" (list float_exact) (List.rev m.values)
                (List.map
                   (fun (x : Metric.sample) -> x.value)
                   (samples_of (History.series h "m")));
              equal ~msg:"summary"
                (list (pair string value))
                (List.sort compare
                   (List.map (fun (k, v) -> (k, `Float v)) m.summary))
                (History.summary h))
            model;
          Array.iter (fun s -> Session.finish s) sessions);
    ]

(* Histories across domains *)

(* A log of [n] samples of "a", 0 to n - 1, read a line at a time with [follow
   ~limit:1]: a history's reference is how many samples it holds. *)
let domain_log =
  lazy
    (let root = Filename.temp_dir "munin-domains" "" in
     at_exit (fun () -> rm_rf root);
     let store = Store.open_ ~root () in
     let evs = List.init 40 (fun i -> Sample ("a", i, Float.of_int i)) in
     (* No metric record: each line after the start is one sample. *)
     spec_run store
       (String.concat ""
          (List.map (fun l -> l ^ "\n") (start_line :: List.mapi ev_line evs))))

let n_samples = 40
let hist_w : (int, History.t) abstract = abstract "h"

let held h =
  List.map
    (fun (x : Metric.sample) -> x.value)
    (samples_of (History.series h "a"))

(* Runs that two domains delete and annotate at once. A run's reference is
   whether it exists; its system side is its id. A call the store refuses
   answers false. *)
let race_store =
  lazy
    (let root = Filename.temp_dir "munin-race" "" in
     at_exit (fun () -> rm_rf root);
     Store.open_ ~root ())

(* Once the calls end, a run's directory holds its log, or the mark of a
   deletion that gc finishes, or is gone. A directory with neither is one that
   nothing removes. *)
let no_stray id =
  let runs = Filename.concat (Store.root (Lazy.force race_store)) "runs" in
  let dir = Filename.concat runs id in
  if
    Sys.file_exists dir
    && not
         (Sys.file_exists (Filename.concat dir "events.jsonl")
         || Sys.file_exists (Filename.concat dir deletion_mark))
  then
    fail
      (Printf.sprintf "%s holds neither a log nor a deletion's mark: %s" dir
         (String.concat " " (Array.to_list (Sys.readdir dir))))

let run_w : (bool ref, string) abstract = abstract ~release:no_stray "r"

let accepted f =
  match f () with () -> true | exception (Failure _ | Sys_error _) -> false

(* [annotate tag store id] adds a tag to the run [id], or sets its notes. *)
let annotate tag store id =
  if tag then Store.add_tags store id [ "t" ]
  else Store.set_notes store id (Some "n")

let domain_tests =
  group "domains"
    [
      stateful ~count:6 ~domains:2
        "a deletion and annotations on two domains leave the run whole or gone"
        [
          command "start"
            (Gen.unit @-> makes run_w)
            (fun () -> ref true)
            (fun () ->
              let s = start (Lazy.force race_store) in
              Session.finish s;
              Session.id s);
          command "delete" (run_w ^-> returns bool)
            (fun r ->
              let was = !r in
              r := false;
              was)
            (fun id ->
              accepted (fun () -> Store.delete_run (Lazy.force race_store) id));
          command "annotate"
            (Gen.bool @-> run_w ^-> returns bool)
            (fun _ r -> !r)
            (fun tag id ->
              accepted (fun () -> annotate tag (Lazy.force race_store) id));
          command "exists" (run_w ^-> returns bool)
            (fun r -> !r)
            (fun id -> Store.run (Lazy.force race_store) id <> None);
        ];
      stateful ~count:30 ~domains:2
        "following one history on two domains leaves both right"
        [
          (* A limit of one byte ends within the start: no sample. *)
          command "read"
            (Gen.unit @-> makes hist_w)
            (fun () -> 0)
            (fun () -> History.read ~limit:1 (Lazy.force domain_log));
          command "follow"
            (hist_w ^-> makes hist_w)
            (fun c -> min (c + 1) n_samples)
            (fun h -> History.follow ~limit:1 h);
          command "samples"
            (hist_w ^-> returns (list float_exact))
            (fun c -> List.init c Float.of_int)
            held;
        ];
    ]

let () =
  exit
    (run "munin"
       [
         values_tests;
         float_text_tests;
         fold_tests;
         session_tests;
         sampling_tests;
         hint_tests;
         process_tests;
         crash_tests;
         store_tests;
         artifact_tests;
         provenance_tests;
         model_tests;
         domain_tests;
       ])
