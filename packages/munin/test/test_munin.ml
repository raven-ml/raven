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

(* Windows removes no read-only file, such as git's objects, until its attribute
   is cleared. *)
let rec rm_rf p =
  match Unix.lstat p with
  | exception Unix.Unix_error _ -> ()
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (fun e -> rm_rf (Filename.concat p e)) (Sys.readdir p);
      Unix.rmdir p
  | _ ->
      if Sys.win32 then Unix.chmod p 0o644;
      Sys.remove p

(* A new store in the test's own directory. *)
let new_store () = Store.open_ ~root:(temp_dir ~prefix:"munin" ()) ()

let start ?(experiment = "e") ?name ?parent ?params ?tags ?steps store =
  Session.start ~store ~provenance ?name ?parent ?params ?tags ?steps
    ~experiment ()

let reread store s = require_some (Store.run store (Session.id s))
let history store s = History.read (reread store s)
let log_path run = Filename.concat (Run.dir run) "events.jsonl"

let write_file path data =
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc data)

let read_file path = In_channel.with_open_bin path In_channel.input_all

(* [source data] is a new file of the test that holds [data]. *)
let source ?(suffix = ".bin") data =
  let p = temp_file ~suffix () in
  write_file p data;
  p

let mkdir_p d =
  try Unix.mkdir d 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()

let lines text = List.filter (( <> ) "") (String.split_on_char '\n' text)

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

(* The name a deletion renames a run's log to (Store_format, resume). *)
let deletion_mark = ".deleted"

(* The kernel's name, as [uname -s] prints it. *)
let kernel =
  lazy
    (if Sys.win32 then "Windows"
     else
       let ic = Unix.open_process_args_in "uname" [| "uname"; "-s" |] in
       let name = input_line ic in
       match Unix.close_process_in ic with
       | WEXITED 0 -> name
       | _ -> failwith "uname -s failed")

(* How a run reads once its writer exited without an end: crashed where the
   writer records the boot it ran under, which only Linux and macOS name
   (Store_format, status). *)
let dead_status () =
  match Lazy.force kernel with "Linux" | "Darwin" -> `Crashed | _ -> `Silent

(* Witnesses *)

let value : Value.t testable =
  Testable.make ~pp:Value.pp ~equal:(fun a b ->
      match (a, b) with
      | `Float x, `Float y -> Testable.equal float_exact x y
      | a, b -> a = b)

let pp_float = Testable.pp float_exact
let same_float = Testable.equal float_exact

let pp_sample ppf (s : Metric.sample) =
  Format.fprintf ppf "{step=%d; time=%a; value=%a}" s.step pp_float s.time
    pp_float s.value

let same_sample (a : Metric.sample) (b : Metric.sample) =
  a.step = b.step && same_float a.time b.time && same_float a.value b.value

let sample : Metric.sample testable =
  Testable.make ~pp:pp_sample ~equal:same_sample

let samples_of s = List.init (Metric.length s) (Metric.sample s)
let values_of s = List.init (Metric.length s) (Metric.value s)

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

(* Statistics compared float by float, bit for bit, as a reader computes
   them. *)
let pp_option pp ppf = function
  | None -> Format.pp_print_string ppf "None"
  | Some x -> Format.fprintf ppf "Some %a" pp x

let pp_stats ppf (st : Metric.stats) =
  Format.fprintf ppf "{count=%d; last=%a; finite=%a; non_finite=%a}" st.count
    pp_sample st.last
    (pp_option (fun ppf (f : Metric.finite) ->
         Format.fprintf ppf "{least=%a; greatest=%a; mean=%a}" pp_sample f.least
           pp_sample f.greatest pp_float f.mean))
    st.finite
    (pp_option (fun ppf (n, s) -> Format.fprintf ppf "(%d, %a)" n pp_sample s))
    st.non_finite

let same_stats (a : Metric.stats) (b : Metric.stats) =
  a.count = b.count && same_sample a.last b.last
  && (match (a.finite, b.finite) with
    | None, None -> true
    | Some f, Some g ->
        same_sample f.least g.least
        && same_sample f.greatest g.greatest
        && same_float f.mean g.mean
    | _ -> false)
  &&
  match (a.non_finite, b.non_finite) with
  | None, None -> true
  | Some (n, s), Some (m, t) -> n = m && same_sample s t
  | _ -> false

let key_stats : History.key_stats testable =
  Testable.make
    ~pp:(fun ppf (k : History.key_stats) ->
      Format.fprintf ppf "{kind=%s; goal=%s; stats=%a}"
        (match k.kind with `Metric -> "metric" | `Reading -> "reading")
        (match k.goal with
        | None -> "None"
        | Some `Minimize -> "minimize"
        | Some `Maximize -> "maximize")
        pp_stats k.stats)
    ~equal:(fun (a : History.key_stats) b ->
      a.kind = b.kind && a.goal = b.goal && same_stats a.stats b.stats)

(* Generators *)

let gen_float = Gen.any_float

let gen_value : Value.t Gen.t =
  Gen.one_of
    [
      Gen.map (fun b -> `Bool b) Gen.bool;
      Gen.map (fun n -> `Int n) Gen.int;
      Gen.map (fun n -> `Int n) (Gen.int_range (-(1 lsl 54)) (1 lsl 54));
      Gen.map (fun x -> `Float x) gen_float;
      (* [any_float] is non-finite in one draw of ten: one value in 70, which
         100 lists of up to 8 values miss once in 300 runs. This arm makes it 11
         values in 80, or 40% of the lists. *)
      Gen.map
        (fun x -> `Float x)
        (Gen.of_list [ Float.nan; Float.infinity; Float.neg_infinity ]);
      Gen.map (fun x -> `Float (Float.of_int x)) Gen.small_int;
      Gen.map (fun s -> `String s) (Gen.string_of (Gen.char_range ' ' '~'));
      Gen.constant (`String "tab\tline\nfeed \xc3\xa9 \"q\" \\");
    ]

(* Logs written from the format's text alone *)

(* The records a test writes by hand: a sample of a metric or of a reading, as
   key, step and value, and a resume at a step. *)
type record =
  | Metrics of string * int * float
  | Reading of string * int * float
  | Resume of int

let pp_record ppf (time, r) =
  match r with
  | Metrics (k, s, v) -> Format.fprintf ppf "metrics %s@%d=%h at %g" k s v time
  | Reading (k, s, v) -> Format.fprintf ppf "reading %s@%d=%h at %g" k s v time
  | Resume s -> Format.fprintf ppf "resume %d at %g" s time

let pp_records = Format.pp_print_list ~pp_sep:Format.pp_print_space pp_record

(* A float as the format writes it: with a fraction or an exponent, and the
   objects that stand for NaN and the infinities (Store_format, values). *)
let json_float x =
  if Float.is_nan x then {|{"float":"NaN"}|}
  else if x = Float.infinity then {|{"float":"Infinity"}|}
  else if x = Float.neg_infinity then {|{"float":"-Infinity"}|}
  else
    let s = Printf.sprintf "%.17g" x in
    if String.exists (fun c -> c = '.' || c = 'e') s then s else s ^ ".0"

let prov_json =
  {|{"command":["t"],"cwd":"/","pid":1,"ocaml":"5","raven":"t","os":"t","arch":"t","seeds":{},"env":{}}|}

let start_line_at time =
  Printf.sprintf
    {|{"type":"start","time":%s,"format":1,"experiment":"spec","params":{},"provenance":%s}|}
    (json_float time) prov_json

let start_line = start_line_at 1.

let metric_line (k, goal) =
  Printf.sprintf {|{"type":"metric","time":1.5,"key":"%s","goal":"%s"}|} k
    (match goal with `Minimize -> "minimize" | `Maximize -> "maximize")

let record_line (time, r) =
  match r with
  | Metrics (k, s, v) ->
      Printf.sprintf
        {|{"type":"metrics","time":%s,"step":%d,"values":{"%s":%s}}|}
        (json_float time) s k (json_float v)
  | Reading (k, s, v) ->
      Printf.sprintf
        {|{"type":"reading","time":%s,"step":%d,"values":{"%s":%s}}|}
        (json_float time) s k (json_float v)
  | Resume s ->
      Printf.sprintf {|{"type":"resume","time":%s,"step":%d,"provenance":%s}|}
        (json_float time) s prov_json

let text_of_lines ls = String.concat "" (List.map (fun l -> l ^ "\n") ls)

(* [log_text records] is a log: its start, a [metric] record of each of
   [metrics], then [records], each at its time. *)
let log_text ?(metrics = [ ("a", `Minimize); ("b", `Minimize) ]) records =
  text_of_lines
    ((start_line :: List.map metric_line metrics) @ List.map record_line records)

(* [timed records] dates the [i]th record [i + 2]. *)
let timed records = List.mapi (fun i r -> (Float.of_int (i + 2), r)) records

(* [written_run store text] is the run [id] of [store] whose log is [text], as a
   writer that exited left it, without its [writer] file. *)
let written_run ?(id = "spec") store text =
  let runs = Filename.concat (Store.root store) "runs" in
  mkdir_p (Store.root store);
  mkdir_p runs;
  let dir = Filename.concat runs id in
  rm_rf dir;
  mkdir_p dir;
  write_file (Filename.concat dir "events.jsonl") text;
  require_some (Store.run store id)

(* The reference: the format's rules read off the record list. A sample of a
   metric is an orphan, and a reading's sample abandoned, when a resume after it
   names a step below its own (Store_format, reading). *)
let gone_back records i s =
  List.exists
    (function _, Resume r -> r < s | _ -> false)
    (List.filteri (fun j _ -> j > i) records)

let series_of records k =
  List.concat
    (List.mapi
       (fun i (time, r) ->
         match r with
         | Metrics (k', step, value)
           when k' = k && not (gone_back records i step) ->
             [ ({ step; time; value } : Metric.sample) ]
         | Reading (k', step, value) when k' = k -> [ { step; time; value } ]
         | _ -> [])
       records)

let surviving_of records k =
  List.filter_map
    (fun x -> x)
    (List.mapi
       (fun i (time, r) ->
         match r with
         | Reading (k', step, value)
           when k' = k && not (gone_back records i step) ->
             Some ({ step; time; value } : Metric.sample)
         | _ -> None)
       records)

let gen_record =
  Gen.frequency
    [
      ( 8,
        Gen.map
          (fun ((k, s), v) -> Metrics (k, s, v))
          Gen.(
            pair
              (* "b" draws few samples, which a resume orphans all of. *)
              (pair
                 (frequency [ (3, constant "a"); (1, constant "b") ])
                 (int_range 0 20))
              (frequency
                 [
                   (6, float_range (-5.) 5.);
                   (1, gen_float);
                   (1, of_list [ 1.; -0.; 0. ]);
                   (* Values whose sums pass the largest float, and one whose
                      addition to it rounds away. *)
                   (1, of_list [ Float.max_float; -.Float.max_float; 0x1p969 ]);
                 ])) );
      (1, Gen.map (fun v -> Reading ("r", 0, v)) (Gen.float_range 0. 100.));
      (* A resume at step 0 orphans every earlier sample but those of step 0. *)
      ( 1,
        Gen.map
          (fun s -> Resume s)
          Gen.(frequency [ (1, constant 0); (2, int_range 0 20) ]) );
    ]

let gen_records =
  Gen.with_pp
    (fun ppf rs -> pp_records ppf (timed rs))
    (Gen.list ~size:(Gen.int_range 0 60) gen_record)

(* [gen_records] half the time; a third of the time a log whose last record
   resumes below every step of one key, so that each of its samples is an
   orphan: its steps move up one, past step 0, and it gets a sample if it had
   none; and otherwise a log that ends with a sample that is not finite and two
   whose sum passes the largest float. *)
let gen_orphaning =
  let orphan_all k rs =
    let rs =
      List.map
        (function
          | Metrics (k', s, v) when k' = k -> Metrics (k', s + 1, v) | r -> r)
        rs
    in
    let steps =
      List.filter_map
        (function Metrics (k', s, _) when k' = k -> Some s | _ -> None)
        rs
    in
    let rs, least =
      match steps with
      | [] -> (rs @ [ Metrics (k, 1, 0.) ], 1)
      | s :: rest -> (rs, List.fold_left min s rest)
    in
    rs @ [ Resume (least - 1) ]
  in
  Gen.(
    frequency
      [
        (3, gen_records);
        ( 2,
          let* k = of_list [ "a"; "b" ] in
          let+ rs = gen_records in
          orphan_all k rs );
        ( 1,
          let* v = of_list [ Float.nan; Float.infinity; Float.neg_infinity ] in
          let* s = int_range 0 20 in
          let+ rs = gen_records in
          rs
          @ [
              Metrics ("a", s, v);
              Metrics ("a", s, Float.max_float);
              Metrics ("a", s, Float.max_float);
            ] );
      ])

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
          (* Strict comparisons keep the earliest of equals and take -0. for 0.
             (Metric, stats). *)
          let pick better =
            List.fold_left
              (fun (b : Metric.sample) (s : Metric.sample) ->
                if better s.value b.value then s else b)
              (List.hd f) (List.tl f)
          in
          equal ~msg:(msg ^ " least") sample (pick ( < )) fs.least;
          equal ~msg:(msg ^ " greatest") sample (pick ( > )) fs.greatest;
          (* The mean by its definition, compared at half scale: each term is
             divided by n and by 2 first, so that no partial sum's rounding
             takes it past the largest float. That naive sum of at most 63 terms
             errs by at most 63 ulps of the largest term, below 1e-12 of it,
             plus a subnormal's rounding per term, below 1e-300 in all. The
             compensated mean is within an ulp of the true one, so no relative
             bound is needed. *)
          let n = Float.of_int (List.length f) in
          let half_mean =
            List.fold_left
              (fun a (s : Metric.sample) -> a +. (s.value /. n /. 2.))
              0. f
          in
          let scale =
            List.fold_left
              (fun a (s : Metric.sample) -> Float.max a (Float.abs s.value))
              0. f
          in
          equal ~msg:(msg ^ " mean")
            (float_rel ~rel:0. ~abs:(Float.max 1e-300 (1e-12 *. scale)))
            half_mean (fs.mean /. 2.);
          (* Adding 0. reads -0. as 0. *)
          at_least
            ~msg:(msg ^ " mean above least")
            float_exact ~than:(fs.least.value +. 0.) (fs.mean +. 0.);
          at_most
            ~msg:(msg ^ " mean below greatest")
            float_exact ~than:(fs.greatest.value +. 0.) (fs.mean +. 0.));
      match (non_finite, st.non_finite) with
      | [], None -> ()
      | (s :: _ as l), Some (n, first) ->
          equal ~msg:(msg ^ " non-finite count") int (List.length l) n;
          equal ~msg:(msg ^ " first non-finite") sample s first
      | _ -> fail (msg ^ ": non-finite statistics disagree"))

(* Whether no sample's step is below the one before. *)
let ordered samples =
  let rec go = function
    | (a : Metric.sample) :: (b :: _ as rest) -> a.step <= b.step && go rest
    | _ -> true
  in
  go samples

let check_history ~msg records h =
  List.iter
    (fun k ->
      let expected = series_of records k in
      equal
        ~msg:(msg ^ " series " ^ k)
        (list sample) expected
        (samples_of (History.series h k));
      equal
        ~msg:(msg ^ " ordered " ^ k)
        bool (ordered expected)
        (Metric.ordered (History.series h k));
      check_stats ~msg:(msg ^ " " ^ k) expected (History.stats h k))
    [ "a"; "b"; "r" ];
  let with_samples = List.filter (fun k -> series_of records k <> []) in
  equal ~msg:(msg ^ " metrics") (list string)
    (with_samples [ "a"; "b" ])
    (History.metrics h);
  equal ~msg:(msg ^ " readings") (list string) (with_samples [ "r" ])
    (History.readings h)

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

(* [exited_run store] is a run whose writer, another process, exited without an
   end. *)
let exited_run store =
  let pid, stdin, out = spawn [ "hold"; Store.root store ] in
  let id = input_line out in
  Unix.close stdin;
  wait pid;
  close_in out;
  require_some (Store.run store id)

(* [limited args] runs the support program under a limit of 4 blocks of 512
   bytes on each file it writes, and is the lines it printed. *)
let limited args =
  if Sys.win32 then skip ~reason:"Windows has no ulimit" ();
  let cmd =
    Printf.sprintf "ulimit -f 4; exec %s"
      (String.concat " " (List.map Filename.quote (child_exe :: args)))
  in
  let ic = Unix.open_process_in cmd in
  let out = In_channel.input_all ic in
  ignore (Unix.close_process_in ic);
  lines out

(* Store_format *)

let numbers_tests =
  group "numbers and values"
    [
      cases "a param is written as the format writes its number"
        ~id:(fun (id, _, _) -> id)
        [
          (* Store_format, values: "[0.1], [1.0], [-0.0], [1e+16], [5e-324]",
             the objects of NaN and the infinities, and an integer past 2^53 as
             [{"int":"<decimal>"}]. *)
          ("0.1", `Float 0.1, "0.1");
          ("1.0", `Float 1., "1.0");
          ("-0.0", `Float (-0.), "-0.0");
          ("1e+16", `Float 1e16, "1e+16");
          ("5e-324", `Float 5e-324, "5e-324");
          ("NaN", `Float Float.nan, {|{"float":"NaN"}|});
          ("Infinity", `Float Float.infinity, {|{"float":"Infinity"}|});
          ("-Infinity", `Float Float.neg_infinity, {|{"float":"-Infinity"}|});
          ("2^53", `Int (1 lsl 53), "9007199254740992");
          ("2^53 + 1", `Int ((1 lsl 53) + 1), {|{"int":"9007199254740993"}|});
          ("-2^53 - 1", `Int (-(1 lsl 53) - 1), {|{"int":"-9007199254740993"}|});
        ]
        (fun (_, v, text) ->
          let store = new_store () in
          let s = start ~params:[ ("p", v) ] store in
          Session.finish s;
          let first = List.hd (lines (read_file (log_path (reread store s)))) in
          contains ~sub:({|"params":{"p":|} ^ text ^ "}") first);
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
          let store = new_store () in
          let params = List.mapi (fun i v -> (Printf.sprintf "p%d" i, v)) vs in
          let s = start ~params store in
          Session.finish s;
          equal (list (pair string value)) params (Run.params (reread store s)));
      prop "samples read back equal"
        (Gen.list ~size:(Gen.int_range 1 20) (Gen.pair gen_float gen_float))
        (fun pairs ->
          cover "NaN" (List.exists (fun (_, v) -> Float.is_nan v) pairs);
          let store = new_store () in
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
    ]

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

let canonical_line (reading, time, step, values) =
  Printf.sprintf {|{"type":"%s","time":%s,"step":%d,"values":{%s}}|}
    (if reading then "reading" else "metrics")
    (json_float time) step
    (String.concat ","
       (List.map (fun (k, v) -> json_key k ^ ":" ^ json_float v) values))

let spaced_line (reading, time, step, values) =
  Printf.sprintf
    {|{ "values" : { %s } , "step" : %d , "time" : %s , "type" : "%s" }|}
    (String.concat " , "
       (List.map (fun (k, v) -> escaped_key k ^ " : " ^ json_float v) values))
    step (json_float time)
    (if reading then "reading" else "metrics")

(* A damaged line of each kind (Store_format, reading). *)
let damage =
  [
    ( "invalid UTF-8",
      "{\"type\":\"metrics\",\"time\":1.0,\"step\":0,\"values\":{\"a\":\xff}}"
    );
    ( "a repeated member",
      {|{"type":"metrics","type":"metrics","time":1.0,"step":0,"values":{}}|} );
    (* Any nesting past the format's depth is the same case. *)
    ("deep nesting", String.make 1_000 '[');
    ("an unknown type", {|{"type":"unknown","time":1.0}|});
    ( "a wrong type",
      {|{"type":"metrics","time":"x","step":0,"values":{"a":1.0}}|} );
    ( "a negative step",
      {|{"type":"metrics","time":1.0,"step":-1,"values":{"a":1.0}}|} );
    ( "a step past 2^53",
      {|{"type":"metrics","time":1.0,"step":9007199254740993,"values":{"a":1.0}}|}
    );
    ( "a bad key",
      {|{"type":"metrics","time":1.0,"step":0,"values":{"../a":1.0}}|} );
    ("not JSON", "not json at all");
    ( "a repeated key",
      {|{"type":"metrics","time":1.0,"step":0,"values":{"a":1.0,"a":2.0}}|} );
    ( "a leading zero",
      {|{"type":"metrics","time":1.0,"step":01,"values":{"a":1.0}}|} );
    ( "a number cut short",
      {|{"type":"metrics","time":1.0,"step":0,"values":{"a":1.}}|} );
    ( "a control character in a key",
      "{\"type\":\"metrics\",\"time\":1.0,\"step\":0,\"values\":{\"a\tb\":1.0}}"
    );
    (* munin never writes one, and it decodes to no UTF-8. *)
    ( "a lone surrogate escape in a key",
      {|{"type":"metrics","time":1.0,"step":0,"values":{"a\ud800":1.0}}|} );
    ( "trailing bytes",
      {|{"type":"metrics","time":1.0,"step":0,"values":{"a":1.0}}x|} );
  ]

let reading_tests =
  group "reading a log"
    [
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
          let store = new_store () in
          let read layout =
            History.read
              (written_run store
                 (text_of_lines (start_line :: List.map layout records)))
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
      prop "a damaged line is skipped and counted"
        (Gen.pair gen_records (Gen.int_range 0 1000))
        (fun (records, at) ->
          let store = new_store () in
          let lines = List.map record_line (timed records) in
          let at = at mod (List.length lines + 1) in
          let text l =
            text_of_lines
              (start_line
              :: metric_line ("a", `Minimize)
              :: metric_line ("b", `Minimize)
              :: l)
          in
          let clean = History.read (written_run store (text lines)) in
          (* Every kind of damage in every case, so that every run checks
             each. *)
          List.iter
            (fun (kind, garbage) ->
              let damaged =
                List.filteri (fun i _ -> i < at) lines
                @ [ garbage ]
                @ List.filteri (fun i _ -> i >= at) lines
              in
              let h = History.read (written_run store (text damaged)) in
              equal ~msg:(kind ^ ": counted") int 1 (History.damaged h);
              equal ~msg:(kind ^ ": skipped") (list sample)
                (samples_of (History.series clean "a"))
                (samples_of (History.series h "a")))
            damage);
      test "members a record's kind does not define are ignored" (fun () ->
          let store = new_store () in
          let line =
            {|{"type":"metrics","note":{"x":[1,2]},"time":2.0,"step":3,"values":{"a":1.5},"z":null}|}
          in
          let h =
            History.read
              (written_run store (text_of_lines [ start_line; line ]))
          in
          equal int 0 (History.damaged h);
          equal (list sample)
            [ { step = 3; time = 2.; value = 1.5 } ]
            (samples_of (History.series h "a")));
      test "a line with a NUL byte stops the reader until it is filled"
        (fun () ->
          let store = new_store () in
          let good = log_text [ (2., Metrics ("a", 1, 1.)) ] in
          let line = record_line (3., Metrics ("a", 2, 2.)) ^ "\n" in
          let hole = String.make (String.length line - 1) '\000' ^ "\n" in
          let after = record_line (4., Metrics ("a", 3, 3.)) ^ "\n" in
          let run = written_run store (good ^ hole ^ after) in
          let h = History.read run in
          equal ~msg:"stops before the hole" (list float_exact) [ 1. ]
            (values_of (History.series h "a"));
          write_file (log_path run) (good ^ line ^ after);
          equal ~msg:"reads on once filled" (list float_exact) [ 1.; 2.; 3. ]
            (values_of (History.series (History.follow h) "a")));
      test "a sample of a key of the other kind is damaged" (fun () ->
          let store = new_store () in
          let run =
            written_run store
              (log_text
                 (timed
                    [
                      Reading ("a", 0, 1.);
                      Reading ("r", 0, 1.);
                      Metrics ("r", 1, 2.);
                    ]))
          in
          let h = History.read run in
          equal ~msg:"damaged" int 2 (History.damaged h);
          equal ~msg:"a" (list float_exact) []
            (values_of (History.series h "a"));
          equal ~msg:"r" (list float_exact) [ 1. ]
            (values_of (History.series h "r")));
      test "a record holding a sample of the other kind is damaged whole"
        (fun () ->
          let store = new_store () in
          let both =
            {|{"type":"metrics","time":3.0,"step":1,"values":{"a":5.0,"r":2.0}}|}
          in
          let run =
            written_run store
              (log_text (timed [ Reading ("r", 0, 1.) ]) ^ both ^ "\n")
          in
          let h = History.read run in
          equal ~msg:"damaged" int 1 (History.damaged h);
          equal ~msg:"a" (list float_exact) []
            (values_of (History.series h "a")));
      test "a run whose start holds another format is unreadable" (fun () ->
          let store = new_store () in
          let line =
            replace_first start_line ~sub:{|"format":1|} ~by:{|"format":2|}
          in
          let runs = Filename.concat (Store.root store) "runs" in
          mkdir_p (Store.root store);
          mkdir_p runs;
          mkdir_p (Filename.concat runs "other");
          write_file
            (Filename.concat (Filename.concat runs "other") "events.jsonl")
            (text_of_lines [ line ]);
          raises_match Exn.failure (fun () -> Store.run store "other");
          equal (list string) [] (List.map Run.id (Store.runs store)));
      prop "a history follows the format's orphan rule and statistics"
        ~examples:
          [
            (* The largest float with additions that each round away but
               together pass it: the mean is 2^1022. *)
            ( `Minimize,
              Metrics ("a", 0, Float.max_float)
              :: List.init 3 (fun _ -> Metrics ("a", 0, 0x1p969)) );
            (* Equal values whose rounded mean can fall a ulp below them. *)
            (`Minimize, List.init 5 (fun _ -> Metrics ("a", 0, Float.max_float)));
            (* An ordered series that a resume orphaned a sample of. *)
            ( `Minimize,
              [
                Metrics ("a", 0, 0.);
                Metrics ("a", 5, 1.);
                Resume 2;
                Metrics ("a", 3, 2.);
              ] );
            (* -0. and 0. tie: the earliest is the least and the greatest. *)
            (`Minimize, [ Metrics ("a", 0, 0.); Metrics ("a", 1, -0.) ]);
            (`Maximize, [ Metrics ("a", 0, -0.); Metrics ("a", 1, 0.) ]);
          ]
        (Gen.pair
           (Gen.of_list
              ~pp:(fun ppf g ->
                Format.pp_print_string ppf
                  (match g with
                  | `Minimize -> "minimize"
                  | `Maximize -> "maximize"))
              [ `Minimize; `Maximize ])
           (Gen.with_pp (fun ppf rs -> pp_records ppf (timed rs)) gen_orphaning))
        (fun (goal, rs) ->
          let records = timed rs in
          let a = series_of records "a" in
          let samples_of_a =
            List.length
              (List.filter
                 (function Metrics ("a", _, _) -> true | _ -> false)
                 rs)
          in
          let resumes =
            List.exists (function Resume _ -> true | _ -> false) rs
          in
          cover "a resume orphans a sample"
            (resumes && List.length a < samples_of_a);
          cover "a key's series is out of order" (not (ordered a));
          cover "a key's series is ordered, with samples a resume orphaned"
            (ordered a && List.length a > 1 && List.length a < samples_of_a);
          cover "a non-finite sample"
            (List.exists
               (function
                 | Metrics (_, _, v) -> not (Float.is_finite v) | _ -> false)
               rs);
          cover "a key's finite values sum past the largest float"
            (List.exists
               (fun k ->
                 let sum =
                   List.fold_left
                     (fun s (x : Metric.sample) ->
                       if Float.is_finite x.value then s +. x.value else s)
                     0. (series_of records k)
                 in
                 not (Float.is_finite sum))
               [ "a"; "b" ]);
          cover "every sample of a key is an orphan"
            (List.exists
               (fun k ->
                 series_of records k = []
                 && List.exists
                      (function Metrics (k', _, _) -> k' = k | _ -> false)
                      rs)
               [ "a"; "b" ]);
          cover "a -0. and a 0. are a key's extreme values"
            (let zeros =
               List.filter (fun (x : Metric.sample) -> x.value = 0.) a
             in
             List.exists
               (fun (x : Metric.sample) -> Float.sign_bit x.value)
               zeros
             && List.exists
                  (fun (x : Metric.sample) -> not (Float.sign_bit x.value))
                  zeros
             && List.for_all
                  (fun (x : Metric.sample) ->
                    (not (Float.is_finite x.value))
                    || if goal = `Minimize then x.value >= 0. else x.value <= 0.)
                  a);
          cover "a key maximized" (goal = `Maximize);
          let store = new_store () in
          let run =
            written_run store
              (log_text ~metrics:[ ("a", goal); ("b", `Minimize) ] records)
          in
          check_history ~msg:"keep all" records (History.read run);
          (* Statistics alone, a resume's re-read included. *)
          let h = History.read ~series:[] run in
          List.iter
            (fun k ->
              check_stats ~msg:("keep none " ^ k) (series_of records k)
                (History.stats h k))
            [ "a"; "b"; "r" ];
          let better =
            match goal with `Minimize -> ( < ) | `Maximize -> ( > )
          in
          equal ~msg:"best" (option sample)
            (match
               List.filter
                 (fun (s : Metric.sample) -> Float.is_finite s.value)
                 a
             with
            | [] -> None
            | x :: rest ->
                Some
                  (List.fold_left
                     (fun (b : Metric.sample) (s : Metric.sample) ->
                       if better s.value b.value then s else b)
                     x rest))
            (match (History.goal h "a", History.stats h "a") with
            | Some g, Some st -> Metric.best g st
            | _ -> None));
    ]

(* This machine's identities, as a session writes them in its run's [writer]
   file: the machine's and the boot's, each when the system names it. *)
let identities store =
  let s = start store in
  let text = read_file (Filename.concat (Run.dir (reread store s)) "writer") in
  Session.finish s;
  let field k =
    let rec find = function
      | k' :: _ :: v :: _ when k' = k -> Some v
      | _ :: rest -> find rest
      | [] -> None
    in
    find (String.split_on_char '"' text)
  in
  (field "machine", field "boot")

let status_tests =
  group "status"
    [
      test "a run whose session is open reads running" (fun () ->
          let store = new_store () in
          let s = start store in
          let st = Run.status (reread store s) in
          Session.finish s;
          equal status `Running st);
      test
        "a run whose writer exited without an end reads crashed under its boot"
        (fun () ->
          let store = new_store () in
          equal status (dead_status ()) (Run.status (exited_run store)));
      (* A run without an end whose lock no process holds, read under each
         [writer] it may have. *)
      cases "an unprobed run reads as its writer and log's time say"
        ~id:(fun (id, _, _, _) -> id)
        [
          ("without a writer, silent", None, false, `Silent);
          ("under this boot, crashed", Some (`Absent, `This), false, `Crashed);
          ("of another machine, silent", Some (`Other, `Other), false, `Silent);
          ( "of this machine, written since this boot, silent",
            Some (`This, `Other),
            false,
            `Silent );
          ( "of this machine, written before this boot, crashed",
            Some (`This, `Other),
            true,
            `Crashed );
        ]
        (fun (_, writer, old, expected) ->
          let store = new_store () in
          let machine, boot = identities store in
          let run =
            written_run store (log_text (timed [ Metrics ("a", 0, 1.) ]))
          in
          let member k this = function
            | `Absent -> []
            | `Other -> [ Printf.sprintf {|"%s":"another"|} k ]
            | `This -> (
                match this with
                | Some v -> [ Printf.sprintf {|"%s":"%s"|} k v ]
                | None -> skip ~reason:("this system names no " ^ k) ())
          in
          (match writer with
          | None -> ()
          | Some (m, b) ->
              write_file
                (Filename.concat (Run.dir run) "writer")
                ("{"
                ^ String.concat ","
                    (member "machine" machine m @ member "boot" boot b)
                ^ "}"));
          if old then Unix.utimes (log_path run) 1. 1.;
          equal status expected
            (Run.status (require_some (Store.run store (Run.id run)))));
    ]

let resume_tests =
  group "resuming a run"
    [
      test "a run whose session is open refuses a resume" (fun () ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.failure (fun () ->
              Session.resume ~provenance (reread store s) ~step:0));
      test "a run whose writer exited is resumed and ended" (fun () ->
          let store = new_store () in
          let run = exited_run store in
          Session.finish (Session.resume ~provenance run ~step:0);
          equal status `Finished
            (Run.status (require_some (Store.run store (Run.id run)))));
      test "a resume cuts the bytes that are not whole lines" (fun () ->
          let store = new_store () in
          let text =
            log_text (timed [ Metrics ("a", 0, 0.); Metrics ("a", 1, 1.) ])
          in
          let run = written_run store (text ^ {|{"type":"metr|}) in
          let s = Session.resume ~provenance run ~step:1 in
          Session.log (Session.metric s ~goal:`Minimize "a") ~step:2 2.;
          Session.finish s;
          let h = History.read run in
          equal ~msg:"damaged" int 0 (History.damaged h);
          equal (list float_exact) [ 0.; 1.; 2. ]
            (values_of (History.series h "a")));
      test "a resume cuts a line of NUL bytes and the records after it"
        (fun () ->
          let store = new_store () in
          let text = log_text (timed [ Metrics ("a", 0, 0.) ]) in
          let hole = String.make 40 '\000' ^ "\n" in
          let lost = record_line (5., Metrics ("a", 1, 1.)) ^ "\n" in
          let run = written_run store (text ^ hole ^ lost) in
          Session.finish (Session.resume ~provenance run ~step:0);
          let log = read_file (log_path run) in
          equal ~msg:"no NUL" int 0
            (String.fold_left
               (fun n c -> if c = '\000' then n + 1 else n)
               0 log);
          equal (list float_exact) [ 0. ]
            (values_of (History.series (History.read run) "a")));
      test "a resumed session learns the kind of each key from the log"
        (fun () ->
          let store = new_store () in
          let run =
            written_run store (log_text (timed [ Reading ("r", 0, 1.) ]))
          in
          let s = Session.resume ~provenance run ~step:0 in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.invalid_arg (fun () -> Session.metric s "r"));
      test "a resumed session's equal declaration writes no record" (fun () ->
          let store = new_store () in
          let run = written_run store (log_text []) in
          let s = Session.resume ~provenance run ~step:0 in
          ignore (Session.metric s ~goal:`Minimize "a");
          Session.finish s;
          let declarations =
            List.filter
              (fun l -> String.starts_with ~prefix:{|{"type":"metric",|} l)
              (lines (read_file (log_path run)))
          in
          equal int 2 (List.length declarations));
      test "a resume's step and provenance read back" (fun () ->
          let store = new_store () in
          let run = written_run store (log_text []) in
          let p = { provenance with command = [ "again" ]; pid = 7 } in
          Session.finish (Session.resume ~provenance:p run ~step:4);
          match History.resumes (History.read run) with
          | [ r ] ->
              equal ~msg:"step" int 4 r.step;
              equal ~msg:"command" (list string) [ "again" ]
                r.provenance.command;
              equal ~msg:"pid" int 7 r.provenance.pid
          | rs -> fail (Printf.sprintf "%d resumes" (List.length rs)));
    ]

(* [listing_digest entries] is the digest of a directory whose listing is
   [entries] (Store_format, blobs). *)
let listing_digest entries =
  "blake2b-256:"
  ^ Digest.BLAKE256.to_hex
      (Digest.BLAKE256.string
         (String.concat "" (List.map (fun l -> l ^ "\n") entries)))

let hex data = Digest.BLAKE256.to_hex (Digest.BLAKE256.string data)

let blobs_tests =
  group "blobs"
    [
      test "a file is stored under the digest of its bytes" (fun () ->
          let store = new_store () in
          let s = start store in
          let v = Session.log_artifact s ~name:"w" (source "weights") in
          Session.finish s;
          equal ~msg:"digest" string
            ("blake2b-256:" ^ hex "weights")
            (Artifact.digest v);
          equal ~msg:"blob" string (hex "weights")
            (Filename.basename (Artifact.path v)));
      test "a directory is stored under the digest of its listing" (fun () ->
          if Sys.win32 then skip ~reason:"a test makes symbolic links" ();
          let store = new_store () in
          let dir = temp_dir () in
          Unix.mkdir (Filename.concat dir "sub") 0o755;
          write_file (Filename.concat dir "a.txt") "a";
          write_file (Filename.concat dir "sub/b") "b";
          (* A link to a file is stored as that file. *)
          Unix.symlink
            (Filename.concat dir "a.txt")
            (Filename.concat dir "link");
          let s = start store in
          let v = Session.log_artifact s ~name:"data" dir in
          Session.finish s;
          equal ~msg:"digest" string
            (listing_digest
               [
                 "file:a.txt:" ^ hex "a";
                 "file:link:" ^ hex "a";
                 "dir:sub";
                 "file:sub/b:" ^ hex "b";
               ])
            (Artifact.digest v);
          equal ~msg:"size" int 3 (Artifact.size_bytes v);
          equal ~msg:"content" string "b"
            (read_file (Filename.concat (Artifact.path v) "sub/b")));
      test "a link to a directory is refused" (fun () ->
          if Sys.win32 then skip ~reason:"a test makes symbolic links" ();
          let store = new_store () in
          let dir = temp_dir () in
          Unix.mkdir (Filename.concat dir "sub") 0o755;
          Unix.symlink (Filename.concat dir "sub") (Filename.concat dir "link");
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Session.log_artifact s ~name:"data" dir));
    ]

let durability_tests =
  group "durability"
    [
      test "a reader sees whole records while a run is written" (fun () ->
          let store = new_store () in
          (* The child logs 40 samples of "big", each beside a key 70 KiB long,
             so that a record takes several writes. *)
          let pid, stdin, out = spawn [ "log"; Store.root store; "40" ] in
          Unix.close stdin;
          let id = input_line out in
          let run = require_some (Store.run store id) in
          let rec poll h k =
            let h = History.follow h in
            let got = values_of (History.series h "big") in
            equal ~msg:"a prefix" (list float_exact)
              (List.init (List.length got) Float.of_int)
              got;
            if Run.status (History.run h) = `Running && k < 10_000 then
              poll h (k + 1)
          in
          poll (History.read run) 0;
          wait pid;
          close_in out;
          equal int 40
            (Metric.length
               (History.series
                  (History.read (require_some (Store.run store id)))
                  "big")));
      test "every byte prefix of a log reads as whole records" (fun () ->
          let store = new_store () in
          let s = start store in
          let m = Session.metric s "m" in
          for step = 0 to 9 do
            Session.log m ~step (Float.of_int step)
          done;
          Session.finish s;
          let text = read_file (log_path (reread store s)) in
          let first = String.index text '\n' + 1 in
          for n = first to String.length text do
            let h = History.read (written_run store (String.sub text 0 n)) in
            let got = values_of (History.series h "m") in
            equal
              ~msg:(Printf.sprintf "prefix of %d bytes" n)
              (list float_exact)
              (List.init (List.length got) Float.of_int)
              got;
            equal ~msg:"no damage" int 0 (History.damaged h)
          done);
    ]

let crash_tests =
  group "crash states"
    [
      cases "a run directory whose log holds no whole first line is no run"
        ~id:fst
        [
          ("no log", None);
          ("a log cut in its start", Some {|{"type":"sta|});
          ("an empty log", Some "");
        ]
        (fun (_, log) ->
          let store = new_store () in
          let runs = Filename.concat (Store.root store) "runs" in
          mkdir_p (Store.root store);
          mkdir_p runs;
          let dir = Filename.concat runs "claimed" in
          mkdir_p dir;
          Option.iter (write_file (Filename.concat dir "events.jsonl")) log;
          equal (option string) None
            (Option.map Run.id (Store.run store "claimed"));
          equal (list string) [] (List.map Run.id (Store.runs store)));
      test "a run directory with a writer and no whole first line is no run"
        (fun () ->
          let store = new_store () in
          let machine, boot = identities store in
          let runs = Filename.concat (Store.root store) "runs" in
          let dir = Filename.concat runs "claimed" in
          mkdir_p dir;
          let quote k = Option.map (Printf.sprintf {|"%s":"%s"|} k) in
          write_file
            (Filename.concat dir "writer")
            ("{"
            ^ String.concat ","
                (List.filter_map Fun.id
                   [ quote "machine" machine; quote "boot" boot ])
            ^ "}");
          write_file (Filename.concat dir "events.jsonl") {|{"type":"sta|};
          equal (option string) None
            (Option.map Run.id (Store.run store "claimed")));
      test "a file under files/ that no record names is ignored" (fun () ->
          let store = new_store () in
          let run = written_run store (log_text []) in
          let dir = Filename.concat (Run.dir run) "files/viz" in
          mkdir_p (Filename.concat (Run.dir run) "files");
          mkdir_p dir;
          write_file (Filename.concat dir "3.png") "png";
          equal (list string) []
            (List.map
               (fun (f : History.file) -> f.path)
               (History.files (History.read run))));
      test "a tag file whose bytes do not digest to its name is ignored"
        (fun () ->
          let store = new_store () in
          let s = start ~tags:[ "a"; "b" ] store in
          Session.finish s;
          let tag =
            Filename.concat (Run.dir (reread store s)) ("tags/" ^ hex "b")
          in
          write_file tag "c";
          equal (list string) [ "a" ] (Run.tags (reread store s)));
    ]

(* Statistics hint *)

let hint_path run = Filename.concat (Run.dir run) "stats.json"

(* [number_after text key] is the integer member [key] of the JSON [text]. *)
let number_after text key =
  let sub = Printf.sprintf {|"%s":|} key in
  let rec find i =
    if String.sub text i (String.length sub) = sub then i + String.length sub
    else find (i + 1)
  in
  let start = find 0 in
  let stop = ref start in
  while
    !stop < String.length text && text.[!stop] >= '0' && text.[!stop] <= '9'
  do
    incr stop
  done;
  int_of_string (String.sub text start (!stop - start))

(* [ended_with_hint store] is an ended run that logged one sample of "a". *)
let ended_with_hint store =
  let s = start store in
  Session.log (Session.metric s "a") ~step:0 1.;
  Session.finish s;
  reread store s

let hint_tests =
  group "statistics of an ended run"
    [
      prop ~count:25 "an ended run's hint gives what its fold gives" gen_records
        (fun rs ->
          let store = new_store () in
          let s = start store in
          let a = Session.metric s ~goal:`Minimize "a" in
          List.iteri
            (fun i r ->
              match r with
              | Metrics (_, st, v) ->
                  Session.log a ~step:st ~time:(Float.of_int i) v
              | _ -> ())
            rs;
          Session.finish s;
          let run = reread store s in
          let hinted = History.stats_of run in
          Sys.remove (hint_path run);
          equal (list (pair string key_stats)) (History.stats_of run) hinted);
      prop ~count:40 "a resumed run's hint gives what its fold gives"
        (Gen.triple gen_records (Gen.int_range 0 20) gen_records)
        (fun (before, step, after) ->
          cover "the resume orphans a sample"
            (List.exists
               (function Metrics (_, s, _) -> s > step | _ -> false)
               before);
          cover "a resume in the log before"
            (List.exists (function Resume _ -> true | _ -> false) before);
          let store = new_store () in
          (* The log a crashed writer left, then a session that resumes it. *)
          let run = written_run store (log_text (timed before)) in
          let s = Session.resume ~provenance run ~step in
          let a = Session.metric s ~goal:`Minimize "a" in
          List.iteri
            (fun i r ->
              match r with
              | Metrics (_, st, v) ->
                  Session.log a ~step:st ~time:(Float.of_int i) v
              | _ -> ())
            after;
          Session.finish s;
          let run = require_some (Store.run store "spec") in
          let hinted = History.stats_of run in
          Sys.remove (hint_path run);
          equal (list (pair string key_stats)) (History.stats_of run) hinted);
      test "a hint that does not parse is ignored" (fun () ->
          let store = new_store () in
          let run = ended_with_hint store in
          let good = History.stats_of run in
          write_file (hint_path run) "{\"format\":1,";
          equal (list (pair string key_stats)) good (History.stats_of run));
      cases "a hint of another format, rules or length is ignored" ~id:fst
        [
          ( "format",
            fun text ->
              replace_first text ~sub:{|"format":1|} ~by:{|"format":2|} );
          ( "rules",
            fun text ->
              replace_first text
                ~sub:(Printf.sprintf {|"rules":%d|} Store_format.rules)
                ~by:(Printf.sprintf {|"rules":%d|} (Store_format.rules + 1)) );
          ( "length",
            fun text ->
              let n = number_after text "length" in
              replace_first text
                ~sub:(Printf.sprintf {|"length":%d|} n)
                ~by:(Printf.sprintf {|"length":%d|} (n + 1)) );
        ]
        (fun (_, change) ->
          let store = new_store () in
          let run = ended_with_hint store in
          let good = History.stats_of run in
          write_file (hint_path run)
            (replace_first
               (change (read_file (hint_path run)))
               ~sub:{|"count":1|} ~by:{|"count":2|});
          equal (list (pair string key_stats)) good (History.stats_of run));
      test "a crashed run has no hint" (fun () ->
          let store = new_store () in
          let run = exited_run store in
          equal (list string) []
            (List.filter (( = ) "stats.json")
               (Array.to_list (Sys.readdir (Run.dir run)))));
    ]

let store_format_tests =
  group "Store_format"
    [
      numbers_tests;
      reading_tests;
      status_tests;
      resume_tests;
      hint_tests;
      blobs_tests;
      durability_tests;
      crash_tests;
    ]

(* Value *)

let value_tests =
  group "Value"
    [
      cases "pp prints each kind of value as its doc states" ~id:fst
        [
          ("true", (`Bool true, "true"));
          ("false", (`Bool false, "false"));
          ("an integer", (`Int (-42), "-42"));
          ("1.0", (`Float 1., "1.0"));
          ("-0.0", (`Float (-0.), "-0.0"));
          ("0.1", (`Float 0.1, "0.1"));
          ("1e+16", (`Float 1e16, "1e+16"));
          ("nan", (`Float Float.nan, "nan"));
          ("inf", (`Float Float.infinity, "inf"));
          ("-inf", (`Float Float.neg_infinity, "-inf"));
          ("a tab and a line feed", (`String "a\tb\nc", {|a\tb\nc|}));
          ("a backslash", (`String {|a\b|}, {|a\\b|}));
          ("a control character", (`String "\x01", {|\u0001|}));
        ]
        (fun (_, (v, text)) ->
          equal string text (Format.asprintf "%a" Value.pp v));
      prop "pp's text holds no tab or line feed" gen_value (fun v ->
          satisfies ~claim:"no tab or line feed" string
            (fun s -> not (String.exists (fun c -> c = '\t' || c = '\n') s))
            (Format.asprintf "%a" Value.pp v));
      test "to_float is a float's value or an integer's nearest float"
        (fun () ->
          equal ~msg:"float" (option float_exact) (Some (-0.))
            (Value.to_float (`Float (-0.)));
          (* 2^53 + 1 lies halfway between 2^53 and 2^53 + 2: the even one. *)
          equal ~msg:"int" (option float_exact) (Some 0x1p53)
            (Value.to_float (`Int ((1 lsl 53) + 1)));
          equal ~msg:"string" (option float_exact) None
            (Value.to_float (`String "1"));
          equal ~msg:"bool" (option float_exact) None
            (Value.to_float (`Bool true)));
      test "to_int, to_string and to_bool take their constructor alone"
        (fun () ->
          equal ~msg:"to_int"
            (list (option int))
            [ Some 3; None; None ]
            (List.map Value.to_int [ `Int 3; `Float 3.; `String "3" ]);
          equal ~msg:"to_string"
            (list (option string))
            [ Some "x"; None ]
            (List.map Value.to_string [ `String "x"; `Int 1 ]);
          equal ~msg:"to_bool"
            (list (option bool))
            [ Some false; None ]
            (List.map Value.to_bool [ `Bool false; `Int 0 ]));
    ]

(* Provenance *)

let git args cwd =
  Unix.system
    (Filename.quote_command "git" ("-C" :: cwd :: args) ~stdout:Filename.null
       ~stderr:Filename.null)

(* [work_tree ()] is a git work tree with one commit of the file [f], or a skip
   without git. *)
let work_tree () =
  let dir = temp_dir ~prefix:"munin-git" () in
  if git [ "init"; "-q" ] dir <> Unix.WEXITED 0 then skip ~reason:"no git" ();
  write_file (Filename.concat dir "f") "1";
  ignore (git [ "add"; "f" ] dir);
  ignore
    (git
       [ "-c"; "user.name=t"; "-c"; "user.email=t@t"; "commit"; "-qm"; "c" ]
       dir);
  dir

let dirty (p : Provenance.t) =
  Option.map (fun (g : Provenance.git) -> g.dirty) p.git

let provenance_tests =
  group "Provenance"
    [
      test "detect finds no work tree outside one" (fun () ->
          equal (option bool) None
            (dirty (Provenance.detect ~cwd:(temp_dir ()) ())));
      test "detect finds a tree clean until a tracked file changes" (fun () ->
          let dir = work_tree () in
          (* An untracked file does not count. *)
          write_file (Filename.concat dir "untracked") "x";
          equal ~msg:"clean" (option bool) (Some false)
            (dirty (Provenance.detect ~cwd:dir ()));
          write_file (Filename.concat dir "f") "2";
          equal ~msg:"dirty" (option bool) (Some true)
            (dirty (Provenance.detect ~cwd:dir ())));
      test "detect records the variables asked for that are set" (fun () ->
          setenv "MUNIN_TEST_SET" (Some "v");
          setenv "MUNIN_TEST_UNSET" None;
          let p =
            Provenance.detect ~cwd:"/"
              ~seeds:[ ("init", 7) ]
              ~capture_env:[ "MUNIN_TEST_SET"; "MUNIN_TEST_UNSET" ]
              ()
          in
          equal ~msg:"env"
            (list (pair string string))
            [ ("MUNIN_TEST_SET", "v") ]
            p.env;
          equal ~msg:"seeds" (list (pair string int)) [ ("init", 7) ] p.seeds);
      cases "detect refuses bad seeds and variable names" ~id:fst
        [
          ("a seed that is not a text", ([ ("", 1) ], []));
          ("a seed named twice", ([ ("s", 1); ("s", 2) ], []));
          (* Both escape to U+FFFD. *)
          ( "two names that escape alike",
            ([], [ "MUNIN_TEST_\xff"; "MUNIN_TEST_\xfe" ]) );
        ]
        (fun (_, (seeds, capture_env)) ->
          raises_match Exn.invalid_arg (fun () ->
              Provenance.detect ~cwd:"/" ~seeds ~capture_env ()));
      test "a start records its provenance" (fun () ->
          let store = new_store () in
          let p =
            {
              provenance with
              hostname = Some "h";
              cpu = Some "c";
              git = Some { commit = String.make 40 'a'; dirty = false };
              seeds = [ ("init", 7); ("data", -1) ];
              env = [ ("K", "v") ];
            }
          in
          let s = Session.start ~store ~provenance:p ~experiment:"e" () in
          Session.finish s;
          let q = Run.provenance (reread store s) in
          equal ~msg:"hostname" (option string) p.hostname q.hostname;
          equal ~msg:"cpu" (option string) p.cpu q.cpu;
          equal ~msg:"git"
            (option (pair string bool))
            (Some (String.make 40 'a', false))
            (Option.map (fun (g : Provenance.git) -> (g.commit, g.dirty)) q.git);
          equal ~msg:"seeds" (list (pair string int)) p.seeds q.seeds;
          equal ~msg:"env" (list (pair string string)) p.env q.env;
          equal ~msg:"command" (list string) p.command q.command);
      test "a start keeps the diff of a dirty work tree" (fun () ->
          let dir = work_tree () in
          write_file (Filename.concat dir "f") "2";
          let store = new_store () in
          let s =
            Session.start ~store
              ~provenance:(Provenance.detect ~cwd:dir ())
              ~experiment:"e" ()
          in
          Session.finish s;
          contains ~sub:"+2"
            (read_file (require_some (Run.diff (reread store s)))));
      test "provenance and failure text that are not UTF-8 are kept, escaped"
        (fun () ->
          let store = new_store () in
          let bad =
            { provenance with cwd = "/data/caf\xe9"; command = [ "x\xff" ] }
          in
          let s = Session.start ~store ~provenance:bad ~experiment:"e" () in
          Session.finish ~status:(`Failed "caf\xe9") s;
          let run = reread store s in
          equal string "/data/caf\xef\xbf\xbd" (Run.provenance run).cwd;
          equal (list string) [ "x\xef\xbf\xbd" ] (Run.provenance run).command;
          equal status (`Failed "caf\xef\xbf\xbd") (Run.status run));
      test "pp prints a field a line" (fun () ->
          let p =
            { provenance with ocaml = "5.4.0"; seeds = [ ("init", 7) ] }
          in
          expect (Format.asprintf "%a" Provenance.pp p)
          @@ __POS_OF__
               {|
            command: test_munin
            cwd: /
            hostname: -
            pid: 0
            ocaml: 5.4.0
            raven: test
            os: test
            arch: test
            cpu: -
            git: -
            seed init: 7
            |});
    ]

(* Metric *)

let metric_tests =
  group "Metric"
    [
      test "a series' accessors raise outside its indices" (fun () ->
          let store = new_store () in
          let run =
            written_run store (log_text (timed [ Metrics ("a", 0, 1.) ]))
          in
          let s = History.series (History.read run) "a" in
          raises_match Exn.invalid_arg (fun () -> Metric.step s 1);
          raises_match Exn.invalid_arg (fun () -> Metric.time s (-1));
          raises_match Exn.invalid_arg (fun () -> Metric.value s 1);
          raises_match Exn.invalid_arg (fun () -> Metric.sample s 1));
    ]

(* Run *)

let run_tests =
  group "Run"
    [
      test "a run reads back what its start stated" (fun () ->
          let store = new_store () in
          let parent = start store in
          Session.finish parent;
          let s =
            start ~experiment:"mnist" ~name:"first" ~parent:(Session.id parent)
              ~steps:99 store
          in
          Session.log (Session.metric s "m") ~step:0 ~time:1. 0.;
          Session.finish s;
          let r = reread store s in
          equal ~msg:"experiment" string "mnist" (Run.experiment r);
          equal ~msg:"name" (option string) (Some "first") (Run.name r);
          equal ~msg:"parent" (option string)
            (Some (Session.id parent))
            (Run.parent r);
          equal ~msg:"steps" (option int) (Some 99) (Run.steps r);
          equal ~msg:"dir" string
            (Filename.concat
               (Filename.concat (Store.root store) "runs")
               (Session.id s))
            (Run.dir r));
      test "a run's times are its start's, its end's and its log's" (fun () ->
          let store = new_store () in
          let s = start store in
          (* A time the program gives its samples moves nothing. *)
          Session.log (Session.metric s "m") ~step:0 ~time:1. 0.;
          Session.finish s;
          let r = reread store s in
          let mtime = (Unix.stat (log_path r)).Unix.st_mtime in
          equal ~msg:"updated" float_exact mtime (Run.updated_at r);
          at_least ~msg:"ended after started" float_exact
            ~than:(Run.started_at r)
            (require_some (Run.ended_at r)));
      cases "a run reads back the status its session ended with" ~id:fst
        [
          ("finished", `Finished);
          ("failed", `Failed "boom");
          ("killed", `Killed);
        ]
        (fun (_, st) ->
          let store = new_store () in
          let s = start store in
          Session.finish ~status:st s;
          equal status (st :> Run.status) (Run.status (reread store s)));
    ]

(* History *)

(* The records [surviving] reads: a sample of the metric "a", a reading of "r"
   at a step and a resume, each at a time any clock gave. *)
let gen_surviving =
  let open Gen in
  let record =
    let* t = float_range 0. 100. in
    let+ r =
      frequency
        [
          (3, map (fun s -> Metrics ("a", s, 1.)) (int_range 0 20));
          ( 3,
            map
              (fun (s, v) -> Reading ("r", s, v))
              (pair (int_range 0 20) (float_range 0. 100.)) );
          (1, map (fun s -> Resume s) (int_range 0 20));
        ]
    in
    (t, r)
  in
  with_pp pp_records (list ~size:(int_range 0 40) record)

let surviving_test =
  prop
    "surviving leaves out the readings a later resume goes back below, \
     whatever the clocks"
    (Gen.pair gen_surviving (Gen.int_range 0 100000))
    (fun (records, cut) ->
      let abandons_later =
        let rec go = function
          | [] -> false
          | (t, Reading (_, s, _)) :: rest ->
              List.exists
                (function t', Resume r -> r < s && t' < t | _ -> false)
                rest
              || go rest
          | _ :: rest -> go rest
        in
        go records
      in
      let readings =
        List.length
          (List.filter (function _, Reading _ -> true | _ -> false) records)
      in
      let expected = surviving_of records "r" in
      cover "a resume abandons a reading" (List.length expected < readings);
      cover "a resume logged at an earlier time abandons a reading"
        abandons_later;
      cover "a reading kept after a resume"
        (List.exists (function _, Resume _ -> true | _ -> false) records
        && expected <> []);
      let store = new_store () in
      let text = log_text ~metrics:[ ("a", `Minimize) ] records in
      let run = written_run store text in
      let h = History.read run in
      equal ~msg:"read at once" (list sample) expected
        (samples_of (History.surviving h "r"));
      equal ~msg:"a metric's, its series" (list sample)
        (samples_of (History.series h "a"))
        (samples_of (History.surviving h "a"));
      (* A resume that orphans samples of "a", which it does not keep, reads the
         log again. *)
      let only = History.read ~series:[ "r" ] run in
      equal ~msg:"keeping r alone" (list sample) expected
        (samples_of (History.surviving only "r"));
      (* Followed in two pieces, a resume of the second after readings of the
         first. *)
      let first = String.length start_line + 1 in
      let cut = first + (cut mod (String.length text - first + 1)) in
      let prefix = String.sub text 0 cut in
      write_file (log_path run) prefix;
      let h0 = History.read run in
      let before = samples_of (History.surviving h0 "r") in
      write_file (log_path run) text;
      let h1 = History.follow h0 in
      let after = samples_of (History.surviving h1 "r") in
      equal ~msg:"followed" (list sample) expected after;
      equal ~msg:"the first piece's, after the follow" (list sample) before
        (samples_of (History.surviving h0 "r"));
      (* The follow extends the first piece's samples unless a resume of the
         second abandons one of them. *)
      let whole_lines = List.length (String.split_on_char '\n' prefix) - 1 in
      let first_records =
        List.filteri (fun i _ -> i < whole_lines - 2) records
      in
      let second = List.filteri (fun i _ -> i >= whole_lines - 2) records in
      let abandons =
        List.exists
          (function
            | _, Resume r ->
                List.exists (fun (s : Metric.sample) -> r < s.step) before
            | _ -> false)
          second
      in
      cover "the follow extends the first piece's readings"
        ((not abandons) && before <> []
        && List.length after > List.length before);
      cover "a resume of the second piece abandons a reading of the first"
        abandons;
      equal ~msg:"the first piece's, by the reference" (list sample)
        (surviving_of first_records "r")
        before;
      if not abandons then
        equal ~msg:"the follow extends them" (list sample) before
          (List.filteri (fun i _ -> i < List.length before) after))

(* [abandoned_file store] is a run that logged a file at step 5, then resumed at
   step 3 and logged one at step 4. *)
let abandoned_file store =
  let run =
    written_run store
      (log_text []
      ^ text_of_lines
          [
            {|{"type":"file","time":2.0,"key":"viz","step":5,"path":"files/viz/5.png"}|};
          ])
  in
  mkdir_p (Filename.concat (Run.dir run) "files");
  mkdir_p (Filename.concat (Run.dir run) "files/viz");
  write_file (Filename.concat (Run.dir run) "files/viz/5.png") "png";
  let s = Session.resume ~provenance run ~step:3 in
  Session.log_file s ~step:4 ~key:"viz" (source ~suffix:".png" "png");
  Session.finish s;
  run

(* A log of [n_samples] samples of "a", 0 to n - 1, read a line at a time with
   [follow ~limit:1]: a history's reference is how many samples it holds. *)
let n_samples = 40

let domain_log =
  lazy
    (written_run (new_store ())
       (log_text ~metrics:[]
          (timed
             (List.init n_samples (fun i -> Metrics ("a", i, Float.of_int i))))))

let hist_w : (int, History.t) abstract = abstract "h"

let history_tests =
  group "History"
    [
      prop
        "following in pieces equals reading at once, and leaves snapshots as \
         they were"
        ~examples:
          [
            (* A resume that orphans a sample of "b", which is not kept. *)
            ( [ Metrics ("b", 5, 1.); Metrics ("a", 0, 1.); Resume 0 ],
              [ 0 ],
              true );
          ]
        (Gen.triple gen_records
           (Gen.list ~size:(Gen.int_range 1 6) (Gen.int_range 0 100000))
           Gen.bool)
        (fun (rs, cuts, keep_a) ->
          let records = timed rs in
          let store = new_store () in
          let text = log_text records in
          let n = String.length text in
          let first = String.length start_line + 1 in
          let cuts =
            List.sort_uniq compare
              (List.map (fun c -> first + (c mod (n - first + 1))) cuts)
          in
          cover "a cut inside a line"
            (List.exists (fun c -> c < n && text.[c - 1] <> '\n') cuts);
          let run = written_run store (String.sub text 0 first) in
          (* Keeping "a" alone, a resume that orphans "b" reads the log
             again. *)
          let series = if keep_a then Some [ "a" ] else None in
          let h0 = History.read ?series run in
          let snaps =
            List.fold_left
              (fun acc c ->
                let h = match acc with (h, _) :: _ -> h | [] -> h0 in
                (* Rewriting keeps the inode; the prefix is unchanged. *)
                write_file (log_path run) (String.sub text 0 c);
                let h' = History.follow h in
                (h', samples_of (History.series h' "a")) :: acc)
              [] cuts
          in
          let last = match snaps with (h, _) :: _ -> h | [] -> h0 in
          write_file (log_path run) text;
          let final = History.follow last in
          cover "a follow restarted"
            (List.exists History.restarted (final :: List.map fst snaps));
          (* Every key's statistics, and the samples of the keys kept. *)
          let check_followed ~msg h =
            if keep_a then (
              equal ~msg:(msg ^ " series a") (list sample)
                (series_of records "a")
                (samples_of (History.series h "a"));
              List.iter
                (fun k ->
                  check_stats
                    ~msg:(msg ^ " " ^ k)
                    (series_of records k) (History.stats h k))
                [ "a"; "b"; "r" ])
            else check_history ~msg records h
          in
          check_followed ~msg:"followed" final;
          List.iter
            (fun (h, seen) ->
              equal ~msg:"a snapshot stays" (list sample) seen
                (samples_of (History.series h "a")))
            snaps;
          (* A second follow of one history copies. *)
          let again = History.follow last in
          check_followed ~msg:"followed twice" again;
          check_followed ~msg:"first follow kept" final);
      surviving_test;
      test "a decimal that strtod misreads somewhere reads back exactly"
        (fun () ->
          let store = new_store () in
          let line =
            {|{"type":"metrics","time":2.0,"step":0,"values":{"a":3.839223843572815e-239}}|}
          in
          let run = written_run store (text_of_lines [ start_line; line ]) in
          equal float_exact
            (Float.of_string "0x1.fffffffffffffp-793")
            (Metric.value (History.series (History.read run) "a") 0));
      test "a replaced log is read again from its start" (fun () ->
          let store = new_store () in
          let run =
            written_run store
              (log_text (timed [ Metrics ("a", 1, 1.); Metrics ("a", 2, 2.) ]))
          in
          let h = History.read run in
          let tmp = log_path run ^ ".new" in
          write_file tmp (log_text [ (2., Metrics ("a", 1, 5.)) ]);
          Sys.rename tmp (log_path run);
          let h' = History.follow h in
          equal ~msg:"restarted" bool true (History.restarted h');
          equal (list sample)
            [ { step = 1; time = 2.; value = 5. } ]
            (samples_of (History.series h' "a")));
      test "a shorter log is read again from its start" (fun () ->
          let store = new_store () in
          let run =
            written_run store
              (log_text (timed [ Metrics ("a", 1, 1.); Metrics ("a", 2, 2.) ]))
          in
          let h = History.read run in
          write_file (log_path run) (log_text [ (2., Metrics ("a", 1, 5.)) ]);
          let h' = History.follow h in
          equal ~msg:"restarted" bool true (History.restarted h');
          equal (list float_exact) [ 5. ] (values_of (History.series h' "a")));
      test "an unchanged log is not read again" (fun () ->
          let store = new_store () in
          let run =
            written_run store (log_text (timed [ Metrics ("a", 1, 1.) ]))
          in
          equal bool false
            (History.restarted (History.follow (History.read run))));
      test "a log read again that no longer starts as a run raises" (fun () ->
          let store = new_store () in
          let run =
            written_run store (log_text (timed [ Metrics ("a", 1, 1.) ]))
          in
          let h = History.read run in
          write_file (log_path run) "not a run\n";
          raises_match Exn.failure (fun () -> History.follow h));
      cases "a limit reads up to the first whole line that ends past it"
        ~id:(fun (n, _) -> string_of_int n)
        (let start = String.length start_line + 1 in
         let line =
           String.length (record_line (2., Metrics ("a", 0, 0.))) + 1
         in
         [
           (1, []);
           (start - 1, []);
           (start, [ 0. ]);
           (start + line - 1, [ 0. ]);
           (start + line, [ 0.; 1. ]);
           (max_int, [ 0.; 1.; 2. ]);
         ])
        (fun (limit, expected) ->
          let store = new_store () in
          let run =
            written_run store
              (log_text ~metrics:[]
                 (timed
                    [
                      Metrics ("a", 0, 0.);
                      Metrics ("a", 1, 1.);
                      Metrics ("a", 2, 2.);
                    ]))
          in
          equal (list float_exact) expected
            (values_of (History.series (History.read ~limit run) "a")));
      test "series, surviving and orphans raise for a key not kept" (fun () ->
          let store = new_store () in
          let run =
            written_run store (log_text (timed [ Metrics ("a", 1, 1.) ]))
          in
          let h = History.read ~series:[ "b" ] run in
          raises_match Exn.invalid_arg (fun () -> History.series h "a");
          raises_match Exn.invalid_arg (fun () -> History.surviving h "a");
          raises_match Exn.invalid_arg (fun () -> History.orphans h "a"));
      test "orphans are a key's samples a resume abandoned, in log order"
        (fun () ->
          let store = new_store () in
          let run =
            written_run store
              (log_text
                 (timed
                    [
                      Metrics ("a", 1, 1.);
                      Metrics ("a", 3, 3.);
                      Metrics ("a", 2, 2.);
                      Resume 1;
                      Metrics ("a", 2, 4.);
                    ]))
          in
          let h = History.read run in
          equal ~msg:"orphans" (list float_exact) [ 3.; 2. ]
            (values_of (History.orphans h "a"));
          equal ~msg:"series" (list float_exact) [ 1.; 4. ]
            (values_of (History.series h "a")));
      test "step_metric is the axis a metric's latest declaration names"
        (fun () ->
          let store = new_store () in
          let s = start store in
          let epoch = Session.metric s "epoch" in
          ignore (Session.metric s ~step_metric:epoch "loss");
          Session.finish s;
          let h = history store s in
          equal ~msg:"loss" (option string) (Some "epoch")
            (History.step_metric h "loss");
          equal ~msg:"epoch" (option string) None
            (History.step_metric h "epoch"));
      test "a later metric record replaces a key's goal" (fun () ->
          let store = new_store () in
          let run =
            written_run store
              (log_text
                 ~metrics:[ ("a", `Minimize); ("a", `Maximize) ]
                 (timed [ Metrics ("a", 0, 1.) ]))
          in
          equal (option string) (Some "maximize")
            (Option.map
               (function `Minimize -> "minimize" | `Maximize -> "maximize")
               (History.goal (History.read run) "a")));
      test "uses are the versions a run used, in log order" (fun () ->
          let store = new_store () in
          let s = start store in
          let a1 = Session.log_artifact s ~name:"x" (source "1") in
          let a2 = Session.log_artifact s ~name:"x" (source "2") in
          Session.use_artifact s a2;
          Session.use_artifact s a1;
          Session.finish s;
          equal
            (list (pair string int))
            [ ("x", 2); ("x", 1) ]
            (History.uses (history store s)));
      test "files are the files a run logged that are not orphans" (fun () ->
          let store = new_store () in
          let run = abandoned_file store in
          equal (list int) [ 4 ]
            (List.map
               (fun (f : History.file) -> f.step)
               (History.files (History.read run))));
      test "a run as its history last read it" (fun () ->
          let store = new_store () in
          let s = start store in
          let h = history store s in
          Store.add_tags store (Session.id s) [ "t" ];
          Store.set_notes store (Session.id s) (Some "n");
          Session.finish s;
          let r = History.run (History.follow h) in
          equal ~msg:"before" status `Running (Run.status (History.run h));
          equal ~msg:"status" status `Finished (Run.status r);
          equal ~msg:"tags" (list string) [ "t" ] (Run.tags r);
          equal ~msg:"notes" (option string) (Some "n") (Run.notes r));
      stateful ~count:10 ~domains:2
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
            (fun h -> values_of (History.series h "a"));
        ];
    ]

(* Store *)

(* The artifacts of one name, as a model: each version's number, content and
   whether it is deleted, the alias "best", and the contents stored. *)
type model = {
  mutable versions : (int * string * bool) list;
  mutable best : int option;
}

let content = Gen.of_list ~pp:Format.pp_print_string [ "a"; "b" ]
let number = Gen.int_range 1 4
let alive m n = List.exists (fun (n', _, live) -> n' = n && live) m.versions

let select_gen =
  Gen.of_list
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | `Latest -> "latest"
        | `Version n -> "version " ^ string_of_int n
        | `Alias a -> "alias " ^ a))
    [ `Latest; `Version 1; `Version 2; `Version 3 ]

(* A store with a file of each content, and a session per domain that publishes.
   Whatever the calls, the store they leave breaks no rule of the format. *)
let store_w : (model, Store.t * string) abstract =
  abstract
    ~release:(fun (store, _) ->
      equal ~msg:"check" (list string) [] (List.map fst (Store.check store)))
    "s"

let session_w : (model, Session.t) abstract =
  abstract ~release:(fun s -> Session.finish s) "p"

(* What [Store.artifact] selects, as the model and the store give it: the
   version and its digest. *)
let selected = option (pair int string)

let select_ref sel m =
  let live n = if alive m n then Some n else None in
  let n =
    match sel with
    | `Latest ->
        List.fold_left
          (fun acc (n, _, live) -> if live then Some n else acc)
          None m.versions
    | `Version n -> live n
    | `Alias _ -> Option.bind m.best live
  in
  Option.map
    (fun n ->
      let _, c, _ = List.find (fun (n', _, _) -> n' = n) m.versions in
      (n, "blake2b-256:" ^ hex c))
    n

(* A blob may go once its version is deleted, so the selection reads the
   version's digest and not its blob. *)
let select_sys sel (store, _) =
  Option.map
    (fun a -> (Artifact.version a, Artifact.digest a))
    (Store.artifact store "ckpt" sel)

let alias_gen =
  Gen.of_list
    ~pp:(fun ppf _ -> Format.pp_print_string ppf "alias best")
    [ `Alias "best" ]

(* [dangling_alias store] is the path of the alias "best" of "x", which names a
   version since deleted. *)
let dangling_alias store =
  let s = start store in
  let a = Session.log_artifact s ~name:"x" (source "1") in
  Session.finish s;
  Store.set_alias store a "best";
  Store.delete_artifact store a;
  Filename.concat (Store.root store) "artifacts/x/aliases/best"

let artifact_tests =
  group "artifacts"
    [
      (* Two sessions publish one name at once in most programs, so the claim
         this reaches on every run is that their numbers are distinct, from 1.
         The rest holds in every order but is reached by chance; the sequential
         tests below hold the cases a race could leave. About a second: each
         publication syncs its blob, its version file and the producing run's
         log (Munin.Store_format, durability), and fewer than 20 programs leave
         a command uncalled on some seeds. *)
      stateful ~count:20 ~domains:2
        "publications, aliases and deletions read as some order left them"
        [
          command "open"
            (Gen.unit @-> makes store_w)
            (fun () -> { versions = []; best = None })
            (fun () ->
              let dir = temp_dir () in
              List.iter
                (fun c -> write_file (Filename.concat dir c) c)
                [ "a"; "b" ];
              (Store.open_ ~root:(Filename.concat dir "store") (), dir));
          command "start"
            (store_w ^-> makes session_w)
            Fun.id
            (fun (store, _) -> start store);
          (* Distinct numbers from 1, whichever session publishes. *)
          command "publish"
            (content @-> session_w ^-> returns int)
            (fun c m ->
              let n = List.length m.versions + 1 in
              m.versions <- m.versions @ [ (n, c, true) ];
              n)
            (fun c s ->
              let dir = Filename.dirname (Store.root (Session.store s)) in
              Artifact.version
                (Session.log_artifact s ~name:"ckpt" (Filename.concat dir c)));
          command "select"
            (select_gen @-> store_w ^-> returns selected)
            select_ref select_sys;
          command "select alias"
            (alias_gen @-> store_w ^-> returns selected)
            select_ref select_sys;
          (* The listing reads the numbers, then each version's file, so it can
             see a version deleted after a later one was published and miss that
             later one (Store.artifacts): it runs where no write runs beside
             it. *)
          command "versions"
            ~pre:(fun _ -> true)
            (store_w ^-> returns (list int))
            (fun m ->
              List.filter_map
                (fun (n, _, live) -> if live then Some n else None)
                m.versions)
            (fun (store, _) ->
              List.map Artifact.version (Store.artifacts ~name:"ckpt" store));
          (* A version deleted before the call is refused; one deleted during it
             leaves the alias naming nothing, as if set first. *)
          command "alias"
            (number @-> store_w ^-> returns bool)
            (fun n m ->
              if alive m n then (
                m.best <- Some n;
                true)
              else false)
            (fun n (store, _) ->
              match Store.artifact store "ckpt" (`Version n) with
              | None -> false
              | Some a -> (
                  match Store.set_alias store a "best" with
                  | () -> true
                  | exception Failure _ -> false));
          command "delete"
            (number @-> store_w ^-> returns unit)
            (fun n m ->
              if alive m n then (
                m.versions <-
                  List.map
                    (fun (n', c, live) -> (n', c, live && n' <> n))
                    m.versions;
                if m.best = Some n then m.best <- None))
            (fun n (store, _) ->
              Option.iter
                (Store.delete_artifact store)
                (Store.artifact store "ckpt" (`Version n)));
          command "unalias"
            (store_w ^-> returns unit)
            (fun m -> m.best <- None)
            (fun (store, _) -> Store.remove_alias store ~name:"ckpt" "best");
          command "aliases"
            (store_w ^-> returns (list (pair string int)))
            (fun m ->
              match m.best with Some n -> [ ("best", n) ] | None -> [])
            (fun (store, _) -> Store.aliases store "ckpt");
        ];
      test "a version reads back what its publication gave" (fun () ->
          let store = new_store () in
          let s = start store in
          let a =
            Session.log_artifact s ~step:7
              ~metadata:[ ("z", `String "raw"); ("a", `Int 1) ]
              ~name:"ckpt" (source "weights")
          in
          Session.finish s;
          let b = require_some (Store.artifact store "ckpt" `Latest) in
          equal ~msg:"name" string "ckpt" (Artifact.name b);
          equal ~msg:"run" string (Session.id s) (Artifact.run b);
          equal ~msg:"step" (option int) (Some 7) (Artifact.step b);
          equal ~msg:"size" int 7 (Artifact.size_bytes b);
          equal ~msg:"metadata"
            (list (pair string value))
            [ ("z", `String "raw"); ("a", `Int 1) ]
            (Artifact.metadata b);
          equal ~msg:"created" float_exact (Artifact.created_at a)
            (Artifact.created_at b);
          equal ~msg:"path" string (Artifact.path a) (Artifact.path b));
      test "the latest is a later version published before the last is deleted"
        (fun () ->
          let store = new_store () in
          let s = start store in
          ignore (Session.log_artifact s ~name:"x" (source "1"));
          let v2 = Session.log_artifact s ~name:"x" (source "2") in
          let v3 = Session.log_artifact s ~name:"x" (source "3") in
          Store.delete_artifact store v2;
          Session.finish s;
          equal (option int)
            (Some (Artifact.version v3))
            (Option.map Artifact.version (Store.artifact store "x" `Latest)));
      test "the latest is a version published while the scan read the last"
        (fun () ->
          if Sys.win32 then skip ~reason:"Windows has no mkfifo" ();
          (* A FIFO in place of version 3 holds the reader inside its scan: it
             has listed 1, 2 and 3, and its read of 3 waits for this test to
             write it. Meanwhile version 4 is published and 3 is written
             deleted, as when a writer publishes a version and prunes the one
             before it. *)
          let store = new_store () in
          let s = start store in
          let v1 = Session.log_artifact s ~name:"x" (source "1") in
          let v2 = Session.log_artifact s ~name:"x" (source "2") in
          Store.delete_artifact store v1;
          Store.delete_artifact store v2;
          let third = Filename.concat (Store.root store) "artifacts/x/3.json" in
          Unix.mkfifo third 0o644;
          let latest = ref None in
          let reader =
            Thread.create
              (fun () -> latest := Store.artifact store "x" `Latest)
              ()
          in
          (* A child writes the FIFO: its open returns once the reader has
             opened it, and a reader that never opens it ends the child at its
             alarm instead of holding the run. *)
          let pid, stdin, out = spawn [ "fifo"; Store.root store; third ] in
          let opened = In_channel.input_line out in
          if opened <> Some "opened" then (
            (* A reader blocked in the open is let go by an open for writing. *)
            (try
               Unix.close
                 (Unix.openfile third [ Unix.O_WRONLY; Unix.O_NONBLOCK ] 0)
             with Unix.Unix_error _ -> ());
            Thread.join reader;
            ignore (Unix.waitpid [] pid);
            fail "the reader never opened version 3's file");
          let v4 = Session.log_artifact s ~name:"x" (source "4") in
          (* The child writes 3's tombstone once this line reaches it. *)
          ignore (Unix.write_substring stdin "go\n" 0 3);
          Unix.close stdin;
          Thread.join reader;
          wait pid;
          close_in out;
          Session.finish s;
          equal (option int)
            (Some (Artifact.version v4))
            (Option.map Artifact.version !latest));
      test "a deleted latest number is never reused" (fun () ->
          let store = new_store () in
          let s = start store in
          ignore (Session.log_artifact s ~name:"x" (source "1"));
          let v2 = Session.log_artifact s ~name:"x" (source "2") in
          Store.delete_artifact store v2;
          let v3 = Session.log_artifact s ~name:"x" (source "3") in
          Session.finish s;
          equal ~msg:"number" int 3 (Artifact.version v3);
          equal ~msg:"check" (list string) [] (List.map fst (Store.check store)));
      test "an alias of a deleted version names nothing" (fun () ->
          let store = new_store () in
          ignore (dangling_alias store);
          equal (option int) None
            (Option.map Artifact.version
               (Store.artifact store "x" (`Alias "best"))));
      test "aliases leaves out an alias of a deleted version" (fun () ->
          let store = new_store () in
          ignore (dangling_alias store);
          equal (list (pair string int)) [] (Store.aliases store "x"));
      test "a deletion leaves its version's aliases until gc" (fun () ->
          let store = new_store () in
          let alias = dangling_alias store in
          equal (list string) [ "best" ]
            (Array.to_list (Sys.readdir (Filename.dirname alias))));
      test "gc removes a blob no version names and keeps a named one" (fun () ->
          let store = new_store () in
          let s = start store in
          let v1 = Session.log_artifact s ~name:"x" (source "1") in
          let v2 = Session.log_artifact s ~name:"x" (source "2") in
          Session.finish s;
          Store.delete_artifact store v1;
          ignore (Store.gc store);
          equal (list string)
            [ Filename.basename (Artifact.path v2) ]
            (Array.to_list (Sys.readdir (Filename.dirname (Artifact.path v1)))));
      test "an alias moved while its old version is deleted reads the new"
        (fun () ->
          if Sys.win32 then skip ~reason:"Windows has no mkfifo" ();
          (* A FIFO in place of version 1 holds the reader after it read the
             alias, which names 1: meanwhile version 2 is published, the alias
             moves to it, and 1 is written deleted, as a "best" checkpoint moves
             on and the old one is pruned. *)
          let store = new_store () in
          let dir = Filename.concat (Store.root store) "artifacts/x" in
          mkdir_p (Store.root store);
          mkdir_p (Filename.dirname dir);
          mkdir_p dir;
          mkdir_p (Filename.concat dir "aliases");
          write_file (Filename.concat dir "aliases/best") "1\n";
          let first = Filename.concat dir "1.json" in
          Unix.mkfifo first 0o644;
          let read = ref None in
          let reader =
            Thread.create
              (fun () -> read := Store.artifact store "x" (`Alias "best"))
              ()
          in
          let pid, stdin, out = spawn [ "fifo"; Store.root store; first ] in
          let opened = In_channel.input_line out in
          if opened <> Some "opened" then (
            (try
               Unix.close
                 (Unix.openfile first [ Unix.O_WRONLY; Unix.O_NONBLOCK ] 0)
             with Unix.Unix_error _ -> ());
            Thread.join reader;
            ignore (Unix.waitpid [] pid);
            fail "the reader never opened version 1's file");
          let s = start store in
          let v2 = Session.log_artifact s ~name:"x" (source "2") in
          Session.finish s;
          Store.set_alias store v2 "best";
          ignore (Unix.write_substring stdin "go\n" 0 3);
          Unix.close stdin;
          Thread.join reader;
          wait pid;
          close_in out;
          equal (option int)
            (Some (Artifact.version v2))
            (Option.map Artifact.version !read));
      test "gc removes an alias of a deleted version" (fun () ->
          let store = new_store () in
          let alias = dangling_alias store in
          ignore (Store.gc store);
          equal (list string) []
            (Array.to_list (Sys.readdir (Filename.dirname alias))));
      test "check passes an alias of a deleted version" (fun () ->
          let store = new_store () in
          ignore (dangling_alias store);
          equal (list string) [] (List.map fst (Store.check store)));
      test "set_alias refuses a version its store cannot read" (fun () ->
          let store = new_store () and other = new_store () in
          let s = start other in
          let a = Session.log_artifact s ~name:"x" (source "1") in
          Session.finish s;
          raises_match Exn.failure (fun () -> Store.set_alias store a "best"));
      test "set_alias refuses a deleted version" (fun () ->
          let store = new_store () in
          let s = start store in
          let a = Session.log_artifact s ~name:"x" (source "1") in
          Session.finish s;
          Store.delete_artifact store a;
          raises_match Exn.failure (fun () -> Store.set_alias store a "best");
          let aliases =
            Filename.concat (Store.root store) "artifacts/x/aliases"
          in
          equal ~msg:"alias files" (list string) []
            (try Array.to_list (Sys.readdir aliases) with Sys_error _ -> []));
      test "an alias spelled as a number names an alias" (fun () ->
          let store = new_store () in
          let s = start store in
          let a1 = Session.log_artifact s ~name:"x" (source "1") in
          ignore (Session.log_artifact s ~name:"x" (source "2"));
          Session.finish s;
          Store.set_alias store a1 "2";
          equal (option int) (Some 1)
            (Option.map Artifact.version
               (Store.artifact store "x" (`Alias "2"))));
      test "a version whose digest is not 64 hexadecimal digits is unreadable"
        (fun () ->
          let store = new_store () in
          let dir = Filename.concat (Store.root store) "artifacts" in
          mkdir_p (Store.root store);
          mkdir_p dir;
          mkdir_p (Filename.concat dir "x");
          write_file
            (Filename.concat dir "x/1.json")
            {|{"format":1,"time":1.0,"run":"r","digest":"blake2b-256:../../../etc/passwd","size":1,"metadata":{}}|};
          equal (option string) None
            (Option.map Artifact.path (Store.artifact store "x" (`Version 1))));
      test "a deleted run's versions stay and name it" (fun () ->
          let store = new_store () in
          let s = start store in
          ignore (Session.log_artifact s ~name:"x" (source "1"));
          Session.finish s;
          Store.delete_run store (Session.id s);
          equal (list string)
            [ Session.id s ]
            (List.map Artifact.run (Store.artifacts store)));
      cases "set_alias refuses a name that is not one segment"
        ~id:(Printf.sprintf "%S") [ ""; "a/b"; ".x"; "a\nb" ] (fun alias ->
          let store = new_store () in
          let s = start store in
          let a = Session.log_artifact s ~name:"x" (source "1") in
          Session.finish s;
          raises_match Exn.invalid_arg (fun () -> Store.set_alias store a alias));
    ]

(* Runs that two domains delete and annotate at once. A run's reference is
   whether it exists; its system side is its id. A call the store refuses
   answers false. *)
let race_store = lazy (new_store ())

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

(* [ended_run ()] is the id of a new finished run of the race's store, its log
   written from the format's text. A session's start and finish sync, and each
   program runs 50 times, every candidate of a shrink included: with synced
   starts a failure on Windows shrank for 509 s, where windtrap enforces no
   timeout. *)
let ended_run =
  let next = Atomic.make 0 in
  fun () ->
    let runs = Filename.concat (Store.root (Lazy.force race_store)) "runs" in
    mkdir_p (Store.root (Lazy.force race_store));
    mkdir_p runs;
    let id = Printf.sprintf "race-%d" (Atomic.fetch_and_add next 1) in
    let dir = Filename.concat runs id in
    Unix.mkdir dir 0o755;
    write_file
      (Filename.concat dir "events.jsonl")
      (text_of_lines
         [ start_line; {|{"type":"end","time":2.0,"status":"finished"}|} ]);
    id

(* A run's notes as two domains set, remove and read them. A reference is the
   notes a run holds; a system side is its id. *)
let notes_w : (string option ref, string) abstract = abstract "n"

(* A conflict needs one run and two domains, so a program's prefix stays short:
   a program syncs or deletes on most calls and runs 50 times, and a failure's
   shrink runs up to 10 000 programs, with no limit on Windows, where one of
   20-call programs took 688 s. *)
let race_steps = 4

(* A store's runs as a program writes and annotates them, and the model of what
   a reader must see: each run's samples, tags, notes, summary and end. *)
type run_model = {
  mutable values : float list; (* reversed *)
  mutable tags : string list;
  mutable notes : string option;
  mutable summary : (string * float) list;
  mutable ended : bool;
}

let written_w : (run_model, Store.t * Session.t * Session.metric) abstract =
  abstract ~release:(fun (_, s, _) -> Session.finish s) "w"

let listing_w : (unit, Store.t) abstract = abstract "t"
let tag_gen = Gen.of_list ~pp:Format.pp_print_string [ "a"; "A"; "b" ]

let read_back =
  pair
    (pair (list string) (option string))
    (pair (pair status (list float_exact)) (list (pair string value)))

let open_run m = not m.ended

(* Runs whose log cannot be opened, in a directory the user cannot search, and
   whose log opens but cannot be read, a directory. Each is the run's id and
   directory. The runner gives the permissions back to remove them. *)
let unreadable_logs =
  [
    ( "unsearchable",
      fun store ->
        if Sys.win32 then
          skip ~reason:"Windows keeps no search permission in a mode" ();
        if Unix.geteuid () = 0 then
          skip ~reason:"root searches any directory" ();
        let s = start store in
        Session.finish s;
        let dir = Run.dir (reread store s) in
        Unix.chmod dir 0o600;
        (Session.id s, dir) );
    ( "a directory",
      fun store ->
        let dir = Filename.concat (Store.root store) "runs/dir" in
        mkdir_p (Store.root store);
        mkdir_p (Filename.dirname dir);
        mkdir_p dir;
        mkdir_p (Filename.concat dir "events.jsonl");
        ("dir", dir) );
  ]

let runs_tests =
  group "runs"
    [
      test "opening and reading a missing root creates nothing" (fun () ->
          let parent = temp_dir () in
          let store = Store.open_ ~root:(Filename.concat parent "store") () in
          equal ~msg:"runs" (list string) []
            (List.map Run.id (Store.runs store));
          equal ~msg:"artifacts" (list string) []
            (List.map Artifact.name (Store.artifacts store));
          equal ~msg:"gc" int 0 (Store.gc store);
          equal ~msg:"files" (list string) []
            (Array.to_list (Sys.readdir parent)));
      cases "the default root is the first variable that applies" ~id:fst
        [
          ("RAVEN_TRACKING_DIR", ([ Some "/r"; Some "/x"; Some "/h" ], "/r"));
          ("XDG_DATA_HOME", ([ None; Some "/x"; Some "/h" ], "/x/raven/munin"));
          ( "a relative XDG_DATA_HOME",
            ([ Some ""; Some "x"; Some "/h" ], "/h/.local/share/raven/munin") );
          ("HOME", ([ None; None; Some "/h" ], "/h/.local/share/raven/munin"));
        ]
        (fun (_, (vars, root)) ->
          if Sys.win32 then skip ~reason:"Windows spells paths with \\" ();
          List.iter2 setenv
            [ "RAVEN_TRACKING_DIR"; "XDG_DATA_HOME"; "HOME" ]
            vars;
          equal string root (Store.root (Store.open_ ())));
      test "open_ without a root to take raises" (fun () ->
          if Sys.win32 then skip ~reason:"Windows takes LOCALAPPDATA" ();
          List.iter
            (fun v -> setenv v None)
            [ "RAVEN_TRACKING_DIR"; "XDG_DATA_HOME"; "HOME" ];
          raises_match Exn.failure (fun () -> Store.open_ ()));
      cases "an id that is not one segment names no run"
        ~id:(Printf.sprintf "%S") [ ""; ".hidden"; "a/b"; "../runs/.hidden" ]
        (fun id ->
          let store = new_store () in
          (* A run's log in a [.]-named directory, which readers ignore. *)
          let dir = Filename.concat (Store.root store) "runs/.hidden" in
          mkdir_p (Filename.dirname dir);
          mkdir_p dir;
          write_file
            (Filename.concat dir "events.jsonl")
            (text_of_lines [ start_line ]);
          equal (option string) None (Option.map Run.id (Store.run store id)));
      test "runs are listed newest first by start time, then by id" (fun () ->
          let store = new_store () in
          let write id time =
            let dir =
              Filename.concat (Filename.concat (Store.root store) "runs") id
            in
            mkdir_p (Store.root store);
            mkdir_p (Filename.dirname dir);
            mkdir_p dir;
            write_file
              (Filename.concat dir "events.jsonl")
              (text_of_lines [ start_line_at time ])
          in
          write "b" 1.;
          write "a" 2.;
          write "c" 1.;
          equal (list string) [ "a"; "c"; "b" ]
            (List.map Run.id (Store.runs store)));
      test "a listing with prev sees changed notes and tags" (fun () ->
          let store = new_store () in
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
          equal (list string) [ "t" ] (Run.tags (List.hd listing)));
      test
        "a listing with prev reads again tags whose time is not a second older \
         than its read" (fun () ->
          let store = new_store () in
          let s = start ~tags:[ "a" ] store in
          Session.finish s;
          let tags = Filename.concat (Run.dir (reread store s)) "tags" in
          (* A file system that keeps whole seconds gives a change the time it
             had, so another tag leaves the directory's time, size and inode as
             they were. An hour after the log's last write is no older than the
             listing's read, whatever the machine's load. *)
          let second =
            Float.of_int (truncate (Run.updated_at (reread store s))) +. 3600.
          in
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
      test "a listing allocates less than 64 KiB a run of small files"
        (fun () ->
          (* A buffer of a read chunk for each of a run's small files cost 64
             KiB apiece. *)
          let store = new_store () in
          let runs = 8 in
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
      cases "a run whose log cannot be opened or read is left out" ~id:fst
        unreadable_logs (fun (_, unreadable) ->
          let store = new_store () in
          ignore (unreadable store);
          equal (list string) [] (List.map Run.id (Store.runs store)));
      cases "run raises Sys_error for a log it cannot open or read" ~id:fst
        unreadable_logs (fun (_, unreadable) ->
          let store = new_store () in
          let id, _ = unreadable store in
          raises_match Exn.sys_error (fun () -> Store.run store id));
      cases "notes, a start and a failure of any length read back whole"
        ~id:string_of_int
        [ 0; 1; 511; 512; 513; 4096; 65535; 65536; 65537; 200_000 ] (fun n ->
          let store = new_store () in
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
      test "tags are equal by their bytes" (fun () ->
          let store = new_store () in
          let s = start ~tags:[ "best" ] store in
          Session.finish s;
          Store.add_tags store (Session.id s) [ "Best"; "caf\xc3\xa9"; "best" ];
          equal (list string)
            [ "Best"; "best"; "caf\xc3\xa9" ]
            (Run.tags (reread store s)));
      cases "add_tags refuses a string that is not a tag"
        ~id:(Printf.sprintf "%S") [ ""; "a,b"; "a\nb"; "\x7f"; "\xff" ]
        (fun tag ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          raises_match Exn.invalid_arg (fun () ->
              Store.add_tags store (Session.id s) [ tag ]));
      test "set_notes refuses text that is not UTF-8" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          raises_match Exn.invalid_arg (fun () ->
              Store.set_notes store (Session.id s) (Some "\xff")));
      test "notes that another handle holds for a moment are removed" (fun () ->
          (* A handle that does not share deletion, as a program other than
             munin may hold, makes Windows refuse the removal until it closes,
             and set_notes waits for it; elsewhere nothing refuses it. *)
          let store = new_store () in
          let s = start store in
          Session.finish s;
          Store.set_notes store (Session.id s) (Some "n");
          let notes = Filename.concat (Run.dir (reread store s)) "notes" in
          let fd = Unix.openfile notes [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 in
          let held =
            Thread.create
              (fun () ->
                if Sys.win32 then Thread.delay 0.2;
                Unix.close fd)
              ()
          in
          Fun.protect ~finally:(fun () -> Thread.join held) @@ fun () ->
          Store.set_notes store (Session.id s) None;
          equal (option string) None (Run.notes (reread store s)));
      stateful ~count:40 "a store's readers see what its writers did"
        [
          command "open" (Gen.unit @-> makes listing_w) (fun () -> ()) new_store;
          command "start"
            (listing_w ^-> makes written_w)
            (fun () ->
              {
                values = [];
                tags = [];
                notes = None;
                summary = [];
                ended = false;
              })
            (fun store ->
              let s = start store in
              (store, s, Session.metric s "m"));
          command "log"
            ~pre:(fun _ m -> open_run m)
            (Gen.pair (Gen.int_range 0 9) (Gen.float_range (-1.) 1.)
            @-> written_w ^-> returns unit)
            (fun (_, v) m -> m.values <- v :: m.values)
            (fun (step, v) (_, _, metric) -> Session.log metric ~step v);
          command "summary"
            ~pre:(fun _ m -> open_run m)
            (Gen.pair
               (Gen.of_list ~pp:Format.pp_print_string [ "k"; "l" ])
               (Gen.float_range 0. 1.)
            @-> written_w ^-> returns unit)
            (fun (k, v) m ->
              m.summary <- (k, v) :: List.remove_assoc k m.summary)
            (fun (k, v) (_, s, _) -> Session.set_summary s [ (k, `Float v) ]);
          command "finish" ~pre:open_run
            (written_w ^-> returns unit)
            (fun m -> m.ended <- true)
            (fun (_, s, _) -> Session.finish s);
          command "tag"
            (tag_gen @-> written_w ^-> returns unit)
            (fun t m -> m.tags <- List.sort_uniq compare (t :: m.tags))
            (fun t (store, s, _) -> Store.add_tags store (Session.id s) [ t ]);
          command "untag"
            (tag_gen @-> written_w ^-> returns unit)
            (fun t m -> m.tags <- List.filter (( <> ) t) m.tags)
            (fun t (store, s, _) ->
              Store.remove_tags store (Session.id s) [ t ]);
          command "note"
            (Gen.option (Gen.of_list ~pp:Format.pp_print_string [ "x"; "y\nz" ])
            @-> written_w ^-> returns unit)
            (fun n m -> m.notes <- n)
            (fun n (store, s, _) -> Store.set_notes store (Session.id s) n);
          command "read"
            (written_w ^-> returns read_back)
            (fun m ->
              ( (m.tags, m.notes),
                ( ((if m.ended then `Finished else `Running), List.rev m.values),
                  List.sort compare
                    (List.map (fun (k, v) -> (k, `Float v)) m.summary) ) ))
            (fun (store, s, _) ->
              let run =
                List.find (fun r -> Run.id r = Session.id s) (Store.runs store)
              in
              let h = History.read run in
              ( (Run.tags run, Run.notes run),
                ( (Run.status run, values_of (History.series h "m")),
                  History.summary h ) ));
        ];
      stateful ~count:30 ~steps:race_steps ~domains:2
        "notes set, removed and read on two domains read as some order left \
         them"
        [
          command "start"
            (Gen.unit @-> makes notes_w)
            (fun () -> ref None)
            ended_run;
          command "set"
            (Gen.bool @-> notes_w ^-> returns unit)
            (fun some r -> r := if some then Some "n" else None)
            (fun some id ->
              Store.set_notes (Lazy.force race_store) id
                (if some then Some "n" else None));
          command "read"
            (notes_w ^-> returns (option string))
            (fun r -> !r)
            (fun id ->
              Run.notes (require_some (Store.run (Lazy.force race_store) id)));
        ];
    ]

let deletion_tests =
  group "deleting runs"
    [
      test "a deleted run goes" (fun () ->
          let store = new_store () in
          let a = start store in
          Session.finish a;
          let b = start store in
          Session.finish b;
          Store.delete_run store (Session.id a);
          equal (list string)
            [ Session.id b ]
            (List.map Run.id (Store.runs store)));
      test "a run whose session is open is not deleted" (fun () ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.failure (fun () ->
              Store.delete_run store (Session.id s)));
      test "a silent run is not deleted" (fun () ->
          let store = new_store () in
          (* A run without its writer file, as one copied, reads silent. *)
          let run = written_run store (log_text []) in
          equal ~msg:"silent" status `Silent (Run.status run);
          raises_match Exn.failure (fun () -> Store.delete_run store "spec"));
      test "a run that does not exist is not deleted" (fun () ->
          raises_match Exn.failure (fun () ->
              Store.delete_run (new_store ()) "none"));
      test "a run whose log the user cannot write is deleted" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          Unix.chmod (log_path (reread store s)) 0o444;
          Store.delete_run store (Session.id s);
          equal (option string) None
            (Option.map Run.id (Store.run store (Session.id s))));
      test "a run whose log another handle holds for a moment is deleted"
        (fun () ->
          (* A handle that does not share deletion makes Windows refuse the
             rename of the log until it closes, and delete_run waits for it;
             elsewhere nothing refuses it. *)
          let store = new_store () in
          let s = start store in
          Session.finish s;
          let id = Session.id s in
          let fd =
            Unix.openfile
              (log_path (reread store s))
              [ Unix.O_RDONLY; Unix.O_CLOEXEC ]
              0
          in
          let held =
            Thread.create
              (fun () ->
                if Sys.win32 then Thread.delay 0.2;
                Unix.close fd)
              ()
          in
          Fun.protect ~finally:(fun () -> Thread.join held) @@ fun () ->
          Store.delete_run store id;
          equal (option string) None (Option.map Run.id (Store.run store id)));
      test "what an interrupted deletion leaves is no run, and gc removes it"
        (fun () ->
          let store = new_store () in
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
          equal ~msg:"listed" (list string) [ kept ]
            (List.map Run.id (Store.runs store));
          equal ~msg:"checked" (list string)
            (List.sort compare [ renamed_dir; emptied_dir; dot ])
            (List.map fst (Store.check store));
          equal ~msg:"removed" int 3 (Store.gc store);
          equal ~msg:"left" (list string) [ kept ]
            (Array.to_list (Sys.readdir runs)));
      stateful ~count:20 ~steps:race_steps ~domains:2
        "a deletion and annotations on two domains leave the run whole or gone"
        [
          command "start"
            (Gen.unit @-> makes run_w)
            (fun () -> ref true)
            ended_run;
          command "delete"
            (run_w ^-> returns bool)
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
          command "exists"
            (run_w ^-> returns bool)
            (fun r -> !r)
            (fun id -> Store.run (Lazy.force race_store) id <> None);
        ];
    ]

(* [gc_while_publishing store] runs [Store.gc] in a child process while this
   process holds the store's shared lock, as a publication in progress does,
   with a [.]-named entry under blobs/. It checks the entry is there while the
   child waits, and is the count the child printed and the entries left once the
   lock is let go. *)
let gc_while_publishing store =
  let blobs = Filename.concat (Store.root store) "blobs/blake2b-256" in
  let entry = Filename.concat blobs ".in-progress" in
  let pid, stdin, out =
    Store.with_publication_lock store (fun () ->
        mkdir_p (Filename.concat (Store.root store) "blobs");
        mkdir_p blobs;
        write_file entry "x";
        let child = spawn [ "gc"; Store.root store ] in
        let pid, _, _ = child in
        (* On Linux the child's wait is the event: /proc/locks lists a blocked
           lock as "-> FLOCK ... <pid> ...". *)
        if Lazy.force kernel = "Linux" then
          eventually ~within:30. (fun () ->
              let waiters =
                List.filter_map
                  (fun l ->
                    match
                      List.filter (( <> ) "") (String.split_on_char ' ' l)
                    with
                    | _ :: "->" :: _ :: _ :: _ :: pid :: _ -> Some pid
                    | _ -> None)
                  (lines (read_file "/proc/locks"))
              in
              mem ~msg:"gc waits" string (string_of_int pid) waiters);
        equal ~msg:"the entry while gc waits" (list string) [ ".in-progress" ]
          (Array.to_list (Sys.readdir blobs));
        child)
  in
  Unix.close stdin;
  let removed = input_line out in
  wait pid;
  close_in out;
  (removed, Array.to_list (Sys.readdir blobs))

let maintenance_tests =
  group "maintenance"
    [
      (* Elsewhere than Linux no event shows the child waiting, so a gc that
         ignored the lock passes there when the child is late. *)
      test "gc waits for a publication in progress" (fun () ->
          let store = new_store () in
          let removed, left = gc_while_publishing store in
          equal ~msg:"removed" string "1" removed;
          equal ~msg:"left" (list string) [] left);
      test "gc removes the work writers left under blobs/ and artifacts/"
        (fun () ->
          let store = new_store () in
          let s = start store in
          ignore (Session.log_artifact s ~name:"x" (source "1"));
          Session.finish s;
          let left =
            [
              Filename.concat (Store.root store) "blobs/blake2b-256/.blob";
              Filename.concat (Store.root store) "artifacts/.name";
              Filename.concat (Store.root store) "artifacts/x/.1.json";
            ]
          in
          List.iter (fun p -> write_file p "x") left;
          equal ~msg:"checked" (list string) (List.sort compare left)
            (List.map fst (Store.check store));
          equal ~msg:"removed" int 3 (Store.gc store);
          equal ~msg:"left" (list string) [] (List.filter Sys.file_exists left));
      test "gc leaves a run directory a start has claimed" (fun () ->
          let store = new_store () in
          let dir = Filename.concat (Store.root store) "runs/claimed" in
          mkdir_p (Store.root store);
          mkdir_p (Filename.dirname dir);
          mkdir_p dir;
          equal ~msg:"removed" int 0 (Store.gc store);
          equal ~msg:"checked" (list string) [ dir ]
            (List.map fst (Store.check store));
          equal ~msg:"left" (list string) [ "claimed" ]
            (Array.to_list (Sys.readdir (Filename.dirname dir))));
      test "check names a damaged line" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          let log = log_path (reread store s) in
          let first, last =
            match lines (read_file log) with
            | [ first; last ] -> (first, last)
            | l -> fail (Printf.sprintf "%d lines" (List.length l))
          in
          write_file log (text_of_lines [ first; "not a record"; last ]);
          equal (list string) [ log ] (List.map fst (Store.check store)));
      test "check names the bytes after an ended run's last line" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          let log = log_path (reread store s) in
          write_file log (read_file log ^ {|{"type":"metr|});
          equal (list string) [ log ] (List.map fst (Store.check store)));
      test "check names the bytes after a crashed run's last line" (fun () ->
          let store = new_store () in
          let run = exited_run store in
          if Run.status run <> `Crashed then
            skip ~reason:"this system names no boot, so the run reads silent" ();
          write_file (log_path run)
            (read_file (log_path run) ^ {|{"type":"metr|});
          equal (list string)
            [ log_path run ]
            (List.map fst (Store.check store)));
      test "check names no bytes after a running run's last line" (fun () ->
          let store = new_store () in
          let pid, stdin, out = spawn [ "hold"; Store.root store ] in
          let id = input_line out in
          let log = log_path (require_some (Store.run store id)) in
          let fd =
            Unix.openfile log [ Unix.O_WRONLY; Unix.O_APPEND; Unix.O_CLOEXEC ] 0
          in
          ignore (Unix.write_substring fd {|{"type":"metr|} 0 13);
          Unix.close fd;
          let checked = List.map fst (Store.check store) in
          Unix.close stdin;
          wait pid;
          close_in out;
          equal (list string) [] checked);
      test "check names a damaged tag file" (fun () ->
          let store = new_store () in
          let s = start ~tags:[ "a" ] store in
          Session.finish s;
          let tag =
            Filename.concat (Run.dir (reread store s)) ("tags/" ^ hex "a")
          in
          write_file tag "b";
          equal (list string) [ tag ] (List.map fst (Store.check store)));
      test "check names a file no record names" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.log_file s ~step:5 ~key:"viz" (source ~suffix:".png" "png");
          Session.finish s;
          let dir = Filename.concat (Run.dir (reread store s)) "files/viz" in
          write_file (Filename.concat dir "9.png") "png";
          equal (list string)
            [ Filename.concat dir "9.png" ]
            (List.map fst (Store.check store)));
      test "check names no file that an orphan record names" (fun () ->
          let store = new_store () in
          ignore (abandoned_file store);
          equal (list string) [] (List.map fst (Store.check store)));
      test "check names an alias that names no version" (fun () ->
          let store = new_store () in
          let s = start store in
          ignore (Session.log_artifact s ~name:"x" (source "1"));
          Session.finish s;
          let alias =
            Filename.concat (Store.root store) "artifacts/x/aliases/best"
          in
          mkdir_p (Filename.dirname alias);
          write_file alias "9\n";
          equal (list string) [ alias ] (List.map fst (Store.check store)));
      test "check names a version whose blob is missing" (fun () ->
          let store = new_store () in
          let s = start store in
          let a = Session.log_artifact s ~name:"x" (source "1") in
          Session.finish s;
          Sys.remove (Artifact.path a);
          equal (list string)
            [ Artifact.path a ]
            (List.map fst (Store.check store)));
      test "check names a version file that does not follow the format"
        (fun () ->
          let store = new_store () in
          let file = Filename.concat (Store.root store) "artifacts/x/1.json" in
          mkdir_p (Filename.concat (Store.root store) "artifacts");
          mkdir_p (Filename.dirname file);
          write_file file {|{"format":1,"time":1.0}|};
          equal (list string) [ file ] (List.map fst (Store.check store)));
      test "check names a blob whose content does not match its digest"
        (fun () ->
          let store = new_store () in
          let s = start store in
          let a = Session.log_artifact s ~name:"x" (source "1") in
          Session.finish s;
          Unix.chmod (Artifact.path a) 0o644;
          write_file (Artifact.path a) "2";
          equal (list string)
            [ Artifact.path a ]
            (List.map fst (Store.check store)));
      test "check names a hint that does not parse" (fun () ->
          let store = new_store () in
          let run = ended_with_hint store in
          write_file (hint_path run) "{\"format\":1,";
          equal (list string)
            [ hint_path run ]
            (List.map fst (Store.check store)));
      test "check names a hint that disagrees with its log" (fun () ->
          let store = new_store () in
          let run = ended_with_hint store in
          write_file (hint_path run)
            (replace_first
               (read_file (hint_path run))
               ~sub:{|"count":1|} ~by:{|"count":2|});
          equal (list string)
            [ hint_path run ]
            (List.map fst (Store.check store)));
      test "check passes a hint of other rules" (fun () ->
          let store = new_store () in
          let run = ended_with_hint store in
          write_file (hint_path run)
            (replace_first
               (replace_first
                  (read_file (hint_path run))
                  ~sub:(Printf.sprintf {|"rules":%d|} Store_format.rules)
                  ~by:(Printf.sprintf {|"rules":%d|} (Store_format.rules + 1)))
               ~sub:{|"count":1|} ~by:{|"count":2|});
          equal (list string) [] (List.map fst (Store.check store)));
      test "check reads a hint's floats bit for bit" (fun () ->
          let store = new_store () in
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
      cases "check names a run whose log cannot be opened or read" ~id:fst
        unreadable_logs (fun (_, unreadable) ->
          let store = new_store () in
          let _, dir = unreadable store in
          equal (list string) [ dir ] (List.map fst (Store.check store)));
    ]

let store_tests =
  group "Store"
    [ runs_tests; deletion_tests; artifact_tests; maintenance_tests ]

(* Session *)

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

let lifecycle_tests =
  group "starting and ending"
    [
      cases "start refuses what the format cannot hold" ~id:fst
        [
          ("a tag with a comma", fun store -> start ~tags:[ "a,b" ] store);
          ("an empty experiment", fun store -> start ~experiment:"" store);
          ( "an experiment with a line feed",
            fun store -> start ~experiment:"a\nb" store );
          ("a name with DEL", fun store -> start ~name:"\x7f" store);
          ( "an empty param key",
            fun store -> start ~params:[ ("", `Int 1) ] store );
          ( "a param key twice",
            fun store -> start ~params:[ ("k", `Int 1); ("k", `Int 2) ] store );
          ( "a string that is not UTF-8",
            fun store -> start ~params:[ ("k", `String "\xff") ] store );
          ("steps below 0", fun store -> start ~steps:(-1) store);
          ("steps past 2^53", fun store -> start ~steps:((1 lsl 53) + 1) store);
        ]
        (fun (_, f) ->
          let store = new_store () in
          raises_match Exn.invalid_arg (fun () -> f store);
          equal ~msg:"no run" (list string) []
            (List.map Run.id (Store.runs store)));
      test "start takes a parent whose log cannot be opened" (fun () ->
          if Sys.win32 then
            skip ~reason:"Windows keeps no search permission in a mode" ();
          if Unix.geteuid () = 0 then
            skip ~reason:"root searches any directory" ();
          let store = new_store () in
          let parent = start store in
          Session.finish parent;
          let dir = Run.dir (reread store parent) in
          Unix.chmod dir 0o600;
          Fun.protect ~finally:(fun () -> Unix.chmod dir 0o755) @@ fun () ->
          let s = start ~parent:(Session.id parent) store in
          Session.finish s;
          equal (option string)
            (Some (Session.id parent))
            (Run.parent (reread store s)));
      test "start refuses a parent the store does not hold" (fun () ->
          let store = new_store () in
          raises_match Exn.failure (fun () -> start ~parent:"none" store));
      test "within finishes as finished and is f's value" (fun () ->
          let store = new_store () in
          let id = Session.within (start store) Session.id in
          equal status `Finished
            (Run.status (require_some (Store.run store id))));
      test "within finishes as failed and raises again" (fun () ->
          let store = new_store () in
          let id = ref "" in
          raises (Failure "boom") (fun () ->
              Session.within (start store) (fun s ->
                  id := Session.id s;
                  failwith "boom"));
          equal status (`Failed "Failure(\"boom\")")
            (Run.status (require_some (Store.run store !id))));
      test "within finishes as killed on Sys.Break" (fun () ->
          let store = new_store () in
          let id = ref "" in
          (* Sys.Break passes through the test's verbs, so it is caught here. *)
          (try
             Session.within (start store) (fun s ->
                 id := Session.id s;
                 raise Sys.Break)
           with Sys.Break -> ());
          equal status `Killed (Run.status (require_some (Store.run store !id))));
      test "within raises f's exception when finishing raises too" (fun () ->
          let store = new_store () in
          let s = start store in
          let reading, await, _ =
            counting (fun n -> if n > 1 then raise Exit else [])
          in
          Session.sample s ~every:0.001 [ reading ];
          await 2;
          (* finish raises the sampler's Exit after it wrote the end. *)
          raises (Failure "boom") (fun () ->
              Session.within s (fun _ -> failwith "boom"));
          equal status (`Failed "Failure(\"boom\")")
            (Run.status (reread store s)));
      test "finishing a finished session does nothing" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          let log = read_file (log_path (reread store s)) in
          Session.finish ~status:`Killed s;
          equal ~msg:"log" string log (read_file (log_path (reread store s)));
          equal status `Finished (Run.status (reread store s)));
      test "writes on a finished session raise" (fun () ->
          let store = new_store () in
          let s = start store in
          let m = Session.metric s "m" in
          Session.finish s;
          raises_match Exn.invalid_arg (fun () -> Session.log m ~step:0 1.));
      test "sync returns on an open session" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.log (Session.metric s "m") ~step:0 1.;
          Session.sync s;
          Session.finish s;
          equal (list float_exact) [ 1. ]
            (values_of (History.series (history store s) "m")));
      test "a session's store and id are its run's" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          equal ~msg:"store" string (Store.root store)
            (Store.root (Session.store s));
          equal ~msg:"id" (list string)
            [ Session.id s ]
            (List.map Run.id (Store.runs store)));
      test "a refused write raises Sys_error and leaves the log as it was"
        (fun () ->
          let store = new_store () in
          match limited [ "refused"; Store.root store ] with
          | [ id; size; raised ] ->
              equal ~msg:"raised" string "Sys_error" raised;
              let run = require_some (Store.run store id) in
              equal ~msg:"length" int (int_of_string size)
                (Unix.stat (log_path run)).Unix.st_size
          | l -> fail ("the child printed: " ^ String.concat " | " l));
      test "resume refuses a run that has ended" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.finish s;
          raises_match Exn.failure (fun () ->
              Session.resume ~provenance (reread store s) ~step:0));
      cases "resume refuses a step outside [0;2^53]" ~id:string_of_int
        [ -1; (1 lsl 53) + 1 ]
        (fun step ->
          let store = new_store () in
          let run = written_run store (log_text []) in
          raises_match Exn.invalid_arg (fun () ->
              Session.resume ~provenance run ~step));
    ]

(* What one session wrote, as a model: each sample's key, step and value, the
   summary and the steps of the files. *)
let logged_w :
    ( (string * int * float) list ref * (string * float) list ref * int list ref,
      Session.t * Session.metric * Session.metric * string )
    abstract =
  abstract ~release:(fun (s, _, _, _) -> Session.finish s) "l"

let metrics_tests =
  group "metrics and summary"
    [
      cases "metric refuses a name that is not a key" ~id:(Printf.sprintf "%S")
        [ ""; "a//b"; ".x"; "a/.b"; "a/.."; "a\nb"; "\x7f"; "\xff"; "/a"; "a/" ]
        (fun key ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.invalid_arg (fun () -> Session.metric s key));
      test "an equal declaration writes no record" (fun () ->
          let store = new_store () in
          let s = start store in
          let a = Session.metric s ~goal:`Minimize "a" in
          let a' = Session.metric s ~goal:`Minimize "a" in
          Session.log a ~step:0 1.;
          Session.log a' ~step:1 2.;
          Session.finish s;
          let declarations =
            List.filter
              (fun l -> String.starts_with ~prefix:{|{"type":"metric",|} l)
              (lines (read_file (log_path (reread store s))))
          in
          equal ~msg:"metric records" int 1 (List.length declarations);
          equal (list float_exact) [ 1.; 2. ]
            (values_of (History.series (history store s) "a")));
      test "a declaration with another goal or axis raises" (fun () ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          let b = Session.metric s "b" in
          ignore (Session.metric s ~goal:`Minimize "a");
          raises_match ~msg:"goal" Exn.invalid_arg (fun () ->
              Session.metric s ~goal:`Maximize "a");
          raises_match ~msg:"axis" Exn.invalid_arg (fun () ->
              Session.metric s ~goal:`Minimize ~step_metric:b "a"));
      test "a step metric of another session is refused" (fun () ->
          let store = new_store () in
          let s1 = start store and s2 = start store in
          Fun.protect ~finally:(fun () ->
              Session.finish s1;
              Session.finish s2)
          @@ fun () ->
          let m = Session.metric s1 "m" in
          raises_match Exn.invalid_arg (fun () ->
              Session.metric s2 ~step_metric:m "a"));
      test "a reading's key cannot become a metric" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.sample s ~every:1000. [ (fun () -> [ ("k", 1.) ]) ];
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.invalid_arg (fun () -> Session.metric s "k"));
      test "key is a handle's key" (fun () ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          equal string "train/loss"
            (Session.key (Session.metric s "train/loss")));
      test "log_metrics writes one record of every pair at one step and time"
        (fun () ->
          let store = new_store () in
          let s = start store in
          let a = Session.metric s "a" and b = Session.metric s "b" in
          Session.log_metrics s ~step:3 ~time:5. [ (a, 1.); (b, 2.) ];
          Session.log_metrics s ~step:4 [];
          Session.finish s;
          let records =
            List.filter
              (fun l -> String.starts_with ~prefix:{|{"type":"metrics",|} l)
              (lines (read_file (log_path (reread store s))))
          in
          equal ~msg:"records" int 1 (List.length records);
          let h = history store s in
          equal ~msg:"a" (list sample)
            [ { step = 3; time = 5.; value = 1. } ]
            (samples_of (History.series h "a"));
          equal ~msg:"b" (list sample)
            [ { step = 3; time = 5.; value = 2. } ]
            (samples_of (History.series h "b")));
      cases "log_metrics refuses a bad step, time or pair" ~id:fst
        [
          ( "a step below 0",
            fun (s, a, _) -> Session.log_metrics s ~step:(-1) [ (a, 0.) ] );
          ( "a step past 2^53",
            fun (s, a, _) ->
              Session.log_metrics s ~step:((1 lsl 53) + 1) [ (a, 0.) ] );
          ( "a NaN time",
            fun (s, a, _) ->
              Session.log_metrics s ~step:0 ~time:Float.nan [ (a, 0.) ] );
          ( "an infinite time",
            fun (s, a, _) ->
              Session.log_metrics s ~step:0 ~time:Float.infinity [ (a, 0.) ] );
          ( "a metric twice",
            fun (s, a, _) -> Session.log_metrics s ~step:0 [ (a, 0.); (a, 1.) ]
          );
          ( "another session's metric",
            fun (s, _, other) -> Session.log_metrics s ~step:0 [ (other, 0.) ]
          );
        ]
        (fun (_, f) ->
          let store = new_store () in
          let s = start store and t = start store in
          Fun.protect ~finally:(fun () ->
              Session.finish s;
              Session.finish t)
          @@ fun () ->
          let a = Session.metric s "a" and other = Session.metric t "o" in
          raises_match Exn.invalid_arg (fun () -> f (s, a, other)));
      test "steps 0 and 2^53 are steps" (fun () ->
          let store = new_store () in
          let s = start store in
          let m = Session.metric s "m" in
          Session.log m ~step:0 0.;
          Session.log m ~step:(1 lsl 53) 1.;
          Session.finish s;
          equal (list int)
            [ 0; 1 lsl 53 ]
            (List.map
               (fun (x : Metric.sample) -> x.step)
               (samples_of (History.series (history store s) "m"))));
      test "a summary keeps each key's latest value, sorted by key" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.set_summary s [ ("note", `String "x"); ("acc", `Float 0.5) ];
          Session.set_summary s [ ("acc", `Float 0.9) ];
          Session.finish s;
          equal
            (list (pair string value))
            [ ("acc", `Float 0.9); ("note", `String "x") ]
            (History.summary (history store s)));
      cases "set_summary refuses a bad key or string" ~id:fst
        [
          ("an empty key", [ ("", `Int 1) ]);
          ("a key twice", [ ("k", `Int 1); ("k", `Int 2) ]);
          ("a string that is not UTF-8", [ ("k", `String "\xff") ]);
        ]
        (fun (_, values) ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.invalid_arg (fun () -> Session.set_summary s values));
      (* The suite's slowest test: each of its 10 programs runs 50 times, and
         each run starts and finishes a session, which syncs the log and its
         directory, and copies a file. About 0.5 s on a quiet machine. *)
      stateful ~count:10 ~domains:2
        "writes of one session on two domains read back whole, each once"
        [
          command "start"
            (Gen.unit @-> makes logged_w)
            (fun () -> (ref [], ref [], ref []))
            (fun () ->
              let s = start (new_store ()) in
              ( s,
                Session.metric s "a",
                Session.metric s "b",
                source ~suffix:".png" "png" ));
          command "log"
            (Gen.pair (Gen.int_range 0 9) (Gen.float_range 0. 1.)
            @-> logged_w ^-> returns unit)
            (fun (step, v) (samples, _, _) ->
              samples := !samples @ [ ("a", step, v) ])
            (fun (step, v) (_, a, _, _) -> Session.log a ~step v);
          command "log_metrics"
            (Gen.int_range 0 9 @-> logged_w ^-> returns unit)
            (fun step (samples, _, _) ->
              samples := !samples @ [ ("b", step, 1.) ])
            (fun step (s, _, b, _) -> Session.log_metrics s ~step [ (b, 1.) ]);
          command "set_summary"
            (Gen.float_range 0. 1. @-> logged_w ^-> returns unit)
            (fun v (_, summary, _) -> summary := [ ("k", v) ])
            (fun v (s, _, _, _) -> Session.set_summary s [ ("k", `Float v) ]);
          command "log_file"
            (Gen.int_range 0 9 @-> logged_w ^-> returns unit)
            (fun step (_, _, files) -> files := !files @ [ step ])
            (fun step (s, _, _, src) -> Session.log_file s ~step ~key:"f" src);
          command "read"
            (logged_w
            ^-> returns
                  (pair
                     (pair (list (pair int float_exact)) (list int))
                     (pair (list (pair string value)) int)))
            (fun (samples, summary, files) ->
              let of_key k =
                List.filter_map
                  (fun (k', step, v) -> if k' = k then Some (step, v) else None)
                  !samples
              in
              ( (of_key "a" @ of_key "b", !files),
                (List.map (fun (k, v) -> (k, `Float v)) !summary, 0) ))
            (fun (s, _, _, _) ->
              let h =
                History.read
                  (require_some (Store.run (Session.store s) (Session.id s)))
              in
              let pairs k =
                List.map
                  (fun (x : Metric.sample) -> (x.step, x.value))
                  (samples_of (History.series h k))
              in
              ( ( pairs "a" @ pairs "b",
                  List.map (fun (f : History.file) -> f.step) (History.files h)
                ),
                (History.summary h, History.damaged h) ));
        ];
    ]

let sampling_tests =
  group "sampling"
    [
      test "readings are logged at the session's latest step" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.log (Session.metric s "m") ~step:7 1.;
          let reading, await, _ =
            counting (fun n -> [ ("sys/x", Float.of_int n) ])
          in
          Session.sample s ~every:0.001 [ reading ];
          await 3;
          Session.finish s;
          let h = history store s in
          equal ~msg:"keys" (list string) [ "sys/x" ] (History.readings h);
          let series = History.series h "sys/x" in
          greater ~msg:"several calls" int ~than:2 (Metric.length series);
          equal ~msg:"at the latest step" (list int)
            (List.init (Metric.length series) (fun _ -> 7))
            (List.init (Metric.length series) (Metric.step series)));
      test "readings before any metrics record are logged at step 0" (fun () ->
          let store = new_store () in
          let s = start store in
          Session.sample s ~every:1000. [ (fun () -> [ ("sys/x", 1.) ]) ];
          Session.finish s;
          equal (list int) [ 0 ]
            (List.map
               (fun (x : Metric.sample) -> x.step)
               (samples_of (History.series (history store s) "sys/x"))));
      test "readings after a resume are logged at its step" (fun () ->
          let store = new_store () in
          let run = written_run store (log_text []) in
          let s = Session.resume ~provenance run ~step:4 in
          Session.sample s ~every:1000. [ (fun () -> [ ("sys/x", 1.) ]) ];
          Session.finish s;
          equal (list int) [ 4 ]
            (List.map
               (fun (x : Metric.sample) -> x.step)
               (samples_of (History.series (History.read run) "sys/x"))));
      test "a reading that raises Sys_error gives nothing that time" (fun () ->
          let store = new_store () in
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
      test "a reading that raises stops the sampler" (fun () ->
          let store = new_store () in
          let s = start store in
          let reading, await, calls =
            counting (fun n -> if n > 1 then raise Exit else [])
          in
          Session.sample s ~every:0.001 [ reading ];
          await 2;
          (try Session.finish s with Exit -> ());
          (* What is observable without waiting: no call between the stop and
             finish. A sampler that went on would call again only after its
             period, which finish may beat; showing no later call needs a
             wait. *)
          equal ~msg:"calls" int 2 (Atomic.get calls));
      test "finish raises what stopped a sampler, after the end" (fun () ->
          let store = new_store () in
          let s = start store in
          let reading, await, _ =
            counting (fun n -> if n > 1 then raise Exit else [])
          in
          Session.sample s ~every:0.001 [ reading ];
          await 2;
          raises Exit (fun () -> Session.finish s);
          equal status `Finished (Run.status (reread store s)));
      cases
        "a reading that gives a pair the format refuses stops the sampler with \
         Invalid_argument"
        ~id:fst
        [
          ("a key that is not a key", [ ("../k", 1.) ]);
          ("a key twice", [ ("k", 1.); ("k", 2.) ]);
          ("a metric's key", [ ("m", 1.) ]);
        ]
        (fun (_, pairs) ->
          let store = new_store () in
          let s = start store in
          ignore (Session.metric s "m");
          let reading, await, calls =
            counting (fun n -> if n = 1 then [] else pairs)
          in
          Session.sample s ~every:0.001 [ reading ];
          await 2;
          raises_match Exn.invalid_arg (fun () -> Session.finish s);
          (* No call between the stop and finish; a later one needs a wait to
             show, as in "a reading that raises stops the sampler". *)
          equal ~msg:"calls" int 2 (Atomic.get calls));
      cases "sample refuses a period that is not positive and finite"
        ~id:(Printf.sprintf "%g") [ 0.; -1.; Float.nan; Float.infinity ]
        (fun every ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.invalid_arg (fun () -> Session.sample s ~every []));
      test "sample raises what a reading's first call raises" (fun () ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises Exit (fun () ->
              Session.sample s ~every:1000. [ (fun () -> raise Exit) ]));
      test "finish stops a sampler before its period" (fun () ->
          (* WINDTRAP_TIMEOUT fails a finish that waits for the hour. *)
          let store = new_store () in
          let s = start store in
          Session.sample s ~every:3600. [ (fun () -> []) ];
          Session.finish s;
          equal status `Finished (Run.status (reread store s)));
    ]

let file_tests =
  group "files"
    [
      test "files are copied and named by step" (fun () ->
          let store = new_store () in
          let src = source ~suffix:".png" "png" in
          let s = start store in
          Session.log_file s ~step:1 ~key:"viz/cm" src;
          Session.log_file s ~step:1 ~key:"viz/cm" src;
          Session.finish s;
          let files = History.files (history store s) in
          equal (list string) [ "1.png"; "1-1.png" ]
            (List.map
               (fun (f : History.file) -> Filename.basename f.path)
               files);
          equal string "png" (read_file (List.hd files).path));
      test "logging a file streams it" (fun () ->
          let store = new_store () in
          let size = 4 * 1024 * 1024 in
          let src = source (String.make size 'x') in
          let s = start store in
          let words () = (Gc.quick_stat ()).Gc.major_words in
          let before = words () in
          Session.log_file s ~step:0 ~key:"big" src;
          let allocated =
            (words () -. before) *. Float.of_int (Sys.word_size / 8)
          in
          Session.finish s;
          less ~msg:"bytes allocated on the major heap" float_exact
            ~than:(Float.of_int (size / 8))
            allocated);
      cases "log_file refuses a bad key or step, and a missing file" ~id:fst
        [
          ("an empty key", (0, "", `Present));
          ("a step below 0", (-1, "k", `Present));
          ("a missing file", (0, "k", `Missing));
        ]
        (fun (_, (step, key, file)) ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          match file with
          | `Present ->
              raises_match Exn.invalid_arg (fun () ->
                  Session.log_file s ~step ~key (source "x"))
          | `Missing ->
              raises_match Exn.sys_error (fun () ->
                  Session.log_file s ~step ~key
                    (Filename.concat (temp_dir ()) "none")));
      test "a file whose copy fails leaves no temporary" (fun () ->
          let store = new_store () in
          let src = source (String.make 65536 'x') in
          match limited [ "logfile"; Store.root store; src ] with
          | [ id; raised ] ->
              equal ~msg:"raised" string "Sys_error" raised;
              let dir =
                Filename.concat
                  (Run.dir (require_some (Store.run store id)))
                  "files/big"
              in
              equal (list string) []
                (Array.to_list (try Sys.readdir dir with Sys_error _ -> [||]))
          | l -> fail ("the child printed: " ^ String.concat " | " l));
    ]

let session_artifact_tests =
  group "artifacts"
    [
      test "the same content makes a new version sharing one blob" (fun () ->
          let store = new_store () in
          let src = source "weights" in
          let s = start store in
          let v1 = Session.log_artifact s ~name:"ckpt" src in
          let v2 = Session.log_artifact s ~name:"ckpt" src in
          Session.finish s;
          equal ~msg:"versions" (list int) [ 1; 2 ]
            [ Artifact.version v1; Artifact.version v2 ];
          equal ~msg:"blob" string (Artifact.path v1) (Artifact.path v2));
      cases "log_artifact refuses a bad name, step or metadata" ~id:fst
        [
          ("a name with /", ("a/b", None, []));
          ("a name starting with .", (".x", None, []));
          ("an empty name", ("", None, []));
          ("a step below 0", ("x", Some (-1), []));
          ("an empty metadata key", ("x", None, [ ("", `Int 1) ]));
          ("a metadata key twice", ("x", None, [ ("k", `Int 1); ("k", `Int 2) ]));
          ("a string that is not UTF-8", ("x", None, [ ("k", `String "\xff") ]));
        ]
        (fun (_, (name, step, metadata)) ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Session.log_artifact s ?step ~metadata ~name (source "x")));
      test "a publication the system refuses raises Sys_error" (fun () ->
          let store = new_store () in
          let src = source (String.make 65536 'x') in
          equal (list string) [ "Sys_error" ]
            (limited [ "fill"; Store.root store; src ]));
      test "log_artifact raises Sys_error for a missing path" (fun () ->
          let store = new_store () in
          let s = start store in
          Fun.protect ~finally:(fun () -> Session.finish s) @@ fun () ->
          raises_match Exn.sys_error (fun () ->
              Session.log_artifact s ~name:"x"
                (Filename.concat (temp_dir ()) "none")));
    ]

let session_tests =
  group "Session"
    [
      lifecycle_tests;
      metrics_tests;
      sampling_tests;
      file_tests;
      session_artifact_tests;
    ]

let () =
  exit
    (run "munin"
       [
         store_format_tests;
         value_tests;
         provenance_tests;
         metric_tests;
         run_tests;
         history_tests;
         store_tests;
         session_tests;
       ])
