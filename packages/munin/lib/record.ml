(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type start = {
  format : int;
  experiment : string;
  name : string option;
  parent : string option;
  steps : int option;
  params : (string * Value.t) list;
  provenance : Provenance.t;
}

type status = [ `Finished | `Failed of string | `Killed ]

type t =
  | Start of start
  | Metric of {
      key : string;
      goal : Metric.goal option;
      step_metric : string option;
    }
  | Metrics of { step : int; values : (string * float) list }
  | Reading of { step : int; values : (string * float) list }
  | File of { key : string; step : int; path : string }
  | Summary of (string * Value.t) list
  | Use of { name : string; version : int }
  | Resume of { step : int; provenance : Provenance.t }
  | End of status

(* Encoding *)

let str s b = Json.add_string b s
let int n b = Json.add_int b n
let opt name f = function None -> [] | Some v -> [ (name, f v) ]
let assoc f kvs b = Json.add_object b (List.map (fun (k, v) -> (k, f v)) kvs)

(* Provenance and a failure's text come from the system, which may hand bytes
   that are not UTF-8; they are written escaped, never refused, so the run stays
   readable (Store_format, section values). *)
let add_provenance b (p : Provenance.t) =
  let str s b = Json.add_string b (File.utf8 s) in
  let assoc f kvs b =
    Json.add_object b (List.map (fun (k, v) -> (File.utf8 k, f v)) kvs)
  in
  Json.add_object b
    ([
       ( "command",
         fun b ->
           Buffer.add_char b '[';
           List.iteri
             (fun i a ->
               if i > 0 then Buffer.add_char b ',';
               str a b)
             p.command;
           Buffer.add_char b ']' );
       ("cwd", str p.cwd);
     ]
    @ opt "hostname" str p.hostname
    @ [
        ("pid", int p.pid);
        ("ocaml", str p.ocaml);
        ("raven", str p.raven);
        ("os", str p.os);
        ("arch", str p.arch);
      ]
    @ opt "cpu" str p.cpu
    @ opt "git"
        (fun (g : Provenance.git) b ->
          Json.add_object b
            [
              ("commit", str g.commit);
              ("dirty", fun b -> Buffer.add_string b (string_of_bool g.dirty));
            ])
        p.git
    @ [ ("seeds", assoc int p.seeds); ("env", assoc str p.env) ])

let goal_name = function `Minimize -> "minimize" | `Maximize -> "maximize"

let encode ?(buf = Buffer.create 128) ~time r =
  let b = buf in
  Buffer.clear b;
  let kind k = ("type", str k) in
  let t = ("time", fun b -> Json.add_float b time) in
  let floats values = assoc (fun x b -> Json.add_float b x) values in
  let members =
    match r with
    | Start s ->
        [
          kind "start";
          t;
          ("format", int s.format);
          ("experiment", str s.experiment);
        ]
        @ opt "name" str s.name @ opt "parent" str s.parent
        @ opt "steps" int s.steps
        @ [
            ("params", assoc (fun v b -> Json.add_value b v) s.params);
            ("provenance", fun b -> add_provenance b s.provenance);
          ]
    | Metric m ->
        [ kind "metric"; t; ("key", str m.key) ]
        @ opt "goal" (fun g -> str (goal_name g)) m.goal
        @ opt "step_metric" str m.step_metric
    | Metrics m ->
        [ kind "metrics"; t; ("step", int m.step); ("values", floats m.values) ]
    | Reading m ->
        [ kind "reading"; t; ("step", int m.step); ("values", floats m.values) ]
    | File f ->
        [
          kind "file";
          t;
          ("key", str f.key);
          ("step", int f.step);
          ("path", str f.path);
        ]
    | Summary values ->
        [
          kind "summary";
          t;
          ("values", assoc (fun v b -> Json.add_value b v) values);
        ]
    | Use u ->
        [ kind "use"; t; ("name", str u.name); ("version", int u.version) ]
    | Resume r ->
        [
          kind "resume";
          t;
          ("step", int r.step);
          ("provenance", fun b -> add_provenance b r.provenance);
        ]
    | End `Finished -> [ kind "end"; t; ("status", str "finished") ]
    | End `Killed -> [ kind "end"; t; ("status", str "killed") ]
    | End (`Failed e) ->
        let e = File.utf8 e in
        [ kind "end"; t; ("status", str "failed"); ("error", str e) ]
  in
  Json.add_object b members;
  Buffer.add_char b '\n';
  Buffer.contents b

(* Decoding *)

let ( let* ) = Option.bind

let provenance j : Provenance.t option =
  let* command = Json.field "command" (Json.list Json.string) j in
  let* cwd = Json.field "cwd" Json.string j in
  let* hostname = Json.opt "hostname" Json.string j in
  let* pid = Json.field "pid" Json.int j in
  let* ocaml = Json.field "ocaml" Json.string j in
  let* raven = Json.field "raven" Json.string j in
  let* os = Json.field "os" Json.string j in
  let* arch = Json.field "arch" Json.string j in
  let* cpu = Json.opt "cpu" Json.string j in
  let* git =
    Json.opt "git"
      (fun g ->
        let* commit = Json.field "commit" Json.string g in
        let* dirty =
          match Json.mem "dirty" g with
          | Some (Json.Bool b) -> Some b
          | _ -> None
        in
        Some { Provenance.commit; dirty })
      j
  in
  let* seeds = Json.field "seeds" (Json.assoc Json.int) j in
  let* env = Json.field "env" (Json.assoc Json.string) j in
  Some
    {
      Provenance.command;
      cwd;
      hostname;
      pid;
      ocaml;
      raven;
      os;
      arch;
      cpu;
      git;
      seeds;
      env;
    }

let step_of j =
  let* s = Json.field "step" Json.int j in
  if s >= 0 && s <= 1 lsl 53 then Some s else None

let goal_of = function
  | Json.String "minimize" -> Some `Minimize
  | Json.String "maximize" -> Some `Maximize
  | _ -> None

(* The keys of a record's values are checked by the fold, once per key the run
   has not named before, rather than on every line here. *)
let floats_of j = Json.field "values" (Json.assoc Json.float) j

let key_of j =
  let* k = Json.field "key" Json.string j in
  if File.is_key k then Some k else None

let decode_json j =
  let* kind = Json.field "type" Json.string j in
  let* time = Json.field "time" Json.float j in
  let* r =
    match kind with
    | "start" ->
        let* format = Json.field "format" Json.int j in
        let* experiment = Json.field "experiment" Json.string j in
        let* name = Json.opt "name" Json.string j in
        let* parent = Json.opt "parent" Json.string j in
        let* steps = Json.opt "steps" Json.int j in
        let* params = Json.field "params" Json.values j in
        let* provenance = Json.field "provenance" provenance j in
        Some
          (Start { format; experiment; name; parent; steps; params; provenance })
    | "metric" ->
        let* key = key_of j in
        let* goal = Json.opt "goal" goal_of j in
        let* step_metric = Json.opt "step_metric" Json.string j in
        Some (Metric { key; goal; step_metric })
    | "metrics" ->
        let* step = step_of j in
        let* values = floats_of j in
        Some (Metrics { step; values })
    | "reading" ->
        let* step = step_of j in
        let* values = floats_of j in
        Some (Reading { step; values })
    | "file" ->
        let* key = key_of j in
        let* step = step_of j in
        let* path = Json.field "path" Json.string j in
        Some (File { key; step; path })
    | "summary" ->
        let* values = Json.field "values" Json.values j in
        Some (Summary values)
    | "use" ->
        let* name = Json.field "name" Json.string j in
        let* version = Json.field "version" Json.int j in
        if version >= 1 then Some (Use { name; version }) else None
    | "resume" ->
        let* step = step_of j in
        let* provenance = Json.field "provenance" provenance j in
        Some (Resume { step; provenance })
    | "end" -> (
        match Json.field "status" Json.string j with
        | Some "finished" -> Some (End `Finished)
        | Some "killed" -> Some (End `Killed)
        | Some "failed" ->
            let* e = Json.field "error" Json.string j in
            Some (End (`Failed e))
        | _ -> None)
    | _ -> None
  in
  Some (time, r)

(* Values records as [encode] writes them

   A [metrics] or [reading] record in [encode]'s layout, its members in that
   order with no space, is read here without a JSON value: such records are most
   of a log, and a JSON value costs hundreds of words a sample. A line in any
   other layout raises [Other] and is left to the JSON reader, which decides
   every line this one does not; on the lines it reads, the two agree. *)

exception Other

(* [has s i lit] is whether [lit] is at [i] in [s]. *)
let has s i lit =
  let n = String.length lit in
  if i < 0 || i + n > String.length s then false
  else
    let k = ref 0 in
    while !k < n && String.unsafe_get s (i + !k) = String.unsafe_get lit !k do
      incr k
    done;
    !k = n

let skip s i lit = if has s i lit then i + String.length lit else raise Other

let number_end s i =
  let j = Munin_float_text.Float_read.number_end s i (String.length s) in
  if j < 0 then raise Other;
  j

let nan_text = {|{"float":"NaN"}|}
let inf_text = {|{"float":"Infinity"}|}
let neg_inf_text = {|{"float":"-Infinity"}|}

(* The readers below read at [pos] and move it past what they read. *)

let skip_at s pos lit = pos := skip s !pos lit

(* [float_at s pos] is the float at [pos], as [add_float] writes it. *)
let float_at s pos =
  let i = !pos in
  if has s i nan_text then (
    pos := i + String.length nan_text;
    Float.nan)
  else if has s i inf_text then (
    pos := i + String.length inf_text;
    Float.infinity)
  else if has s i neg_inf_text then (
    pos := i + String.length neg_inf_text;
    Float.neg_infinity)
  else
    let j = number_end s i in
    match Munin_float_text.Float_read.of_substring s i (j - i) with
    | Some x ->
        pos := j;
        x
    | None -> raise Other

(* [step_at s pos] is the step at [pos], an integer from [0] to 2^53. *)
let step_at s pos =
  let i = !pos in
  let j = number_end s i in
  if j - i > 16 then raise Other;
  let n = ref 0 in
  for k = i to j - 1 do
    (* A minus, a point or an exponent is the JSON reader's to judge. *)
    if s.[k] < '0' || s.[k] > '9' then raise Other;
    n := (10 * !n) + Char.code s.[k] - 48
  done;
  if !n > 1 lsl 53 then raise Other;
  pos := j;
  !n

(* [key_at s pos] is the key whose quote is at [pos], with no escape. *)
let key_at s pos =
  let i = skip s !pos "\"" in
  let j = ref i in
  while
    !j < String.length s
    && s.[!j] <> '"'
    && s.[!j] <> '\\'
    && Char.code s.[!j] >= 0x20
  do
    incr j
  done;
  if !j >= String.length s || s.[!j] <> '"' then raise Other;
  pos := !j + 1;
  String.sub s i (!j - i)

let rec mem_key k = function
  | [] -> false
  | (k', _) :: rest -> String.equal k k' || mem_key k rest

let values_at s pos =
  skip_at s pos "{";
  if has s !pos "}" then (
    incr pos;
    [])
  else
    let rec members acc =
      let k = key_at s pos in
      (* A repeated name is the JSON reader's to refuse. *)
      if mem_key k acc then raise Other;
      skip_at s pos ":";
      let acc = (k, float_at s pos) :: acc in
      if has s !pos "," then (
        incr pos;
        members acc)
      else (
        skip_at s pos "}";
        List.rev acc)
    in
    members []

let values_record line =
  let reading =
    if has line 0 {|{"type":"metrics","time":|} then false
    else if has line 0 {|{"type":"reading","time":|} then true
    else raise Other
  in
  if not (Json.utf8 line) then raise Other;
  let pos = ref (String.length {|{"type":"metrics","time":|}) in
  let time = float_at line pos in
  skip_at line pos {|,"step":|};
  let step = step_at line pos in
  skip_at line pos {|,"values":|};
  let values = values_at line pos in
  skip_at line pos "}";
  if !pos <> String.length line then raise Other;
  (time, if reading then Reading { step; values } else Metrics { step; values })

let decode line =
  match values_record line with
  | r -> Some r
  | exception Other -> (
      match Json.of_string line with Error _ -> None | Ok j -> decode_json j)

let start_format line =
  match Json.of_string line with
  | Ok j when Json.mem "type" j = Some (Json.String "start") ->
      Json.field "format" Json.int j
  | _ -> None
