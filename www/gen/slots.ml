(* What a page's slots are filled with: the parts of a page that must
   follow the repository rather than be written by hand.

   - [release]: the newest version in CHANGES.md, as "beta 1".
   - [library-count]: how many core libraries there are, in words.
   - [tabs]: the landing page's code tabs, one per library in
     [Registry.tabs], each the first tested block of its doc/index.md, or
     of the section of a guide the registry names.
   - [excerpt FILE NAME]: the top-level definition NAME in FILE, verbatim
     and highlighted.
   - [libraries]: the landing page's cards, the core by group, then
     contrib.
   - [libraries index]: the docs landing's compact index: a row per
     library, with its guides and examples counted.
   - [posts N] and [posts all]: the latest N posts, or all of them.
   - [index]: the index, A to Z, of the docs and the API.
   - [uses LIB...]: the libraries a project uses, each its glyph and name,
     linked to its docs.
   - [gpt-oss PART]: gpt-oss's figures from docs/performance.md: [goal],
     the goal post's rows; [bars] and [runs], the measured runs as bars
     and as a table; [stat], the landing page's figure; [baselines], the
     goal post's count of baselines, in words; [date], the goal post's
     date for the NVIDIA columns. A figure missing, malformed, given
     twice, or differing between the goal post and the runs fails,
     naming its line. *)

open Html

let ( / ) = Filename.concat

let number_word n =
  let words =
    [| "zero"; "one"; "two"; "three"; "four"; "five"; "six"; "seven"; "eight"; "nine"; "ten"; "eleven";
       "twelve"; "thirteen"; "fourteen"; "fifteen"; "sixteen"; "seventeen"; "eighteen"; "nineteen"; "twenty" |]
  in
  if n < Array.length words then String.capitalize_ascii words.(n) else string_of_int n

let plural n one many = sprintf "%d %s" n (if n = 1 then one else many)

(* Without JS, the panels show one under another, each under its heading. *)
let tabs ~root =
  let tab (name, what, source) =
    let e = Registry.find name in
    let doc = root / "packages" / name / "doc" in
    let code =
      match source with
      | None -> Repo.tested_code (doc / "index.md")
      | Some (file, section) -> Repo.tested_code ~section (doc / file)
    in
    sprintf
      {|<section class="panel" id="tab-%s">
<h3 class="panel-h">%s%s</h3>
<pre class="panel-code" tabindex="0" role="region" aria-label="%s example">%s</pre>
<div class="panel-foot caption"><span class="hide-sm">%s · like %s</span><a href="/docs/%s/" class="u">%s docs →</a></div>
</section>|}
      name (glyph e) name name (Highlight.to_html code) what (esc e.like) name name
  in
  concat_map tab Registry.tabs

let libraries ~root =
  let card (e : Registry.entry) =
    sprintf
      {|<a href="/docs/%s/" class="card lib"><div class="glyph-row"><span class="lib-glyph">%s</span><span class="caption">like %s</span></div><div class="stack" style="gap: 6px; flex: 1"><span class="mono h4">%s</span><span class="small">%s</span></div></a>|}
      e.name (glyph e) (esc e.like) e.name (esc e.about)
  in
  let group (title, entries) =
    sprintf {|
<div class="group-row"><h3 class="label group-title">%s</h3><div class="grid-auto%s">%s</div></div>|}
      title (if List.length entries = 4 then " four" else "") (concat_map card entries)
  in
  let contrib (p : Repo.package) =
    let g = match List.assoc_opt p.name Registry.contrib with Some g -> g | None -> Repo.fail "www/gen/registry.ml: no contrib entry for %s" p.name in
    let href =
      if Sys.file_exists (root / "contrib" / p.name / "doc" / "index.md") then sprintf "/docs/%s/" p.name
      else sprintf "%s/tree/main/contrib/%s" github p.name
    in
    sprintf {|<a href="%s" class="contrib"><span class="glyph" style="color: var(--dim); font-size: 20px" aria-hidden="true">%s</span><span class="mono">%s</span><span class="small">%s</span></a>|}
      href (Option.value g ~default:"") p.name (esc p.synopsis)
  in
  sprintf
    {|<div class="stack groups">%s
<div class="group-row"><div class="stack group-title" style="gap: 6px"><h3 class="label">contrib</h3><p class="caption">Built against every change to the core; released on their own schedule.</p></div><div class="grid-auto">%s</div></div>
</div>|}
    (concat_map group Registry.groups) (concat_map contrib (Repo.contrib root))

(* The docs landing's index of libraries: one row each, grouped, with what
   it is, its Python counterpart and how many guides and examples it has,
   so the right library is found without scrolling. Contrib comes last. *)
let library_index ~root =
  let count dir =
    match (Repo.guides (dir / "doc"), Repo.examples (dir / "examples")) with
    | 0, 0 -> "1 page"
    | 0, x -> plural x "example" "examples"
    | g, 0 -> plural g "guide" "guides"
    | g, x -> plural g "guide" "guides" ^ " · " ^ plural x "example" "examples"
  in
  let row ~href ~glyph ~name ~about ~like ~count =
    sprintf {|
<li><a href="%s" class="librow">%s<span class="mono librow-name">%s</span><span class="librow-about">%s</span><span class="caption librow-like">%s</span><span class="caption librow-count">%s</span></a></li>|}
      href glyph name about like count
  in
  let core (e : Registry.entry) =
    row ~href:(sprintf "/docs/%s/" e.name) ~glyph:(glyph e) ~name:e.name ~about:(esc e.about)
      ~like:("like " ^ esc e.like) ~count:(count (root / "packages" / e.name))
  in
  let contrib (p : Repo.package) =
    let g = match List.assoc_opt p.name Registry.contrib with Some g -> g | None -> Repo.fail "www/gen/registry.ml: no contrib entry for %s" p.name in
    let dir = root / "contrib" / p.name in
    let href = if Sys.file_exists (dir / "doc" / "index.md") then sprintf "/docs/%s/" p.name else sprintf "%s/tree/main/contrib/%s" github p.name in
    row ~href ~glyph:(sprintf {|<span class="glyph" style="color: var(--dim)" aria-hidden="true">%s</span>|} (Option.value g ~default:""))
      ~name:p.name ~about:(esc p.synopsis) ~like:"" ~count:(count dir)
  in
  let group title rows = sprintf {|
<div class="libgroup"><h3 class="label">%s</h3><ul>%s
</ul></div>|} title rows in
  sprintf {|<div class="libindex">%s%s
</div>|}
    (concat_map (fun (title, entries) -> group title (concat_map core entries)) Registry.groups)
    (group "contrib" (concat_map contrib (Repo.contrib root)))

(* [Some n]: the latest n, as the landing page lists them; [None]: all of
   them, as the blog lists them, each title a heading. *)
let posts ~root n =
  let posts = Repo.posts root in
  let blurb (p : Repo.post) = match p.blurb with Some b -> sprintf {|<span class="small" style="max-width: 36rem">%s</span>|} (esc b) | None -> "" in
  match n with
  | Some n ->
      let post (p : Repo.post) =
        sprintf {|
<li><a href="/blog/%s/" class="row meta"><span class="label">%s</span><span class="stack" style="gap: 6px"><span class="h3">%s</span>%s</span></a></li>|}
          p.slug p.date (esc p.title) (blurb p)
      in
      sprintf {|<ul class="rows">%s
</ul>|} (concat_map post (List.filteri (fun i _ -> i < n) posts))
  | None ->
      let post (p : Repo.post) =
        sprintf {|
<li><a href="/blog/%s/" class="row meta" style="--meta-w: 120px; padding: 28px 0"><span class="label" style="line-height: 30px">%s</span><span class="stack" style="gap: 6px"><h2 class="h3">%s</h2>%s</span></a></li>|}
          p.slug p.date (esc p.title) (blurb p)
      in
      sprintf {|<ol reversed class="rows">%s
</ol>|} (concat_map post posts)

(* The index, A to Z, of the docs and the API: a group per letter, each
   entry its name, what it is, and its library. The search page filters
   it; without JS it is the whole list. *)
let index ~root =
  let entries = Docs.index ~root in
  let letter (e : Docs.entry) =
    match Char.lowercase_ascii (if e.name = "" then '#' else e.name.[0]) with 'a' .. 'z' as c -> String.make 1 c | _ -> "#"
  in
  let compare_entries (a : Docs.entry) (b : Docs.entry) =
    compare (String.lowercase_ascii a.name, String.lowercase_ascii a.qualified) (String.lowercase_ascii b.name, String.lowercase_ascii b.qualified)
  in
  let entries = List.sort compare_entries entries in
  (* What a library is for, so that "dataframe join" finds talon's joins. *)
  let topic lib =
    match List.find_opt (fun (r : Registry.entry) -> r.name = lib) (List.concat_map snd Registry.groups) with
    | Some r -> r.like ^ " " ^ r.about
    | None -> ""
  in
  let item (e : Docs.entry) =
    let lib = match e.lib with Some l -> l.name | None -> "" in
    let path, last =
      match String.rindex_opt e.qualified '.' with
      | Some i when e.qualified <> e.name && i + 1 < String.length e.qualified -> (String.sub e.qualified 0 (i + 1), String.sub e.qualified (i + 1) (String.length e.qualified - i - 1))
      | _ -> ("", e.qualified)
    in
    sprintf
      {|<li data-n="%s" data-k="%s"><a href="%s" class="azitem"><span class="azglyph">%s</span><span class="stack" style="gap: 2px; min-width: 0"><span class="mono-ui az-name"><span class="az-path">%s</span>%s</span>%s<span class="caption">%s%s</span></span></a></li>|}
      (esc (String.lowercase_ascii e.name))
      (esc (String.lowercase_ascii (String.concat " " [ e.qualified; e.kind; lib; topic lib; e.about ])))
      (esc e.url)
      (match e.lib with Some l -> l.glyph | None -> {|<span class="glyph" aria-hidden="true"></span>|})
      (esc path) (esc last)
      (if e.about = "" then "" else sprintf {|<span class="small">%s</span>|} (esc e.about))
      (esc e.kind) (if lib = "" then "" else " · " ^ esc lib)
  in
  let groups =
    List.fold_left
      (fun acc e -> match acc with (l, es) :: rest when l = letter e -> (l, e :: es) :: rest | _ -> (letter e, [ e ]) :: acc)
      [] entries
    |> List.rev_map (fun (l, es) -> (l, List.rev es))
    |> List.sort (fun (a, _) (b, _) -> compare (a = "#", a) (b = "#", b))
  in
  let group (l, es) =
    let id = if l = "#" then "az-symbols" else "az-" ^ l in
    sprintf {|
<section class="azgroup" aria-labelledby="%s"><h2 id="%s" class="azletter mono">%s</h2><ul class="azlist">%s</ul></section>|} id id l
      (concat_map item es)
  in
  sprintf {|<div class="azindex" data-index data-count="%d">%s
</div>|} (List.length entries) (concat_map group groups)

let uses ~page libs =
  let lib name =
    match List.find_opt (fun (e : Registry.entry) -> e.name = name) Registry.entries with
    | Some e -> sprintf {|<li><a href="/docs/%s/">%s%s</a></li>|} e.name (glyph e) e.name
    | None -> Repo.fail "%s: uses %s, which the registry doesn't list" page name
  in
  sprintf {|<ul class="uses" aria-label="Raven libraries it uses">%s</ul>|} (concat_map lib libs)

(* gpt-oss's figures, from docs/performance.md: the goal post's table, and
   the table of runs under the measured section's heading. A cell of the
   goal post is "blocked: REASON", "not measured", or clauses "VERDICT on
   WHAT, A against B" joined by "; ", each optionally followed by ", NOTE",
   which the page doesn't show. In the NVIDIA columns each clause's A and B
   must be the runs' cells for that column, baseline and measure, so the
   two tables can't disagree, and a cell whose runs exist gives one clause
   for each measure.
   - WHAT is one word, a measure: decode or prefill.
   - An "ahead" or "behind" clause must agree with its figures. A "level"
     clause is a judgement of the runs' ranges, which the Markdown doesn't
     hold, so its verdict isn't checked against its figures. *)

let perf = "docs/performance.md"
let goal_heading = "The goal post"
let runs_heading = "gpt-oss-20b on an NVIDIA RTX 5000 Ada"
let goal_header = [ "As fast as"; "NVIDIA, CUDA"; "NVIDIA, NV"; "AMD" ]
let runs_header = [ ""; "Attention, head"; "Decode"; "Prefill, 512 tokens"; "First token" ]

(* The sentence of the goal post that dates its NVIDIA columns. *)
let dated = "The NVIDIA columns are from "

(* The goal post's NVIDIA columns, by their raven run; each baseline it
   names, by its run and the storage raven matches it in; each measure, by
   its column among the runs. *)
let raven_cuda = "Raven, CUDA"
let llama = ("llama.cpp, CUDA", "Q8_0")
let vllm = ("vLLM", "bfloat16")
let columns = [ (1, raven_cuda); (2, "Raven, NV") ]
let baselines = [ ("llama.cpp", llama); ("vLLM", vllm) ]
let measures = [ ("decode", 2); ("prefill", 3) ]

type run = { line : int; cells : string list }
type figures = { goal : run list; runs : run list; runs_line : int; date : string }

(* [s] cut at each [sep]. *)
let split sep s =
  let n = String.length sep in
  let rec go acc i j =
    if j + n > String.length s then List.rev (String.sub s i (String.length s - i) :: acc)
    else if String.sub s j n = sep then go (String.sub s i (j - i) :: acc) (j + n) (j + n)
    else go acc i (j + 1)
  in
  go [] 0 0

(* The rows of the first table under [heading], its header checked. *)
let rows ~root ~header heading =
  let blocks = Repo.under ~file:perf (root / perf) heading in
  match Repo.table ~file:perf blocks with
  | (line, cells) :: rows ->
      if cells <> header then Repo.fail "%s:%d: the table's header is not | %s |" perf line (String.concat " | " header);
      (blocks, line, List.map (fun (line, cells) -> { line; cells }) rows)
  | [] -> Repo.fail "%s: an empty table under %s" perf heading

(* The YYYY-MM-DD after [dated] in the goal post's paragraphs. *)
let date blocks =
  let at (line, s) =
    match Repo.find s dated with
    | None -> None
    | Some i ->
        let i = i + String.length dated in
        let digits i n = i + n <= String.length s && String.for_all Repo.is_digit (String.sub s i n) in
        if digits i 4 && digits (i + 5) 2 && digits (i + 8) 2 && s.[i + 4] = '-' && s.[i + 7] = '-' then Some (String.sub s i 10)
        else Repo.fail "%s:%d: no YYYY-MM-DD after %S" perf line dated
  in
  match List.find_map at (Repo.paragraphs blocks) with
  | Some d -> d
  | None -> Repo.fail "%s: no paragraph under %s says %S" perf goal_heading dated

(* A second run of the same system and storage fails at its line. *)
let distinct runs =
  let key r = (List.nth r.cells 0, List.nth r.cells 1) in
  ignore
    (List.fold_left
       (fun seen r ->
         match List.assoc_opt (key r) seen with
         | Some first -> Repo.fail "%s:%d: a second run of %s with %s, after line %d" perf r.line (fst (key r)) (snd (key r)) first
         | None -> (key r, r.line) :: seen)
       [] runs)

let figures ~root =
  let blocks, _, goal = rows ~root ~header:goal_header goal_heading in
  let _, runs_line, runs = rows ~root ~header:runs_header runs_heading in
  distinct runs;
  { goal; runs; runs_line; date = date blocks }

let find_run f (system, head) = List.find_opt (fun r -> List.nth r.cells 0 = system && List.nth r.cells 1 = head) f.runs

let run f (system, head) =
  match find_run f (system, head) with
  | Some r -> r
  | None -> Repo.fail "%s:%d: no run of %s with %s" perf f.runs_line system head

let number r col =
  let s = List.nth r.cells col in
  match float_of_string_opt (String.concat "" (String.split_on_char ',' s)) with
  | Some x -> x
  | None -> Repo.fail "%s:%d: %S is not a number" perf r.line s

type clause = { verdict : string; what : string; ours : string; theirs : string }
type cell = Blocked of string | Unmeasured | Clauses of clause list

let clause ~line s =
  let bad () = Repo.fail "%s:%d: %S is not VERDICT on WHAT, A against B" perf line s in
  match split ", " s with
  | head :: figures :: _ -> (
      match (String.split_on_char ' ' head, split " against " figures) with
      | [ verdict; "on"; what ], [ ours; theirs ] when List.mem verdict [ "ahead"; "level"; "behind" ] ->
          { verdict; what; ours; theirs }
      | _ -> bad ())
  | _ -> bad ()

let blocked = "blocked: "

let cell ~line s =
  if s = "not measured" then Unmeasured
  else if String.starts_with ~prefix:blocked s then Blocked (String.sub s (String.length blocked) (String.length s - String.length blocked))
  else Clauses (List.map (clause ~line) (split "; " s))

(* A cell of an NVIDIA column against the runs: one clause for each
   measure where both runs exist, each clause's figures the runs' cells. *)
let agree f (r : run) col c =
  let name = List.nth r.cells 0 in
  match (List.assoc_opt col columns, List.assoc_opt name baselines, c) with
  | None, _, _ | Some _, None, (Blocked _ | Unmeasured) -> ()
  | Some _, None, Clauses _ -> Repo.fail "%s:%d: no runs for %s" perf r.line name
  | Some system, Some (base, head), (Blocked _ | Unmeasured) -> (
      match (find_run f (system, head), find_run f (base, head)) with
      | Some ours, Some theirs ->
          Repo.fail "%s:%d: no figures for %s in %s, where lines %d and %d hold its runs" perf r.line name (List.nth goal_header col)
            ours.line theirs.line
      | _ -> ())
  | Some system, Some (base, head), Clauses cs ->
      let ours = run f (system, head) and theirs = run f (base, head) in
      let whats = List.sort compare (List.map (fun c -> c.what) cs) in
      if whats <> List.sort compare (List.map fst measures) then
        Repo.fail "%s:%d: %s in %s gives %s, where it must give each of %s once" perf r.line name (List.nth goal_header col)
          (String.concat ", " whats) (String.concat ", " (List.map fst measures));
      List.iter
        (fun c ->
          let m = List.assoc c.what measures in
          let o = List.nth ours.cells m and t = List.nth theirs.cells m in
          if (c.ours, c.theirs) <> (o, t) then
            Repo.fail "%s:%d: %s %s against %s, where lines %d and %d give %s against %s" perf r.line c.what c.ours c.theirs ours.line
              theirs.line o t;
          (* Tokens a second: ahead is more, behind less; level is a
             judgement of the runs' ranges, so either. *)
          let a = number ours m and b = number theirs m in
          if (c.verdict = "ahead" && a <= b) || (c.verdict = "behind" && a >= b) then
            Repo.fail "%s:%d: %s on %s, at %s against %s" perf r.line c.verdict c.what c.ours c.theirs)
        cs

let goal f =
  let td (r : run) col =
    let c = cell ~line:r.line (List.nth r.cells col) in
    agree f r col c;
    let dot, state, text =
      match c with
      | Blocked why -> ("border: 1px dashed var(--dim)", "blocked", why)
      | Unmeasured -> ("border: 1px solid var(--dim)", "not measured", "—")
      | Clauses cs ->
          let text = String.concat "; " (List.map (fun c -> sprintf "%s %s against %s, %s" c.what c.ours c.theirs c.verdict) cs) in
          if List.exists (fun c -> c.verdict = "behind") cs then ("background: var(--text-1)", "behind", text)
          else ("background: var(--raven)", "as fast", text)
    in
    sprintf
      {|<td style="text-align: left"><span class="stack" style="gap: 4px"><span class="state"><span aria-hidden="true" style="width: 10px; height: 10px; border-radius: 50%%; flex: none; %s"></span>%s</span><span style="font-size: 14px">%s</span></span></td>|}
      dot state (esc text)
  in
  let tr (r : run) =
    sprintf {|<tr><th scope="row" class="mono" style="color: var(--text-0); font-size: 14px">%s</th>%s%s%s</tr>|}
      (esc (List.nth r.cells 0)) (td r 1) (td r 2) (td r 3)
  in
  String.concat "\n" (List.map tr f.goal)

(* The bars: a group per measure and baseline, each raven run named for
   its bar and for the label that states its difference. *)
let groups =
  let cuda = ("raven", "raven") and nv = ("raven · NV", "NV") in
  let q8 = snd llama and bf16 = snd vllm in
  let by_llama = ("llama.cpp", llama) and by_vllm = ("vLLM", vllm) in
  [
    ("Decode, Q8_0 attention and head", 2, [ (cuda, (raven_cuda, q8)); (nv, ("Raven, NV", q8)) ], by_llama);
    ("Decode, bfloat16", 2, [ (cuda, (raven_cuda, bf16)) ], by_vllm);
    ("Prefill, 512 tokens, Q8_0", 3, [ (cuda, (raven_cuda, q8)) ], by_llama);
    ("Prefill, 512 tokens, bfloat16", 3, [ (cuda, (raven_cuda, bf16)) ], by_vllm);
  ]

(* Tokens a second [a] against [b]: "equal", or [a / b]'s distance from 1
   in percent: "within 0.1%" where one decimal rounds it to 0, an integer
   where one decimal rounds it to 2 or more, else one decimal. *)
let difference a b =
  let pct = Float.abs ((a /. b) -. 1.) *. 100. in
  let tenths = Float.round (pct *. 10.) /. 10. in
  let way = if a > b then "faster" else "slower" in
  if a = b then "equal"
  else if tenths = 0. then "within 0.1%"
  else if tenths >= 2. then sprintf "%.0f%% %s" pct way
  else sprintf "%.1f%% %s" pct way

let bars f =
  let group (title, m, ours, (base_name, base)) =
    let ours = List.map (fun ((who, short), key) -> (who, short, run f key)) ours and base = run f base in
    let top = List.fold_left (fun acc (_, _, r) -> Float.max acc (number r m)) (number base m) ours in
    let label = String.concat ", " (List.map (fun (_, short, r) -> short ^ " " ^ difference (number r m) (number base m)) ours) in
    let bar colour (who, r) =
      sprintf
        {|<div class="bar"><span class="who">%s</span><span class="track"><span class="fill" style="display: block; width: %.1f%%; background: var(%s)"></span></span><span class="v">%s tok/s</span></div>|}
        (esc who) (number r m /. top *. 100.) colour (esc (List.nth r.cells m))
    in
    let ours = List.map (fun (who, _, r) -> (who, r)) ours in
    let all = String.concat ", " (List.map (fun (who, r) -> sprintf "%s %s tok/s" who (List.nth r.cells m)) (ours @ [ (base_name, base) ])) in
    sprintf
      {|<li aria-label="%s: %s; %s">
<div style="display: flex; justify-content: space-between; gap: 12px"><span style="font-size: 15px; color: var(--text-0)">%s</span><span class="caption" style="color: var(--text-3); white-space: nowrap">%s</span></div>
%s</li>|}
      (esc title) (esc all) (esc label) (esc title) (esc label)
      (String.concat "\n" (List.map (bar "--raven") ours @ [ bar "--ref" (base_name, base) ]))
  in
  String.concat "\n" (List.map group groups)

(* The runs' table: "Raven, CUDA" as "raven · CUDA", raven's decode
   highlighted. *)
let runs f =
  let tr (r : run) =
    match r.cells with
    | [ system; head; decode; prefill; first ] ->
        let raven = String.starts_with ~prefix:"Raven, " system in
        let system = String.concat " · " (split ", " (if raven then "r" ^ String.sub system 1 (String.length system - 1) else system)) in
        sprintf {|<tr><th scope="row">%s</th><td>%s</td><td%s>%s</td><td>%s</td><td>%s</td></tr>|} (esc system) (esc head)
          (if raven then {| class="hi"|} else "")
          (esc decode) (esc prefill) (esc first)
    | _ -> Repo.fail "%s:%d: a run is not five cells" perf r.line
  in
  String.concat "\n" (List.map tr f.runs)

(* The goal post's count of baselines, its rows, in words. *)
let baselines_word f = String.lowercase_ascii (number_word (List.length f.goal))

(* The landing page's figure: raven's decode through CUDA at Q8_0 against
   llama.cpp's, and at bfloat16 against vLLM's, on the runs' machine, and
   the goal post's count of baselines. *)
let stat f =
  let decode key = esc (List.nth (run f key).cells 2) in
  sprintf
    {|<p class="stat-big">%s tok/s</p>
<p class="small">decoding %s through CUDA, jitted, with attention and head in Q8_0 as llama.cpp stores them, against llama.cpp's %s; at the checkpoint's bfloat16, %s against vLLM's %s. The goal post on the road to 1.0: at least as fast as %s baselines.</p>|}
    (decode (raven_cuda, snd llama))
    (esc runs_heading) (decode llama)
    (decode (raven_cuda, snd vllm))
    (decode vllm) (baselines_word f)

let gpt_oss ~root ~page part =
  let f = figures ~root in
  match part with
  | "goal" -> goal f
  | "bars" -> bars f
  | "runs" -> runs f
  | "stat" -> stat f
  | "baselines" -> baselines_word f
  | "date" -> esc f.date
  | _ -> Repo.fail "%s: gpt-oss takes goal, bars, runs, stat, baselines or date" page

let fill ~root ~page name args =
  match (name, args) with
  | "release", [] -> esc (Repo.release root)
  | "index", [] -> index ~root
  | "uses", (_ :: _ as libs) -> uses ~page libs
  | "library-count", [] -> number_word (List.length Registry.entries)
  | "tabs", [] -> tabs ~root
  | "excerpt", [ file; name ] -> Highlight.to_html (Repo.definition (root / file) name)
  | "libraries", [] -> libraries ~root
  | "libraries", [ "index" ] -> library_index ~root
  | "posts", [ "all" ] -> posts ~root None
  | "posts", [ n ] -> (
      match int_of_string_opt n with Some n -> posts ~root (Some n) | None -> Repo.fail "%s: posts takes a number or all" page)
  | "gpt-oss", [ part ] -> gpt_oss ~root ~page part
  | _ -> Repo.fail "%s: no slot %s %s" page name (String.concat " " args)
