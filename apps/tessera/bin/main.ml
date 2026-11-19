(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* main.exe COMMAND [OPTION...]: TESSERA v2 on raven.

   [preprocess] makes a tile of a region's Sentinel-2 and Sentinel-1
   observations from Microsoft's Planetary Computer; [encode] embeds a tile's
   pixels with a model; [corpus] makes a corpus of tiles' pixels, [pretrain]
   trains a model on one by Barlow Twins with mixup, and [targets] and [distil]
   train a student from a teacher's embeddings; [probe] scores a model's
   embeddings of a tile on labelled fields. [main.exe COMMAND --help] lists a
   command's options. *)

open Tessera

let fail fmt = Printf.ksprintf failwith fmt

(* [Arg] over the command's arguments, those after its name. *)
let parse usage specs =
  let usage = "main.exe " ^ usage in
  let anon a = raise (Arg.Bad ("unexpected argument " ^ a)) in
  try Arg.parse_argv ~current:(ref 1) Sys.argv (Arg.align specs) anon usage with
  | Arg.Help m ->
      print_string m;
      exit 0
  | Arg.Bad m ->
      prerr_string m;
      exit 2

let required name = function "" -> fail "%s is required" name | v -> v

let rec mkdir_p d =
  if not (Sys.file_exists d) then begin
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755
  end

(* Devices *)

type 'd target = { at : 'd Nx.Placement.t; jit : bool }

let device_doc =
  "D the devices, as Rig_gpus.open_all names them: cpu (the default), a GPU \
   (cuda, nv, amd or metal; cuda:1 the second), or a comma list"

let eager_doc =
  " compute eagerly, not compiled by Rune.jit: on the host, whose set has \
   kernels"

(* The set of the devices [names] names. *)
let devices names =
  Nx.devices (Result.error_to_failure (Rig_gpus.open_all names))

(* [m]'s forward pass at [t], compiled where [t] says, in bfloat16 with
   [bf16]. *)
let forward (type d) (t : d target) ~bf16 m : d Batch.t -> d Nx.float32_t =
  let run (type s) (m : ((float, s, Nx.host) Nx.t, Nx.host) Model.t) =
    let m = Nx.Ptree.place (module Model) t.at m in
    if not t.jit then Model.apply m
    else
      let model = Nx.Ptree.instantiate (module Model) in
      Rune.jit
        Nx.Ptree.(model @-> v Batch.walk @-> returns tensor)
        Model.apply m
  in
  if bf16 then run (Nx.Ptree.cast (module Model) Nx.bfloat16 m) else run m

(* Models *)

(* The released models on the Hugging Face Hub, at the revisions read. *)
let releases =
  let v2 r = "geotessera/TESSERA-V-2.0-2B-" ^ r in
  [
    ( "nano",
      (v2 "N", "bd08c90bf56ca6a1e7609a395cd47115fa12f1d8", "student_nano.pt") );
    ( "small",
      (v2 "S", "9e1a725f1ac1a3a8ab93fd85b08633754cacb508", "student_small.pt")
    );
    ( "medium",
      (v2 "M", "8c80b926f67cb0b44c2cd41d0226e61741503b9a", "student_medium.pt")
    );
    ( "large",
      (v2 "L", "24e186649bdd544e7c770f277444f3f88c71571c", "student_large.pt")
    );
    ( "teacher",
      ( v2 "Teacher",
        "2cd386f7dbda5db0489c868b0ba862f53c0e013b",
        "tessera_v2_2B_teacher.pt" ) );
  ]

let model_doc =
  "M a release, nano, small, medium, large or teacher, downloaded once; or a \
   checkpoint file, as a distil run's OUT/student.safetensors"

let model name =
  match List.assoc_opt name releases with
  | Some (repo, revision, file) ->
      Model.load (Kaun_hf.download_file ~revision ~file:("ckpt/" ^ file) repo)
  | None -> Model.load name

(* [dims], or all of [m]'s values if [0], refused unless a prefix of [m]. *)
let prefix m dims =
  if dims = 0 then Model.dims m
  else if List.mem dims m.Model.prefixes then dims
  else
    fail "--dims %d: the model's prefixes are [%s]" dims
      (String.concat "; " (List.map string_of_int m.prefixes))

(* The grid of TESSERA's tile of the landmask [file] (Tile.of_landmask), refused
   unless it has the tile's [h × w] cells and, if [place] states the tile's
   grid, lies on it. *)
let landmask (h, w) ?place file =
  let g = (Tile.load_landmask ~source:"mpc" file).grid in
  let text (g : Jord.Grid.t) =
    Printf.sprintf "%d × %d cells from (%.17g, %.17g)" g.rows g.cols
      (fst g.origin) (snd g.origin)
  in
  if g.rows <> h || g.cols <> w then
    fail "%s: a tile of %d × %d cells, and the tile is %d × %d" file g.rows
      g.cols h w;
  (match place with
  | Some (p : Tile.place) when not (Jord.Grid.equal p.grid g) ->
      fail "%s: a tile on %s, and the tile lies on %s" file (text g)
        (text p.grid)
  | _ -> ());
  g

let read_json file =
  In_channel.with_open_bin file In_channel.input_all
  |> Jsont_bytesrw.decode_string Jsont.json
  |> Result.map_error (fun e -> file ^ ": " ^ e)
  |> Result.error_to_failure

(* encode *)

let encode () =
  let file = ref "" and tile = ref "" and out = ref "" and device = ref "cpu" in
  let eager = ref false in
  let dims = ref 0 and int8 = ref false and geotiff = ref "" in
  let bf16 = ref false and pixels = ref 4096 in
  parse "encode --model M --tile DIR --out NAME [OPTION...]"
    [
      ("--model", Arg.Set_string file, model_doc);
      ("--tile", Arg.Set_string tile, "DIR the tile");
      ("--out", Arg.Set_string out, "NAME write NAME.npy, or NAME.tiff");
      ("--device", Arg.Set_string device, device_doc);
      ("--eager", Arg.Set eager, eager_doc);
      ("--dims", Arg.Set_int dims, "K write the Matryoshka prefix of K values");
      ("--int8", Arg.Set int8, " write GeoTessera's int8 codes and scales");
      ( "--landmask",
        Arg.Set_string geotiff,
        "F write NAME.tiff, GeoTessera's float32 GeoTIFF, on the grid of \
         TESSERA's tile of the landmask F" );
      ("--bf16", Arg.Set bf16, " compute the products in bfloat16");
      ("--pixels", Arg.Set_int pixels, "N pixels a batch (default 4096)");
    ];
  if !int8 && !geotiff <> "" then
    fail "--int8 and --landmask: GeoTessera's GeoTIFFs hold float32";
  let out = required "--out" !out in
  let m = model (required "--model" !file) in
  let dims = prefix m !dims in
  let t = Tile.load (required "--tile" !tile) in
  let px, (h, w) = Pixels.of_tile t in
  let grid =
    if !geotiff = "" then None
    else Some (landmask (h, w) ?place:t.place !geotiff)
  in
  let shape = Batch.typical ~pixels:!pixels ~slots:32 in
  let (module D) = devices !device in
  let t = { at = D.on; jit = not !eager } in
  let e = Embedding.encode ~shape t.at (forward t ~bf16:!bf16 m) px in
  let e = Nx.slice [ A; A; R (0, dims) ] (Nx.reshape [| h; w; -1 |] e) in
  match grid with
  | Some grid -> Embedding.save_geotiff out ~grid e
  | None when !int8 -> Embedding.save_int8 out e
  | None -> Embedding.save out e

(* corpus *)

let corpus () =
  let tiles = ref "" and out = ref "" and seed = ref 0 and least = ref 0 in
  parse "corpus --tiles DIR[,DIR...] --out DIR [--seed N] [--min N]"
    [
      ("--tiles", Arg.Set_string tiles, "DIRS the tiles, comma-separated");
      ("--out", Arg.Set_string out, "DIR the corpus");
      ("--seed", Arg.Set_int seed, "N the shuffle's key (default 0)");
      ( "--min",
        Arg.Set_int least,
        "N keep the pixels of N observations of each sensor or more, as \
         pretraining's views need one of each (default: of any observation)" );
    ];
  let tiles = String.split_on_char ',' (required "--tiles" !tiles) in
  let out = required "--out" !out in
  (* Every tile is read before [out] is made, so that one that cannot be read
     leaves it as it was; each is read again for its part, so that no two tiles'
     arrays are held at once. *)
  List.iter (fun tile -> ignore (Tile.load tile : Tile.t)) tiles;
  mkdir_p out;
  let part i tile =
    let px, _ = Pixels.of_tile (Tile.load tile) in
    let l2 = Nx.to_array (Nx_ragged.lengths px.s2)
    and l1 = Nx.to_array (Nx_ragged.lengths px.s1) in
    let keep p =
      if !least = 0 then l2.(p) > 0L || l1.(p) > 0L
      else
        let m = Int64.of_int !least in
        l2.(p) >= m && l1.(p) >= m
    in
    let kept = List.filter keep (List.init (Pixels.length px) Fun.id) in
    let indices =
      Nx.create Nx.int64
        [| List.length kept |]
        (Array.of_list (List.map Int64.of_int kept))
    in
    let dir = Filename.concat out (Printf.sprintf "part%d" i) in
    Pixels.save dir (Pixels.take ~indices px);
    Printf.printf "%s: %d pixels, %d kept\n%!" tile (Pixels.length px)
      (List.length kept);
    dir
  in
  let parts = List.mapi part tiles in
  let n = Pixels.shuffle ~key:(Nx.Rng.key !seed) ~bytes:(1 lsl 30) out parts in
  let remove d =
    Array.iter (fun f -> Sys.remove (Filename.concat d f)) (Sys.readdir d);
    Sys.rmdir d
  in
  List.iter remove parts;
  Printf.printf "%s: %d pixels, shuffled\n" out n

(* targets *)

let targets () =
  let teacher = ref "" and corpus = ref "" and out = ref "" in
  let device = ref "cpu" and eager = ref false in
  let bf16 = ref false and pixels = ref 512 in
  parse "targets --teacher M --corpus DIR --out FILE [OPTION...]"
    [
      ("--teacher", Arg.Set_string teacher, model_doc);
      ("--corpus", Arg.Set_string corpus, "DIR the corpus");
      ("--out", Arg.Set_string out, "FILE the targets, float32 [pixels; dims]");
      ("--device", Arg.Set_string device, device_doc);
      ("--eager", Arg.Set eager, eager_doc);
      ("--bf16", Arg.Set bf16, " compute the products in bfloat16");
      ("--pixels", Arg.Set_int pixels, "N pixels a batch (default 512)");
    ];
  let out = required "--out" !out in
  let px = Pixels.load (required "--corpus" !corpus) in
  let m = model (required "--teacher" !teacher) in
  let shape = Batch.typical ~pixels:!pixels ~slots:32 in
  let n = Pixels.length px and chunk = 8192 in
  let (module D) = devices !device in
  let t = { at = D.on; jit = not !eager } in
  let f = forward t ~bf16:!bf16 m in
  let block pos =
    let len = min chunk (n - pos) in
    let e = Embedding.encode ~shape t.at f (Pixels.sub px ~pos ~len) in
    Printf.printf "targets: %d / %d pixels\n%!" (pos + len) n;
    e
  in
  Seq.init ((n + chunk - 1) / chunk) (fun i -> i * chunk)
  |> Seq.map block
  |> Nx_io.save_npy_concat out Nx.float32 [| Model.dims m |]

(* Training runs *)

(* A run of key [k] draws from [Nx.Rng.fold_in k site] at each site: the
   corpus's order after the first epoch, a step's draws (dropout, and
   pretraining's views), the heads or the projector, and the model. *)
let epoch_site = 0
let draws_site = 1
let head_site = 2
let init_site = 3

let arch ~teacher = function
  | "nano" -> Model.nano
  | "small" -> Model.small
  | "medium" -> Model.medium
  | "large" -> Model.large
  | "teacher" when teacher -> Model.teacher
  | a when teacher -> fail "--arch %S: nano, small, medium, large or teacher" a
  | a -> fail "--arch %S: nano, small, medium or large" a

(* The corpus index of each stream position of [[p, p + len)]. *)
let stream ~key ~n p len =
  Pixels.stream ~key:(Nx.Rng.fold_in key epoch_site) ~n ~first:p len

let state_file out = Filename.concat out "state.safetensors"

let check_out ~out ~resume =
  let file = state_file out in
  if resume && not (Sys.file_exists file) then
    fail "%s: no state to resume" file;
  if (not resume) && Sys.file_exists file then
    fail "%s holds a run: --resume it, or write to another --out" out

(* The run's log, holding the lines a resumed run keeps: those of the steps up
   to its state's. *)
let open_log file ~resume ~upto =
  let kept =
    if not (resume && Sys.file_exists file) then []
    else
      In_channel.with_open_text file In_channel.input_lines
      |> List.filter (fun l ->
          int_of_string (List.hd (String.split_on_char '\t' l)) <= upto)
  in
  let log = open_out file in
  List.iter (fun l -> output_string log (l ^ "\n")) kept;
  log

(* The steps of a run in [out] to [steps], from [start] or, with [resume], from
   the state there. Each step takes its groups, the positions they took and the
   pixels they hold ([groups position]) from the stream position the one before
   stopped at, and writes a line of OUT/log.tsv: its number, first position,
   positions taken, pixels, [step]'s values, its loss first, and seconds. Every
   100 steps and at the last, the state, its key and position go to
   OUT/state.safetensors and its [model] to [model_file]. *)
let train (type d) (t : d target) ~out ~resume ~steps ~key ~model_file
    (module S : Nx.Ptree.S) (module G : Nx.Ptree.S)
    ~(start : (d Nx.float32_t, d) S.t) ~counter ~model ~groups step =
  let s = Nx.Ptree.instantiate (module S) in
  let st, position =
    if not resume then (start, 0)
    else
      let st, k, position =
        Checkpoint.load (state_file out) (module S) ~like:start
      in
      if Nx.to_array (Nx.Rng.to_tensor k) <> Nx.to_array (Nx.Rng.to_tensor key)
      then fail "%s: a run of another --seed" (state_file out);
      (st, position)
  in
  let step =
    if not t.jit then step
    else
      Rune.jit
        Nx.Ptree.(pair (instantiate (module G)) s @-> returns (pair tensor s))
        step
  in
  (* A compiled step keys its variants on its arguments' layouts, and a state
     read from a file lies as the file's records do: it is stored afresh, as the
     step stores its results, by a compiled copy that a set without kernels runs
     too (a gap of rune's: compiled variants keyed on layouts). *)
  let st = if t.jit then Checkpoint.fresh s st else st in
  let first = Int32.to_int (Nx.item [] (Nx.place Nx.Host.on (counter st))) in
  mkdir_p out;
  let log = open_log (Filename.concat out "log.tsv") ~resume ~upto:first in
  let rec go st position i =
    if i < steps then begin
      let t0 = Unix.gettimeofday () in
      let g, taken, pixels = groups position in
      (* The compiled step writes the new state over the old one. *)
      let st = if t.jit then Nx.Ptree.donate s st else st in
      let values, st = step (g, st) in
      let values = Nx.to_array (Nx.place Nx.Host.on values) in
      let dt = Unix.gettimeofday () -. t0 in
      Printf.fprintf log "%d\t%d\t%d\t%d\t%s\t%.4f\n%!" (i + 1) position taken
        pixels
        (String.concat "\t"
           (Array.to_list (Array.map (Printf.sprintf "%.9g") values)))
        dt;
      if (i + 1) mod 10 = 0 || i < 3 then
        Printf.printf "step %d position %d pixels %d loss %.6f %.3f s\n%!"
          (i + 1) position pixels values.(0) dt;
      let position = position + taken in
      if (i + 1) mod 100 = 0 || i + 1 = steps then begin
        Checkpoint.save (state_file out) (module S) st ~key ~position;
        Model.save model_file (model st)
      end;
      go st position (i + 1)
    end
  in
  go st position first;
  close_out log

let run_options ~steps ~seed ~device ~eager ~bf16 ~resume =
  [
    ("--device", Arg.Set_string device, device_doc);
    ("--eager", Arg.Set eager, eager_doc);
    ("--steps", Arg.Set_int steps, "N steps (default 1000)");
    ("--seed", Arg.Set_int seed, "N the run's key (default 0)");
    ("--bf16", Arg.Set bf16, " compute the products in bfloat16");
    ("--resume", Arg.Set resume, " continue the run in OUT to --steps");
  ]

(* pretrain *)

let pretrain () =
  let r = Barlow.v1_recipe in
  let corpus = ref "" and out = ref "" and arch_ = ref "" and widths = ref "" in
  let device = ref "cpu" and eager = ref false in
  let steps = ref 1000 and groups = ref 1 in
  let pixels = ref r.pixels and seed = ref 0 and bf16 = ref false in
  let resume = ref false in
  parse "pretrain --corpus DIR --arch A --out DIR [OPTION...]"
    ([
       ("--corpus", Arg.Set_string corpus, "DIR the corpus (corpus --min 1)");
       ( "--arch",
         Arg.Set_string arch_,
         "A the model: nano, small, medium, large or teacher" );
       ("--out", Arg.Set_string out, "DIR the run's state, log and model");
       ( "--projector",
         Arg.Set_string widths,
         "W,... the projector's widths after the model's (default six of \
          16384, v1.0's)" );
       ("--groups", Arg.Set_int groups, "G groups a step (default 1)");
       ("--pixels", Arg.Set_int pixels, "N pixels a group (default 2048)");
     ]
    @ run_options ~steps ~seed ~device ~eager ~bf16 ~resume);
  let dir = required "--corpus" !corpus and out = required "--out" !out in
  let arch = arch ~teacher:true (required "--arch" !arch_) in
  let widths =
    if !widths = "" then r.projector
    else
      match List.map int_of_string_opt (String.split_on_char ',' !widths) with
      | ws when List.for_all (function Some w -> w > 0 | _ -> false) ws ->
          List.map Option.get ws
      | _ -> fail "--projector %S: widths of 1 or more" !widths
  in
  let resume = !resume and steps = !steps and groups = !groups in
  let pixels = !pixels in
  check_out ~out ~resume;
  let px = Pixels.load dir in
  let n = Pixels.length px in
  if n = 0 then fail "%s: a corpus of no pixel" dir;
  let key = Nx.Rng.key !seed in
  let m = Model.init ~key:(Nx.Rng.fold_in key init_site) arch in
  let projector =
    Barlow.projector ~key:(Nx.Rng.fold_in key head_site) (Model.dims m :: widths)
  in
  let (module D) = devices !device in
  let t = { at = D.on; jit = not !eager } in
  let lr = Barlow.schedule ~lr:r.lr ~warmup:r.warmup ~steps in
  let weight_decay = Vega.rate r.weight_decay in
  let w = Barlow.v1_weights and dropout = r.dropout and clip = r.clip in
  let step g =
    let terms, st =
      if !bf16 then
        Barlow.step ~compute:Nx.bfloat16 w ~dropout ~lr ~weight_decay
          ~clip g
      else
        Barlow.step ~compute:Nx.float32 w ~dropout ~lr ~weight_decay
          ~clip g
    in
    ( Nx.stack
        [
          Barlow.objective w terms;
          terms.on_diagonal;
          terms.off_diagonal;
          terms.mixup;
        ],
      st )
  in
  (* A step's groups from stream position [p], read as one window, each
     group's views drawn by its positions. *)
  let groups_at p =
    let window = groups * pixels in
    let views =
      Pixels.window dir px ~first:p (stream ~key ~n p window)
    in
    let group g =
      Barlow.group
        ~key:(Nx.Rng.fold_in key draws_site)
        ~length:r.length
        ~first:(p + (g * pixels))
        (Pixels.sub views ~pos:(g * pixels) ~len:pixels)
    in
    let gs = Barlow.stack (List.init groups group) in
    (Nx.Ptree.place (module Barlow.Group) t.at gs, window, window)
  in
  train t ~out ~resume ~steps ~key
    ~model_file:(Filename.concat out "model.safetensors")
    (module Barlow.State)
    (module Barlow.Group)
    ~start:
      (Nx.Ptree.place
         (module Barlow.State)
         t.at (Barlow.init m projector))
    ~counter:(fun (st : (D.d Nx.float32_t, D.d) Barlow.state) ->
      st.opt.step)
    ~model:(fun st -> st.model)
    ~groups:groups_at step

(* distil *)

(* The most pixels a distillation step computes on at once. Its loss is a sum
   over pixels, so a step cuts its pixels into groups and sums their gradients
   (Distil.step), and holds what one group's backward reads: about 28 MB a
   pixel for the medium student in float32 on a 2024 Sentinel tile, where one
   group of 2048 pixels asked 58 GB of a 32 GB GPU (2026-11-19). Groups of 256
   also ran a step faster than groups of 512 or 1024 there. *)
let group_pixels = 256

let distil () =
  let corpus = ref ""
  and targets = ref ""
  and out = ref ""
  and arch_ = ref "" in
  let device = ref "cpu" and eager = ref false and steps = ref 1000 in
  let pixels = ref 2048 and seed = ref 0 and bf16 = ref false in
  let resume = ref false in
  parse "distil --corpus DIR --targets FILE --arch A --out DIR [OPTION...]"
    ([
       ("--corpus", Arg.Set_string corpus, "DIR the corpus");
       ("--targets", Arg.Set_string targets, "FILE the teacher's targets of it");
       ( "--arch",
         Arg.Set_string arch_,
         "A the student: nano, small, medium or large" );
       ("--out", Arg.Set_string out, "DIR the run's state, log and student");
       ( "--pixels",
         Arg.Set_int pixels,
         Printf.sprintf "N pixels a step, %d a group at most (default 2048)"
           group_pixels );
     ]
    @ run_options ~steps ~seed ~device ~eager ~bf16 ~resume);
  let dir = required "--corpus" !corpus in
  let targets_file = required "--targets" !targets in
  let out = required "--out" !out in
  let arch = arch ~teacher:false (required "--arch" !arch_) in
  if !pixels < 1 then fail "--pixels %d: 1 or more" !pixels;
  let resume = !resume and steps = !steps in
  (* The fewest groups of equal size that hold the step's pixels, which hold
     fewer than one more a group where they do not divide them. *)
  let groups = (!pixels + group_pixels - 1) / group_pixels in
  check_out ~out ~resume;
  let px = Pixels.load dir in
  let n = Pixels.length px in
  if n = 0 then fail "%s: a corpus of no pixel" dir;
  let (Nx.P t as targets) = Nx_io.load_npy targets_file in
  let d =
    match Nx.shape t with
    | [| rows; d |] when rows = n -> d
    | _ -> fail "%s: targets of another shape than [%d; d]" targets_file n
  in
  let key = Nx.Rng.key !seed in
  let m = Model.init ~key:(Nx.Rng.fold_in key init_site) arch in
  let heads =
    Distil.heads ~key:(Nx.Rng.fold_in key head_site) ~prefixes:m.prefixes d
  in
  let shape =
    Distil.shape ~pixels:((!pixels + groups - 1) / groups) ~slots:32 px
  in
  let r = Distil.v2_recipe in
  let (module D) = devices !device in
  let t = { at = D.on; jit = not !eager } in
  let lr = Barlow.schedule ~lr:r.lr ~warmup:r.warmup ~steps in
  let weight_decay = Vega.rate r.weight_decay in
  let dropout = Model.v2_dropout and clip = r.clip in
  let step g =
    let loss, st =
      if !bf16 then
        Distil.step ~compute:Nx.bfloat16 ~dropout ~lr ~weight_decay
          ~clip g
      else
        Distil.step ~compute:Nx.float32 ~dropout ~lr ~weight_decay ~clip
          g
    in
    (Nx.reshape [| 1 |] loss, st)
  in
  (* A step's groups from stream position [p], read as one window. *)
  let groups_at p =
    let idx = stream ~key ~n p (groups * shape.pixels) in
    let g, taken =
      Distil.groups
        ~key:(Nx.Rng.fold_in key draws_site)
        ~shape ~groups ~first:p
        (Pixels.window dir px ~first:p idx)
        (Distil.targets targets_file targets ~first:p idx)
    in
    let filled =
      Array.fold_left
        (fun a p -> if p >= 0L then a + 1 else a)
        0
        (Nx.to_array g.batch.pixel)
    in
    (Nx.Ptree.place (module Distil.Group) t.at g, taken, filled)
  in
  train t ~out ~resume ~steps ~key
    ~model_file:(Filename.concat out "student.safetensors")
    (module Distil.State)
    (module Distil.Group)
    ~start:
      (Nx.Ptree.place (module Distil.State) t.at (Distil.init m heads))
    ~counter:(fun (st : (D.d Nx.float32_t, D.d) Distil.state) ->
      st.opt.step)
    ~model:(fun st -> st.model)
    ~groups:groups_at step

(* probe *)

(* The owner of each cell of [grid]: the feature whose region holds the cell's
   centre, or -1, as v1.0 rasterises its probe's fields
   (github.com/ucam-eo/tessera-util, commit 290feaf,
   src/utils/convert_shp_to_tiff.py, lines 42-49, rasterio's default). Fields
   that tile the plane hold each centre once, so the owner does not depend on
   their order. *)
let owners (grid : Jord.Grid.t) polys =
  let feature, cell = Jord.Rasterize.centres grid polys in
  let none = Nx.full Nx.int64 [| grid.rows * grid.cols |] (-1L) in
  Nx.to_array (Nx.scatter ~axis:0 cell feature none)

(* Each feature's label, its [property] as text. *)
let labels_of ~property file props =
  Array.mapi
    (fun i (p : Jsont.json) ->
      let value =
        match p with
        | Object (ms, _) -> Option.map snd (Jsont.Json.find_mem property ms)
        | _ -> None
      in
      match value with
      | Some (String (s, _)) -> s
      | Some (Number (x, _)) -> Printf.sprintf "%g" x
      | _ -> fail "%s: feature %d has no %S" file i property)
    props

let probe () =
  let file = ref "" and tile = ref "" and embedding = ref "" in
  let landmask_file = ref "" and labels = ref "" and property = ref "class" in
  let dims = ref 0 and device = ref "cpu" and eager = ref false in
  let pixels = ref 4096 in
  parse
    "probe (--model M --tile DIR | --embedding NAME) --landmask F --labels F \
     [OPTION...]"
    [
      ("--model", Arg.Set_string file, model_doc);
      ("--tile", Arg.Set_string tile, "DIR the tile the model embeds");
      ( "--embedding",
        Arg.Set_string embedding,
        "NAME the tile's embedding in NAME.npy, float32 or GeoTessera's int8 \
         with NAME_scales.npy, in place of a model and a tile" );
      ("--landmask", Arg.Set_string landmask_file, "F the tile's landmask");
      ("--labels", Arg.Set_string labels, "F the fields, GeoJSON");
      ( "--property",
        Arg.Set_string property,
        "P the fields' class (default class)" );
      ("--dims", Arg.Set_int dims, "K probe the Matryoshka prefix of K values");
      ("--device", Arg.Set_string device, device_doc);
      ("--eager", Arg.Set eager, eager_doc);
      ("--pixels", Arg.Set_int pixels, "N pixels a batch (default 4096)");
    ];
  let landmask_file = required "--landmask" !landmask_file in
  let labels = required "--labels" !labels in
  let e, (grid : Jord.Grid.t) =
    if !embedding <> "" then (
      let e = Embedding.load !embedding in
      let h = Nx.dim 0 e and w = Nx.dim 1 e and d = Nx.dim 2 e in
      if !dims < 0 || !dims > d then
        fail "--dims %d: the embedding has %d values" !dims d;
      let d' = if !dims = 0 then d else !dims in
      ( Nx.slice [ A; R (0, d') ] (Nx.reshape [| h * w; d |] e),
        landmask (h, w) landmask_file ))
    else
      let m = model (required "--model" !file) in
      let dims = prefix m !dims in
      let t = Tile.load (required "--tile" !tile) in
      let px, hw = Pixels.of_tile t in
      let grid = landmask hw ?place:t.place landmask_file in
      let shape = Batch.typical ~pixels:!pixels ~slots:32 in
      let (module D) = devices !device in
      let t = { at = D.on; jit = not !eager } in
      let e = Embedding.encode ~shape t.at (forward t ~bf16:false m) px in
      (Nx.slice [ A; R (0, dims) ] e, grid)
  in
  let polys, props =
    Result.error_to_failure (Jord_geojson.polygons (read_json labels))
  in
  let names = labels_of ~property:!property labels props in
  let owner = owners grid (Jord.Polygons.to_crs grid.crs polys) in
  (* A pixel with no observation embeds as NaN, and is left out. *)
  let finite = Nx.to_array (Nx.all ~axes:[ 1 ] (Nx.logical_not (Nx.isnan e))) in
  let labelled =
    List.filter
      (fun i -> owner.(i) >= 0L)
      (List.init (grid.rows * grid.cols) Fun.id)
  in
  let cells, empty = List.partition (fun i -> finite.(i)) labelled in
  if cells = [] then
    fail "%s: no field holds the centre of a pixel with an observation" labels;
  let field i = Int64.to_int owner.(i) in
  let classes =
    List.sort_uniq compare (List.map (fun i -> names.(field i)) cells)
  in
  let id name =
    let rec go k = function
      | x :: _ when x = name -> k
      | _ :: r -> go (k + 1) r
      | [] -> assert false
    in
    go 0 classes
  in
  let area = Array.make (Array.length names) 0 in
  List.iter (fun i -> area.(field i) <- area.(field i) + 1) cells;
  (* v1.0's probe: its split's share and its inverse penalty. *)
  let share = 0.3 and c = 0.1 in
  let training = Eval.split ~share ~classes:names ~areas:area in
  List.iter
    (fun cl ->
      let fields =
        List.filter
          (fun f -> area.(f) > 0 && names.(f) = cl)
          (List.init (Array.length names) Fun.id)
      in
      let taken = List.filter (fun f -> training.(f)) fields in
      let pixels l = List.fold_left (fun a f -> a + area.(f)) 0 l in
      Printf.printf
        "probe: class %s: %d fields, %d pixels, %d training fields of %d pixels\n"
        cl (List.length fields) (pixels fields) (List.length taken)
        (pixels taken))
    classes;
  let train, test = List.partition (fun i -> training.(field i)) cells in
  if train = [] || test = [] then
    fail "v1.0's split of these fields leaves no %s pixel"
      (if train = [] then "training" else "test");
  let rows l =
    Nx.create Nx.int64
      [| List.length l |]
      (Array.of_list (List.map Int64.of_int l))
  in
  let y l = rows (List.map (fun i -> id names.(field i)) l) in
  let x l = Nx.take ~axis:0 (rows l) e in
  Printf.printf
    "probe: %d classes, %d training pixels, %d test pixels, %d labelled pixels \
     without an observation left out, c %g\n"
    (List.length classes) (List.length train) (List.length test)
    (List.length empty) c;
  let probe = Eval.fit ~c (x train) (y train) in
  let s = Eval.score ~predicted:(Eval.predict probe (x test)) (y test) in
  Printf.printf "probe: accuracy %.6f weighted F1 %.6f\n" s.accuracy s.f1;
  Printf.printf "rankme: %.9f over the labelled pixels\n"
    (Eval.rankme (x cells))

(* preprocess *)

let rec rm_rf path =
  if Sys.file_exists path then
    if Sys.is_directory path then begin
      Array.iter (fun n -> rm_rf (Filename.concat path n)) (Sys.readdir path);
      Sys.rmdir path
    end
    else Sys.remove path

let json_text json =
  Result.error_to_failure (Jsont_bytesrw.encode_string Jsont.json json)

(* [file] holding [json], whole or not at all: written beside it, then renamed
   over it. *)
let write_json file json =
  let part = file ^ ".part" in
  Out_channel.with_open_bin part (fun oc -> output_string oc (json_text json));
  Sys.rename part file

(* The longitude and latitude bounds of [g]'s boundary, 21 points between each
   two corners, as TESSERA's preprocessing densifies them. *)
let lonlat_bbox (g : Jord.Grid.t) =
  let n = 22 in
  let r = Float.of_int g.rows and c = Float.of_int g.cols in
  let edge (r0, c0) (r1, c1) =
    List.init n (fun i ->
        let t = Float.of_int i /. Float.of_int n in
        [| r0 +. (t *. (r1 -. r0)); c0 +. (t *. (c1 -. c0)) |])
  in
  let corners = [ (0., 0.); (0., c); (r, c); (r, 0.); (0., 0.) ] in
  let rec edges = function
    | a :: (b :: _ as l) -> edge a b @ edges l
    | _ -> []
  in
  let p = Nx.create Nx.float64 [| 4 * n; 2 |] (Array.concat (edges corners)) in
  let ll =
    Jord.Crs.transform ~src:g.crs ~dst:Jord.Crs.lonlat (Jord.Grid.world g p)
  in
  let col k = Nx.slice [ A; I k ] ll in
  let lo k = Nx.item [] (Nx.min (col k))
  and hi k = Nx.item [] (Nx.max (col k)) in
  (lo 0, lo 1, hi 0, hi 1)

let date_of_string s =
  let day d =
    match Jord.Time.of_date d with
    | _ -> true
    | exception Invalid_argument _ -> false
  in
  match Scanf.sscanf_opt s "%4u-%2u-%2u%!" (fun y m d -> (y, m, d)) with
  | Some d when day d -> d
  | _ -> fail "%S is not a date YYYY-MM-DD" s

let date_text (y, m, d) = Printf.sprintf "%04d-%02d-%02d" y m d

(* What a run asks: the catalogue and collections, the area and days searched,
   the grid, and the region by its count of cells and the MD5 of where they lie.
   A tile's directory records it, and serves no other, so that a run resumed
   there reads the items of its own request. *)
let request (place : Tile.place) ~bbox (first, last) =
  let module J = Jsont.Json in
  let field n v = J.mem (J.name n) v in
  let pair (a, b) = J.list [ J.number a; J.number b ] in
  let w, s, e, n = bbox in
  let g = place.grid in
  let epsg =
    match Jord.Crs.epsg g.crs with
    | Some code -> code
    | None ->
        fail "the grid's system, %s, has no EPSG code"
          (Format.asprintf "%a" Jord.Crs.pp g.crs)
  in
  let region =
    String.concat ""
      (List.map
         (fun c -> if c = 0 then "0" else "1")
         (Array.to_list (Nx.to_array (Nx.reshape [| -1 |] place.roi))))
  in
  let cells =
    String.fold_left (fun k c -> if c = '1' then k + 1 else k) 0 region
  in
  J.object'
    [
      field "source" (J.string place.source);
      field "collections"
        (J.list [ J.string Stac.sentinel2_l2a; J.string Stac.sentinel1_rtc ]);
      field "bbox" (J.list (List.map J.number [ w; s; e; n ]));
      field "days"
        (J.list [ J.string (date_text first); J.string (date_text last) ]);
      field "grid"
        (J.object'
           [
             field "epsg" (J.int epsg);
             field "rows" (J.int g.rows);
             field "cols" (J.int g.cols);
             field "origin" (pair g.origin);
             field "row_step" (pair g.row_step);
             field "col_step" (pair g.col_step);
           ]);
      field "roi"
        (J.object'
           [
             field "cells" (J.int cells);
             field "md5" (J.string (Digest.to_hex (Digest.string region)));
           ]);
    ]

(* The request [dir] serves: [r], recorded by the first run in it. *)
let serve dir r =
  let file = Filename.concat dir "request.json" in
  if Sys.file_exists file then begin
    let held = read_json file in
    if not (Jsont.Json.equal held r) then
      fail "%s holds the tile of another request:\n  %s\nthis run asks:\n  %s"
        dir (json_text held) (json_text r)
  end
  else if Sys.file_exists dir && Sys.readdir dir <> [||] then
    fail "%s holds files and no request.json: give a new directory" dir
  else begin
    mkdir_p dir;
    write_json file r
  end

(* A collection's items over the span, searched once and kept in [file], so that
   a resumed run reads the same ones. *)
let recorded_items ~file ~collection ~bbox span =
  let json =
    if Sys.file_exists file then read_json file
    else
      let j = Stac.search ~collection ~bbox span in
      write_json file j;
      j
  in
  Result.error_to_failure
    (Result.map_error (fun e -> file ^ ": " ^ e) (Jord_stac.items json))

let preprocess () =
  let landmask_file = ref "" and polygons = ref "" and year = ref 0 in
  let days = ref "" and out = ref "" in
  parse
    "preprocess (--landmask F | --polygons F) (--year Y | --days FIRST,LAST) \
     --out DIR"
    [
      ( "--landmask",
        Arg.Set_string landmask_file,
        "F TESSERA's tile of GeoTessera's landmask F" );
      ( "--polygons",
        Arg.Set_string polygons,
        "F the cells of 10 m covering the GeoJSON F's regions" );
      ("--year", Arg.Set_int year, "Y the days of the year Y");
      ( "--days",
        Arg.Set_string days,
        "FIRST,LAST the days FIRST to LAST, YYYY-MM-DD" );
      ("--out", Arg.Set_string out, "DIR the tile");
    ];
  let out = required "--out" !out in
  let place =
    match (!landmask_file, !polygons) with
    | f, "" when f <> "" -> Tile.load_landmask ~source:"mpc" f
    | "", f when f <> "" ->
        Tile.of_polygons ~source:"mpc"
          (fst (Result.error_to_failure (Jord_geojson.polygons (read_json f))))
    | _ -> fail "one of --landmask and --polygons is required"
  in
  let span =
    match (!year, !days) with
    | y, "" when y > 0 && y <= 9999 -> ((y, 1, 1), (y, 12, 31))
    | 0, d when d <> "" -> (
        match String.split_on_char ',' d with
        | [ a; b ] ->
            let first = date_of_string a and last = date_of_string b in
            if compare first last > 0 then
              fail "--days %s: the last day precedes the first" d;
            (first, last)
        | _ -> fail "--days FIRST,LAST")
    | _ -> fail "one of --year and --days is required"
  in
  let grid = place.grid in
  let roi = Nx.not_equal_s place.roi 0 in
  let bbox = lonlat_bbox grid in
  serve out (request place ~bbox span);
  mkdir_p (Filename.concat out "days");
  let items name collection =
    recorded_items ~file:(Filename.concat out name) ~collection ~bbox span
  in
  let s2 = items "s2_items.json" Stac.sentinel2_l2a in
  let s1 = items "s1_items.json" Stac.sentinel1_rtc in
  let s2_days, s2_left = Sentinel2.days s2 in
  let s1_days, s1_left = Sentinel1.days s1 in
  List.iter
    (fun (it : Jord_stac.item) ->
      Printf.printf "left out, lacking an asset: %s\n%!" it.id)
    (s2_left @ s1_left);
  Printf.printf "%d Sentinel-2 items (%d distinct), %d Sentinel-1 items\n%!"
    (List.length s2)
    (List.fold_left (fun n (_, l) -> n + List.length l) 0 s2_days)
    (List.length s1);
  let session = Stac.session ~now:Unix.gettimeofday in
  (* One grid's centres per system, as the span's items need them. *)
  let systems = ref [] in
  let centres crs =
    match List.find_opt (fun (c, _) -> Jord.Crs.equal c crs) !systems with
    | Some (_, x) -> x
    | None ->
        let x = Jord.Warp.centres ~at:Nx.Host.on crs grid in
        systems := (crs, x) :: !systems;
        x
  in
  let dates =
    List.sort_uniq compare
      (List.map fst s2_days @ List.map (fun ((d, _), _) -> d) s1_days)
  in
  (* A day's shard, made once in its directory: a resumed run reads it. *)
  let shard date =
    let dir = Filename.concat out ("days/" ^ date_text date) in
    if not (Sys.file_exists dir) then begin
      let s2 =
        Option.bind
          (List.assoc_opt date s2_days)
          (Sentinel2.day session ~roi ~centres)
      in
      let s1 o =
        Option.bind
          (List.assoc_opt (date, o) s1_days)
          (Sentinel1.day session ~roi ~centres)
      in
      let ascending = s1 Jord_stac.Ascending and descending = s1 Descending in
      let part = dir ^ ".part" in
      rm_rf part;
      let frame (f : Sentinel2.frame) = (f.bands, f.mask) in
      Tile.save part
        (Tile.shard ~place (grid.rows, grid.cols) date ~s2:(Option.map frame s2)
           ~ascending ~descending);
      Sys.rename part dir;
      let says = function Some _ -> "kept" | None -> "-" in
      Printf.printf "%s s2 %s, ascending %s, descending %s\n%!" (date_text date)
        (says s2) (says ascending) (says descending)
    end;
    Tile.load dir
  in
  let shards =
    match List.map shard dates with
    | [] -> [ Tile.empty ~place (grid.rows, grid.cols) ]
    | l -> l
  in
  Tile.concat ~place:(Some place) out shards;
  let t = Tile.load out in
  Printf.printf "%s: %d Sentinel-2 frames, %d ascending, %d descending\n%!" out
    (Nx.dim 0 t.bands) (Nx.dim 0 t.ascending) (Nx.dim 0 t.descending)

let commands =
  [
    ("preprocess", preprocess);
    ("encode", encode);
    ("corpus", corpus);
    ("pretrain", pretrain);
    ("targets", targets);
    ("distil", distil);
    ("probe", probe);
  ]

(* A failure, such as a file the command cannot read, is printed with the
   command's refusals. *)
let () =
  match Array.to_list Sys.argv with
  | _ :: name :: _ when List.mem_assoc name commands -> (
      try (List.assoc name commands) ()
      with Failure m | Sys_error m ->
        prerr_endline ("tessera: " ^ m);
        exit 2)
  | _ ->
      prerr_endline
        ("usage: main.exe COMMAND [OPTION...], COMMAND one of "
        ^ String.concat ", " (List.map fst commands)
        ^ "; main.exe COMMAND --help lists its options");
      exit 2
