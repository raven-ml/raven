(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* main.exe: embeddings of TESSERA tiles by the released v2 models, training,
   and their checks against the upstream reference's outputs.

   encode reads a tile, embeds its pixels and writes GeoTessera's files.
   validate embeds the fixture tiles with each released model and compares the
   result with the reference's, embedding by embedding and, for medium, block by
   block against its layer dumps; with --training or --barlow it compares one
   training objective's gradients with the reference's. corpus, targets, distil
   and pretrain train models; probe scores a model's embeddings on a labelled
   tile. *)

open Tessera

let usage =
  {|usage: main.exe encode --model FILE --tile DIR --out NAME [OPTION...]
       main.exe validate --weights DIR --fixtures DIR [OPTION...]
       main.exe validate --training DIR --tile DIR [--masks | --signs]
                [--device D] [--bf16] [--out DIR]
       main.exe corpus --tiles DIR[,DIR...] --out DIR [--seed N] [--bytes B]
       main.exe targets --teacher FILE --corpus DIR --out FILE [OPTION...]
       main.exe distil --corpus DIR --targets FILE --out DIR
                       (--init FILE [--heads FILE] | --arch A) [OPTION...]
       main.exe pretrain --corpus DIR --out DIR
                       (--init FILE [--projector-weights FILE] | --arch A)
                       [OPTION...]
       main.exe validate --barlow DIR --tile DIR [--masks | --signs]
                [--device D] [--out DIR]
       main.exe probe (--model FILE | --arch A) --tile DIR --landmask FILE
                       --labels FILE [OPTION...]

encode options:
  --device D    cpu (eager, the default), cpu-jit, cuda, nv, amd or metal
  --dims K      write the Matryoshka prefix of K values (students only)
  --int8        write GeoTessera's int8 codes and scales
  --landmask F  write NAME.tiff, GeoTessera's float32 GeoTIFF, on the grid of
                the tile's landmask F, in place of the npy files
  --bf16        compute the linear layers and attention in bfloat16
  --pixels N    pixels per batch (default 4096)
  --slots S     pixels per row at most (default 32)

validate options:
  --device D, --pixels N, --slots S
                as for encode, for the students
  --models M    comma-separated, of nano, small, medium, large, teacher
                (default the four students)
  --bf16        validate the teacher in bfloat16 too

validate --training compares one distillation step's loss and gradients
with gen/grads.py's on a small student, dropout masks fed to it; --masks
writes raven's tokens and masks for it first. --signs writes the signs of
the inputs of the device's ReLUs to DIR/DEVICE-PREC (cuda-f32),
which the oracle multiplies its own by; the run then compares its gradients
on that branch, and judges each sign that differs from float64's apart,
listing them in OUT/flips-FIXTURE-BRANCH.tsv for --out OUT (default the
working directory, made if absent). A comparison only reads the fixture's
directory.

corpus writes the pixels of the tiles with an observation, or with --min N
of each sensor (v1.0 pretrains on pixels of 10), shuffled by --seed, in runs
of --bytes (default 1 GiB).

targets options:
  --device D, --bf16, --pixels N (default 512), --slots S
                as for encode
  --fp16        write float16 targets (default float32)
  --chunk N     pixels a block of the file (default 8192)

distil options:
  --init FILE   a student checkpoint, its heads from --heads FILE (npz of
                PyTorch's k.weight and k.bias) or drawn
  --arch A      a student drawn from --seed: nano, small, medium or large
  --device D    as for encode
  --steps N     (default 1000), --groups G (default 1) per step, --pixels N
                per group (default 2048), --slots S (default 32)
  --seed N      the run's key: corpus order after the first epoch, dropout,
                and a drawn student's and heads' parameters (default 0)
  --lr X (8e-4), --warmup F (0.05 of the steps), --decay X (1e-4, every
                leaf), --clip X (1.0), --dropout X (0.1), --reducer-dropout X
                (0.2): v2 App. J, Table 9
  --bf16        compute the products in bfloat16, parameters float32
  --memory B    refuse a step whose footprint, state and groups pass B bytes
  --every K     write OUT/state.safetensors every K steps (default 100)
  --until N     stop after step N of the --steps the schedule spans
  --resume      continue from OUT/state.safetensors, with the options of
                the run that wrote it (a run without --resume refuses an
                OUT that holds one)
  --heldout C:T report each prefix's mean cosine to the targets T of the
                corpus C after the last step

pretrain options (Barlow Twins with mixup, TESSERA v1.0's objective):
  --init FILE   a model checkpoint, its projector from --projector-weights
                FILE (npz of PyTorch's net.i.weight and net.i.bias) or drawn
  --arch A      a model drawn from --seed: nano, small, medium, large or
                teacher
  --projector W,...
                the projector's widths after the model's (default six of
                16384, v1.0's: a compiled float32 step of the default
                holds about 38 GB, more than one 32 GB device)
  --device D    as for encode
  --steps N (1000), --groups G (1) per step, --pixels B (2048, at least
                2) a group, --length L (40) observations of each sensor a
                view
  --seed N      the run's key: corpus order after the first epoch, views,
                dropout, mixing, and a drawn model's and projector's
                parameters (default 0)
  --lr X (2e-3), --warmup F (0.1), --decay X (0, on linear layers'
                weights: v1.0's code writes 1e-6, and its run applied 0),
                --clip X (2.0), --dropout X (0.1), --reducer-dropout X
                (0.2): v1.0's recipe (Barlow.v1_recipe)
  --bf16, --memory B, --every K, --until N, --resume
                as for distil
  --remat       recompute each encoder layer in the backward pass
  --rankme N    after the last step, print RankMe of the projector's and the
                representation's values of the stream's first N pixels
A group of B pixels is the objective's batch, and every pixel of the corpus
needs an observation of each sensor (corpus --min).

validate --barlow compares one pretraining group's loss and gradients
with gen/grads.py barlow-oracle's on a small teacher or student, dropout
masks fed to it; --masks writes raven's views and masks for it first, and
--signs the branch, as for --training.

probe embeds a tile with a model and labels each pixel by the GeoJSON
feature whose region holds its centre, on the grid of the tile's landmask,
as v1.0's in-training probe rasterises its fields (rasterio's default).
tessera-eval's downstream tasks label every cell a field touches (rasterio's
all_touched); the probe does not, since a cell on a shared edge would then
take the label of whichever field comes last, and a test field's edge could
take a training field's. Within each class (--property, default class) the
smallest fields go to training until they hold --share (0.3) of the class's
pixels, as v1.0 splits its fields, and the others to testing; a pixel with
no observation is left out:
  --model FILE  a model checkpoint, or
  --arch A      the model pretrain --arch A --seed 0 starts from
  --state FILE  the model of a distil or pretrain run's state, of --model's
                or --arch's structure
  --c X         the probe's inverse penalty (default 0.1)
  --dims K      probe the Matryoshka prefix of K values
  --device D, --pixels N, --slots S
                as for encode
  --out PREFIX  write the rows, labels and probe as PREFIX_*.npy

Memory grows linearly with --pixels. At 4096 pixels a batch takes about
7.7 GB for medium and 11.4 GB for large eager on the CPU, and 3.8 GB and
5.1 GB of a CUDA GPU's memory compiled.|}

let die fmt =
  Printf.ksprintf
    (fun m ->
      prerr_endline ("tessera: " ^ m);
      exit 2)
    fmt

(* Arguments *)

(* A command's [--name value] options and [--flag]s, in any order; any other
   argument is refused. *)
let parse ~options ~flags args =
  let rec go opts = function
    | [] -> opts
    | a :: rest when List.mem a flags -> go ((a, "") :: opts) rest
    | a :: v :: rest when List.mem a options -> go ((a, v) :: opts) rest
    | [ a ] when List.mem a options -> die "%s needs a value\n%s" a usage
    | a :: _ -> die "unknown argument %S\n%s" a usage
  in
  go [] args

let opt opts name = List.assoc_opt name opts

let req opts name =
  match opt opts name with
  | Some v -> v
  | None -> die "%s is required\n%s" name usage

let flag opts name = List.mem_assoc name opts

let int_opt opts name default =
  match opt opts name with
  | None -> default
  | Some v -> (
      match int_of_string_opt v with Some n -> n | None -> die "%s: %S" name v)

(* Devices *)

type 'd target = { at : 'd Nx.Placement.t; jit : bool }
type run = { run : 'd. 'd target -> unit }

let gpus =
  [
    ("cuda", fun () -> Rig.open_ (Rig_cuda.gpu 0));
    ("nv", fun () -> Rig.open_ (Rig_nv_nvidia.gpu 0));
    ("amd", fun () -> Rig.open_ (Rig_amd_amdgpu.gpu 0));
    ("metal", fun () -> Rig.open_ (Rig_metal.gpu 0));
  ]

let on device k =
  match device with
  | "cpu" -> k.run { at = Nx.Host.on; jit = false }
  | "cpu-jit" -> k.run { at = Nx.Host.on; jit = true }
  | gpu -> (
      match List.assoc_opt gpu gpus with
      | None -> die "%S is not a device" gpu
      | Some open_ -> (
          match open_ () with
          | Error e -> die "%s: %s" gpu e
          | Ok g ->
              let module D = (val Nx.devices [ g ]) in
              k.run { at = D.on; jit = true }))

(* A model's forward pass at [t], compiled where [t] asks. *)
let forward (type s d) (t : d target) (m : ((float, s, d) Nx.t, d) Model.t) :
    d Batch.t -> d Nx.float32_t =
  if not t.jit then Model.apply m
  else
    let model = Nx.Ptree.instantiate (module Model) in
    let batch = Nx.Ptree.v Batch.walk in
    let f =
      Rune.jit Nx.Ptree.(model @-> batch @-> returns tensor) Model.apply
    in
    f m

let place (type d) (t : d target) m = Nx.Ptree.place (module Model) t.at m
let bf16 m = Nx.Ptree.cast (module Model) Nx.bfloat16 m

(* encode *)

(* The tile's grid, from its landmask's tags, as GeoTessera takes it. A tile of
   another shape is refused here, before it is embedded, as save_geotiff would
   refuse it after. *)
let landmask ~tile:(h, w) file =
  let tags = Nx_io.Tiff.tags (List.hd (Nx_io.load_tiff file)) in
  match Jord.Geotiff.grid tags with
  | Error e -> die "%s: %s" file e
  | Ok (g : Jord.Grid.t) when g.rows <> h || g.cols <> w ->
      die "%s: a grid of %d × %d, and the tile is %d × %d" file g.rows g.cols h
        w
  | Ok g -> g

let encode opts =
  let file = req opts "--model" and tile = req opts "--tile" in
  let out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 4096)
      ~slots:(int_opt opts "--slots" 32)
  in
  let m = Model.load file in
  let dims =
    match opt opts "--dims" with
    | None -> Model.dims m
    | Some k -> (
        match int_of_string_opt k with
        | Some k when List.mem k m.prefixes -> k
        | _ ->
            die "--dims %s: the model's prefixes are [%s]" k
              (String.concat "; " (List.map string_of_int m.prefixes)))
  in
  let px, (h, w) = Pixels.of_tile tile in
  let geotiff = Option.map (landmask ~tile:(h, w)) (opt opts "--landmask") in
  if flag opts "--int8" && Option.is_some geotiff then
    die "--int8 and --landmask: GeoTessera's GeoTIFFs hold float32";
  let embed t =
    if flag opts "--bf16" then
      Embedding.encode ~shape t.at (forward t (place t (bf16 m))) px
    else Embedding.encode ~shape t.at (forward t (place t m)) px
  in
  on device
    {
      run =
        (fun t ->
          let e = embed t in
          let e = Nx.reshape [| h; w; -1 |] e in
          let e = Nx.slice [ Nx.A; Nx.A; Nx.R (0, dims) ] e in
          match geotiff with
          | Some grid -> Embedding.save_geotiff out ~grid e
          | None when flag opts "--int8" -> Embedding.save_int8 out e
          | None -> Embedding.save out e);
    }

(* validate *)

let f64 x = Nx.to_array (Nx.cast Nx.float64 x)
let load_f32 file = Nx.unpack Nx.float32 (Nx_io.load_npy file)
let checks = ref 0
let failed = ref []

let report ~name ~pass text =
  Printf.printf "%-48s %s %s\n%!" name (if pass then "ok  " else "FAIL") text;
  incr checks;
  if not pass then failed := name :: !failed

(* A comparison no bound judges, reported for its numbers. *)
let note ~name text = Printf.printf "%-48s %s %s\n%!" name "info" text

(* Tolerances: twice the reference's spread against itself over devices, batch
   sizes and fused paths (max |Δ| 9.8e-6, 1 − cos 2e-12). *)
let max_abs_tol = 2e-5
let cos_tol = 1e-11
let int8_share_tol = 1e-4

(* Max |Δ| and least cosine over the rows [valid] marks, rows of [dims]. *)
let compare_rows ~dims ~valid a b =
  let worst = ref 0. and least = ref 1. in
  Array.iteri
    (fun p v ->
      if v then begin
        let dot = ref 0. and na = ref 0. and nb = ref 0. in
        for c = 0 to dims - 1 do
          let x = a.((p * dims) + c) and y = b.((p * dims) + c) in
          worst := Float.max !worst (Float.abs (x -. y));
          if Float.is_nan x || Float.is_nan y then worst := Float.nan;
          dot := !dot +. (x *. y);
          na := !na +. (x *. x);
          nb := !nb +. (y *. y)
        done;
        least := Float.min !least (!dot /. Float.sqrt (!na *. !nb))
      end)
    valid;
  (!worst, !least)

(* Compares [e] with [ref_file], judged at [tol] if given. *)
let validate_embedding ~label ~valid ?tol e ref_file =
  let r = f64 (load_f32 ref_file) in
  let dims = Nx.dim 1 e in
  let mine = f64 e in
  let worst, least = compare_rows ~dims ~valid mine r in
  let undefined_nan =
    let ok = ref true in
    Array.iteri
      (fun p v ->
        if not v then
          for c = 0 to dims - 1 do
            if not (Float.is_nan mine.((p * dims) + c)) then ok := false
          done)
      valid;
    !ok
  in
  let text =
    Printf.sprintf "max|d| %.3g  1-cos %.3g  undefined NaN %b" worst
      (1. -. least) undefined_nan
  in
  match tol with
  | None -> note ~name:label text
  | Some (abs_tol, cos_tol) ->
      let pass = worst <= abs_tol && 1. -. least <= cos_tol && undefined_nan in
      report ~name:label ~pass text

let validate_int8 ~label ~valid e ~codes ~scales =
  let c, s = Embedding.quantize e in
  let c = Nx.to_array c and s = f64 s in
  let rc = Nx.to_array (Nx.unpack Nx.int8 (Nx_io.load_npy codes)) in
  let rs = f64 (load_f32 scales) in
  let dims = Nx.dim 1 e in
  let differ = ref 0 and total = ref 0 and worst = ref 0 and scale = ref 0. in
  let undefined_ok = ref true in
  Array.iteri
    (fun p v ->
      let codes = Array.sub c (p * dims) dims in
      if not v then
        undefined_ok :=
          !undefined_ok && Float.is_nan s.(p) && Array.for_all (( = ) 0) codes
      else begin
        scale :=
          Float.max !scale (Float.abs (s.(p) -. rs.(p)) /. Float.abs rs.(p));
        Array.iteri
          (fun k code ->
            let d = abs (code - rc.((p * dims) + k)) in
            incr total;
            if d > 0 then incr differ;
            worst := max !worst d)
          codes
      end)
    valid;
  let share = Float.of_int !differ /. Float.of_int !total in
  let pass = !worst <= 1 && share <= int8_share_tol && !undefined_ok in
  report ~name:label ~pass
    (Printf.sprintf
       "codes differing %d / %d (max %d)  scale rel %.3g  undefined 0, NaN %b"
       !differ !total !worst !scale !undefined_ok)

let student_names = [ "nano"; "small"; "medium"; "large" ]

let weight_file dir = function
  | "teacher" -> Filename.concat dir "tessera_v2_2B_teacher.pt"
  | m -> Filename.concat dir ("student_" ^ m ^ ".pt")

let tiles = [ "small_64"; "nodesc_32" ]

let valid_of fixtures tile =
  let v =
    Nx.unpack Nx.uint8
      (Nx_io.load_npy (Filename.concat fixtures ("emb/" ^ tile ^ "/valid.npy")))
  in
  Array.map (fun x -> x <> 0) (Nx.to_array v)

(* The reference's name for a device's fixtures. *)
let ref_device = function "cpu" | "cpu-jit" -> "cpu" | _ -> "cuda"

(* Blocks *)

(* The position of each dumped pixel, from the dumps' index.json. *)
let dumped_pixels dir =
  let text =
    In_channel.with_open_bin
      (Filename.concat dir "index.json")
      In_channel.input_all
  in
  let find from sub =
    let n = String.length sub in
    let rec go i =
      if i + n > String.length text then die "%s: no %S" dir sub
      else if String.sub text i n = sub then i + n
      else go (i + 1)
    in
    go from
  in
  let number from key =
    let i = find from (Printf.sprintf "\"%s\": " key) in
    let j = ref i in
    while !j < String.length text && text.[!j] >= '0' && text.[!j] <= '9' do
      incr j
    done;
    int_of_string (String.sub text i (!j - i))
  in
  Sys.readdir dir |> Array.to_list
  |> List.filter (fun n -> Sys.is_directory (Filename.concat dir n))
  |> List.sort compare
  |> List.map (fun n ->
      let i = find 0 (Printf.sprintf "\"%s\": {" n) in
      (n, number i "y", number i "x"))

(* Row [i]'s first [k] tokens of a [[rows; width; …]] block, as [k] rows of [c]
   values. *)
let tokens_of block i k =
  let c = Nx.numel block / (Nx.dim 0 block * Nx.dim 1 block) in
  f64 (Nx.reshape [| k; c |] (Nx.slice [ Nx.I i; Nx.R (0, k) ] block))

let max_abs a b =
  let w = ref 0. in
  Array.iteri (fun i x -> w := Float.max !w (Float.abs (x -. b.(i)))) a;
  !w

(* Token values as multisets: the reference's copies against our counts. *)
let same_multiset ~c dump mine counts =
  let tbl = Hashtbl.create 64 in
  let key a i = Array.to_list (Array.sub a (i * c) c) in
  for i = 0 to (Array.length dump / c) - 1 do
    let k = key dump i in
    Hashtbl.replace tbl k (1 + Option.value ~default:0 (Hashtbl.find_opt tbl k))
  done;
  Array.iteri
    (fun i m ->
      let k = key mine i in
      Hashtbl.replace tbl k
        (Option.value ~default:0 (Hashtbl.find_opt tbl k) - m))
    counts;
  Hashtbl.fold (fun _ v ok -> ok && v = 0) tbl true

let input name = name = "s2_in" || name = "s1_in"

(* Block tolerance: relative to the largest magnitude of the reference's block,
   the reference's own float32 spread (1e-5 of the output) with a margin for
   blocks that are not normalised. *)
let block_tol = 1e-4

let validate_blocks (type s d) (t : d target)
    (m : ((float, s, d) Nx.t, d) Model.t) ~fixtures ~device =
  let dir =
    Filename.concat fixtures
      (Printf.sprintf "layers/medium__%s__noflag" (ref_device device))
  in
  let pixels = dumped_pixels dir in
  let n = List.length pixels in
  let px, (_, w) = Pixels.of_tile (Filename.concat fixtures "tiles/small_64") in
  let indices =
    Nx.create Nx.int64 [| n |]
      (Array.of_list
         (List.map (fun (_, y, x) -> Int64.of_int ((y * w) + x)) pixels))
  in
  let sub =
    {
      Pixels.s2 = Nx_ragged.take ~indices px.s2;
      s1 = Nx_ragged.take ~indices px.s1;
    }
  in
  let shape =
    { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 1; width = 256 }
  in
  let b, _ = Batch.pack shape sub (Batch.v2_counts sub) ~from:0 in
  let blocks =
    let bt = Batch.place t.at b in
    if not t.jit then Model.blocks m (Model.inputs m bt)
    else
      let names = ref [] in
      let model = Nx.Ptree.instantiate (module Model) in
      let batch = Nx.Ptree.v Batch.walk in
      let f =
        Rune.jit
          Nx.Ptree.(model @-> batch @-> returns (list tensor))
          (fun m b ->
            let l = Model.blocks m (Model.inputs m b) in
            names := List.map fst l;
            List.map snd l)
      in
      let vs = f m bt in
      List.combine !names vs
  in
  let blocks = List.map (fun (k, v) -> (k, Nx.place Nx.Host.on v)) blocks in
  let tokens (r : Nx.host Batch.rows) i =
    Int32.to_int (Nx.item [ i; 0 ] r.stop)
  in
  let counts (r : Nx.host Batch.rows) i k =
    Array.init k (fun j ->
        Float.to_int (Float.round (Float.exp (Nx.item [ i; j ] r.log_count))))
  in
  List.iter
    (fun (name, v) ->
      let worst = ref 0. and scale = ref 0. and exact = ref true in
      let dumped = ref false in
      List.iteri
        (fun i (pixel, _, _) ->
          let file =
            Filename.concat dir (Filename.concat pixel (name ^ ".npy"))
          in
          if Sys.file_exists file then begin
            dumped := true;
            let dump = f64 (load_f32 file) in
            let sensor = String.sub name 0 2 in
            let per_token =
              (sensor = "s2" || sensor = "s1")
              && not (String.ends_with ~suffix:"pooled" name)
            in
            let mine, dump =
              if per_token then begin
                let r = if sensor = "s2" then b.s2 else b.s1 in
                let k = tokens r i in
                let mine = tokens_of v i k in
                let c = Array.length mine / k in
                if input name then
                  exact :=
                    !exact
                    && same_multiset ~c dump mine (counts r i k)
                    && max_abs mine (Array.sub dump 0 (k * c)) = 0.;
                (mine, Array.sub dump 0 (k * c))
              end
              else (f64 (Nx.slice [ Nx.I i ] v), dump)
            in
            Array.iter (fun x -> scale := Float.max !scale (Float.abs x)) dump;
            worst := Float.max !worst (max_abs mine dump)
          end)
        pixels;
      let tol = block_tol *. Float.max 1. !scale in
      let pass = !worst <= tol && !exact in
      (* The reference dumps every value but the embedding's before its ReLU and
         a fusion's, which only a teacher has. *)
      let undumped =
        Filename.check_suffix name "_embed_hidden"
        || String.starts_with ~prefix:"fusion_" name
      in
      if not !dumped then
        begin if not undumped then die "%s: no dump of %s" dir name
        end
      else
        report
          ~name:(Printf.sprintf "medium/%s/blocks/%s" device name)
          ~pass
          (Printf.sprintf "max|d| %.3g  max|ref| %.3g%s" !worst !scale
             (if input name then
                Printf.sprintf "  tokens and counts exact %b" !exact
              else "")))
    blocks

(* Training *)

let float_opt opts name default =
  match opt opts name with
  | None -> default
  | Some v -> (
      match float_of_string_opt v with
      | Some x -> x
      | None -> die "%s: %S" name v)

let now = Unix.gettimeofday

(* Words the program allocated so far. *)
let words () =
  let s = Gc.quick_stat () in
  s.minor_words +. s.major_words -. s.promoted_words

let host_f32 a name =
  match Nx_io.Archive.find name a with
  | None -> die "%s: no entry" name
  | Some (Nx.P x) -> Nx_io.Archive.float ~shape:(Nx.shape x) Nx.float32 name a

(* A student's heads in PyTorch's layout, [k.weight] [[d; k]] and [k.bias]. *)
let load_heads file prefixes =
  let a = Nx_io.load_npz file in
  List.map
    (fun k ->
      let w = Nx.matrix_transpose (host_f32 a (Printf.sprintf "%d.weight" k)) in
      let b = host_f32 a (Printf.sprintf "%d.bias" k) in
      (k, [ { Kaun.Linear.w; b = Some b } ]))
    prefixes

(* Pixels at the corpus indices [indices]. *)
let take_pixels (px : Nx.host Pixels.t) indices =
  {
    Pixels.s2 = Nx_ragged.take ~indices px.s2;
    s1 = Nx_ragged.take ~indices px.s1;
  }

(* Each place's key: the run's dropout key indexed by the place's stream
   position, the first position at an empty place. *)
let place_keys key ~first (b : Nx.host Batch.t) =
  let p =
    Nx.add_s (Nx.maximum b.pixel (Nx.scalar Nx.int64 0L)) (Int64.of_int first)
  in
  Nx.Rng.fold_in_tensor key p

(* corpus *)

let corpus opts =
  let tiles = String.split_on_char ',' (req opts "--tiles") in
  let out = req opts "--out" in
  let key = Nx.Rng.key (int_opt opts "--seed" 0) in
  let bytes = int_opt opts "--bytes" (1 lsl 30) in
  let least = int_opt opts "--min" 0 in
  if not (Sys.file_exists out) then Sys.mkdir out 0o755;
  let parts =
    List.mapi
      (fun i tile ->
        let px, _ = Pixels.of_tile tile in
        let l2 = Nx.to_array (Nx_ragged.lengths px.s2) in
        let l1 = Nx.to_array (Nx_ragged.lengths px.s1) in
        let keep p =
          if least = 0 then l2.(p) > 0L || l1.(p) > 0L
          else
            let m = Int64.of_int least in
            l2.(p) >= m && l1.(p) >= m
        in
        let kept = List.filter keep (List.init (Pixels.length px) Fun.id) in
        let indices =
          Nx.create Nx.int64
            [| List.length kept |]
            (Array.of_list (List.map Int64.of_int kept))
        in
        let dir = Filename.concat out (Printf.sprintf "part%d" i) in
        Pixels.save dir (take_pixels px indices);
        Printf.printf "%s: %d pixels, %d kept\n%!" tile (Pixels.length px)
          (List.length kept);
        dir)
      tiles
  in
  let n = Pixels.shuffle ~key ~bytes out parts in
  List.iter
    (fun d ->
      Array.iter (fun f -> Sys.remove (Filename.concat d f)) (Sys.readdir d);
      Sys.rmdir d)
    parts;
  Printf.printf "%s: %d pixels, shuffled\n" out n

(* targets *)

let targets opts =
  let teacher = req opts "--teacher" and dir = req opts "--corpus" in
  let out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let chunk = int_opt opts "--chunk" 8192 in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 512)
      ~slots:(int_opt opts "--slots" 32)
  in
  let px = Pixels.load dir in
  let n = Pixels.length px in
  let m = Model.load teacher in
  on device
    {
      run =
        (fun t ->
          let f =
            if flag opts "--bf16" then forward t (place t (bf16 m))
            else forward t (place t m)
          in
          let start = now () in
          let block pos =
            let len = min chunk (n - pos) in
            let e = Embedding.encode ~shape t.at f (Pixels.sub px ~pos ~len) in
            Printf.printf "targets: %d / %d pixels, %.0f px/s\n%!" (pos + len) n
              (Float.of_int (pos + len) /. (now () -. start));
            e
          in
          let blocks =
            Seq.map block
              (Seq.init ((n + chunk - 1) / chunk) (fun i -> i * chunk))
          in
          let d = Model.dims m in
          if flag opts "--fp16" then
            Nx_io.save_npy_concat out Nx.float16 [| d |]
              (Seq.map (Nx.cast Nx.float16) blocks)
          else Nx_io.save_npy_concat out Nx.float32 [| d |] blocks);
    }

(* distil *)

(* A run's keys: the corpus order of each epoch after the first, and
   dropout's. *)
let epoch_site = 0
let dropout_site = 1
let heads_site = 2
let init_site = 3

(* The corpus index of each stream position of [p, p + len): epoch [e]'s
   ordering of [n], the corpus's own for epoch 0. *)
let indices key ~n p len =
  let pos = Nx.arange Nx.int64 p (p + len) 1 in
  let n' = Int64.of_int n in
  let e = Nx.div_s pos n' and i = Nx.mod_s pos n' in
  let shuffled =
    Nx.Rng.permute
      ~key:(Nx.Rng.fold_in_tensor (Nx.Rng.fold_in key epoch_site) e)
      ~n i
  in
  Nx.where (Nx.equal_s e 0L) i shuffled

(* The pixels a step's groups hold: from position [p], each group packed from
   where the previous one stopped, and the positions they took. *)
let step_groups ~key ~shape ~groups corpus targets p =
  let n = Pixels.length corpus in
  let window = groups * shape.Batch.pixels in
  let idx = indices key ~n p window in
  let px = take_pixels corpus idx in
  let px, counts = Batch.expand px (Batch.v2_counts px) in
  let dkey = Nx.Rng.fold_in key dropout_site in
  let rec go g from acc =
    if g = groups then (List.rev acc, from)
    else
      let b, next = Batch.pack shape px counts ~from in
      let places = Nx.maximum b.pixel (Nx.scalar Nx.int64 0L) in
      let keys = place_keys dkey ~first:p b in
      let t = Nx.take ~axis:0 (Nx.take ~axis:0 places idx) targets in
      go (g + 1) next ({ Distil.batch = b; keys; targets = t } :: acc)
  in
  let gs, taken = go 0 0 [] in
  let stack f = Nx.stack (List.map f gs) in
  ( {
      Distil.batch =
        Batch.stack (List.map (fun (g : _ Distil.group) -> g.batch) gs);
      keys =
        Nx.Rng.of_tensor
          (stack (fun (g : _ Distil.group) -> Nx.Rng.to_tensor g.keys));
      targets = stack (fun (g : _ Distil.group) -> g.targets);
    },
    taken )

let tree_bytes s x = Nx.Ptree.fold s (fun _ a acc -> acc + Nx.nbytes a) x 0

(* A run's files under [out]: its state, and its log, which a resumed run keeps
   up to the steps its state holds. A run without [--resume] refuses an [out]
   that holds either. *)
type files = {
  ckpt : string;
  log : out_channel;
  keep : int -> unit;
  resume : bool;
}

let run_files opts out =
  if not (Sys.file_exists out) then Sys.mkdir out 0o755;
  let ckpt = Filename.concat out "state.safetensors" in
  let log_file = Filename.concat out "log.tsv" in
  let resume = flag opts "--resume" in
  if resume && not (Sys.file_exists ckpt) then die "%s: no state to resume" ckpt;
  if (not resume) && (Sys.file_exists ckpt || Sys.file_exists log_file) then
    die "%s holds a run: --resume it, or write to another --out" out;
  let kept =
    if not resume then []
    else
      In_channel.with_open_bin log_file In_channel.input_all
      |> String.split_on_char '\n'
      |> List.filter (fun l -> l <> "")
  in
  let log = open_out log_file in
  let keep upto =
    List.iter
      (fun l ->
        match String.split_on_char '\t' l with
        | s :: _ when int_of_string s <= upto -> output_string log (l ^ "\n")
        | _ -> ())
      kept;
    flush log
  in
  { ckpt; log; keep; resume }

(* The steps of a run from [start], or from the state [files] holds, to step
   [until]: each step's groups from the position the previous one stopped at,
   its log line, and the state every [every] steps and at the last. Compiled,
   its footprint is reported and checked against [--memory]. *)
let train (type d) (t : d target) opts files ~key ~seed ~run ~until ~every
    (module S : Nx.Ptree.S) (module G : Nx.Ptree.S)
    ~(start : (d Nx.float32_t, d) S.t)
    ~(counter : (d Nx.float32_t, d) S.t -> d Nx.int32_t)
    ~(groups : int -> (d Nx.float32_t, d) G.t * int)
    ~(filled : (d Nx.float32_t, d) G.t -> int)
    (step_fn :
      (d Nx.float32_t, d) G.t * (d Nx.float32_t, d) S.t ->
      d Nx.float32_t * (d Nx.float32_t, d) S.t) =
  let st_ptree = Nx.Ptree.instantiate (module S) in
  let group = Nx.Ptree.instantiate (module G) in
  let st, position =
    if files.resume then begin
      let st, k, position, saved =
        Checkpoint.load files.ckpt (module S) ~like:start
      in
      if saved <> run then
        die "%s is a run of\n  %s\nnot of\n  %s" files.ckpt saved run;
      if Nx.to_array (Nx.Rng.to_tensor k) <> Nx.to_array (Nx.Rng.to_tensor key)
      then die "%s: the run's key is not --seed %d's" files.ckpt seed;
      (st, position)
    end
    else (start, 0)
  in
  let signature =
    Nx.Ptree.(pair group st_ptree @-> returns (pair tensor st_ptree))
  in
  let step = if t.jit then Rune.jit signature step_fn else step_fn in
  let first = Int32.to_int (Nx.item [] (Nx.place Nx.Host.on (counter st))) in
  files.keep first;
  let gs0 = groups position in
  if t.jit then begin
    (* Over the donated state, as the step runs: the new state then takes the
       old one's storage. *)
    let fp =
      Rune.footprint signature step_fn (fst gs0, Nx.Ptree.donate st_ptree st)
    in
    let live = tree_bytes st_ptree st + tree_bytes group (fst gs0) in
    List.iter
      (fun (r, b) ->
        Printf.printf
          "footprint: %s %.2f GB, with the state and a step's groups %.2f GB\n\
           %!"
          (Rig.name r)
          (Float.of_int b /. 1e9)
          (Float.of_int (b + live) /. 1e9))
      fp;
    match opt opts "--memory" with
    | None -> ()
    | Some s ->
        let budget = float_of_string s in
        List.iter
          (fun (r, b) ->
            if Rig.name r <> "host" && Float.of_int (b + live) > budget then
              die "%s needs %.2f GB, above --memory %.2f GB" (Rig.name r)
                (Float.of_int (b + live) /. 1e9)
                (budget /. 1e9))
          fp
  end;
  let rec go st position i ~gs =
    if i >= until then st
    else begin
      let t0 = now () and w0 = words () in
      let gs, taken = match gs with Some g -> g | None -> groups position in
      (* A compiled step writes the new state over the old one. *)
      let st = if t.jit then Nx.Ptree.donate st_ptree st else st in
      let loss, st = step (gs, st) in
      let loss = Nx.item [] (Nx.place Nx.Host.on loss) in
      let dt = now () -. t0 and dw = words () -. w0 in
      let filled = filled gs in
      Printf.fprintf files.log "%d\t%d\t%d\t%d\t%.9g\t%.4f\t%.0f\n%!" (i + 1)
        position taken filled loss dt dw;
      if (i + 1) mod 10 = 0 || i < 3 then
        Printf.printf
          "step %d position %d pixels %d loss %.6f %.3f s %.0f words\n%!"
          (i + 1) position filled loss dt dw;
      let position = position + taken in
      if (i + 1) mod every = 0 || i + 1 = until then
        Checkpoint.save files.ckpt (module S) st ~key ~position ~run;
      go st position (i + 1) ~gs:None
    end
  in
  go st position first ~gs:(Some gs0)

(* Mean cosine of each prefix's head to the teacher over [px]. *)
let fidelity (type d) (t : d target) shape
    (st : (d Nx.float32_t, d) Distil.state) px targets =
  let e = Embedding.encode ~shape t.at (forward t st.model) px in
  let host =
    Nx.Ptree.place
      (module Distil.Params)
      Nx.Host.on
      { Distil.net = st.model.net; heads = st.heads }
  in
  let c = Distil.cosines host.heads e ~targets in
  let mean = Nx.to_array (Nx.mean ~axes:[ 0 ] c) in
  List.mapi (fun i (p, _) -> (p, mean.(i))) host.heads

let distil opts =
  let dir = req opts "--corpus" and out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let seed = int_opt opts "--seed" 0 in
  let key = Nx.Rng.key seed in
  let steps = int_opt opts "--steps" 1000 in
  let until = min steps (int_opt opts "--until" steps) in
  let groups = int_opt opts "--groups" 1 in
  let every = int_opt opts "--every" 100 in
  let lr = float_opt opts "--lr" 8e-4 in
  let warmup = float_opt opts "--warmup" 0.05 in
  let decay = float_opt opts "--decay" 1e-4 in
  let clip = float_opt opts "--clip" 1.0 in
  let dropout =
    {
      Model.layers = float_opt opts "--dropout" 0.1;
      reducer = float_opt opts "--reducer-dropout" 0.2;
    }
  in
  let corpus = Pixels.load dir in
  let targets =
    match Nx_io.load_npy (req opts "--targets") with
    | Nx.P x -> Nx.cast Nx.float32 x
  in
  let d_t = Nx.dim 1 targets in
  let m, heads =
    match (opt opts "--init", opt opts "--arch") with
    | Some file, _ ->
        let m = Model.load file in
        let heads =
          match opt opts "--heads" with
          | Some h -> load_heads h m.prefixes
          | None ->
              Distil.heads
                ~key:(Nx.Rng.fold_in key heads_site)
                ~prefixes:m.prefixes d_t
        in
        (m, heads)
    | None, Some a ->
        let arch =
          match a with
          | "nano" -> Model.nano
          | "small" -> Model.small
          | "medium" -> Model.medium
          | "large" -> Model.large
          | a -> die "--arch %S: nano, small, medium or large" a
        in
        let m = Model.init ~key:(Nx.Rng.fold_in key init_site) arch in
        ( m,
          Distil.heads
            ~key:(Nx.Rng.fold_in key heads_site)
            ~prefixes:m.prefixes d_t )
    | None, None -> die "--init or --arch is required\n%s" usage
  in
  (* Rows for the duplicate-filled tokens of the corpus's first pixels. *)
  let pixels = int_opt opts "--pixels" 2048
  and slots = int_opt opts "--slots" 32 in
  let width = 256 in
  let shape =
    let sample =
      Pixels.sub corpus ~pos:0 ~len:(min 4096 (Pixels.length corpus))
    in
    let _, c = Batch.expand sample (Batch.v2_counts sample) in
    let mean x =
      Float.of_int (Nx.numel x) /. Float.of_int (Pixels.length sample)
    in
    let rows tokens =
      max 1
        (Float.to_int
           (Float.ceil
              (1.05 *. tokens *. Float.of_int pixels /. Float.of_int width))
        + 1)
    in
    {
      Batch.pixels;
      s2_rows = rows (mean c.s2);
      s1_rows = rows (mean c.s1);
      slots;
      width;
    }
  in
  let warmup_steps =
    max 1 (Float.to_int (Float.round (warmup *. Float.of_int steps)))
  in
  let sched () =
    Vega.Schedule.warmup_cosine_decay ~init_value:0. ~peak_value:lr
      ~warmup_steps
      ~decay_steps:(max 1 (steps - warmup_steps))
      ()
  in
  let compute_bf16 = flag opts "--bf16" in
  let heldout =
    match opt opts "--heldout" with
    | None -> None
    | Some s -> (
        match String.split_on_char ':' s with
        | [ c; t ] ->
            let px = Pixels.load c in
            let t =
              match Nx_io.load_npy t with Nx.P x -> Nx.cast Nx.float32 x
            in
            Some (px, t)
        | _ -> die "--heldout CORPUS:TARGETS")
  in
  let files = run_files opts out in
  (* The options a resume must repeat: every one that changes the steps. *)
  let run =
    String.concat " "
      ([
         Printf.sprintf "--corpus %s --targets %s" dir (req opts "--targets");
         (match (opt opts "--init", opt opts "--arch") with
         | Some f, _ ->
             Printf.sprintf "--init %s --heads %s" f
               (Option.value ~default:"drawn" (opt opts "--heads"))
         | None, a -> "--arch " ^ Option.value ~default:"" a);
         Printf.sprintf
           "--seed %d --steps %d --groups %d --pixels %d --slots %d --lr %h \
            --warmup %h --decay %h --clip %h --dropout %h --reducer-dropout %h"
           seed steps groups pixels slots lr warmup decay clip dropout.layers
           dropout.reducer;
       ]
      @ if compute_bf16 then [ "--bf16" ] else [])
  in
  Printf.printf
    "distil: %d pixels, shape {pixels %d; s2_rows %d; s1_rows %d; slots %d; \
     width %d}, %d groups, dropout %g/%g, %s\n\
     %!"
    (Pixels.length corpus) shape.pixels shape.s2_rows shape.s1_rows shape.slots
    shape.width groups dropout.layers dropout.reducer
    (if compute_bf16 then "bf16" else "f32");
  on device
    {
      run =
        (fun (type d) (t : d target) ->
          let fresh =
            Nx.Ptree.place
              (module Distil.Params)
              t.at
              { Distil.net = m.net; heads }
          in
          let start =
            Distil.init
              { (Nx.Ptree.place (module Model) t.at m) with net = fresh.net }
              fresh.heads
          in
          let sched = sched () in
          let step_fn
              (g :
                (d Nx.float32_t, d) Distil.group
                * (d Nx.float32_t, d) Distil.state) :
              d Nx.float32_t * (d Nx.float32_t, d) Distil.state =
            if compute_bf16 then
              Distil.step ~compute:Nx.bfloat16 ~dropout ~lr:sched
                ~weight_decay:(Vega.rate decay) ~clip g
            else
              Distil.step ~compute:Nx.float32 ~dropout ~lr:sched
                ~weight_decay:(Vega.rate decay) ~clip g
          in
          let groups position =
            let g, taken =
              step_groups ~key ~shape ~groups corpus targets position
            in
            (Nx.Ptree.place (module Distil.Group) t.at g, taken)
          in
          let filled (g : (d Nx.float32_t, d) Distil.group) =
            Nx.to_array (Nx.place Nx.Host.on g.batch.pixel)
            |> Array.fold_left (fun a p -> if p >= 0L then a + 1 else a) 0
          in
          let st =
            train t opts files ~key ~seed ~run ~until ~every
              (module Distil.State)
              (module Distil.Group)
              ~start
              ~counter:(fun (st : (d Nx.float32_t, d) Distil.state) ->
                st.opt.step)
              ~groups ~filled step_fn
          in
          match heldout with
          | None -> ()
          | Some (px, tg) ->
              let shape = Batch.typical ~pixels:4096 ~slots:32 in
              List.iter
                (fun (k, c) ->
                  Printf.printf "fidelity: prefix %d mean cos %.6f\n" k c)
                (fidelity t shape st px tg));
    };
  close_out files.log

(* pretrain *)

(* A pretraining's key of its groups' draws. *)
let views_site = 1

(* A step's [groups] groups of [pixels] from stream position [p], their views
   drawn by their positions, and the positions they take. *)
let barlow_groups ~key ~length ~pixels ~groups corpus p =
  let n = Pixels.length corpus in
  let gkey = Nx.Rng.fold_in key views_site in
  let group g =
    let first = p + (g * pixels) in
    Barlow.group ~key:gkey ~length ~first
      (take_pixels corpus (indices key ~n first pixels))
  in
  (Barlow.stack (List.init groups group), groups * pixels)

(* A projector in PyTorch's layout ([net.i.weight] and [net.i.bias] of a
   sequence of linear, batch-norm and ReLU modules), from an npz. *)
let load_projector file =
  let a = Nx_io.load_npz file in
  (* The modules with parameters, in order; ReLUs have none. *)
  let modules =
    List.filter_map
      (fun n ->
        match String.split_on_char '.' n with
        | [ "net"; i; "weight" ] -> int_of_string_opt i
        | _ -> None)
      (Nx_io.Archive.names a)
    |> List.sort compare
  in
  let layer i =
    let name f = Printf.sprintf "net.%d.%s" i f in
    let w = host_f32 a (name "weight") and b = host_f32 a (name "bias") in
    if Nx.ndim w = 2 then
      `Linear { Kaun.Linear.w = Nx.matrix_transpose w; b = Some b }
    else `Norm { Kaun.Batch_norm.gamma = w; beta = b }
  in
  let rec build = function
    | [ `Linear out ] -> ([], out)
    | `Linear l :: `Norm n :: rest ->
        let hidden, out = build rest in
        ((l, n) :: hidden, out)
    | _ -> die "%s: not a projector of linear and batch-norm layers" file
  in
  let hidden, out = build (List.map layer modules) in
  { Barlow.hidden; out }

let arch_named = function
  | "nano" -> Model.nano
  | "small" -> Model.small
  | "medium" -> Model.medium
  | "large" -> Model.large
  | "teacher" -> Model.teacher
  | a -> die "--arch %S: nano, small, medium, large or teacher" a

(* RankMe of the projector's and the representation's values of [v]'s first
   views, without dropout. *)
let rankmes (st : (Nx.host Nx.float32_t, Nx.host) Barlow.state)
    (v : Nx.host Barlow.views) =
  let e = Model.apply st.model v.a1 in
  let z = Barlow.project st.projector e in
  (Eval.rankme z, Eval.rankme e)

let pretrain opts =
  let dir = req opts "--corpus" and out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let seed = int_opt opts "--seed" 0 in
  let key = Nx.Rng.key seed in
  let steps = int_opt opts "--steps" 1000 in
  let until = min steps (int_opt opts "--until" steps) in
  let groups = int_opt opts "--groups" 1 in
  let v1 = Barlow.v1_recipe in
  let pixels = int_opt opts "--pixels" v1.pixels in
  if pixels < 2 then die "--pixels %d: a group needs 2 pixels or more" pixels;
  let length = int_opt opts "--length" v1.length in
  let every = int_opt opts "--every" 100 in
  let lr = float_opt opts "--lr" v1.lr in
  let warmup = float_opt opts "--warmup" v1.warmup in
  let decay = float_opt opts "--decay" v1.weight_decay in
  let clip = float_opt opts "--clip" v1.clip in
  let remat = flag opts "--remat" in
  let dropout =
    {
      Model.layers = float_opt opts "--dropout" v1.dropout.layers;
      reducer = float_opt opts "--reducer-dropout" v1.dropout.reducer;
    }
  in
  let corpus = Pixels.load dir in
  let m =
    match (opt opts "--init", opt opts "--arch") with
    | Some file, _ -> Model.load file
    | None, Some a ->
        Model.init ~key:(Nx.Rng.fold_in key init_site) (arch_named a)
    | None, None -> die "--init or --arch is required\n%s" usage
  in
  let projector =
    match opt opts "--projector-weights" with
    | Some file -> load_projector file
    | None ->
        let widths =
          match opt opts "--projector" with
          | None -> v1.projector
          | Some l -> List.map int_of_string (String.split_on_char ',' l)
        in
        Barlow.projector
          ~key:(Nx.Rng.fold_in key heads_site)
          (Model.dims m :: widths)
  in
  let warmup_steps =
    max 1 (Float.to_int (Float.round (warmup *. Float.of_int steps)))
  in
  let sched () =
    Vega.Schedule.warmup_cosine_decay ~init_value:0. ~peak_value:lr
      ~warmup_steps
      ~decay_steps:(max 1 (steps - warmup_steps))
      ()
  in
  let compute_bf16 = flag opts "--bf16" in
  let files = run_files opts out in
  let run =
    String.concat " "
      ([
         Printf.sprintf "--corpus %s" dir;
         (match (opt opts "--init", opt opts "--arch") with
         | Some f, _ ->
             Printf.sprintf "--init %s --projector-weights %s" f
               (Option.value ~default:"drawn" (opt opts "--projector-weights"))
         | None, a -> "--arch " ^ Option.value ~default:"" a);
         Printf.sprintf
           "--projector %s --seed %d --steps %d --groups %d --pixels %d \
            --length %d --lr %h --warmup %h --decay %h --clip %h --dropout %h \
            --reducer-dropout %h"
           (String.concat ","
              (List.map
                 (fun (l, _) -> string_of_int (Nx.dim 1 l.Kaun.Linear.w))
                 projector.hidden
              @ [ string_of_int (Nx.dim 1 projector.out.w) ]))
           seed steps groups pixels length lr warmup decay clip dropout.layers
           dropout.reducer;
       ]
      @ (if compute_bf16 then [ "--bf16" ] else [])
      @ if remat then [ "--remat" ] else [])
  in
  Printf.printf
    "pretrain: %d pixels, %d groups of %d, %d observations a view, projector \
     %s, dropout %g/%g, %s\n\
     %!"
    (Pixels.length corpus) groups pixels length
    (String.concat "→"
       (List.map string_of_int
          (Model.dims m
           :: List.map (fun (l, _) -> Nx.dim 1 l.Kaun.Linear.w) projector.hidden
          @ [ Nx.dim 1 projector.out.w ])))
    dropout.layers dropout.reducer
    (if compute_bf16 then "bf16" else "f32");
  on device
    {
      run =
        (fun (type d) (t : d target) ->
          let fresh =
            Nx.Ptree.place
              (module Barlow.Params)
              t.at
              { Barlow.net = m.net; projector }
          in
          let start =
            Barlow.init
              { (Nx.Ptree.place (module Model) t.at m) with net = fresh.net }
              fresh.projector
          in
          let sched = sched () in
          let step_fn
              (g :
                (d Nx.float32_t, d) Barlow.group
                * (d Nx.float32_t, d) Barlow.state) :
              d Nx.float32_t * (d Nx.float32_t, d) Barlow.state =
            let weight_decay = Vega.rate decay in
            if compute_bf16 then
              Barlow.step ~compute:Nx.bfloat16 Barlow.v1_weights ~dropout ~remat
                ~lr:sched ~weight_decay ~decays:Barlow.decays ~clip g
            else
              Barlow.step ~compute:Nx.float32 Barlow.v1_weights ~dropout ~remat
                ~lr:sched ~weight_decay ~decays:Barlow.decays ~clip g
          in
          let groups_at position =
            let g, taken =
              barlow_groups ~key ~length ~pixels ~groups corpus position
            in
            (Nx.Ptree.place (module Barlow.Group) t.at g, taken)
          in
          let st =
            train t opts files ~key ~seed ~run ~until ~every
              (module Barlow.State)
              (module Barlow.Group)
              ~start
              ~counter:(fun (st : (d Nx.float32_t, d) Barlow.state) ->
                st.opt.step)
              ~groups:groups_at
              ~filled:(fun _ -> groups * pixels)
              step_fn
          in
          (* RankMe of the stream's first pixels as a group, on the host. *)
          let n = int_opt opts "--rankme" 0 in
          if n > 0 then begin
            let host = Nx.Ptree.place (module Barlow.State) Nx.Host.on st in
            let first =
              Barlow.group
                ~key:(Nx.Rng.fold_in key views_site)
                ~length ~first:0
                (take_pixels corpus (indices key ~n:(Pixels.length corpus) 0 n))
            in
            let z, e = rankmes host first in
            Printf.printf "rankme: projector %.6f representation %.6f\n" z e
          end);
    };
  close_out files.log

(* validate --training *)

(* The oracle's distillation: the student [student.pt], its heads [heads.npz],
   the targets and the tile's pixels [pixels.npy] of [gen/grads.py init], at
   v2's dropout, keyed by each pixel's index there. *)
let training_inputs dir tile =
  let file f = Filename.concat dir f in
  let m = Model.load (file "student.pt") in
  let heads = load_heads (file "heads.npz") m.prefixes in
  let targets = Nx.unpack Nx.float32 (Nx_io.load_npy (file "targets.npy")) in
  let px, _ = Pixels.of_tile tile in
  let px =
    take_pixels px (Nx.unpack Nx.int64 (Nx_io.load_npy (file "pixels.npy")))
  in
  let px, counts = Batch.expand px (Batch.v2_counts px) in
  (m, heads, targets, px, counts)

let oracle_key = Nx.Rng.key 0

(* float32's spacing relative to its value, at most. *)
let f32_ulp = Float.ldexp 1. (-23)

(* Writes the tokens and masks of each pixel, one a row, for the oracle. *)
let write_masks dir tile =
  let m, _, _, px, counts = training_inputs dir tile in
  let n = Pixels.length px in
  let shape =
    { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 1; width = 256 }
  in
  let b, _ = Batch.pack shape px counts ~from:0 in
  let keys = place_keys oracle_key ~first:0 b in
  let masks = Model.masks ~dropout:Model.v2_dropout ~keys m b in
  List.iter
    (fun (name, x) ->
      Nx_io.save_npy
        (Filename.concat dir ("mask_" ^ name ^ ".npy"))
        (Nx.cast Nx.uint8 x))
    masks;
  let blocks = Model.blocks m (Model.inputs m b) in
  List.iter
    (fun s ->
      Nx_io.save_npy
        (Filename.concat dir ("raven_" ^ s ^ ".npy"))
        (List.assoc s blocks))
    [ "s2_in"; "s1_in" ];
  let len (r : Nx.host Batch.rows) =
    Nx.cast Nx.int64 (Nx.slice [ Nx.A; Nx.I 0 ] r.stop)
  in
  Nx_io.save_npy
    (Filename.concat dir "raven_lengths.npy")
    (Nx.stack ~axis:1 [ len b.s2; len b.s1 ]);
  Out_channel.with_open_bin (Filename.concat dir "raven_masks.json") (fun oc ->
      Printf.fprintf oc "{\"order\": [%s]}\n"
        (String.concat ", "
           (List.map (fun (k, _) -> Printf.sprintf "%S" k) masks)));
  Printf.printf "%s: %d masks of %d pixels\n" dir (List.length masks) n

let leaves s x =
  List.rev
    (Nx.Ptree.fold s
       (fun p a acc ->
         ( Nx.Ptree.Path.to_string p,
           Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on a)) )
         :: acc)
       x [])

let rel a b =
  let d = ref 0. and n = ref 0. in
  Array.iteri
    (fun i x ->
      d := !d +. ((x -. b.(i)) ** 2.);
      n := !n +. (b.(i) ** 2.))
    a;
  Float.sqrt (!d /. Float.max !n 1e-300)

(* Fed branches. A ReLU's sign is discrete, so the gradients are compared on
   raven's branch: [--signs] writes the signs of each ReLU's input in the
   device's step, the oracle multiplies its own by them, and the values where
   raven's sign differs from float64's are judged apart. A device and
   precision's branch lives in its own directory, [cuda-f32]. *)

let branch_dir dir ~device ~prec = Filename.concat dir (device ^ "-" ^ prec)

(* A ReLU's inputs [z] in the oracle's layout, [[pixels; width; c]] with pixel
   [r]'s tokens from column [0] to [len.(r)], zeros after; a pixel's values,
   [[pixels; c]], are [[pixels; 1; c]]. *)
type preact = {
  name : string;
  dims : int array;
  len : int array;
  z : float array;
}

(* Each ReLU's input a step records, under the name the oracle feeds:
   [s2_layer0] for a feed-forward's [s2_layer0_linear1], [s2_embed] for
   [s2_embed_hidden], [reducer] for [reducer1_ln1] and [projector0] for
   [projector0_bn], after a pretraining view's prefix. Each must be the ReLU's
   input, not its output: the oracle is fed the input's signs, so a ReLU that
   cuts elsewhere than [0] runs another branch than the one fed and shows in the
   gradients, where the output's signs would follow it. *)
let relu_inputs =
  [
    ("_linear1", "");
    ("_embed_hidden", "_embed");
    ("reducer1_ln1", "reducer");
    ("_bn", "");
  ]

(* The loss's inputs, which the loss's checks need: a distillation's [embedding]
   and a pretraining view's [v0_projection]. *)
let loss_input name =
  name = "embedding" || Filename.check_suffix name "_projection"

let recorder acc name v =
  let chop (suffix, by) =
    if Filename.check_suffix name suffix then
      Some (Filename.chop_suffix name suffix ^ by)
    else None
  in
  match List.find_map chop relu_inputs with
  | Some n -> acc := (n, v ()) :: !acc
  | None -> if loss_input name then acc := (name, v ()) :: !acc

(* [z] of a batch of one pixel a row, every token filled, or of pixels. *)
let whole name z =
  let z = Nx.place Nx.Host.on z in
  let dims =
    match Nx.shape z with [| p; c |] -> [| p; 1; c |] | dims -> dims
  in
  { name; dims; len = Array.make dims.(0) dims.(1); z = f64 z }

(* [z] [[rows; width; ffn]] of [r]'s packed rows, by pixel place. *)
let by_place (r : Nx.host Batch.rows) name z =
  let z = Nx.place Nx.Host.on z in
  let width = Nx.dim 1 z and c = Nx.dim 2 z in
  let slots = Nx.dim 1 r.slot_start and places = Nx.dim 0 r.at in
  let a = f64 z and out = Array.make (places * width * c) 0. in
  let len =
    Array.init places (fun k ->
        let s = Int64.to_int (Nx.item [ k ] r.at) in
        let row = s / slots and j = s mod slots in
        let start = Int32.to_int (Nx.item [ row; j ] r.slot_start) in
        let n = Int32.to_int (Nx.item [ row; j ] r.slot_stop) - start in
        Array.blit a (((row * width) + start) * c) out (k * width * c) (n * c);
        n)
  in
  { name; dims = [| places; width; c |]; len; z = out }

let sign_file dir name = Filename.concat dir ("sign_" ^ name ^ ".npy")
let signs p = Array.map (fun z -> if z > 0. then 1 else 0) p.z

let save_signs dir ps =
  (try Sys.mkdir dir 0o755 with Sys_error _ -> ());
  List.iter
    (fun p ->
      Nx_io.save_npy (sign_file dir p.name)
        (Nx.create Nx.uint8 p.dims (signs p)))
    ps;
  Printf.printf "%s: the signs of %d layers' pre-activations\n" dir
    (List.length ps)

(* The run's branch is the one the oracle was fed: a compiled program rounds as
   it did when it wrote them. *)
let check_branch dir ps =
  List.iter
    (fun p ->
      let file = sign_file dir p.name in
      if not (Sys.file_exists file) then
        die "%s: no signs: write them with --signs, then run the oracle" file;
      let fed = Nx.to_array (Nx.unpack Nx.uint8 (Nx_io.load_npy file)) in
      if fed <> signs p then
        die "%s: the run's signs differ from the ones the oracle was fed" file)
    ps

(* Where a comparison of fixture [dir] on [branch] lists its flips: in
   [--out], made here if absent so that a bad path fails before the run, the
   working directory by default; named for the fixture and the branch, so runs
   of others keep theirs. Never the fixture's directory, which a comparison
   only reads. *)
let flips_file opts dir branch =
  let out = Option.value ~default:"." (opt opts "--out") in
  (try if not (Sys.file_exists out) then Sys.mkdir out 0o755
   with Sys_error e -> die "--out: %s" e);
  Filename.concat out
    (Printf.sprintf "flips-%s-%s.tsv" (Filename.basename dir)
       (Filename.basename branch))

(* Each flip of [ps], a value where raven's sign is not float64's on raven's
   branch, by the oracle's [preact_f64_*] in [dir]. A flip passes where the
   float64 pre-activation's magnitude is at most the largest distance of the
   reference's own pre-activation at the run's precision ([preact_<prec>_*])
   from float64 over the layer's tokens: a sign the reference's rounding could
   have given. The flips go to [out]. *)
let check_flips ~label ~prec ~out dir ps =
  let load p pr =
    let file =
      Filename.concat dir (Printf.sprintf "preact_%s_%s.npy" pr p.name)
    in
    let a = Nx.to_array (Nx.unpack Nx.float64 (Nx_io.load_npy file)) in
    if Array.length a <> Array.length p.z then die "%s: of another shape" file;
    a
  in
  let rows = ref [] in
  List.iter
    (fun p ->
      let z64 = load p "f64" and zr = load p prec in
      let w = p.dims.(1) and c = p.dims.(2) in
      let tokens f =
        Array.iteri
          (fun r n ->
            for t = 0 to n - 1 do
              for u = 0 to c - 1 do
                f r t u ((((r * w) + t) * c) + u)
              done
            done)
          p.len
      in
      let ss = ref 0. and n = ref 0 and bound = ref 0. in
      tokens (fun _ _ _ i ->
          let e = Float.abs (zr.(i) -. z64.(i)) in
          ss := !ss +. (e *. e);
          bound := Float.max !bound e;
          incr n);
      let sigma = Float.sqrt (!ss /. Float.of_int (max 1 !n)) in
      let flips = ref 0 and worst = ref 0. in
      tokens (fun r t u i ->
          if p.z.(i) > 0. <> (z64.(i) > 0.) then begin
            let ratio = Float.abs z64.(i) /. !bound in
            incr flips;
            worst := Float.max !worst ratio;
            rows :=
              Printf.sprintf
                "%s\t%d\t%d\t%d\t%.6g\t%.6g\t%.6g\t%.4g\t%.4g\t%.3f" p.name r t
                u z64.(i) p.z.(i) zr.(i) sigma !bound ratio
              :: !rows
          end);
      report
        ~name:(Printf.sprintf "%s/flips/%s" label p.name)
        ~pass:(!worst <= 1.)
        (Printf.sprintf
           "%d flips of %d  torch %s sigma %.3g  bound %.3g  worst |f64|/bound \
            %.3f"
           !flips !n prec sigma !bound !worst))
    ps;
  Out_channel.with_open_text out (fun oc ->
      output_string oc
        "layer\tpixel\ttoken\tunit\tf64\traven\ttorch\tsigma\tbound\t|f64|/bound\n";
      List.iter (fun l -> output_string oc (l ^ "\n")) (List.rev !rows))

(* A leaf whose exact gradient is 0 by the softmax's invariance to a shift of
   its scores (attention's key biases, the pooling's bias) is a sum of terms
   that cancel. The oracle gives each element the sum [T] of its terms'
   magnitudes, in float64 ([grads_zero.pt]), and an element passes where its
   value is at most [u · T], [u] the unit roundoff of the products' dtype. [u ·
   T] is a first-order estimate, not a bound: one rounding of every term and no
   cancellation among the errors, leaving out the inner sums' growth and relying
   on errors cancelling across tokens. Here it lies 6-20 times above the
   measured error, a margin that moves with the batch's size. The result is the
   largest [|g| / (u · T)]. *)
let zero_ratio ~u g t =
  let worst = ref 0. in
  Array.iteri
    (fun i x ->
      let r = if t.(i) > 0. then Float.abs x /. (u *. t.(i)) else Float.abs x in
      worst := Float.max !worst (if x = 0. then 0. else r))
    g;
  !worst

let is_zero_leaf t = Array.exists (fun x -> x > 0.) t

let unit_roundoff = function
  | "bf16" -> Float.ldexp 1. (-8)
  | _ -> Float.ldexp 1. (-24)

(* A leaf of exact gradient 0 by the batch normalisation's or the whitening's
   invariance to a shift, which the oracle gives no terms for: its float64 norm,
   rounding alone, below this share of the whole gradient's (1e-15 to 2e-14 of
   600-1600 in the fixtures). *)
let zero_leaf = 1e-12
let norm a = Float.sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0. a)

(* The run's leaves [mine] against float64's [r64] and the reference's [rref] at
   [prec], [terms] the oracle's exact-zero terms. A leaf with terms passes by
   [zero_ratio]; one of float64 norm below [zero_leaf] of the gradient's by its
   distance against float32's spacing at the gradient's norm; any other within
   twice the reference's distance from float64. *)
let check_leaves ~label ~prec mine ~r64 ~rref ~terms =
  let u = unit_roundoff prec and worst = ref 0. in
  let scale =
    Float.sqrt (List.fold_left (fun s (_, a) -> s +. (norm a ** 2.)) 0. r64)
  in
  let judge name ratio text =
    worst := Float.max !worst ratio;
    report ~name ~pass:(ratio <= 1.) text
  in
  List.iter2
    (fun (path, g) ((path', r64), ((_, rp), (_, tz))) ->
      if path <> path' then die "%s against %s" path path';
      let name = Printf.sprintf "%s/%s" label path in
      if is_zero_leaf tz then
        let ratio = zero_ratio ~u g tz in
        judge name ratio
          (Printf.sprintf "exact 0: |g|/(u T) %.3g  torch %s %.3g  f64 %.3g"
             ratio prec (zero_ratio ~u rp tz) (zero_ratio ~u r64 tz))
      else if norm r64 <= zero_leaf *. scale then
        let dist a b = norm (Array.map2 ( -. ) a b) /. scale in
        let d = dist g r64 in
        judge name (d /. f32_ulp)
          (Printf.sprintf
             "exact 0: distance %.3g of the gradient's norm, torch %s %.3g, \
              bound %.3g"
             d prec (dist rp r64) f32_ulp)
      else
        let mine = rel g r64 and theirs = rel rp r64 in
        let ratio = mine /. (2. *. theirs) in
        judge name ratio
          (Printf.sprintf "rel %.3g  torch %s rel %.3g  ratio %.2f" mine prec
             theirs ratio))
    mine
    (List.combine r64 (List.combine rref terms));
  Printf.printf "worst ratio to its bound: %.3f\n" !worst

(* The loss against float64. A loss is one value, its error its input's error
   projected onto its gradient in a random direction, so one reference run's
   distance from float64 says little about the next. The run's loss must lie
   within [u M] of its first-order prediction from float64, [l64 + ∂L/∂x · (x −
   x₆₄)], [x] the objective's inputs (the embeddings, the projections), [∂L/∂x]
   the float64 loss's gradient there. [M] is the sum over the objective's
   rounded values [y] of [|∂L/∂y|] times the magnitude of [y]'s terms
   ([objective_terms.npy]): like [zero_ratio]'s [u · T], a first-order estimate
   of the objective's own rounding, not a bound, here 2.5-90 times the measured
   residual. The remainder past first order is 1e-11 to 3e-10 here. [x] itself
   must lie within twice the reference's distance from float64, as a leaf must.
   With [~judge:false] the loss is reported, not judged: bfloat16's [u M] is too
   wide for its residual ever to reach it. [files] name each input's files in
   [dir] by kind: ["f64"], the reference's precision and ["grad"]. *)
let check_loss ~label ~prec ~judge ~input dir ~loss ~l64 ~mine ~files =
  let load f = Nx.unpack Nx.float64 (Nx_io.load_npy (Filename.concat dir f)) in
  let inputs kind =
    Array.concat (List.map (fun f -> Nx.to_array (load (f kind))) files)
  in
  let x64 = inputs "f64" and xref = inputs prec and g = inputs "grad" in
  let x = Array.concat mine in
  let mine_rel = rel x x64 and theirs = rel xref x64 in
  report
    ~name:(label ^ "/" ^ input)
    ~pass:(mine_rel <= 2. *. theirs)
    (Printf.sprintf "rel %.3g  torch %s rel %.3g  ratio %.2f" mine_rel prec
       theirs
       (mine_rel /. (2. *. theirs)));
  let terms = unit_roundoff prec *. Nx.item [] (load "objective_terms.npy") in
  let first =
    Array.fold_left ( +. ) 0.
      (Array.map2 (fun g d -> g *. d) g (Array.map2 ( -. ) x x64))
  in
  let residual = Float.abs (loss -. l64 -. first) in
  let text =
    Printf.sprintf
      "raven %.10g  f64 %.10g  first order %.3g  residual %.3g  bound u M %.3g"
      loss l64 first residual terms
  in
  let name = label ^ "/loss" in
  if judge then report ~name ~pass:(residual <= terms) text else note ~name text

let validate_training opts =
  let dir = req opts "--training" and tile = req opts "--tile" in
  if flag opts "--masks" then write_masks dir tile
  else begin
    let device = Option.value ~default:"cpu" (opt opts "--device") in
    let prec = if flag opts "--bf16" then "bf16" else "f32" in
    let branch = branch_dir dir ~device ~prec in
    let flips = flips_file opts dir branch in
    let m, heads, targets, px, counts = training_inputs dir tile in
    let n = Pixels.length px in
    let shape =
      { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 4; width = 256 }
    in
    let b, _ = Batch.pack shape px counts ~from:0 in
    let keys = place_keys oracle_key ~first:0 b in
    let host = Nx.Ptree.instantiate (module Distil.Params) in
    let file f = Filename.concat branch f in
    let ref_grads prec =
      let g = Model.load (file ("grads_" ^ prec ^ ".pt")) in
      let h = load_heads (file ("heads_grads_" ^ prec ^ ".npz")) m.prefixes in
      leaves host { Distil.net = g.net; heads = h }
    in
    let ref_loss prec =
      Nx.item []
        (Nx.unpack Nx.float64 (Nx_io.load_npy (file ("loss_" ^ prec ^ ".npy"))))
    in
    on device
      {
        run =
          (fun (type d) (t : d target) ->
            let params =
              Nx.Ptree.place
                (module Distil.Params)
                t.at
                { Distil.net = m.net; heads }
            in
            let mt = Nx.Ptree.place (module Model) t.at m in
            let p = Nx.Ptree.instantiate (module Distil.Params) in
            let names = ref [] in
            let f (params, b, keys, targets) =
              let objective (type c) (dt : (float, c) Nx.dtype) q =
                let q = Nx.Ptree.cast (module Distil.Params) dt q in
                let acc = ref [] in
                let l =
                  Distil.loss ~record:(recorder acc) ~dropout:Model.v2_dropout
                    ~keys mt q b ~targets
                in
                names := List.rev_map fst !acc;
                (Nx.div_s l (Float.of_int n), List.rev_map snd !acc)
              in
              let aux = Nx.Ptree.(list tensor) in
              let loss, g, z =
                if prec = "bf16" then
                  Rune.value_and_grad_aux p aux (objective Nx.bfloat16) params
                else Rune.value_and_grad_aux p aux (objective Nx.float32) params
              in
              (loss, (g, z))
            in
            let args =
              ( params,
                Batch.place t.at b,
                Nx.Rng.place t.at keys,
                Nx.place t.at targets )
            in
            let loss, (grads, z) =
              if not t.jit then f args
              else
                Rune.jit
                  Nx.Ptree.(
                    quad p (v Batch.walk) Nx.Rng.ptree tensor
                    @-> returns (pair tensor (pair p (list tensor))))
                  f args
            in
            let values, relus =
              List.partition
                (fun (n, _) -> loss_input n)
                (List.combine !names z)
            in
            let ps =
              List.map
                (fun (name, z) ->
                  match String.sub name 0 3 with
                  | "s2_" -> by_place b.s2 name z
                  | "s1_" -> by_place b.s1 name z
                  | _ -> whole name z)
                relus
            in
            if flag opts "--signs" then save_signs branch ps
            else begin
              check_branch branch ps;
              let label = Printf.sprintf "training/%s/%s" device prec in
              let loss = Nx.item [] (Nx.place Nx.Host.on loss) in
              let embedding kind =
                match kind with
                | "grad" -> "embedding_grad.npy"
                | k -> "embedding_" ^ k ^ ".npy"
              in
              check_loss ~label ~prec ~judge:(prec <> "bf16") ~input:"embedding"
                branch ~loss ~l64:(ref_loss "f64")
                ~mine:(List.map (fun (_, e) -> (whole "" e).z) values)
                ~files:[ embedding ];
              check_leaves ~label ~prec (leaves p grads) ~r64:(ref_grads "f64")
                ~rref:(ref_grads prec) ~terms:(ref_grads "zero");
              check_flips ~label ~prec ~out:flips branch ps
            end);
      }
  end

(* validate --barlow *)

(* The oracle's pretraining: the small teacher or student [model.pt] and the
   projector [projector.npz] of [gen/grads.py barlow-init], its group of the
   tile's pixels [pixels.npy], its mixing weight and permutation, and the views
   raven draws of them at [length] observations, keyed by each pixel's index. *)
let barlow_inputs dir tile =
  let file f = Filename.concat dir f in
  let m = Model.load (file "model.pt") in
  let projector = load_projector (file "projector.npz") in
  let px, _ = Pixels.of_tile tile in
  let px =
    take_pixels px (Nx.unpack Nx.int64 (Nx_io.load_npy (file "pixels.npy")))
  in
  let length =
    Int64.to_int
      (Nx.item [] (Nx.unpack Nx.int64 (Nx_io.load_npy (file "length.npy"))))
  in
  let v = Barlow.group ~key:oracle_key ~length ~first:0 px in
  let v =
    {
      v with
      alpha = Nx.unpack Nx.float32 (Nx_io.load_npy (file "alpha.npy"));
      perm = Nx.unpack Nx.int64 (Nx_io.load_npy (file "perm.npy"));
    }
  in
  (m, projector, v)

(* The oracle's dropout: the reference's encoder layers and its reducer's. *)
let barlow_dropout = Model.v2_dropout

let write_views dir tile =
  let m, _, v = barlow_inputs dir tile in
  let save name x = Nx_io.save_npy (Filename.concat dir name) x in
  let keys i = Nx.Rng.of_tensor (Nx.get [ i ] (Nx.Rng.to_tensor v.keys)) in
  let order = ref [] in
  List.iteri
    (fun i b ->
      let blocks = Model.blocks m (Model.inputs m b) in
      List.iter
        (fun s ->
          save
            (Printf.sprintf "raven_a%d_%s.npy" (i + 1) s)
            (List.assoc s blocks))
        [ "s2_in"; "s1_in" ])
    [ v.a1; v.a2 ];
  (* The mixed view's masks are drawn on the first view's layout. *)
  List.iteri
    (fun i b ->
      let masks = Model.masks ~dropout:barlow_dropout ~keys:(keys i) m b in
      order := List.map fst masks;
      List.iter
        (fun (name, x) ->
          save (Printf.sprintf "mask_v%d_%s.npy" i name) (Nx.cast Nx.uint8 x))
        masks)
    [ v.a1; v.a2; v.a1 ];
  Out_channel.with_open_bin (Filename.concat dir "raven_masks.json") (fun oc ->
      Printf.fprintf oc "{\"order\": [%s]}\n"
        (String.concat ", " (List.map (Printf.sprintf "%S") !order)));
  Printf.printf "%s: views of %d pixels, %d masks a view\n" dir
    (Nx.dim 0 v.perm) (List.length !order)

let validate_barlow opts =
  let dir = req opts "--barlow" and tile = req opts "--tile" in
  if flag opts "--masks" then write_views dir tile
  else begin
    let device = Option.value ~default:"cpu" (opt opts "--device") in
    let branch = branch_dir dir ~device ~prec:"f32" in
    let flips = flips_file opts dir branch in
    let m, projector, v = barlow_inputs dir tile in
    let host = Nx.Ptree.instantiate (module Barlow.Params) in
    let file f = Filename.concat branch f in
    let ref_grads prec =
      let g = Model.load (file ("grads_" ^ prec ^ ".pt")) in
      let pr = load_projector (file ("projector_grads_" ^ prec ^ ".npz")) in
      leaves host { Barlow.net = g.net; projector = pr }
    in
    let ref_loss prec =
      Nx.item []
        (Nx.unpack Nx.float64 (Nx_io.load_npy (file ("loss_" ^ prec ^ ".npy"))))
    in
    on device
      {
        run =
          (fun (type d) (t : d target) ->
            let p = Nx.Ptree.instantiate (module Barlow.Params) in
            let params =
              Nx.Ptree.place
                (module Barlow.Params)
                t.at
                { Barlow.net = m.net; projector }
            in
            let mt = Nx.Ptree.place (module Model) t.at m in
            let names = ref [] in
            let f (params, v) =
              let loss, g, z =
                Rune.value_and_grad_aux p
                  Nx.Ptree.(list tensor)
                  (fun q ->
                    let acc = ref [] in
                    let l =
                      Barlow.loss ~record:(recorder acc) Barlow.v1_weights
                        ~dropout:barlow_dropout ~remat:false mt q v
                    in
                    names := List.rev_map fst !acc;
                    (l, List.rev_map snd !acc))
                  params
              in
              (loss, (g, z))
            in
            let args = (params, Nx.Ptree.place (module Barlow.Group) t.at v) in
            let loss, (grads, z) =
              if not t.jit then f args
              else
                Rune.jit
                  Nx.Ptree.(
                    pair p (instantiate (module Barlow.Group))
                    @-> returns (pair tensor (pair p (list tensor))))
                  f args
            in
            let values, relus =
              List.partition
                (fun (n, _) -> loss_input n)
                (List.combine !names z)
            in
            let ps = List.map (fun (n, z) -> whole n z) relus in
            if flag opts "--signs" then save_signs branch ps
            else begin
              check_branch branch ps;
              let label = Printf.sprintf "barlow/%s" device in
              let loss = Nx.item [] (Nx.place Nx.Host.on loss) in
              let projection v kind =
                match kind with
                | "grad" -> Printf.sprintf "projection_grad_v%d.npy" v
                | k -> Printf.sprintf "projection_%s_v%d.npy" k v
              in
              let mine =
                List.map
                  (fun v ->
                    (whole ""
                       (List.assoc (Printf.sprintf "v%d_projection" v) values))
                      .z)
                  [ 0; 1; 2 ]
              in
              check_loss ~label ~prec:"f32" ~judge:true ~input:"projection"
                branch ~loss ~l64:(ref_loss "f64") ~mine
                ~files:(List.map projection [ 0; 1; 2 ]);
              check_leaves ~label ~prec:"f32" (leaves p grads)
                ~r64:(ref_grads "f64") ~rref:(ref_grads "f32")
                ~terms:(ref_grads "zero");
              check_flips ~label ~prec:"f32" ~out:flips branch ps
            end);
      }
  end

(* probe *)

(* Each feature's label, its [property] as text, from GeoJSON properties. *)
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
      | _ -> die "%s: feature %d has no %S" file i property)
    props

(* The owner of each cell of [grid]: the feature whose region holds the cell's
   centre, or -1. Fields that tile the plane edge to edge hold each centre once,
   so the owner does not depend on the features' order. v1.0 rasterises its
   probe's fields with rasterio's default, this rule (tessera-util@290feaf
   src/utils/convert_shp_to_tiff.py:42-49); tessera-eval@9274405
   tessera_eval/rasterize.py:37-44 passes all_touched. *)
let owners grid polys =
  let feature, cell = Jord.Rasterize.centres grid polys in
  let none = Nx.full Nx.int64 [| grid.Jord.Grid.rows * grid.cols |] (-1L) in
  Nx.to_array (Nx.scatter ~axis:0 cell feature none)

(* v1.0's split of a class's fields: the smallest by area first, each taken
   while the fields taken hold less than [share] of the class's area
   (utils/metrics.py@290feaf:553-561). The first is always taken. [fields] are
   (field, area) pairs; ties go by field index. *)
let training_fields ~share fields =
  let total = List.fold_left (fun a (_, n) -> a + n) 0 fields in
  let target = share *. Float.of_int total in
  let sorted = List.sort (fun (f, n) (g, m) -> compare (n, f) (m, g)) fields in
  let rec take acc = function
    | (f, n) :: rest when Float.of_int acc < target -> f :: take (acc + n) rest
    | _ -> []
  in
  take 0 sorted

let probe opts =
  let tile = req opts "--tile" and labels = req opts "--labels" in
  let property = Option.value ~default:"class" (opt opts "--property") in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let c = float_opt opts "--c" 0.1 in
  let share = float_opt opts "--share" 0.3 in
  let m =
    match (opt opts "--model", opt opts "--arch") with
    | Some file, _ -> Model.load file
    | None, Some a ->
        Model.init ~key:(Nx.Rng.fold_in (Nx.Rng.key 0) init_site) (arch_named a)
    | None, None -> die "--model or --arch is required\n%s" usage
  in
  let m =
    match opt opts "--state" with
    | None -> m
    | Some st -> Checkpoint.part st (module Model) ~path:"model" ~like:m
  in
  let px, (h, w) = Pixels.of_tile tile in
  let grid = landmask ~tile:(h, w) (req opts "--landmask") in
  let polys, props =
    match
      Result.bind
        (Jsont_bytesrw.decode_string Jsont.json
           (In_channel.with_open_bin labels In_channel.input_all))
        Jord.Geojson.polygons
    with
    | Ok x -> x
    | Error e -> die "%s: %s" labels e
  in
  let names = labels_of ~property labels props in
  let owner = owners grid (Jord.Polygons.to_crs grid.crs polys) in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 4096)
      ~slots:(int_opt opts "--slots" 32)
  in
  let e =
    let r = ref None in
    on device
      {
        run =
          (fun t ->
            r := Some (Embedding.encode ~shape t.at (forward t (place t m)) px));
      };
    Option.get !r
  in
  let e =
    match opt opts "--dims" with
    | None -> e
    | Some k -> Nx.slice [ Nx.A; Nx.R (0, int_of_string k) ] e
  in
  (* A pixel with no observation embeds as NaN, and is left out. *)
  let finite = Nx.to_array (Nx.all ~axes:[ 1 ] (Nx.logical_not (Nx.isnan e))) in
  let labelled =
    List.filter (fun i -> owner.(i) >= 0L) (List.init (h * w) Fun.id)
  in
  let cells, empty = List.partition (fun i -> finite.(i)) labelled in
  let field i = Int64.to_int owner.(i) in
  let classes =
    List.sort_uniq compare (List.map (fun i -> names.(field i)) cells)
  in
  let id n =
    let rec go k = function
      | x :: _ when x = n -> k
      | _ :: r -> go (k + 1) r
      | [] -> assert false
    in
    go 0 classes
  in
  let area = Array.make (Array.length names) 0 in
  List.iter (fun i -> area.(field i) <- area.(field i) + 1) cells;
  let training = Array.make (Array.length names) false in
  List.iter
    (fun cl ->
      let fields =
        List.filter
          (fun (f, n) -> n > 0 && names.(f) = cl)
          (List.mapi (fun f n -> (f, n)) (Array.to_list area))
      in
      let taken = training_fields ~share fields in
      List.iter (fun f -> training.(f) <- true) taken;
      let pixels l = List.fold_left (fun a f -> a + area.(f)) 0 l in
      Printf.printf
        "probe: class %s: %d fields, %d pixels, %d training fields of %d pixels\n"
        cl (List.length fields)
        (pixels (List.map fst fields))
        (List.length taken) (pixels taken))
    classes;
  let train, test = List.partition (fun i -> training.(field i)) cells in
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
    (Eval.rankme (x cells));
  match opt opts "--out" with
  | None -> ()
  | Some out ->
      let save n a = Nx_io.save_npy (out ^ "_" ^ n ^ ".npy") a in
      save "x_train" (x train);
      save "y_train" (y train);
      save "x_test" (x test);
      save "y_test" (y test);
      save "w" probe.w;
      save "b" probe.b

(* Main *)

(* The teacher's reference has one float32 fixture and two bfloat16 ones of
   equal bits, so it has no measured spread of its own at either precision.
   Float32 is judged at the students' float32 bound; bfloat16 against bfloat16
   autocast at twice the reference's bfloat16-against-float32 distance (max |Δ|
   2.9e-2, 1 − cos 8.8e-6), the most rounding to bfloat16 moved it. A pair
   across precisions is reported, not judged. *)
let teacher_bf16_tol = (2. *. 2.9e-2, 2. *. 8.8e-6)

let validate_teacher (type d) (t : d target) m ~fixtures ~bf16:with_bf16 =
  if not t.jit then die "the teacher validates on a GPU";
  let tile = "small_64" in
  let px, _ = Pixels.of_tile (Filename.concat fixtures ("tiles/" ^ tile)) in
  let valid = valid_of fixtures tile in
  let shape = Batch.typical ~pixels:512 ~slots:32 in
  let reference kind =
    Printf.sprintf "%s/emb/%s/teacher__cuda__%s__bp512__sdpa.npy" fixtures tile
      kind
  in
  let check label e ~same ~tol ~other =
    let name kind = Printf.sprintf "teacher/%s/%s vs %s" label tile kind in
    validate_embedding ~label:(name same) ~valid ~tol e (reference same);
    validate_embedding ~label:(name other) ~valid e (reference other)
  in
  let e = Embedding.encode ~shape t.at (forward t (place t m)) px in
  check "fp32" e ~same:"fp32" ~tol:(max_abs_tol, cos_tol) ~other:"bf16-autocast";
  Gc.full_major ();
  if with_bf16 then begin
    let e = Embedding.encode ~shape t.at (forward t (place t (bf16 m))) px in
    check "bf16" e ~same:"bf16-autocast" ~tol:teacher_bf16_tol ~other:"fp32"
  end

let validate_student (type d) (t : d target) m ~fixtures ~shape ~device name =
  let m = place t m in
  let f = forward t m in
  List.iter
    (fun tile ->
      let px, _ = Pixels.of_tile (Filename.concat fixtures ("tiles/" ^ tile)) in
      let e = Embedding.encode ~shape t.at f px in
      let valid = valid_of fixtures tile in
      let base =
        Printf.sprintf "%s/emb/%s/%s__%s__fp32__bp4096__fast" fixtures tile name
          (ref_device device)
      in
      let label = Printf.sprintf "%s/%s/%s" name device tile in
      validate_embedding ~label ~valid ~tol:(max_abs_tol, cos_tol) e
        (base ^ ".npy");
      validate_int8 ~label:(label ^ "/int8") ~valid e
        ~codes:(base ^ "_int8.npy") ~scales:(base ^ "_scales.npy"))
    tiles;
  if name = "medium" then validate_blocks t m ~fixtures ~device

let summary () =
  Printf.printf "%d checks, %d failed\n" !checks (List.length !failed);
  List.iter (Printf.printf "failed: %s\n") (List.rev !failed);
  exit (if !failed = [] then 0 else 1)

let validate opts =
  if opt opts "--training" <> None then begin
    validate_training opts;
    summary ()
  end;
  if opt opts "--barlow" <> None then begin
    validate_barlow opts;
    summary ()
  end;
  let weights = req opts "--weights" and fixtures = req opts "--fixtures" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let models =
    match opt opts "--models" with
    | None -> student_names
    | Some l ->
        let ms = String.split_on_char ',' l in
        List.iter
          (fun m ->
            if not (List.mem m ("teacher" :: student_names)) then
              die "--models: %S is not a model\n%s" m usage)
          ms;
        ms
  in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 4096)
      ~slots:(int_opt opts "--slots" 32)
  in
  let one t name =
    match Model.load (weight_file weights name) with
    | exception Failure why -> report ~name ~pass:false why
    | m when name = "teacher" ->
        validate_teacher t m ~fixtures ~bf16:(flag opts "--bf16")
    | m -> validate_student t m ~fixtures ~shape ~device name
  in
  on device
    {
      run =
        (fun t ->
          List.iter
            (fun name ->
              one t name;
              Gc.full_major ())
            models);
    };
  summary ()

let () =
  match List.tl (Array.to_list Sys.argv) with
  | "encode" :: args ->
      encode
        (parse ~flags:[ "--int8"; "--bf16" ]
           ~options:
             [
               "--model";
               "--tile";
               "--out";
               "--landmask";
               "--device";
               "--dims";
               "--pixels";
               "--slots";
             ]
           args)
  | "validate" :: args ->
      validate
        (parse
           ~flags:[ "--bf16"; "--masks"; "--signs" ]
           ~options:
             [
               "--weights";
               "--fixtures";
               "--device";
               "--models";
               "--pixels";
               "--slots";
               "--training";
               "--barlow";
               "--tile";
               "--out";
             ]
           args)
  | "corpus" :: args ->
      corpus
        (parse ~flags:[]
           ~options:[ "--tiles"; "--out"; "--seed"; "--bytes"; "--min" ]
           args)
  | "targets" :: args ->
      targets
        (parse ~flags:[ "--bf16"; "--fp16" ]
           ~options:
             [
               "--teacher";
               "--corpus";
               "--out";
               "--device";
               "--chunk";
               "--pixels";
               "--slots";
             ]
           args)
  | "distil" :: args ->
      distil
        (parse ~flags:[ "--bf16"; "--resume" ]
           ~options:
             [
               "--corpus";
               "--targets";
               "--out";
               "--init";
               "--heads";
               "--arch";
               "--device";
               "--seed";
               "--steps";
               "--groups";
               "--pixels";
               "--slots";
               "--every";
               "--lr";
               "--warmup";
               "--decay";
               "--clip";
               "--dropout";
               "--reducer-dropout";
               "--memory";
               "--heldout";
               "--until";
             ]
           args)
  | "pretrain" :: args ->
      pretrain
        (parse
           ~flags:[ "--bf16"; "--resume"; "--remat" ]
           ~options:
             [
               "--corpus";
               "--out";
               "--init";
               "--projector-weights";
               "--arch";
               "--projector";
               "--device";
               "--seed";
               "--steps";
               "--groups";
               "--pixels";
               "--length";
               "--every";
               "--lr";
               "--warmup";
               "--decay";
               "--clip";
               "--dropout";
               "--reducer-dropout";
               "--memory";
               "--until";
               "--rankme";
             ]
           args)
  | "probe" :: args ->
      probe
        (parse ~flags:[]
           ~options:
             [
               "--model";
               "--arch";
               "--tile";
               "--landmask";
               "--labels";
               "--state";
               "--property";
               "--device";
               "--c";
               "--share";
               "--dims";
               "--pixels";
               "--slots";
               "--out";
             ]
           args)
  | ("-h" | "--help" | "help") :: _ -> print_endline usage
  | _ -> die "%s" usage
