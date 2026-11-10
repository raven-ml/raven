(* What the site shows about each package that no file in the repository
   states: its glyph and colour, its group, the Python library it stands in
   for, and the one line that introduces it. A package's name and synopsis
   come from its dune-project. Every package there is either listed here or
   in [hidden], so a new package can't go missing from the site. *)

type entry = {
  name : string;
  glyph : (string * string) option;  (** The rune and its colour. *)
  like : string;  (** Its counterpart in Python. *)
  about : string;  (** One line, for its card. *)
}

let v ?glyph name ~like about = { name; glyph; like; about }

(* The core's groups, in the order the site lists them. *)
let groups =
  [
    ( "foundation",
      [
        v "rig" ~like:"CUDA driver API" "Devices such as GPUs, and the order of the work they run.";
        v "nx" ~like:"NumPy" "N-dimensional arrays on the host and on devices, with linear algebra and FFT.";
        v "tolk" ~glyph:("ᛏ", "#8b9bb4") ~like:"tinygrad" "A minimal ML compiler that fuses tensor programs into GPU kernels.";
        v "rune" ~glyph:("ᚱ", "#f59e0b") ~like:"JAX" "Automatic differentiation, vectorizing maps and compilation.";
      ] );
    ( "machine learning",
      [
        v "kaun" ~glyph:("ᚴ", "#ef4444") ~like:"Flax" "Neural networks and training, on typed parameter trees.";
        v "vega" ~glyph:("ᚹ", "#eab308") ~like:"Optax" "Composable gradient-based optimizers.";
        v "brot" ~glyph:("ᚨ", "#06b6d4") ~like:"HF Tokenizers" "Tokenization that reads HuggingFace tokenizer.json.";
      ] );
    ( "scientific computing",
      [
        v "jera" ~glyph:("ᛄ", "#14b8a6") ~like:"SciPy" "Systems, minima, integrals, interpolation, ODEs and SDEs.";
        v "norn" ~glyph:("ᚾ", "#818cf8") ~like:"NumPyro / Stan" "Probabilistic models, samplers and diagnostics.";
      ] );
    ( "data and visualization",
      [
        v "talon" ~glyph:("ᛃ", "#ec4899") ~like:"pandas / Polars" "Dataframes with typed queries, CSV and Parquet.";
        v "hugin" ~glyph:("ᛞ", "#b197fc") ~like:"Matplotlib" "Publication-quality plotting.";
      ] );
    ( "tools",
      [
        v "quill" ~glyph:("ᛈ", "#10b981") ~like:"Jupyter" "An interactive REPL and notebooks as markdown files.";
        v "munin" ~glyph:("ᛗ", "#f97316") ~like:"W&B / MLflow" "Local experiment tracking with a live terminal dashboard.";
      ] );
  ]

(* The manual, doc/manual/<name>.md, in the order the docs list it. *)
let manual = [ "introduction"; "installation"; "quickstart"; "coming-from-python"; "ecosystem-overview" ]

(* Packages in dune-project the site doesn't list. *)
let hidden = [ "raven"; "ppx_ptree"; "lz" ]

(* Contrib packages and their glyphs; their one line is their synopsis. *)
let contrib = [ ("fehu", Some "ᚠ"); ("sowilo", Some "ᛋ"); ("ymir", None) ]

(* The landing page's code tabs, in order, with what each library is. Each
   tab shows the first tested ocaml block of the library's doc/index.md, or
   of the section a tab names, as a guide of the library and a heading's
   id. *)
let tabs =
  [ ("nx", "n-dimensional arrays", None);
    ("rune", "automatic differentiation", Some ("06-compilation.md", "a-training-step"));
    ("kaun", "neural networks", None); ("brot", "tokenization", None); ("talon", "dataframes", None);
    ("jera", "numerical methods", None); ("hugin", "plotting", None) ]

let entries = List.concat_map snd groups
let find name = List.find (fun e -> e.name = name) entries
