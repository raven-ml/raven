# Nx-Datasets Developer Guide

## Architecture

Nx-datasets provides automatic download, caching, and loading of popular ML datasets as Nx tensors. It handles fetching, parsing binary formats, and converting to standardized tensor representations.

### Core Components

- **[lib/dataset_utils.ml](lib/dataset_utils.ml)**: Download, caching, and file utilities
- **[lib/datasets/](lib/datasets/)**: Individual dataset loaders (MNIST, CIFAR-10, etc.)
- **[lib/generators.ml](lib/generators.ml)**: Synthetic dataset generators
- **[lib/nx_datasets.ml](lib/nx_datasets.ml)**: Public API aggregating all datasets

### Key Design Principles

1. **Automatic caching**: Download once, cache in `~/.cache/ocaml-nx/datasets/`
2. **Lazy loading**: Download only when needed
3. **Nx-native**: Return data as Nx tensors ready for ML
4. **Error handling**: Clear messages for network/parsing failures
5. **Standardized formats**: Consistent shapes and dtypes across datasets

## Data Flow

```
User calls load_mnist()
        ↓
Check cache: ~/.cache/ocaml-nx/datasets/mnist/
        ↓
   Not cached? → Download from source
        ↓
Parse binary format (IDX for MNIST)
        ↓
Convert to Nx tensors
        ↓
Return (train_data, test_data)
```

## Download and Caching

### Cache Directory

```ocaml
let cache_dir = Xdg.cache_base ^ dataset_name ^ "/"
(* ~/.cache/ocaml-nx/datasets/mnist/ *)
```

**Cache structure:**

```
~/.cache/ocaml-nx/datasets/
├── mnist/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── cifar10/
│   └── cifar-10-batches-bin/
└── ...
```

### Download Mechanism

Using libcurl for HTTP downloads:

```ocaml
let download_file url dest_path =
  let h = new Curl.handle in
  h#set_url url;
  h#set_followlocation true;  (* Follow redirects *)
  h#set_timeout 300;          (* 5 min timeout *)

  let oc = open_out_bin dest_path in
  h#set_writefunction (fun data ->
    output_string oc data;
    String.length data
  );

  h#perform;
  close_out oc
```

**Error handling:**
- Network failures: Clear error with URL
- Disk full: IO exception with path
- Invalid format: Parsing error with dataset name

### Decompression

Handle gzip-compressed downloads:

```ocaml
let gunzip_file gz_path output_path =
  let cmd = Printf.sprintf "gunzip -c %s > %s"
    (Filename.quote gz_path)
    (Filename.quote output_path)
  in
  match Sys.command cmd with
  | 0 -> ()
  | _ -> failwith ("gunzip failed: " ^ gz_path)
```

## Dataset Loading

### MNIST Format (IDX)

MNIST uses IDX binary format:

```
Magic number (4 bytes): 0x00000803 (images) or 0x00000801 (labels)
Dimensions (4 bytes each): [num_images, height, width] or [num_labels]
Data (1 byte per pixel/label): uint8 values
```

**Parsing:**

```ocaml
let parse_idx_images path =
  let ic = open_in_bin path in

  (* Read magic number *)
  let magic = read_int32_be ic in
  assert (magic = 0x00000803l);

  (* Read dimensions *)
  let n_images = read_int32_be ic in
  let height = read_int32_be ic in
  let width = read_int32_be ic in

  (* Read pixels *)
  let size = n_images * height * width in
  let data = Array.make size 0 in
  for i = 0 to size - 1 do
    data.(i) <- input_byte ic
  done;
  close_in ic;

  (* Convert to Nx tensor *)
  Nx.create Nx.uint8 [|n_images; height; width; 1|] data
```

### CIFAR-10 Format

Binary batches with interleaved labels and images:

```
Label (1 byte)
Red channel (1024 bytes)
Green channel (1024 bytes)
Blue channel (1024 bytes)
[Repeat for 10000 images per batch]
```

**Parsing:**

```ocaml
let parse_cifar_batch path =
  let ic = open_in_bin path in
  let images = Array.make (10000 * 32 * 32 * 3) 0 in
  let labels = Array.make 10000 0 in

  for i = 0 to 9999 do
    labels.(i) <- input_byte ic;  (* Label *)

    (* Read RGB channels *)
    for c = 0 to 2 do
      for pixel = 0 to 1023 do
        let idx = i * (32 * 32 * 3) + pixel * 3 + c in
        images.(idx) <- input_byte ic
      done
    done
  done;
  close_in ic;

  let x = Nx.create Nx.uint8 [|10000; 32; 32; 3|] images in
  let y = Nx.create Nx.uint8 [|10000; 1|] labels in
  (x, y)
```

### CSV Datasets (Iris, etc.)

Parse CSV with header:

```ocaml
let parse_csv path =
  let lines = read_lines path in
  let header, data_lines = List.hd lines, List.tl lines in

  (* Parse data rows *)
  let rows = List.map (String.split_on_char ',') data_lines in

  (* Convert to arrays *)
  let features = List.map (fun row ->
    List.take (List.length row - 1) row
    |> List.map float_of_string
    |> Array.of_list
  ) rows |> Array.concat in

  let labels = List.map (fun row ->
    List.nth row (List.length row - 1) |> int_of_string
  ) rows |> Array.of_list in

  let n_samples = List.length rows in
  let n_features = List.length (List.hd rows) - 1 in

  (Nx.create Nx.float64 [|n_samples; n_features|] features,
   Nx.create Nx.int32 [|n_samples; 1|] labels)
```

## Synthetic Datasets

### Make Blobs (Clustering)

Generate clustered data:

```ocaml
let make_blobs ~n_samples ~n_features ~centers =
  let n_centers = Array.length centers in
  let samples_per_center = n_samples / n_centers in

  (* For each center *)
  let data = List.init n_centers (fun i ->
    let center = centers.(i) in

    (* Generate samples around center *)
    List.init samples_per_center (fun _j ->
      (* Add Gaussian noise *)
      Array.init n_features (fun k ->
        center.(k) +. gaussian_noise ~std:cluster_std ()
      )
    )
  ) |> List.concat in

  Nx.create Nx.float32 [|n_samples; n_features|] (Array.concat data)
```

### Make Regression

Linear model with noise:

```ocaml
let make_regression ~n_samples ~n_features ~noise =
  (* Random coefficients *)
  let coef = Array.init n_features (fun _ -> Random.float 2. -. 1.) in

  (* Generate X *)
  let x_data = Array.init (n_samples * n_features) (fun _ ->
    Random.float 2. -. 1.
  ) in
  let x = Nx.create Nx.float32 [|n_samples; n_features|] x_data in

  (* Compute y = X * coef + noise *)
  let y_data = Array.init n_samples (fun i ->
    let dot = ref 0. in
    for j = 0 to n_features - 1 do
      dot := !dot +. x_data.(i * n_features + j) *. coef.(j)
    done;
    !dot +. gaussian_noise ~std:noise ()
  ) in
  let y = Nx.create Nx.float32 [|n_samples; 1|] y_data in

  (x, y)
```

## Development Workflow

### Building and Testing

```bash
# Build nx-datasets
dune build nx-datasets/

# Run tests
dune build nx-datasets/test/test_nx_datasets.exe && _build/default/nx-datasets/test/test_nx_datasets.exe

# Clear cache for testing
rm -rf ~/.cache/ocaml-nx/datasets/mnist
```

### Testing Datasets

```ocaml
let test_mnist_loading () =
  let (x_train, y_train), (x_test, y_test) = load_mnist () in

  (* Check shapes *)
  Alcotest.(check (array int)) "train images shape"
    [|60000; 28; 28; 1|] (Nx.shape x_train);

  Alcotest.(check (array int)) "train labels shape"
    [|60000; 1|] (Nx.shape y_train);

  (* Check dtypes *)
  Alcotest.(check bool) "images are uint8"
    true (Nx.dtype x_train = Nx.uint8);

  (* Check value ranges *)
  let min_val = Nx.min x_train |> Nx.item in
  let max_val = Nx.max x_train |> Nx.item in
  Alcotest.(check bool) "pixel values in [0, 255]"
    true (min_val >= 0 && max_val <= 255)
```

### Mock Downloads for Testing

Avoid network calls in tests:

```ocaml
(* Create fake cached files *)
let setup_mock_cache () =
  let cache_dir = get_cache_dir "mnist" in
  mkdir_p cache_dir;

  (* Write minimal valid IDX files *)
  write_mock_idx (cache_dir ^ "train-images-idx3-ubyte") ~n:100;
  write_mock_idx (cache_dir ^ "train-labels-idx1-ubyte") ~n:100

let test_with_mock () =
  setup_mock_cache ();
  let data = load_mnist () in
  (* Test with mock data *)
  ...
```

## Adding Datasets

### New Dataset Loader

1. Create loader module:

```ocaml
(* lib/datasets/my_dataset.ml *)
let url = "https://example.com/dataset.tar.gz"

let load () =
  let cache_dir = Dataset_utils.get_cache_dir "my_dataset" in

  (* Download if not cached *)
  if not (Sys.file_exists (cache_dir ^ "data.bin")) then
    Dataset_utils.download_file url (cache_dir ^ "archive.tar.gz");
    (* Extract and process *)
    ...

  (* Parse and return tensors *)
  let data = parse_binary (cache_dir ^ "data.bin") in
  (features, labels)
```

2. Add to public API:

```ocaml
(* lib/nx_datasets.ml *)
let load_my_dataset = My_dataset.load
```

3. Document in `.mli`:

```ocaml
val load_my_dataset : unit -> Nx.float32_t * Nx.int32_t
(** [load_my_dataset ()] loads My Dataset.

    Returns features as float32 tensor of shape [|N; D|] and labels. *)
```

### Binary Format Parsing

For custom binary formats:

```ocaml
let parse_custom_format path =
  let ic = open_in_bin path in

  (* Read header *)
  let magic = really_input_string ic 4 in
  assert (magic = "CUST");

  (* Read metadata *)
  let n_samples = input_binary_int ic in
  let n_features = input_binary_int ic in

  (* Read data *)
  let data = Array.init (n_samples * n_features) (fun _ ->
    input_float ic
  ) in

  close_in ic;
  Nx.create Nx.float32 [|n_samples; n_features|] data
```

## Common Pitfalls

### Byte Order

MNIST uses big-endian integers:

```ocaml
(* Wrong: platform-dependent *)
let n = input_binary_int ic

(* Correct: explicit big-endian *)
let read_int32_be ic =
  let b1 = input_byte ic in
  let b2 = input_byte ic in
  let b3 = input_byte ic in
  let b4 = input_byte ic in
  Int32.of_int ((b1 lsl 24) lor (b2 lsl 16) lor (b3 lsl 8) lor b4)
```

### File Permissions

Cache directory might not exist:

```ocaml
(* Wrong: assume directory exists *)
let oc = open_out (cache_dir ^ "file.bin")

(* Correct: create directory first *)
Dataset_utils.mkdir_p cache_dir;
let oc = open_out (cache_dir ^ "file.bin")
```

### Channel Ordering

CIFAR-10 is channel-first (CHW), but we want HWC:

```ocaml
(* Wrong: leave as CHW *)
Nx.create Nx.uint8 [|n; 3; 32; 32|] data

(* Correct: transpose to HWC *)
let chw = Nx.create Nx.uint8 [|n; 3; 32; 32|] data in
Nx.transpose chw ~axes:[|0; 2; 3; 1|]  (* → [n; 32; 32; 3] *)
```

### Download Failures

Handle network issues gracefully:

```ocaml
try
  download_file url dest_path
with
| Curl.CurlException (code, _, msg) ->
    failwith (Printf.sprintf "Download failed: %s (error %d)" msg code)
| Sys_error msg ->
    failwith (Printf.sprintf "File error: %s" msg)
```

## Performance

- **Lazy downloads**: Only download when `load_*` is called
- **Binary parsing**: Use `input_byte` for efficiency
- **Preallocate arrays**: Avoid reallocations during parsing
- **Cache aggressively**: Never re-download

## Code Style

- **Dataset modules**: One module per dataset in `datasets/`
- **Errors**: Descriptive failures with dataset name
- **Shapes**: Document expected tensor shapes in docstrings
- **URLs**: Use stable, official sources

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [nx/HACKING.md](../nx/HACKING.md): Nx tensor operations
- Dataset format specifications (IDX, CIFAR, etc.)
