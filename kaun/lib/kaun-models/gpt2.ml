open Rune

(* Configuration *)

type config = {
  vocab_size : int;
  n_positions : int;
  n_embd : int;
  n_layer : int;
  n_head : int;
  n_inner : int option;
  activation_function : [ `gelu | `relu | `swish | `gelu_new ];
  resid_pdrop : float;
  embd_pdrop : float;
  attn_pdrop : float;
  layer_norm_epsilon : float;
  initializer_range : float;
  scale_attn_weights : bool;
  use_cache : bool;
  scale_attn_by_inverse_layer_idx : bool;
  reorder_and_upcast_attn : bool;
  bos_token_id : int option;
  eos_token_id : int option;
  pad_token_id : int option;
}

let default_config =
  {
    vocab_size = 50257;
    n_positions = 1024;
    n_embd = 768;
    n_layer = 12;
    n_head = 12;
    n_inner = None;
    (* Defaults to 4 * n_embd *)
    activation_function = `gelu_new;
    resid_pdrop = 0.1;
    embd_pdrop = 0.1;
    attn_pdrop = 0.1;
    layer_norm_epsilon = 1e-5;
    initializer_range = 0.02;
    scale_attn_weights = true;
    use_cache = true;
    scale_attn_by_inverse_layer_idx = false;
    reorder_and_upcast_attn = false;
    bos_token_id = Some 50256;
    eos_token_id = Some 50256;
    pad_token_id = None;
  }

let gpt2_small = default_config

let gpt2_medium =
  { default_config with n_embd = 1024; n_layer = 24; n_head = 16 }

let gpt2_large =
  { default_config with n_embd = 1280; n_layer = 36; n_head = 20 }

let gpt2_xl = { default_config with n_embd = 1600; n_layer = 48; n_head = 25 }

(* Input type *)
type inputs = {
  input_ids : (int32, int32_elt) Rune.t;
  attention_mask : (int32, int32_elt) Rune.t option;
  position_ids : (int32, int32_elt) Rune.t option;
}

(* GPT-2 specific tokenizer with BPE *)
module Tokenizer = struct
  type t = {
    tokenizer : Saga.Tokenizer.t;
        (* Store the actual tokenizer for encoding/decoding *)
    vocab_size : int;
    bos_token_id : int;
    eos_token_id : int;
    pad_token_id : int option;
  }

  let download_vocab_and_merges model_id =
    (* Download vocab and merges files from HuggingFace if not present *)
    let cache_dir =
      match Sys.getenv_opt "XDG_CACHE_HOME" with
      | Some dir -> dir
      | None -> (
          match Sys.getenv_opt "HOME" with
          | Some home -> Filename.concat home ".cache"
          | None -> "/tmp/.cache")
    in
    let kaun_cache = Filename.concat cache_dir "kaun" in
    let gpt2_cache = Filename.concat kaun_cache "gpt2" in
    let model_cache = Filename.concat gpt2_cache model_id in
    let vocab_file = Filename.concat model_cache "vocab.json" in
    let merges_file = Filename.concat model_cache "merges.txt" in

    (* Create cache directories if they don't exist *)
    if not (Sys.file_exists model_cache) then
      Sys.command (Printf.sprintf "mkdir -p %s" model_cache) |> ignore;

    (* Download vocab.json if it doesn't exist *)
    if not (Sys.file_exists vocab_file) then (
      Printf.printf "Downloading vocab.json for %s...\n%!" model_id;
      let url =
        Printf.sprintf "https://huggingface.co/%s/resolve/main/vocab.json"
          model_id
      in
      let cmd =
        Printf.sprintf
          "curl -L -o %s %s 2>/dev/null || wget -O %s %s 2>/dev/null" vocab_file
          url vocab_file url
      in
      let exit_code = Sys.command cmd in
      if exit_code <> 0 then
        failwith
          (Printf.sprintf "Failed to download vocab.json for %s" model_id));

    (* Download merges.txt if it doesn't exist *)
    if not (Sys.file_exists merges_file) then (
      Printf.printf "Downloading merges.txt for %s...\n%!" model_id;
      let url =
        Printf.sprintf "https://huggingface.co/%s/resolve/main/merges.txt"
          model_id
      in
      let cmd =
        Printf.sprintf
          "curl -L -o %s %s 2>/dev/null || wget -O %s %s 2>/dev/null"
          merges_file url merges_file url
      in
      let exit_code = Sys.command cmd in
      if exit_code <> 0 then
        failwith
          (Printf.sprintf "Failed to download merges.txt for %s" model_id));

    (vocab_file, merges_file)

  let create ?vocab_file ?merges_file ?model_id () =
    (* Either provide vocab_file and merges_file paths, or a model_id to
       download from *)
    let vocab_file, merges_file =
      match (vocab_file, merges_file, model_id) with
      | Some vf, Some mf, _ -> (vf, mf)
      | None, None, Some id -> download_vocab_and_merges id
      | None, None, None -> download_vocab_and_merges "gpt2" (* Default *)
      | _ ->
          failwith "Either provide both vocab_file and merges_file, or model_id"
    in

    (* Create GPT-2 tokenizer with ByteLevel pre-tokenizer Use use_regex:true to
       enable GPT-2 pattern splitting *)
    let tokenizer =
      Saga.Tokenizer.from_model_file ~vocab:vocab_file ~merges:merges_file
        ~pre:
          (Saga.Pre_tokenizers.byte_level ~add_prefix_space:false
             ~use_regex:true ())
        ~decoder:(Saga.Decoders.byte_level ())
        ()
    in
    {
      tokenizer;
      vocab_size = 50257;
      (* GPT-2 vocab size *)
      bos_token_id = 50256;
      (* <|endoftext|> *)
      eos_token_id = 50256;
      (* <|endoftext|> *)
      pad_token_id = None;
      (* GPT-2 doesn't use padding by default *)
    }

  let encode_to_array t text =
    (* Use the tokenizer's encode method *)
    let encoding = Saga.Tokenizer.encode t.tokenizer text in
    Saga.Encoding.get_ids encoding

  let encode t text =
    let token_ids = encode_to_array t text in
    let seq_len = Array.length token_ids in

    (* Convert to tensors *)
    let input_ids =
      Rune.create Int32 [| 1; seq_len |] (Array.map Int32.of_int token_ids)
    in

    (* Return inputs record *)
    { input_ids; attention_mask = None; position_ids = None }

  let encode_batch t ?(max_length = 1024) ?(padding = false) texts =
    let encoded = List.map (encode_to_array t) texts in

    (* Find max length or use specified max_length *)
    let actual_max =
      if padding then max_length
      else
        List.fold_left (fun acc arr -> Int.max acc (Array.length arr)) 0 encoded
    in

    (* Ensure we don't exceed vocabulary size *)
    let vocab_size = t.vocab_size in
    let validate_tokens arr =
      Array.map
        (fun token_id ->
          if token_id >= 0 && token_id < vocab_size then token_id
          else t.eos_token_id (* Replace invalid tokens with EOS *))
        arr
    in

    (* Pad sequences if padding is enabled *)
    let padded =
      if padding then
        let pad_token_id =
          Option.value t.pad_token_id ~default:t.eos_token_id
        in
        List.map
          (fun arr ->
            let validated = validate_tokens arr in
            let len = Array.length validated in
            if len >= actual_max then Array.sub validated 0 actual_max
            else
              Array.append validated
                (Array.make (actual_max - len) pad_token_id))
          encoded
      else
        (* Truncate to max length if no padding *)
        List.map
          (fun arr ->
            let validated = validate_tokens arr in
            let len = Array.length validated in
            if len > actual_max then Array.sub validated 0 actual_max
            else validated)
          encoded
    in

    (* Convert to tensor *)
    let batch_size = List.length padded in
    let flat_data = Array.concat padded in
    let nx_tensor =
      let data = Array.map Int32.of_int flat_data in
      Nx.create Int32 [| batch_size; actual_max |] data
    in
    (* Convert to Rune tensor *)
    Rune.of_nx nx_tensor

  let decode t token_ids =
    (* Decode token IDs back to text using tokenizer *)
    Saga.Tokenizer.decode t.tokenizer token_ids

  (* Get special token IDs *)
  let get_bos_token_id t = t.bos_token_id
  let get_eos_token_id t = t.eos_token_id
  let get_pad_token_id t = t.pad_token_id
  let get_vocab_size t = t.vocab_size
end

(* GPT-2 Embeddings *)
let embeddings ~config () =
  let open Kaun.Layer in
  (* GPT-2 embeddings: token + position *)
  let token_embeddings =
    embedding ~vocab_size:config.vocab_size ~embed_dim:config.n_embd ()
  in
  let position_embeddings =
    embedding ~vocab_size:config.n_positions ~embed_dim:config.n_embd ()
  in
  let dropout = dropout ~rate:config.embd_pdrop () in

  (* Custom module that applies both embeddings and sums them *)
  {
    Kaun.init =
      (fun ~rngs ~dtype ->
        let keys = Rune.Rng.split ~n:3 rngs in
        Kaun.Ptree.record_of
          [
            ("token_embeddings", token_embeddings.init ~rngs:keys.(0) ~dtype);
            ( "position_embeddings",
              position_embeddings.init ~rngs:keys.(1) ~dtype );
            ("dropout", dropout.init ~rngs:keys.(2) ~dtype);
          ]);
    Kaun.apply =
      (fun params ~training ?rngs x ->
        (* x is expected to be float tensor, but we need int indices *)
        let input_ids = Rune.cast Rune.int32 x in
        match params with
        | Kaun.Ptree.Record fields ->
            (* Get each embedding layer's params *)
            let get_params name =
              match Kaun.Ptree.Record.find_opt name fields with
              | Some p -> p
              | None -> failwith ("Embeddings: missing " ^ name)
            in

            (* Manually perform embedding lookups *)
            let get_embedding_table name =
              match get_params name with
              | Tensor t -> t
              | _ -> failwith ("Expected tensor for " ^ name)
            in

            let token_embeddings_table =
              get_embedding_table "token_embeddings"
            in
            let position_embeddings_table =
              get_embedding_table "position_embeddings"
            in

            (* Perform embedding lookups using differentiable operations *)
            let lookup_embeddings embedding_table indices =
              let batch_size = (Rune.shape indices).(0) in
              let seq_len = (Rune.shape indices).(1) in
              let table_shape = Rune.shape embedding_table in
              if Array.length table_shape <> 2 then
                failwith
                  (Printf.sprintf
                     "Embedding table has wrong shape: %d dims, expected 2"
                     (Array.length table_shape));
              let embed_dim = table_shape.(1) in

              (* Flatten indices for gather operation *)
              let indices_flat =
                Rune.reshape [| batch_size * seq_len |] indices
              in

              (* Use take to gather embeddings - this is differentiable *)
              let gathered = Rune.take ~axis:0 indices_flat embedding_table in

              (* Reshape to [batch_size, seq_len, embed_dim] *)
              Rune.reshape [| batch_size; seq_len; embed_dim |] gathered
            in

            (* Apply token embeddings *)
            let token_embeds =
              lookup_embeddings token_embeddings_table input_ids
            in

            (* Create position ids: [0, 1, 2, ..., seq_len-1] *)
            let seq_len =
              (Rune.shape input_ids).(Array.length (Rune.shape input_ids) - 1)
            in
            let batch_size = (Rune.shape input_ids).(0) in
            let position_ids =
              let pos_ids = Rune.zeros Rune.int32 [| batch_size; seq_len |] in
              for b = 0 to batch_size - 1 do
                for s = 0 to seq_len - 1 do
                  Rune.set [ b; s ] pos_ids
                    (Rune.scalar Rune.int32 (Int32.of_int s))
                done
              done;
              pos_ids
            in
            let position_embeds =
              lookup_embeddings position_embeddings_table position_ids
            in

            (* Sum embeddings *)
            let embeddings = Rune.add token_embeds position_embeds in

            (* Apply dropout *)
            let embeddings =
              dropout.apply (get_params "dropout") ~training ?rngs embeddings
            in

            embeddings
        | _ -> failwith "Embeddings: invalid params");
  }

(* Main Model *)

type 'a gpt2 = {
  model : Kaun.Layer.module_;
  params : 'a Kaun.Ptree.t;
  config : config;
  dtype : (float, 'a) Rune.dtype;
}

type 'a output = {
  last_hidden_state : (float, 'a) Rune.t;
  hidden_states : (float, 'a) Rune.t list option;
  attentions : (float, 'a) Rune.t list option;
}

module Gpt2_block = struct
  (* Generate causal attention mask *)
  let causal_mask ~seq_len ~dtype =
    (* Create a lower triangular matrix for causal masking *)
    let mask = ones dtype [| seq_len; seq_len |] in
    tril mask ~k:0

  (* GPT-2 uses GELU activation *)
  let gelu x =
    (* Use exact GELU with erf for numerical stability *)
    (* GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) *)
    Rune.gelu x

  (* Multi-head attention with causal masking *)
  let causal_attention ~n_head ~hidden_size ~params x =
    let batch_size = (shape x).(0) in
    let seq_len = (shape x).(1) in
    let head_dim = hidden_size / n_head in
    let dtype = dtype x in

    (* Get Q, K, V weights and biases *)
    let get_param name =
      match Kaun.Ptree.Record.find_opt name params with
      | Some (Kaun.Ptree.Tensor t) -> t
      | _ -> failwith ("Missing parameter: " ^ name)
    in

    let qkv_weight = get_param "qkv_weight" in
    let qkv_bias = get_param "qkv_bias" in
    let out_weight = get_param "attn_out_weight" in
    let out_bias = get_param "attn_out_bias" in

    (* Compute combined QKV *)
    let qkv =
      add (matmul x qkv_weight) (reshape [| 1; 1; 3 * hidden_size |] qkv_bias)
    in

    (* Split into Q, K, V *)
    let query = slice [ A; A; R (0, hidden_size) ] qkv in
    let key = slice [ A; A; R (hidden_size, 2 * hidden_size) ] qkv in
    let value = slice [ A; A; R (2 * hidden_size, 3 * hidden_size) ] qkv in

    (* Reshape for multi-head attention: [batch, seq, n_head, head_dim] *)
    let reshape_for_heads t =
      let t = reshape [| batch_size; seq_len; n_head; head_dim |] t in
      (* Transpose to [batch, n_head, seq, head_dim] *)
      transpose ~axes:[ 0; 2; 1; 3 ] t
    in

    let query = reshape_for_heads query in
    let key = reshape_for_heads key in
    let value = reshape_for_heads value in

    (* Scaled dot-product attention with causal mask *)
    (* scores = Q @ K^T / sqrt(head_dim) *)
    let key_t = transpose ~axes:[ 0; 1; 3; 2 ] key in
    let scores = matmul query key_t in
    let scores = div_s scores (Float.sqrt (Float.of_int head_dim)) in

    (* Apply causal mask *)
    let mask = causal_mask ~seq_len ~dtype in
    (* Expand mask for batch and heads: [1, 1, seq_len, seq_len] *)
    let mask = reshape [| 1; 1; seq_len; seq_len |] mask in
    (* Where mask is 0, set score to large negative value *)
    let neg_inf = full dtype [| 1; 1; 1; 1 |] (-1e10) in
    let scores = where (equal_s mask 0.0) neg_inf scores in

    (* Softmax over last dimension *)
    let attn_weights = softmax scores ~axes:[ 3 ] in

    (* Apply attention to values *)
    let attn_output = matmul attn_weights value in

    (* Transpose back and reshape *)
    let attn_output = transpose ~axes:[ 0; 2; 1; 3 ] attn_output in
    (* Make contiguous before reshaping - force a copy to ensure it's
       contiguous *)
    let attn_output = copy attn_output in
    let attn_output =
      reshape [| batch_size; seq_len; hidden_size |] attn_output
    in

    (* Output projection *)
    add
      (matmul attn_output out_weight)
      (reshape [| 1; 1; hidden_size |] out_bias)

  (* Feed-forward network *)
  let mlp ~n_inner ~hidden_size ~params x =
    let get_param name =
      match Kaun.Ptree.Record.find_opt name params with
      | Some (Kaun.Ptree.Tensor t) -> t
      | _ -> failwith ("Missing parameter: " ^ name)
    in

    let inter_weight = get_param "inter_weight" in
    let inter_bias = get_param "inter_bias" in
    let out_weight = get_param "out_weight" in
    let out_bias = get_param "out_bias" in

    (* First linear layer *)
    let h =
      add (matmul x inter_weight) (reshape [| 1; 1; n_inner |] inter_bias)
    in
    (* GELU activation *)
    let h = gelu h in
    (* Second linear layer *)
    add (matmul h out_weight) (reshape [| 1; 1; hidden_size |] out_bias)

  (* GPT-2 transformer block with pre-layer normalization *)
  let gpt2_block ~config ~params x =
    let get_param name =
      match Kaun.Ptree.Record.find_opt name params with
      | Some (Kaun.Ptree.Tensor t) -> t
      | _ -> failwith ("Missing parameter: " ^ name)
    in

    let ln1_weight = get_param "attn_gamma" in
    let ln1_bias = get_param "attn_beta" in
    let ln2_weight = get_param "ffn_gamma" in
    let ln2_bias = get_param "ffn_beta" in

    (* Pre-layer norm for attention *)
    let normed =
      layer_norm ~gamma:ln1_weight ~beta:ln1_bias
        ~epsilon:config.layer_norm_epsilon x
    in

    (* Self-attention with residual *)
    let attn_out =
      causal_attention ~n_head:config.n_head ~hidden_size:config.n_embd ~params
        normed
    in
    let x = add x attn_out in

    (* Pre-layer norm for FFN *)
    let normed =
      layer_norm ~gamma:ln2_weight ~beta:ln2_bias
        ~epsilon:config.layer_norm_epsilon x
    in

    (* FFN with residual *)
    let n_inner = Option.value config.n_inner ~default:(4 * config.n_embd) in
    let ffn_out = mlp ~n_inner ~hidden_size:config.n_embd ~params normed in
    add x ffn_out

  (* Stack of GPT-2 blocks *)
  let gpt2_transformer ~config ~layer_params x =
    (* Apply each transformer block sequentially *)
    let rec apply_layers x = function
      | [] -> x
      | params :: rest ->
          let x = gpt2_block ~config ~params x in
          apply_layers x rest
    in
    apply_layers x layer_params
end

let create ?(config = default_config) () =
  (* GPT-2 uses a custom architecture with causal attention *)
  (* We'll implement it as a custom module *)
  {
    Kaun.init =
      (fun ~rngs ~dtype ->
        (* Initialize embeddings *)
        let embeddings_layer = embeddings ~config () in
        let embeddings_params = Kaun.init embeddings_layer ~rngs ~dtype in

        (* Initialize transformer blocks *)
        let layer_params =
          List.init config.n_layer (fun _ ->
              (* Each layer needs initialized parameters *)
              (* For now, return empty - will be filled by from_pretrained *)
              Kaun.Ptree.Record.empty)
        in

        (* Initialize final layer norm *)
        let ln_f_gamma = Rune.ones dtype [| config.n_embd |] in
        let ln_f_beta = Rune.zeros dtype [| config.n_embd |] in

        (* Return params structure *)
        Kaun.Ptree.List
          [
            embeddings_params;
            Kaun.Ptree.List
              (List.map (fun p -> Kaun.Ptree.Record p) layer_params);
            Kaun.Ptree.Record
              (Kaun.Ptree.Record.empty
              |> Kaun.Ptree.Record.add "gamma" (Kaun.Ptree.Tensor ln_f_gamma)
              |> Kaun.Ptree.Record.add "beta" (Kaun.Ptree.Tensor ln_f_beta));
          ]);
    Kaun.apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Kaun.Ptree.List [ embeddings_params; layer_params_list; ln_f_params ]
          -> (
            (* Apply embeddings *)
            let embeddings_layer = embeddings ~config () in
            let x =
              Kaun.apply embeddings_layer embeddings_params ~training:false x
            in

            (* Extract layer params *)
            let layer_params =
              match layer_params_list with
              | Kaun.Ptree.List lst ->
                  List.map
                    (function
                      | Kaun.Ptree.Record r -> r
                      | _ -> failwith "Invalid layer params structure")
                    lst
              | _ -> failwith "Invalid layer params list"
            in

            (* Apply GPT-2 transformer blocks *)
            let x = Gpt2_block.gpt2_transformer ~config ~layer_params x in

            (* Apply final layer norm *)
            match ln_f_params with
            | Kaun.Ptree.Record fields ->
                let gamma =
                  match Kaun.Ptree.Record.find_opt "gamma" fields with
                  | Some (Kaun.Ptree.Tensor t) -> t
                  | _ -> failwith "Missing ln_f gamma"
                in
                let beta =
                  match Kaun.Ptree.Record.find_opt "beta" fields with
                  | Some (Kaun.Ptree.Tensor t) -> t
                  | _ -> failwith "Missing ln_f beta"
                in
                layer_norm x ~gamma ~beta ~epsilon:config.layer_norm_epsilon
            | _ -> failwith "Invalid ln_f params")
        | _ -> failwith "Invalid params structure");
  }

let from_pretrained ?(model_id = "gpt2") ?revision ?cache_config ~dtype () =
  (* Load config and weights from HuggingFace *)
  let cache_config =
    Option.value cache_config ~default:Kaun_huggingface.Config.default
  in
  let revision = Option.value revision ~default:Kaun_huggingface.Latest in

  (* Load config JSON from HuggingFace *)
  let config_json =
    match
      Kaun_huggingface.load_config ~config:cache_config ~revision ~model_id ()
    with
    | Cached json | Downloaded (json, _) -> json
  in

  (* Parse GPT-2 specific config *)
  let gpt2_config =
    let open Yojson.Safe.Util in
    {
      vocab_size = config_json |> member "vocab_size" |> to_int;
      n_positions = config_json |> member "n_positions" |> to_int;
      n_embd = config_json |> member "n_embd" |> to_int;
      n_layer = config_json |> member "n_layer" |> to_int;
      n_head = config_json |> member "n_head" |> to_int;
      n_inner = config_json |> member "n_inner" |> to_int_option;
      activation_function =
        (match
           config_json |> member "activation_function" |> to_string_option
         with
        | Some "gelu_new" -> `gelu_new
        | Some "gelu" -> `gelu
        | Some "relu" -> `relu
        | Some "swish" | Some "silu" -> `swish
        | _ -> `gelu_new);
      resid_pdrop =
        config_json |> member "resid_pdrop" |> to_float_option
        |> Option.value ~default:0.1;
      embd_pdrop =
        config_json |> member "embd_pdrop" |> to_float_option
        |> Option.value ~default:0.1;
      attn_pdrop =
        config_json |> member "attn_pdrop" |> to_float_option
        |> Option.value ~default:0.1;
      layer_norm_epsilon =
        config_json
        |> member "layer_norm_epsilon"
        |> to_float_option |> Option.value ~default:1e-5;
      initializer_range =
        config_json |> member "initializer_range" |> to_float_option
        |> Option.value ~default:0.02;
      scale_attn_weights =
        config_json
        |> member "scale_attn_weights"
        |> to_bool_option |> Option.value ~default:true;
      use_cache =
        config_json |> member "use_cache" |> to_bool_option
        |> Option.value ~default:true;
      scale_attn_by_inverse_layer_idx =
        config_json
        |> member "scale_attn_by_inverse_layer_idx"
        |> to_bool_option
        |> Option.value ~default:false;
      reorder_and_upcast_attn =
        config_json
        |> member "reorder_and_upcast_attn"
        |> to_bool_option
        |> Option.value ~default:false;
      bos_token_id = config_json |> member "bos_token_id" |> to_int_option;
      eos_token_id = config_json |> member "eos_token_id" |> to_int_option;
      pad_token_id = config_json |> member "pad_token_id" |> to_int_option;
    }
  in

  (* Load weights using HuggingFace infrastructure *)
  let hf_params =
    Kaun_huggingface.from_pretrained ~config:cache_config ~revision ~model_id
      ~dtype ()
  in

  (* Map HuggingFace parameter names to our expected structure *)
  let map_huggingface_to_kaun hf_params =
    (* Flatten the nested HuggingFace structure *)
    let rec flatten_ptree prefix tree =
      match tree with
      | Kaun.Ptree.Tensor t -> [ (prefix, t) ]
      | Kaun.Ptree.List lst ->
          List.concat
            (List.mapi
               (fun i subtree ->
                 flatten_ptree (prefix ^ "." ^ string_of_int i) subtree)
               lst)
      | Kaun.Ptree.Record fields ->
          Kaun.Ptree.Record.fold
            (fun name subtree acc ->
              let new_prefix =
                if prefix = "" then name else prefix ^ "." ^ name
              in
              flatten_ptree new_prefix subtree @ acc)
            fields []
    in

    let flat_params = flatten_ptree "" hf_params in
    let embeddings_params = ref Kaun.Ptree.Record.empty in
    let decoder_layers = ref [] in
    let final_layer_norm_params = ref Kaun.Ptree.Record.empty in

    List.iter
      (fun (hf_name, tensor) ->
        match hf_name with
        (* Embeddings *)
        | s when String.starts_with ~prefix:"wte.weight" s ->
            embeddings_params :=
              Kaun.Ptree.Record.add "token_embeddings"
                (Kaun.Ptree.Tensor tensor) !embeddings_params
        | s when String.starts_with ~prefix:"wpe.weight" s ->
            embeddings_params :=
              Kaun.Ptree.Record.add "position_embeddings"
                (Kaun.Ptree.Tensor tensor) !embeddings_params
        (* Transformer blocks *)
        | s when String.starts_with ~prefix:"h." s -> (
            let rest = String.sub s 2 (String.length s - 2) in
            match String.split_on_char '.' rest with
            | layer_idx :: params -> (
                let layer_idx_int = int_of_string layer_idx in
                let param_name = String.concat "." params in

                (* Ensure we have enough layers *)
                while List.length !decoder_layers <= layer_idx_int do
                  decoder_layers :=
                    !decoder_layers @ [ ref Kaun.Ptree.Record.empty ]
                done;

                (* Get the layer params ref *)
                let layer_params = List.nth !decoder_layers layer_idx_int in

                (* Handle different parameter types *)
                match param_name with
                | "attn.c_attn.weight" ->
                    (* Keep combined QKV weight, will split after matmul *)
                    (* GPT-2 stores as [in_features, out_features] = [768, 2304] *)
                    layer_params :=
                      Kaun.Ptree.Record.add "qkv_weight"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "attn.c_attn.bias" ->
                    (* Keep combined QKV bias *)
                    layer_params :=
                      Kaun.Ptree.Record.add "qkv_bias"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "attn.c_proj.weight" ->
                    (* GPT-2 stores as [in, out] which is [768, 768] *)
                    (* We need [768, 768] for x[*,*,768] @ W -> [*,*,768] *)
                    (* So NO transpose needed! *)
                    layer_params :=
                      Kaun.Ptree.Record.add "attn_out_weight"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "attn.c_proj.bias" ->
                    layer_params :=
                      Kaun.Ptree.Record.add "attn_out_bias"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "mlp.c_fc.weight" ->
                    (* GPT-2 stores as [in, out] which is [768, 3072] *)
                    (* We need [768, 3072] for x[*,*,768] @ W -> [*,*,3072] *)
                    (* So NO transpose needed! *)
                    layer_params :=
                      Kaun.Ptree.Record.add "inter_weight"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "mlp.c_proj.weight" ->
                    (* GPT-2 stores as [in, out] which is [3072, 768] *)
                    (* We need [3072, 768] for h[*,*,3072] @ W -> [*,*,768] *)
                    (* So NO transpose needed! *)
                    layer_params :=
                      Kaun.Ptree.Record.add "out_weight"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "mlp.c_fc.bias" ->
                    layer_params :=
                      Kaun.Ptree.Record.add "inter_bias"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "mlp.c_proj.bias" ->
                    layer_params :=
                      Kaun.Ptree.Record.add "out_bias"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "ln_1.weight" ->
                    layer_params :=
                      Kaun.Ptree.Record.add "attn_gamma"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "ln_1.bias" ->
                    layer_params :=
                      Kaun.Ptree.Record.add "attn_beta"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "ln_2.weight" ->
                    layer_params :=
                      Kaun.Ptree.Record.add "ffn_gamma"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | "ln_2.bias" ->
                    layer_params :=
                      Kaun.Ptree.Record.add "ffn_beta"
                        (Kaun.Ptree.Tensor tensor) !layer_params
                | _ -> () (* Ignore other parameters like attn.bias *))
            | _ -> ())
        (* Final layer norm *)
        | s when String.starts_with ~prefix:"ln_f.weight" s ->
            final_layer_norm_params :=
              Kaun.Ptree.Record.add "gamma" (Kaun.Ptree.Tensor tensor)
                !final_layer_norm_params
        | s when String.starts_with ~prefix:"ln_f.bias" s ->
            final_layer_norm_params :=
              Kaun.Ptree.Record.add "beta" (Kaun.Ptree.Tensor tensor)
                !final_layer_norm_params
        | _ -> () (* Ignore other parameters *))
      flat_params;

    (* Build the final sequential structure *)
    let decoder_list =
      List.map (fun r -> Kaun.Ptree.Record !r) !decoder_layers
    in

    (* Add dropout placeholder for embeddings *)
    embeddings_params :=
      Kaun.Ptree.Record.add "dropout" (Kaun.Ptree.List []) !embeddings_params;

    (* Create sequential structure: embeddings, decoder layers, final layer
       norm *)
    Kaun.Ptree.List
      [
        Kaun.Ptree.Record !embeddings_params;
        Kaun.Ptree.List decoder_list;
        Kaun.Ptree.Record !final_layer_norm_params;
      ]
  in

  let mapped_params = map_huggingface_to_kaun hf_params in
  let model = create ~config:gpt2_config () in

  { model; params = mapped_params; config = gpt2_config; dtype }

let forward gpt2 inputs ?(training = false) ?(output_hidden_states = false)
    ?(output_attentions = false) () =
  let { model; params; _ } = gpt2 in
  let { input_ids; attention_mask = _; position_ids = _ } = inputs in

  (* GPT-2 forward pass using the Kaun model *)
  let open Rune in
  (* Get dtype from params to maintain polymorphism *)
  let dtype_tensor =
    match Kaun.Ptree.flatten_with_paths params with
    | [] -> failwith "Empty params"
    | (_, t) :: _ -> t
  in
  let target_dtype = dtype dtype_tensor in
  let float_input = cast target_dtype input_ids in

  (* Apply the model using Kaun *)
  let model_output = Kaun.apply model params ~training float_input in

  (* The model output is the final hidden state *)
  let last_hidden_state = model_output in

  (* For output_hidden_states and output_attentions, we would need to modify the
     model architecture to return intermediate values. For now, return minimal
     info *)
  let hidden_states =
    if output_hidden_states then Some [ last_hidden_state ] else None
  in
  let attentions = if output_attentions then None else None in

  (* Return structured output *)
  { last_hidden_state; hidden_states; attentions }

(* Language Modeling Head *)

module For_causal_lm = struct
  let create ?(config = default_config) () =
    let open Kaun.Layer in
    sequential
      [
        (* GPT-2 base model *)
        create ~config ();
        (* LM head: project to vocabulary *)
        linear ~in_features:config.n_embd ~out_features:config.vocab_size ();
      ]

  let forward ~model ~params ~input_ids ?attention_mask:_ ?position_ids:_
      ?labels ~training () =
    let open Rune in
    (* Get the dtype from params to maintain polymorphism *)
    let dtype_tensor =
      match Kaun.Ptree.flatten_with_paths params with
      | [] -> failwith "Empty params"
      | (_, t) :: _ -> t
    in
    let target_dtype = dtype dtype_tensor in
    let float_input = cast target_dtype input_ids in
    let hidden_states = Kaun.apply model params ~training float_input in

    (* Apply LM head: project hidden states to vocabulary GPT-2 uses weight
       tying, so we use the transposed token embeddings *)
    let logits =
      (* Try to find token embeddings in params structure *)
      match params with
      | Kaun.Ptree.List param_list -> (
          (* params is a list where first element is embeddings module *)
          match List.nth_opt param_list 0 with
          | Some (Kaun.Ptree.Record emb_fields) -> (
              match
                Kaun.Ptree.Record.find_opt "token_embeddings" emb_fields
              with
              | Some (Kaun.Ptree.Tensor wte) ->
                  (* Token embeddings have shape [vocab_size, hidden_size] We
                     need to use them as [hidden_size, vocab_size] for the LM
                     head *)
                  let wte_transposed = transpose ~axes:[ 1; 0 ] wte in
                  matmul hidden_states wte_transposed
              | _ -> hidden_states)
          | _ -> hidden_states)
      | _ -> hidden_states
    in

    (* Compute loss if labels provided *)
    let loss =
      match labels with
      | Some labels ->
          (* Shift labels for next token prediction *)
          let batch_size = (shape labels).(0) in
          let seq_length = (shape labels).(1) in
          let vocab_size = (shape logits).(2) in

          (* Shift logits and labels: predict next token *)
          let shift_logits = slice [ A; R (0, seq_length - 1); A ] logits in
          let shift_labels = slice [ A; R (1, seq_length); A ] labels in

          let flat_logits =
            Rune.reshape
              [| batch_size * (seq_length - 1); vocab_size |]
              shift_logits
          in
          let flat_labels =
            Rune.reshape [| batch_size * (seq_length - 1) |] shift_labels
          in
          Some
            (Kaun.Loss.softmax_cross_entropy_with_indices flat_logits
               flat_labels)
      | None -> None
    in

    (logits, loss)
end

(* Utilities *)

let parse_gpt2_config json =
  (* Parse GPT-2 specific configuration from HuggingFace JSON *)
  let open Yojson.Safe.Util in
  {
    vocab_size = json |> member "vocab_size" |> to_int;
    n_positions = json |> member "n_positions" |> to_int;
    n_embd = json |> member "n_embd" |> to_int;
    n_layer = json |> member "n_layer" |> to_int;
    n_head = json |> member "n_head" |> to_int;
    n_inner = json |> member "n_inner" |> to_int_option;
    activation_function =
      (match json |> member "activation_function" |> to_string_option with
      | Some "gelu_new" -> `gelu_new
      | Some "gelu" -> `gelu
      | Some "relu" -> `relu
      | Some "swish" | Some "silu" -> `swish
      | _ -> `gelu_new);
    resid_pdrop =
      json |> member "resid_pdrop" |> to_float_option
      |> Option.value ~default:0.1;
    embd_pdrop =
      json |> member "embd_pdrop" |> to_float_option
      |> Option.value ~default:0.1;
    attn_pdrop =
      json |> member "attn_pdrop" |> to_float_option
      |> Option.value ~default:0.1;
    layer_norm_epsilon =
      json
      |> member "layer_norm_epsilon"
      |> to_float_option |> Option.value ~default:1e-5;
    initializer_range =
      json |> member "initializer_range" |> to_float_option
      |> Option.value ~default:0.02;
    scale_attn_weights =
      json
      |> member "scale_attn_weights"
      |> to_bool_option |> Option.value ~default:true;
    use_cache =
      json |> member "use_cache" |> to_bool_option |> Option.value ~default:true;
    scale_attn_by_inverse_layer_idx =
      json
      |> member "scale_attn_by_inverse_layer_idx"
      |> to_bool_option
      |> Option.value ~default:false;
    reorder_and_upcast_attn =
      json
      |> member "reorder_and_upcast_attn"
      |> to_bool_option
      |> Option.value ~default:false;
    bos_token_id = json |> member "bos_token_id" |> to_int_option;
    eos_token_id = json |> member "eos_token_id" |> to_int_option;
    pad_token_id = json |> member "pad_token_id" |> to_int_option;
  }

let num_parameters params =
  let tensors = Kaun.Ptree.flatten_with_paths params in
  List.fold_left
    (fun acc (_, t) -> acc + Array.fold_left ( * ) 1 (Rune.shape t))
    0 tensors

let parameter_stats params =
  let total_params = num_parameters params in
  let total_bytes = total_params * 4 in
  (* Assuming float32 *)
  Printf.sprintf "GPT-2 parameters: %d (%.2f MB)" total_params
    (float_of_int total_bytes /. 1024. /. 1024.)

(* Common GPT-2 model configurations *)
let load_gpt2_small ~dtype () = from_pretrained ~model_id:"gpt2" ~dtype ()

let load_gpt2_medium ~dtype () =
  from_pretrained ~model_id:"gpt2-medium" ~dtype ()

let load_gpt2_large ~dtype () = from_pretrained ~model_id:"gpt2-large" ~dtype ()
let load_gpt2_xl ~dtype () = from_pretrained ~model_id:"gpt2-xl" ~dtype ()
