open Rune

(* Configuration *)

type config = {
  vocab_size : int;
  hidden_size : int;
  num_hidden_layers : int;
  num_attention_heads : int;
  intermediate_size : int;
  hidden_act : [ `gelu | `relu | `swish | `gelu_new ];
  hidden_dropout_prob : float;
  attention_probs_dropout_prob : float;
  max_position_embeddings : int;
  type_vocab_size : int;
  layer_norm_eps : float;
  pad_token_id : int;
  position_embedding_type : [ `absolute | `relative ];
  use_cache : bool;
  classifier_dropout : float option;
}

let default_config =
  {
    vocab_size = 30522;
    hidden_size = 768;
    num_hidden_layers = 12;
    num_attention_heads = 12;
    intermediate_size = 3072;
    hidden_act = `gelu;
    hidden_dropout_prob = 0.1;
    attention_probs_dropout_prob = 0.1;
    max_position_embeddings = 512;
    type_vocab_size = 2;
    layer_norm_eps = 1e-12;
    pad_token_id = 0;
    position_embedding_type = `absolute;
    use_cache = true;
    classifier_dropout = None;
  }

let bert_base_uncased = default_config

let bert_large_uncased =
  {
    default_config with
    hidden_size = 1024;
    num_hidden_layers = 24;
    num_attention_heads = 16;
    intermediate_size = 4096;
  }

let bert_base_cased =
  default_config (* Same architecture, different tokenizer *)

let bert_base_multilingual =
  {
    default_config with
    vocab_size = 105879;
    (* Larger vocabulary for multilingual *)
  }

(* Move inputs type definition before Tokenizer *)
type inputs = {
  input_ids : (int32, int32_elt) Rune.t;
  attention_mask : (int32, int32_elt) Rune.t;
  token_type_ids : (int32, int32_elt) Rune.t option;
  position_ids : (int32, int32_elt) Rune.t option;
}

module Tokenizer = struct
  (* Simple BERT WordPiece tokenizer implementation *)

  type t = {
    vocab : (string, int) Hashtbl.t;
    inv_vocab : (int, string) Hashtbl.t;
    unk_token_id : int;
    cls_token_id : int;
    sep_token_id : int;
    pad_token_id : int;
    max_input_chars_per_word : int;
  }

  let download_vocab_file model_id =
    (* Download vocab file from HuggingFace if not present *)
    let cache_dir =
      match Sys.getenv_opt "XDG_CACHE_HOME" with
      | Some dir -> dir
      | None -> (
          match Sys.getenv_opt "HOME" with
          | Some home -> Filename.concat home ".cache"
          | None -> "/tmp/.cache")
    in
    let kaun_cache = Filename.concat cache_dir "kaun" in
    let vocab_cache = Filename.concat kaun_cache "vocab" in
    let vocab_file = Filename.concat vocab_cache (model_id ^ "-vocab.txt") in

    (* Create cache directory if it doesn't exist *)
    if not (Sys.file_exists vocab_cache) then
      Sys.command (Printf.sprintf "mkdir -p %s" vocab_cache) |> ignore;

    (* Download if file doesn't exist *)
    if not (Sys.file_exists vocab_file) then (
      Printf.printf "Downloading vocab file for %s...\n%!" model_id;
      let url =
        Printf.sprintf "https://huggingface.co/%s/resolve/main/vocab.txt"
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
          (Printf.sprintf "Failed to download vocab file for %s" model_id));
    vocab_file

  let load_vocab vocab_file =
    let vocab = Hashtbl.create 30000 in
    let inv_vocab = Hashtbl.create 30000 in
    let ic = open_in vocab_file in
    let idx = ref 0 in
    (try
       while true do
         let line = input_line ic in
         let token = String.trim line in
         if String.length token > 0 then (
           Hashtbl.add vocab token !idx;
           Hashtbl.add inv_vocab !idx token;
           incr idx)
       done
     with End_of_file -> ());
    close_in ic;
    (vocab, inv_vocab)

  let create ?vocab_file ?model_id () =
    (* Either provide a vocab_file path or a model_id to download from *)
    let vocab_file =
      match (vocab_file, model_id) with
      | Some file, _ -> file
      | None, Some id -> download_vocab_file id
      | None, None -> download_vocab_file "bert-base-uncased" (* Default *)
    in
    let vocab, inv_vocab = load_vocab vocab_file in
    {
      vocab;
      inv_vocab;
      unk_token_id = 100;
      (* [UNK] is at index 100 in BERT vocab *)
      cls_token_id = 101;
      (* [CLS] *)
      sep_token_id = 102;
      (* [SEP] *)
      pad_token_id = 0;
      (* [PAD] *)
      max_input_chars_per_word = 100;
    }

  (* Basic tokenization: lowercase and split on whitespace/punctuation *)
  let basic_tokenize text =
    let text = String.lowercase_ascii text in
    let tokens = ref [] in
    let current = Buffer.create 16 in

    String.iter
      (fun c ->
        match c with
        | 'a' .. 'z' | '0' .. '9' -> Buffer.add_char current c
        | _ ->
            if Buffer.length current > 0 then (
              tokens := Buffer.contents current :: !tokens;
              Buffer.clear current);
            (* Don't add whitespace as tokens *)
            if c <> ' ' && c <> '\t' && c <> '\n' && c <> '\r' then
              tokens := String.make 1 c :: !tokens)
      text;

    if Buffer.length current > 0 then
      tokens := Buffer.contents current :: !tokens;

    List.rev !tokens

  (* WordPiece tokenization on a single word *)
  let wordpiece_tokenize_word t word =
    let n = String.length word in
    if n > t.max_input_chars_per_word then [ t.unk_token_id ]
    else
      let output = ref [] in
      let start = ref 0 in
      while !start < n do
        let end_idx = ref n in
        let found = ref false in
        while !start < !end_idx && not !found do
          let substr =
            if !start > 0 then "##" ^ String.sub word !start (!end_idx - !start)
            else String.sub word !start (!end_idx - !start)
          in
          match Hashtbl.find_opt t.vocab substr with
          | Some token_id ->
              output := token_id :: !output;
              found := true;
              start := !end_idx
          | None -> decr end_idx
        done;
        if not !found then (
          output := [ t.unk_token_id ];
          start := n (* Exit loop *))
      done;
      List.rev !output

  let encode_to_array t text =
    let basic_tokens = basic_tokenize text in
    let token_ids = ref [ t.cls_token_id ] in

    List.iter
      (fun word ->
        let word_tokens = wordpiece_tokenize_word t word in
        token_ids := !token_ids @ word_tokens)
      basic_tokens;

    token_ids := !token_ids @ [ t.sep_token_id ];
    Array.of_list !token_ids

  let encode t text =
    let token_ids = encode_to_array t text in
    let seq_len = Array.length token_ids in

    (* Convert to tensors *)
    let input_ids =
      Rune.create Int32 [| 1; seq_len |] (Array.map Int32.of_int token_ids)
    in
    let attention_mask = Rune.ones Int32 [| 1; seq_len |] in

    (* Return inputs record *)
    { input_ids; attention_mask; token_type_ids = None; position_ids = None }

  let encode_batch t ?(max_length = 512) ?(padding = true) texts =
    let encoded = List.map (encode_to_array t) texts in

    (* Find max length or use specified max_length *)
    let actual_max =
      if padding then max_length
      else
        List.fold_left (fun acc arr -> Int.max acc (Array.length arr)) 0 encoded
    in

    (* Pad sequences *)
    let padded =
      List.map
        (fun arr ->
          let len = Array.length arr in
          if len >= actual_max then Array.sub arr 0 actual_max
          else Array.append arr (Array.make (actual_max - len) t.pad_token_id))
        encoded
    in

    (* Convert to tensor *)
    let batch_size = List.length padded in
    let flat_data = Array.concat padded in
    (* Create Nx tensor with int values *)
    let nx_tensor =
      let data = Array.map Int32.of_int flat_data in
      Nx.create Int32 [| batch_size; max_length |] data
    in
    (* Convert to Rune tensor *)
    Rune.of_nx nx_tensor

  let decode t token_ids =
    let tokens = ref [] in
    Array.iter
      (fun id ->
        if id <> t.cls_token_id && id <> t.sep_token_id && id <> t.pad_token_id
        then
          match Hashtbl.find_opt t.inv_vocab id with
          | Some token ->
              (* Remove ## prefix for subword tokens *)
              let token =
                if String.length token > 2 && String.sub token 0 2 = "##" then
                  String.sub token 2 (String.length token - 2)
                else token ^ " " (* Add space after whole words *)
              in
              tokens := token :: !tokens
          | None -> ())
      token_ids;
    String.concat "" (List.rev !tokens) |> String.trim

  let create_wordpiece ?vocab_file ?model_id () =
    create ?vocab_file ?model_id ()
end

let embeddings ~config () =
  let open Kaun.Layer in
  (* BERT embeddings: token + position + token_type *)
  (* We'll create a custom module that combines them *)
  let token_embeddings =
    embedding ~vocab_size:config.vocab_size ~embed_dim:config.hidden_size ()
  in
  let position_embeddings =
    embedding ~vocab_size:config.max_position_embeddings
      ~embed_dim:config.hidden_size ()
  in
  let token_type_embeddings =
    embedding ~vocab_size:config.type_vocab_size ~embed_dim:config.hidden_size
      ()
  in
  let layer_norm =
    layer_norm ~dim:config.hidden_size ~eps:config.layer_norm_eps ()
  in
  let dropout = dropout ~rate:config.hidden_dropout_prob () in

  (* Custom module that applies all embeddings and sums them *)
  {
    Kaun.init =
      (fun ~rngs ~dtype ->
        let keys = Rune.Rng.split ~n:5 rngs in
        Kaun.Ptree.record_of
          [
            ("token_embeddings", token_embeddings.init ~rngs:keys.(0) ~dtype);
            ( "position_embeddings",
              position_embeddings.init ~rngs:keys.(1) ~dtype );
            ( "token_type_embeddings",
              token_type_embeddings.init ~rngs:keys.(2) ~dtype );
            ("layer_norm", layer_norm.init ~rngs:keys.(3) ~dtype);
            ("dropout", dropout.init ~rngs:keys.(4) ~dtype);
          ]);
    Kaun.apply =
      (fun params ~training ?rngs x ->
        (* x is expected to be float tensor for compatibility with Kaun modules *)
        (* Cast back to int32 for embedding lookups *)
        let input_ids = Rune.cast Rune.int32 x in
        match params with
        | Kaun.Ptree.Record fields ->
            (* Get each embedding layer's params *)
            let get_params name =
              match Kaun.Ptree.Record.find_opt name fields with
              | Some p -> p
              | None -> failwith ("Embeddings: missing " ^ name)
            in

            (* Manually perform embedding lookups since Module.apply expects float tensors *)
            (* Extract embedding tables from params *)
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
            let token_type_embeddings_table =
              get_embedding_table "token_type_embeddings"
            in

            (* Perform embedding lookups using Rune.take for
               differentiability *)
            let lookup_embeddings embedding_table indices =
              let batch_size = (Rune.shape indices).(0) in
              let seq_len = (Rune.shape indices).(1) in
              let embed_dim = (Rune.shape embedding_table).(1) in

              (* Flatten indices for take operation *)
              let indices_flat =
                Rune.reshape [| batch_size * seq_len |] indices
              in

              (* Use take to gather embeddings *)
              let gathered = Rune.take ~axis:0 indices_flat embedding_table in

              (* Reshape to [batch_size, seq_len, embed_dim] *)
              Rune.reshape [| batch_size; seq_len; embed_dim |] gathered
            in

            (* Apply token embeddings *)
            let token_embeds =
              lookup_embeddings token_embeddings_table input_ids
            in

            (* Create position ids if not provided: [0, 1, 2, ..., seq_len-1] *)
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

            (* Create token type ids (all zeros for single sentence) *)
            let token_type_ids = Rune.zeros Rune.int32 (Rune.shape input_ids) in
            let token_type_embeds =
              lookup_embeddings token_type_embeddings_table token_type_ids
            in

            (* Sum all embeddings *)
            let embeddings =
              Rune.add token_embeds (Rune.add position_embeds token_type_embeds)
            in

            (* Apply layer norm and dropout *)
            let embeddings =
              layer_norm.apply (get_params "layer_norm") ~training ?rngs
                embeddings
            in
            let embeddings =
              dropout.apply (get_params "dropout") ~training ?rngs embeddings
            in

            embeddings
        | _ -> failwith "Embeddings: invalid params");
  }

let pooler ~hidden_size () =
  (* Create a module that extracts CLS token and applies dense + tanh *)
  {
    Kaun.init =
      (fun ~rngs ~dtype ->
        (* Initialize dense layer weights *)
        let key = Rune.Rng.to_int rngs in
        let init_fn = (Kaun.Initializers.normal ~stddev:0.02 ()).f in
        let dense_weight = init_fn key [| hidden_size; hidden_size |] dtype in
        let dense_bias = Rune.zeros dtype [| hidden_size |] in
        Kaun.Ptree.(
          Record
            (Record.of_seq
            @@ List.to_seq
                 [
                   ("dense_weight", Tensor dense_weight);
                   ("dense_bias", Tensor dense_bias);
                 ])));
    Kaun.apply =
      (fun params ~training:_ ?rngs:_ x ->
        let open Rune in
        (* Extract CLS token: hidden_states[:, 0, :] *)
        let cls_token = slice [ A; I 0; A ] x in

        (* Apply dense layer *)
        let dense_weight =
          match params with
          | Record fields -> (
              match Kaun.Ptree.Record.find_opt "dense_weight" fields with
              | Some (Tensor t) -> t
              | _ -> failwith "Pooler: missing dense_weight")
          | _ -> failwith "Pooler: invalid params"
        in
        let dense_bias =
          match params with
          | Record fields -> (
              match Kaun.Ptree.Record.find_opt "dense_bias" fields with
              | Some (Tensor t) -> t
              | _ -> failwith "Pooler: missing dense_bias")
          | _ -> failwith "Pooler: invalid params"
        in
        let pooled = add (matmul cls_token dense_weight) dense_bias in

        (* Apply tanh activation *)
        tanh pooled);
  }

(* Main Model *)

type 'a bert = {
  model : Kaun.Layer.module_;
  params : 'a Kaun.Ptree.t;
  config : config;
  dtype : (float, 'a) Rune.dtype;
}

type 'a output = {
  last_hidden_state : (float, 'a) Rune.t;
  pooler_output : (float, 'a) Rune.t option;
  hidden_states : (float, 'a) Rune.t list option;
  attentions : (float, 'a) Rune.t list option;
}

let create_bert_layers ~config ~add_pooling_layer =
  (* Build BERT as a sequential model *)
  let layers =
    [
      (* Use the proper BERT embeddings module that combines token + position +
         segment *)
      embeddings ~config ();
      (* Use the transformer encoder from Kaun *)
      Kaun.Layer.transformer_encoder ~num_layers:config.num_hidden_layers
        ~hidden_size:config.hidden_size
        ~num_attention_heads:config.num_attention_heads
        ~intermediate_size:config.intermediate_size
        ~hidden_dropout_prob:config.hidden_dropout_prob
        ~attention_probs_dropout_prob:config.attention_probs_dropout_prob
        ~layer_norm_eps:config.layer_norm_eps
        ~hidden_act:
          (match config.hidden_act with
          | `gelu | `gelu_new -> `gelu
          | `relu -> `relu
          | `swish -> `swish)
        ~use_bias:true (* BERT uses bias *)
        ();
    ]
    @
    (* Optional pooler for [CLS] token *)
    if add_pooling_layer then [ pooler ~hidden_size:config.hidden_size () ]
    else []
  in
  layers

let create ?(config = default_config) ?(add_pooling_layer = true) () =
  let open Kaun.Layer in
  sequential (create_bert_layers ~config ~add_pooling_layer)

let from_pretrained ?(model_id = "bert-base-uncased") ?revision ?cache_config
    ~dtype () =
  (* Load config and weights from HuggingFace, but handle BERT-specific
     conversion here *)
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

  (* Parse BERT-specific config *)
  let bert_config =
    let open Yojson.Safe.Util in
    {
      vocab_size = config_json |> member "vocab_size" |> to_int;
      hidden_size = config_json |> member "hidden_size" |> to_int;
      num_hidden_layers = config_json |> member "num_hidden_layers" |> to_int;
      num_attention_heads =
        config_json |> member "num_attention_heads" |> to_int;
      intermediate_size = config_json |> member "intermediate_size" |> to_int;
      hidden_act =
        (match config_json |> member "hidden_act" |> to_string_option with
        | Some "gelu" | Some "gelu_new" -> `gelu
        | Some "relu" -> `relu
        | Some "swish" | Some "silu" -> `swish
        | _ -> `gelu);
      hidden_dropout_prob =
        config_json
        |> member "hidden_dropout_prob"
        |> to_float_option |> Option.value ~default:0.1;
      attention_probs_dropout_prob =
        config_json
        |> member "attention_probs_dropout_prob"
        |> to_float_option |> Option.value ~default:0.1;
      max_position_embeddings =
        config_json |> member "max_position_embeddings" |> to_int;
      type_vocab_size =
        config_json |> member "type_vocab_size" |> to_int_option
        |> Option.value ~default:2;
      layer_norm_eps =
        config_json |> member "layer_norm_eps" |> to_float_option
        |> Option.value ~default:1e-12;
      pad_token_id = 0;
      position_embedding_type = `absolute;
      use_cache = true;
      classifier_dropout = None;
    }
  in

  (* Load weights using HuggingFace infrastructure *)
  let hf_params =
    Kaun_huggingface.from_pretrained ~config:cache_config ~revision ~model_id
      ~dtype ()
  in

  (* Map HuggingFace parameter names to our expected structure *)
  let map_huggingface_to_kaun hf_params =
    (* First, flatten the nested HuggingFace structure to get dot-separated
       names *)
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

    (* Flatten the HuggingFace parameters *)
    let flat_params = flatten_ptree "" hf_params in

    (* Map flat HF names to Kaun sequential structure *)
    let embeddings_params = ref Kaun.Ptree.Record.empty in
    let encoder_layers = ref [] in
    let pooler_params = ref Kaun.Ptree.Record.empty in

    List.iter
      (fun (hf_name, tensor) ->
        (* Strip "bert." prefix if present *)
        let name =
          if String.starts_with ~prefix:"bert." hf_name then
            String.sub hf_name 5 (String.length hf_name - 5)
          else hf_name
        in

        (* Map based on the name structure *)
        match name with
        (* Embeddings *)
        | s
          when String.starts_with ~prefix:"embeddings.word_embeddings.weight" s
          ->
            embeddings_params :=
              Kaun.Ptree.Record.add "token_embeddings"
                (Kaun.Ptree.Tensor tensor) !embeddings_params
        | s
          when String.starts_with
                 ~prefix:"embeddings.position_embeddings.weight" s ->
            embeddings_params :=
              Kaun.Ptree.Record.add "position_embeddings"
                (Kaun.Ptree.Tensor tensor) !embeddings_params
        | s
          when String.starts_with
                 ~prefix:"embeddings.token_type_embeddings.weight" s ->
            embeddings_params :=
              Kaun.Ptree.Record.add "token_type_embeddings"
                (Kaun.Ptree.Tensor tensor) !embeddings_params
        | s when String.starts_with ~prefix:"embeddings.LayerNorm" s ->
            let ln_params =
              match
                Kaun.Ptree.Record.find_opt "layer_norm" !embeddings_params
              with
              | Some (Kaun.Ptree.Record r) -> r
              | _ -> Kaun.Ptree.Record.empty
            in
            let field =
              if String.ends_with ~suffix:"weight" s then "gamma"
              else if String.ends_with ~suffix:"bias" s then "beta"
              else if String.ends_with ~suffix:"gamma" s then "gamma"
              else "beta"
            in
            let updated_ln =
              Kaun.Ptree.Record.add field (Kaun.Ptree.Tensor tensor) ln_params
            in
            embeddings_params :=
              Kaun.Ptree.Record.add "layer_norm" (Kaun.Ptree.Record updated_ln)
                !embeddings_params
        (* Encoder layers *)
        | s when String.starts_with ~prefix:"encoder.layer." s -> (
            let rest = String.sub s 14 (String.length s - 14) in
            match String.split_on_char '.' rest with
            | layer_idx :: params ->
                let layer_idx_int = int_of_string layer_idx in
                let param_name = String.concat "." params in

                (* Map HF param names to Kaun param names *)
                (* HuggingFace stores weights transposed, so we need to transpose them *)
                let kaun_param, needs_transpose =
                  match param_name with
                  | "attention.self.query.weight" -> ("q_weight", true)
                  | "attention.self.key.weight" -> ("k_weight", true)
                  | "attention.self.value.weight" -> ("v_weight", true)
                  | "attention.output.dense.weight" -> ("attn_out_weight", true)
                  | "intermediate.dense.weight" -> ("inter_weight", true)
                  | "output.dense.weight" -> ("out_weight", true)
                  | "attention.self.query.bias" -> ("q_bias", false)
                  | "attention.self.key.bias" -> ("k_bias", false)
                  | "attention.self.value.bias" -> ("v_bias", false)
                  | "attention.output.dense.bias" -> ("attn_out_bias", false)
                  | "intermediate.dense.bias" -> ("inter_bias", false)
                  | "output.dense.bias" -> ("out_bias", false)
                  | "attention.output.LayerNorm.weight" -> ("attn_gamma", false)
                  | "attention.output.LayerNorm.bias" -> ("attn_beta", false)
                  | "output.LayerNorm.weight" -> ("ffn_gamma", false)
                  | "output.LayerNorm.bias" -> ("ffn_beta", false)
                  | "attention.output.LayerNorm.gamma" -> ("attn_gamma", false)
                  | "attention.output.LayerNorm.beta" -> ("attn_beta", false)
                  | "output.LayerNorm.gamma" -> ("ffn_gamma", false)
                  | "output.LayerNorm.beta" -> ("ffn_beta", false)
                  | _ -> (param_name, false)
                in

                (* Ensure we have enough layers *)
                while List.length !encoder_layers <= layer_idx_int do
                  encoder_layers :=
                    !encoder_layers @ [ ref Kaun.Ptree.Record.empty ]
                done;

                (* Add param to the appropriate layer *)
                let layer_params = List.nth !encoder_layers layer_idx_int in
                (* Transpose weight matrices if needed (HuggingFace stores them
                   transposed) *)
                let final_tensor =
                  if needs_transpose then Rune.transpose tensor ~axes:[| 1; 0 |]
                  else tensor
                in
                layer_params :=
                  Kaun.Ptree.Record.add kaun_param
                    (Kaun.Ptree.Tensor final_tensor) !layer_params
            | _ -> ())
        (* Pooler *)
        | s when String.starts_with ~prefix:"pooler.dense.weight" s ->
            (* Transpose the pooler weight too (HuggingFace stores it
               transposed) *)
            let transposed_tensor = Rune.transpose tensor ~axes:[| 1; 0 |] in
            pooler_params :=
              Kaun.Ptree.Record.add "dense_weight"
                (Kaun.Ptree.Tensor transposed_tensor) !pooler_params
        | s when String.starts_with ~prefix:"pooler.dense.bias" s ->
            pooler_params :=
              Kaun.Ptree.Record.add "dense_bias" (Kaun.Ptree.Tensor tensor)
                !pooler_params
        | _ -> () (* Ignore other parameters *))
      flat_params;

    (* Build the final sequential structure *)
    let encoder_list =
      List.map (fun r -> Kaun.Ptree.Record !r) !encoder_layers
    in

    (* Add dropout placeholder for embeddings *)
    embeddings_params :=
      Kaun.Ptree.Record.add "dropout" (Kaun.Ptree.List []) !embeddings_params;

    (* Create a structure with both encoder params and pooler params *)
    (* The encoder params are for the sequential model *)
    let encoder_params =
      Kaun.Ptree.List
        [ Kaun.Ptree.Record !embeddings_params; Kaun.Ptree.List encoder_list ]
    in

    (* Return a record with both encoder and pooler params *)
    Kaun.Ptree.record_of
      [
        ("encoder", encoder_params); ("pooler", Kaun.Ptree.Record !pooler_params);
      ]
  in

  let mapped_params = map_huggingface_to_kaun hf_params in

  (* Create the bert model structure without pooler since we handle it
     separately *)
  let model = create ~config:bert_config ~add_pooling_layer:false () in

  { model; params = mapped_params; config = bert_config; dtype }

let forward bert inputs ?(training = false) ?(output_hidden_states = false)
    ?(output_attentions = false) () =
  let { model; params; _ } = bert in
  let { input_ids; attention_mask; token_type_ids; position_ids = _ } =
    inputs
  in
  (* BERT forward pass using the Kaun model properly *)
  let open Rune in
  (* Get batch size and seq length *)
  let input_shape = shape input_ids in
  let batch_size = input_shape.(0) in
  let seq_len = input_shape.(1) in

  (* Default token_type_ids to zeros if not provided *)
  let _token_type_ids =
    match token_type_ids with
    | Some ids -> ids
    | None -> zeros int32 [| batch_size; seq_len |]
  in

  (* TODO: Handle attention_mask properly when Kaun supports it *)
  let _ = attention_mask in

  (* Extract encoder and pooler params using path-based access *)
  let encoder_params =
    try Kaun.Ptree.get_by_path "encoder" params
    with Invalid_argument _ -> failwith "forward: missing encoder params"
  in
  let pooler_params =
    try Kaun.Ptree.get_by_path "pooler" params
    with Invalid_argument _ -> Kaun.Ptree.Record Kaun.Ptree.Record.empty
  in

  (* Convert input_ids to float for Kaun model *)
  let dtype_tensor =
    match Kaun.Ptree.flatten_with_paths encoder_params with
    | [] -> failwith "Empty encoder params"
    | (_, t) :: _ -> t
  in
  let target_dtype = dtype dtype_tensor in
  let float_input = cast target_dtype input_ids in

  (* Apply the model using Kaun - this handles embeddings and encoder layers *)
  (* The model should be created without pooler since we handle it separately *)
  let model_output = Kaun.apply model encoder_params ~training float_input in

  (* The model output is the encoder's last hidden state *)
  let last_hidden_state = model_output in

  (* For output_hidden_states and output_attentions, we would need to modify the
     model architecture to return intermediate values. For now, return minimal
     info *)
  let hidden_states =
    if output_hidden_states then Some [ last_hidden_state ] else None
  in
  let attentions =
    if output_attentions then None
      (* Our current Kaun encoder doesn't expose attention weights *)
    else None
  in

  (* Apply pooler if present in parameters *)
  let pooler_output =
    match pooler_params with
    | Kaun.Ptree.Record fields when Kaun.Ptree.Record.cardinal fields > 0 ->
        (* Get pooler weights using path access *)
        let pooler_weight =
          match
            Kaun.Ptree.get_by_path "dense_weight" pooler_params
            |> Kaun.Ptree.get_tensor
          with
          | Some t -> t
          | None -> failwith "Pooler: missing dense_weight"
        in
        let pooler_bias =
          match
            Kaun.Ptree.get_by_path "dense_bias" pooler_params
            |> Kaun.Ptree.get_tensor
          with
          | Some t -> t
          | None -> failwith "Pooler: missing dense_bias"
        in

        (* Extract CLS token from encoder output *)
        let cls_token = slice [ A; I 0; A ] last_hidden_state in

        (* Apply dense + tanh - weights are already in correct shape after
           loading *)
        let pooled = add (matmul cls_token pooler_weight) pooler_bias in
        Some (tanh pooled)
    | _ -> None
  in

  (* Return structured output *)
  { last_hidden_state; pooler_output; hidden_states; attentions }

(* Task-Specific Heads *)

module For_masked_lm = struct
  let create ?(config = default_config) () =
    let open Kaun.Layer in
    sequential
      [
        (* BERT encoder *)
        create ~config ~add_pooling_layer:false ();
        (* MLM head: project back to vocabulary *)
        linear ~in_features:config.hidden_size ~out_features:config.hidden_size
          ();
        gelu ();
        layer_norm ~dim:config.hidden_size ~eps:config.layer_norm_eps ();
        linear ~in_features:config.hidden_size ~out_features:config.vocab_size
          ();
      ]

  let forward ~model ~params ~input_ids ?attention_mask:_ ?token_type_ids:_
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
    let logits = Kaun.apply model params ~training float_input in

    (* Compute loss if labels provided *)
    let loss =
      match labels with
      | Some labels ->
          (* Reshape for cross-entropy *)
          let batch_size = (shape logits).(0) in
          let seq_length = (shape logits).(1) in
          let vocab_size = (shape logits).(2) in
          let flat_logits =
            Rune.reshape [| batch_size * seq_length; vocab_size |] logits
          in
          let flat_labels = Rune.reshape [| batch_size * seq_length |] labels in
          Some
            (Kaun.Loss.softmax_cross_entropy_with_indices flat_logits
               flat_labels)
      | None -> None
    in

    (logits, loss)
end

module For_sequence_classification = struct
  let create ?(config = default_config) ~num_labels () =
    let open Kaun.Layer in
    sequential
      [
        (* BERT encoder with pooler *)
        create ~config ~add_pooling_layer:true ();
        (* Classification head *)
        dropout
          ~rate:
            (Option.value config.classifier_dropout
               ~default:config.hidden_dropout_prob)
          ();
        linear ~in_features:config.hidden_size ~out_features:num_labels ();
      ]

  let forward ~model ~params ~input_ids ?attention_mask:_ ?token_type_ids:_
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
    let logits = Kaun.apply model params ~training float_input in

    (* Compute loss if labels provided *)
    let loss =
      match labels with
      | Some labels ->
          Some (Kaun.Loss.softmax_cross_entropy_with_indices logits labels)
      | None -> None
    in

    (logits, loss)
end

module For_token_classification = struct
  let create ?(config = default_config) ~num_labels () =
    let open Kaun.Layer in
    sequential
      [
        (* BERT encoder without pooler *)
        create ~config ~add_pooling_layer:false ();
        (* Token classification head *)
        dropout ~rate:config.hidden_dropout_prob ();
        linear ~in_features:config.hidden_size ~out_features:num_labels ();
      ]

  let forward ~model ~params ~input_ids ?attention_mask:_ ?token_type_ids:_
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
    let logits = Kaun.apply model params ~training float_input in

    (* Compute loss if labels provided *)
    let loss =
      match labels with
      | Some labels ->
          (* Reshape for cross-entropy *)
          let batch_size = (shape logits).(0) in
          let seq_length = (shape logits).(1) in
          let num_labels = (shape logits).(2) in
          let flat_logits =
            Rune.reshape [| batch_size * seq_length; num_labels |] logits
          in
          let flat_labels = Rune.reshape [| batch_size * seq_length |] labels in
          Some
            (Kaun.Loss.softmax_cross_entropy_with_indices flat_logits
               flat_labels)
      | None -> None
    in

    (logits, loss)
end

(* BERT-specific configurations *)

let parse_bert_config json =
  (* Parse BERT-specific configuration from HuggingFace JSON *)
  let open Yojson.Safe.Util in
  {
    vocab_size = json |> member "vocab_size" |> to_int;
    hidden_size = json |> member "hidden_size" |> to_int;
    num_hidden_layers = json |> member "num_hidden_layers" |> to_int;
    num_attention_heads = json |> member "num_attention_heads" |> to_int;
    intermediate_size = json |> member "intermediate_size" |> to_int;
    hidden_act =
      (match json |> member "hidden_act" |> to_string_option with
      | Some "gelu" | Some "gelu_new" -> `gelu
      | Some "relu" -> `relu
      | Some "swish" | Some "silu" -> `swish
      | _ -> `gelu);
    hidden_dropout_prob =
      json
      |> member "hidden_dropout_prob"
      |> to_float_option |> Option.value ~default:0.1;
    attention_probs_dropout_prob =
      json
      |> member "attention_probs_dropout_prob"
      |> to_float_option |> Option.value ~default:0.1;
    max_position_embeddings = json |> member "max_position_embeddings" |> to_int;
    type_vocab_size =
      json |> member "type_vocab_size" |> to_int_option
      |> Option.value ~default:2;
    layer_norm_eps =
      json |> member "layer_norm_eps" |> to_float_option
      |> Option.value ~default:1e-12;
    pad_token_id = 0;
    position_embedding_type = `absolute;
    use_cache = true;
    classifier_dropout = None;
  }

(* Utilities *)

let create_attention_mask (type a) ~(input_ids : (int32, int32_elt) Rune.t)
    ~pad_token_id ~(dtype : (float, a) dtype) : (float, a) Rune.t =
  (* Create mask where 1.0 for real tokens, 0.0 for padding *)
  let input_dtype = Rune.dtype input_ids in
  let pad_tensor = Rune.scalar input_dtype (Int32.of_int pad_token_id) in
  let mask = Rune.not_equal input_ids pad_tensor in
  (* Cast to the requested float dtype *)
  Rune.cast dtype mask

let get_embeddings ~model:_ ~params:_ ~input_ids:_ ?attention_mask:_
    ~layer_index:_ () =
  (* Would extract embeddings from specific layer *)
  failwith "get_embeddings not fully implemented"

let num_parameters params =
  let tensors = Kaun.Ptree.flatten_with_paths params in
  List.fold_left
    (fun acc (_, t) -> acc + Array.fold_left ( * ) 1 (Rune.shape t))
    0 tensors

let parameter_stats params =
  let total_params = num_parameters params in
  let total_bytes = total_params * 4 in
  (* Assuming float32 *)
  Printf.sprintf "BERT parameters: %d (%.2f MB)" total_params
    (float_of_int total_bytes /. 1024. /. 1024.)

(* Common BERT model configurations *)
let load_bert_base_uncased ~dtype () =
  from_pretrained ~model_id:"bert-base-uncased" ~dtype ()

let load_bert_large_uncased ~dtype () =
  from_pretrained ~model_id:"bert-large-uncased" ~dtype ()

let load_bert_base_cased ~dtype () =
  from_pretrained ~model_id:"bert-base-cased" ~dtype ()

let load_bert_base_multilingual_cased ~dtype () =
  from_pretrained ~model_id:"bert-base-multilingual-cased" ~dtype ()
