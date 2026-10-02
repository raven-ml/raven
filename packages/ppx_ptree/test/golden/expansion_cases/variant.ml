type 'a t =
  | Float of 'a
  | Mxfp4 of { blocks : Nx.uint8_t; scales : Nx.uint8_t }
  | Scaled of 'a * Nx.float32_t
  | Tied
[@@deriving ptree]
