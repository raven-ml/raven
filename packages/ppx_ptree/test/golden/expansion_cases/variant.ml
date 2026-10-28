type ('a, 'd) t =
  | Float of 'a
  | Mxfp4 of { blocks : 'd Nx.uint8_t; scales : 'd Nx.uint8_t }
  | Scaled of 'a * 'd Nx.float32_t
  | Tied
[@@deriving ptree]
