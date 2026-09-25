type 'a helper = { state : Nx.int64_t; w : 'a }

and 'a t = {
  helper : 'a helper;
  optional : 'a option;
  name : string; [@ptree.skip]
}

and state = { steps : Nx.int32_t } [@@deriving ptree]
