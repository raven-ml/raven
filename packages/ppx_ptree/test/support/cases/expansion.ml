type ('a, 'd) helper = { state : 'd Nx.int64_t; w : 'a }

and ('a, 'd) t = {
  helper : ('a, 'd) helper;
  optional : 'a option;
  name : string; [@ptree.skip]
}

and 'd state = { steps : 'd Nx.int32_t } [@@deriving ptree]
