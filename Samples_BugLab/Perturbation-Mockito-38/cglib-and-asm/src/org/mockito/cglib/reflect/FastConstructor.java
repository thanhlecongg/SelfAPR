[buglab_swap_variables]^super ( constructor, fc, fc.getIndex ( constructor.getParameterTypes (  )  )  ) ;^24^^^^^23^25^super ( fc, constructor, fc.getIndex ( constructor.getParameterTypes (  )  )  ) ;^[CLASS] FastConstructor  [METHOD] <init> [RETURN_TYPE] Constructor)   FastClass fc Constructor constructor [VARIABLES] boolean  FastClass  fc  Constructor  constructor  
[buglab_swap_variables]^super (  constructor, fc.getIndex ( constructor.getParameterTypes (  )  )  ) ;^24^^^^^23^25^super ( fc, constructor, fc.getIndex ( constructor.getParameterTypes (  )  )  ) ;^[CLASS] FastConstructor  [METHOD] <init> [RETURN_TYPE] Constructor)   FastClass fc Constructor constructor [VARIABLES] boolean  FastClass  fc  Constructor  constructor  
[buglab_swap_variables]^super ( fc,  fc.getIndex ( constructor.getParameterTypes (  )  )  ) ;^24^^^^^23^25^super ( fc, constructor, fc.getIndex ( constructor.getParameterTypes (  )  )  ) ;^[CLASS] FastConstructor  [METHOD] <init> [RETURN_TYPE] Constructor)   FastClass fc Constructor constructor [VARIABLES] boolean  FastClass  fc  Constructor  constructor  
[buglab_swap_variables]^return index.newInstance ( fc, null ) ;^36^^^^^35^37^return fc.newInstance ( index, null ) ;^[CLASS] FastConstructor  [METHOD] newInstance [RETURN_TYPE] Object   [VARIABLES] boolean  
[buglab_swap_variables]^return fc.newInstance (  null ) ;^36^^^^^35^37^return fc.newInstance ( index, null ) ;^[CLASS] FastConstructor  [METHOD] newInstance [RETURN_TYPE] Object   [VARIABLES] boolean  
[buglab_swap_variables]^return args.newInstance ( index, fc ) ;^40^^^^^39^41^return fc.newInstance ( index, args ) ;^[CLASS] FastConstructor  [METHOD] newInstance [RETURN_TYPE] Object   Object[] args [VARIABLES] boolean  Object[]  args  
[buglab_swap_variables]^return fc.newInstance ( index ) ;^40^^^^^39^41^return fc.newInstance ( index, args ) ;^[CLASS] FastConstructor  [METHOD] newInstance [RETURN_TYPE] Object   Object[] args [VARIABLES] boolean  Object[]  args  
[buglab_swap_variables]^return index.newInstance ( fc, args ) ;^40^^^^^39^41^return fc.newInstance ( index, args ) ;^[CLASS] FastConstructor  [METHOD] newInstance [RETURN_TYPE] Object   Object[] args [VARIABLES] boolean  Object[]  args  
[buglab_swap_variables]^return fc.newInstance (  args ) ;^40^^^^^39^41^return fc.newInstance ( index, args ) ;^[CLASS] FastConstructor  [METHOD] newInstance [RETURN_TYPE] Object   Object[] args [VARIABLES] boolean  Object[]  args  
[buglab_swap_variables]^return fc.newInstance ( args, index ) ;^40^^^^^39^41^return fc.newInstance ( index, args ) ;^[CLASS] FastConstructor  [METHOD] newInstance [RETURN_TYPE] Object   Object[] args [VARIABLES] boolean  Object[]  args  