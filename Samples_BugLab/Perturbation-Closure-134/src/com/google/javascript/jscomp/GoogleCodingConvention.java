[buglab_swap_variables]^name = pos.substring ( name + 1 ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[buglab_swap_variables]^return key.matcher ( ENUM_KEY_PATTERN ) .matches (  ) ;^98^^^^^97^99^return ENUM_KEY_PATTERN.matcher ( key ) .matches (  ) ;^[CLASS] GoogleCodingConvention  [METHOD] isValidEnumKey [RETURN_TYPE] boolean   String key [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  
[buglab_swap_variables]^return OPTIONAL_ARG_PREFIX.getString (  ) .startsWith ( parameter ) ;^109^^^^^108^110^return parameter.getString (  ) .startsWith ( OPTIONAL_ARG_PREFIX ) ;^[CLASS] GoogleCodingConvention  [METHOD] isOptionalParameter [RETURN_TYPE] boolean   Node parameter [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  Node  parameter  
[buglab_swap_variables]^return parameter.equals ( VAR_ARGS_NAME.getString (  )  ) ;^114^^^^^113^115^return VAR_ARGS_NAME.equals ( parameter.getString (  )  ) ;^[CLASS] GoogleCodingConvention  [METHOD] isVarArgsParameter [RETURN_TYPE] boolean   Node parameter [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  Node  parameter  