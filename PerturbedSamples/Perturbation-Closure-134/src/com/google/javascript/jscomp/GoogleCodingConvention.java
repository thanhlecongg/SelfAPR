[REPLACE]^private static  String OPTIONAL_ARG_PREFIX = "opt_";^34^^^^^^^[REPLACE] private static final String OPTIONAL_ARG_PREFIX = "opt_";^ [CLASS] GoogleCodingConvention  
[REPLACE]^private static final String VAR_ARGS_NAME ;^36^^^^^^^[REPLACE] private static final String VAR_ARGS_NAME = "var_args";^ [CLASS] GoogleCodingConvention  
[REPLACE]^private static final Pattern ENUM_KEY_PATTERN ;^38^39^^^^38^39^[REPLACE] private static final Pattern ENUM_KEY_PATTERN = Pattern.compile ( "[A-Z0-9][A-Z0-9_]*" ) ;^ [CLASS] GoogleCodingConvention  
[REPLACE]^if  ( name .endsWith ( VAR_ARGS_NAME )   + 1 <= 1 )  {^61^^^^^60^81^[REPLACE] if  ( name.length (  )  <= 1 )  {^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REMOVE]^if  ( ! ( isUpperCase ( name.charAt ( 0 )  )  )  )  {     return false; }^61^^^^^60^81^[REMOVE] ^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^return true;^62^^^^^60^81^[REPLACE] return false;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^int pos = VAR_ARGS_NAME.lastIndexOf ( '$' ) ;^67^^^^^60^81^[REPLACE] int pos = name.lastIndexOf ( '$' ) ;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^if  ( pos  <=  0 )  {^68^^^^^60^81^[REPLACE] if  ( pos >= 0 )  {^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^if  ( name.size (  )  == 0 )  {^70^^^^^60^81^[REPLACE] if  ( name.length (  )  == 0 )  {^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^return true;^71^^^^^60^81^[REPLACE] return false;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^name = name.substring ( pos  >  2 ) ;^69^^^^^60^81^[REPLACE] name = name.substring ( pos + 1 ) ;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[ADD]^^69^^^^^60^81^[ADD] name = name.substring ( pos + 1 ) ;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^if  ( name.length (  )  /  2 == 0 )  {^70^^^^^60^81^[REPLACE] if  ( name.length (  )  == 0 )  {^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REMOVE]^if  ( ! ( isUpperCase ( name.charAt ( 0 )  )  )  )  {     return false; }^70^^^^^60^81^[REMOVE] ^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^name = name.substring ( pos  <=  1 ) ;^69^^^^^60^81^[REPLACE] name = name.substring ( pos + 1 ) ;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^if  ( !Character.isUpperCase ( VAR_ARGS_NAME.charAt ( 1 )  )  )  {^75^^^^^60^81^[REPLACE] if  ( !Character.isUpperCase ( name.charAt ( 0 )  )  )  {^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REMOVE]^if  (  ( name.length (  )  )  <= 1 )  {     return false; }^75^^^^^60^81^[REMOVE] ^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^return true;^76^^^^^60^81^[REPLACE] return false;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^return name.length (  ) .equals ( name ) ;^80^^^^^60^81^[REPLACE] return name.toUpperCase (  ) .equals ( name ) ;^[METHOD] isConstant [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  [TYPE]  boolean false  true  [TYPE]  int pos 
[REPLACE]^return ENUM_KEY_PATTERN.matcher ( OPTIONAL_ARG_PREFIX ) .matches (  ) ;^98^^^^^97^99^[REPLACE] return ENUM_KEY_PATTERN.matcher ( key ) .matches (  ) ;^[METHOD] isValidEnumKey [TYPE] boolean [PARAMETER] String key [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  [TYPE]  boolean false  true 
[REPLACE]^return ENUM_KEY_PATTERN.matcher ( key ) .matches (  ) ;^109^^^^^108^110^[REPLACE] return parameter.getString (  ) .startsWith ( OPTIONAL_ARG_PREFIX ) ;^[METHOD] isOptionalParameter [TYPE] boolean [PARAMETER] Node parameter [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  [TYPE]  boolean false  true  [TYPE]  Node parameter 
[REPLACE]^return ENUM_KEY_PATTERN.matcher ( key ) .matches (  ) ;^114^^^^^113^115^[REPLACE] return VAR_ARGS_NAME.equals ( parameter.getString (  )  ) ;^[METHOD] isVarArgsParameter [TYPE] boolean [PARAMETER] Node parameter [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  [TYPE]  boolean false  true  [TYPE]  Node parameter 
[REPLACE]^return !local && name.endsWith ( "_" ) ;^125^^^^^124^126^[REPLACE] return !local && name.startsWith ( "_" ) ;^[METHOD] isExported [TYPE] boolean [PARAMETER] String name boolean local [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  [TYPE]  boolean false  local  true 
[REPLACE]^return name.endsWith ( "_" )  || !isExported ( name ) ;^136^^^^^135^137^[REPLACE] return name.endsWith ( "_" )  && !isExported ( name ) ;^[METHOD] isPrivate [TYPE] boolean [PARAMETER] String name [CLASS] GoogleCodingConvention   [TYPE]  Pattern ENUM_KEY_PATTERN  [TYPE]  String OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  [TYPE]  boolean false  true 