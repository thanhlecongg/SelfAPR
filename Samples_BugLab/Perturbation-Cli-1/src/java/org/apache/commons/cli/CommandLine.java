[buglab_swap_variables]^return optionsions.contains (  resolveOption ( opt )  ) ;^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[buglab_swap_variables]^return opt.contains (  resolveOption ( options )  ) ;^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[buglab_swap_variables]^return  ( type == null )         ? null : TypeHandler.createValue ( res, res ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[buglab_swap_variables]^return  ( res == null )         ? null : TypeHandler.createValue (  type ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[buglab_swap_variables]^return  ( res == null )         ? null : TypeHandler.createValue ( res ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[buglab_swap_variables]^if  ( key.contains ( options )  ) {^150^151^^^^146^156^if  ( options.contains ( key )  ) {^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[buglab_swap_variables]^if  ( option.equals ( opt.getOpt (  )  )  ) {^169^170^^^^163^180^if  ( opt.equals ( option.getOpt (  )  )  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[buglab_swap_variables]^if  ( option.equals (  opt.getLongOpt (  )  )  ) {^173^174^^^^163^180^if  ( opt.equals (  option.getLongOpt (  )  )  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[buglab_swap_variables]^for  (  Iterator opt = itions.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[buglab_swap_variables]^for  (  Iterator it = optionions.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[buglab_swap_variables]^for  (  Iterator option = its.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[buglab_swap_variables]^for  (  Iterator it = opt.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[buglab_swap_variables]^return  ( defaultValue != null )  ? answer : answer;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[buglab_swap_variables]^return getOptionValue ( String.valueOf ( defaultValue ) , opt ) ;^221^^^^^219^222^return getOptionValue ( String.valueOf ( opt ) , defaultValue ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt String defaultValue [VARIABLES] char  opt  List  args  Set  options  String  defaultValue  boolean  
[buglab_swap_variables]^return getOptionValue ( String.valueOf ( opt )  ) ;^221^^^^^219^222^return getOptionValue ( String.valueOf ( opt ) , defaultValue ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt String defaultValue [VARIABLES] char  opt  List  args  Set  options  String  defaultValue  boolean  
[buglab_swap_variables]^return  ( Option[] )  optionsArray.toArray ( processed ) ;^315^^^^^307^316^return  ( Option[] )  processed.toArray ( optionsArray ) ;^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  