[P1_Replace_Type]^private  Set  args = new Linked Set  (  ) ;^43^^^^^38^48^private List args = new LinkedList (  ) ;^[CLASS] CommandLine   [VARIABLES] 
[P1_Replace_Type]^private  List  options = new Hash List  (  ) ;^46^^^^^41^51^private Set options = new HashSet (  ) ;^[CLASS] CommandLine   [VARIABLES] 
[P5_Replace_Variable]^return optionsions.contains (  resolveOption ( opt )  ) ;^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[P5_Replace_Variable]^return opt.contains (  resolveOption ( options )  ) ;^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^return options.add (  resolveOption ( opt )  ) ;^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^return options .iterator (  )  ;^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^return options.contains (  hasOption ( opt )  ) ;^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[P12_Insert_Condition]^if  ( options.contains ( key )  ) { return options.contains (  resolveOption ( opt )  ) ; }^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[P14_Delete_Statement]^^67^^^^^65^68^return options.contains (  resolveOption ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   String opt [VARIABLES] List  args  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^return resolveOption ( String.valueOf ( opt )  ) ;^78^^^^^76^79^return hasOption ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P7_Replace_Invocation]^return hasOption ( String.equals ( opt )  ) ;^78^^^^^76^79^return hasOption ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P14_Delete_Statement]^^78^^^^^76^79^return hasOption ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] hasOption [RETURN_TYPE] boolean   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P1_Replace_Type]^char res = getOptionValue ( opt ) ;^89^^^^^87^100^String res = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P7_Replace_Invocation]^String res = getOptionValues ( opt ) ;^89^^^^^87^100^String res = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P11_Insert_Donor_Statement]^String[] values = getOptionValues ( opt ) ;String res = getOptionValue ( opt ) ;^89^^^^^87^100^String res = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P11_Insert_Donor_Statement]^String answer = getOptionValue ( opt ) ;String res = getOptionValue ( opt ) ;^89^^^^^87^100^String res = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P5_Replace_Variable]^String res = getOptionValue ( res ) ;^89^^^^^87^100^String res = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P14_Delete_Statement]^^89^^^^^87^100^String res = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P5_Replace_Variable]^Option resion = resolveOption ( opt ) ;^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P7_Replace_Invocation]^Option option = hasOption ( opt ) ;^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P8_Replace_Mix]^Option resion = hasOption ( opt ) ;^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P11_Insert_Donor_Statement]^Option option =  ( Option )  it.next (  ) ;Option option = resolveOption ( opt ) ;^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P11_Insert_Donor_Statement]^Option key = resolveOption (  opt  ) ;Option option = resolveOption ( opt ) ;^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P14_Delete_Statement]^^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P11_Insert_Donor_Statement]^return options.iterator (  ) ;Option option = resolveOption ( opt ) ;^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P11_Insert_Donor_Statement]^return options.contains (  resolveOption ( opt )  ) ;Option option = resolveOption ( opt ) ;^91^^^^^87^100^Option option = resolveOption ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P2_Replace_Operator]^if  ( option != null ) {^92^93^^^^87^100^if  ( option == null ) {^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P8_Replace_Mix]^if  ( option == this ) {^92^93^^^^87^100^if  ( option == null ) {^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P15_Unwrap_Block]^return null;^92^93^94^95^^87^100^if  ( option == null ) { return null; }^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P16_Remove_Block]^^92^93^94^95^^87^100^if  ( option == null ) { return null; }^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P13_Insert_Block]^if  ( options.contains ( key )  )  {     return key.getValues (  ) ; }^92^^^^^87^100^[Delete]^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P13_Insert_Block]^if  ( opt.equals ( option.getOpt (  )  )  )  {     return option; }^92^^^^^87^100^[Delete]^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P8_Replace_Mix]^return true;^94^^^^^87^100^return null;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P7_Replace_Invocation]^Object type = option .getOpt (  )  ;^97^^^^^87^100^Object type = option.getType (  ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P14_Delete_Statement]^^97^^^^^87^100^Object type = option.getType (  ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P2_Replace_Operator]^return  ( res != null )         ? null : TypeHandler.createValue ( res, type ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P5_Replace_Variable]^return  ( opt == null )         ? null : TypeHandler.createValue ( res, type ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P5_Replace_Variable]^return  ( res == null )         ? null : TypeHandler.createValue (  type ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P5_Replace_Variable]^return  ( res == null )         ? null : TypeHandler.createValue ( res ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P5_Replace_Variable]^return  ( type == null )         ? null : TypeHandler.createValue ( res, res ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P6_Replace_Expression]^return  ( TypeHandler.createValue ( res, type ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P8_Replace_Mix]^return TypeHandler.createValue ( res, type ) ;^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P14_Delete_Statement]^^99^^^^^87^100^return  ( res == null )         ? null : TypeHandler.createValue ( res, type ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   String opt [VARIABLES] List  args  Option  option  Object  type  Set  options  String  opt  res  boolean  
[P7_Replace_Invocation]^return getOptionValue ( String.valueOf ( opt )  ) ;^110^^^^^108^111^return getOptionObject ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P7_Replace_Invocation]^return getOptionObject ( String.equals ( opt )  ) ;^110^^^^^108^111^return getOptionObject ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P14_Delete_Statement]^^110^^^^^108^111^return getOptionObject ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionObject [RETURN_TYPE] Object   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P7_Replace_Invocation]^String[] values = getOptionValue ( opt ) ;^122^^^^^120^125^String[] values = getOptionValues ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt [VARIABLES] List  args  Set  options  String  opt  String[]  values  boolean  
[P11_Insert_Donor_Statement]^String answer = getOptionValue ( opt ) ;String[] values = getOptionValues ( opt ) ;^122^^^^^120^125^String[] values = getOptionValues ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt [VARIABLES] List  args  Set  options  String  opt  String[]  values  boolean  
[P11_Insert_Donor_Statement]^String res = getOptionValue ( opt ) ;String[] values = getOptionValues ( opt ) ;^122^^^^^120^125^String[] values = getOptionValues ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt [VARIABLES] List  args  Set  options  String  opt  String[]  values  boolean  
[P14_Delete_Statement]^^122^^^^^120^125^String[] values = getOptionValues ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt [VARIABLES] List  args  Set  options  String  opt  String[]  values  boolean  
[P2_Replace_Operator]^return  ( values != null )  ? null : values[0];^124^^^^^120^125^return  ( values == null )  ? null : values[0];^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt [VARIABLES] List  args  Set  options  String  opt  String[]  values  boolean  
[P6_Replace_Expression]^return  ( values[0];^124^^^^^120^125^return  ( values == null )  ? null : values[0];^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt [VARIABLES] List  args  Set  options  String  opt  String[]  values  boolean  
[P8_Replace_Mix]^return  ( values != null )  ? null : values[0];;^124^^^^^120^125^return  ( values == null )  ? null : values[0];^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt [VARIABLES] List  args  Set  options  String  opt  String[]  values  boolean  
[P7_Replace_Invocation]^return getOptionValues ( String.valueOf ( opt )  ) ;^136^^^^^134^137^return getOptionValue ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P7_Replace_Invocation]^return getOptionValue ( String.equals ( opt )  ) ;^136^^^^^134^137^return getOptionValue ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P7_Replace_Invocation]^return getOptionValue ( String .equals ( null )   ) ;^136^^^^^134^137^return getOptionValue ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P14_Delete_Statement]^^136^^^^^134^137^return getOptionValue ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P7_Replace_Invocation]^Option key = hasOption (  opt  ) ;^148^^^^^146^156^Option key = resolveOption (  opt  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P11_Insert_Donor_Statement]^Option option = resolveOption ( opt ) ;Option key = resolveOption (  opt  ) ;^148^^^^^146^156^Option key = resolveOption (  opt  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P14_Delete_Statement]^^148^^^^^146^156^Option key = resolveOption (  opt  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P11_Insert_Donor_Statement]^return options.contains (  resolveOption ( opt )  ) ;Option key = resolveOption (  opt  ) ;^148^^^^^146^156^Option key = resolveOption (  opt  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P5_Replace_Variable]^if  ( key.contains ( options )  ) {^150^151^^^^146^156^if  ( options.contains ( key )  ) {^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^if  ( options.add ( key )  ) {^150^151^^^^146^156^if  ( options.contains ( key )  ) {^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P15_Unwrap_Block]^return key.getValues();^150^151^152^153^^146^156^if  ( options.contains ( key )  ) { return key.getValues (  ) ; }^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P16_Remove_Block]^^150^151^152^153^^146^156^if  ( options.contains ( key )  ) { return key.getValues (  ) ; }^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P13_Insert_Block]^if  ( option == null )  {     return null; }^150^^^^^146^156^[Delete]^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^return key .getType (  )  ;^152^^^^^146^156^return key.getValues (  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P14_Delete_Statement]^^152^^^^^146^156^return key.getValues (  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P8_Replace_Mix]^return false;^155^^^^^146^156^return null;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   String opt [VARIABLES] List  args  Option  key  Set  options  String  opt  boolean  
[P8_Replace_Mix]^opt =  Util.stripLeadingHyphens ( null ) ;^165^^^^^163^180^opt = Util.stripLeadingHyphens ( opt ) ;^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P14_Delete_Statement]^^165^^^^^163^180^opt = Util.stripLeadingHyphens ( opt ) ;^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P5_Replace_Variable]^if  ( option.equals ( opt.getOpt (  )  )  ) {^169^170^^^^163^180^if  ( opt.equals ( option.getOpt (  )  )  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P9_Replace_Statement]^if  ( opt.equals (  option.getLongOpt (  )  )  ) {^169^170^^^^163^180^if  ( opt.equals ( option.getOpt (  )  )  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P15_Unwrap_Block]^return option;^169^170^171^172^^163^180^if  ( opt.equals ( option.getOpt (  )  )  ) { return option; }^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P16_Remove_Block]^^169^170^171^172^^163^180^if  ( opt.equals ( option.getOpt (  )  )  ) { return option; }^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P13_Insert_Block]^if  ( option == null )  {     return null; }^169^^^^^163^180^[Delete]^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P13_Insert_Block]^if  ( opt.equals ( option.getLongOpt (  )  )  )  {     return option; }^169^^^^^163^180^[Delete]^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P5_Replace_Variable]^if  ( option.equals (  opt.getLongOpt (  )  )  ) {^173^174^^^^163^180^if  ( opt.equals (  option.getLongOpt (  )  )  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^if  ( opt.equals (  option .getOpt (  )   )  ) {^173^174^^^^163^180^if  ( opt.equals (  option.getLongOpt (  )  )  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P15_Unwrap_Block]^return option;^173^174^175^176^^163^180^if  ( opt.equals (  option.getLongOpt (  )  )  ) { return option; }^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P16_Remove_Block]^^173^174^175^176^^163^180^if  ( opt.equals (  option.getLongOpt (  )  )  ) { return option; }^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P13_Insert_Block]^if  ( opt.equals ( option.getOpt (  )  )  )  {     return option; }^173^^^^^163^180^[Delete]^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P5_Replace_Variable]^for  (  Iterator opt = itions.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P5_Replace_Variable]^for  (  Iterator it = optionsions.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P5_Replace_Variable]^for  (  Iterator option = its.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P5_Replace_Variable]^for  (  Iterator it = option.iterator (  ) ; it.hasNext (  ) ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P8_Replace_Mix]^if  ( opt .valueOf ( null )   ) {^173^174^^^^163^180^if  ( opt.equals (  option.getLongOpt (  )  )  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P11_Insert_Donor_Statement]^Option option = resolveOption ( opt ) ;Option option =  ( Option )  it.next (  ) ;^168^^^^^163^180^Option option =  ( Option )  it.next (  ) ;^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P11_Insert_Donor_Statement]^Option[] optionsArray = new Option[processed.size (  ) ];Option option =  ( Option )  it.next (  ) ;^168^^^^^163^180^Option option =  ( Option )  it.next (  ) ;^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^Option option =  ( Option )  it .hasNext (  )  ;^168^^^^^163^180^Option option =  ( Option )  it.next (  ) ;^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P14_Delete_Statement]^^168^^^^^163^180^Option option =  ( Option )  it.next (  ) ;^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P13_Insert_Block]^if  ( opt.equals ( option.getLongOpt (  )  )  )  {     return option; }^173^^^^^163^180^[Delete]^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^for  (  Iterator it = options.iterator (  ) ; it .next (  )  ;  ) {^166^167^^^^163^180^for  (  Iterator it = options.iterator (  ) ; it.hasNext (  ) ;  ) {^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P8_Replace_Mix]^return true;^179^^^^^163^180^return null;^[CLASS] CommandLine  [METHOD] resolveOption [RETURN_TYPE] Option   String opt [VARIABLES] Iterator  it  List  args  Option  option  Set  options  String  opt  boolean  
[P7_Replace_Invocation]^return getOptionValue ( String.valueOf ( opt )  ) ;^191^^^^^189^192^return getOptionValues ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P7_Replace_Invocation]^return getOptionValues ( String.equals ( opt )  ) ;^191^^^^^189^192^return getOptionValues ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P14_Delete_Statement]^^191^^^^^189^192^return getOptionValues ( String.valueOf ( opt )  ) ;^[CLASS] CommandLine  [METHOD] getOptionValues [RETURN_TYPE] String[]   char opt [VARIABLES] char  opt  List  args  Set  options  boolean  
[P1_Replace_Type]^char answer = getOptionValue ( opt ) ;^205^^^^^203^208^String answer = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P7_Replace_Invocation]^String answer = getOptionValues ( opt ) ;^205^^^^^203^208^String answer = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P8_Replace_Mix]^String answer = getOptionValue ( defaultValue ) ;^205^^^^^203^208^String answer = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P11_Insert_Donor_Statement]^String[] values = getOptionValues ( opt ) ;String answer = getOptionValue ( opt ) ;^205^^^^^203^208^String answer = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P11_Insert_Donor_Statement]^String[] answer = new String[args.size (  ) ];String answer = getOptionValue ( opt ) ;^205^^^^^203^208^String answer = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P11_Insert_Donor_Statement]^String res = getOptionValue ( opt ) ;String answer = getOptionValue ( opt ) ;^205^^^^^203^208^String answer = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P14_Delete_Statement]^^205^^^^^203^208^String answer = getOptionValue ( opt ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P2_Replace_Operator]^return  ( answer == null )  ? answer : defaultValue;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P5_Replace_Variable]^return  ( opt != null )  ? answer : defaultValue;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P5_Replace_Variable]^return  ( answer != null )  ? answer : opt;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P5_Replace_Variable]^return  ( defaultValue != null )  ? answer : answer;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P6_Replace_Expression]^return  ( answer ! =  defaultValue;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P6_Replace_Expression]^return  ( answer ! =  answer ;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P8_Replace_Mix]^return  ( answer ;^207^^^^^203^208^return  ( answer != null )  ? answer : defaultValue;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   String opt String defaultValue [VARIABLES] List  args  Set  options  String  answer  defaultValue  opt  boolean  
[P5_Replace_Variable]^return getOptionValue ( String.valueOf ( opt )  ) ;^221^^^^^219^222^return getOptionValue ( String.valueOf ( opt ) , defaultValue ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt String defaultValue [VARIABLES] char  opt  List  args  Set  options  String  defaultValue  boolean  
[P5_Replace_Variable]^return getOptionValue ( String.valueOf ( defaultValue ) , opt ) ;^221^^^^^219^222^return getOptionValue ( String.valueOf ( opt ) , defaultValue ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt String defaultValue [VARIABLES] char  opt  List  args  Set  options  String  defaultValue  boolean  
[P7_Replace_Invocation]^return getOptionValue ( String.equals ( opt ) , defaultValue ) ;^221^^^^^219^222^return getOptionValue ( String.valueOf ( opt ) , defaultValue ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt String defaultValue [VARIABLES] char  opt  List  args  Set  options  String  defaultValue  boolean  
[P14_Delete_Statement]^^221^^^^^219^222^return getOptionValue ( String.valueOf ( opt ) , defaultValue ) ;^[CLASS] CommandLine  [METHOD] getOptionValue [RETURN_TYPE] String   char opt String defaultValue [VARIABLES] char  opt  List  args  Set  options  String  defaultValue  boolean  
[P3_Replace_Literal]^String[] answer = new String[args.size() + 3 ];^231^^^^^229^236^String[] answer = new String[args.size (  ) ];^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P11_Insert_Donor_Statement]^String answer = getOptionValue ( opt ) ;String[] answer = new String[args.size (  ) ];^231^^^^^229^236^String[] answer = new String[args.size (  ) ];^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P11_Insert_Donor_Statement]^Option[] optionsArray = new Option[processed.size (  ) ];String[] answer = new String[args.size (  ) ];^231^^^^^229^236^String[] answer = new String[args.size (  ) ];^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P3_Replace_Literal]^String[] answer = new String[args.size() - 3 ];^231^^^^^229^236^String[] answer = new String[args.size (  ) ];^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P14_Delete_Statement]^^231^^^^^229^236^String[] answer = new String[args.size (  ) ];^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P7_Replace_Invocation]^args.add ( answer ) ;^233^^^^^229^236^args.toArray ( answer ) ;^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P14_Delete_Statement]^^233^^^^^229^236^args.toArray ( answer ) ;^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P11_Insert_Donor_Statement]^args.add ( arg ) ;args.toArray ( answer ) ;^233^^^^^229^236^args.toArray ( answer ) ;^[CLASS] CommandLine  [METHOD] getArgs [RETURN_TYPE] String[]   [VARIABLES] List  args  Set  options  String[]  answer  boolean  
[P7_Replace_Invocation]^args.toArray ( arg ) ;^277^^^^^275^278^args.add ( arg ) ;^[CLASS] CommandLine  [METHOD] addArg [RETURN_TYPE] void   String arg [VARIABLES] List  args  Set  options  String  arg  boolean  
[P14_Delete_Statement]^^277^^^^^275^278^args.add ( arg ) ;^[CLASS] CommandLine  [METHOD] addArg [RETURN_TYPE] void   String arg [VARIABLES] List  args  Set  options  String  arg  boolean  
[P11_Insert_Donor_Statement]^options.add ( opt ) ;args.add ( arg ) ;^277^^^^^275^278^args.add ( arg ) ;^[CLASS] CommandLine  [METHOD] addArg [RETURN_TYPE] void   String arg [VARIABLES] List  args  Set  options  String  arg  boolean  
[P11_Insert_Donor_Statement]^args.toArray ( answer ) ;args.add ( arg ) ;^277^^^^^275^278^args.add ( arg ) ;^[CLASS] CommandLine  [METHOD] addArg [RETURN_TYPE] void   String arg [VARIABLES] List  args  Set  options  String  arg  boolean  
[P7_Replace_Invocation]^options.contains ( opt ) ;^288^^^^^286^289^options.add ( opt ) ;^[CLASS] CommandLine  [METHOD] addOption [RETURN_TYPE] void   Option opt [VARIABLES] List  args  Option  opt  Set  options  boolean  
[P14_Delete_Statement]^^288^^^^^286^289^options.add ( opt ) ;^[CLASS] CommandLine  [METHOD] addOption [RETURN_TYPE] void   Option opt [VARIABLES] List  args  Option  opt  Set  options  boolean  
[P11_Insert_Donor_Statement]^args.add ( arg ) ;options.add ( opt ) ;^288^^^^^286^289^options.add ( opt ) ;^[CLASS] CommandLine  [METHOD] addOption [RETURN_TYPE] void   Option opt [VARIABLES] List  args  Option  opt  Set  options  boolean  
[P14_Delete_Statement]^^299^^^^^297^300^return options.iterator (  ) ;^[CLASS] CommandLine  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] List  args  Set  options  boolean  
[P3_Replace_Literal]^Option[] optionsArray = new Option[processed.size() + 3 ];^312^^^^^307^316^Option[] optionsArray = new Option[processed.size (  ) ];^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  
[P7_Replace_Invocation]^Option[] optionsArray = new Option[processed.toArray (  ) ];^312^^^^^307^316^Option[] optionsArray = new Option[processed.size (  ) ];^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  
[P11_Insert_Donor_Statement]^Option option =  ( Option )  it.next (  ) ;Option[] optionsArray = new Option[processed.size (  ) ];^312^^^^^307^316^Option[] optionsArray = new Option[processed.size (  ) ];^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  
[P11_Insert_Donor_Statement]^String[] answer = new String[args.size (  ) ];Option[] optionsArray = new Option[processed.size (  ) ];^312^^^^^307^316^Option[] optionsArray = new Option[processed.size (  ) ];^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  
[P3_Replace_Literal]^Option[] optionsArray = new Option[processed.size() + 1 ];^312^^^^^307^316^Option[] optionsArray = new Option[processed.size (  ) ];^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  
[P14_Delete_Statement]^^312^^^^^307^316^Option[] optionsArray = new Option[processed.size (  ) ];^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  
[P5_Replace_Variable]^return  ( Option[] )  optionsArray.toArray ( processed ) ;^315^^^^^307^316^return  ( Option[] )  processed.toArray ( optionsArray ) ;^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  
[P14_Delete_Statement]^^315^^^^^307^316^return  ( Option[] )  processed.toArray ( optionsArray ) ;^[CLASS] CommandLine  [METHOD] getOptions [RETURN_TYPE] Option[]   [VARIABLES] Collection  processed  List  args  Set  options  Option[]  optionsArray  boolean  