[P1_Replace_Type]^private static final  short  serialVersionUID = -3520677225766901240L;^36^^^^^31^41^private static final long serialVersionUID = -3520677225766901240L;^[CLASS] ConstantFactory   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = -3520677225766901240;^36^^^^^31^41^private static final long serialVersionUID = -3520677225766901240L;^[CLASS] ConstantFactory   [VARIABLES] 
[P4_Replace_Constructor]^public static final Factory NULL_INSTANCE = public static final Factory NULL_INSTANCE =  new ConstantFactory<T> ( constantToReturn )  ;^40^^^^^35^45^public static final Factory NULL_INSTANCE = new ConstantFactory<Object> ( null ) ;^[CLASS] ConstantFactory   [VARIABLES] 
[P8_Replace_Mix]^public static final Factory NULL_INSTANCE = new ConstantFactory<Object> ( false ) ;^40^^^^^35^45^public static final Factory NULL_INSTANCE = new ConstantFactory<Object> ( null ) ;^[CLASS] ConstantFactory   [VARIABLES] 
[P14_Delete_Statement]^^67^^^^^66^69^super (  ) ;^[CLASS] ConstantFactory  [METHOD] <init> [RETURN_TYPE] ConstantFactory(T)   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P5_Replace_Variable]^iConstant = iConstant;^68^^^^^66^69^iConstant = constantToReturn;^[CLASS] ConstantFactory  [METHOD] <init> [RETURN_TYPE] ConstantFactory(T)   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P8_Replace_Mix]^iConstant =  null;^68^^^^^66^69^iConstant = constantToReturn;^[CLASS] ConstantFactory  [METHOD] <init> [RETURN_TYPE] ConstantFactory(T)   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P2_Replace_Operator]^if  ( constantToReturn != null )  {^54^^^^^53^58^if  ( constantToReturn == null )  {^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P5_Replace_Variable]^if  ( iConstant == null )  {^54^^^^^53^58^if  ( constantToReturn == null )  {^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P8_Replace_Mix]^if  ( constantToReturn == false )  {^54^^^^^53^58^if  ( constantToReturn == null )  {^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P15_Unwrap_Block]^return ((org.apache.commons.collections4.Factory<T>) (org.apache.commons.collections4.functors.ConstantFactory.NULL_INSTANCE));^54^55^56^^^53^58^if  ( constantToReturn == null )  { return  ( Factory<T> )  NULL_INSTANCE; }^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P16_Remove_Block]^^54^55^56^^^53^58^if  ( constantToReturn == null )  { return  ( Factory<T> )  NULL_INSTANCE; }^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P13_Insert_Block]^if  ( constantToReturn == null )  {     return  (  ( Factory<T> )   ( NULL_INSTANCE )  ) ; }^55^^^^^53^58^[Delete]^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P8_Replace_Mix]^return new ConstantFactory<T> ( iConstant ) ;^57^^^^^53^58^return new ConstantFactory<T> ( constantToReturn ) ;^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P8_Replace_Mix]^return constantToReturn;^77^^^^^76^78^return iConstant;^[CLASS] ConstantFactory  [METHOD] create [RETURN_TYPE] T   [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[P5_Replace_Variable]^return constantToReturn;^87^^^^^86^88^return iConstant;^[CLASS] ConstantFactory  [METHOD] getConstant [RETURN_TYPE] T   [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  