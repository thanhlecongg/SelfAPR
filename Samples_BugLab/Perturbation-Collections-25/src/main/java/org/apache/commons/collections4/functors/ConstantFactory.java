[BugLab_Variable_Misuse]^iConstant = iConstant;^68^^^^^66^69^iConstant = constantToReturn;^[CLASS] ConstantFactory  [METHOD] <init> [RETURN_TYPE] ConstantFactory(T)   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( iConstant == null )  {^54^^^^^53^58^if  ( constantToReturn == null )  {^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( constantToReturn != null )  {^54^^^^^53^58^if  ( constantToReturn == null )  {^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new ConstantFactory<T> ( iConstant ) ;^57^^^^^53^58^return new ConstantFactory<T> ( constantToReturn ) ;^[CLASS] ConstantFactory  [METHOD] constantFactory [RETURN_TYPE] <T>   final T constantToReturn [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[BugLab_Variable_Misuse]^return constantToReturn;^77^^^^^76^78^return iConstant;^[CLASS] ConstantFactory  [METHOD] create [RETURN_TYPE] T   [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  
[BugLab_Variable_Misuse]^return constantToReturn;^87^^^^^86^88^return iConstant;^[CLASS] ConstantFactory  [METHOD] getConstant [RETURN_TYPE] T   [VARIABLES] Factory  NULL_INSTANCE  boolean  T  constantToReturn  iConstant  long  serialVersionUID  