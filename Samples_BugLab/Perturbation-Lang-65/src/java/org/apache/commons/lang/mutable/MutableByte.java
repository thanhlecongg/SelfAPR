[buglab_swap_variables]^byte anotherVal = other.value.value;^263^^^^^261^265^byte anotherVal = other.value;^[CLASS] MutableByte  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  boolean  MutableByte  other  byte  anotherVal  operand  value  long  serialVersionUID  
[buglab_swap_variables]^byte anotherVal = other;^263^^^^^261^265^byte anotherVal = other.value;^[CLASS] MutableByte  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  boolean  MutableByte  other  byte  anotherVal  operand  value  long  serialVersionUID  
[buglab_swap_variables]^return anotherVal < value ? -1 :  ( value == anotherVal ? 0 : 1 ) ;^264^^^^^261^265^return value < anotherVal ? -1 :  ( value == anotherVal ? 0 : 1 ) ;^[CLASS] MutableByte  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  boolean  MutableByte  other  byte  anotherVal  operand  value  long  serialVersionUID  