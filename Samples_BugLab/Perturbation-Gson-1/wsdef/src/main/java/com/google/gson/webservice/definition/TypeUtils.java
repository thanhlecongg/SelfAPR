[BugLab_Wrong_Operator]^if  ( type  <  Class )  {^44^^^^^43^54^if  ( type instanceof Class )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^if  ( type  >=  Class )  {^44^^^^^43^54^if  ( type instanceof Class )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^if  ( type  ==  Class )  {^44^^^^^43^54^if  ( type instanceof Class )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  &  ParameterizedType )  {^46^^^^^43^54^} else if  ( type instanceof ParameterizedType )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  ==  GenericArrayType )  {^48^^^^^43^54^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ==  type  ==  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ==  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >=  type  >=  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  &&  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >  type  >  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Literal]^return  (  ( ParameterizedType ) type ) .getActualTypeArguments (  ) [1];^47^^^^^43^54^return  (  ( ParameterizedType ) type ) .getActualTypeArguments (  ) [0];^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  !=  GenericArrayType )  {^48^^^^^43^54^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <  type  <  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"   instanceof   type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Literal]^return  (  ( ParameterizedType ) type ) .getActualTypeArguments (  ) [this];^47^^^^^43^54^return  (  ( ParameterizedType ) type ) .getActualTypeArguments (  ) [0];^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ^  type  ^  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  |  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  !=  ParameterizedType )  {^46^^^^^43^54^} else if  ( type instanceof ParameterizedType )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  ==  ParameterizedType )  {^46^^^^^43^54^} else if  ( type instanceof ParameterizedType )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  ^  GenericArrayType )  {^48^^^^^43^54^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  &  type  &  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ||  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >>  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <<  type  <<  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >=  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  &  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  &  GenericArrayType )  {^48^^^^^43^54^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <=  type  <=  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"   instanceof   type   instanceof   "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  !=  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ^  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^51^52^^^^43^54^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] getActualTypeForFirstTypeVariable [RETURN_TYPE] Type   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^if  ( type  >  Class )  {^57^^^^^56^64^if  ( type instanceof Class )  {^[CLASS] TypeUtils  [METHOD] isArray [RETURN_TYPE] boolean   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^if  ( type  <  Class )  {^57^^^^^56^64^if  ( type instanceof Class )  {^[CLASS] TypeUtils  [METHOD] isArray [RETURN_TYPE] boolean   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  >=  GenericArrayType )  {^59^^^^^56^64^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] isArray [RETURN_TYPE] boolean   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Literal]^return true;^62^^^^^56^64^return false;^[CLASS] TypeUtils  [METHOD] isArray [RETURN_TYPE] boolean   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Literal]^return false;^60^^^^^56^64^return true;^[CLASS] TypeUtils  [METHOD] isArray [RETURN_TYPE] boolean   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^} else if  ( type  <=  GenericArrayType )  {^59^^^^^56^64^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] isArray [RETURN_TYPE] boolean   Type type [VARIABLES] Type  type  boolean  
[BugLab_Wrong_Operator]^if  ( type  <  Class )  {^70^^^^^69^83^if  ( type instanceof Class )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^if  ( type  >>  Class )  {^70^^^^^69^83^if  ( type instanceof Class )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^} else if  ( type  <  ParameterizedType )  {^72^^^^^69^83^} else if  ( type instanceof ParameterizedType )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^} else if  ( type  &&  ParameterizedType )  {^72^^^^^69^83^} else if  ( type instanceof ParameterizedType )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^} else if  ( type  ||  GenericArrayType )  {^75^^^^^69^83^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <=  type  <=  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"   instanceof   type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  &&  type  &&  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  !=  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Variable_Misuse]^return wrapWithArray ( null ) ;^78^^^^^69^83^return wrapWithArray ( rawClass ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <<  type  <<  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >>  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >  type  >  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ==  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^} else if  ( type  |  GenericArrayType )  {^75^^^^^69^83^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ^  type  ^  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  &  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >=  type  >=  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ||  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >=  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^} else if  ( type  |  ParameterizedType )  {^72^^^^^69^83^} else if  ( type instanceof ParameterizedType )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^} else if  ( type  ^  ParameterizedType )  {^72^^^^^69^83^} else if  ( type instanceof ParameterizedType )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^} else if  ( type  <<  GenericArrayType )  {^75^^^^^69^83^} else if  ( type instanceof GenericArrayType )  {^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  ==  type  ==  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <<  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  >  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  &  type  &  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  <  type  <  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  |  type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Variable_Misuse]^return wrapWithArray ( 3 ) ;^78^^^^^69^83^return wrapWithArray ( rawClass ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Type \'"  |  type  |  "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^80^81^^^^69^83^throw new IllegalArgumentException ( "Type \'" + type + "\' is not a Class, " + "ParameterizedType, or GenericArrayType. Can't extract class." ) ;^[CLASS] TypeUtils  [METHOD] toRawClass [RETURN_TYPE] Class   Type type [VARIABLES] ParameterizedType  actualType  Type  type  boolean  Class  rawClass  GenericArrayType  actualType  
[BugLab_Wrong_Literal]^return Array.newInstance ( rawClass, -1 ) .getClass (  ) ;^86^^^^^85^87^return Array.newInstance ( rawClass, 0 ) .getClass (  ) ;^[CLASS] TypeUtils  [METHOD] wrapWithArray [RETURN_TYPE] Class   Class<?> rawClass [VARIABLES] boolean  Class  rawClass  
[BugLab_Wrong_Literal]^return Array.newInstance ( rawClass, 1 ) .getClass (  ) ;^86^^^^^85^87^return Array.newInstance ( rawClass, 0 ) .getClass (  ) ;^[CLASS] TypeUtils  [METHOD] wrapWithArray [RETURN_TYPE] Class   Class<?> rawClass [VARIABLES] boolean  Class  rawClass  