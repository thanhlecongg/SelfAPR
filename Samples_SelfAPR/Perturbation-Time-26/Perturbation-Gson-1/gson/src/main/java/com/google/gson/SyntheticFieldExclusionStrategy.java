[P8_Replace_Mix]^this.skipSyntheticFields =  null;^33^^^^^32^34^this.skipSyntheticFields = skipSyntheticFields;^[CLASS] SyntheticFieldExclusionStrategy  [METHOD] <init> [RETURN_TYPE] SyntheticFieldExclusionStrategy(boolean)   boolean skipSyntheticFields [VARIABLES] boolean  skipSyntheticFields  
[P3_Replace_Literal]^return true;^37^^^^^36^38^return false;^[CLASS] SyntheticFieldExclusionStrategy  [METHOD] shouldSkipClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] Class  clazz  boolean  skipSyntheticFields  
[P2_Replace_Operator]^return skipSyntheticFields || f.isSynthetic (  ) ;^41^^^^^40^42^return skipSyntheticFields && f.isSynthetic (  ) ;^[CLASS] SyntheticFieldExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] FieldAttributes  f  boolean  skipSyntheticFields  
[P5_Replace_Variable]^return f && skipSyntheticFields.isSynthetic (  ) ;^41^^^^^40^42^return skipSyntheticFields && f.isSynthetic (  ) ;^[CLASS] SyntheticFieldExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] FieldAttributes  f  boolean  skipSyntheticFields  
[P8_Replace_Mix]^return   f.isSynthetic (  ) ;^41^^^^^40^42^return skipSyntheticFields && f.isSynthetic (  ) ;^[CLASS] SyntheticFieldExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] FieldAttributes  f  boolean  skipSyntheticFields  
[P14_Delete_Statement]^^41^^^^^40^42^return skipSyntheticFields && f.isSynthetic (  ) ;^[CLASS] SyntheticFieldExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] FieldAttributes  f  boolean  skipSyntheticFields  