[buglab_swap_variables]^TypeSerializer d2 = prop.forProperty ( _delegate ) ;^42^^^^^41^47^TypeSerializer d2 = _delegate.forProperty ( prop ) ;^[CLASS] TypeSerializerWrapper  [METHOD] forProperty [RETURN_TYPE] TypeSerializer   BeanProperty prop [VARIABLES] TypeSerializer  _delegate  d2  delegate  Object  _value  value  boolean  BeanProperty  prop  
[buglab_swap_variables]^if  ( _delegate == d2 )  {^43^^^^^41^47^if  ( d2 == _delegate )  {^[CLASS] TypeSerializerWrapper  [METHOD] forProperty [RETURN_TYPE] TypeSerializer   BeanProperty prop [VARIABLES] TypeSerializer  _delegate  d2  delegate  Object  _value  value  boolean  BeanProperty  prop  
[buglab_swap_variables]^return new TypeSerializerWrapper ( _value, d2 ) ;^46^^^^^41^47^return new TypeSerializerWrapper ( d2, _value ) ;^[CLASS] TypeSerializerWrapper  [METHOD] forProperty [RETURN_TYPE] TypeSerializer   BeanProperty prop [VARIABLES] TypeSerializer  _delegate  d2  delegate  Object  _value  value  boolean  BeanProperty  prop  
[buglab_swap_variables]^return new TypeSerializerWrapper (  _value ) ;^46^^^^^41^47^return new TypeSerializerWrapper ( d2, _value ) ;^[CLASS] TypeSerializerWrapper  [METHOD] forProperty [RETURN_TYPE] TypeSerializer   BeanProperty prop [VARIABLES] TypeSerializer  _delegate  d2  delegate  Object  _value  value  boolean  BeanProperty  prop  
[buglab_swap_variables]^return new TypeSerializerWrapper ( d2 ) ;^46^^^^^41^47^return new TypeSerializerWrapper ( d2, _value ) ;^[CLASS] TypeSerializerWrapper  [METHOD] forProperty [RETURN_TYPE] TypeSerializer   BeanProperty prop [VARIABLES] TypeSerializer  _delegate  d2  delegate  Object  _value  value  boolean  BeanProperty  prop  
[buglab_swap_variables]^_delegate.writeTypePrefixForScalar ( _value ) ;^73^^^^^71^74^_delegate.writeTypePrefixForScalar ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypePrefixForScalar (  jgen ) ;^73^^^^^71^74^_delegate.writeTypePrefixForScalar ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypePrefixForObject ( jgen, _value ) ;^79^^^^^77^80^_delegate.writeTypePrefixForObject ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypePrefixForObject ( _value ) ;^79^^^^^77^80^_delegate.writeTypePrefixForObject ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypePrefixForObject (  jgen ) ;^79^^^^^77^80^_delegate.writeTypePrefixForObject ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypePrefixForArray ( jgen, _value ) ;^85^^^^^83^86^_delegate.writeTypePrefixForArray ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypePrefixForArray ( _value ) ;^85^^^^^83^86^_delegate.writeTypePrefixForArray ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypePrefixForArray (  jgen ) ;^85^^^^^83^86^_delegate.writeTypePrefixForArray ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForScalar ( jgen, _value ) ;^91^^^^^89^92^_delegate.writeTypeSuffixForScalar ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForScalar ( _value ) ;^91^^^^^89^92^_delegate.writeTypeSuffixForScalar ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForScalar (  jgen ) ;^91^^^^^89^92^_delegate.writeTypeSuffixForScalar ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForObject ( jgen, _value ) ;^97^^^^^95^98^_delegate.writeTypeSuffixForObject ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForObject ( _value ) ;^97^^^^^95^98^_delegate.writeTypeSuffixForObject ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForObject (  jgen ) ;^97^^^^^95^98^_delegate.writeTypeSuffixForObject ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForArray ( _value ) ;^103^^^^^101^104^_delegate.writeTypeSuffixForArray ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeTypeSuffixForArray (  jgen ) ;^103^^^^^101^104^_delegate.writeTypeSuffixForArray ( _value, jgen ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeTypeSuffixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForScalar ( _value,  typeId ) ;^109^^^^^107^110^_delegate.writeCustomTypePrefixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForScalar ( _value, jgen ) ;^109^^^^^107^110^_delegate.writeCustomTypePrefixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForScalar ( typeId, jgen, _value ) ;^109^^^^^107^110^_delegate.writeCustomTypePrefixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForScalar (  jgen, typeId ) ;^109^^^^^107^110^_delegate.writeCustomTypePrefixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForObject ( _value, typeId, jgen ) ;^115^^^^^113^116^_delegate.writeCustomTypePrefixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForObject ( _value,  typeId ) ;^115^^^^^113^116^_delegate.writeCustomTypePrefixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForObject ( typeId, jgen, _value ) ;^115^^^^^113^116^_delegate.writeCustomTypePrefixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForObject ( _value, jgen ) ;^115^^^^^113^116^_delegate.writeCustomTypePrefixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForObject (  jgen, typeId ) ;^115^^^^^113^116^_delegate.writeCustomTypePrefixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForArray ( jgen, _value, typeId ) ;^121^^^^^119^122^_delegate.writeCustomTypePrefixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForArray ( _value,  typeId ) ;^121^^^^^119^122^_delegate.writeCustomTypePrefixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForArray ( _value, typeId, jgen ) ;^121^^^^^119^122^_delegate.writeCustomTypePrefixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForArray ( _value, jgen ) ;^121^^^^^119^122^_delegate.writeCustomTypePrefixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForArray ( typeId, jgen, _value ) ;^121^^^^^119^122^_delegate.writeCustomTypePrefixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypePrefixForArray (  jgen, typeId ) ;^121^^^^^119^122^_delegate.writeCustomTypePrefixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypePrefixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForScalar ( _value, typeId, jgen ) ;^127^^^^^125^128^_delegate.writeCustomTypeSuffixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForScalar ( _value,  typeId ) ;^127^^^^^125^128^_delegate.writeCustomTypeSuffixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForScalar ( typeId, jgen, _value ) ;^127^^^^^125^128^_delegate.writeCustomTypeSuffixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForScalar ( _value, jgen ) ;^127^^^^^125^128^_delegate.writeCustomTypeSuffixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForScalar (  jgen, typeId ) ;^127^^^^^125^128^_delegate.writeCustomTypeSuffixForScalar ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForScalar [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForObject ( jgen, _value, typeId ) ;^134^^^^^131^135^_delegate.writeCustomTypeSuffixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForObject ( _value,  typeId ) ;^134^^^^^131^135^_delegate.writeCustomTypeSuffixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForObject ( _value, typeId, jgen ) ;^134^^^^^131^135^_delegate.writeCustomTypeSuffixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForObject ( _value, jgen ) ;^134^^^^^131^135^_delegate.writeCustomTypeSuffixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForObject (  jgen, typeId ) ;^134^^^^^131^135^_delegate.writeCustomTypeSuffixForObject ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForObject [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForArray ( _value, typeId, jgen ) ;^140^^^^^138^141^_delegate.writeCustomTypeSuffixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForArray ( _value,  typeId ) ;^140^^^^^138^141^_delegate.writeCustomTypeSuffixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForArray ( typeId, jgen, _value ) ;^140^^^^^138^141^_delegate.writeCustomTypeSuffixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForArray ( _value, jgen ) ;^140^^^^^138^141^_delegate.writeCustomTypeSuffixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  
[buglab_swap_variables]^_delegate.writeCustomTypeSuffixForArray (  jgen, typeId ) ;^140^^^^^138^141^_delegate.writeCustomTypeSuffixForArray ( _value, jgen, typeId ) ;^[CLASS] TypeSerializerWrapper  [METHOD] writeCustomTypeSuffixForArray [RETURN_TYPE] void   Object value JsonGenerator jgen String typeId [VARIABLES] TypeSerializer  _delegate  d2  delegate  JsonGenerator  jgen  Object  _value  value  String  typeId  boolean  