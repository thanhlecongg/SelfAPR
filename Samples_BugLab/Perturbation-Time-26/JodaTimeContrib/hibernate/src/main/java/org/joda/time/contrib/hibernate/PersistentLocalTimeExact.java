[buglab_swap_variables]^if  ( y == x )  {^49^^^^^48^58^if  ( x == y )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[buglab_swap_variables]^if  ( y == null || x == null )  {^52^^^^^48^58^if  ( x == null || y == null )  {^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[buglab_swap_variables]^return dty.equals ( dtx ) ;^57^^^^^48^58^return dtx.equals ( dty ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] equals [RETURN_TYPE] boolean   Object x Object y [VARIABLES] PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  x  y  boolean  LocalTime  dtx  dty  
[buglab_swap_variables]^return nullSafeGet ( strings, resultSet[0] ) ;^65^^^^^64^67^return nullSafeGet ( resultSet, strings[0] ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String[] strings Object object [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  object  String[]  strings  boolean  
[buglab_swap_variables]^return nullSafeGet (  strings[0] ) ;^65^^^^^64^67^return nullSafeGet ( resultSet, strings[0] ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String[] strings Object object [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  object  String[]  strings  boolean  
[buglab_swap_variables]^return nullSafeGet ( resultSet[0] ) ;^65^^^^^64^67^return nullSafeGet ( resultSet, strings[0] ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String[] strings Object object [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  object  String[]  strings  boolean  
[buglab_swap_variables]^Object timestamp = Hibernate.INTEGER.nullSafeGet ( string, resultSet ) ;^70^^^^^69^76^Object timestamp = Hibernate.INTEGER.nullSafeGet ( resultSet, string ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String string [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  timestamp  String  string  boolean  
[buglab_swap_variables]^Object timestamp = Hibernate.INTEGER.nullSafeGet (  string ) ;^70^^^^^69^76^Object timestamp = Hibernate.INTEGER.nullSafeGet ( resultSet, string ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String string [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  timestamp  String  string  boolean  
[buglab_swap_variables]^Object timestamp = Hibernate.INTEGER.nullSafeGet ( resultSet ) ;^70^^^^^69^76^Object timestamp = Hibernate.INTEGER.nullSafeGet ( resultSet, string ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeGet [RETURN_TYPE] Object   ResultSet resultSet String string [VARIABLES] ResultSet  resultSet  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  Object  timestamp  String  string  boolean  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet ( lt, new Integer ( preparedStatement.getMillisOfDay (  )  ) , index ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet (  new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( index.getMillisOfDay (  )  ) , lt ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  )  ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet ( index, null, preparedStatement ) ;^80^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, null, index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet (  null, index ) ;^80^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, null, index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet ( preparedStatement, null ) ;^80^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, null, index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  
[buglab_swap_variables]^Hibernate.INTEGER.nullSafeSet ( index, new Integer ( lt.getMillisOfDay (  )  ) , preparedStatement ) ;^83^^^^^78^85^Hibernate.INTEGER.nullSafeSet ( preparedStatement, new Integer ( lt.getMillisOfDay (  )  ) , index ) ;^[CLASS] PersistentLocalTimeExact  [METHOD] nullSafeSet [RETURN_TYPE] void   PreparedStatement preparedStatement Object value int index [VARIABLES] boolean  LocalTime  lt  PersistentLocalTimeExact  INSTANCE  int[]  SQL_TYPES  PreparedStatement  preparedStatement  Object  value  int  index  