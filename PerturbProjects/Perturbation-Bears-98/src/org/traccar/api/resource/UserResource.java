[REPLACE]^if  ( Context.getPermissionsManager (  ) .isAdmin ( getUserId (  )  )  )  {^51^^^^^50^66^[REPLACE] if  ( !Context.getPermissionsManager (  ) .isAdmin ( getUserId (  )  )  )  {^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[ADD]^^51^52^53^54^55^50^66^[ADD] if  ( !Context.getPermissionsManager (  ) .isAdmin ( getUserId (  )  )  )  { Context.getPermissionsManager (  ) .checkRegistration ( getUserId (  )  ) ; Context.getPermissionsManager (  ) .checkUserUpdate ( getUserId (  ) , new User (  ) , entity ) ; entity.setDeviceLimit ( Context.getConfig (  ) .getInteger ( "users.defaultDeviceLimit" )  ) ; int expirationDays = Context.getConfig (  ) .getInteger ( "users.defaultExpirationDays" ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^if  ( expirationDays  <  3 )  {^56^^^^^50^66^[REPLACE] if  ( expirationDays > 0 )  {^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^entity.setExpirationTime ( new Date ( System.currentTimeMillis (  )    ( long )  expirationDays * 24 * 3600 * 1000 )  ) ;^57^58^^^^50^66^[REPLACE] entity.setExpirationTime ( new Date ( System.currentTimeMillis (  )  +  ( long )  expirationDays * 24 * 3600 * 1000 )  ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[ADD]^^57^58^^^^50^66^[ADD] entity.setExpirationTime ( new Date ( System.currentTimeMillis (  )  +  ( long )  expirationDays * 24 * 3600 * 1000 )  ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^int expirationDays = Context.getConfig (  ) .removeUser ( "users.defaultExpirationDays" ) ;^55^^^^^50^66^[REPLACE] int expirationDays = Context.getConfig (  ) .getInteger ( "users.defaultExpirationDays" ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^if  ( expirationDays  <=  2 )  {^56^^^^^50^66^[REPLACE] if  ( expirationDays > 0 )  {^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^int expirationDays = Context.getConfig (  )  .getUser (  )  ;^55^^^^^50^66^[REPLACE] int expirationDays = Context.getConfig (  ) .getInteger ( "users.defaultExpirationDays" ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^Context.getPermissionsManager (  ) .updateUser ( entity ) ;^61^^^^^50^66^[REPLACE] Context.getPermissionsManager (  ) .addUser ( entity ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^Context.getNotificationManager (  ) .addUser ( entity ) ;^61^^^^^50^66^[REPLACE] Context.getPermissionsManager (  ) .addUser ( entity ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^if  ( Context .getPermissionsManager (   )  || expirationDays > 0  )   == false )  {^62^^^^^50^66^[REPLACE] if  ( Context.getNotificationManager (  )  != null )  {^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[ADD]^Context.getNotificationManager (  ) .refresh (  ) ;^62^63^64^^^50^66^[ADD] if  ( Context.getNotificationManager (  )  != null )  { Context.getNotificationManager (  ) .refresh (  ) ; }^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^Context.getNotificationManager (  )  .removeUser ( true )  ;^63^^^^^50^66^[REPLACE] Context.getNotificationManager (  ) .refresh (  ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^Context .getPermissionsManager (  )  .refresh (  ) ;^63^^^^^50^66^[REPLACE] Context.getNotificationManager (  ) .refresh (  ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
[REPLACE]^return noContent (  ) .build (  )  ;^65^^^^^50^66^[REPLACE] return Response.ok ( entity ) .build (  ) ;^[METHOD] add [TYPE] Response [PARAMETER] User entity [CLASS] UserResource   [TYPE]  User entity  [TYPE]  boolean false  true  [TYPE]  int expirationDays 
