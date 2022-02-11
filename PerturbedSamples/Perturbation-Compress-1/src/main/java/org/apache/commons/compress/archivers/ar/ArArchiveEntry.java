[REPLACE]^private final  char  name;^28^^^^^^^[REPLACE] private final String name;^ [CLASS] ArArchiveEntry  
[REPLACE]^private  int userId;^29^^^^^^^[REPLACE] private final int userId;^ [CLASS] ArArchiveEntry  
[REPLACE]^private  int groupId;^30^^^^^^^[REPLACE] private final int groupId;^ [CLASS] ArArchiveEntry  
[REPLACE]^private  int mode;^31^^^^^^^[REPLACE] private final int mode;^ [CLASS] ArArchiveEntry  
[REPLACE]^this ( name, length, 0 , 0 , 33188, System.currentTimeMillis (  )  ) ;^36^^^^^35^37^[REPLACE] this ( name, length, 0, 0, 33188, System.currentTimeMillis (  )  ) ;^[METHOD] <init> [TYPE] String,long) [PARAMETER] String name long length [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[ADD]^^36^^^^^35^37^[ADD] this ( name, length, 0, 0, 33188, System.currentTimeMillis (  )  ) ;^[METHOD] <init> [TYPE] String,long) [PARAMETER] String name long length [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return this.getLength (  ) ;^36^^^^^35^37^[REPLACE] this ( name, length, 0, 0, 33188, System.currentTimeMillis (  )  ) ;^[METHOD] <init> [TYPE] String,long) [PARAMETER] String name long length [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^this.name =  null;^40^^^^^39^46^[REPLACE] this.name = name;^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^this.length =  lastModified;^41^^^^^39^46^[REPLACE] this.length = length;^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^this.userId =  groupId;^42^^^^^39^46^[REPLACE] this.userId = userId;^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^this.groupId =  userId;^43^^^^^39^46^[REPLACE] this.groupId = groupId;^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^this.mode =  null;^44^^^^^39^46^[REPLACE] this.mode = mode;^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[ADD]^^44^^^^^39^46^[ADD] this.mode = mode;^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^this.lastModified =  length;^45^^^^^39^46^[REPLACE] this.lastModified = lastModified;^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[ADD]^^45^46^^^^39^46^[ADD] this.lastModified = lastModified; }^[METHOD] <init> [TYPE] String,long,int,int,int,long) [PARAMETER] String name long length int userId int groupId int mode long lastModified [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return length;^49^^^^^48^50^[REPLACE] return this.getLength (  ) ;^[METHOD] getSize [TYPE] long [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return length;^53^^^^^52^54^[REPLACE] return name;^[METHOD] getName [TYPE] String [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return length;^57^^^^^56^58^[REPLACE] return userId;^[METHOD] getUserId [TYPE] int [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return length;^61^^^^^60^62^[REPLACE] return groupId;^[METHOD] getGroupId [TYPE] int [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return length;^65^^^^^64^66^[REPLACE] return mode;^[METHOD] getMode [TYPE] int [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return length;^69^^^^^68^70^[REPLACE] return lastModified;^[METHOD] getLastModified [TYPE] long [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return userId;^73^^^^^72^74^[REPLACE] return length;^[METHOD] getLength [TYPE] long [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 
[REPLACE]^return true;^77^^^^^76^78^[REPLACE] return false;^[METHOD] isDirectory [TYPE] boolean [PARAMETER] [CLASS] ArArchiveEntry   [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  int groupId  mode  userId  [TYPE]  long lastModified  length 