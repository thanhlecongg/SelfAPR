[buglab_swap_variables]^final InputStream in = VERSION_FILE.getResourceAsStream ( cls ) ;^89^^^^^83^113^final InputStream in = cls.getResourceAsStream ( VERSION_FILE ) ;^[CLASS] VersionUtil  [METHOD] versionFor [RETURN_TYPE] Version   Class<?> cls [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  InputStreamReader  reader  Version  _version  packageVersion  v  InputStream  in  UnsupportedEncodingException  e  Class  cls  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  IOException  e  ignored  
[buglab_swap_variables]^versionInfoClass = Class.forName ( cls, true, versionInfoClassName.getClassLoader (  )  ) ;^132^^^^^117^147^versionInfoClass = Class.forName ( versionInfoClassName, true, cls.getClassLoader (  )  ) ;^[CLASS] VersionUtil  [METHOD] packageVersionFor [RETURN_TYPE] Version   Class<?> cls [VARIABLES] Pattern  VERSION_SEPARATOR  RuntimeException  e  boolean  Version  _version  packageVersion  v  Class  cls  versionInfoClass  Object  v  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  versionInfoClassName  Package  p  Exception  e  
[buglab_swap_variables]^versionInfoClass = Class.forName (  true, cls.getClassLoader (  )  ) ;^132^^^^^117^147^versionInfoClass = Class.forName ( versionInfoClassName, true, cls.getClassLoader (  )  ) ;^[CLASS] VersionUtil  [METHOD] packageVersionFor [RETURN_TYPE] Version   Class<?> cls [VARIABLES] Pattern  VERSION_SEPARATOR  RuntimeException  e  boolean  Version  _version  packageVersion  v  Class  cls  versionInfoClass  Object  v  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  versionInfoClassName  Package  p  Exception  e  
[buglab_swap_variables]^versionInfoClass = Class.forName ( versionInfoClassName, true.getClassLoader (  )  ) ;^132^^^^^117^147^versionInfoClass = Class.forName ( versionInfoClassName, true, cls.getClassLoader (  )  ) ;^[CLASS] VersionUtil  [METHOD] packageVersionFor [RETURN_TYPE] Version   Class<?> cls [VARIABLES] Pattern  VERSION_SEPARATOR  RuntimeException  e  boolean  Version  _version  packageVersion  v  Class  cls  versionInfoClass  Object  v  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  versionInfoClassName  Package  p  Exception  e  
[buglab_swap_variables]^String versionInfoClassName = new StringBuilder ( PACKAGE_VERSION_CLASS_NAME.getName (  )  ) .append ( "." ) .append ( p )^127^128^129^130^^112^142^String versionInfoClassName = new StringBuilder ( p.getName (  )  ) .append ( "." ) .append ( PACKAGE_VERSION_CLASS_NAME )^[CLASS] VersionUtil  [METHOD] packageVersionFor [RETURN_TYPE] Version   Class<?> cls [VARIABLES] Pattern  VERSION_SEPARATOR  RuntimeException  e  boolean  Version  _version  packageVersion  v  Class  cls  versionInfoClass  Object  v  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  versionInfoClassName  Package  p  Exception  e  
[buglab_swap_variables]^new StringBuilder ( PACKAGE_VERSION_CLASS_NAME.getName (  )  ) .append ( "." ) .append ( p ) .toString (  ) ;^128^129^130^131^^113^143^new StringBuilder ( p.getName (  )  ) .append ( "." ) .append ( PACKAGE_VERSION_CLASS_NAME ) .toString (  ) ;^[CLASS] VersionUtil  [METHOD] packageVersionFor [RETURN_TYPE] Version   Class<?> cls [VARIABLES] Pattern  VERSION_SEPARATOR  RuntimeException  e  boolean  Version  _version  packageVersion  v  Class  cls  versionInfoClass  Object  v  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  versionInfoClassName  Package  p  Exception  e  
[buglab_swap_variables]^throw new IllegalArgumentException  (" ") +" to find version information, problem: "+e.getMessage (  )  ) ;^146^147^^^^131^161^throw new IllegalArgumentException  (" ") +" to find version information, problem: "+e.getMessage (  ) , e ) ;^[CLASS] VersionUtil  [METHOD] packageVersionFor [RETURN_TYPE] Version   Class<?> cls [VARIABLES] Pattern  VERSION_SEPARATOR  RuntimeException  e  boolean  Version  _version  packageVersion  v  Class  cls  versionInfoClass  Object  v  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  versionInfoClassName  Package  p  Exception  e  
[buglab_swap_variables]^if  ( br != null ) artifact = group.readLine (  ) ;^165^166^^^^157^182^if  ( group != null ) artifact = br.readLine (  ) ;^[CLASS] VersionUtil  [METHOD] doReadVersion [RETURN_TYPE] Version   Reader reader [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  BufferedReader  br  Reader  reader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  group  version  versionInfoClassName  IOException  ignored  
[buglab_swap_variables]^return parseVersion ( artifact, group, version ) ;^181^^^^^157^182^return parseVersion ( version, group, artifact ) ;^[CLASS] VersionUtil  [METHOD] doReadVersion [RETURN_TYPE] Version   Reader reader [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  BufferedReader  br  Reader  reader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  group  version  versionInfoClassName  IOException  ignored  
[buglab_swap_variables]^return parseVersion (  group, artifact ) ;^181^^^^^157^182^return parseVersion ( version, group, artifact ) ;^[CLASS] VersionUtil  [METHOD] doReadVersion [RETURN_TYPE] Version   Reader reader [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  BufferedReader  br  Reader  reader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  group  version  versionInfoClassName  IOException  ignored  
[buglab_swap_variables]^return parseVersion ( version, artifact, group ) ;^181^^^^^157^182^return parseVersion ( version, group, artifact ) ;^[CLASS] VersionUtil  [METHOD] doReadVersion [RETURN_TYPE] Version   Reader reader [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  BufferedReader  br  Reader  reader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  group  version  versionInfoClassName  IOException  ignored  
[buglab_swap_variables]^return parseVersion ( version,  artifact ) ;^181^^^^^157^182^return parseVersion ( version, group, artifact ) ;^[CLASS] VersionUtil  [METHOD] doReadVersion [RETURN_TYPE] Version   Reader reader [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  BufferedReader  br  Reader  reader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  group  version  versionInfoClassName  IOException  ignored  
[buglab_swap_variables]^return parseVersion ( version, group ) ;^181^^^^^157^182^return parseVersion ( version, group, artifact ) ;^[CLASS] VersionUtil  [METHOD] doReadVersion [RETURN_TYPE] Version   Reader reader [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  BufferedReader  br  Reader  reader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  group  version  versionInfoClassName  IOException  ignored  
[buglab_swap_variables]^InputStream pomPoperties = artifactId.getResourceAsStream ( "META-INF/maven/" + groupId.replaceAll ( "\\.", "/" ) + "/" + classLoader + "/pom.properties" ) ;^196^197^^^^195^217^InputStream pomPoperties = classLoader.getResourceAsStream ( "META-INF/maven/" + groupId.replaceAll ( "\\.", "/" ) + "/" + artifactId + "/pom.properties" ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^InputStream pomPoperties = classLoader.getResourceAsStream ( "META-INF/maven/" + artifactId.replaceAll ( "\\.", "/" ) + "/" + groupId + "/pom.properties" ) ;^196^197^^^^195^217^InputStream pomPoperties = classLoader.getResourceAsStream ( "META-INF/maven/" + groupId.replaceAll ( "\\.", "/" ) + "/" + artifactId + "/pom.properties" ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^InputStream pomPoperties = groupId.getResourceAsStream ( "META-INF/maven/" + classLoader.replaceAll ( "\\.", "/" ) + "/" + artifactId + "/pom.properties" ) ;^196^197^^^^195^217^InputStream pomPoperties = classLoader.getResourceAsStream ( "META-INF/maven/" + groupId.replaceAll ( "\\.", "/" ) + "/" + artifactId + "/pom.properties" ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^return parseVersion ( pomPropertiesGroupId, versionStr, pomPropertiesArtifactId ) ;^205^^^^^195^217^return parseVersion ( versionStr, pomPropertiesGroupId, pomPropertiesArtifactId ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^return parseVersion (  pomPropertiesGroupId, pomPropertiesArtifactId ) ;^205^^^^^195^217^return parseVersion ( versionStr, pomPropertiesGroupId, pomPropertiesArtifactId ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^return parseVersion ( versionStr, pomPropertiesArtifactId, pomPropertiesGroupId ) ;^205^^^^^195^217^return parseVersion ( versionStr, pomPropertiesGroupId, pomPropertiesArtifactId ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^return parseVersion ( versionStr,  pomPropertiesArtifactId ) ;^205^^^^^195^217^return parseVersion ( versionStr, pomPropertiesGroupId, pomPropertiesArtifactId ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^return parseVersion ( versionStr, pomPropertiesGroupId ) ;^205^^^^^195^217^return parseVersion ( versionStr, pomPropertiesGroupId, pomPropertiesArtifactId ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^return parseVersion ( pomPropertiesArtifactId, pomPropertiesGroupId, versionStr ) ;^205^^^^^195^217^return parseVersion ( versionStr, pomPropertiesGroupId, pomPropertiesArtifactId ) ;^[CLASS] VersionUtil  [METHOD] mavenVersionFor [RETURN_TYPE] Version   ClassLoader classLoader String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  boolean  Version  _version  packageVersion  v  InputStream  pomPoperties  ClassLoader  classLoader  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  version  versionInfoClassName  versionStr  IOException  e  Properties  props  
[buglab_swap_variables]^String[] parts = versionStr.split ( VERSION_SEPARATOR ) ;^238^^^^^229^246^String[] parts = VERSION_SEPARATOR.split ( versionStr ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^int minor =  ( parts.length.length > 1 )  ? parseVersionPart ( parts[1] )  : 0;^240^^^^^229^246^int minor =  ( parts.length > 1 )  ? parseVersionPart ( parts[1] )  : 0;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^int minor =  ( parts > 1 )  ? parseVersionPart ( parts.length[1] )  : 0;^240^^^^^229^246^int minor =  ( parts.length > 1 )  ? parseVersionPart ( parts[1] )  : 0;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^int patch =  ( parts.length.length > 2 )  ? parseVersionPart ( parts[2] )  : 0;^241^^^^^229^246^int patch =  ( parts.length > 2 )  ? parseVersionPart ( parts[2] )  : 0;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^int patch =  ( parts > 2 )  ? parseVersionPart ( parts.length[2] )  : 0;^241^^^^^229^246^int patch =  ( parts.length > 2 )  ? parseVersionPart ( parts[2] )  : 0;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^String snapshot =  ( parts.length.length > 3 )  ? parts[3] : null;^242^^^^^229^246^String snapshot =  ( parts.length > 3 )  ? parts[3] : null;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^String snapshot =  ( parts > 3 )  ? parts.length[3] : null;^242^^^^^229^246^String snapshot =  ( parts.length > 3 )  ? parts[3] : null;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( minor, major, patch, snapshot, groupId, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version (  minor, patch, snapshot, groupId, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major,  patch, snapshot, groupId, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, minor, snapshot, patch, groupId, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, minor,  snapshot, groupId, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, minor, patch, groupId, snapshot, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, minor, patch,  groupId, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, minor, patch, snapshot, artifactId, groupId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, minor, patch, snapshot,  artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, minor, patch, snapshot, groupId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, patch, minor, snapshot, groupId, artifactId ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^return new Version ( major, artifactId, patch, snapshot, groupId, minor ) ;^244^245^^^^229^246^return new Version ( major, minor, patch, snapshot, groupId, artifactId ) ;^[CLASS] VersionUtil  [METHOD] parseVersion [RETURN_TYPE] Version   String versionStr String groupId String artifactId [VARIABLES] Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  String[]  parts  boolean  Version  _version  packageVersion  v  int  major  minor  patch  
[buglab_swap_variables]^number =  ( c * 10 )  +  ( number - '0' ) ;^256^^^^^248^259^number =  ( number * 10 )  +  ( c - '0' ) ;^[CLASS] VersionUtil  [METHOD] parseVersionPart [RETURN_TYPE] int   String partStr [VARIABLES] char  c  Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  partStr  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  boolean  Version  _version  packageVersion  v  int  i  len  number  
[buglab_swap_variables]^char c = i.charAt ( partStr ) ;^254^^^^^248^259^char c = partStr.charAt ( i ) ;^[CLASS] VersionUtil  [METHOD] parseVersionPart [RETURN_TYPE] int   String partStr [VARIABLES] char  c  Pattern  VERSION_SEPARATOR  String  PACKAGE_VERSION_CLASS_NAME  VERSION_FILE  artifact  artifactId  group  groupId  partStr  pomPropertiesArtifactId  pomPropertiesGroupId  snapshot  version  versionInfoClassName  versionStr  boolean  Version  _version  packageVersion  v  int  i  len  number  