[REPLACE]^private Object device;^33^^^^^^^[REPLACE] private Device device;^ [CLASS] GroupTree TreeNode  
[REPLACE]^private Collection<Object children = new HashSet<> (  ) ;^34^^^^^^^[REPLACE] private Collection<TreeNode> children = new HashSet<> (  ) ;^ [CLASS] GroupTree TreeNode  
[REPLACE]^private  Map<Long, TreeNode> groupMap = new HashMap<> (  ) ;^90^^^^^^^[REPLACE] private final Map<Long, TreeNode> groupMap = new HashMap<> (  ) ;^ [CLASS] GroupTree TreeNode  
[REPLACE]^for  ( Group group : null )  {^94^^^^^92^116^[REPLACE] for  ( Group group : groups )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^groupMap.put ( group.getGroupId (  ) , new TreeNode ( group )  ) ;^95^^^^^92^116^[REPLACE] groupMap.put ( group.getId (  ) , new TreeNode ( group )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REMOVE]^groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ;^95^^^^^92^116^[REMOVE] ^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[ADD]^^95^^^^^92^116^[ADD] groupMap.put ( group.getId (  ) , new TreeNode ( group )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^Device device  ;^94^^^^^92^116^[REPLACE] for  ( Group group : groups )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^if  ( node.getGroup (  ) .getGroupId (  )   ==  1 )  {^99^^^^^92^116^[REPLACE] if  ( node.getGroup (  ) .getGroupId (  )  != 0 )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[ADD]^node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^99^100^101^^^92^116^[ADD] if  ( node.getGroup (  ) .getGroupId (  )  != 0 )  { node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ; }^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  )  ;^100^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ;^100^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^( node.getGroup (  ) .getGroupId (  )  )  ;^100^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^100^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^if  ( node.getGroup (  ) .getGroupId (  )   ==  0 )  {^99^^^^^92^116^[REPLACE] if  ( node.getGroup (  ) .getGroupId (  )  != 0 )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[ADD]^^100^^^^^92^116^[ADD] node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^( node.getGroup (  )  )  ;^100^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[ADD]^^98^99^100^101^^92^116^[ADD] for  ( TreeNode node : groupMap.values (  )  )  { if  ( node.getGroup (  ) .getGroupId (  )  != 0 )  { node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ; }^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^TreeNode> results = new HashSet<> (  )  ;^104^^^^^92^116^[REPLACE] Map<Long, TreeNode> deviceMap = new HashMap<> (  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^deviceMap.get ( device.getId (  ) , new TreeNode ( device )  ) ;^107^^^^^92^116^[REPLACE] deviceMap.put ( device.getId (  ) , new TreeNode ( device )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^( device.getId (  )  )  ;^107^^^^^92^116^[REPLACE] deviceMap.put ( device.getId (  ) , new TreeNode ( device )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^Group group  ;^106^^^^^92^116^[REPLACE] for  ( Device device : devices )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^if  ( node.getDevice (  ) .getGroupId (  )   ==  3 )  {^111^^^^^92^116^[REPLACE] if  ( node.getDevice (  ) .getGroupId (  )  != 0 )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[ADD]^^111^112^113^^^92^116^[ADD] if  ( node.getDevice (  ) .getGroupId (  )  != 0 )  { node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ; }^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  )  ;^112^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ;^112^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[ADD]^^112^^^^^92^116^[ADD] node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^node.setParent ( groupMap.get ( node.getGroup (  ) .getGroupId (  )  )  ) ;^112^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[ADD]^^110^111^112^113^^92^116^[ADD] for  ( TreeNode node : deviceMap.values (  )  )  { if  ( node.getDevice (  ) .getGroupId (  )  != 0 )  { node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ; }^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^if  ( node.getDevice (  ) .getGroupId (  )   ==  0 )  {^111^^^^^92^116^[REPLACE] if  ( node.getDevice (  ) .getGroupId (  )  != 0 )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^node.setParent ( groupMap .put ( null , 0 )  ( node^112^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^( node.getDevice (  ) .getGroupId (  )  )  ;^112^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^( node.getDevice (  )  )  ;^112^^^^^92^116^[REPLACE] node.setParent ( groupMap.get ( node.getDevice (  ) .getGroupId (  )  )  ) ;^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^TreeNode child  ;^110^^^^^92^116^[REPLACE] for  ( TreeNode node : deviceMap.values (  )  )  {^[METHOD] <init> [TYPE] Collection) [PARAMETER] Group> groups Device> devices [CLASS] GroupTree TreeNode   [TYPE]  Group group  [TYPE]  Collection children  devices  groups  [TYPE]  boolean false  true  [TYPE]  Map deviceMap  groupMap  [TYPE]  Device device  [TYPE]  TreeNode node 
[REPLACE]^private Object group;^32^^^^^^^[REPLACE] private Group group;^[METHOD] getNodes [TYPE] void [PARAMETER] TreeNode> results TreeNode node [CLASS] TreeNode   [TYPE]  Group group  [TYPE]  Set results  [TYPE]  boolean false  true  [TYPE]  Device device  [TYPE]  TreeNode child  node  [TYPE]  Collection children  devices  groups  [TYPE]  Map deviceMap  groupMap 
[REPLACE]^private Collection<Device children = new HashSet<> (  ) ;^34^^^^^^^[REPLACE] private Collection<TreeNode> children = new HashSet<> (  ) ;^[METHOD] getNodes [TYPE] void [PARAMETER] TreeNode> results TreeNode node [CLASS] TreeNode   [TYPE]  Group group  [TYPE]  Set results  [TYPE]  boolean false  true  [TYPE]  Device device  [TYPE]  TreeNode child  node  [TYPE]  Collection children  devices  groups  [TYPE]  Map deviceMap  groupMap 
