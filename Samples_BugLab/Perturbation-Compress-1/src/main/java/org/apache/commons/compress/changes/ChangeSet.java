[buglab_swap_variables]^changes.add ( new Change ( pInput, pEntry )  ) ;^47^^^^^46^48^changes.add ( new Change ( pEntry, pInput )  ) ;^[CLASS] ChangeSet  [METHOD] add [RETURN_TYPE] void   ArchiveEntry pEntry InputStream pInput [VARIABLES] ArchiveEntry  pEntry  Set  changes  boolean  InputStream  pInput  
[buglab_swap_variables]^changes.add ( new Change (  pInput )  ) ;^47^^^^^46^48^changes.add ( new Change ( pEntry, pInput )  ) ;^[CLASS] ChangeSet  [METHOD] add [RETURN_TYPE] void   ArchiveEntry pEntry InputStream pInput [VARIABLES] ArchiveEntry  pEntry  Set  changes  boolean  InputStream  pInput  
[buglab_swap_variables]^changes.add ( new Change ( pEntry )  ) ;^47^^^^^46^48^changes.add ( new Change ( pEntry, pInput )  ) ;^[CLASS] ChangeSet  [METHOD] add [RETURN_TYPE] void   ArchiveEntry pEntry InputStream pInput [VARIABLES] ArchiveEntry  pEntry  Set  changes  boolean  InputStream  pInput  
[buglab_swap_variables]^while  (  ( in = entry.getNextEntry (  )  )  != null )  {^57^^^^^42^72^while  (  ( entry = in.getNextEntry (  )  )  != null )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^if  ( Change.TYPE_ADD.type (  )  == change )  {^63^^^^^48^78^if  ( change.type (  )  == Change.TYPE_ADD )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( out.getInput (  ) , change, change.getEntry (  )  ) ;^64^^^^^49^79^copyStream ( change.getInput (  ) , out, change.getEntry (  )  ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( change.getInput (  ) , out.getEntry (  )  ) ;^64^^^^^49^79^copyStream ( change.getInput (  ) , out, change.getEntry (  )  ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( change.getInput (  ) ,  change.getEntry (  )  ) ;^64^^^^^49^79^copyStream ( change.getInput (  ) , out, change.getEntry (  )  ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^if  ( Change.TYPE_DELETE.type (  )  == change && entry.getName (  )  != null )  {^68^69^^^^53^83^if  ( change.type (  )  == Change.TYPE_DELETE && entry.getName (  )  != null )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^if  ( change.type (  )  == entry && Change.TYPE_DELETE.getName (  )  != null )  {^68^69^^^^53^83^if  ( change.type (  )  == Change.TYPE_DELETE && entry.getName (  )  != null )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^} else if  ( change.getName (  ) .matches ( entry.targetFile (  )  + "/.*" )  )  {^74^75^^^^68^79^} else if  ( entry.getName (  ) .matches ( change.targetFile (  )  + "/.*" )  )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^if  ( change.getName (  ) .equals ( entry.targetFile (  )  )  )  {^70^^^^^68^79^if  ( entry.getName (  ) .equals ( change.targetFile (  )  )  )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^if  ( change.getName (  ) .equals ( entry.targetFile (  )  )  )  {^70^^^^^55^85^if  ( entry.getName (  ) .equals ( change.targetFile (  )  )  )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^} else if  ( change.getName (  ) .matches ( entry.targetFile (  )  + "/.*" )  )  {^74^75^^^^59^89^} else if  ( entry.getName (  ) .matches ( change.targetFile (  )  + "/.*" )  )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( entry, out, in ) ;^84^^^^^69^99^copyStream ( in, out, entry ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream (  out, entry ) ;^84^^^^^69^99^copyStream ( in, out, entry ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( out, in, entry ) ;^84^^^^^69^99^copyStream ( in, out, entry ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( in,  entry ) ;^84^^^^^69^99^copyStream ( in, out, entry ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( in, out ) ;^84^^^^^69^99^copyStream ( in, out, entry ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^copyStream ( in, entry, out ) ;^84^^^^^69^99^copyStream ( in, out, entry ) ;^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^for  ( Iterator changes = it.iterator (  ) ; it.hasNext (  ) ; )  {^60^^^^^45^75^for  ( Iterator it = changes.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^if  ( entry.type (  )  == Change.TYPE_DELETE && change.getName (  )  != null )  {^68^69^^^^53^83^if  ( change.type (  )  == Change.TYPE_DELETE && entry.getName (  )  != null )  {^[CLASS] ChangeSet  [METHOD] perform [RETURN_TYPE] void   ArchiveInputStream in ArchiveOutputStream out [VARIABLES] ArchiveInputStream  in  Set  changes  boolean  copy  Iterator  it  ArchiveEntry  entry  Change  change  ArchiveOutputStream  out  
[buglab_swap_variables]^if  ( pChange != Change.TYPE_DELETE.type (  ) || pChange.targetFile (  )  == null )  {^91^92^^^^90^113^if  ( Change.TYPE_DELETE != pChange.type (  ) || pChange.targetFile (  )  == null )  {^[CLASS] ChangeSet  [METHOD] addDeletion [RETURN_TYPE] void   Change pChange [VARIABLES] Iterator  it  Set  changes  Change  change  pChange  String  source  target  boolean  
[buglab_swap_variables]^if  ( target.equals ( source )  )  {^104^^^^^90^113^if  ( source.equals ( target )  )  {^[CLASS] ChangeSet  [METHOD] addDeletion [RETURN_TYPE] void   Change pChange [VARIABLES] Iterator  it  Set  changes  Change  change  pChange  String  source  target  boolean  
[buglab_swap_variables]^} else if  ( source.matches ( target + "/.*" )  )  {^106^^^^^90^113^} else if  ( target.matches ( source + "/.*" )  )  {^[CLASS] ChangeSet  [METHOD] addDeletion [RETURN_TYPE] void   Change pChange [VARIABLES] Iterator  it  Set  changes  Change  change  pChange  String  source  target  boolean  
[buglab_swap_variables]^for  ( Iterator change = its.iterator (  ) ; it.hasNext (  ) ; )  {^98^^^^^90^113^for  ( Iterator it = changes.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] ChangeSet  [METHOD] addDeletion [RETURN_TYPE] void   Change pChange [VARIABLES] Iterator  it  Set  changes  Change  change  pChange  String  source  target  boolean  
[buglab_swap_variables]^if  ( Change.TYPE_ADD.type (  )  == change && change.getEntry (  )  != null )  {^100^101^^^^90^113^if  ( change.type (  )  == Change.TYPE_ADD && change.getEntry (  )  != null )  {^[CLASS] ChangeSet  [METHOD] addDeletion [RETURN_TYPE] void   Change pChange [VARIABLES] Iterator  it  Set  changes  Change  change  pChange  String  source  target  boolean  
[buglab_swap_variables]^if  ( Change.TYPE_DELETE.type (  )  == change )  {^121^^^^^115^133^if  ( change.type (  )  == Change.TYPE_DELETE )  {^[CLASS] ChangeSet  [METHOD] isDeletedLater [RETURN_TYPE] boolean   ArchiveEntry entry [VARIABLES] Iterator  it  ArchiveEntry  entry  Set  changes  String  source  target  Change  change  boolean  
[buglab_swap_variables]^if  ( target.equals ( source )  )  {^124^^^^^115^133^if  ( source.equals ( target )  )  {^[CLASS] ChangeSet  [METHOD] isDeletedLater [RETURN_TYPE] boolean   ArchiveEntry entry [VARIABLES] Iterator  it  ArchiveEntry  entry  Set  changes  String  source  target  Change  change  boolean  
[buglab_swap_variables]^return target.matches ( source + "/.*" ) ;^128^^^^^115^133^return source.matches ( target + "/.*" ) ;^[CLASS] ChangeSet  [METHOD] isDeletedLater [RETURN_TYPE] boolean   ArchiveEntry entry [VARIABLES] Iterator  it  ArchiveEntry  entry  Set  changes  String  source  target  Change  change  boolean  
[buglab_swap_variables]^for  ( Iterator change = its.iterator (  ) ; it.hasNext (  ) ; )  {^119^^^^^115^133^for  ( Iterator it = changes.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] ChangeSet  [METHOD] isDeletedLater [RETURN_TYPE] boolean   ArchiveEntry entry [VARIABLES] Iterator  it  ArchiveEntry  entry  Set  changes  String  source  target  Change  change  boolean  
[buglab_swap_variables]^for  ( Iterator changes = it.iterator (  ) ; it.hasNext (  ) ; )  {^119^^^^^115^133^for  ( Iterator it = changes.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] ChangeSet  [METHOD] isDeletedLater [RETURN_TYPE] boolean   ArchiveEntry entry [VARIABLES] Iterator  it  ArchiveEntry  entry  Set  changes  String  source  target  Change  change  boolean  
[buglab_swap_variables]^for  ( Iterator it = change.iterator (  ) ; it.hasNext (  ) ; )  {^119^^^^^115^133^for  ( Iterator it = changes.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] ChangeSet  [METHOD] isDeletedLater [RETURN_TYPE] boolean   ArchiveEntry entry [VARIABLES] Iterator  it  ArchiveEntry  entry  Set  changes  String  source  target  Change  change  boolean  
[buglab_swap_variables]^IOUtils.copy ( out, in ) ;^138^^^^^135^140^IOUtils.copy ( in, out ) ;^[CLASS] ChangeSet  [METHOD] copyStream [RETURN_TYPE] void   InputStream in ArchiveOutputStream out ArchiveEntry entry [VARIABLES] ArchiveEntry  entry  Set  changes  boolean  InputStream  in  ArchiveOutputStream  out  
[buglab_swap_variables]^IOUtils.copy (  out ) ;^138^^^^^135^140^IOUtils.copy ( in, out ) ;^[CLASS] ChangeSet  [METHOD] copyStream [RETURN_TYPE] void   InputStream in ArchiveOutputStream out ArchiveEntry entry [VARIABLES] ArchiveEntry  entry  Set  changes  boolean  InputStream  in  ArchiveOutputStream  out  
[buglab_swap_variables]^IOUtils.copy ( in ) ;^138^^^^^135^140^IOUtils.copy ( in, out ) ;^[CLASS] ChangeSet  [METHOD] copyStream [RETURN_TYPE] void   InputStream in ArchiveOutputStream out ArchiveEntry entry [VARIABLES] ArchiveEntry  entry  Set  changes  boolean  InputStream  in  ArchiveOutputStream  out  