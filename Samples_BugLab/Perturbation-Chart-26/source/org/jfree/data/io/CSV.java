[buglab_swap_variables]^if  ( lineIndexIndex == 0 )  {^104^^^^^96^115^if  ( lineIndex == 0 )  {^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^extractRowKeyAndData ( dataset, line, columnKeys ) ;^108^^^^^96^115^extractRowKeyAndData ( line, dataset, columnKeys ) ;^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^extractRowKeyAndData (  dataset, columnKeys ) ;^108^^^^^96^115^extractRowKeyAndData ( line, dataset, columnKeys ) ;^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^extractRowKeyAndData ( line,  columnKeys ) ;^108^^^^^96^115^extractRowKeyAndData ( line, dataset, columnKeys ) ;^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^extractRowKeyAndData ( line, columnKeys, dataset ) ;^108^^^^^96^115^extractRowKeyAndData ( line, dataset, columnKeys ) ;^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^extractRowKeyAndData ( line, dataset ) ;^108^^^^^96^115^extractRowKeyAndData ( line, dataset, columnKeys ) ;^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^extractRowKeyAndData ( columnKeys, dataset, line ) ;^108^^^^^96^115^extractRowKeyAndData ( line, dataset, columnKeys ) ;^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^if  ( line == 0 )  {^104^^^^^96^115^if  ( lineIndex == 0 )  {^[CLASS] CSV  [METHOD] readCategoryDataset [RETURN_TYPE] CategoryDataset   Reader in [VARIABLES] boolean  char  fieldDelimiter  textDelimiter  BufferedReader  reader  Reader  in  List  columnKeys  String  line  DefaultCategoryDataset  dataset  int  lineIndex  
[buglab_swap_variables]^String key = start.substring ( line, i ) ;^132^^^^^124^142^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring ( i, start ) ;^132^^^^^124^142^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring (  i ) ;^132^^^^^124^142^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring ( start ) ;^132^^^^^124^142^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = i.substring ( start, line ) ;^132^^^^^124^142^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^if  ( i.charAt ( line )  == this.fieldDelimiter )  {^129^^^^^124^142^if  ( line.charAt ( i )  == this.fieldDelimiter )  {^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = start.substring ( line, line.length (  )  ) ;^139^^^^^124^142^String key = line.substring ( start, line.length (  )  ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring ( start.length (  )  ) ;^139^^^^^124^142^String key = line.substring ( start, line.length (  )  ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring (  line.length (  )  ) ;^139^^^^^124^142^String key = line.substring ( start, line.length (  )  ) ;^[CLASS] CSV  [METHOD] extractColumnKeys [RETURN_TYPE] List   String line [VARIABLES] char  fieldDelimiter  textDelimiter  List  keys  String  key  line  boolean  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( i.substring ( start, line )  ) ) ;^164^165^166^^^159^171^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( start.substring ( line, i )  ) ) ;^164^165^166^^^159^171^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring (  i )  ) ) ;^164^165^166^^^159^171^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start )  ) ) ;^164^165^166^^^159^171^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( i, start )  ) ) ;^164^165^166^^^159^171^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( start.substring ( line, i )  ) ) ;^165^166^^^^159^171^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring ( i, start )  ) ) ;^165^166^^^^159^171^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring (  i )  ) ) ;^165^166^^^^159^171^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring ( start )  ) ) ;^165^166^^^^159^171^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( columnKeys, rowKey, ( Comparable )  value.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^159^171^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue (  rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^159^171^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value, fieldIndex, ( Comparable )  columnKeys.get ( rowKey - 1 ) ) ;^167^168^169^170^^159^171^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^159^171^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value, rowKey, ( Comparable )  fieldIndex.get ( columnKeys - 1 ) ) ;^167^168^169^170^^159^171^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( fieldIndex, rowKey, ( Comparable )  columnKeys.get ( value - 1 ) ) ;^167^168^169^170^^159^171^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^( Comparable )  fieldIndex.get ( columnKeys - 1 ) ) ;^169^170^^^^159^171^( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = start.substring ( line, i ) ;^160^^^^^153^182^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring (  i ) ;^160^^^^^153^182^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring ( i, start ) ;^160^^^^^153^182^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = line.substring ( start ) ;^160^^^^^153^182^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^String key = i.substring ( start, line ) ;^160^^^^^153^182^String key = line.substring ( start, i ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( i.substring ( start, line )  ) ) ;^164^165^166^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( i, start )  ) ) ;^164^165^166^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring (  i )  ) ) ;^164^165^166^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start )  ) ) ;^164^165^166^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( start.substring ( line, i )  ) ) ;^164^165^166^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( start.substring ( line, i )  ) ) ;^165^166^^^^153^182^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring (  i )  ) ) ;^165^166^^^^153^182^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( i.substring ( start, line )  ) ) ;^165^166^^^^153^182^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring ( start )  ) ) ;^165^166^^^^153^182^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring ( i, start )  ) ) ;^165^166^^^^153^182^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( fieldIndex, rowKey, ( Comparable )  columnKeys.get ( value - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue (  rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value, columnKeys, ( Comparable )  rowKey.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value, rowKey, ( Comparable )  fieldIndex.get ( columnKeys - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^( Comparable )  fieldIndex.get ( columnKeys - 1 ) ) ;^169^170^^^^153^182^( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( i.substring ( start, line )  ) ) ;^165^166^^^^159^171^removeStringDelimiters ( line.substring ( start, i )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^if  ( i.charAt ( line )  == this.fieldDelimiter )  {^158^^^^^153^182^if  ( line.charAt ( i )  == this.fieldDelimiter )  {^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( rowKey, value, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( columnKeys, rowKey, ( Comparable )  value.get ( fieldIndex - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value, fieldIndex, ( Comparable )  columnKeys.get ( rowKey - 1 ) ) ;^167^168^169^170^^153^182^dataset.addValue ( value, rowKey, ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( start.substring ( line, line.length (  )  )  ) ) ;^176^177^178^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, line.length (  )  )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start.length (  )  )  ) ) ;^176^177^178^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, line.length (  )  )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^Double value = Double.valueOf ( removeStringDelimiters ( line.substring (  line.length (  )  )  ) ) ;^176^177^178^^^153^182^Double value = Double.valueOf ( removeStringDelimiters ( line.substring ( start, line.length (  )  )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( start.substring ( line, line.length (  )  )  ) ) ;^177^178^^^^153^182^removeStringDelimiters ( line.substring ( start, line.length (  )  )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring ( start.length (  )  )  ) ) ;^177^178^^^^153^182^removeStringDelimiters ( line.substring ( start, line.length (  )  )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^removeStringDelimiters ( line.substring (  line.length (  )  )  ) ) ;^177^178^^^^153^182^removeStringDelimiters ( line.substring ( start, line.length (  )  )  ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue (  rowKey,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^179^180^181^^^153^182^dataset.addValue ( value, rowKey,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( rowKey, value,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^179^180^181^^^153^182^dataset.addValue ( value, rowKey,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( value,   ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^179^180^181^^^153^182^dataset.addValue ( value, rowKey,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^dataset.addValue ( fieldIndex, rowKey,  ( Comparable )  columnKeys.get ( value - 1 ) ) ;^179^180^181^^^153^182^dataset.addValue ( value, rowKey,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^value, rowKey,  ( Comparable )  fieldIndex.get ( columnKeys - 1 ) ) ;^180^181^^^^153^182^value, rowKey,  ( Comparable )  columnKeys.get ( fieldIndex - 1 ) ) ;^[CLASS] CSV  [METHOD] extractRowKeyAndData [RETURN_TYPE] void   String line DefaultCategoryDataset dataset List columnKeys [VARIABLES] Comparable  rowKey  boolean  char  fieldDelimiter  textDelimiter  List  columnKeys  String  key  line  DefaultCategoryDataset  dataset  Double  value  int  fieldIndex  i  start  
[buglab_swap_variables]^if  ( this.textDelimiter.charAt ( 0 )  == k )  {^194^^^^^192^201^if  ( k.charAt ( 0 )  == this.textDelimiter )  {^[CLASS] CSV  [METHOD] removeStringDelimiters [RETURN_TYPE] String   String key [VARIABLES] char  fieldDelimiter  textDelimiter  String  k  key  boolean  
[buglab_swap_variables]^if  ( this.textDelimiter.charAt ( k.length (  )  - 1 )  == k )  {^197^^^^^192^201^if  ( k.charAt ( k.length (  )  - 1 )  == this.textDelimiter )  {^[CLASS] CSV  [METHOD] removeStringDelimiters [RETURN_TYPE] String   String key [VARIABLES] char  fieldDelimiter  textDelimiter  String  k  key  boolean  