[P1_Replace_Type]^private final  short  wantedCount;^9^^^^^4^14^private final int wantedCount;^[CLASS] Discrepancy   [VARIABLES] 
[P8_Replace_Mix]^private final  long  wantedCount;^9^^^^^4^14^private final int wantedCount;^[CLASS] Discrepancy   [VARIABLES] 
[P1_Replace_Type]^private final  short  actualCount;^10^^^^^5^15^private final int actualCount;^[CLASS] Discrepancy   [VARIABLES] 
[P8_Replace_Mix]^private final  long  actualCount;^10^^^^^5^15^private final int actualCount;^[CLASS] Discrepancy   [VARIABLES] 
[P5_Replace_Variable]^this.wantedCount = actualCount;^13^^^^^12^15^this.wantedCount = wantedCount;^[CLASS] Discrepancy  [METHOD] <init> [RETURN_TYPE] Discrepancy(int,int)   int wantedCount int actualCount [VARIABLES] int  actualCount  wantedCount  boolean  
[P11_Insert_Donor_Statement]^this.actualCount = actualCount;this.wantedCount = wantedCount;^13^^^^^12^15^this.wantedCount = wantedCount;^[CLASS] Discrepancy  [METHOD] <init> [RETURN_TYPE] Discrepancy(int,int)   int wantedCount int actualCount [VARIABLES] int  actualCount  wantedCount  boolean  
[P5_Replace_Variable]^this.actualCount = wantedCount;^14^^^^^12^15^this.actualCount = actualCount;^[CLASS] Discrepancy  [METHOD] <init> [RETURN_TYPE] Discrepancy(int,int)   int wantedCount int actualCount [VARIABLES] int  actualCount  wantedCount  boolean  
[P11_Insert_Donor_Statement]^this.wantedCount = wantedCount;this.actualCount = actualCount;^14^^^^^12^15^this.actualCount = actualCount;^[CLASS] Discrepancy  [METHOD] <init> [RETURN_TYPE] Discrepancy(int,int)   int wantedCount int actualCount [VARIABLES] int  actualCount  wantedCount  boolean  
[P5_Replace_Variable]^return actualCount;^18^^^^^17^19^return wantedCount;^[CLASS] Discrepancy  [METHOD] getWantedCount [RETURN_TYPE] int   [VARIABLES] int  actualCount  wantedCount  boolean  
[P5_Replace_Variable]^return Pluralizer.pluralize ( actualCount ) ;^22^^^^^21^23^return Pluralizer.pluralize ( wantedCount ) ;^[CLASS] Discrepancy  [METHOD] getPluralizedWantedCount [RETURN_TYPE] String   [VARIABLES] int  actualCount  wantedCount  boolean  
[P14_Delete_Statement]^^22^^^^^21^23^return Pluralizer.pluralize ( wantedCount ) ;^[CLASS] Discrepancy  [METHOD] getPluralizedWantedCount [RETURN_TYPE] String   [VARIABLES] int  actualCount  wantedCount  boolean  
[P5_Replace_Variable]^return wantedCount;^26^^^^^25^27^return actualCount;^[CLASS] Discrepancy  [METHOD] getActualCount [RETURN_TYPE] int   [VARIABLES] int  actualCount  wantedCount  boolean  
[P5_Replace_Variable]^return Pluralizer.pluralize ( wantedCount ) ;^30^^^^^29^31^return Pluralizer.pluralize ( actualCount ) ;^[CLASS] Discrepancy  [METHOD] getPluralizedActualCount [RETURN_TYPE] String   [VARIABLES] int  actualCount  wantedCount  boolean  
[P14_Delete_Statement]^^30^^^^^29^31^return Pluralizer.pluralize ( actualCount ) ;^[CLASS] Discrepancy  [METHOD] getPluralizedActualCount [RETURN_TYPE] String   [VARIABLES] int  actualCount  wantedCount  boolean  