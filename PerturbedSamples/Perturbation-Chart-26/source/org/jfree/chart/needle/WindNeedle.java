[REPLACE]^private static final long serialVersionUID  = null ;^60^^^^^^^[REPLACE] private static final long serialVersionUID = -2861061368907167834L;^ [CLASS] WindNeedle  
[REPLACE]^super ( true ) ;^66^^^^^65^67^[REPLACE] super ( false ) ;^[METHOD] <init> [TYPE] WindNeedle() [PARAMETER] [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^^66^^^^^65^67^[ADD] super ( false ) ;^[METHOD] <init> [TYPE] WindNeedle() [PARAMETER] [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^super .getSize (  )  ;^80^^^^^77^97^[REPLACE] super.drawNeedle ( g2, plotArea, rotate, angle ) ;^[METHOD] drawNeedle [TYPE] void [PARAMETER] Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [CLASS] WindNeedle   [TYPE]  boolean false  true  [TYPE]  Point2D newRotate  rotate  [TYPE]  double angle  [TYPE]  Rectangle2D newArea  plotArea  [TYPE]  long serialVersionUID  [TYPE]  int spacing  [TYPE]  Graphics2D g2 
[ADD]^^80^^^^^77^97^[ADD] super.drawNeedle ( g2, plotArea, rotate, angle ) ;^[METHOD] drawNeedle [TYPE] void [PARAMETER] Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [CLASS] WindNeedle   [TYPE]  boolean false  true  [TYPE]  Point2D newRotate  rotate  [TYPE]  double angle  [TYPE]  Rectangle2D newArea  plotArea  [TYPE]  long serialVersionUID  [TYPE]  int spacing  [TYPE]  Graphics2D g2 
[REPLACE]^if  (  ( rotate == null )  ) {^81^^^^^77^97^[REPLACE] if  (  ( rotate != null )  &&  ( plotArea != null )  )  {^[METHOD] drawNeedle [TYPE] void [PARAMETER] Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [CLASS] WindNeedle   [TYPE]  boolean false  true  [TYPE]  Point2D newRotate  rotate  [TYPE]  double angle  [TYPE]  Rectangle2D newArea  plotArea  [TYPE]  long serialVersionUID  [TYPE]  int spacing  [TYPE]  Graphics2D g2 
[REPLACE]^int spacing = getSize (  )   ;^83^^^^^77^97^[REPLACE] int spacing = getSize (  )  * 3;^[METHOD] drawNeedle [TYPE] void [PARAMETER] Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [CLASS] WindNeedle   [TYPE]  boolean false  true  [TYPE]  Point2D newRotate  rotate  [TYPE]  double angle  [TYPE]  Rectangle2D newArea  plotArea  [TYPE]  long serialVersionUID  [TYPE]  int spacing  [TYPE]  Graphics2D g2 
[ADD]^Rectangle2D newArea = new Rectangle2D.Double (  ) ;^83^84^^^^77^97^[ADD] int spacing = getSize (  )  * 3; Rectangle2D newArea = new Rectangle2D.Double (  ) ;^[METHOD] drawNeedle [TYPE] void [PARAMETER] Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [CLASS] WindNeedle   [TYPE]  boolean false  true  [TYPE]  Point2D newRotate  rotate  [TYPE]  double angle  [TYPE]  Rectangle2D newArea  plotArea  [TYPE]  long serialVersionUID  [TYPE]  int spacing  [TYPE]  Graphics2D g2 
[REPLACE]^int spacing = getSize (  )  * 3;^84^^^^^77^97^[REPLACE] Rectangle2D newArea = new Rectangle2D.Double (  ) ;^[METHOD] drawNeedle [TYPE] void [PARAMETER] Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [CLASS] WindNeedle   [TYPE]  boolean false  true  [TYPE]  Point2D newRotate  rotate  [TYPE]  double angle  [TYPE]  Rectangle2D newArea  plotArea  [TYPE]  long serialVersionUID  [TYPE]  int spacing  [TYPE]  Graphics2D g2 
[REPLACE]^Point2D newRotate = newRotate;^86^^^^^77^97^[REPLACE] Point2D newRotate = rotate;^[METHOD] drawNeedle [TYPE] void [PARAMETER] Graphics2D g2 Rectangle2D plotArea Point2D rotate double angle [CLASS] WindNeedle   [TYPE]  boolean false  true  [TYPE]  Point2D newRotate  rotate  [TYPE]  double angle  [TYPE]  Rectangle2D newArea  plotArea  [TYPE]  long serialVersionUID  [TYPE]  int spacing  [TYPE]  Graphics2D g2 
[REPLACE]^if  ( object != this )  {^107^^^^^106^117^[REPLACE] if  ( object == null )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object object [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  Object object  [TYPE]  boolean false  true 
[REPLACE]^return true;^108^^^^^106^117^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object object [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  Object object  [TYPE]  boolean false  true 
[REPLACE]^if  ( object  ||  this )  {^110^^^^^106^117^[REPLACE] if  ( object == this )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object object [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  Object object  [TYPE]  boolean false  true 
[REPLACE]^return false;^111^^^^^106^117^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object object [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  Object object  [TYPE]  boolean false  true 
[REPLACE]^if  (  object instanceof WindNeedle )  {^113^^^^^106^117^[REPLACE] if  ( super.equals ( object )  && object instanceof WindNeedle )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object object [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  Object object  [TYPE]  boolean false  true 
[REPLACE]^return false;^114^^^^^106^117^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object object [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  Object object  [TYPE]  boolean false  true 
[REPLACE]^return true;^116^^^^^106^117^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object object [CLASS] WindNeedle   [TYPE]  long serialVersionUID  [TYPE]  Object object  [TYPE]  boolean false  true 