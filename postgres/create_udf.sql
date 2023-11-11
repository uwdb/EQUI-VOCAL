CREATE OR REPLACE FUNCTION Near(double precision, text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Near'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Far(double precision, text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Far'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION LeftOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'LeftOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'RightOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Behind(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Behind'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION FrontOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'FrontOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION LeftQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'LeftQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'RightQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION TopQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'TopQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION BottomQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'BottomQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Color(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Color'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Shape(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Shape'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Material(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Material'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Red(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Red'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Gray(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Gray'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Blue(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Blue'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Green(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Green'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Brown(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Brown'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cyan(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Cyan'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Purple(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Purple'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Yellow(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Yellow'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cube(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Cube'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Sphere(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Sphere'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cylinder(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Cylinder'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Metal(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Metal'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Rubber(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Rubber'
LANGUAGE C STRICT;

-- For Shibuya dataset
CREATE OR REPLACE FUNCTION LeftPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'LeftPoly'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'RightPoly'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION TopPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'TopPoly'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION BottomPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'BottomPoly'
LANGUAGE C STRICT;

-- For Warsaw dataset
CREATE OR REPLACE FUNCTION Eastward4(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Eastward4'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Eastward3(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Eastward3'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Eastward2(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Eastward2'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Westward2(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Westward2'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Southward1Upper(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Southward1Upper'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Stopped(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Stopped'
LANGUAGE C;

CREATE OR REPLACE FUNCTION HighAccel(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'HighAccel'
LANGUAGE C;

CREATE OR REPLACE FUNCTION DistanceSmall(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'DistanceSmall'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Faster(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/postgres/functors', 'Faster'
LANGUAGE C;
