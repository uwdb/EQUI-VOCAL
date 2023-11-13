\set functors_path '/home/enhao/EQUI-VOCAL/postgres/functors'

CREATE OR REPLACE FUNCTION Near(double precision, text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Near'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Far(double precision, text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Far'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION LeftOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'LeftOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'RightOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Behind(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Behind'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION FrontOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'FrontOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION LeftQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'LeftQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'RightQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION TopQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'TopQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION BottomQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'BottomQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Color(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Color'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Shape(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Shape'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Material(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Material'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Red(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Red'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Gray(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Gray'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Blue(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Blue'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Green(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Green'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Brown(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Brown'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cyan(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Cyan'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Purple(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Purple'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Yellow(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Yellow'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cube(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Cube'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Sphere(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Sphere'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cylinder(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Cylinder'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Metal(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Metal'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Rubber(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Rubber'
LANGUAGE C STRICT;

-- For Shibuya dataset
CREATE OR REPLACE FUNCTION LeftPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'LeftPoly'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'RightPoly'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION TopPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'TopPoly'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION BottomPoly(double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'BottomPoly'
LANGUAGE C STRICT;

-- For Warsaw dataset
CREATE OR REPLACE FUNCTION Eastward4(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Eastward4'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Eastward3(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Eastward3'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Eastward2(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Eastward2'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Westward2(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Westward2'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Southward1Upper(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Southward1Upper'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Stopped(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Stopped'
LANGUAGE C;

CREATE OR REPLACE FUNCTION HighAccel(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'HighAccel'
LANGUAGE C;

CREATE OR REPLACE FUNCTION DistanceSmall(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'DistanceSmall'
LANGUAGE C;

CREATE OR REPLACE FUNCTION Faster(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean AS
:'functors_path', 'Faster'
LANGUAGE C;
