CREATE OR REPLACE FUNCTION Near(double precision, text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Near'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Far(double precision, text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Far'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION LeftOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'LeftOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'RightOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Behind(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Behind'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION FrontOf(text, text, text, double precision, double precision, double precision, double precision, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'FrontOf'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION LeftQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'LeftQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION RightQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'RightQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION TopQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'TopQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION BottomQuadrant(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'BottomQuadrant'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Color(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Color'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Shape(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Shape'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Material(text, text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Material'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Red(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Red'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Gray(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Gray'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Blue(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Blue'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Green(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Green'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Brown(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Brown'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cyan(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Cyan'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Purple(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Purple'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Yellow(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Yellow'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cube(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Cube'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Sphere(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Sphere'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Cylinder(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Cylinder'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Metal(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Metal'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION Rubber(text, text, text, double precision, double precision, double precision, double precision) RETURNS boolean AS
'/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Rubber'
LANGUAGE C STRICT;
