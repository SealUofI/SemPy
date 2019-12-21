//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {-1, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, -1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {3, 2};
//+
Line(3) = {3, 4};
//+
Line(4) = {1, 4};
//+
Curve Loop(1) = {2, -1, 4, -3};
//+
Plane Surface(1) = {1};
//+
Transfinite Surface {1} = {2, 3, 4, 1};
//+
Transfinite Curve {1, 3} = 2 Using Progression 1;
//+
Transfinite Curve {2, 4} = 2 Using Progression 1;
//+
Recombine Surface {1};
