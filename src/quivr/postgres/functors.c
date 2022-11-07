#include "postgres.h"
#include <stdint.h>
#include <math.h>
#include "fmgr.h"
#include "utils/builtins.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(get_iou);

Datum get_iou(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);
    float x1p = PG_GETARG_FLOAT8(4);
    float y1p = PG_GETARG_FLOAT8(5);
    float x2p = PG_GETARG_FLOAT8(6);
    float y2p = PG_GETARG_FLOAT8(7);

    // determine the coordinates of the intersection rectangle
    float x_left = (x1 > x1p) ? x1 : x1p;
    float y_top = (y1 > y1p) ? y1 : y1p;
    float x_right = (x2 < x2p) ? x2 : x2p;
    float y_bottom = (y2 < y2p)? y2 : y2p;

    if (x_right < x_left || y_bottom < y_top) {
        PG_RETURN_FLOAT8(0.0);
    }
    // The intersection of two axis-aligned bounding boxes is always an
    // axis-aligned bounding box
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);

    // compute the area of both AABBs
    float bb1_area = (x2 - x1) * (y2 - y1);
    float bb2_area = (x2p - x1p) * (y2p - y1p);

    // compute the intersection over union by taking the intersection
    // area and dividing it by the sum of prediction + ground-truth
    // areas - the interesection area
    float iou = intersection_area / (bb1_area + bb2_area - intersection_area);
    PG_RETURN_FLOAT8(iou);
}

PG_FUNCTION_INFO_V1(Near);

Datum Near(PG_FUNCTION_ARGS) {
    float theta = PG_GETARG_FLOAT8(0);
    float x1 = PG_GETARG_FLOAT8(4);
    float y1 = PG_GETARG_FLOAT8(5);
    float x2 = PG_GETARG_FLOAT8(6);
    float y2 = PG_GETARG_FLOAT8(7);
    float x3 = PG_GETARG_FLOAT8(11);
    float y3 = PG_GETARG_FLOAT8(12);
    float x4 = PG_GETARG_FLOAT8(13);
    float y4 = PG_GETARG_FLOAT8(14);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    float distance = sqrt(pow(cx1 - cx2, 2.0) + pow(cy1 - cy2, 2.0)) / ((x2 - x1 + x4 - x3) / 2);

    PG_RETURN_BOOL(distance < theta);
}

PG_FUNCTION_INFO_V1(Far);

Datum Far(PG_FUNCTION_ARGS) {
    float theta = PG_GETARG_FLOAT8(0);
    float x1 = PG_GETARG_FLOAT8(4);
    float y1 = PG_GETARG_FLOAT8(5);
    float x2 = PG_GETARG_FLOAT8(6);
    float y2 = PG_GETARG_FLOAT8(7);
    float x3 = PG_GETARG_FLOAT8(11);
    float y3 = PG_GETARG_FLOAT8(12);
    float x4 = PG_GETARG_FLOAT8(13);
    float y4 = PG_GETARG_FLOAT8(14);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    float distance = sqrt(pow(cx1 - cx2, 2.0) + pow(cy1 - cy2, 2.0)) / ((x2 - x1 + x4 - x3) / 2);

    PG_RETURN_BOOL(distance > theta);
}

PG_FUNCTION_INFO_V1(LeftOf);

Datum LeftOf(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);
    float x3 = PG_GETARG_FLOAT8(10);
    float y3 = PG_GETARG_FLOAT8(11);
    float x4 = PG_GETARG_FLOAT8(12);
    float y4 = PG_GETARG_FLOAT8(13);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    PG_RETURN_BOOL(cx1 < cx2);
}

PG_FUNCTION_INFO_V1(RightOf);

Datum RightOf(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);
    float x3 = PG_GETARG_FLOAT8(10);
    float y3 = PG_GETARG_FLOAT8(11);
    float x4 = PG_GETARG_FLOAT8(12);
    float y4 = PG_GETARG_FLOAT8(13);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    PG_RETURN_BOOL(cx1 > cx2);
}

PG_FUNCTION_INFO_V1(Behind);

Datum Behind(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);
    float x3 = PG_GETARG_FLOAT8(10);
    float y3 = PG_GETARG_FLOAT8(11);
    float x4 = PG_GETARG_FLOAT8(12);
    float y4 = PG_GETARG_FLOAT8(13);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    PG_RETURN_BOOL(cy1 < cy2);
}

PG_FUNCTION_INFO_V1(FrontOf);

Datum FrontOf(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);
    float x3 = PG_GETARG_FLOAT8(10);
    float y3 = PG_GETARG_FLOAT8(11);
    float x4 = PG_GETARG_FLOAT8(12);
    float y4 = PG_GETARG_FLOAT8(13);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    PG_RETURN_BOOL(cy1 > cy2);
}

PG_FUNCTION_INFO_V1(LeftQuadrant);

Datum LeftQuadrant(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;

    PG_RETURN_BOOL(cx1 >= 0 && cx1 < 240);
}

PG_FUNCTION_INFO_V1(RightQuadrant);

Datum RightQuadrant(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;

    PG_RETURN_BOOL(cx1 >= 240 && cx1 <= 480);
}

PG_FUNCTION_INFO_V1(TopQuadrant);

Datum TopQuadrant(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;

    PG_RETURN_BOOL(cy1 >= 0 && cy1 < 160);
}

PG_FUNCTION_INFO_V1(BottomQuadrant);

Datum BottomQuadrant(PG_FUNCTION_ARGS) {
    float x1 = PG_GETARG_FLOAT8(3);
    float y1 = PG_GETARG_FLOAT8(4);
    float x2 = PG_GETARG_FLOAT8(5);
    float y2 = PG_GETARG_FLOAT8(6);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;

    PG_RETURN_BOOL(cy1 >= 160 && cy1 <= 320);
}

// COLOR
//
PG_FUNCTION_INFO_V1(Color);

Datum Color(PG_FUNCTION_ARGS) {
    char* target_color = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(2));

    PG_RETURN_BOOL(strcmp(color_a, target_color) == 0);
}

PG_FUNCTION_INFO_V1(Red);

Datum Red(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "red") == 0);
}

PG_FUNCTION_INFO_V1(Gray);

Datum Gray(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "gray") == 0);
}

PG_FUNCTION_INFO_V1(Blue);

Datum Blue(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "blue") == 0);
}

PG_FUNCTION_INFO_V1(Green);

Datum Green(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "green") == 0);
}

PG_FUNCTION_INFO_V1(Brown);

Datum Brown(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "brown") == 0);
}

PG_FUNCTION_INFO_V1(Cyan);

Datum Cyan(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "cyan") == 0);
}

PG_FUNCTION_INFO_V1(Purple);

Datum Purple(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "purple") == 0);
}

PG_FUNCTION_INFO_V1(Yellow);

Datum Yellow(PG_FUNCTION_ARGS) {
    char* color_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(color_a, "yellow") == 0);
}

// SHAPE
PG_FUNCTION_INFO_V1(Shape);

Datum Shape(PG_FUNCTION_ARGS) {
    char* target_shape = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* shape_a = text_to_cstring(PG_GETARG_TEXT_PP(1));

    PG_RETURN_BOOL(strcmp(shape_a, target_shape) == 0);
}

PG_FUNCTION_INFO_V1(Cube);

Datum Cube(PG_FUNCTION_ARGS) {
    char* shape_a = text_to_cstring(PG_GETARG_TEXT_PP(0));

    PG_RETURN_BOOL(strcmp(shape_a, "cube") == 0);
}

PG_FUNCTION_INFO_V1(Sphere);

Datum Sphere(PG_FUNCTION_ARGS) {
    char* shape_a = text_to_cstring(PG_GETARG_TEXT_PP(0));

    PG_RETURN_BOOL(strcmp(shape_a, "sphere") == 0);
}

PG_FUNCTION_INFO_V1(Cylinder);

Datum Cylinder(PG_FUNCTION_ARGS) {
    char* shape_a = text_to_cstring(PG_GETARG_TEXT_PP(0));

    PG_RETURN_BOOL(strcmp(shape_a, "cylinder") == 0);
}

// MATERIAL
//
PG_FUNCTION_INFO_V1(Material);

Datum Material(PG_FUNCTION_ARGS) {
    char* target_material = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* material_a = text_to_cstring(PG_GETARG_TEXT_PP(3));

    PG_RETURN_BOOL(strcmp(material_a, target_material) == 0);
}

PG_FUNCTION_INFO_V1(Metal);

Datum Metal(PG_FUNCTION_ARGS) {
    char* material_a = text_to_cstring(PG_GETARG_TEXT_PP(2));

    PG_RETURN_BOOL(strcmp(material_a, "metal") == 0);
}

PG_FUNCTION_INFO_V1(Rubber);

Datum Rubber(PG_FUNCTION_ARGS) {
    char* material_a = text_to_cstring(PG_GETARG_TEXT_PP(2));

    PG_RETURN_BOOL(strcmp(material_a, "rubber") == 0);
}

// compile:
// cc -I /mmfs1/gscratch/balazinska/enhaoz/env/include/server/ -fpic -c functors.c
// cc -shared -o functors.so functors.o